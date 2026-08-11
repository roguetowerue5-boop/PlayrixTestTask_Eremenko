"""LoRA Trainer: обучение Flux LoRA на Icons через fal.ai.

OpenRouter свои LoRA не хостит — обучение и инференс идут через fal
(провайдер fal_lora), текст/откат картинок — через OpenRouter.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import ROOT, invalidate_caches, providers_config
from app import settings as app_settings

log = logging.getLogger("offerforge.lora")
router = APIRouter(prefix="/api/lora", tags=["lora"])

LORA_DIR = ROOT / "lora"
DATASET_DIR = LORA_DIR / "dataset"
DATASET_ZIP = LORA_DIR / "dataset.zip"
JOBS_FILE = LORA_DIR / "jobs.json"
ICONS_DIR = ROOT.parent / "Icons"

FAL_TRAIN = "fal-ai/flux-lora-fast-training"
QUEUE = "https://queue.fal.run"
TRIGGER = "plrxcard"


class KeyBody(BaseModel):
    api_key: str


class TrainBody(BaseModel):
    steps: int = Field(1100, ge=200, le=2500)
    trigger_word: str = TRIGGER
    rebuild_dataset: bool = False
    # True = не грузить zip заново, взять zip_url из прошлого job (если есть).
    reuse_zip: bool = True


class ApplyBody(BaseModel):
    lora_url: str
    scale: float = Field(0.9, ge=0.1, le=2.0)
    enable: bool = True


def _fal_key() -> str | None:
    return app_settings.resolve_key("fal_lora", "FAL_KEY")


def _auth_headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
    }


def _load_jobs() -> dict:
    if not JOBS_FILE.exists():
        return {"jobs": [], "active_request_id": None, "last_lora_url": None}
    try:
        return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"jobs": [], "active_request_id": None, "last_lora_url": None}


def _save_jobs(data: dict) -> None:
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dataset_stats() -> dict:
    pngs = list(DATASET_DIR.glob("*.png")) if DATASET_DIR.is_dir() else []
    return {
        "ready": DATASET_ZIP.is_file() and bool(pngs),
        "images": len(pngs),
        "zip_mb": round(DATASET_ZIP.stat().st_size / 1e6, 1) if DATASET_ZIP.is_file() else 0,
        "zip_path": str(DATASET_ZIP) if DATASET_ZIP.is_file() else None,
        "icons_ok": ICONS_DIR.is_dir(),
        "icons_count": len(list(ICONS_DIR.glob("*.png"))) if ICONS_DIR.is_dir() else 0,
    }


def _fal_lora_cfg() -> dict:
    cfg = (providers_config().get("providers") or {}).get("fal_lora") or {}
    path = ""
    try:
        path = (((cfg.get("image") or {}).get("body") or {}).get("loras") or [{}])[0].get("path") or ""
    except (IndexError, AttributeError, TypeError):
        path = ""
    return {
        "enabled": bool(cfg.get("enabled")),
        "lora_path": path,
        "configured": bool(path) and "ЗАМЕНИ" not in path,
    }


@router.get("/status")
async def lora_status() -> dict:
    key = _fal_key()
    admin = (
        app_settings.resolve_key("fal_admin", "FAL_ADMIN_KEY")
        or app_settings.resolve_key("fal_admin", "FAL_KEY_ADMIN")
    )
    jobs = _load_jobs()
    return {
        "has_key": bool(key),
        "key_mask": app_settings.mask(key),
        "has_admin_key": bool(admin),
        "admin_key_mask": app_settings.mask(admin),
        "trigger": TRIGGER,
        "dataset": _dataset_stats(),
        "fal_lora": _fal_lora_cfg(),
        "active_request_id": jobs.get("active_request_id"),
        "last_lora_url": jobs.get("last_lora_url"),
        "jobs": (jobs.get("jobs") or [])[-8:],
        "hint": (
            "Ключ fal и OpenRouter работают вместе: brief/concept → OpenRouter, "
            "render LoRA → fal, откат картинок → OpenRouter. "
            "Баланс Fal в сайдбаре — Admin key (FAL_ADMIN_KEY)."
        ),
    }


@router.post("/key")
async def save_fal_key(body: KeyBody) -> dict:
    key = (body.api_key or "").strip()
    if not key:
        raise HTTPException(400, "Вставь FAL_KEY")
    app_settings.save_secret("fal_lora", key)
    # Не включаем провайдер автоматически — сначала нужно обучить / подставить URL.
    return {"ok": True, "key_mask": app_settings.mask(key)}


@router.post("/admin-key")
async def save_fal_admin_key(body: KeyBody) -> dict:
    """Admin scope key для /v1/account/billing (остаток в сайдбаре)."""
    key = (body.api_key or "").strip()
    if not key:
        raise HTTPException(400, "Вставь FAL_ADMIN_KEY (scope ADMIN)")
    app_settings.save_secret("fal_admin", key)
    return {"ok": True, "key_mask": app_settings.mask(key)}


def _rebuild_dataset() -> None:
    script = ROOT / "tools" / "build_lora_dataset.py"
    if not script.is_file():
        raise HTTPException(500, "tools/build_lora_dataset.py не найден")
    if not ICONS_DIR.is_dir():
        raise HTTPException(400, f"Нет папки Icons: {ICONS_DIR}")
    cmd = [
        sys.executable, str(script),
        "--src", str(ICONS_DIR),
        "--out", str(DATASET_DIR),
        "--size", "1024",
        "--art-refs", "6",
    ]
    log.info("Сборка датасета: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"Сборка датасета упала: {proc.stderr or proc.stdout}")


def _cached_zip_url() -> str | None:
    """URL датасета с прошлого успешного старта — повторная заливка ~127 МБ не нужна."""
    jobs = _load_jobs()
    for j in reversed(jobs.get("jobs") or []):
        url = (j.get("zip_url") or "").strip()
        if url.startswith("http"):
            return url
    return None


async def _upload_zip(key: str, zip_path: Path) -> str:
    """Загрузка zip на fal CDN. fal-client сам делает multipart для файлов >90 МБ."""
    try:
        import fal_client
    except ImportError as e:
        raise HTTPException(
            500,
            "Нет пакета fal-client. В venv: pip install fal-client",
        ) from e

    size_mb = zip_path.stat().st_size / 1e6
    log.info("Загружаю %s (%.1f МБ) через fal-client…", zip_path.name, size_mb)
    prev = os.environ.get("FAL_KEY")
    os.environ["FAL_KEY"] = key
    url = None
    err: Exception | None = None
    try:
        # Sync multipart в thread — async-путь у fal_client иногда падает на больших zip.
        url = await asyncio.to_thread(fal_client.upload_file, zip_path)
    except Exception as e1:  # noqa: BLE001
        err = e1
        log.warning("fal sync upload failed: %s — пробую async…", e1)
        try:
            url = await fal_client.upload_file_async(zip_path)
            err = None
        except Exception as e2:  # noqa: BLE001
            err = e2
    finally:
        if prev is None:
            os.environ.pop("FAL_KEY", None)
        else:
            os.environ["FAL_KEY"] = prev
    if err or not url:
        raise HTTPException(502, f"fal upload: {err or 'пустой URL'}") from err
    log.info("Zip на CDN: %s", url)
    return url


@router.post("/rebuild-dataset")
async def rebuild_dataset() -> dict:
    _rebuild_dataset()
    return {"ok": True, "dataset": _dataset_stats()}


@router.post("/train")
async def start_train(body: TrainBody) -> dict:
    key = _fal_key()
    if not key:
        raise HTTPException(400, "Сначала сохрани FAL_KEY на этом экране")

    if body.rebuild_dataset or not DATASET_ZIP.is_file():
        _rebuild_dataset()
    if not DATASET_ZIP.is_file():
        raise HTTPException(400, "Нет lora/dataset.zip — нажми «Собрать датасет»")

    zip_url = None
    if body.reuse_zip and not body.rebuild_dataset:
        zip_url = _cached_zip_url()
        if zip_url:
            log.info("Переиспользую zip с CDN: %s", zip_url)
    if not zip_url:
        try:
            log.info("Загружаю %s на fal storage…", DATASET_ZIP)
            zip_url = await _upload_zip(key, DATASET_ZIP)
        except HTTPException as upload_err:
            # Последний шанс: прошлый zip ещё жив на fal.
            cached = _cached_zip_url()
            if cached:
                log.warning(
                    "Upload упал (%s) — стартую на кэшированном zip %s",
                    upload_err.detail,
                    cached,
                )
                zip_url = cached
            else:
                raise
    payload = {
        "images_data_url": zip_url,
        "trigger_word": (body.trigger_word or TRIGGER).strip() or TRIGGER,
        "is_style": True,
        "create_masks": False,
        "steps": body.steps,
    }
    async with httpx.AsyncClient() as client:
        submit = await client.post(
            f"{QUEUE}/{FAL_TRAIN}",
            headers=_auth_headers(key),
            json=payload,
            timeout=120,
        )
        if submit.status_code >= 400:
            raise HTTPException(
                502,
                f"fal train submit: {submit.status_code} {submit.text[:400]}",
            )
        info = submit.json()

    request_id = info.get("request_id") or info.get("requestId")
    if not request_id:
        raise HTTPException(502, f"fal не вернул request_id: {info}")

    jobs = _load_jobs()
    entry = {
        "request_id": request_id,
        "status": info.get("status") or "IN_QUEUE",
        "steps": body.steps,
        "trigger_word": payload["trigger_word"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "zip_url": zip_url,
        "status_url": info.get("status_url"),
        "response_url": info.get("response_url"),
        "lora_url": None,
        "error": None,
    }
    jobs.setdefault("jobs", []).append(entry)
    jobs["active_request_id"] = request_id
    _save_jobs(jobs)

    return {
        "ok": True,
        "request_id": request_id,
        "message": (
            f"Обучение запущено ({body.steps} steps, стиль, trigger={payload['trigger_word']}). "
            "Обновляй статус кнопкой ниже — обычно несколько минут."
        ),
        "job": entry,
    }


@router.get("/job")
async def job_status(request_id: str | None = None) -> dict:
    key = _fal_key()
    if not key:
        raise HTTPException(400, "Нет FAL_KEY")

    jobs = _load_jobs()
    rid = (request_id or jobs.get("active_request_id") or "").strip()
    if not rid:
        raise HTTPException(400, "Нет активного обучения")

    async with httpx.AsyncClient() as client:
        st = await client.get(
            f"{QUEUE}/{FAL_TRAIN}/requests/{rid}/status",
            headers=_auth_headers(key),
            params={"logs": "1"},
            timeout=60,
        )
        if st.status_code == 404:
            raise HTTPException(404, f"Запрос {rid} не найден на fal")
        if st.status_code >= 400:
            raise HTTPException(502, f"fal status: {st.status_code} {st.text[:300]}")
        status_body = st.json()

    status = status_body.get("status") or "UNKNOWN"
    logs = []
    for row in status_body.get("logs") or []:
        if isinstance(row, dict):
            logs.append(row.get("message") or str(row))
        else:
            logs.append(str(row))

    result: dict[str, Any] = {
        "request_id": rid,
        "status": status,
        "logs": logs[-30:],
        "queue_position": status_body.get("queue_position"),
    }

    if status == "COMPLETED":
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{QUEUE}/{FAL_TRAIN}/requests/{rid}",
                headers=_auth_headers(key),
                timeout=120,
            )
        if res.status_code >= 400:
            raise HTTPException(502, f"fal result: {res.status_code} {res.text[:300]}")
        data = res.json()
        # Ответ может быть обёрнут в {response: ...} или плоский.
        payload = data.get("response") if isinstance(data.get("response"), dict) else data
        lora_file = payload.get("diffusers_lora_file") or {}
        lora_url = lora_file.get("url") if isinstance(lora_file, dict) else None
        result["lora_url"] = lora_url
        result["raw"] = {
            "diffusers_lora_file": lora_file,
            "config_file": payload.get("config_file"),
        }
        # Обновить журнал.
        for j in jobs.get("jobs") or []:
            if j.get("request_id") == rid:
                j["status"] = "COMPLETED"
                j["lora_url"] = lora_url
                j["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if lora_url:
            jobs["last_lora_url"] = lora_url
        _save_jobs(jobs)

    elif status in ("FAILED", "CANCELLED", "ERROR"):
        err = status_body.get("error") or status_body.get("detail") or status
        result["error"] = err
        for j in jobs.get("jobs") or []:
            if j.get("request_id") == rid:
                j["status"] = status
                j["error"] = str(err)
        _save_jobs(jobs)

    else:
        for j in jobs.get("jobs") or []:
            if j.get("request_id") == rid:
                j["status"] = status
        _save_jobs(jobs)

    return result


@router.post("/apply")
async def apply_lora(body: ApplyBody) -> dict:
    url = (body.lora_url or "").strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Нужен http(s) URL на .safetensors")

    # Читаем YAML с диска, чтобы не потерять комментарии через полный rewrite —
    # но safe_dump всё равно сотрёт комментарии. Делаем точечную правку файла.
    path = app_settings.PROVIDERS_FILE
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    providers = raw.setdefault("providers", {})
    fal = providers.get("fal_lora")
    if not isinstance(fal, dict):
        raise HTTPException(500, "В providers.yaml нет fal_lora")

    fal["enabled"] = bool(body.enable)
    image = fal.setdefault("image", {})
    body_cfg = image.setdefault("body", {})
    loras = body_cfg.setdefault("loras", [{}])
    if not loras:
        loras.append({})
        body_cfg["loras"] = loras
    loras[0] = {"path": url, "scale": float(body.scale)}

    app_settings._dump_yaml(path, raw, app_settings.PROVIDERS_HEADER)
    invalidate_caches()

    jobs = _load_jobs()
    jobs["last_lora_url"] = url
    _save_jobs(jobs)

    return {
        "ok": True,
        "enabled": fal["enabled"],
        "lora_path": url,
        "message": (
            "fal_lora включён. В наполнении выбери пресет «LoRA (Icons)» — "
            "картинки пойдут через fal, текст останется на OpenRouter."
        ),
    }
