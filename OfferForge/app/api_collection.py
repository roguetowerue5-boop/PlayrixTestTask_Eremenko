"""Сборка страницы коллекции через OfferBuilder.

Отдельный экран от «Сборка оффера»: там композитор OfferForge + арт из
прогонов, здесь — готовые материалы Cutted и 10 слотов коллекции.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import ROOT, RUNS_DIR, is_offline, load_preset, providers_config, resolve_variant
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry

log = logging.getLogger("offerforge.collection")
router = APIRouter(prefix="/api/collection", tags=["collection"])

PLAYRIX = ROOT.parent
BUILDER_DIR = PLAYRIX / "OfferBuilder"
CUTTED_DIR = PLAYRIX / "Cutted"
ICONS_DIR = PLAYRIX / "Icons"
REFERENCE = PLAYRIX / "RuleForBuilding" / "blue_template.png"
COLLECTION_RUNS = RUNS_DIR / "collection"


def _load_builder():
    if str(BUILDER_DIR) not in sys.path:
        sys.path.insert(0, str(BUILDER_DIR))
    import build_offer as ob  # noqa: WPS433 — соседний пакет в Playrix
    return ob


class SlotIn(BaseModel):
    name: str = "Item"
    rarity: int = Field(1, ge=1, le=5)
    enchant_on: bool = False
    enchant_power: int = Field(2, ge=1, le=99)
    # Относительный путь под runs/ (например collection-fill/.../slot_01.png)
    image_rel: str | None = None


def _resolve_run_image(rel: str | None, assets_dir: Path, slot_idx: int) -> str | None:
    """Копирует картинку из runs/ в assets прогона сборки."""
    if not rel:
        return None
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.startswith("runs/"):
        rel = rel[5:]
    # query-string от cache-bust не нужна на диске
    rel = rel.split("?", 1)[0]
    if ".." in rel or rel.startswith("/") or ":" in rel:
        raise HTTPException(400, f"Недопустимый путь картинки: {rel}")
    src = (RUNS_DIR / rel).resolve()
    try:
        src.relative_to(RUNS_DIR.resolve())
    except ValueError as e:
        raise HTTPException(400, f"Путь вне runs/: {rel}") from e
    if not src.is_file():
        raise HTTPException(404, f"Картинка не найдена: {rel}")
    dest = assets_dir / f"slot_{slot_idx:02d}{src.suffix or '.png'}"
    dest.write_bytes(src.read_bytes())
    return str(dest)


def _read_run_image_bytes(rel: str) -> bytes:
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.startswith("runs/"):
        rel = rel[5:]
    rel = rel.split("?", 1)[0]
    if ".." in rel or rel.startswith("/") or ":" in rel:
        raise HTTPException(400, f"Недопустимый путь: {rel}")
    src = (RUNS_DIR / rel).resolve()
    try:
        src.relative_to(RUNS_DIR.resolve())
    except ValueError as e:
        raise HTTPException(400, f"Путь вне runs/: {rel}") from e
    if not src.is_file():
        raise HTTPException(404, f"Картинка не найдена: {rel}")
    return src.read_bytes()


@router.post("/name-from-art")
async def name_from_art(
    image: UploadFile | None = File(None),
    image_rel: str | None = Form(None),
    preset_id: str | None = Form(None),
) -> dict:
    """Vision: подобрать name_ru / name_en по картинке слота."""
    data: bytes | None = None
    if image is not None and image.filename:
        data = await image.read()
    elif image_rel:
        data = _read_run_image_bytes(image_rel)
    if not data:
        raise HTTPException(400, "Нужна картинка слота (файл или image_rel)")

    try:
        preset = load_preset(preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    # critic — vision-роль; иначе default (тоже часто умеет картинки).
    variant = resolve_variant(preset, "critic")
    prompt = engine.render("name_from_art")
    reg = Registry(providers_config(), offline=is_offline())
    try:
        resp = await reg.text(variant, prompt, images=[data], json_mode=True)
        parsed = engine.parse_json_response(resp.text)
    except (ProviderError, ValueError) as e:
        raise HTTPException(502, str(e)) from e

    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise HTTPException(502, "Модель вернула не объект")

    name_ru = str(parsed.get("name_ru") or "").strip()
    name_en = str(parsed.get("name_en") or "").strip()
    if not name_ru and not name_en:
        raise HTTPException(502, "Пустые имена в ответе модели")

    # Формат для неймплейта: «Хлопушка / Clapperboard» (или одно, если второго нет).
    if name_ru and name_en:
        display = f"{name_ru} / {name_en}"
    else:
        display = name_ru or name_en

    return {
        "name_ru": name_ru,
        "name_en": name_en,
        "name": display,
        "model": getattr(resp, "model", None) or variant.model_name,
        "cost_usd": float(getattr(resp, "cost_usd", 0) or 0),
    }


@router.get("/defaults")
async def defaults() -> dict:
    """Пути и дефолты для UI."""
    return {
        "slots": 10,
        "title": "Collection",
        "page_enabled": False,  # временно: футер из заглушки BackGround
        "cutted_ok": CUTTED_DIR.is_dir(),
        "icons_ok": ICONS_DIR.is_dir(),
        "reference_ok": REFERENCE.is_file(),
        "builder_ok": (BUILDER_DIR / "build_offer.py").is_file(),
    }


@router.post("/build")
async def build_collection(
    title: str = Form(...),
    slots: str = Form(...),  # JSON-массив из 10 слотов
    # Номер страницы временно отключён — в футере остаётся заглушка из BackGround.
    image_0: UploadFile | None = File(None),
    image_1: UploadFile | None = File(None),
    image_2: UploadFile | None = File(None),
    image_3: UploadFile | None = File(None),
    image_4: UploadFile | None = File(None),
    image_5: UploadFile | None = File(None),
    image_6: UploadFile | None = File(None),
    image_7: UploadFile | None = File(None),
    image_8: UploadFile | None = File(None),
    image_9: UploadFile | None = File(None),
) -> dict:
    """Собирает PNG коллекции и кладёт в runs/collection/<id>/result.png."""
    if not (BUILDER_DIR / "build_offer.py").is_file():
        raise HTTPException(500, f"OfferBuilder не найден: {BUILDER_DIR}")
    if not CUTTED_DIR.is_dir():
        raise HTTPException(400, f"Нет материалов Cutted: {CUTTED_DIR}")
    if not REFERENCE.is_file():
        raise HTTPException(400, f"Нет референса: {REFERENCE}")

    try:
        raw_slots = json.loads(slots)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"slots: невалидный JSON ({e})") from e
    if not isinstance(raw_slots, list) or len(raw_slots) != 10:
        raise HTTPException(400, "Нужно ровно 10 слотов в поле slots")

    uploads = [
        image_0, image_1, image_2, image_3, image_4,
        image_5, image_6, image_7, image_8, image_9,
    ]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = COLLECTION_RUNS / run_id
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ob = _load_builder()
    items = []
    for i, raw in enumerate(raw_slots):
        try:
            slot = SlotIn(**raw)
        except Exception as e:  # pydantic ValidationError
            raise HTTPException(400, f"Слот {i + 1}: {e}") from e

        image_path = None
        up = uploads[i]
        if up is not None and up.filename:
            data = await up.read()
            if data:
                ext = Path(up.filename).suffix.lower() or ".png"
                if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                    ext = ".png"
                dest = assets_dir / f"slot_{i + 1:02d}{ext}"
                dest.write_bytes(data)
                image_path = str(dest)
        if image_path is None and slot.image_rel:
            image_path = _resolve_run_image(slot.image_rel, assets_dir, i + 1)

        badge = f"+{slot.enchant_power}" if slot.enchant_on else None
        items.append(ob.CardItem(
            name=slot.name.strip() or f"Item {i + 1}",
            rarity=slot.rarity,
            image=image_path,
            badge=badge,
        ))

    config = ob.OfferConfig(
        title=(title or "").strip() or "Collection",
        page=None,  # футер SET N/M — стартовая заглушка из BackGround.png
        show_progressbar=True,
        items=items,
    )

    try:
        result = ob.build_offer(config, CUTTED_DIR, REFERENCE, ICONS_DIR)
    except Exception as e:
        log.exception("Сборка коллекции упала")
        raise HTTPException(500, str(e)) from e

    out_file = out_dir / "result.png"
    result.save(out_file)
    meta = {
        "run_id": run_id,
        "title": config.title,
        "page": None,
        "page_total": None,
        "slots": [
            {
                "name": it.name,
                "rarity": it.rarity,
                "badge": it.badge,
                "has_image": bool(it.image),
            }
            for it in items
        ],
        "size": list(result.size),
        "file": f"collection/{run_id}/result.png",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("Коллекция собрана: %s (%dx%d)", out_file, *result.size)
    return meta
