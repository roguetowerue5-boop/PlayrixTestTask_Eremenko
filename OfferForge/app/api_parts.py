"""Эндпоинты экрана «Составляющие».

Разбор эталонной карточки на элементы, библиотека версий и выбор версии
по умолчанию. Всё, что здесь лежит, потом собирается композитором — и
не проходит через генератор изображений.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import parts
from app.config import is_offline, load_preset, providers_config
from app.pipeline import dissect, regen, segment
from app.providers.registry import Registry

log = logging.getLogger("offerforge.parts")
router = APIRouter(prefix="/api/parts", tags=["parts"])


@router.get("")
async def get_parts() -> dict:
    return {
        "kinds": parts.list_kinds(),
        "defaults": parts.defaults(),
    }


@router.get("/{kind}/{version}/asset/{name}")
async def get_asset(kind: str, version: str, name: str):
    """Превью файла составляющей."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Недопустимое имя файла")
    meta = parts.load_version(kind, version)
    if meta is None:
        raise HTTPException(404, f"Версия {kind}/{version} не найдена")
    path = meta.asset_path(name)
    if not path.exists():
        raise HTTPException(404, f"Файл {name} не найден")
    return FileResponse(path, media_type="image/png")


class DefaultBody(BaseModel):
    version: str


@router.put("/{kind}/default")
async def put_default(kind: str, body: DefaultBody) -> dict:
    if parts.load_version(kind, body.version) is None:
        raise HTTPException(404, f"Версия {kind}/{body.version} не найдена")
    parts.set_default(kind, body.version)
    return {"ok": True, "kind": kind, "default": body.version}


@router.delete("/{kind}/{version}")
async def remove_version(kind: str, version: str) -> dict:
    try:
        parts.delete_version(kind, version)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True}


class Target(BaseModel):
    """Одна цель поиска: элемент, который просят найти отдельным запросом."""
    kind: str
    title: str = ""
    en: str = ""
    hint: str = ""
    level: str = "cell"
    zone: list[float] = [0.0, 0.0, 1.0, 1.0]
    counted: bool = False
    hollow: bool = False


@router.get("/targets")
async def get_targets() -> dict:
    from app.config import segment_targets

    return {"targets": segment_targets()}


@router.put("/targets")
async def put_targets(body: list[Target]) -> dict:
    """Перезаписывает список целей.

    Набор элементов у каждого скина свой, поэтому список правится из
    интерфейса, а не в коде. Порядок сохраняется: запросы идут в нём.
    """
    from app.config import save_segment_targets

    seen: set[str] = set()
    clean: list[dict] = []
    for t in body:
        kind = t.kind.strip()
        if not kind or kind in seen:
            continue
        seen.add(kind)
        zone = t.zone if len(t.zone) == 4 else [0.0, 0.0, 1.0, 1.0]
        clean.append({
            "kind": kind,
            "title": t.title.strip() or kind,
            "en": t.en.strip() or kind,
            "hint": t.hint.strip(),
            "level": "cell" if t.level == "cell" else "screen",
            "zone": [round(float(v), 3) for v in zone],
            "counted": t.counted,
            "hollow": t.hollow,
        })
    if not clean:
        raise HTTPException(400, "Список целей пуст — искать будет нечего")

    save_segment_targets(clean)
    return {"ok": True, "count": len(clean)}


@router.post("/dissect")
async def dissect_upload(
    file: UploadFile = File(...),
    version: str = Form(...),
    note: str = Form(""),
    preset_id: str | None = Form(None),
    make_default: bool = Form(True),
) -> dict:
    """Разбирает загруженную карточку на составляющие.

    Vision-модель размечает элементы, Pillow режет по разметке, результат
    ложится в библиотеку новой версией. Дальше сборка берёт её как есть —
    интерфейс перестаёт зависеть от генератора.
    """
    image = await file.read()
    if not image:
        raise HTTPException(400, "Пустой файл")
    if len(image) > 12 * 1024 * 1024:
        raise HTTPException(400, "Файл больше 12 МБ")

    preset = load_preset(preset_id)
    reg = Registry(providers_config(), offline=is_offline() or preset.id == "offline")

    events: list[dict] = []
    try:
        result = await dissect.dissect_card(
            reg, preset, image, version,
            note=note, make_default=make_default,
            on_event=lambda kind, payload: events.append({"kind": kind, **payload}),
        )
    except Exception as e:  # noqa: BLE001 — причину показываем пользователю
        log.exception("Разбор карточки упал")
        raise HTTPException(400, str(e).replace("\n", " ")[:400]) from e

    # Возвращаем превью нарезанного, чтобы результат было видно сразу.
    previews = {}
    for item in result["saved"]:
        kind = item["kind"]
        for name in item.get("assets", []):
            meta = parts.load_version(kind, item["version"])
            if meta:
                p = meta.asset_path(name)
                if p.exists() and p.stat().st_size < 2_000_000:
                    previews[f"{kind}/{name}"] = (
                        "data:image/png;base64,"
                        + base64.b64encode(p.read_bytes()).decode()
                    )

    return {
        "version": result["version"],
        "summary": result["summary"],
        "palette": result["palette"],
        "saved": result["saved"],
        "previews": previews,
        "events": events,
    }


@router.post("/segment")
async def segment_upload(
    file: UploadFile = File(...),
    version: str = Form(...),
    note: str = Form(""),
    preset_id: str | None = Form(None),
    make_default: bool = Form(True),
    extract_mode: str = Form("model"),
) -> dict:
    """Сегментация игрового UI-скина на элементы экрана.

    Отличается от /dissect тем, что разбирает ЭКРАН целиком: панель, ленту,
    прогресс-бар, крестик, футер и один слот карточки. Вырезанные куски
    дополнительно сохраняются как референсы стиля для генерации арта.

    extract_mode задаёт, чем снимать фон: "raw" — ничем, "local" — своим
    матированием, "model" — image-edit моделью.
    """
    if extract_mode not in ("raw", "local", "model"):
        raise HTTPException(400, f"Неизвестный режим извлечения: {extract_mode}")
    image = await file.read()
    if not image:
        raise HTTPException(400, "Пустой файл")
    if len(image) > 40 * 1024 * 1024:
        raise HTTPException(400, "Файл больше 40 МБ")

    preset = load_preset(preset_id)
    reg = Registry(providers_config(), offline=is_offline() or preset.id == "offline")

    events: list[dict] = []
    try:
        result = await segment.segment_skin(
            reg, preset, image, version, note=note, make_default=make_default,
            extract_mode=extract_mode,
            on_event=lambda kind, payload: events.append({"kind": kind, **payload}),
        )
    except Exception as e:  # noqa: BLE001 — причину показываем пользователю
        log.exception("Сегментация скина упала")
        raise HTTPException(400, str(e).replace("\n", " ")[:400]) from e

    previews = {}
    for item in result["saved"]:
        meta = parts.load_version(item["kind"], item["version"])
        for name in item.get("assets", []):
            if not meta:
                continue
            p = meta.asset_path(name)
            if p.exists() and p.stat().st_size < 3_000_000:
                previews[f"{item['kind']}/{name}"] = (
                    "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
                )

    from app.api_markup import _rebuild_preview

    return {
        "version": result["version"],
        "summary": result["summary"],
        "palette": result["palette"],
        "slots": result["slots"],
        "saved": result["saved"],
        "references": result["references"],
        "previews": previews,
        "events": events,
        "checks": result.get("checks") or [],
        "rebuild": _rebuild_preview(image, result),
    }


class RegenBody(BaseModel):
    source: str                          # версия, полученная сегментацией
    target: str | None = None            # куда сложить, по умолчанию "<source>-gen"
    kinds: list[str] | None = None       # None — все, что есть в источнике
    preset_id: str | None = None
    make_default: bool = True


@router.post("/regenerate")
async def regenerate(body: RegenBody) -> dict:
    """Перерисовывает вырезанные элементы моделью по образцу.

    Сегментация даёт куски скриншота — с подложкой панели, без прозрачности
    и в одном экземпляре. Здесь каждый кусок идёт в image-модель референсом
    с требованием повторить один в один, но изолированно и на чистом фоне.
    В сборку после этого попадает настоящий ассет, а не фрагмент скрина.
    """
    target = body.target or f"{body.source}-gen"
    preset = load_preset(body.preset_id)
    reg = Registry(providers_config(), offline=is_offline() or preset.id == "offline")

    events: list[dict] = []
    try:
        result = await regen.regenerate_skin(
            reg, preset, body.source, target,
            kinds=body.kinds, make_default=body.make_default,
            on_event=lambda kind, payload: events.append({"kind": kind, **payload}),
        )
    except Exception as e:  # noqa: BLE001 — причину показываем пользователю
        log.exception("Регенерация элементов упала")
        raise HTTPException(400, str(e).replace("\n", " ")[:400]) from e

    # Пары «образец → результат»: без сравнения бок о бок непонятно,
    # насколько модель попала.
    pairs = []
    for item in result["parts"]:
        kind = item["kind"]
        src = parts.load_version(kind, body.source)
        made = parts.load_version(kind, item["version"])
        if not (src and made and src.assets and made.assets):
            continue
        sp, mp = src.asset_path(src.assets[0]), made.asset_path(made.assets[0])
        if sp.exists() and mp.exists():
            pairs.append({
                "kind": kind,
                "sample": "data:image/png;base64," + base64.b64encode(sp.read_bytes()).decode(),
                "made": "data:image/png;base64," + base64.b64encode(mp.read_bytes()).decode(),
                "assets": made.assets,
            })

    return {
        "source": result["source"], "target": result["target"],
        "ok": len(result["parts"]), "failed": result["failed"],
        "pairs": pairs, "events": events,
    }


@router.get("/skins")
async def get_skins() -> dict:
    """Скины, для которых нарезаны референсы стиля."""
    from app.config import available_skins

    return {"skins": available_skins()}


class ManualBody(BaseModel):
    kind: str
    version: str
    title: str = ""
    note: str = ""
    make_default: bool = False


@router.post("/manual")
async def add_manual(
    file: UploadFile | None = File(None),
    kind: str = Form(...),
    version: str = Form(...),
    asset_name: str = Form(""),
    note: str = Form(""),
    make_default: bool = Form(False),
) -> dict:
    """Добавить готовый файл составляющей без разбора."""
    assets = {}
    if file is not None:
        data = await file.read()
        if data:
            name = asset_name or Path(file.filename or "asset.png").name
            assets[name] = data
    try:
        meta = parts.save_version(
            kind, version, assets=assets, title=version,
            note=note, source="загружено вручную", make_default=make_default,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return json.loads(meta.model_dump_json())
