"""Эндпоинты ручной разметки скина.

Vision ставит боксы приблизительно и путает соседние элементы. Курсором
это делается точно и один раз, поэтому размеченное вручную имеет приоритет
над автоматикой: если разметка есть, модель о координатах не спрашивают.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from app import parts, skins
from app.config import is_offline, load_preset, providers_config
from app.pipeline import regen, segment
from app.providers.registry import Registry

log = logging.getLogger("offerforge.markup")
router = APIRouter(prefix="/api/markup", tags=["markup"])


@router.get("")
async def list_all() -> dict:
    """Загруженные скины и типы элементов, которыми можно размечать."""
    return {
        "skins": skins.list_skins(),
        "kinds": [
            {"kind": k, "title": v["title"], "note": v["note"]}
            for k, v in parts.PART_KINDS.items() if k != "font"
        ] + [{"kind": "art", "title": "Окно под арт",
              "note": "Область внутри слота, куда встанет картинка предмета."}],
    }


@router.post("/upload")
async def upload(file: UploadFile = File(...), version: str = Form(...)) -> dict:
    """Загружает скин и отдаёт его как data-URL для холста разметки."""
    image = await file.read()
    if not image:
        raise HTTPException(400, "Пустой файл")
    if len(image) > 40 * 1024 * 1024:
        raise HTTPException(400, "Файл больше 40 МБ")

    skins.save_screen(version, image)
    img = Image.open(BytesIO(image))
    # Для холста отдаём уменьшенную копию: скины бывают 5000+ px, и гонять
    # их в браузер целиком незачем — разметка всё равно в долях.
    preview = img.convert("RGBA")
    preview.thumbnail((1600, 1600), Image.LANCZOS)
    buf = BytesIO()
    preview.save(buf, format="PNG")

    return {
        "version": version,
        "width": img.width, "height": img.height,
        "preview": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "markup": skins.load_markup(version),
    }


@router.get("/{version}/screen")
async def screen(version: str):
    path = skins.skin_dir(version) / "screen.png"
    if not path.exists():
        raise HTTPException(404, f"Скин {version} не загружен")
    return FileResponse(path, media_type="image/png")


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class MarkupPart(BaseModel):
    kind: str
    box: Box
    # Контур лассо: точки [x, y] в долях экрана. Если задан — режем по нему,
    # а бокс остаётся только описанием габаритов.
    polygon: list[list[float]] | None = None
    stars_count: int | None = None
    note: str = ""


class MarkupBody(BaseModel):
    parts: list[MarkupPart] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    summary: str = ""
    slots: dict | None = None
    font: dict | None = None


@router.put("/{version}")
async def save(version: str, body: MarkupBody) -> dict:
    if not skins.has_screen(version):
        raise HTTPException(404, f"Скин {version} не загружен")
    if not body.parts:
        raise HTTPException(400, "Разметка пуста — выдели хотя бы один элемент")

    markup = body.model_dump(exclude_none=True)
    skins.save_markup(version, markup, source="вручную")
    return {"ok": True, "version": version, "parts": len(body.parts)}


@router.get("/{version}")
async def get(version: str) -> dict:
    markup = skins.load_markup(version)
    if markup is None:
        raise HTTPException(404, f"Разметки для {version} нет")
    return markup


@router.delete("/{version}")
async def remove(version: str) -> dict:
    skins.delete_skin(version)
    return {"ok": True}


class ApplyBody(BaseModel):
    version: str
    make_default: bool = True
    preset_id: str | None = None
    # Чем снимать фон вокруг вырезанного: ничем / своим матированием /
    # image-edit моделью.
    extract_mode: str = "model"
    regenerate: bool = False        # ещё и перерисовать элементы заново


@router.post("/apply")
async def apply(body: ApplyBody) -> dict:
    """Режет скин по ручной разметке и при желании сразу перерисовывает.

    Vision здесь не вызывается вовсе: координаты уже известны точно.
    """
    if body.extract_mode not in ("raw", "local", "model"):
        raise HTTPException(400, f"Неизвестный режим извлечения: {body.extract_mode}")
    image = skins.load_screen(body.version)
    if image is None:
        raise HTTPException(404, f"Скин {body.version} не загружен")
    markup = skins.load_markup(body.version)
    if markup is None:
        raise HTTPException(400, "Сначала размести элементы на холсте")

    preset = load_preset(body.preset_id)
    reg = Registry(providers_config(), offline=is_offline() or preset.id == "offline")
    events: list[dict] = []

    try:
        cut_result = await segment.segment_skin(
            reg, preset, image, body.version,
            make_default=body.make_default, markup=markup,
            extract_mode=body.extract_mode,
            on_event=lambda k, p: events.append({"kind": k, **p}),
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Нарезка по разметке упала")
        raise HTTPException(400, str(e).replace("\n", " ")[:400]) from e

    result: dict = {
        "version": body.version,
        "cut": [s["kind"] for s in cut_result["saved"]],
        "references": len(cut_result["references"]),
        "events": events,
        # Показываем именно вырезанное: это и есть результат, а не полуфабрикат
        # перед генерацией. Из этих кусков собирается оффер.
        "pieces": _gallery(body.version, cut_result["saved"]),
        "checks": cut_result.get("checks") or [],
        "rebuild": _rebuild_preview(image, cut_result),
    }

    if body.regenerate:
        target = f"{body.version}-gen"
        try:
            # Полный экран уходит в модель вместе с образцом: по одному
            # обрезку она не понимает ни назначения элемента, ни масштаба.
            gen = await regen.regenerate_skin(
                reg, preset, body.version, target,
                make_default=body.make_default, screen=image,
                on_event=lambda k, p: events.append({"kind": k, **p}),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Регенерация по разметке упала")
            raise HTTPException(400, str(e).replace("\n", " ")[:400]) from e

        result["regenerated"] = {
            "target": target,
            "ok": [p["kind"] for p in gen["parts"]],
            "failed": [f["kind"] for f in gen["failed"]],
        }
        result["pairs"] = _pairs(body.version, target, gen)

    return result


def _rebuild_preview(screen: bytes, result: dict) -> dict | None:
    """Сборка из деталей и карта расхождений — рядом с оригиналом.

    Числа в отчёте показывают, что не сошлось, но не показывают почему.
    Картинка показывает: видно, что рамка приехала вместе с плашкой, что
    звёзды встали не на ту высоту, что шаг сетки уехал.
    """
    from app.pipeline import verify as verify_lib

    markup = result.get("markup") or {}
    pieces: dict[str, dict[str, bytes]] = {}
    for item in result.get("saved", []):
        kind = item["kind"]
        if kind == "font":
            continue
        meta = parts.load_version(kind, item["version"])
        if not meta or not meta.assets:
            continue
        path = meta.asset_path(meta.assets[0])
        if path.exists():
            pieces[kind] = {meta.assets[0]: path.read_bytes()}
    if not pieces:
        return None

    original = Image.open(BytesIO(screen)).convert("RGBA")
    built, _ = verify_lib.rebuild(original.size, markup, pieces)

    def url(img: Image.Image, side: int = 900) -> str:
        copy = img.copy()
        copy.thumbnail((side, side), Image.LANCZOS)
        buf = BytesIO()
        copy.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "original": url(original),
        "rebuilt": url(built),
        "diff": url(verify_lib.diff_map(original, built)),
    }


def _gallery(version: str, saved: list[dict]) -> list[dict]:
    """Вырезанные куски как data-URL — по всем ассетам каждого элемента."""
    out = []
    for item in saved:
        kind = item["kind"]
        if kind == "font":
            continue
        meta = parts.load_version(kind, version)
        if not meta:
            continue
        shots = []
        for name in meta.assets:
            path = meta.asset_path(name)
            if not path.exists():
                continue
            with Image.open(path) as im:
                w, h = im.size
            shots.append({
                "name": name, "w": w, "h": h,
                "src": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
            })
        if shots:
            out.append({
                "kind": kind,
                "title": parts.PART_KINDS.get(kind, {}).get("title", kind),
                "assets": shots,
            })
    return out


def _pairs(source: str, target: str, gen: dict) -> list[dict]:
    """Пары «вырезано → перерисовано» для показа рядом."""
    out = []
    for item in gen["parts"]:
        kind = item["kind"]
        src, made = parts.load_version(kind, source), parts.load_version(kind, target)
        if not (src and made and src.assets and made.assets):
            continue
        sp, mp = src.asset_path(src.assets[0]), made.asset_path(made.assets[0])
        if sp.exists() and mp.exists():
            out.append({
                "kind": kind,
                "sample": "data:image/png;base64," + base64.b64encode(sp.read_bytes()).decode(),
                "made": "data:image/png;base64," + base64.b64encode(mp.read_bytes()).decode(),
                "assets": made.assets,
            })
    return out
