"""Разбор эталонной карточки на составляющие.

Генератор изображений плохо рисует интерфейс: рамки кривые, надписи
искажены, бейджи не читаются. Поэтому UI не генерируется, а разбирается
из готовой карточки один раз и потом переиспользуется.

Vision-модель размечает, где какой элемент, Pillow режет по разметке и
складывает в библиотеку как новую версию. Дальше сборка берёт конкретные
версии — и результат перестаёт зависеть от настроения модели.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw

from app import parts
from app.config import resolve_variant
from app.models import Preset
from app.parts import PART_KINDS, FontStyle
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry

log = logging.getLogger("offerforge.dissect")

# Разбираем только то, что имеет смысл вырезать картинкой.
# font описывается метаданными, art — это окно под генерацию.
CUTTABLE = ("frame", "stars", "button", "badge", "nameplate")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _to_pixels(box: dict, w: int, h: int) -> tuple[int, int, int, int]:
    x = _clamp01(box.get("x", 0)) * w
    y = _clamp01(box.get("y", 0)) * h
    bw = _clamp01(box.get("w", 0)) * w
    bh = _clamp01(box.get("h", 0)) * h
    left, top = int(round(x)), int(round(y))
    right, bottom = int(round(min(x + bw, w))), int(round(min(y + bh, h)))
    return left, top, max(right, left + 1), max(bottom, top + 1)


async def analyze(
    reg: Registry, preset: Preset, image: bytes
) -> dict[str, Any]:
    """Просит vision-модель разметить элементы карточки."""
    img = Image.open(BytesIO(image))
    prompt = engine.render(
        "dissect", width=img.width, height=img.height,
        kinds={k: v for k, v in PART_KINDS.items() if k != "font"},
    )
    variant = resolve_variant(preset, "critic")   # роль с гарантированным vision

    resp = await reg.text(variant, prompt, images=[image], json_mode=True)
    data = engine.parse_json_response(resp.text)
    if not isinstance(data, dict) or not data.get("parts"):
        raise ProviderError("модель не вернула разметку составляющих")
    return data


def _punch_window(frame: Image.Image, window: tuple[int, int, int, int]) -> Image.Image:
    """Делает прозрачным окно под арт внутри рамки.

    Рамка вырезается прямоугольником вместе со всем, что внутри. Если не
    выбить в ней окно, она ляжет поверх арта и закроет его целиком —
    именно так и выглядит «рамка есть, картинки нет».
    """
    left, top, right, bottom = window
    if right - left < 4 or bottom - top < 4:
        return frame

    mask = Image.new("L", frame.size, 255)
    radius = max(4, min(right - left, bottom - top) // 12)
    ImageDraw.Draw(mask).rounded_rectangle(
        [left, top, right - 1, bottom - 1], radius=radius, fill=0)
    frame = frame.copy()
    frame.putalpha(Image.composite(
        frame.getchannel("A"), Image.new("L", frame.size, 0), mask))
    return frame


def cut(image: bytes, markup: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    """Режет карточку по разметке. Возвращает {тип: {имя_файла: png}}."""
    img = Image.open(BytesIO(image)).convert("RGBA")
    out: dict[str, dict[str, bytes]] = {}

    # Где на эталоне находится арт — понадобится, чтобы выбить окно в рамке.
    art_box = next(
        (i.get("box") for i in markup.get("parts", []) if i.get("kind") == "art"),
        None,
    )

    for item in markup.get("parts", []):
        kind = item.get("kind")
        if kind not in CUTTABLE:
            continue
        box = item.get("box") or {}
        left, top, right, bottom = _to_pixels(box, img.width, img.height)
        if right - left < 4 or bottom - top < 4:
            log.warning("Пропускаю %s: бокс вырожденный", kind)
            continue

        crop = img.crop((left, top, right, bottom))

        if kind == "frame":
            if art_box:
                al, at, ar, ab = _to_pixels(art_box, img.width, img.height)
                crop = _punch_window(crop, (al - left, at - top, ar - left, ab - top))
            else:
                # Модель не разметила арт — вырезаем окно по внутреннему
                # отступу рамки. Лучше приблизительное окно, чем глухая
                # заливка поверх картинки.
                w, h = crop.size
                pad_x, pad_y = int(w * 0.09), int(h * 0.13)
                crop = _punch_window(
                    crop, (pad_x, pad_y, w - pad_x, int(h * 0.78)))
                log.info("Разбор: арт не размечен, окно в рамке взято по отступу")

        buf = BytesIO()
        crop.save(buf, format="PNG")

        # Звёзды на эталоне показаны одним количеством. Кладём вырезанное
        # как есть, а недостающие варианты 1..5 добираются композитором из
        # того, что нашлось, — иначе пришлось бы требовать пять эталонов.
        name = "frame.png" if kind == "frame" else {
            "stars": "stars.png", "button": "buy.png",
            "badge": "badge.png", "nameplate": "plate.png",
        }[kind]
        out.setdefault(kind, {})[name] = buf.getvalue()

    return out


def anchors(markup: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Относительные координаты элементов — подсказка для шаблона."""
    return {
        item["kind"]: item["box"]
        for item in markup.get("parts", [])
        if item.get("kind") and item.get("box")
    }


async def dissect_card(
    reg: Registry,
    preset: Preset,
    image: bytes,
    version: str,
    *,
    note: str = "",
    make_default: bool = True,
    on_event=None,
) -> dict[str, Any]:
    """Полный цикл: разметить, нарезать, сохранить как версию каждого типа."""
    if on_event:
        on_event("stage", {"name": "analyze", "status": "start"})
    markup = await analyze(reg, preset, image)
    found = [p.get("kind") for p in markup.get("parts", [])]
    if on_event:
        on_event("stage", {"name": "analyze", "status": "done",
                           "found": found, "summary": markup.get("summary", "")})

    pieces = cut(image, markup)
    boxes = anchors(markup)
    saved: list[dict] = []

    for kind, assets in pieces.items():
        meta = parts.save_version(
            kind, version,
            assets=assets,
            title=version,
            note=note or markup.get("summary", "")[:200],
            source="разбор карточки",
            anchor=boxes.get(kind),
            make_default=make_default,
        )
        saved.append({"kind": kind, "version": meta.version,
                      "assets": meta.assets, "anchor": meta.anchor})
        if on_event:
            on_event("part", {"kind": kind, "version": meta.version,
                              "assets": meta.assets})

    # Шрифт — не картинка: сохраняем описанием, чтобы композитор рисовал
    # текст сам и не получал его от генератора искажённым.
    font_spec = markup.get("font") or {}
    if font_spec:
        img = Image.open(BytesIO(image))
        size = int(round(float(font_spec.get("size_rel", 0.055)) * img.height))
        font = FontStyle(
            size=max(size, 10),
            color=font_spec.get("color") or "#FFF3D0",
            stroke_color=font_spec.get("stroke_color") or "#5A2E00",
            stroke_width=int(font_spec.get("stroke_width", 5)),
            uppercase=bool(font_spec.get("uppercase", True)),
        )
        meta = parts.save_version(
            "font", version, title=version,
            note=font_spec.get("note", ""), source="разбор карточки",
            font=font, make_default=make_default,
        )
        saved.append({"kind": "font", "version": meta.version,
                      "assets": [], "font": font.model_dump()})
        if on_event:
            on_event("part", {"kind": "font", "version": meta.version, "assets": []})

    if on_event:
        on_event("done", {"version": version, "saved": len(saved),
                          "kinds": [s["kind"] for s in saved]})

    return {
        "version": version,
        "summary": markup.get("summary", ""),
        "palette": markup.get("palette", []),
        "saved": saved,
        "markup": markup,
    }
