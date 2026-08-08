"""Сегментация игрового UI-скина на переиспользуемые элементы.

Генератор изображений не умеет рисовать интерфейс: рамки кривые, надписи
поедут, бейджи не читаются. Поэтому интерфейс берётся из готового скина —
его один раз режут на элементы, а дальше собирают композитором.

Работает в два шага, потому что ни один из них по отдельности не годится:

  1. Vision размечает боксы — он видит семантику («вот рамка слота»),
     но координаты даёт приблизительные.
  2. Pillow режет по боксу и обрезает прозрачные поля по альфа-каналу —
     это возвращает точные границы, которые vision дать не может.

Автосегментация только по альфе не работает: в реальном скине весь экран —
один связный непрозрачный компонент.
"""
from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app import parts
from app.config import resolve_variant, segment_targets
from app.models import Preset
from app.parts import PART_KINDS, SCREEN_KINDS, FontStyle
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry

log = logging.getLogger("offerforge.segment")

# Максимальная сторона картинки, уходящей в vision. Скины бывают 5000+ px,
# а модели такое либо режут сами, либо считают втридорога.
VISION_MAX_SIDE = 1600

# Элемент меньше этого в пикселях — почти наверняка промах разметки.
MIN_PIECE = 8


def _opaque_bounds(img: Image.Image) -> dict[str, int] | None:
    """Границы непрозрачного содержимого — подсказка для vision."""
    if "A" not in img.mode:
        return None
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        return None
    return {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}


def _downscale(img: Image.Image, max_side: int = VISION_MAX_SIDE) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    k = max_side / max(img.size)
    return img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)


def _to_pixels(box: dict, w: int, h: int) -> tuple[int, int, int, int]:
    def clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    x, y = clamp(box.get("x", 0)) * w, clamp(box.get("y", 0)) * h
    bw, bh = clamp(box.get("w", 0)) * w, clamp(box.get("h", 0)) * h
    left, top = int(round(x)), int(round(y))
    right = int(round(min(x + bw, w)))
    bottom = int(round(min(y + bh, h)))
    return left, top, max(right, left + 1), max(bottom, top + 1)


def _trim(img: Image.Image, pad: int = 2) -> Image.Image:
    """Срезает прозрачные поля вокруг элемента.

    Именно это чинит неточность vision: он ставит бокс «примерно», а
    альфа-канал знает настоящую границу до пикселя.
    """
    if "A" not in img.mode:
        return img
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        return img
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.width, bbox[2] + pad)
    bottom = min(img.height, bbox[3] + pad)
    return img.crop((left, top, right, bottom))


# Элементы, которые ищутся на общем плане, и те, что внутри карточки.
SCREEN_LEVEL = ("panel", "ribbon", "progress")
CELL_LEVEL = ("frame", "stars", "badge", "nameplate")

# Подсказки для крупного плана: на кропе одной карточки формулировки могут
# быть предметными, потому что элемент занимает заметную часть кадра.
CELL_HINTS = {
    "frame": "The outer border of the card, covering the whole close-up.",
    "stars": "Row of small stars above the card, showing rarity.",
    "badge": "Small tag with a plus and a number, usually top-right.",
    "nameplate": "Plate with the item name along the bottom of the card.",
}

# Зона поиска внутри карточки — доли бокса ячейки (x, y, w, h) с полями.
# Спрашивать про плашку имени, показывая всю карточку, значит дать модели
# шанс перепутать её со звёздами и бейджем. В своей зоне элемент занимает
# половину кадра, и путать его не с чем.
CELL_ZONES = {
    "frame":     (-0.10, -0.10, 1.20, 1.20),   # рамка — вся карточка с полем
    "stars":     (-0.15, -0.22, 1.30, 0.45),   # над карточкой
    # Бейдж сидит на правом краю и заметно выходит за него — зона шире
    # карточки, иначе центр бейджа в неё не попадает и он теряется.
    "badge":     ( 0.35, -0.15, 1.05, 0.70),
    "nameplate": (-0.10,  0.62, 1.20, 0.50),   # низ карточки
}

# Элементы, для которых нужно число копий в кадре.
COUNTED = {"stars"}


async def analyze(reg: Registry, preset: Preset, image: bytes) -> dict[str, Any]:
    """Размечает экран в два прохода.

    Одним проходом не выходит. На полном скине 5482×2529 звезда занимает
    меньше процента кадра, и модель называет её координаты наугад — отсюда
    и брались боксы, попадавшие в пустоту. Поэтому:

    1. Общий план — только крупное: панель, лента, прогресс и, главное,
       границы сетки карточек.
    2. По найденной сетке вырезается ОДНА карточка и увеличивается. На
       крупном плане звезда и бейдж занимают уже десятки процентов кадра,
       и модель показывает их уверенно.

    Координаты второго прохода пересчитываются в координаты экрана.
    """
    img = Image.open(BytesIO(image)).convert("RGBA")
    small = _downscale(img)
    buf = BytesIO()
    small.save(buf, format="PNG")

    variant = resolve_variant(preset, "critic")   # роль с гарантированным vision

    prompt = engine.render(
        "segment",
        width=small.width, height=small.height,
        opaque=_opaque_bounds(small),
        kinds={k: PART_KINDS[k] for k in SCREEN_LEVEL if k in PART_KINDS},
    )
    resp = await reg.text(variant, prompt, images=[buf.getvalue()], json_mode=True)
    data = engine.parse_json_response(resp.text)
    if not isinstance(data, dict) or not data.get("parts"):
        raise ProviderError("модель не вернула разметку элементов экрана")

    data["parts"] = [p for p in data["parts"]
                     if p.get("kind") in SCREEN_LEVEL]

    # Добираем поштучно то, что общий план потерял или поставил не туда.
    # Прогресс-бар мелкий относительно экрана, и в общем ответе его бокс
    # регулярно приезжал мимо.
    data["parts"] = await _refine_screen(reg, preset, img, data["parts"])

    cell = await analyze_cell(reg, preset, img, data.get("slots") or {})
    data["parts"].extend(cell)

    if len(data["parts"]) < 3:
        # Модель не справилась. Лучше отработать по типовой раскладке, чем
        # не отработать: координаты в ней сняты с реального скина и
        # проверены вырезкой.
        from app.providers.rest import FALLBACK_SCREEN_MARKUP

        log.warning("Vision разметил только %d элементов — беру типовую раскладку",
                    len(data["parts"]))
        fallback = dict(FALLBACK_SCREEN_MARKUP)
        fallback["summary"] = (data.get("summary")
                               or fallback["summary"]) + " (типовая раскладка)"
        return fallback
    return data


# Разумные пределы размера элемента в долях экрана: (мин площадь, макс).
# Взяты с запасом — задача не придраться, а поймать явную чушь вроде
# крестика во весь экран или панели размером с иконку.
SANE_AREA = {
    "panel":     (0.15, 1.00),
    "ribbon":    (0.02, 0.40),
    "progress":  (0.005, 0.30),
    "close":     (0.0001, 0.02),
    "footer":    (0.001, 0.20),
    "frame":     (0.002, 0.20),
    "stars":     (0.0002, 0.05),
    "badge":     (0.0001, 0.03),
    "nameplate": (0.0005, 0.06),
    "button":    (0.0002, 0.10),
    "art":       (0.001, 0.15),
}


def validate_markup(markup: dict[str, Any]) -> list[str]:
    """Ищет в разметке явно невозможное.

    Vision ошибается предсказуемо: даёт крестику половину экрана, ставит
    бокс за границей кадра, забывает сетку. Дешевле поймать это здесь, чем
    потом гадать, почему вырезался кусок фона.
    """
    problems: list[str] = []
    seen: set[str] = set()

    for item in markup.get("parts", []):
        kind = item.get("kind")
        box = item.get("box") or {}
        if not kind:
            continue
        try:
            x, y = float(box["x"]), float(box["y"])
            w, h = float(box["w"]), float(box["h"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{kind}: бокс не читается")
            continue

        if w <= 0 or h <= 0:
            problems.append(f"{kind}: нулевой размер")
        if x < -0.01 or y < -0.01 or x + w > 1.01 or y + h > 1.01:
            problems.append(f"{kind}: бокс выходит за кадр")

        lo, hi = SANE_AREA.get(kind, (0.0, 1.0))
        area = max(0.0, w) * max(0.0, h)
        if area < lo:
            problems.append(f"{kind}: подозрительно мал ({area:.4f} площади)")
        elif area > hi:
            problems.append(f"{kind}: подозрительно велик ({area:.3f} площади)")

        if kind in seen and kind not in ("stars", "badge", "button"):
            problems.append(f"{kind}: размечен дважды")
        seen.add(kind)

    slots = markup.get("slots") or {}
    if not first_cell(slots):
        problems.append("сетка карточек не определена — "
                        "без неё не вырезать одну ячейку")
    return problems


def first_cell(slots: dict[str, Any]) -> dict[str, float] | None:
    """Бокс первой ячейки сетки — из области слотов и числа столбцов."""
    box = slots.get("box") or {}
    cols = int(slots.get("cols", 0) or 0)
    rows = int(slots.get("rows", 0) or 0)
    if cols < 1 or rows < 1 or not box:
        return None
    try:
        w, h = float(box["w"]) / cols, float(box["h"]) / rows
        return {"x": float(box["x"]), "y": float(box["y"]), "w": w, "h": h}
    except (KeyError, TypeError, ValueError):
        return None


# Зоны поиска на уровне экрана — доли непрозрачной области панели.
SCREEN_ZONES = {
    "ribbon":   (0.00, 0.00, 1.00, 0.28),
    "progress": (0.10, 0.10, 0.80, 0.28),
}

# Эти элементы уточняем поштучно. Панель не уточняем: она и так вся
# непрозрачная область, общий план с ней справляется.
REFINE = tuple(SCREEN_ZONES)


async def _refine_screen(
    reg: Registry, preset: Preset, img: Image.Image, parts_list: list[dict],
) -> list[dict]:
    """Переспрашивает мелкие элементы экрана по одному, в своих зонах.

    Общий план видит панель и ленту уверенно, а прогресс-бар и крестик
    занимают проценты кадра — их боксы приезжали мимо, а иногда не
    приезжали вовсе. Отдельный запрос по своей зоне решает и то и другое.
    """
    W, H = img.size
    bounds = _opaque_bounds(img)
    if bounds:
        left, top = int(bounds["x0"]), int(bounds["y0"])
        right, bottom = int(bounds["x1"]), int(bounds["y1"])
    else:
        left, top, right, bottom = 0, 0, W, H

    variant = resolve_variant(preset, "critic")
    # Цели берём из segment-targets.yaml: набор элементов у каждого скина
    # свой, и добавление ещё одного не должно требовать правки кода.
    targets = [t for t in segment_targets()
               if t["level"] == "screen" and t["kind"] != "panel"]
    if not targets:
        return parts_list

    results = await asyncio.gather(*[
        _find_one(reg, variant, img, (left, top, right, bottom),
                  t["kind"], target=t)
        for t in targets
    ], return_exceptions=True)

    pw, ph = (right - left) / W, (bottom - top) / H
    ox, oy = left / W, top / H
    by_kind = {p.get("kind"): p for p in parts_list}

    for kind, res in zip([t["kind"] for t in targets], results):
        if isinstance(res, Exception) or not res:
            continue
        b = res["box"]
        entry = {"kind": kind, "box": {
            "x": round(ox + b["x"] * pw, 4), "y": round(oy + b["y"] * ph, 4),
            "w": round(b["w"] * pw, 4), "h": round(b["h"] * ph, 4)}}
        if "at" in res:
            entry["at"] = [round(ox + res["at"][0] * pw, 4),
                           round(oy + res["at"][1] * ph, 4)]
        by_kind[kind] = entry
        log.info("%s уточнён поштучно", kind)

    return list(by_kind.values())


async def _find_one(
    reg: Registry,
    variant,
    img: Image.Image,
    cell_px: tuple[int, int, int, int],
    kind: str,
    target: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Ищет один элемент в его зоне карточки.

    Возвращает бокс и точку в координатах ЯЧЕЙКИ (не зоны и не экрана):
    вызывающий переведёт их в координаты экрана тем же способом, что и
    раньше, и ему не нужно знать про зоны.
    """
    W, H = img.size
    left, top, right, bottom = cell_px
    cw, chh = right - left, bottom - top

    if target:
        zx, zy, zw, zh = target["zone"]
    else:
        zx, zy, zw, zh = CELL_ZONES.get(kind, (-0.1, -0.1, 1.2, 1.2))

    # Зона переводится в доли ВСЕГО кадра и уходит в промпт подсказкой.
    # Кроп зоны мы больше не отправляем: на вырезанном куске модель не
    # видит, что вокруг, и путает плашку карточки с лентой заголовка —
    # они похожи, а отличить их можно только по месту на экране. С полным
    # кадром такой ошибки нет, а промах мимо зоны отсекается проверкой.
    zone_hint = {
        "x0": round(max(0.0, (left + zx * cw) / W), 3),
        "y0": round(max(0.0, (top + zy * chh) / H), 3),
        "x1": round(min(1.0, (left + (zx + zw) * cw) / W), 3),
        "y1": round(min(1.0, (top + (zy + zh) * chh) / H), 3),
    }
    if zone_hint["x1"] - zone_hint["x0"] < 0.005:
        return None

    shot = _downscale(img)
    buf = BytesIO()
    shot.save(buf, format="PNG")

    prompt = engine.render(
        "segment_one",
        title=(target or {}).get("en")
        or (PART_KINDS.get(kind) or {}).get("en") or kind,
        hint=(target or {}).get("hint") or CELL_HINTS.get(kind, ""),
        zone=zone_hint,
        counted=target["counted"] if target else kind in COUNTED,
        repeated=zone_hint["x1"] - zone_hint["x0"] < 0.5,
    )
    try:
        resp = await reg.text(variant, prompt, images=[buf.getvalue()],
                              json_mode=True)
        data = engine.parse_json_response(resp.text)
    except (ProviderError, ValueError) as e:
        log.warning("%s: не нашёлся (%s)", kind, str(e)[:120])
        return None

    if not isinstance(data, dict) or data.get("found") is False:
        log.info("%s: модель говорит, что его тут нет", kind)
        return None
    box = data.get("box") or {}
    if not box:
        return None

    # Ответ уже в долях кадра — переводим в доли области поиска, потому что
    # так его ждёт вызывающий.
    try:
        bx, by = float(box["x"]), float(box["y"])
        bw, bh = float(box["w"]), float(box["h"])
    except (KeyError, TypeError, ValueError):
        return None

    # Промах мимо подсказанной зоны: центр найденного должен в неё попадать
    # с запасом на половину размера зоны. Без этой проверки модель, увидев
    # весь экран, приносит похожий элемент из другого его конца.
    cx, cy = bx + bw / 2, by + bh / 2
    mx = (zone_hint["x1"] - zone_hint["x0"]) * 0.5
    my = (zone_hint["y1"] - zone_hint["y0"]) * 0.5
    if not (zone_hint["x0"] - mx <= cx <= zone_hint["x1"] + mx
            and zone_hint["y0"] - my <= cy <= zone_hint["y1"] + my):
        log.warning("%s: найденное лежит вне подсказанной зоны "
                    "(центр %.2f,%.2f) — отбрасываю", kind, cx, cy)
        return None

    entry: dict[str, Any] = {"kind": kind, "box": {
        "x": (bx * W - left) / cw, "y": (by * H - top) / chh,
        "w": bw * W / cw, "h": bh * H / chh}}

    at = data.get("at")
    if isinstance(at, (list, tuple)) and len(at) == 2:
        try:
            entry["at"] = [(float(at[0]) * W - left) / cw,
                           (float(at[1]) * H - top) / chh]
        except (TypeError, ValueError):
            pass
    if (target["counted"] if target else kind in COUNTED) and data.get("count"):
        try:
            entry["count"] = int(data["count"])
        except (TypeError, ValueError):
            pass

    log.info("%s найден", kind)
    return entry


async def analyze_cell(
    reg: Registry, preset: Preset, img: Image.Image, slots: dict[str, Any],
) -> list[dict[str, Any]]:
    """Второй проход: размечает одну карточку крупным планом.

    Возвращает боксы уже в координатах экрана. Если сетку не нашли или
    модель промолчала — пустой список, вызывающий разберётся.
    """
    cell = first_cell(slots)
    if not cell:
        log.warning("Сетка слотов не определена — крупный план пропускаю")
        return []

    W, H = img.size
    # Берём с запасом: звёзды сидят над карточкой, а плашка иногда свисает
    # ниже, и без полей они не попали бы в кадр.
    pad_x, pad_y = cell["w"] * 0.12, cell["h"] * 0.12
    left = max(0, int((cell["x"] - pad_x) * W))
    top = max(0, int((cell["y"] - pad_y) * H))
    right = min(W, int((cell["x"] + cell["w"] + pad_x) * W))
    bottom = min(H, int((cell["y"] + cell["h"] + pad_y) * H))
    if right - left < 32 or bottom - top < 32:
        return []

    # По одному запросу на элемент, каждый — в своей зоне карточки. Один
    # запрос «найди всё» заставляет модель делить внимание: она уверенно
    # берёт крупное (рамку, футер) и молча теряет мелкое — бейдж пропадал
    # целиком, плашка и прогресс приезжали не там. В отдельном запросе,
    # где спрашивают одну вещь и показывают только её окрестность, путать
    # не с чем.
    variant = resolve_variant(preset, "critic")
    cell_px = (left, top, right, bottom)
    targets = [t for t in segment_targets() if t["level"] == "cell"]
    if not targets:
        return []

    found = await asyncio.gather(*[
        _find_one(reg, variant, img, cell_px, t["kind"], target=t)
        for t in targets
    ], return_exceptions=True)

    items: list[dict[str, Any]] = []
    for kind, res in zip([t["kind"] for t in targets], found):
        if isinstance(res, Exception):
            log.warning("Поиск %s упал: %s", kind, res)
            continue
        if res:
            items.append(res)

    cw, ch = (right - left) / W, (bottom - top) / H
    ox, oy = left / W, top / H

    out: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("kind")
        box = item.get("box") or {}
        if not kind or not box:
            continue
        try:
            entry = {
                "kind": kind,
                # Из координат кропа — в координаты экрана.
                "box": {
                    "x": round(ox + float(box["x"]) * cw, 4),
                    "y": round(oy + float(box["y"]) * ch, 4),
                    "w": round(float(box["w"]) * cw, 4),
                    "h": round(float(box["h"]) * ch, 4),
                },
            }
            if kind == "stars" and item.get("count"):
                entry["stars_count"] = int(item["count"])
            # Точка на теле элемента. По ней автолассо обводит контур, и
            # она важнее бокса: указать «вот здесь звезда» модели проще,
            # чем провести её границу.
            at = item.get("at")
            if isinstance(at, (list, tuple)) and len(at) == 2:
                entry["at"] = [round(ox + float(at[0]) * cw, 4),
                               round(oy + float(at[1]) * ch, 4)]
            out.append(entry)
        except (KeyError, TypeError, ValueError):
            continue

    # Окно под арт вычисляем сами: модель путает его с рамкой, а нам оно
    # нужно точно — по нему пробивается прозрачная дыра в рамке.
    frame = next((p for p in out if p["kind"] == "frame"), None)
    plate = next((p for p in out if p["kind"] == "nameplate"), None)
    if frame and not any(p["kind"] == "art" for p in out):
        fb = frame["box"]
        inset = fb["w"] * 0.07
        art_top = fb["y"] + fb["h"] * 0.05
        art_bottom = (plate["box"]["y"] if plate
                      else fb["y"] + fb["h"] * 0.88) - fb["h"] * 0.02
        if art_bottom > art_top:
            out.append({"kind": "art", "box": {
                "x": round(fb["x"] + inset, 4), "y": round(art_top, 4),
                "w": round(fb["w"] - inset * 2, 4),
                "h": round(art_bottom - art_top, 4)}})

    log.info("Крупный план дал %d элементов карточки: %s",
             len(out), ", ".join(p["kind"] for p in out))
    return out


def cut_polygon(img: Image.Image, points: list, feather: float = 0.6) -> Image.Image:
    """Вырезает область по произвольному контуру — как лассо в редакторе.

    Прямоугольник неизбежно захватывает куски соседей и подложку панели.
    Контур повторяет форму элемента, поэтому вырезается ровно он: всё вне
    полигона становится прозрачным.

    points — список [x, y] в долях от размера изображения.
    """
    from PIL import ImageDraw, ImageFilter

    w, h = img.size
    pts = [(max(0.0, min(1.0, float(x))) * w, max(0.0, min(1.0, float(y))) * h)
           for x, y in points]
    if len(pts) < 3:
        return img

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    if feather > 0:
        # Лёгкое размытие снимает лесенку по краю контура: рисованный от
        # руки полигон иначе даёт рваную кромку.
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(Image.composite(out.getchannel("A"),
                                 Image.new("L", (w, h), 0), mask))
    return out


def cut(image: bytes, markup: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    """Режет скин по разметке и обрезает прозрачные края каждого куска.

    Если у элемента есть polygon — режем по контуру, иначе по боксу.
    """
    img = Image.open(BytesIO(image)).convert("RGBA")
    out: dict[str, dict[str, bytes]] = {}

    art_box = next(
        (i.get("box") for i in markup.get("parts", []) if i.get("kind") == "art"),
        None,
    )

    for item in markup.get("parts", []):
        kind = item.get("kind")
        if kind not in PART_KINDS or kind == "font":
            continue

        poly = item.get("polygon") or None
        if poly and len(poly) >= 3:
            # Контур сам задаёт границы: бокс считаем по нему, а не по
            # прямоугольнику разметки — иначе в кусок попадёт лишний воздух.
            xs = [max(0.0, min(1.0, float(p[0]))) for p in poly]
            ys = [max(0.0, min(1.0, float(p[1]))) for p in poly]
            left, top = int(min(xs) * img.width), int(min(ys) * img.height)
            right, bottom = int(max(xs) * img.width), int(max(ys) * img.height)
        else:
            left, top, right, bottom = _to_pixels(item.get("box") or {},
                                                  img.width, img.height)

        piece = img.crop((left, top, right, bottom))

        # Автолассо: если модель указала точку на теле элемента, обводим
        # контур от неё. Это тот же результат, что от лассо руками, только
        # границу ищет заливка, а не человек.
        at = item.get("at")
        if not poly and at and kind not in ("panel", "art"):
            from app.pipeline import extract as extract_lib

            traced = extract_lib.wand_cut(
                img, [(float(at[0]), float(at[1]))],
                (left, top, right, bottom))
            if traced is not None:
                piece = traced
                log.info("Автолассо обвело %s по точке %s", kind, at)

        if poly and len(poly) >= 3 and piece.width > 1 and piece.height > 1:
            # Точки переводим в доли вырезанного куска.
            local = [((float(px) * img.width - left) / piece.width,
                      (float(py) * img.height - top) / piece.height)
                     for px, py in poly]
            piece = cut_polygon(piece, local)

        crop = _trim(piece)
        if crop.width < MIN_PIECE or crop.height < MIN_PIECE:
            log.warning("Пропускаю %s: после обрезки осталось %dx%d",
                        kind, crop.width, crop.height)
            continue

        if kind == "frame":
            crop = _punch_frame_window(crop, art_box, img, left, top)

        # Плашка и лента приходят с чужой надписью. Свою мы рисуем сами,
        # поэтому исходную затираем — иначе два текста наложатся.
        if kind in ("nameplate", "ribbon"):
            crop = clean_text_area(crop, side_frac=0.18 if kind == "ribbon" else 0.14)

        buf = BytesIO()
        crop.save(buf, format="PNG")

        # Звёзды приходят одной полоской с известным количеством: кладём её
        # под своё число, остальные значения доберутся фолбэком композитора.
        if kind == "stars":
            n = int(item.get("stars_count") or 1)
            name = f"{max(1, min(5, n))}.png"
        else:
            name = (PART_KINDS[kind]["assets"] or ["asset.png"])[0]

        out.setdefault(kind, {})[name] = buf.getvalue()

    return out


def clean_text_area(img: Image.Image, side_frac: float = 0.12) -> Image.Image:
    """Затирает надпись, впечатанную в плашку.

    Плашки и ленты в скине идут с готовым текстом («Star Map», «FARMING»).
    Если оставить как есть, наша подпись ляжет поверх чужой и получится
    каша. Текст всегда по центру, а края — чистый фон, поэтому центр
    заменяется растянутой полоской края: это классический 9-slice, только
    по горизонтали.
    """
    w, h = img.size
    side = max(2, int(w * side_frac))
    if w < side * 3:
        return img

    out = img.copy()
    # Полоска сразу за левым кантом — там уже фон, но ещё не буквы.
    strip = out.crop((side, 0, side + max(2, side // 3), h))
    filler = strip.resize((w - side * 2, h), Image.LANCZOS)
    out.paste(filler, (side, 0))
    return out


def _punch_frame_window(crop: Image.Image, art_box, src: Image.Image,
                        off_x: int, off_y: int) -> Image.Image:
    """Делает прозрачным окно под картинку внутри рамки слота.

    Без окна рамка ляжет поверх арта сплошным пятном — ровно так выглядит
    «рамка есть, картинки нет».
    """
    from PIL import ImageDraw

    w, h = crop.size
    if art_box:
        al, at, ar, ab = _to_pixels(art_box, src.width, src.height)
        window = (al - off_x, at - off_y, ar - off_x, ab - off_y)
    else:
        # Разметки арта нет — берём типовой внутренний отступ слота.
        window = (int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.80))

    left, top, right, bottom = window
    if right - left < MIN_PIECE or bottom - top < MIN_PIECE:
        return crop

    mask = Image.new("L", crop.size, 255)
    radius = max(4, min(right - left, bottom - top) // 12)
    ImageDraw.Draw(mask).rounded_rectangle(
        [left, top, right - 1, bottom - 1], radius=radius, fill=0)
    out = crop.copy()
    out.putalpha(Image.composite(
        out.getchannel("A"), Image.new("L", crop.size, 0), mask))
    return out


async def segment_skin(
    reg: Registry,
    preset: Preset,
    image: bytes,
    version: str,
    *,
    note: str = "",
    make_default: bool = True,
    save_references: bool = True,
    markup: dict[str, Any] | None = None,
    extract_mode: str = "local",
    verify: bool = True,
    on_event=None,
) -> dict[str, Any]:
    """Полный цикл: разметить экран, нарезать, сохранить версию каждого типа.

    markup — готовая разметка. Если она передана, vision не вызывается:
    размеченное курсором точнее любой автоматики, и переспрашивать модель
    об уже известных координатах незачем.

    extract_mode — что делать с нарезанным: "raw" оставить обрезком скрина,
    "local" снять фон локально, "model" отдать image-edit модели.

    verify — после извлечения сложить экран обратно из деталей и сверить с
    оригиналом. Не сошедшиеся элементы вырезаются заново.
    """
    from app import skins

    skins.save_screen(version, image)

    if markup:
        if on_event:
            on_event("stage", {"name": "analyze", "status": "done",
                               "found": [p.get("kind") for p in markup.get("parts", [])],
                               "summary": "разметка задана вручную"})
    else:
        if on_event:
            on_event("stage", {"name": "analyze", "status": "start"})
        markup = await analyze(reg, preset, image)

        # Отсекаем заведомо неверные боксы до вырезания: иначе в библиотеку
        # попадёт кусок фона, и разбираться придётся уже по картинкам.
        bad = validate_markup(markup)
        if bad:
            log.warning("Разметка сомнительна: %s", "; ".join(bad))
            if on_event:
                on_event("warn", {"text": "разметка сомнительна: "
                                          + "; ".join(bad[:4])})
            drop = {p.split(":")[0] for p in bad if "подозрительно" in p
                    or "за кадр" in p or "нулевой" in p}
            if drop:
                markup["parts"] = [p for p in markup.get("parts", [])
                                   if p.get("kind") not in drop]
                log.warning("Убрал из разметки: %s", ", ".join(sorted(drop)))

        skins.save_markup(version, markup, source="vision")
    found = [p.get("kind") for p in markup.get("parts", [])]
    if on_event:
        on_event("stage", {"name": "analyze", "status": "done", "found": found,
                           "summary": markup.get("summary", "")})

    pieces = cut(image, markup)
    boxes = {i["kind"]: i["box"] for i in markup.get("parts", [])
             if i.get("kind") and i.get("box")}

    # Референсы стиля берём ДО извлечения: для них нужен кусок скрина с
    # окружением, а не изолированный ассет.
    raw_pieces = pieces

    checks: list[dict] = []

    if extract_mode != "raw":
        from app.pipeline import extract as extract_lib

        if on_event:
            on_event("stage", {"name": "extract", "status": "start",
                               "mode": extract_mode})
        pieces = await extract_lib.extract_pieces(
            reg, preset, pieces, mode=extract_mode, screen=image,
            boxes=boxes, on_event=on_event,
            # Контуром вырезано и то, что обвели руками (polygon), и то,
            # что обвело автолассо по точке (at). И там и там граница уже
            # найдена, и матирование поверх пошло бы от края силуэта,
            # съедая сам элемент.
            lassoed={i["kind"] for i in markup.get("parts", [])
                     if i.get("kind") and (i.get("polygon") or i.get("at"))})
        if on_event:
            on_event("stage", {"name": "extract", "status": "done",
                               "mode": extract_mode})

    if verify and extract_mode != "raw":
        # Проверка сборкой: детали раскладываются обратно по своим
        # координатам и сравниваются с оригиналом. Что не сошлось — то и
        # вырезано плохо, и оно уходит на повторную попытку.
        from app.pipeline import verify as verify_lib

        if on_event:
            on_event("stage", {"name": "verify", "status": "start"})
        pieces, checks = await verify_lib.verify_and_repair(
            reg, preset, screen=image, markup=markup, pieces=pieces,
            raw_pieces=raw_pieces, mode=extract_mode, on_event=on_event)
        if on_event:
            on_event("stage", {"name": "verify", "status": "done",
                               "passes": len(checks),
                               "ok": bool(checks and checks[-1]["ok"])})

    saved: list[dict] = []

    for kind, assets in pieces.items():
        meta = parts.save_version(
            kind, version, assets=assets, title=version,
            note=note or markup.get("summary", "")[:200],
            source="сегментация скина", anchor=boxes.get(kind),
            make_default=make_default,
        )
        saved.append({"kind": kind, "version": meta.version, "assets": meta.assets})
        if on_event:
            on_event("part", {"kind": kind, "version": meta.version,
                              "assets": meta.assets})

    # Шрифт — описанием, не картинкой: текст рисует композитор.
    font_spec = markup.get("font") or {}
    if font_spec:
        img = Image.open(BytesIO(image))
        size = int(round(float(font_spec.get("size_rel", 0.03)) * img.height))
        font = FontStyle(
            size=max(size, 10),
            color=font_spec.get("color") or "#FFF3D0",
            stroke_color=font_spec.get("stroke_color") or "#5A2E00",
            stroke_width=int(font_spec.get("stroke_width", 4)),
            uppercase=bool(font_spec.get("uppercase", True)),
        )
        meta = parts.save_version(
            "font", version, title=version, note=markup.get("summary", "")[:200],
            source="сегментация скина", font=font, make_default=make_default,
        )
        saved.append({"kind": "font", "version": meta.version, "assets": []})

    refs: list[str] = []
    if save_references:
        refs = save_skin_references(version, image, raw_pieces, markup)
        if on_event:
            on_event("refs", {"count": len(refs), "files": refs})

    if on_event:
        on_event("done", {"version": version, "saved": len(saved),
                          "kinds": [s["kind"] for s in saved],
                          "references": len(refs)})

    return {
        "version": version,
        "summary": markup.get("summary", ""),
        "palette": markup.get("palette", []),
        "slots": markup.get("slots") or {},
        "saved": saved,
        "references": refs,
        "markup": markup,
        "checks": checks,
    }


def references_dir(version: str) -> Path:
    from app.config import STYLE_DIR

    return STYLE_DIR / "refs" / version


def save_skin_references(
    version: str, image: bytes, pieces: dict[str, dict[str, bytes]],
    markup: dict[str, Any],
) -> list[str]:
    """Складывает референсы стиля, которые уйдут в генерацию арта.

    Это и есть «максимальная важность»: модель видит не абстрактное описание
    стиля словами, а вырезанные куски того самого скина, под который рисуется
    арт. Палитра и материалы попадают в цель без уговоров промптом.
    """
    out_dir = references_dir(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    written: list[str] = []
    # Порядок важен: первым идёт то, что сильнее задаёт стиль арта.
    priority = ("frame", "nameplate", "badge", "stars", "button",
                "ribbon", "progress", "panel")
    for kind in priority:
        for name, data in (pieces.get(kind) or {}).items():
            path = out_dir / f"{len(written):02d}_{kind}_{name}"
            path.write_bytes(data)
            written.append(path.name)
            break   # по одному файлу на тип — референсов нужно немного

    # Плюс общий вид экрана: он задаёт освещение и общую гамму.
    full = Image.open(BytesIO(image)).convert("RGBA")
    full = _trim(full)
    full.thumbnail((1024, 1024), Image.LANCZOS)
    buf = BytesIO()
    full.save(buf, format="PNG")
    path = out_dir / f"{len(written):02d}_screen.png"
    path.write_bytes(buf.getvalue())
    written.append(path.name)

    return written
