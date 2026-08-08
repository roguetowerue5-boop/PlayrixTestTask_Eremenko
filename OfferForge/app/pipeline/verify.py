"""Проверка нарезки сборкой обратно.

Идея простая: если детали вырезаны правильно, из них можно сложить исходный
экран. Раскладываем каждый элемент по его координатам из разметки и
сравниваем с оригиналом. Что не сошлось — то и вырезано плохо.

Это дешевле и честнее, чем спрашивать модель «хорошо ли получилось».
Сборка либо совпадает с оригиналом, либо нет, и расхождение показывает
пальцем на конкретный элемент.

Два типа брака ловятся по-разному.

**Дыра.** В оригинале на этом месте что-то есть, в сборке пусто — элемент
не вырезался или выело матированием.

**Поглощение.** Один элемент утащил в себя соседей: рамка карточки
приходит вместе с плашкой имени, звездой и бейджем, потому что в скине они
лежат внутри её habitats. Ловится пересечением непрозрачных пикселей: если
два элемента закрашивают одну зону, один содержит другого. Само по себе
касание нормально, поэтому смотрим долю чужой области, которую элемент
покрыл собой.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Callable

from PIL import Image, ImageChops, ImageFilter

log = logging.getLogger("offerforge.verify")

# Доля чужой площади, выше которой считаем, что элемент поглотил соседа.
# Ниже — обычное касание: рамка и плашка соприкасаются краями всегда.
SWALLOW_FRAC = 0.35

# Средняя разница по каналу, выше которой область считается несошедшейся.
# 0–255; на глаз заметно примерно с 25.
DIFF_TOL = 34

# Доля площади элемента, которую он обязан закрасить в своей зоне.
# Меньше — значит от него осталась одна кромка.
MIN_FILL = 0.08

# Подложки: они по замыслу лежат под другими элементами и содержат их в
# своих границах. Панель — фон всего экрана, на ленте сидит крестик.
# Считать это поглощением нельзя, иначе брак найдётся всегда.
BACKDROP = {"panel", "ribbon", "footer", "progress"}

# Кольца: внутри обязана быть дыра. Рамка карточки приходит от модели
# вместе с плашкой имени, звездой и бейджем, потому что визуально это один
# объект — вот это и есть настоящее поглощение.
HOLLOW = {"frame"}

# Элементы, от которых намеренно оставляют один экземпляр: в скине их
# несколько подряд, а в библиотеку нужен один — полоску под нужное число
# соберёт композитор. Судить их по расхождению нельзя, они обязаны
# отличаться от оригинала.
ATOMIZED = {"stars", "button"}


def _box_px(box: dict, w: int, h: int) -> tuple[int, int, int, int]:
    left = max(0, min(w - 1, int(float(box.get("x", 0)) * w)))
    top = max(0, min(h - 1, int(float(box.get("y", 0)) * h)))
    right = max(left + 1, min(w, int((float(box.get("x", 0))
                                      + float(box.get("w", 0))) * w)))
    bottom = max(top + 1, min(h, int((float(box.get("y", 0))
                                      + float(box.get("h", 0))) * h)))
    return left, top, right, bottom


CARD_PARTS = ("frame", "stars", "badge", "nameplate", "art")


def card_offsets(markup: dict[str, Any],
                 box: dict | None = None) -> list[tuple[float, float]]:
    """Смещения ячеек сетки относительно той, где элемент размечен.

    Карточные элементы размечаются на одном слоте, а на экране их десять.
    Без размножения сборка отличалась бы от оригинала девятью пустыми
    слотами, и вина за это ложилась бы на панель, которая тут ни при чём.

    Отсчёт именно от своей ячейки, а не от первой: разметку делают на том
    слоте, где элемент виднее, и бейдж вполне может быть отмечен на втором.
    Считая от первой, мы уводили его на ячейку вправо за пределы панели.
    """
    slots = markup.get("slots") or {}
    grid = slots.get("box") or {}
    cols, rows = int(slots.get("cols", 1) or 1), int(slots.get("rows", 1) or 1)
    if cols < 1 or rows < 1 or not grid:
        return [(0.0, 0.0)]

    cw = float(grid.get("w", 0)) / cols
    ch = float(grid.get("h", 0)) / rows
    if cw <= 0 or ch <= 0:
        return [(0.0, 0.0)]

    col = row = 0
    if box:
        cx = float(box.get("x", 0)) + float(box.get("w", 0)) / 2
        cy = float(box.get("y", 0)) + float(box.get("h", 0)) / 2
        col = max(0, min(cols - 1, int((cx - float(grid.get("x", 0))) / cw)))
        row = max(0, min(rows - 1, int((cy - float(grid.get("y", 0))) / ch)))

    return [((c - col) * cw, (r - row) * ch)
            for r in range(rows) for c in range(cols)]


def rebuild(
    size: tuple[int, int],
    markup: dict[str, Any],
    pieces: dict[str, dict[str, bytes]],
) -> tuple[Image.Image, dict[str, Image.Image]]:
    """Складывает экран обратно из вырезанных деталей.

    Возвращает саму сборку и по маске на каждый тип — маски нужны, чтобы
    потом искать пересечения.
    """
    w, h = size
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    masks: dict[str, Image.Image] = {}

    for item in markup.get("parts", []):
        kind = item.get("kind")
        assets = pieces.get(kind or "")
        if not kind or not assets:
            continue
        blob = next(iter(assets.values()))
        piece = Image.open(BytesIO(blob)).convert("RGBA")

        left, top, right, bottom = _box_px(item.get("box") or {}, w, h)
        bw, bh = right - left, bottom - top
        if bw < 2 or bh < 2:
            continue

        # Вписываем деталь в её бокс. Матирование обрезало прозрачные поля,
        # поэтому размер уже не тот, что был при вырезании, — масштабируем
        # по большей стороне, чтобы не растянуть.
        scale = min(bw / piece.width, bh / piece.height)
        fitted = piece.resize((max(1, round(piece.width * scale)),
                               max(1, round(piece.height * scale))),
                              Image.LANCZOS)
        ox = left + (bw - fitted.width) // 2
        oy = top + (bh - fitted.height) // 2

        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Карточные элементы повторяются по всей сетке, остальные — один раз.
        cells = (card_offsets(markup, item.get("box"))
                 if kind in CARD_PARTS else [(0.0, 0.0)])
        for dx, dy in cells:
            layer.paste(fitted, (ox + round(dx * w), oy + round(dy * h)))
        canvas = Image.alpha_composite(canvas, layer)

        mask = layer.getchannel("A").point(lambda v: 255 if v > 32 else 0)
        if kind in masks:
            masks[kind] = ImageChops.lighter(masks[kind], mask)
        else:
            masks[kind] = mask

    return canvas, masks


def _area(mask: Image.Image) -> int:
    return sum(mask.histogram()[128:])


def find_swallowed(
    masks: dict[str, Image.Image],
    markup: dict[str, Any],
    size: tuple[int, int],
) -> list[dict]:
    """Ищет элементы, утащившие в себя соседей.

    Сравниваем не маску с маской, а маску элемента с *боксом* соседа: если
    рамка закрасила больше трети площади, отведённой под плашку имени,
    значит плашка внутри рамки. Именно так выглядит брак, который видно
    глазом на превью.
    """
    w, h = size
    boxes = {i["kind"]: i["box"] for i in markup.get("parts", [])
             if i.get("kind") and i.get("box")}
    problems: list[dict] = []

    for kind, mask in masks.items():
        if kind in BACKDROP:
            continue        # подложке содержать других — её работа
        own = boxes.get(kind)
        for other, obox in boxes.items():
            if other == kind or other == "art" or other not in masks:
                continue
            # Интересуют только соседи, которые лежат внутри нашего бокса:
            # иначе поймаем обычное соседство по краю.
            if own and not _inside(obox, own):
                continue
            left, top, right, bottom = _box_px(obox, w, h)
            region = mask.crop((left, top, right, bottom))
            covered = _area(region) / max(1, (right - left) * (bottom - top))
            if covered >= SWALLOW_FRAC:
                problems.append({
                    "kind": kind, "swallowed": other,
                    "covered": round(covered, 3),
                    "reason": f"«{kind}» закрывает {covered:.0%} области «{other}» — "
                              f"похоже, вырезан вместе с ним",
                })
    return problems


def _inside(inner: dict, outer: dict, slack: float = 0.02) -> bool:
    """Лежит ли один бокс внутри другого (с небольшим допуском)."""
    return (float(inner["x"]) >= float(outer["x"]) - slack
            and float(inner["y"]) >= float(outer["y"]) - slack
            and float(inner["x"]) + float(inner["w"])
            <= float(outer["x"]) + float(outer["w"]) + slack
            and float(inner["y"]) + float(inner["h"])
            <= float(outer["y"]) + float(outer["h"]) + slack)


def compare(
    original: Image.Image,
    rebuilt: Image.Image,
    markup: dict[str, Any],
    masks: dict[str, Image.Image],
) -> dict[str, Any]:
    """Сверяет сборку с оригиналом и по каждому элементу выносит вердикт."""
    w, h = original.size
    orig = original.convert("RGB")
    # Сборку кладём на тот же фон, что у оригинала, иначе прозрачные зоны
    # дадут ложную разницу просто потому, что они прозрачные.
    flat = Image.new("RGB", (w, h), (0, 0, 0))
    flat.paste(orig, (0, 0))
    built = Image.new("RGB", (w, h), (0, 0, 0))
    built.paste(rebuilt.convert("RGB"), (0, 0), rebuilt.getchannel("A"))

    diff = ImageChops.difference(orig, built).convert("L")
    diff = diff.filter(ImageFilter.GaussianBlur(1.5))

    boxes = {i["kind"]: i["box"] for i in markup.get("parts", [])
             if i.get("kind") and i.get("box")}

    per_kind: list[dict] = []
    for item in markup.get("parts", []):
        kind = item.get("kind")
        if not kind or kind == "art":
            continue
        left, top, right, bottom = _box_px(item.get("box") or {}, w, h)
        area = max(1, (right - left) * (bottom - top))

        region = diff.crop((left, top, right, bottom))

        # Из зоны элемента вычитаем зоны тех, кто лежит внутри него. Иначе
        # панель всегда «не сходится»: её бокс — весь экран, а на экране
        # десять карточек против одного вырезанного слота, и эта разница
        # приписывается панели, хотя панель тут ни при чём.
        inner = [b for k, b in boxes.items()
                 if k != kind and k != "art" and _inside(b, item["box"])]
        if inner:
            keep = Image.new("L", region.size, 255)
            for b in inner:
                il, it, ir, ib = _box_px(b, w, h)
                keep.paste(0, (max(0, il - left), max(0, it - top),
                               max(0, ir - left), max(0, ib - top)))
            region = Image.composite(region, Image.new("L", region.size, 0), keep)
            counted = _area(keep.point(lambda v: 255 if v > 128 else 0))
        else:
            counted = region.width * region.height

        hist = region.histogram()
        mean = sum(i * n for i, n in enumerate(hist)) / max(1, counted)

        fill = _area(masks[kind].crop((left, top, right, bottom))) / area \
            if kind in masks else 0.0

        troubles = []
        if kind not in masks:
            troubles.append("элемент не вырезался")
        elif fill < MIN_FILL:
            troubles.append(f"в своей зоне закрашено {fill:.0%} — почти пусто")
        # У подложки расхождение складывается из всего, что на ней лежит:
        # артов внутри карточек у нас нет и не будет, они рисуются отдельно.
        # Судить панель за это нельзя — у каждого элемента своя строка.
        # Атомизированные тоже: одна звезда против пяти в оригинале даёт
        # огромное расхождение, хотя всё сделано правильно.
        if mean > DIFF_TOL and kind not in BACKDROP and kind not in ATOMIZED:
            troubles.append(f"расхождение с оригиналом {mean:.0f}/255")

        per_kind.append({
            "kind": kind, "diff": round(mean, 1), "fill": round(fill, 3),
            "ok": not troubles, "troubles": troubles,
        })

    swallowed = find_swallowed(masks, markup, (w, h))
    for s in swallowed:
        for row in per_kind:
            if row["kind"] == s["kind"]:
                row["ok"] = False
                row["troubles"].append(s["reason"])

    overall = sum(r["diff"] for r in per_kind) / max(1, len(per_kind))
    return {
        "parts": per_kind,
        "swallowed": swallowed,
        "mean_diff": round(overall, 1),
        "failed": [r["kind"] for r in per_kind if not r["ok"]],
        "ok": all(r["ok"] for r in per_kind),
    }


def diff_map(original: Image.Image, rebuilt: Image.Image,
             max_side: int = 900) -> Image.Image:
    """Карта расхождений для показа: где темнее — там сошлось."""
    orig = original.convert("RGB")
    built = Image.new("RGB", orig.size, (0, 0, 0))
    built.paste(rebuilt.convert("RGB"), (0, 0), rebuilt.getchannel("A"))
    d = ImageChops.difference(orig, built).convert("L")
    # Растягиваем контраст: расхождения обычно тихие, на глаз незаметны.
    d = d.point(lambda v: min(255, int(v * 3)))
    out = Image.merge("RGB", (d, d.point(lambda v: v // 3), d.point(lambda v: 0)))
    out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out


def subtract_neighbours(
    kind: str,
    piece: bytes,
    markup: dict[str, Any],
    pieces: dict[str, dict[str, bytes]],
    size: tuple[int, int],
    *,
    grow: int = 3,
) -> bytes | None:
    """Стирает из элемента формы соседей, лежащих внутри него.

    Уговаривать модель убрать плашку из рамки бесполезно: она возвращает
    то же самое, потому что визуально это один объект. Но нам и не нужно
    её уговаривать — координаты соседей заданы разметкой, а их формы уже
    вырезаны. Достаточно вычесть.

    Вычитается именно форма соседа, а не его прямоугольник. Плашка имени
    сидит на нижней перекладине рамки: убрав прямоугольник, мы вырезали бы
    из рамки кусок вместе с кантом. Форма же убирает ровно плашку и
    оставляет кант вокруг неё.

    grow — на сколько пикселей расширить вычитаемую форму. Кромка соседа
    полупрозрачна после матирования, и без запаса от неё остаётся ореол.
    """
    w, h = size
    boxes = {i["kind"]: i["box"] for i in markup.get("parts", [])
             if i.get("kind") and i.get("box")}
    own = boxes.get(kind)
    if not own:
        return None

    inner = [k for k, b in boxes.items()
             if k != kind and k != "art" and k in pieces and _inside(b, own)]
    if not inner:
        return None

    left, top, right, bottom = _box_px(own, w, h)
    bw, bh = right - left, bottom - top
    img = Image.open(BytesIO(piece)).convert("RGBA")

    # Работаем в координатах бокса: вырезанный кусок мог быть обрезан
    # матированием, поэтому масштабируем его обратно на бокс, вычитаем и
    # возвращаем к прежнему размеру. Так координаты соседей совпадают.
    scale = min(bw / img.width, bh / img.height)
    fitted = img.resize((max(1, round(img.width * scale)),
                         max(1, round(img.height * scale))), Image.LANCZOS)
    ox, oy = (bw - fitted.width) // 2, (bh - fitted.height) // 2

    from app.pipeline import matting

    stencil = Image.new("L", (bw, bh), 0)
    for other in inner:
        ob = boxes[other]
        ol, ot, orr, obm = _box_px(ob, w, h)
        ow, oh = max(1, orr - ol), max(1, obm - ot)
        blob = next(iter(pieces[other].values()))
        shape = Image.open(BytesIO(blob)).convert("RGBA")

        s = min(ow / shape.width, oh / shape.height)
        shape = shape.resize((max(1, round(shape.width * s)),
                              max(1, round(shape.height * s))), Image.LANCZOS)

        if matting.coverage(shape) >= 0.3:
            # У соседа есть внятная форма — вычитаем ровно её. Так у рамки
            # уцелеет кант вокруг плашки.
            mask = shape.getchannel("A").point(lambda v: 255 if v > 24 else 0)
            px = ol - left + (ow - shape.width) // 2
            py = ot - top + (oh - shape.height) // 2
        else:
            # От соседа осталась одна обводка — его форме верить нельзя.
            # Вычтем область бокса, отступив от краёв: там проходит кант
            # рамки, и без отступа мы срезали бы именно его, а плашку
            # оставили бы на месте.
            inset = max(2, int(min(ow, oh) * 0.10))
            mask = Image.new("L", (max(1, ow - inset * 2),
                                   max(1, oh - inset * 2)), 255)
            px, py = ol - left + inset, ot - top + inset

        area = stencil.crop((px, py, px + mask.width, py + mask.height))
        stencil.paste(ImageChops.lighter(area, mask), (px, py))

    if grow > 0:
        stencil = stencil.filter(ImageFilter.MaxFilter(grow * 2 + 1))

    keep = ImageChops.invert(stencil).crop(
        (-ox, -oy, -ox + fitted.width, -oy + fitted.height))
    cleaned = fitted.copy()
    cleaned.putalpha(ImageChops.multiply(cleaned.getchannel("A"), keep))
    cleaned = matting.trim(cleaned)
    if matting.coverage(cleaned) < 0.01:
        return None         # вычли всё подчистую — значит промахнулись

    out = BytesIO()
    cleaned.save(out, format="PNG")
    return out.getvalue()


async def verify_and_repair(
    reg,
    preset,
    *,
    screen: bytes,
    markup: dict[str, Any],
    pieces: dict[str, dict[str, bytes]],
    raw_pieces: dict[str, dict[str, bytes]] | None = None,
    mode: str = "model",
    attempts: int = 2,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[dict[str, dict[str, bytes]], list[dict]]:
    """Собирает экран из деталей, а не сошедшиеся вырезает заново.

    Повторная попытка идёт только по проблемным типам: перегонять всё
    заново дорого и незачем, а поглотивший соседей элемент получает
    уточнённое требование — «только контур, внутри пусто».

    Повтор всегда стартует от ИСХОДНОГО обрезка, а не от результата
    прошлой попытки. Иначе обработка накладывается сама на себя: матирование
    второй раз съедает края, которые первый раз оставило, и с каждой
    попыткой становится хуже, а не лучше.

    Для локального матирования повторов нет вовсе: оно детерминировано и
    подсказок не понимает, второй прогон вернул бы тот же результат.
    """
    from app.pipeline import extract as extract_lib

    original = Image.open(BytesIO(screen)).convert("RGBA")
    source = raw_pieces or pieces
    history: list[dict] = []
    current = pieces
    limit = attempts if mode == "model" else 1

    for attempt in range(1, limit + 1):
        rebuilt, masks = rebuild(original.size, markup, current)
        report = compare(original, rebuilt, markup, masks)
        report["attempt"] = attempt
        history.append(report)

        if on_event:
            on_event("verify", {"attempt": attempt, "ok": report["ok"],
                                "mean_diff": report["mean_diff"],
                                "failed": report["failed"]})
        if report["ok"]:
            break

        # Чиним локально, не обращаясь к модели: оба случая решаются тем,
        # что мы и так знаем, и потому срабатывают всегда.
        patched = dict(current)
        cleaned: list[str] = []

        # 1. Поглощение — вычитаем соседей. Координаты и формы вложенных
        #    элементов заданы разметкой, гадать не о чем.
        for s in report["swallowed"]:
            kind = s["kind"]
            assets = patched.get(kind)
            if not assets:
                continue
            name, blob = next(iter(assets.items()))
            fixed_blob = subtract_neighbours(
                kind, blob, markup, patched, original.size)
            if fixed_blob:
                patched[kind] = {name: fixed_blob}
                cleaned.append(f"{kind} (вычтены соседи)")

        # 2. Пустота — откатываем к сырому вырезу. Матирование иногда
        #    выедает элемент почти целиком: ленту заголовка оно оставляет
        #    от 91% плотности всего 1%. Обрезок скрина с фоном по краям
        #    хуже чистого ассета, но несравнимо лучше пустоты.
        for row in report["parts"]:
            kind = row["kind"]
            if row["ok"] or kind not in source or kind in ATOMIZED:
                continue
            if row["fill"] >= MIN_FILL:
                continue
            name = next(iter(source[kind]))
            patched[kind] = dict(source[kind])
            cleaned.append(f"{kind} (возврат к исходному вырезу)")

        if cleaned:
            if on_event:
                on_event("repair", {"attempt": attempt, "kinds": cleaned})
            current = patched
            # Пересчитываем: обычно на этом всё и заканчивается.
            rebuilt, masks = rebuild(original.size, markup, current)
            report = compare(original, rebuilt, markup, masks)
            report["attempt"] = attempt
            report["repaired"] = cleaned
            history[-1] = report
            if on_event:
                on_event("verify", {"attempt": attempt, "ok": report["ok"],
                                    "mean_diff": report["mean_diff"],
                                    "failed": report["failed"]})

        if report["ok"] or attempt == limit:
            break

        # Что осталось — отдаём модели, ей же говорим прямо, чего мы ждём.
        hints = {s["kind"]: "The element is a hollow outline only. Everything "
                            "that sits inside it — name plate, stars, badge, "
                            "artwork — belongs to other elements and must be "
                            "erased, leaving the middle empty."
                 for s in report["swallowed"]}

        retry = {k: v for k, v in source.items() if k in report["failed"]}
        if not retry:
            break
        if on_event:
            on_event("repair", {"attempt": attempt, "kinds": list(retry)})

        fixed = await extract_lib.extract_pieces(
            reg, preset, retry, mode=mode, screen=screen,
            boxes={i["kind"]: i["box"] for i in markup.get("parts", [])
                   if i.get("kind") and i.get("box")},
            hints=hints, on_event=on_event)

        # Берём новый вариант, только если он и правда лучше. Модель может
        # вернуть что-то хуже прежнего, и тогда откат к прошлой попытке —
        # единственный способ не ухудшить результат проверкой.
        candidate = {**current, **fixed}
        cand_built, cand_masks = rebuild(original.size, markup, candidate)
        cand_report = compare(original, cand_built, markup, cand_masks)
        if len(cand_report["failed"]) <= len(report["failed"]):
            current = candidate
        elif on_event:
            on_event("warn", {"text": "повторная попытка вышла хуже — "
                                      "оставляю прежний вариант"})

    return current, history
