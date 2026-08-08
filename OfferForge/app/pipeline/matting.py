"""Вырезание элемента из сгенерированной картинки.

Image-модели отдают непрозрачный PNG: элемент лежит на каком-то фоне.
Для сборки нужен именно элемент с альфа-каналом, иначе он ляжет на панель
прямоугольным пятном.

Поэтому генерация просит однотонный фон известного цвета, а здесь он
срезается заливкой от краёв. Заливка, а не «все пиксели похожего цвета»:
иначе выест такой же цвет внутри самого элемента.
"""
from __future__ import annotations

import logging
from collections import deque

from PIL import Image, ImageFilter

log = logging.getLogger("offerforge.matting")

# Фон, который просим у модели. Ярко-зелёный почти не встречается в игровом
# UI, поэтому не рискует совпасть с самим элементом.
CHROMA = (0, 255, 0)
CHROMA_NAME = "pure chroma green (#00FF00)"


def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def detect_key(img: Image.Image) -> tuple[int, int, int]:
    """Цвет фона по углам — модель редко попадает в запрошенный оттенок точно."""
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()
    corners = [px[0, 0][:3], px[w - 1, 0][:3], px[0, h - 1][:3], px[w - 1, h - 1][:3]]
    return next((c for c in corners if sum(_dist(c, o) < 40 for o in corners) >= 2),
                corners[0])


def remove_background(
    img: Image.Image,
    *,
    key: tuple[int, int, int] | None = None,
    tolerance: int = 120,
    feather: float = 1.0,
    kill_spill: bool = True,
) -> Image.Image:
    """Делает фон прозрачным заливкой от краёв изображения.

    key=None — цвет фона определяется по углам: модель редко попадает в
    запрошенный оттенок точно, зато углы почти всегда чистый фон.
    """
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()

    if key is None:
        key = detect_key(img)

    mask = Image.new("L", (w, h), 255)      # 255 = элемент, 0 = фон
    mp = mask.load()
    seen = bytearray(w * h)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        idx = y * w + x
        if seen[idx]:
            continue
        seen[idx] = 1
        if _dist(px[x, y][:3], key) > tolerance:
            continue
        mp[x, y] = 0
        if x > 0:
            queue.append((x - 1, y))
        if x < w - 1:
            queue.append((x + 1, y))
        if y > 0:
            queue.append((x, y - 1))
        if y < h - 1:
            queue.append((x, y + 1))

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    alpha = out.getchannel("A").point(lambda v: v)
    out.putalpha(Image.composite(alpha, Image.new("L", (w, h), 0), mask))

    # Хромакей подмешивается в полупрозрачные края зелёной каймой — гасим её.
    # Но только если дальше не ищутся замкнутые области фона: despill меняет
    # их цвет, и они перестают опознаваться как фон.
    return despill(out, key) if kill_spill else out


def despill(img: Image.Image, key: tuple[int, int, int]) -> Image.Image:
    """Убирает подмес цвета фона в краевых пикселях."""
    if _dist(key, CHROMA) > 200:
        return img          # фон был не хромакейный, кайма не грозит

    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a and g > r + 18 and g > b + 18:
                g = (r + b) // 2 + 18
                px[x, y] = (r, g, b, a)
    return img


def remove_enclosed(
    img: Image.Image,
    *,
    key: tuple[int, int, int] | None = None,
    tolerance: int = 120,
    min_area_frac: float = 0.02,
) -> Image.Image:
    """Убирает замкнутые области фонового цвета внутри элемента.

    Заливка от краёв не проникает в окно рамки слота: оно со всех сторон
    закрыто самой рамкой. В результате внутри остаётся кусок хромакея, и
    рамка перекрывает арт зелёным пятном.

    Поэтому для элементов с окном отдельно ищутся замкнутые области цвета
    фона. Порог по площади защищает мелкие детали самого элемента —
    блик или тень цвета, близкого к ключу, не выест.
    """
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()
    if key is None:
        key = px[0, 0][:3]

    visited = bytearray(w * h)
    min_area = int(w * h * min_area_frac)

    for sy in range(h):
        for sx in range(w):
            idx = sy * w + sx
            if visited[idx] or px[sx, sy][3] == 0:
                continue
            if _dist(px[sx, sy][:3], key) > tolerance:
                visited[idx] = 1
                continue

            region: list[tuple[int, int]] = []
            queue = deque([(sx, sy)])
            visited[idx] = 1
            while queue:
                x, y = queue.popleft()
                region.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    ni = ny * w + nx
                    if visited[ni] or px[nx, ny][3] == 0:
                        continue
                    if _dist(px[nx, ny][:3], key) > tolerance:
                        continue
                    visited[ni] = 1
                    queue.append((nx, ny))

            if len(region) >= min_area:
                for x, y in region:
                    r, g, b, _ = px[x, y]
                    px[x, y] = (r, g, b, 0)

    return img


def trim(img: Image.Image, pad: int = 2) -> Image.Image:
    """Срезает прозрачные поля, оставляя небольшой отступ."""
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        return img
    return img.crop((
        max(0, bbox[0] - pad), max(0, bbox[1] - pad),
        min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad),
    ))


def coverage(img: Image.Image) -> float:
    """Доля непрозрачных пикселей — признак того, что вырезание удалось.

    Около 1.0 означает, что фон не нашёлся и осталось сплошное полотно;
    около 0.0 — что выело весь элемент. И то и другое — брак.
    """
    a = img.getchannel("A")
    total = img.width * img.height
    if not total:
        return 0.0
    hist = a.histogram()
    opaque = sum(hist[128:])
    return opaque / total
