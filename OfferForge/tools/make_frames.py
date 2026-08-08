"""Генератор дефолтных ассетов рамки.

Рамка, плашка и звёзды — это UI-ассеты, а не арт: они одинаковы на всех
карточках и потому рисуются кодом, а не моделью. Скрипт даёт рабочий
комплект «из коробки»; когда появятся настоящие ассеты проекта, файлы
в style/frames/ просто заменяются, шаблоны трогать не нужно.

    python tools/make_frames.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "style" / "frames"

CARD_W, CARD_H = 512, 720
RIM_OUTER = (198, 214, 232, 255)
RIM_INNER = (118, 142, 176, 255)
GOLD = (255, 196, 62, 255)
GOLD_DARK = (176, 112, 12, 255)
PLATE = (58, 34, 12, 235)


def rounded(size, radius, fill, outline=None, width=0) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [0, 0, size[0] - 1, size[1] - 1], radius=radius,
        fill=fill, outline=outline, width=width,
    )
    return img


def card_frame() -> Image.Image:
    """Скруглённая металлическая рамка с прозрачным окном под арт."""
    img = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Внешний кант
    d.rounded_rectangle([0, 0, CARD_W - 1, CARD_H - 1], radius=34,
                        outline=RIM_OUTER, width=14)
    d.rounded_rectangle([13, 13, CARD_W - 14, CARD_H - 14], radius=26,
                        outline=RIM_INNER, width=6)

    # Внутреннее окно вырезаем в альфе, чтобы арт был виден насквозь
    window = Image.new("L", (CARD_W, CARD_H), 255)
    wd = ImageDraw.Draw(window)
    wd.rounded_rectangle([36, 92, CARD_W - 36, 562], radius=18, fill=0)
    alpha = img.getchannel("A")
    img.putalpha(Image.composite(alpha, Image.new("L", img.size, 0), window))

    # Тонкая обводка вокруг окна
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([34, 90, CARD_W - 34, 564], radius=20,
                        outline=RIM_INNER, width=5)
    return img


def nameplate() -> Image.Image:
    img = rounded((400, 84), 20, PLATE, outline=GOLD_DARK, width=4)
    d = ImageDraw.Draw(img)
    d.line([(16, 8), (384, 8)], fill=(255, 255, 255, 40), width=3)
    return img


def stars(n: int) -> Image.Image:
    """Полоса из n звёзд, отцентрованная по ширине."""
    w, h = 256, 72
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = 22
    step = min(46, (w - 20) // max(n, 1))
    total = step * (n - 1)
    cx0 = (w - total) // 2
    for i in range(n):
        cx, cy = cx0 + i * step, h // 2
        pts = []
        for k in range(10):
            import math
            ang = math.pi / 2 + k * math.pi / 5
            rad = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
        d.polygon(pts, fill=GOLD, outline=GOLD_DARK)
    return img


def set_panel() -> Image.Image:
    """Обводка панели набора."""
    w, h = 1680, 1120
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=40, outline=GOLD_DARK, width=10)
    d.rounded_rectangle([18, 18, w - 19, h - 19], radius=32,
                        outline=(255, 255, 255, 34), width=3)
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    for name, img in (
        ("card_frame.png", card_frame()),
        ("nameplate.png", nameplate()),
        ("set_panel.png", set_panel()),
    ):
        img.save(OUT / name)
        written.append(name)

    for n in range(1, 6):
        stars(n).save(OUT / f"stars_{n}.png")
        written.append(f"stars_{n}.png")

    print(f"Записано в {OUT}:")
    for name in written:
        print(f"  {name}")
    print("\nЗамени эти файлы настоящими ассетами проекта — шаблоны менять не нужно.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
