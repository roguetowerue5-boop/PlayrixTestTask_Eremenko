"""Базовая версия составляющих — чтобы приложение работало из коробки.

Рамка, звёзды, плашка, бейдж и кнопка рисуются кодом и кладутся в
библиотеку как версия `classic`. Когда появится эталонная карточка,
экран «Составляющие» разберёт её и добавит рядом новую версию —
переключение делается в один выпадающий список, шаблоны не трогаются.

    python tools/make_parts.py [имя-версии]
"""
from __future__ import annotations

import math
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import parts  # noqa: E402
from app.parts import FontStyle  # noqa: E402

CARD_W, CARD_H = 512, 720
RIM_OUTER = (198, 214, 232, 255)
RIM_INNER = (118, 142, 176, 255)
GOLD = (255, 196, 62, 255)
GOLD_DARK = (176, 112, 12, 255)
PLATE = (58, 34, 12, 235)


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def frame() -> Image.Image:
    """Скруглённая рамка с прозрачным окном под арт."""
    img = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, CARD_W - 1, CARD_H - 1], radius=34,
                        outline=RIM_OUTER, width=14)
    d.rounded_rectangle([13, 13, CARD_W - 14, CARD_H - 14], radius=26,
                        outline=RIM_INNER, width=6)

    window = Image.new("L", (CARD_W, CARD_H), 255)
    ImageDraw.Draw(window).rounded_rectangle(
        [36, 92, CARD_W - 36, 562], radius=18, fill=0)
    alpha = img.getchannel("A")
    img.putalpha(Image.composite(alpha, Image.new("L", img.size, 0), window))

    ImageDraw.Draw(img).rounded_rectangle(
        [34, 90, CARD_W - 34, 564], radius=20, outline=RIM_INNER, width=5)
    return img


def nameplate() -> Image.Image:
    img = Image.new("RGBA", (400, 84), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 399, 83], radius=20, fill=PLATE,
                        outline=GOLD_DARK, width=4)
    d.line([(16, 8), (384, 8)], fill=(255, 255, 255, 40), width=3)
    return img


def stars(n: int) -> Image.Image:
    w, h, r = 256, 72, 22
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    step = min(46, (w - 20) // max(n, 1))
    cx0 = (w - step * (n - 1)) // 2
    for i in range(n):
        cx, cy = cx0 + i * step, h // 2
        pts = []
        for k in range(10):
            ang = math.pi / 2 + k * math.pi / 5
            rad = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
        d.polygon(pts, fill=GOLD, outline=GOLD_DARK)
    return img


def badge() -> Image.Image:
    """Круглый бейдж «+N» — цифра дорисовывается композитором."""
    size = 104
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, size - 5, size - 5], fill=(206, 44, 58, 255),
              outline=(255, 236, 190, 255), width=5)
    d.ellipse([16, 12, size - 17, size // 2], fill=(255, 255, 255, 38))
    return img


def button() -> Image.Image:
    w, h = 320, 96
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 6, w - 1, h - 1], radius=24, fill=(46, 128, 44, 255))
    d.rounded_rectangle([0, 0, w - 1, h - 9], radius=24, fill=(88, 196, 74, 255),
                        outline=(226, 255, 200, 255), width=4)
    d.rounded_rectangle([18, 12, w - 19, h // 2 - 4], radius=16,
                        fill=(255, 255, 255, 46))
    return img


def panel() -> Image.Image:
    w, h = 1680, 1120
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=40, outline=GOLD_DARK, width=10)
    d.rounded_rectangle([18, 18, w - 19, h - 19], radius=32,
                        outline=(255, 255, 255, 34), width=3)
    return img


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "classic"

    made = [
        ("frame", {"frame.png": _png(frame())},
         "Металлическая рамка со скруглёнными углами и окном под арт"),
        ("stars", {f"{n}.png": _png(stars(n)) for n in range(1, 6)},
         "Золотые звёзды, отдельный файл на каждое количество"),
        ("nameplate", {"plate.png": _png(nameplate())},
         "Тёмная плашка с золотым кантом"),
        ("badge", {"badge.png": _png(badge())},
         "Красный круглый бейдж под количество"),
        ("button", {"buy.png": _png(button())},
         "Зелёная кнопка покупки"),
        ("panel", {"panel.png": _png(panel())},
         "Рамка вокруг всего набора"),
    ]

    for kind, assets, note in made:
        meta = parts.save_version(
            kind, version, assets=assets, title=version,
            note=note, source="сгенерировано кодом", make_default=True,
        )
        print(f"  {kind:10} {version}  ({', '.join(meta.assets)})")

    parts.save_version(
        "font", version, title=version,
        note="Жирный гротеск с тёмной обводкой и мягкой тенью",
        source="сгенерировано кодом", make_default=True,
        font=FontStyle(size=40, color="#FFF3D0", stroke_color="#5A2E00",
                       stroke_width=5, uppercase=True),
    )
    print(f"  {'font':10} {version}  (описание стиля)")

    print(f"\nВерсия «{version}» записана в {parts.PARTS_DIR} и выбрана по умолчанию.")
    print("Разбери эталонную карточку на экране «Составляющие», чтобы добавить свою.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
