#!/usr/bin/env python
"""Готовит вырезанные руками элементы к сборке.

Вырезанное вручную точнее любой автоматики, но собирается не сразу: у
файлов остаются мелочи, из-за которых композитор ставит их не так, как
ожидается. Скрипт делает КОПИИ в отдельной папке и правит их, не трогая
исходники.

Что правится и почему:

* **Звёзды приводятся к общей ширине.** В скине полоска из пяти звёзд
  шире, чем одна звезда, — это естественно. Но композитор вписывает
  каждый файл в один и тот же бокс, и одна звезда 122×118 раздувалась до
  ширины полоски, теряя форму. Поля дописываются прозрачным, звёзды
  прижимаются к левому краю, как в игре.

* **Надписи затираются, только если они есть.** Плашка и лента могут
  прийти с текстом «Star Map», «FARMING» — свою подпись рисует композитор,
  и тогда два текста лягут друг на друга. Но вырезанные вручную файлы
  часто уже чистые, и затирание такой плашки её только портит: край
  растягивается на середину и оставляет серую кляксу. Поэтому текст
  сначала ищется, и молча ничего не трётся.

* **Обрезаются пустые поля.** Прозрачная кайма вокруг элемента съедает
  место в боксе и сдвигает элемент от края.

Запуск:
    python tools\\prepare_parts.py --src ..\\Cutted
    python tools\\prepare_parts.py --src ..\\Cutted --out ..\\Cutted-ready
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

# Сколько звёзд в файле — по имени. Ключ: имя без не-буквоцифр, в нижнем
# регистре. «starts» вместо «stars» — частая опечатка, принимаем как есть.
STAR_COUNT = {
    "star": 1, "1star": 1, "1stars": 1,
    "2stars": 2, "2starts": 2,
    "3stars": 3, "3starts": 3,
    "4stars": 4, "4starts": 4,
    "5stars": 5, "5starts": 5,
}

# Элементы с впечатанной надписью → доля края, которая считается фоном.
TEXTED = {"nameplate": 0.14, "plate": 0.14, "namebar": 0.14,
          "ribbon": 0.18, "title": 0.18, "header": 0.18,
          "footer": 0.16, "setplate": 0.16}


def key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def trim(img: Image.Image, pad: int = 1) -> Image.Image:
    """Срезает прозрачные поля, оставляя пиксель на сглаженный край."""
    a = img.getchannel("A").point(lambda v: 255 if v > 8 else 0)
    bbox = a.getbbox()
    if not bbox:
        return img
    return img.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                     min(img.width, bbox[2] + pad),
                     min(img.height, bbox[3] + pad)))


def has_text(img: Image.Image, side_frac: float, tol: float = 12.0) -> bool:
    """Есть ли на плашке впечатанная надпись.

    Текст всегда по центру, а края — ровный фон. Значит если середина
    пестрее краёв, там что-то написано. Сравниваем разброс яркости: у
    чистой плашки он в центре такой же, как по краям.

    Проверка нужна, чтобы не портить уже чистые файлы: затирание
    растягивает край на середину, и на плашке без текста остаётся серая
    полоса поперёк.
    """
    from PIL import ImageStat

    w, h = img.size
    side = max(2, int(w * side_frac))
    if w < side * 3:
        return False

    band = (0, int(h * 0.25), w, int(h * 0.75))
    grey = img.convert("RGBA")
    # Прозрачное мешает статистике — подкладываем ровный фон.
    flat = Image.new("RGB", grey.size, (0, 0, 0))
    flat.paste(grey.convert("RGB"), (0, 0), grey.getchannel("A"))
    flat = flat.crop(band).convert("L")

    left = ImageStat.Stat(flat.crop((0, 0, side, flat.height))).stddev[0]
    right = ImageStat.Stat(
        flat.crop((flat.width - side, 0, flat.width, flat.height))).stddev[0]
    middle = ImageStat.Stat(
        flat.crop((side, 0, flat.width - side, flat.height))).stddev[0]
    return middle > max(left, right) + tol


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="куда положить копии, по умолчанию <src>-ready")
    ap.add_argument("--clean-text", choices=("auto", "always", "never"),
                    default="auto",
                    help="затирать надписи: auto — только если текст найден")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = (root / args.src).resolve()
    out = (root / args.out).resolve() if args.out else src.parent / f"{src.name}-ready"
    if not src.is_dir():
        print(f"Папки нет: {src}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir())
             if p.suffix.lower() in (".png", ".webp")]
    if not files:
        print(f"В {src} нет картинок", file=sys.stderr)
        return 1

    # Сначала обрезаем всё и запоминаем звёзды: их ширину нужно знать целиком,
    # прежде чем выравнивать.
    prepared: dict[Path, Image.Image] = {}
    stars: dict[int, Path] = {}
    for path in files:
        with Image.open(path) as im:
            img = trim(im.convert("RGBA"))
        prepared[path] = img
        n = STAR_COUNT.get(key(path.name))
        if n:
            stars[n] = path

    # Звёзды: общая ширина по самой длинной полоске, высота по самой высокой.
    # Прижимаем к левому краю — в игре звёзды растут вправо от левого края.
    if stars:
        wide = max(prepared[p].width for p in stars.values())
        tall = max(prepared[p].height for p in stars.values())
        for n, path in stars.items():
            img = prepared[path]
            canvas = Image.new("RGBA", (wide, tall), (0, 0, 0, 0))
            canvas.paste(img, (0, (tall - img.height) // 2))
            prepared[path] = canvas
        print(f"звёзды выровнены по {wide}x{tall} "
              f"({', '.join(str(n) for n in sorted(stars))} шт)")

    from app.pipeline.segment import clean_text_area

    for path, img in prepared.items():
        k = key(path.name)
        note = ""
        if k in TEXTED and args.clean_text != "never":
            found = has_text(img, TEXTED[k])
            if args.clean_text == "always" or found:
                img = clean_text_area(img, side_frac=TEXTED[k])
                note = "  надпись затёрта"
            else:
                note = "  надписи нет, оставляю как есть"

        # Имя сохраняем: import_parts узнаёт элемент по нему.
        dst = out / f"{path.stem}.png"
        img.save(dst)
        with Image.open(path) as before:
            was = f"{before.width}x{before.height}"
        print(f"  {path.name:22} {was:>11} → {img.width}x{img.height}{note}")

    print(f"\nГотово: {len(prepared)} файлов в {out}")
    print("Дальше: python tools\\import_parts.py --src "
          f"{out.relative_to(root.parent) if out.is_relative_to(root.parent) else out}"
          " --version blue --keep-text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
