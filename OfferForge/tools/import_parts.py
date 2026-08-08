#!/usr/bin/env python
"""Импорт готовых элементов в библиотеку составляющих.

Автоматика вырезает не всё и не всегда точно. Рамку, например, взять
надёжно так и не удалось: она кольцо, лежит вплотную к плашке и звёздам, и
любой автоматический способ либо тащит соседей, либо съедает кант.

Вырезанное руками — лучший материал, какой может быть. Этот скрипт кладёт
такие файлы в библиотеку как обычную версию: дальше они ничем не
отличаются от полученных сегментацией, и сборка берёт их так же.

Соответствие имён гибкое: сравниваются только буквы и цифры в нижнем
регистре, поэтому `BackGround.png`, `background.PNG` и `back_ground.png` —
одно и то же. Опечатки вида `5starts` тоже учтены: их проще принять, чем
просить переименовать файл.

Запуск:
    python tools\\import_parts.py --src ..\\Cutted --version blue
    python tools\\import_parts.py --src ..\\Cutted --version blue --default
    python tools\\import_parts.py --src ..\\Cutted --list
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app import parts  # noqa: E402

# Имя файла (только буквы и цифры) → (тип, имя ассета в библиотеке).
# Синонимов много намеренно: называть файлы по-своему естественно, а
# заставлять переименовывать — лишний повод для ошибки.
MAPPING: dict[str, tuple[str, str]] = {
    "border": ("frame", "frame.png"),
    "frame": ("frame", "frame.png"),
    "cardframe": ("frame", "frame.png"),

    "background": ("panel", "panel.png"),
    "panel": ("panel", "panel.png"),
    "backpanel": ("panel", "panel.png"),
    "ribbonwithribbon": ("panel", "panel.png"),   # панель со встроенной лентой

    "ribbon": ("ribbon", "ribbon.png"),
    "title": ("ribbon", "ribbon.png"),
    "header": ("ribbon", "ribbon.png"),

    "progressbar": ("progress", "progress.png"),
    "progress": ("progress", "progress.png"),

    "nameplate": ("nameplate", "plate.png"),
    "plate": ("nameplate", "plate.png"),
    "namebar": ("nameplate", "plate.png"),

    "badge": ("badge", "badge.png"),
    "plus": ("badge", "badge.png"),
    "quantity": ("badge", "badge.png"),

    "close": ("close", "close.png"),
    "cross": ("close", "close.png"),
    "closebutton": ("close", "close.png"),

    "footer": ("footer", "footer.png"),
    "setplate": ("footer", "footer.png"),

    "button": ("button", "buy.png"),
    "buy": ("button", "buy.png"),

    # Звёзды: по файлу на количество. «starts» — не опечатка в коде, а
    # частая опечатка в именах файлов.
    "star": ("stars", "1.png"),
    "1star": ("stars", "1.png"),
    "2stars": ("stars", "2.png"),
    "3stars": ("stars", "3.png"),
    "4stars": ("stars", "4.png"),
    "5stars": ("stars", "5.png"),
    "2starts": ("stars", "2.png"),
    "3starts": ("stars", "3.png"),
    "4starts": ("stars", "4.png"),
    "5starts": ("stars", "5.png"),
}


# Элементы с впечатанной надписью и доля края, которая считается чистым
# фоном. Текст всегда по центру, поэтому центр заменяется растянутой
# полоской края — это 9-slice, только по горизонтали.
TEXTED = {"nameplate": 0.14, "ribbon": 0.18, "footer": 0.16}


def key(name: str) -> str:
    """Имя файла без расширения, только буквы и цифры, в нижнем регистре."""
    return re.sub(r"[^a-z0-9]", "", Path(name).stem.lower())


def check_alpha(path: Path) -> tuple[bool, str]:
    """Есть ли у файла прозрачность и не пустой ли он.

    Элемент без альфы ляжет на панель прямоугольным пятном — это первое,
    что стоит проверить, потому что глазом в файловом менеджере такое не
    видно.
    """
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        lo, hi = rgba.getchannel("A").getextrema()
    if hi == 0:
        return False, "пустой — альфа нулевая целиком"
    if lo == 255:
        return False, "нет прозрачности — ляжет прямоугольником"
    opaque = sum(rgba.getchannel("A").histogram()[128:])
    frac = opaque / (rgba.width * rgba.height)
    if frac > 0.995:
        return False, f"почти непрозрачный ({frac:.0%})"
    return True, f"{rgba.width}x{rgba.height}, непрозрачно {frac:.0%}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="папка с вырезанными элементами")
    ap.add_argument("--version", default="hand",
                    help="имя версии в библиотеке")
    ap.add_argument("--note", default="вырезано руками")
    ap.add_argument("--default", action="store_true",
                    help="сделать версией по умолчанию")
    ap.add_argument("--list", action="store_true",
                    help="только показать, что куда ляжет")
    ap.add_argument("--keep-text", action="store_true",
                    help="не затирать надписи на плашках (по умолчанию затираются)")
    args = ap.parse_args()

    src = (Path(__file__).resolve().parent.parent / args.src).resolve()
    if not src.is_dir():
        print(f"Папки нет: {src}", file=sys.stderr)
        return 1

    # Собираем ассеты по типам: у звёзд их пять, у остальных по одному.
    by_kind: dict[str, dict[str, bytes]] = {}
    skipped: list[str] = []

    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in (".png", ".webp"):
            continue
        target = MAPPING.get(key(path.name))
        if not target:
            skipped.append(f"{path.name}: непонятно, какой это элемент")
            continue
        kind, asset = target

        ok, detail = check_alpha(path)
        mark = "ok  " if ok else " !  "
        print(f"{mark}{path.name:22} → {kind}/{asset:11} {detail}")
        if not ok:
            skipped.append(f"{path.name}: {detail}")
            continue
        if args.list:
            continue

        data = path.read_bytes()
        # Плашки приходят с чужой надписью — «Star Map» на плашке, «FARMING»
        # на ленте. Свою подпись рисует композитор, и без затирания два
        # текста накладываются друг на друга. Затираем растянутой полоской
        # края: центр плашки — текст, края — чистый фон.
        if kind in TEXTED and not args.keep_text:
            from io import BytesIO

            from app.pipeline.segment import clean_text_area

            # Только если надпись действительно есть. Вырезанные вручную
            # плашки часто уже чистые, и затирание такой портит её: край
            # растягивается на середину и оставляет серую полосу.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from prepare_parts import has_text

            with Image.open(BytesIO(data)) as im:
                rgba = im.convert("RGBA")
                if has_text(rgba, TEXTED[kind]):
                    buf = BytesIO()
                    clean_text_area(rgba, side_frac=TEXTED[kind]).save(
                        buf, format="PNG")
                    data = buf.getvalue()
                    print(f"    надпись затёрта (край {TEXTED[kind]:.0%})")
                else:
                    print("    надписи нет, оставляю как есть")

        by_kind.setdefault(kind, {})[asset] = data

    if args.list:
        return 0
    if not by_kind:
        print("\nНечего импортировать", file=sys.stderr)
        return 1

    print()
    for kind, assets in by_kind.items():
        meta = parts.save_version(
            kind, args.version, assets=assets, title=args.version,
            note=args.note, source=f"импорт из {src.name}",
            make_default=args.default)
        print(f"сохранено {kind}/{meta.version}: {', '.join(sorted(meta.assets))}")

    missing = [k for k in parts.PART_KINDS
               if k not in by_kind and k != "font"]
    if missing:
        print(f"\nНе нашлось в папке: {', '.join(missing)}. "
              "Эти типы возьмутся из версии по умолчанию.")
    if skipped:
        print("\nПропущено:")
        for s in skipped:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
