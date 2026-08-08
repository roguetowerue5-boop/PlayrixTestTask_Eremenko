#!/usr/bin/env python
"""Готовит датасет для дообучения LoRA на стиле игры.

Зачем LoRA. Промптом можно описать стиль словами, но словами не передаётся
то, что делает картинку «из этой игры»: конкретная мера глянца, характер
скруглений, как именно ложится блик, насколько плотная тень. Модель, дообученная
на полутора сотнях эталонных артов, попадает в стиль без длинных описаний
и без референсов в каждом запросе.

Что делает скрипт:

* берёт готовые арты из папки Icons — это уже вырезанные карточки, лучший
  материал, какой может быть;
* приводит их к квадрату нужной стороны, не растягивая: недостающее поле
  дописывается краевым цветом, потому что кроп срезал бы объект, а чёрные
  поля модель выучила бы как часть стиля;
* пишет к каждой картинке подпись с триггер-словом.

Подписи. Обучение без подписей даёт LoRA, которая тянет стиль на всё подряд.
С подписью вида «<trigger>, стилевые слова, описание объекта» модель разводит
понятия: по триггеру — стиль, по остальному — содержание. Описание объекта
взять неоткуда автоматически, поэтому по умолчанию пишется общая строка, а
`--captions` подхватывает готовый JSON вида {"CardC6_card_01": "rose arch"}.

Запуск:
    python tools/build_lora_dataset.py --src ../Icons --out lora/dataset
    python tools/build_lora_dataset.py --src ../Icons --out lora/dataset \
        --captions lora/captions.json --size 1024
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

# Триггер-слово. Редкий токен, чтобы не пересечься с тем, что модель уже
# знает: на «playrix» или «cartoon» у неё есть свои представления, и LoRA
# будет спорить с ними вместо того, чтобы дополнять.
TRIGGER = "plrxcard"

BASE_CAPTION = (
    "stylized 3D render game card art, glossy materials, bold saturated colors, "
    "soft key light from upper left, rounded chunky shapes, single centered object"
)


def squarify(img: Image.Image, size: int) -> Image.Image:
    """Вписывает картинку в квадрат, продолжая её края в поля.

    Растяжение исказило бы пропорции — LoRA выучила бы их как стиль. Кроп
    срезал бы часть объекта. Заливка полей одним цветом даёт видимую рамку,
    и модель выучивает уже её.

    Поэтому поля — растянутые краевые полоски самой картинки. Фон у карточек
    ровный, поэтому продолжение выглядит как естественное продолжение фона,
    а границы поля попросту не видно.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    rw, rh = resized.size
    if (rw, rh) == (size, size):
        return resized

    canvas = Image.new("RGB", (size, size))
    ox, oy = (size - rw) // 2, (size - rh) // 2

    if ox > 0:      # поля слева и справа
        left = resized.crop((0, 0, 1, rh)).resize((ox, rh), Image.NEAREST)
        right = resized.crop((rw - 1, 0, rw, rh)).resize((size - ox - rw, rh),
                                                         Image.NEAREST)
        canvas.paste(left, (0, oy))
        canvas.paste(right, (ox + rw, oy))
    if oy > 0:      # поля сверху и снизу
        top = resized.crop((0, 0, rw, 1)).resize((rw, oy), Image.NEAREST)
        bottom = resized.crop((0, rh - 1, rw, rh)).resize((rw, size - oy - rh),
                                                          Image.NEAREST)
        canvas.paste(top, (ox, 0))
        canvas.paste(bottom, (ox, oy + rh))

    canvas.paste(resized, (ox, oy))
    return canvas


def build(src: Path, out: Path, size: int, captions: dict[str, str],
          make_zip: bool) -> int:
    files = sorted(p for p in src.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    if not files:
        print(f"В {src} нет картинок", file=sys.stderr)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if old.suffix.lower() in (".png", ".txt"):
            old.unlink()

    written = 0
    for i, path in enumerate(files, 1):
        try:
            with Image.open(path) as im:
                square = squarify(im, size)
        except OSError as e:
            print(f"  пропускаю {path.name}: {e}", file=sys.stderr)
            continue

        stem = f"{i:03d}_{path.stem}"
        square.save(out / f"{stem}.png")

        subject = captions.get(path.stem) or captions.get(path.name)
        caption = f"{TRIGGER}, {BASE_CAPTION}"
        if subject:
            caption += f", {subject}"
        (out / f"{stem}.txt").write_text(caption, encoding="utf-8")
        written += 1

    print(f"Готово: {written} пар картинка+подпись в {out}")
    print(f"Триггер-слово: {TRIGGER}")
    if not captions:
        print("Подписи общие. Индивидуальные описания объектов заметно "
              "улучшают разделение стиля и содержания — см. --captions.")

    if make_zip:
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                z.write(f, f.name)
        print(f"Архив для загрузки в тренер: {archive} "
              f"({archive.stat().st_size / 1e6:.1f} МБ)")
    return written


def pick_art_refs(src: Path, count: int) -> None:
    """Кладёт несколько эталонных артов в style/art-refs.

    Это работает уже сейчас, без всякого обучения: генерация прикладывает
    их референсами к каждому запросу, и модель видит, как должна выглядеть
    картинка внутри карточки. Раньше в референсы шли только куски скина —
    они задают палитру интерфейса, но не подачу самого арта.

    Берём с равномерным шагом по всему списку: подряд идущие карточки
    обычно из одной коллекции и показали бы модели одну тему вместо
    разнообразия композиций.
    """
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in
                   (".png", ".jpg", ".jpeg", ".webp"))
    if not files:
        return

    out = Path(__file__).resolve().parent.parent / "style" / "art-refs"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    step = max(1, len(files) // count)
    for i, path in enumerate(files[::step][:count]):
        with Image.open(path) as im:
            ref = im.convert("RGB")
            ref.thumbnail((768, 768), Image.LANCZOS)
            ref.save(out / f"{i:02d}_{path.stem}.png")
    print(f"Эталонных артов в референсы: {min(count, len(files))} → {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="папка с эталонными артами")
    ap.add_argument("--out", type=Path, default=Path("lora/dataset"))
    ap.add_argument("--size", type=int, default=1024,
                    help="сторона квадрата, обычно 1024")
    ap.add_argument("--captions", type=Path,
                    help='JSON {"имя_файла_без_расширения": "описание объекта"}')
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--art-refs", type=int, metavar="N",
                    help="дополнительно положить N эталонных артов в "
                         "style/art-refs — они пойдут референсами в каждую "
                         "генерацию и работают уже сейчас, без обучения")
    args = ap.parse_args()

    if args.art_refs:
        pick_art_refs(args.src, args.art_refs)

    captions: dict[str, str] = {}
    if args.captions and args.captions.exists():
        captions = json.loads(args.captions.read_text(encoding="utf-8"))
        print(f"Подписей загружено: {len(captions)}")

    return 0 if build(args.src, args.out, args.size, captions,
                      not args.no_zip) else 1


if __name__ == "__main__":
    raise SystemExit(main())
