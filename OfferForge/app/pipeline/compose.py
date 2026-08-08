"""Композитор: собирает оффер из отдельных элементов.

Всё, что можно собрать локально, собирается локально. Нейросеть отвечает
только за содержимое слоёв image_gen — рамки, плашки, звёзды и текст
детерминированы, поэтому одинаковы на всех офферах по построению.

Слои сортируются по z и накладываются по очереди на RGBA-канву.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.config import STYLE_DIR
from app.models import Box, Layer, OfferPlan, OfferTemplate, TextStyle

log = logging.getLogger("offerforge.compose")

_VAR = re.compile(r"\$\{([\w.]+)\}")


# ---------------------------------------------------------------------------
# Подстановка переменных в поля шаблона
# ---------------------------------------------------------------------------

def _lookup(path: str, ctx: dict) -> str:
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return ""
        else:
            return ""
        if cur is None:
            return ""
    return str(cur)


def interpolate(value: str | None, ctx: dict) -> str:
    if not value:
        return ""
    return _VAR.sub(lambda m: _lookup(m.group(1), ctx), value)


def _hex_to_rgba(value: str | None, fallback=(0, 0, 0, 0)) -> tuple[int, int, int, int]:
    if not value:
        return fallback
    s = value.strip().lstrip("#")
    try:
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)
        if len(s) == 8:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
        if len(s) == 3:
            return tuple(int(c * 2, 16) for c in s) + (255,)  # type: ignore[return-value]
    except ValueError:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------

def _fit(img: Image.Image, w: int, h: int, mode: str) -> Image.Image:
    if w <= 0 or h <= 0:
        return img
    if mode == "stretch":
        return img.resize((w, h), Image.LANCZOS)
    src_ratio, dst_ratio = img.width / img.height, w / h
    if (mode == "cover") == (src_ratio > dst_ratio):
        new_h = h if mode == "cover" else int(w / src_ratio)
        new_w = int(new_h * src_ratio) if mode == "cover" else w
    else:
        new_w = w if mode == "cover" else int(h * src_ratio)
        new_h = int(new_w / src_ratio) if mode == "cover" else h
    img = img.resize((max(new_w, 1), max(new_h, 1)), Image.LANCZOS)
    if mode == "cover":
        left, top = (img.width - w) // 2, (img.height - h) // 2
        img = img.crop((left, top, left + w, top + h))
    return img


def _font(style: TextStyle) -> ImageFont.FreeTypeFont:
    candidates = [
        STYLE_DIR / "fonts" / f"{style.font}.ttf",
        STYLE_DIR / "fonts" / "default.ttf",
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), style.size)
            except OSError:
                continue
    for sys_font in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(sys_font, style.size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(canvas: Image.Image, text: str, box: Box, style: TextStyle) -> None:
    if not text:
        return
    if style.uppercase:
        text = text.upper()

    font = _font(style)
    d = ImageDraw.Draw(canvas)

    # Ужимаем шрифт, пока строка не влезет в бокс.
    size = style.size
    while size > 8:
        bbox = d.textbbox((0, 0), text, font=font, stroke_width=style.stroke_width)
        if bbox[2] - bbox[0] <= box.w:
            break
        size -= 2
        font = _font(style.model_copy(update={"size": size}))

    bbox = d.textbbox((0, 0), text, font=font, stroke_width=style.stroke_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = {"left": box.x, "center": box.x + (box.w - tw) // 2, "right": box.x + box.w - tw}[style.align]
    y = box.y + (box.h - th) // 2 - bbox[1]

    if style.shadow_offset:
        dx, dy = style.shadow_offset
        d.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 90))

    d.text(
        (x, y),
        text,
        font=font,
        fill=_hex_to_rgba(style.color, (255, 255, 255, 255)),
        stroke_width=style.stroke_width,
        stroke_fill=_hex_to_rgba(style.stroke_color, (0, 0, 0, 255)) if style.stroke_color else None,
    )


def _gradient(w: int, h: int, c1, c2) -> Image.Image:
    img = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(c1, c2)))
    return img


def _grid_cells(box: Box, cols: int, rows: int, gap: int) -> list[Box]:
    cw = (box.w - gap * (cols - 1)) // cols
    ch = (box.h - gap * (rows - 1)) // rows
    return [
        Box(x=box.x + c * (cw + gap), y=box.y + r * (ch + gap), w=cw, h=ch)
        for r in range(rows)
        for c in range(cols)
    ]


# ---------------------------------------------------------------------------
# Основной проход
# ---------------------------------------------------------------------------

class Compositor:
    def __init__(self, template: OfferTemplate, assets_dir: Path | None = None,
                 part_versions: dict[str, str] | None = None):
        self.tpl = template
        self.assets_dir = assets_dir or STYLE_DIR
        # Версии составляющих на этот прогон. Задаются снаружи, чтобы одна
        # и та же раскладка могла собираться разными наборами элементов.
        self.part_versions = part_versions or {}
        self._part_cache: dict[tuple[str, str], Any] = {}

    def _load_asset(self, rel: str) -> Image.Image | None:
        p = self.assets_dir / rel
        if not p.exists():
            return None
        try:
            return Image.open(p).convert("RGBA")
        except OSError:
            return None

    def _load_part(self, kind: str, asset_name: str | None) -> Image.Image | None:
        """Берёт файл из выбранной версии составляющей."""
        from app import parts as parts_lib

        key = (kind, self.part_versions.get(kind, ""))
        if key not in self._part_cache:
            self._part_cache[key] = parts_lib.resolve(kind, self.part_versions.get(kind))
        version = self._part_cache[key]
        if version is None:
            return None

        candidates = [asset_name] if asset_name else []
        candidates += version.assets
        for name in candidates:
            if not name:
                continue
            path = version.asset_path(name)
            if path.exists():
                try:
                    return Image.open(path).convert("RGBA")
                except OSError:
                    continue
        return None

    def font_style(self, base: TextStyle) -> TextStyle:
        """Стиль надписи из версии составляющей font, если она выбрана.

        Шрифт — такая же составляющая, как рамка: его нельзя отдавать
        генератору, иначе текст поедет.
        """
        from app import parts as parts_lib

        version = parts_lib.resolve("font", self.part_versions.get("font"))
        if version is None or version.font is None:
            return base
        f = version.font
        patch = {
            "size": f.size, "color": f.color, "stroke_color": f.stroke_color,
            "stroke_width": f.stroke_width, "uppercase": f.uppercase,
        }
        if f.file:
            patch["font"] = Path(f.file).stem
        return base.model_copy(update=patch)

    def render(
        self,
        plan: OfferPlan,
        art: dict[str, bytes],
        *,
        placeholder_missing: bool = True,
    ) -> Image.Image:
        """Собирает финальный композит.

        art — байты сгенерированных картинок по ключу слота. Для сеточных
        слоёв ключи вида "<slot>#<index>".
        """
        # Редкость берём из элемента, если он в плане один: так вложенная
        # карточка получает свою звёздность, а не единицу по умолчанию.
        rarity = plan.elements[0].rarity if len(plan.elements) == 1 else 1
        ctx = {
            "palette": plan.palette or ["#2C2050", "#150C2E"],
            "title": plan.title,
            "rarity": rarity,
            **plan.texts,
        }

        canvas = Image.new(
            "RGBA", (self.tpl.canvas_w, self.tpl.canvas_h), _hex_to_rgba(self.tpl.background)
        )

        for layer in sorted(self.tpl.layers, key=lambda l: l.z):
            if layer.repeat_for:
                self._render_repeated(canvas, layer, plan, art, ctx, placeholder_missing)
            else:
                self._render_single(canvas, layer, plan, art, ctx, placeholder_missing)

        return canvas

    # -- отдельный слой ------------------------------------------------------

    def _render_single(self, canvas, layer: Layer, plan, art, ctx, placeholder) -> None:
        box = layer.box
        tile = self._layer_image(layer, plan, art, ctx, box, placeholder, index=None)
        if tile is not None:
            self._paste(canvas, tile, box, layer)

        if layer.kind == "text":
            text = interpolate(f"${{{layer.text_slot}}}", ctx) if layer.text_slot else ""
            draw_text(canvas, text, box, self.font_style(layer.style))

    def _render_nested(self, canvas, layer: Layer, plan, art, cell, index: int) -> None:
        """Собирает вложенный шаблон и вклеивает результат в ячейку.

        Так набор состоит из настоящих карточек: рамка, звёзды и подпись
        рисуются в собственном масштабе шаблона и потом уменьшаются целиком,
        а не растягиваются каждая под размер ячейки.
        """
        from app.config import load_template

        try:
            sub_tpl = load_template(layer.template or "card")
        except FileNotFoundError:
            log.warning("Вложенный шаблон '%s' не найден", layer.template)
            return

        element = plan.elements[index]
        sub_plan = plan.model_copy(update={
            "elements": [element],
            "texts": {**plan.texts,
                      "title": plan.texts.get(f"element_{index}") or element.subject},
        })

        # Ключи арта у набора вида "cards#3"; вложенная карточка ждёт "art".
        sub_art = {}
        raw = art.get(f"{layer.id}#{index}")
        if raw:
            gen = next(iter(sub_tpl.gen_layers()), None)
            sub_art[gen.id if gen else "art"] = raw

        sub = Compositor(sub_tpl, self.assets_dir, self.part_versions)
        tile = sub.render(sub_plan, sub_art, placeholder_missing=True)

        # contain, а не cover: карточку нельзя обрезать по краю ячейки,
        # иначе срежется рамка. Остаток центрируем.
        tile = _fit(tile, cell.w, cell.h, "contain")
        canvas.alpha_composite(tile, (
            cell.x + max(0, (cell.w - tile.width) // 2),
            cell.y + max(0, (cell.h - tile.height) // 2),
        ))

    def _render_repeated(self, canvas, layer: Layer, plan, art, ctx, placeholder) -> None:
        cols, rows = layer.grid or (len(plan.elements), 1)
        cells = _grid_cells(layer.box, cols, rows, layer.gap)

        for i, cell in enumerate(cells):
            if i >= len(plan.elements):
                break
            element = plan.elements[i]
            local_ctx = {**ctx, "rarity": element.rarity, "subject": element.subject}

            if layer.kind == "template":
                self._render_nested(canvas, layer, plan, art, cell, i)
                continue

            if layer.kind == "text":
                style = self.font_style(layer.style)
                draw_text(canvas, element.subject, Box(
                    x=cell.x, y=cell.y + cell.h - style.size - 12,
                    w=cell.w, h=style.size + 8,
                ), style)
                continue

            tile = self._layer_image(layer, plan, art, local_ctx, cell, placeholder, index=i)
            if tile is not None:
                self._paste(canvas, tile, cell, layer)

    def _layer_image(self, layer: Layer, plan, art, ctx, box: Box, placeholder, index) -> Image.Image | None:
        if layer.kind == "solid":
            return Image.new("RGBA", (box.w, box.h), _hex_to_rgba(interpolate(layer.color, ctx)))

        if layer.kind == "gradient":
            return _gradient(
                box.w, box.h,
                _hex_to_rgba(interpolate(layer.color, ctx), (40, 30, 80, 255)),
                _hex_to_rgba(interpolate(layer.color_to, ctx), (20, 12, 46, 255)),
            )

        if layer.kind == "asset":
            img = self._load_asset(interpolate(layer.asset, ctx))
            return _fit(img, box.w, box.h, "stretch") if img else None

        if layer.kind == "part":
            img = self._load_part(
                layer.part or layer.id,
                interpolate(layer.part_asset, ctx) or None,
            )
            if not img:
                return None
            # Вписываем без искажения. Растяжение уродует всё, у чего
            # пропорции не совпали с боксом: одна звезда 122×118 в боксе
            # 256×72 расплывалась в жёлтую кляксу. Кто хочет заполнения —
            # ставит fit: stretch в шаблоне явно.
            mode = layer.fit if layer.fit == "stretch" else "contain"
            return _fit(img, box.w, box.h, mode)

        if layer.kind == "image_gen":
            key = f"{layer.id}#{index}" if index is not None else layer.id
            raw = art.get(key) or art.get(layer.prompt_slot or layer.id)
            if raw:
                return _fit(Image.open(BytesIO(raw)).convert("RGBA"), box.w, box.h, layer.fit)
            if placeholder:
                ph = Image.new("RGBA", (box.w, box.h), (60, 60, 70, 255))
                d = ImageDraw.Draw(ph)
                d.rectangle([0, 0, box.w - 1, box.h - 1], outline=(140, 140, 160, 255), width=3)
                d.line([(0, 0), (box.w, box.h)], fill=(90, 90, 105, 255), width=2)
                d.line([(0, box.h), (box.w, 0)], fill=(90, 90, 105, 255), width=2)
                return ph
            return None

        return None

    @staticmethod
    def _paste(canvas: Image.Image, tile: Image.Image, box: Box, layer: Layer) -> None:
        if layer.rotate:
            tile = tile.rotate(layer.rotate, expand=True, resample=Image.BICUBIC)
        if layer.opacity < 1.0:
            alpha = tile.getchannel("A").point(lambda v: int(v * layer.opacity))
            tile.putalpha(alpha)
        canvas.alpha_composite(tile, (box.x, box.y))


def contact_sheet(images: list[Image.Image], cols: int = 3, gap: int = 24,
                  bg=(24, 18, 40, 255)) -> Image.Image:
    """Лист со всеми вариантами оффера рядом — то, что уходит на доску."""
    if not images:
        return Image.new("RGBA", (400, 200), bg)
    tw = max(i.width for i in images)
    th = max(i.height for i in images)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new(
        "RGBA", (cols * tw + gap * (cols + 1), rows * th + gap * (rows + 1)), bg
    )
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        sheet.alpha_composite(img, (gap + c * (tw + gap), gap + r * (th + gap)))
    return sheet
