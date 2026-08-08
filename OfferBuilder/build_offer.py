#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Сборщик офферов.

Берёт нарезанные материалы из C:\\Playrix\\Cutted и собирает из них финальную
картинку оффера (подложка + 10 карточек товара), строго следуя алгоритму,
описанному в C:\\Playrix\\RuleForBuilding\\rule_building.txt:

    1. Обращение к картинке-референсу (blue_template.png), чтобы понять,
       какой результат должен получиться.
    2. Обращение к материалам (Cutted/*).
    3. Проверка всех материалов на наличие.
    4. Обращение к разметке (поиск области карточек на подложке, расчёт
       сетки 5x2 от её центра).
    5. Подстановка материалов в разметку.
    6. Проверка, что материалы встали на нужные позиции.

Что собирается на карточке (по разделу "Логика сборки" в rule_building.txt),
в порядке наложения слоёв:
    подложка -> рамка (border) -> неймплейт -> картинка (или заглушка
    из Icons) -> звёзды -> бейдж -> текст названия -> цифра заточки на бейдже.

Координаты элементов карточки заданы в ЛОКАЛЬНОЙ системе координат рамки,
где (0, 0) — это центр Border.png (см. класс BorderSpace ниже), как и
требует правило: "используются локальные координаты бордера, где 0.0 -
это центр бордера".

Запуск:
    python build_offer.py --config example_config.yaml --out result.png

Вход по умолчанию:
    Cutted:          C:\\Playrix\\Cutted
    Референс:        C:\\Playrix\\RuleForBuilding\\blue_template.png
    Заглушки картинок: C:\\Playrix\\Icons
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

log = logging.getLogger("offer_builder")


# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------

DEFAULT_CUTTED_DIR = Path(r"C:\Playrix\Cutted")
DEFAULT_RULE_DIR = Path(r"C:\Playrix\RuleForBuilding")
DEFAULT_ICONS_DIR = Path(r"C:\Playrix\Icons")
DEFAULT_REFERENCE = DEFAULT_RULE_DIR / "blue_template.png"
DEFAULT_FONT_DIR = Path(r"C:\Playrix\OfferForge\style\fonts")
# Nunito (OFL) — есть кириллица; Fredoka оставляем как латинский запасной.
DEFAULT_FONT_FILE = DEFAULT_FONT_DIR / "default.ttf"
DEFAULT_FONT_WEIGHT = 800  # ExtraBold для variable Nunito

# Материалы, обязательные для сборки (шаг 2-3 правила: "обращение к
# материалам" + "проверка всех материалов на наличие").
REQUIRED_MATERIALS = {
    "background": "BackGround.png",
    "border": "Border.png",
    "nameplate": "Nameplate.png",
    "badge": "Badge.png",
    "star_1": "Star.png",
    "star_2": "2stars.png",
    "star_3": "3stars.png",
    "star_4": "4stars.png",
    "star_5": "5starts.png",
}
# Не обязателен по тексту правила (там не описан), но есть среди материалов —
# используем, если явно включён в конфиге коллекции.
OPTIONAL_MATERIALS = {
    "progressbar": "Progressbar.png",
}

CARDS_PER_PAGE = 10
GRID_COLS, GRID_ROWS = 5, 2

# Итоговый размер оффера. Сборка всегда идёт в "родном" масштабе
# BackGround.png (3630x2414 — так устроены все локальные координаты и
# сетка), а на выходе картинку приводим к требуемому финальному размеру.
# 3587x2426 почти точно совпадает по пропорциям с содержимым референса
# (RuleForBuilding/blue_template.png, обрезанного по контенту: 3648x2467,
# те же ~1.48), так что перегонка в этот размер — лёгкое единое
# масштабирование (~1-2%), а не искажение картинки.
OUTPUT_SIZE = (3587, 2426)


# ---------------------------------------------------------------------------
# Локальные координаты рамки: (0, 0) = центр Border.png.
#
# Числа ниже вымерены двумя способами и сведены вместе:
#   - геометрия самого Border.png (где в нём прозрачное окно под арт и где
#     проходит горизонтальная плашка-разделитель под неймплейт);
#   - сверка с RuleForBuilding/blue_template.png (референс), на котором
#     измерены реальные позиции звёзд и бейджа относительно рамки карточки.
# Обе системы совпали в одном масштабе (1 px референса = 1 px Cutted),
# поэтому числа ниже — координаты в "натуральных" пикселях Border.png.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BorderSpace:
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.w / 2

    @property
    def cy(self) -> float:
        return self.h / 2

    def to_canvas_topleft(self, local_x: float, local_y: float, box_w: float, box_h: float) -> tuple[int, int]:
        """Локальные координаты (центр бокса или его якорь) -> верхний левый
        угол на канве рамки, готовый для Image.paste/alpha_composite."""
        return (round(self.cx + local_x - box_w / 2), round(self.cy + local_y - box_h / 2))


# Окно под картинку товара внутри рамки.
# Реальная дыра Border.png (flood-fill по alpha<40 от центра) ≈
# x:[52,458], y:[63,515]. Картинку кладём COVER/STRETCH на этот бокс
# (с запасом под полупрозрачную кромку) и обрезаем маской дыры —
# иначе сверху остаётся щель, через которую светится фон подложки.
IMAGE_HOLE_ALPHA = 40          # порог «дырки» в альфе рамки
IMAGE_MASK_DILATE = 31         # MaxFilter: арт заходит глубоко под antialias-кромку
# Fallback-бокс, если flood вдруг не сработает (старое окно + сдвиг вверх).
IMAGE_WINDOW_LOCAL = dict(cx=0.0, cy=(63 + 515) / 2 - 353.5, w=458 - 52 + 1, h=515 - 63 + 1)

# Неймплейт: его верхний край стоит ровно на разделительной плашке рамки
# (y=520 в Border.png) и по ширине центрирован на рамке. Подтверждено
# на референсе: центр неймплейта по x совпал с центром рамки день в день.
NAMEPLATE_LOCAL_TOP = 520 - 353.5  # y верхнего края неймплейта, локально

# Звёзды: замер Coolness — star_center − rim_top ≈ +0.5px (центр на верхнем
# краю канта), tip_overhang/h ≈ 0.48. У Border.png rim_top (центр) ≈ y=12;
# золотое ядро пресета чуть выше геометрического центра PNG (~4px), поэтому
# якорь пресета = 12+5 = 17, чтобы визуальный центр совпал с rim_top.
BORDER_TOP_MIDLINE = 17
# Макс. вылет кончика 5★: 131/2 − 17 ≈ 49px.
STAR_TIP_OVERHANG = 52

# Бейдж: по референсу левый верхний угол бейджа стоит вплотную к левому
# краю рамки и на border_top + 104px.
BADGE_LOCAL_TOPLEFT = (-255.5, 104 - 353.5)

# Центр текста на Badge.png (новый ассет пустой, без вшитого "+2").
# Координаты относительно самого бейджа; патч-подложку больше не кладём —
# она вырывала кусок из градиента.
BADGE_TEXT_CENTER = (98, 68)
BADGE_TEXT_MAX_W_FRAC = 0.72
BADGE_TEXT_SIZE = 56
BADGE_TEXT_COLOR = (90, 45, 10, 255)

# Неймплейт в Cutted чуть узковат относительно рамки → при апскейле
# финального оффера кромка даёт артефакты. Растягиваем до доли ширины
# Border.png перед вставкой (LANCZOS).
NAMEPLATE_WIDTH_FRAC = 0.92

# Канва карточки чуть выше рамки — только под вылет кончиков звёзд.
CARD_TOP_PAD = STAR_TIP_OVERHANG + 8
STAR_OVERHANG_MAX = STAR_TIP_OVERHANG + 4
# Плотный шаг как на Coolness (референс ~0.07–0.1 ширины карточки).
GRID_GAP_X = 28
GRID_GAP_Y = STAR_OVERHANG_MAX + 6
# Пресеты звёзд в Cutted чуть крупноваты относительно рамки (особенно 3–5),
# из‑за чего острия соседних звёзд в пресете визуально слипаются.
STAR_PRESET_SCALE = 0.88
# Progressbar.png в Cutted выше, чем полоса на Coolness — чуть уменьшаем,
# чтобы сетка карточек получила больше высоты и могла вырасти.
PROGRESS_BAR_SCALE = 0.78


# ---------------------------------------------------------------------------
# Данные оффера (то, что задаёт пользователь конфигом)
# ---------------------------------------------------------------------------

@dataclass
class CardItem:
    name: str
    rarity: int = 1                 # число звёзд, 1..5 (в тексте правила упомянуто 1..4,
                                     # но в Cutted есть и пресет на 5 — поддерживаем оба)
    image: str | None = None        # путь к картинке товара; пусто -> заглушка из Icons
    badge: str | None = None        # текст на бейдже ("+2" и т.п.); пусто -> бейджа нет


@dataclass
class OfferConfig:
    title: str
    set_progress: str | None = None     # напр. "8/25" — зарезервировано на будущее: Progressbar.png
                                         # используется как есть, со своим примером значения, т.к.
                                         # прогресс-бар не описан в rule_building.txt.
    show_progressbar: bool = False
    page: int | None = None             # номер страницы коллекции → футер «SET N/M»
    page_total: int = 10                # знаменатель в футере (обычно 10)
    items: list[CardItem] = field(default_factory=list)

    def __post_init__(self):
        if len(self.items) != CARDS_PER_PAGE:
            raise ValueError(
                f"На странице коллекции должно быть ровно {CARDS_PER_PAGE} карточек, "
                f"получено {len(self.items)}."
            )


def load_config(path: Path) -> OfferConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [CardItem(**it) for it in data.get("items", [])]
    return OfferConfig(
        title=data["title"],
        set_progress=data.get("set_progress"),
        show_progressbar=bool(data.get("show_progressbar", False)),
        page=data.get("page"),
        page_total=int(data.get("page_total", 10)),
        items=items,
    )


# ---------------------------------------------------------------------------
# Шаг 1-3: референс и материалы
# ---------------------------------------------------------------------------

def load_reference(reference_path: Path) -> Image.Image:
    log.info("Шаг 1. Обращение к картинке-референсу: %s", reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"Референс не найден: {reference_path}")
    return Image.open(reference_path).convert("RGBA")


def load_materials(cutted_dir: Path, use_progressbar: bool) -> dict[str, Image.Image]:
    log.info("Шаг 2. Обращение к материалам: %s", cutted_dir)
    needed = dict(REQUIRED_MATERIALS)
    if use_progressbar:
        needed.update(OPTIONAL_MATERIALS)

    missing = [fname for fname in needed.values() if not (cutted_dir / fname).exists()]
    log.info("Шаг 3. Проверка всех материалов на наличие (%d шт.)", len(needed))
    if missing:
        raise FileNotFoundError(
            "Не найдены обязательные материалы в " + str(cutted_dir) + ": " + ", ".join(missing)
        )

    materials = {key: Image.open(cutted_dir / fname).convert("RGBA") for key, fname in needed.items()}
    for key, fname in needed.items():
        log.info("  ok  %-12s %-16s %s", key, fname, materials[key].size)
    return materials


STAR_ASSET_BY_RARITY = {1: "star_1", 2: "star_2", 3: "star_3", 4: "star_4", 5: "star_5"}


# ---------------------------------------------------------------------------
# Шаг 4: разметка — находим область карточек на подложке и считаем сетку
# ---------------------------------------------------------------------------

def _content_bbox_ratio(img: Image.Image) -> float:
    """Соотношение сторон непрозрачного содержимого картинки."""
    bbox = img.getbbox()
    if not bbox:
        return 0.0
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w / h if h else 0.0


def verify_background_matches_reference(bg: Image.Image, reference: Image.Image, tolerance: float = 0.05) -> None:
    """Шаг: "проверяется что границы подложки соотносятся с референсом,
    проверяется что размер изображений одинаковый".

    Подложка (Cutted/BackGround.png) обрезана вплотную к контенту, а
    референс — нет, поэтому сравниваем не абсолютные размеры, а
    соотношение сторон непрозрачного содержимого."""
    bg_ratio = _content_bbox_ratio(bg)
    ref_ratio = _content_bbox_ratio(reference)
    diff = abs(bg_ratio - ref_ratio) / ref_ratio if ref_ratio else 1.0
    status = "OK" if diff <= tolerance else "ПРЕДУПРЕЖДЕНИЕ"
    log.info(
        "  Сверка с референсом: соотношение сторон подложки=%.3f, референса=%.3f, "
        "расхождение=%.1f%% [%s]",
        bg_ratio, ref_ratio, diff * 100, status,
    )
    if diff > tolerance:
        log.warning(
            "Подложка заметно отличается по пропорциям от референса (%.1f%% > %.0f%%). "
            "Продолжаю сборку, но результат стоит сверить вручную.",
            diff * 100, tolerance * 100,
        )


def _find_field_edge(mask, x: int, y0: int, dy: int, limit: int) -> int:
    y = y0
    while 0 <= y + dy < limit and mask[y + dy, x]:
        y += dy
    return y


def _find_field_edge_x(mask, y: int, x0: int, dx: int, limit: int) -> int:
    x = x0
    while 0 <= x + dx < limit and mask[y, x + dx]:
        x += dx
    return x


def detect_card_area(bg: Image.Image) -> tuple[int, int, int, int]:
    """Разбивает подложку на "название" и "область карточек" и возвращает
    прямоугольник области карточек (x0, y0, x1, y1).

    Область карточек — это залитое градиентом синее поле внутри золотой
    рамки подложки (всё, что ниже ленты с заголовком). Определяем его не
    захардкоженными числами, а по цвету: поле заметно бледнее и менее
    насыщенное, чем лента и внешняя синяя окантовка (у поля R-канал заметно
    выше нуля, а у ленты/окантовки R практически 0 при похожей яркости).

    Поиск края идёт "изнутри наружу" контурным проходом (а не глобальным
    min/max по маске), чтобы не спутать поле с нижним вырезом под футер
    и не зацепить скруглённые углы панели — ровно то, что требует правило:
    "проверяется положение бордеров относительно референса".
    """
    import numpy as np

    arr = np.array(bg).astype(int)
    h, w = arr.shape[:2]
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    field_mask = (a > 200) & (b > 150) & (r > 15) & (r < 150) & ((b - r) > 60)

    if not field_mask.any():
        raise ValueError("Не удалось найти область карточек на подложке (поле не распознано по цвету).")

    ys, xs = np.where(field_mask)
    seed_y, seed_x = int(np.median(ys)), int(np.median(xs))
    if not field_mask[seed_y, seed_x]:
        seed_y, seed_x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])

    sample_cols = [max(0, seed_x - w // 6), seed_x, min(w - 1, seed_x + w // 6)]
    sample_rows = [max(0, seed_y - h // 6), seed_y, min(h - 1, seed_y + h // 6)]

    tops = [_find_field_edge(field_mask, x, seed_y, -1, h) for x in sample_cols if field_mask[seed_y, x]]
    bots = [_find_field_edge(field_mask, x, seed_y, +1, h) for x in sample_cols if field_mask[seed_y, x]]
    lefts = [_find_field_edge_x(field_mask, y, seed_x, -1, w) for y in sample_rows if field_mask[y, seed_x]]
    rights = [_find_field_edge_x(field_mask, y, seed_x, +1, w) for y in sample_rows if field_mask[y, seed_x]]

    # Берём самые "тесные" значения по каждой стороне — так прямоугольник
    # гарантированно вписан в поле и не залезает на скруглённые углы или
    # вырез под футер.
    x0, x1 = max(lefts), min(rights)
    y0, y1 = max(tops), min(bots)
    return x0, y0, x1, y1


# Зазор между рамками — это НЕ "что осталось от области карточек после
# сетки", а собственный шаг раскладки: на референсе (blue_template.png)
# рамки стоят плотно, с небольшим постоянным зазором, а лишнее место в
# области карточек уходит в поля по краям (сетка просто центрируется в
# области, как и требует правило: "отступы от центра области карточек").
# Раньше зазор считался растяжением на всю ширину/высоту области — из-за
# этого рамки расползались с огромными промежутками на подложках, где
# область карточек шире, чем у референса.
#
# Числа по горизонтали измерены на референсе: шаг между колонками 562px
# при рамке 511px (зазор 51px). По вертикали зазор увеличен до GRID_GAP_Y,
# чтобы вылет звёзд нижнего ряда не наезжал на верхний ряд (см. выше).


def compute_grid(card_area: tuple[int, int, int, int], border_w: int, border_h: int,
                  gap_x: int = GRID_GAP_X, gap_y: int = GRID_GAP_Y) -> list[tuple[int, int]]:
    """10 позиций рамок (верхний левый угол) — 5 сверху, 5 снизу, плотным
    шагом gap_x/gap_y, сетка целиком центрируется в области карточек через
    отступы от её центра, как требует правило."""
    x0, y0, x1, y1 = card_area
    area_w, area_h = x1 - x0, y1 - y0
    area_cx, area_cy = (x0 + x1) / 2, (y0 + y1) / 2

    # Зазоры ФИКСИРОВАННЫЕ: только сжимаем, если сетка не влезает.
    # Раньше при нехватке высоты gap_x пересчитывался «на всю ширину» и
    # раздувался (51 → 95) — карточки расползались по горизонтали.
    if GRID_COLS * border_w + (GRID_COLS - 1) * gap_x > area_w:
        gap_x = max(4, (area_w - GRID_COLS * border_w) // (GRID_COLS - 1)) if area_w > GRID_COLS * border_w else 4
        log.warning("Область карточек узкая — gap_x сжат до %dpx.", gap_x)
    if GRID_ROWS * border_h + (GRID_ROWS - 1) * gap_y > area_h:
        min_gap_y = STAR_OVERHANG_MAX + 4
        natural_y = max(4, (area_h - GRID_ROWS * border_h) // (GRID_ROWS - 1)) if area_h > GRID_ROWS * border_h else 4
        gap_y = min(gap_y, max(min_gap_y, natural_y) if area_h > GRID_ROWS * border_h + min_gap_y else natural_y)
        log.warning("Область карточек низкая — gap_y сжат до %dpx.", gap_y)

    grid_w = GRID_COLS * border_w + (GRID_COLS - 1) * gap_x
    grid_h = GRID_ROWS * border_h + (GRID_ROWS - 1) * gap_y

    start_x = area_cx - grid_w / 2
    start_y = area_cy - grid_h / 2

    positions = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = start_x + col * (border_w + gap_x)
            y = start_y + row * (border_h + gap_y)
            positions.append((round(x), round(y)))
    return positions


def fit_card_scale(card_area: tuple[int, int, int, int], border_w: int, border_h: int) -> float:
    """Скейл карточки, чтобы сетка 5×2 с ФИКСИРОВАННЫМИ зазорами
    (включая вылет звёзд) заполнила область. 1.0 — родной размер Border.png.

    Зазоры не скейлим: они задают «воздух» под вылет звёзд в пикселях
    подложки. Скейл считает размер рамок; если место есть — можно чуть
    апскейлить (как на Coolness/Tour), а не оставлять пустые поля:
        COLS * border_w * s + (COLS-1) * gap_x <= area_w
    """
    x0, y0, x1, y1 = card_area
    area_w, area_h = x1 - x0, y1 - y0
    room_w = area_w - (GRID_COLS - 1) * GRID_GAP_X
    room_h = area_h - (GRID_ROWS - 1) * GRID_GAP_Y
    if room_w <= 0 or room_h <= 0:
        return 0.5
    scale = min(room_w / (GRID_COLS * border_w), room_h / (GRID_ROWS * border_h))
    return max(0.5, min(scale, 1.12))


def verify_grid_fits(card_area: tuple[int, int, int, int], positions: list[tuple[int, int]],
                      border_w: int, border_h: int, tolerance_px: int = 4) -> None:
    """"Проверяется положение бордеров относительно референса — они должны
    подходить друг к другу идеально." Здесь — проверка, что вся сетка
    вписана в область карточек без наложений и без выхода за края."""
    x0, y0, x1, y1 = card_area
    gx0 = min(p[0] for p in positions)
    gy0 = min(p[1] for p in positions)
    gx1 = max(p[0] + border_w for p in positions)
    gy1 = max(p[1] + border_h for p in positions)

    overflow = max(0, x0 - gx0, y0 - gy0, gx1 - x1, gy1 - y1)
    status = "OK" if overflow <= tolerance_px else "ПРЕДУПРЕЖДЕНИЕ"
    log.info(
        "  Сетка карточек: %s (область карточек %s), выход за границы=%dpx [%s]",
        (gx0, gy0, gx1, gy1), card_area, overflow, status,
    )
    if overflow > tolerance_px:
        log.warning("Сетка карточек выходит за пределы области на %dpx — рамки не поместились впритык.", overflow)


# ---------------------------------------------------------------------------
# Шрифт
# ---------------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    """Жирный гротеск для title / nameplate / footer.

    По умолчанию — Nunito ExtraBold (кириллица + латиница). Fredoka в
    репозитории без кириллицы: русские буквы рисовались как □□□.
    """
    candidates = [
        DEFAULT_FONT_FILE,
        DEFAULT_FONT_DIR / "Nunito-Variable.ttf",
        DEFAULT_FONT_DIR / "Nunito-ExtraBold.ttf",
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\comicbd.ttf"),
        DEFAULT_FONT_DIR / "Fredoka-Bold.ttf",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            continue
        # Variable Nunito: поднимаем вес до ExtraBold.
        try:
            names = font.get_variation_names() if hasattr(font, "get_variation_names") else []
            if names and hasattr(font, "set_variation_by_name"):
                for label in (b"ExtraBold", b"Black", b"Bold", "ExtraBold", "Black", "Bold"):
                    try:
                        font.set_variation_by_name(label)
                        break
                    except (OSError, ValueError, TypeError):
                        continue
            elif hasattr(font, "set_variation_by_axes"):
                try:
                    font.set_variation_by_axes([DEFAULT_FONT_WEIGHT])
                except (OSError, ValueError, TypeError):
                    pass
        except Exception:  # noqa: BLE001 — шрифт и так уже загружен
            pass
        return font
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Стили текста сверены с Packs/05_Collection_Coolness.png:
#   title     — жёлтый→оранжевый градиент + тёмно-коричневая обводка + тень
#   nameplate — белый fill + тёмно-синяя обводка/тень (как «VENDING» / «FAN»)
#   footer    — по умолчанию впечатан в BackGround; при config.page перерисовываем
TITLE_STYLE = dict(
    size=122, color=(255, 217, 61, 255), stroke_color=(74, 29, 0, 255),
    stroke_width=9, shadow=(0, 5),
    gradient=((255, 230, 80, 255), (255, 160, 40, 255)),  # top → bottom
)
NAMEPLATE_STYLE = dict(
    size=52, color=(255, 255, 255, 255), stroke_color=(26, 42, 107, 255),
    stroke_width=3, shadow=(0, 4),
)
FOOTER_STYLE = dict(
    size=54, color=(92, 48, 18, 255), stroke_color=None,
    stroke_width=0, shadow=None,
)


def _draw_fitted_text(canvas: Image.Image, text: str, cx: float, cy: float, max_w: float,
                       size: int, color, stroke_color, stroke_width: int, uppercase: bool = True,
                       shadow: tuple[int, int] | None = (0, 3),
                       gradient: tuple[tuple[int, ...], tuple[int, ...]] | None = None) -> None:
    if not text:
        return
    if uppercase:
        text = text.upper()
    if canvas.mode != "RGBA":
        raise ValueError("текст рисуется только на RGBA-канве")
    d = ImageDraw.Draw(canvas)
    font = _font(size)
    while size > 10:
        bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 2
        font = _font(size)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = cx - tw / 2 - bbox[0]
    y = cy - th / 2 - bbox[1]

    if shadow and (shadow[0] or shadow[1]):
        d.text(
            (x + shadow[0], y + shadow[1]), text, font=font,
            fill=(0, 0, 0, 110),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 110) if stroke_color else None,
        )

    if gradient is None:
        d.text((x, y), text, font=font, fill=color,
               stroke_width=stroke_width, stroke_fill=stroke_color)
        return

    # Обводка отдельно, заливка — вертикальный градиент через маску глифов
    # (как жёлто-оранжевый title в Coolness).
    if stroke_color and stroke_width:
        d.text((x, y), text, font=font, fill=(0, 0, 0, 0),
               stroke_width=stroke_width, stroke_fill=stroke_color)

    mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask).text((x, y), text, font=font, fill=255)
    ink = mask.getbbox()
    if ink is None:
        return
    x0, y0, x1, y1 = ink
    top, bot = gradient
    grad = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    gp = grad.load()
    h = max(y1 - y0, 1)
    for yy in range(y0, y1):
        t = (yy - y0) / h
        rgba = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(4))
        for xx in range(x0, x1):
            if mask.getpixel((xx, yy)):
                gp[xx, yy] = rgba
    canvas.alpha_composite(grad)


def _draw_badge_text(badge: Image.Image, text: str, cx: float, cy: float,
                      max_w: float, size: int, color: tuple[int, int, int, int]) -> None:
    """Рисует текст бейджа без подложки и без мягкого «ореола».

    Глифы рендерятся в L-маску (2×), даунскейлятся, жёстко отсекаются
    по порогу альфы и впечатываются сплошным цветом — градиент бейджа
    под текстом не затирается прямоугольником и не мутнеет.
    """
    if not text:
        return
    scale = 2
    mw, mh = badge.width * scale, badge.height * scale
    mask_hi = Image.new("L", (mw, mh), 0)
    d = ImageDraw.Draw(mask_hi)
    font_size = size * scale
    font = _font(font_size)
    while font_size > 10 * scale:
        bbox = d.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w * scale:
            break
        font_size -= 2 * scale
        font = _font(font_size)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = cx * scale - tw / 2 - bbox[0]
    y = cy * scale - th / 2 - bbox[1]
    d.text((x, y), text, font=font, fill=255)
    mask = mask_hi.resize(badge.size, Image.LANCZOS).point(lambda a: 255 if a >= 160 else 0)
    colored = Image.new("RGBA", badge.size, color)
    badge.paste(colored, (0, 0), mask)


# ---------------------------------------------------------------------------
# Заглушка картинки товара
# ---------------------------------------------------------------------------

def resolve_item_image(item: CardItem, icons_dir: Path) -> Image.Image:
    if item.image:
        p = Path(item.image)
        if not p.is_absolute():
            candidate = icons_dir / item.image
            p = candidate if candidate.exists() else p
        if p.exists():
            return Image.open(p).convert("RGBA")
        log.warning("Картинка '%s' для '%s' не найдена, подставляю заглушку из Icons.", item.image, item.name)

    placeholder = icons_dir / "CardC6_card_01.png"
    if not placeholder.exists():
        raise FileNotFoundError(f"Нет ни картинки товара, ни заглушки в {icons_dir}")
    return Image.open(placeholder).convert("RGBA")


def _fit_stretch(img: Image.Image, w: int, h: int) -> Image.Image:
    """Растягивает картинку под размер бокса без сохранения пропорций."""
    return img.resize((max(w, 1), max(h, 1)), Image.LANCZOS)


def _fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Масштабирует с сохранением пропорций так, чтобы полностью покрыть
    бокс, лишнее обрезает по центру."""
    w, h = max(w, 1), max(h, 1)
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    fitted = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return fitted.crop((left, top, left + w, top + h))


def _slot_mask(border: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Маска дырки рамки (арт-слот) и её bbox.

    Flood-fill от центра по alpha < IMAGE_HOLE_ALPHA — так не захватываем
    внешнюю прозрачную подложку PNG. Затем слегка расширяем маску
    (MaxFilter), чтобы арт зашёл под полупрозрачную antialias-кромку
    рамки и не оставлял щель сверху/по бокам.
    """
    w, h = border.size
    alpha = border.getchannel("A")
    ap = alpha.load()
    sx, sy = w // 2, h // 2
    # Если центр вдруг непрозрачен — ищем ближайшую дырку вниз/вверх.
    if ap[sx, sy] >= IMAGE_HOLE_ALPHA:
        for dy in range(h):
            for sign in (-1, 1):
                y = sy + sign * dy
                if 0 <= y < h and ap[sx, y] < IMAGE_HOLE_ALPHA:
                    sy = y
                    break
            else:
                continue
            break

    from collections import deque
    vis = bytearray(w * h)
    q = deque([(sx, sy)])
    vis[sy * w + sx] = 1
    mask = Image.new("L", (w, h), 0)
    mp = mask.load()
    minx, miny, maxx, maxy = w, h, -1, -1
    while q:
        x, y = q.popleft()
        if ap[x, y] >= IMAGE_HOLE_ALPHA:
            continue
        mp[x, y] = 255
        if x < minx: minx = x
        if y < miny: miny = y
        if x > maxx: maxx = x
        if y > maxy: maxy = y
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not vis[ny * w + nx]:
                vis[ny * w + nx] = 1
                if ap[nx, ny] < IMAGE_HOLE_ALPHA:
                    q.append((nx, ny))

    if maxx < 0:
        # Fallback на константное окно.
        cx = border.width / 2
        cy = border.height / 2
        bw, bh = IMAGE_WINDOW_LOCAL["w"], IMAGE_WINDOW_LOCAL["h"]
        x0 = round(cx + IMAGE_WINDOW_LOCAL["cx"] - bw / 2)
        y0 = round(cy + IMAGE_WINDOW_LOCAL["cy"] - bh / 2)
        bbox = (x0, y0, x0 + int(bw), y0 + int(bh))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle(bbox, fill=255)
        return mask, bbox

    if IMAGE_MASK_DILATE >= 3 and IMAGE_MASK_DILATE % 2 == 1:
        mask = mask.filter(ImageFilter.MaxFilter(IMAGE_MASK_DILATE))
        # После дилатации подтянем bbox.
        bbox2 = mask.getbbox()
        if bbox2:
            minx, miny, maxx, maxy = bbox2

    return mask, (minx, miny, maxx + 1, maxy + 1)


def _interior_mask(border: Image.Image) -> Image.Image:
    """Маска «внутри карточки»: всё, что не внешний прозрачный паддинг PNG.

    Flood от углов по низкой альфе = снаружи. Инверсия = дырка слота +
    сама рамка + неймплейт-зона. Арт кладём на всю эту маску, а рамка
    сверху сама скроет края — так не остаётся щели у верхней кромки.
    """
    w, h = border.size
    alpha = border.getchannel("A")
    ap = alpha.load()
    from collections import deque
    exterior = bytearray(w * h)
    q: deque[tuple[int, int]] = deque()
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ap[x, y] < IMAGE_HOLE_ALPHA:
            q.append((x, y))
            exterior[y * w + x] = 1
    # Также точки вдоль краёв с низкой альфой.
    for x in range(w):
        for y in (0, h - 1):
            if ap[x, y] < IMAGE_HOLE_ALPHA and not exterior[y * w + x]:
                q.append((x, y)); exterior[y * w + x] = 1
    for y in range(h):
        for x in (0, w - 1):
            if ap[x, y] < IMAGE_HOLE_ALPHA and not exterior[y * w + x]:
                q.append((x, y)); exterior[y * w + x] = 1
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not exterior[ny * w + nx]:
                if ap[nx, ny] < IMAGE_HOLE_ALPHA:
                    exterior[ny * w + nx] = 1
                    q.append((nx, ny))
    mask = Image.new("L", (w, h), 0)
    mp = mask.load()
    for y in range(h):
        row = y * w
        for x in range(w):
            if not exterior[row + x]:
                mp[x, y] = 255
    return mask


def _place_art_in_slot(border: Image.Image, art: Image.Image) -> Image.Image:
    """Накладывает арт на весь слот карточки и обрезает маской интерьера.

    1) COVER на полный размер Border.png — без полей сверху/снизу.
    2) Маска интерьера (не внешний паддинг) — арт не вылезает за карточку.
    3) Рамка потом ляжет сверху и скроет края по своему альфа-каналу.
    """
    fitted = _fit_cover(art.convert("RGBA"), border.width, border.height)
    interior = _interior_mask(border)
    # Дополнительно расширяем слот-дырку, чтобы под antialias-кромкой
    # точно был арт (на случай тонких зазоров у верхней дуги).
    hole, _ = _slot_mask(border)
    combined = ImageChops.lighter(interior, hole)
    r, g, b, a = fitted.split()
    a = ImageChops.multiply(a, combined)
    return Image.merge("RGBA", (r, g, b, a))


# ---------------------------------------------------------------------------
# Сборка одной карточки
# ---------------------------------------------------------------------------

def compose_card(item: CardItem, materials: dict[str, Image.Image], icons_dir: Path,
                  scale: float = 1.0) -> Image.Image:
    """Собирает одну карточку товара в её собственном масштабе (размер
    Border.png), слоями, как описано в разделе "Дальше на каждый бордер
    добавляется" правила:
        1. Неймплейт
        2. Название в неймплейт
        3. Картинка (или заглушка)
        4. Звёзды
        5. Бейдж (если нужен)
        6. Текст названия (см. п.2 — плашка и текст рисуются в одной точке)
        7. Показатель заточки на бейдже (если бейдж есть)
    """
    border = materials["border"]
    space = BorderSpace(border.width, border.height)
    # Канва карточки выше рамки на CARD_TOP_PAD: звёзды по дизайну торчат
    # над верхним краем рамки, и без этого запаса их верхушку срезало бы
    # по границе канвы. pad_xy сдвигает любую посчитанную через BorderSpace
    # позицию вниз на этот запас перед вставкой в канву.
    canvas = Image.new("RGBA", (border.width, border.height + CARD_TOP_PAD), (0, 0, 0, 0))

    def pad_xy(pos: tuple[int, int]) -> tuple[int, int]:
        return (pos[0], pos[1] + CARD_TOP_PAD)

    # 0. Картинка товара — COVER на весь слот + маска дырки рамки.
    art = resolve_item_image(item, icons_dir)
    art_layer = _place_art_in_slot(border, art)
    canvas.alpha_composite(art_layer, pad_xy((0, 0)))

    # 1. Неймплейт ПОД рамкой: серебряный кант Border.png закрывает
    #    зубчатую кромку ассета (иначе при зуме видны артефакты).
    #    Сначала срезаем прозрачный паддинг PNG, иначе верх плашки
    #    визуально «проседает» и по углам светится щель.
    plate = materials["nameplate"]
    opaque = plate.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox()
    if opaque:
        plate = plate.crop(opaque)
    target_w = max(plate.width, round(border.width * NAMEPLATE_WIDTH_FRAC))
    if target_w != plate.width:
        scale_p = target_w / plate.width
        plate = plate.resize(
            (target_w, max(1, round(plate.height * scale_p))),
            Image.LANCZOS,
        )
    plate_pos = space.to_canvas_topleft(0.0, NAMEPLATE_LOCAL_TOP + plate.height / 2, plate.width, plate.height)
    canvas.alpha_composite(plate, pad_xy(plate_pos))

    # 2. Рамка — поверх арта и неймплейта
    canvas.alpha_composite(border, pad_xy((0, 0)))

    # 3. Название в неймплейт (стиль как в Coolness: белый + тёмно-синяя обводка)
    _draw_fitted_text(
        canvas, item.name,
        cx=space.cx, cy=space.cy + NAMEPLATE_LOCAL_TOP + plate.height / 2 + CARD_TOP_PAD,
        max_w=plate.width * 0.86,
        size=NAMEPLATE_STYLE["size"],
        color=NAMEPLATE_STYLE["color"],
        stroke_color=NAMEPLATE_STYLE["stroke_color"],
        stroke_width=NAMEPLATE_STYLE["stroke_width"],
        shadow=NAMEPLATE_STYLE["shadow"],
    )

    # 4. Звёзды — пресет по количеству; центр пресета на середине верхнего
    #    канта рамки, по центру по x. Пресет цельный (не из одиночных Star.png).
    star_key = STAR_ASSET_BY_RARITY.get(item.rarity, STAR_ASSET_BY_RARITY[1])
    stars = materials[star_key]
    if STAR_PRESET_SCALE != 1.0:
        stars = stars.resize(
            (max(1, round(stars.width * STAR_PRESET_SCALE)),
             max(1, round(stars.height * STAR_PRESET_SCALE))),
            Image.LANCZOS,
        )
    stars_pos = (
        round(space.cx - stars.width / 2),
        round(BORDER_TOP_MIDLINE - stars.height / 2),
    )
    canvas.alpha_composite(stars, pad_xy(stars_pos))

    # 5-6. Бейдж: новый Badge.png без вшитого текста — рисуем "+N" прямо
    #    на градиенте. Рендер 2× → downsample: без прямоугольной подложки
    #    и с более чистой кромкой глифов на жёлтом градиенте.
    if item.badge:
        badge = materials["badge"].copy()
        cx, cy = BADGE_TEXT_CENTER
        if badge.size != (197, 145):
            cx = round(badge.width * (BADGE_TEXT_CENTER[0] / 197))
            cy = round(badge.height * (BADGE_TEXT_CENTER[1] / 145))
        _draw_badge_text(
            badge, item.badge, cx=cx, cy=cy,
            max_w=badge.width * BADGE_TEXT_MAX_W_FRAC,
            size=BADGE_TEXT_SIZE,
            color=BADGE_TEXT_COLOR,
        )
        badge_pos = (round(space.cx + BADGE_LOCAL_TOPLEFT[0]), round(space.cy + BADGE_LOCAL_TOPLEFT[1]))
        canvas.alpha_composite(badge, pad_xy(badge_pos))

    if scale != 1.0:
        new_w = max(1, round(canvas.width * scale))
        new_h = max(1, round(canvas.height * scale))
        canvas = canvas.resize((new_w, new_h), Image.LANCZOS)

    return canvas


# ---------------------------------------------------------------------------
# Сборка страницы коллекции целиком
# ---------------------------------------------------------------------------

def build_offer(config: OfferConfig, cutted_dir: Path, reference_path: Path, icons_dir: Path) -> Image.Image:
    reference = load_reference(reference_path)
    materials = load_materials(cutted_dir, config.show_progressbar)

    bg = materials["background"].copy()
    log.info("Проверка подложки против референса.")
    verify_background_matches_reference(bg, reference)

    log.info("Шаг 4. Обращение к разметке: поиск области карточек на подложке.")
    card_area = detect_card_area(bg)
    log.info("  Подложка %s разбита на: название y=[0:%d], область карточек %s",
             bg.size, card_area[1], card_area)

    border = materials["border"]
    canvas = bg  # подложка уже несёт в себе ленту, рамку и футер

    # Заголовок коллекции — стиль как в Packs/05_Collection_Coolness (жёлтый градиент + коричневая обводка).
    _draw_fitted_text(
        canvas, config.title,
        cx=bg.width / 2, cy=card_area[1] / 2,
        max_w=bg.width * 0.55,
        size=TITLE_STYLE["size"],
        color=TITLE_STYLE["color"],
        stroke_color=TITLE_STYLE["stroke_color"],
        stroke_width=TITLE_STYLE["stroke_width"],
        shadow=TITLE_STYLE["shadow"],
        gradient=TITLE_STYLE.get("gradient"),
    )

    # Прогресс-бар (если включён) занимает полосу в верхней части области
    # карточек — резервируем её ДО расчёта сетки, иначе бар налезет на
    # первый ряд рамок.
    grid_area = card_area
    if config.show_progressbar and "progressbar" in materials:
        # Progressbar.png несёт слабый (alpha ~10-60) полупрозрачный "хвост"
        # справа — след вырезки, не часть самой полосы. По нему getbbox()
        # (порог alpha>0) даёт бокс почти во всю ширину картинки, из-за чего
        # реальная полоса, занимающая только левые ~70%, при центровании по
        # полной картинке уезжала влево и казалась мельче поля. Обрезаем по
        # уверенно непрозрачным пикселям (alpha>128), чтобы центровать сам
        # бар, а не пустой хвост рядом с ним.
        # Порог подобран по плато: alpha>85..110 даёт один и тот же бокс
        # (15,50,1898,341) — ниже (61-80) в бокс ещё лезет полупрозрачный
        # хвост-артефакт, выше (120+) уже обрезает законную мягкую тень
        # снизу самой полосы, отчего бар казался приплюснутым по высоте.
        pb_full = materials["progressbar"]
        pb_bbox = Image.eval(pb_full.getchannel("A"), lambda a: 255 if a > 100 else 0).getbbox() or pb_full.getbbox()
        pb = pb_full.crop(pb_bbox)
        if PROGRESS_BAR_SCALE != 1.0:
            pb = pb.resize(
                (max(1, round(pb.width * PROGRESS_BAR_SCALE)),
                 max(1, round(pb.height * PROGRESS_BAR_SCALE))),
                Image.LANCZOS,
            )
        margin_top, margin_bottom = 8, 8
        pb_x = round((bg.width - pb.width) / 2)
        pb_y = card_area[1] + margin_top
        canvas.alpha_composite(pb, (pb_x, pb_y))
        reserved = margin_top + pb.height + margin_bottom
        grid_area = (card_area[0], card_area[1] + reserved, card_area[2], card_area[3])

    card_scale = fit_card_scale(grid_area, border.width, border.height)
    scaled_w = max(1, round(border.width * card_scale))
    scaled_h = max(1, round(border.height * card_scale))
    scaled_pad = max(1, round(CARD_TOP_PAD * card_scale))
    if card_scale < 0.999:
        log.info("  Карточки уменьшены до %.1f%% (%dx%d), чтобы сетка с вылетом звёзд влезла в область.",
                 card_scale * 100, scaled_w, scaled_h)

    positions = compute_grid(grid_area, scaled_w, scaled_h)
    verify_grid_fits(grid_area, positions, scaled_w, scaled_h)

    log.info("Шаг 5. Подстановка материалов в разметку: сборка 10 карточек.")
    if len(config.items) != CARDS_PER_PAGE:
        raise ValueError(f"Ожидается {CARDS_PER_PAGE} карточек, получено {len(config.items)}.")

    for idx, (item, (x, y)) in enumerate(zip(config.items, positions), start=1):
        log.info("  Карточка %2d/%d: %-24s rarity=%d badge=%s", idx, CARDS_PER_PAGE,
                  item.name, item.rarity, item.badge or "-")
        card = compose_card(item, materials, icons_dir, scale=card_scale)
        # compose_card возвращает канву с запасом CARD_TOP_PAD сверху (там,
        # где над рамкой торчат звёзды) — компенсируем его здесь, чтобы сама
        # рамка легла ровно в (x, y), как посчитано сеткой.
        paste_y = y - scaled_pad
        if paste_y < 0:
            log.warning("Карточка %d: не хватает %dpx сверху для вылета звёзд — обрежутся.", idx, -paste_y)
            paste_y = 0
        canvas.alpha_composite(card, (x, paste_y))

    log.info("Шаг 6. Проверка размещения — см. предупреждения выше (их отсутствие значит, что всё встало штатно).")
    # Футер SET N/M оставляем как впечатан в BackGround.png — не перерисовываем.

    if canvas.size != OUTPUT_SIZE:
        log.info("Приведение к финальному размеру: %s -> %s", canvas.size, OUTPUT_SIZE)
        canvas = canvas.resize(OUTPUT_SIZE, Image.LANCZOS)

    return canvas


def _redraw_footer_set(
    canvas: Image.Image,
    page: int,
    total: int,
    card_area: tuple[int, int, int, int],
) -> None:
    """Перекрывает впечатанный SET N/M и рисует актуальный номер страницы."""
    import numpy as np

    w, h = canvas.size
    y0 = min(h - 1, card_area[3] + 8)
    # Нижняя бежевая полоса футера — ниже синего поля карточек.
    band = canvas.crop((0, y0, w, h))
    arr = np.array(band)
    if arr.size == 0:
        return
    opaque = arr[:, :, 3] > 180
    if not opaque.any():
        return
    # Бежевый футер: высокий R/G, умеренный B.
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    beige = opaque & (r > 160) & (g > 130) & (b > 80) & (r > b) & ((r - b) > 40)
    if not beige.any():
        beige = opaque
    ys, xs = np.where(beige)
    fy0, fy1 = int(ys.min()) + y0, int(ys.max()) + y0 + 1
    fx0, fx1 = int(xs.min()), int(xs.max()) + 1
    sample = arr[beige]
    fill = tuple(int(v) for v in sample.mean(axis=0))
    # Широкий патч по центру — закрывает старый «SET …».
    patch_w = max(280, (fx1 - fx0) // 3)
    cx = (fx0 + fx1) // 2
    cy = (fy0 + fy1) // 2
    patch = Image.new("RGBA", (patch_w, max(40, fy1 - fy0)), fill)
    canvas.alpha_composite(patch, (cx - patch_w // 2, fy0))
    label = f"SET {page}/{total}"
    _draw_fitted_text(
        canvas, label,
        cx=cx, cy=cy,
        max_w=patch_w * 0.95,
        size=FOOTER_STYLE["size"],
        color=FOOTER_STYLE["color"],
        stroke_color=FOOTER_STYLE["stroke_color"],
        stroke_width=FOOTER_STYLE["stroke_width"],
        shadow=FOOTER_STYLE["shadow"],
        uppercase=True,
    )
    log.info("  Футер: %s", label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборщик офферов из материалов Cutted по правилам RuleForBuilding.")
    parser.add_argument("--config", required=True, type=Path, help="YAML-конфиг коллекции (см. example_config.yaml)")
    parser.add_argument("--out", type=Path, default=Path("result.png"), help="Куда сохранить результат")
    parser.add_argument("--cutted-dir", type=Path, default=DEFAULT_CUTTED_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--icons-dir", type=Path, default=DEFAULT_ICONS_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(message)s")

    config = load_config(args.config)
    result = build_offer(config, args.cutted_dir, args.reference, args.icons_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)
    log.info("Готово: %s (%dx%d)", args.out, result.width, result.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
