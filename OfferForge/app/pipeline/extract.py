"""Изъятие элемента интерфейса из скина — без перерисовки.

Разница с regen.py принципиальная. Там модель рисует элемент заново по
образцу, и результат неизбежно расходится с оригиналом: рамка становится
пустым прямоугольником, бейдж — жёлтым овалом. Здесь пиксели остаются
теми же, убирается только фон вокруг них.

Два пути к одному результату:

* `auto_matte` — локально, без сети. Заливка от краёв по локальной разнице
  соседей, а не по одному «ключевому» цвету: фон в игровом UI почти всегда
  градиентный, и глобальный порог либо не добирает тёмный край, либо
  выедает половину элемента.
* роль `extract` у image-edit модели — кроп уходит в модель с требованием
  положить его на хромакей, ничего не меняя. Дороже, зато справляется там,
  где фон и элемент одного тона.

После матирования вступает `split_atoms`: полоска из пяти звёзд — это пять
звёзд, а не один элемент. Композитор собирает полоску сам под нужное число,
поэтому в библиотеку кладётся одна.
"""
from __future__ import annotations

import logging
from collections import deque
from io import BytesIO
from typing import Any, Callable

from PIL import Image, ImageChops, ImageFilter

from app.config import resolve_variant
from app.models import Preset
from app.pipeline import matting
from app.prompts import engine
from app.providers.registry import Registry

log = logging.getLogger("offerforge.extract")

# Ниже этого элемент считается выеденным, выше — фон не нашёлся вовсе.
COVERAGE_MIN, COVERAGE_MAX = 0.01, 0.97


def _d(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def edge_colors(img: Image.Image) -> list[tuple[int, int, int]]:
    """Цвета периметра, сгруппированные по похожести.

    Периметр кропа — это фон плюс то, что случайно задело выделение: край
    соседней рамки, кусок подложки. Поэтому берём не один цвет, а несколько
    кластеров, и заливка стартует от каждого.
    """
    px = img.convert("RGBA").load()
    w, h = img.size
    # Прозрачные пиксели в выборку не берём. После лассо весь периметр
    # прозрачный, и без этого условия «фоном» назначался чёрный из-под
    # нулевой альфы, а заливка потом выедала сам элемент.
    coords = [(x, y) for x in range(w) for y in (0, h - 1)]
    coords += [(x, y) for y in range(h) for x in (0, w - 1)]
    samples = [px[x, y][:3] for x, y in coords if px[x, y][3] > 24]

    clusters: list[list] = []
    for c in samples:
        for cl in clusters:
            if _d(c, cl[0]) < 60:
                cl[1] += 1
                break
        else:
            clusters.append([c, 1])
    clusters.sort(key=lambda c: -c[1])
    if not clusters:
        return []

    # Порог относительный, а не абсолютный. Элемент часто касается края
    # выделения — звезда упирается в границу, и её жёлтый попадает в
    # периметр. С абсолютным порогом такой цвет проходит как фон, заливка
    # стартует прямо со звезды и съедает её. Относительный оставляет только
    # то, чего на периметре сопоставимо много с главным фоном.
    top = clusters[0][1]
    return [c[0] for c in clusters[:3] if c[1] >= top * 0.55]


def auto_matte(
    img: Image.Image,
    *,
    local: int = 42,
    drift: int = 190,
    feather: float = 0.8,
) -> Image.Image:
    """Убирает фон заливкой от краёв по локальной разнице соседей.

    local — насколько сосед может отличаться от текущего пикселя, чтобы
    заливка через него прошла. Малое значение держит заливку внутри
    плавного градиента и не даёт ей перепрыгнуть на элемент через резкую
    границу.

    drift — предохранитель: суммарный уход от стартового цвета. Без него
    цепочка мелких шагов проползает через весь градиент прямо в элемент,
    если тот меняет тон постепенно.
    """
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()

    starts = edge_colors(img)
    if not starts:
        return img          # весь периметр прозрачен — резать уже нечего

    mask = Image.new("L", (w, h), 255)      # 255 — элемент, 0 — фон
    mp = mask.load()
    seen = bytearray(w * h)
    # В очереди: координаты, цвет предыдущего пикселя и цвет, с которого
    # эта ветка заливки началась.
    queue: deque[tuple[int, int, tuple, tuple]] = deque()

    def seed(x: int, y: int) -> None:
        r, g, b, a = px[x, y]
        if a <= 24:
            # Уже прозрачно — вырезано лассо. Отсюда заливка идёт дальше,
            # но цвет за ориентир не берётся: под нулевой альфой мусор.
            mp[x, y] = 0
            seen[y * w + x] = 1
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and px[nx, ny][3] > 24:
                    c = px[nx, ny][:3]
                    if any(_d(c, s) <= 70 for s in starts):
                        queue.append((nx, ny, c, c))
            return
        # Стартуем только с тех краёв, что похожи на фон. Иначе ветка,
        # начавшаяся на элементе, поползёт вглубь него.
        c = (r, g, b)
        if any(_d(c, s) <= 70 for s in starts):
            queue.append((x, y, c, c))

    for x in range(w):
        seed(x, 0)
        seed(x, h - 1)
    for y in range(h):
        seed(0, y)
        seed(w - 1, y)

    while queue:
        x, y, prev, origin = queue.popleft()
        idx = y * w + x
        if seen[idx]:
            continue
        cur = px[x, y][:3]
        # Шаг допустим, если сосед близок к предыдущему пикселю (плавный
        # градиент) и вся ветка не ушла слишком далеко от своего старта.
        # Без второго условия цепочка мелких шагов проползает через
        # градиент прямо в элемент.
        if _d(cur, prev) > local or _d(cur, origin) > drift:
            continue
        seen[idx] = 1
        mp[x, y] = 0
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx]:
                queue.append((nx, ny, cur, origin))

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(Image.composite(out.getchannel("A"),
                                 Image.new("L", (w, h), 0), mask))
    return out


def _labels(alpha, w: int, h: int, min_area: int) -> list[tuple[int, int, int, int, int]]:
    """Связные области непрозрачных пикселей: (площадь, x0, y0, x1, y1)."""
    seen = bytearray(w * h)
    found: list[tuple[int, int, int, int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy * w + sx] or alpha[sx, sy] < 24:
                continue
            x0 = x1 = sx
            y0 = y1 = sy
            area = 0
            queue = deque([(sx, sy)])
            seen[sy * w + sx] = 1
            while queue:
                x, y = queue.popleft()
                area += 1
                x0, x1 = min(x0, x), max(x1, x)
                y0, y1 = min(y0, y), max(y1, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    ni = ny * w + nx
                    if seen[ni] or alpha[nx, ny] < 24:
                        continue
                    seen[ni] = 1
                    queue.append((nx, ny))
            if area >= min_area:
                found.append((area, x0, y0, x1, y1))
    return found


def separate(img: Image.Image, erode: int = 3) -> Image.Image:
    """Разрывает тонкие перемычки между слипшимися элементами.

    Звезда, стоящая вплотную к канту рамки, образует с ним одну связную
    область, и разбор на компоненты выдаёт их одним куском. Эрозия альфы
    съедает перемычку — она тоньше самих элементов, — после чего они
    расходятся. Форму возвращает не дилатация, а пересечение с исходной
    альфой: так края остаются точно теми же, что были.
    """
    a = img.getchannel("A")
    thin = a.filter(ImageFilter.MinFilter(erode))
    out = img.copy()
    out.putalpha(thin)
    return out


def wand(
    img: Image.Image,
    seeds: list[tuple[float, float]],
    *,
    box: tuple[int, int, int, int] | None = None,
    local: int = 60,
    drift: int = 150,
    feather: float = 0.8,
) -> Image.Image | None:
    """Волшебная палочка: растит область от точек внутри элемента.

    Прямоугольный бокс не описывает форму, а обводить руками пользователь
    не должен. Но модель уверенно показывает ТОЧКУ внутри элемента — это
    вопрос «где звезда», а не «где именно проходит её граница». Границу
    дальше находит заливка: от точки наружу, пока цвет похож.

    Так получается автоматическое лассо. Оно точнее бокса, потому что
    идёт по реальному контуру, и надёжнее вопроса «дай координаты
    границы», потому что от модели требуется только попасть внутрь.

    seeds — точки в долях изображения. box — ограничитель в пикселях:
    заливка не выйдет за него, даже если цвет продолжается дальше.
    """
    img = img.convert("RGBA")
    w, h = img.size
    px = img.load()
    x0, y0, x1, y1 = box or (0, 0, w, h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    mask = Image.new("L", (w, h), 0)
    mp = mask.load()
    seen = bytearray(w * h)
    queue: deque[tuple[int, int, tuple, tuple]] = deque()

    for sx, sy in seeds:
        ix, iy = int(sx * w), int(sy * h)
        if not (x0 <= ix < x1 and y0 <= iy < y1):
            continue
        if px[ix, iy][3] < 24:
            continue
        c = px[ix, iy][:3]
        queue.append((ix, iy, c, c))

    if not queue:
        return None

    while queue:
        x, y, prev, origin = queue.popleft()
        idx = y * w + x
        if seen[idx]:
            continue
        r, g, b, a = px[x, y]
        if a < 24:
            continue
        cur = (r, g, b)
        # Шаг допустим, если сосед близок к предыдущему пикселю и вся
        # ветка не ушла далеко от точки старта. Первое ведёт заливку по
        # градиенту внутри элемента, второе не даёт ей перетечь на фон
        # через плавную границу.
        if _d(cur, prev) > local or _d(cur, origin) > drift:
            continue
        seen[idx] = 1
        mp[x, y] = 255
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if x0 <= nx < x1 and y0 <= ny < y1 and not seen[ny * w + nx]:
                queue.append((nx, ny, cur, origin))

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out


def fill_holes(alpha: Image.Image) -> Image.Image:
    """Заращивает дыры внутри маски, не трогая внешнюю границу.

    Заливка не заходит на контрастные детали внутри элемента: коричневая
    цифра на жёлтом бейдже, белая надпись на плашке остаются проколами.
    Морфологическое закрытие заодно скругляет и внешний контур, а нам
    нужно ровно наоборот.

    Поэтому ищем дыры честно: заливаем прозрачное ОТ КРАЁВ. Что не
    достигнуто — заперто внутри элемента, значит дыра, и её закрываем.
    """
    w, h = alpha.size
    px = alpha.load()
    seen = bytearray(w * h)
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if px[x, y] < 128:
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if px[x, y] < 128:
                queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        idx = y * w + x
        if seen[idx] or px[x, y] >= 128:
            continue
        seen[idx] = 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx]:
                queue.append((nx, ny))

    out = alpha.copy()
    op = out.load()
    for y in range(h):
        row = y * w
        for x in range(w):
            if px[x, y] < 128 and not seen[row + x]:
                op[x, y] = 255
    return out


def wand_cut(
    img: Image.Image,
    seeds: list[tuple[float, float]],
    box: tuple[int, int, int, int],
    *,
    fill_inner: bool = True,
) -> Image.Image | None:
    """Автолассо: заливка от точки, заращивание дыр, обрезка полей.

    Пробует несколько стартовых точек. Одна точка ненадёжна: у крестика
    центр бокса приходится на просвет между лучами, заливка уходит на
    подложку и возвращает залитый прямоугольник. Поэтому берётся набор
    точек, и выигрывает та, что дала правдоподобную область — не пустую
    и не во весь бокс.
    """
    # Работаем на кропе бокса, а не на всём полотне: скин бывает 5482×2529,
    # и заливка по нему для каждой точки — это минуты вместо миллисекунд.
    x0, y0, x1, y1 = box
    crop = img.convert("RGBA").crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if cw < 4 or ch < 4:
        return None

    # Крупные элементы уменьшаем: точность границы от этого почти не
    # страдает, а заливка ускоряется на порядок.
    scale = 1.0
    if max(cw, ch) > 640:
        scale = 640 / max(cw, ch)
        work = crop.resize((max(2, round(cw * scale)), max(2, round(ch * scale))),
                           Image.LANCZOS)
    else:
        work = crop

    local: list[tuple[float, float]] = []
    for sx, sy in seeds:
        lx = (sx * img.width - x0) / max(1, cw)
        ly = (sy * img.height - y0) / max(1, ch)
        if 0.0 <= lx <= 1.0 and 0.0 <= ly <= 1.0:
            local.append((lx, ly))
    if not local:
        return None

    def is_backdrop(a: Image.Image) -> bool:
        """Залился фон вокруг элемента, а не сам элемент.

        Точный признак — углы. Бокс облегает элемент, поэтому в углах у
        звезды, крестика и любой непрямоугольной формы остаётся фон. Если
        маска забрала все четыре угла, значит заливка пошла снаружи, и
        заращивание дыр потом закроет ею сам элемент.
        """
        w2, h2 = a.size
        m = 2
        corners = [a.getpixel(p) for p in
                   ((m, m), (w2 - 1 - m, m), (m, h2 - 1 - m), (w2 - 1 - m, h2 - 1 - m))]
        return sum(1 for c in corners if c > 128) >= 4

    best, best_cov = None, 0.0
    for seed in local:
        picked = wand(work, [seed])
        if picked is None:
            continue
        alpha = picked.getchannel("A").point(lambda v: 255 if v > 100 else 0)
        if is_backdrop(alpha):
            continue
        if fill_inner:
            alpha = fill_holes(alpha)
        cov = sum(alpha.histogram()[128:]) / max(1, alpha.width * alpha.height)
        # Меньше 2% — заливка сорвалась на детали. Больше 97% — залился
        # весь бокс, значит точка попала на фон, а не на элемент.
        if 0.02 < cov < 0.97 and cov > best_cov:
            best, best_cov = alpha, cov

    if best is None:
        return None

    if scale != 1.0:
        best = best.resize((cw, ch), Image.LANCZOS)
    best = best.filter(ImageFilter.GaussianBlur(0.8))

    out = crop.copy()
    out.putalpha(ImageChops.multiply(out.getchannel("A"), best))
    return matting.trim(out)


def probe_seeds(box: dict, spread: float = 0.22) -> list[tuple[float, float]]:
    """Точки-кандидаты внутри бокса: центр и смещения от него.

    Модель указывает, где элемент, с точностью «примерно тут». Центр
    бокса чаще всего попадает в тело, но у элементов с просветом —
    крестика, рамки — промахивается. Несколько точек закрывают этот случай
    без вопросов к модели.
    """
    cx = float(box.get("x", 0)) + float(box.get("w", 0)) / 2
    cy = float(box.get("y", 0)) + float(box.get("h", 0)) / 2
    dx, dy = float(box.get("w", 0)) * spread, float(box.get("h", 0)) * spread
    return [
        (cx, cy),
        (cx - dx, cy), (cx + dx, cy), (cx, cy - dy), (cx, cy + dy),
        (cx - dx, cy - dy), (cx + dx, cy + dy),
    ]


def split_atoms(
    img: Image.Image,
    *,
    min_area_frac: float = 0.004,
    pad: int = 2,
) -> list[Image.Image]:
    """Разбивает картинку на связные непрозрачные куски.

    Выделение «звёзды» — это пять звёзд рядом. В библиотеку нужна одна:
    полоску под нужное число композитор соберёт сам. Заодно отсекается
    мусор, попавший в выделение: обрывок соседней рамки — отдельный
    компонент, и по площади он отваливается.
    """
    img = img.convert("RGBA")
    w, h = img.size
    min_area = max(16, int(w * h * min_area_frac))

    boxes = _labels(img.getchannel("A").load(), w, h, min_area)
    # Если всё слиплось в одну кляксу — рвём перемычки и пробуем снова.
    # Границы берём по утончённой маске, а пиксели — из оригинала, поэтому
    # эрозия влияет только на разделение, не на качество краёв. Радиус
    # наращиваем: перемычка бывает и в пару пикселей, и в десяток.
    if len(boxes) < 2:
        for radius in (3, 5, 9, 15):
            retry = _labels(separate(img, radius).getchannel("A").load(),
                            w, h, min_area)
            if len(retry) > 1:
                boxes = retry
                break

    boxes.sort(key=lambda b: -b[0])
    out: list[Image.Image] = []
    for _, x0, y0, x1, y1 in boxes:
        piece = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                          min(w, x1 + 1 + pad), min(h, y1 + 1 + pad)))
        # Запоминаем, упирался ли кусок в границу выделения и где был его
        # центр: по этому потом отличают целевой элемент от обрывка соседа.
        piece.info["touches_edge"] = (x0 <= 1 or y0 <= 1
                                      or x1 >= w - 2 or y1 >= h - 2)
        piece.info["center"] = ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)
        out.append(piece)
    return out


def pick_atom(atoms: list[Image.Image], _depth: int = 0) -> Image.Image | None:
    """Выбирает представителя группы.

    Внутри выделения обычно лежит не только целевой элемент: рядом край
    соседней рамки, угол бейджа, полоска подложки. Отличаем по трём
    признакам — обрывок соседа упирается в границу выделения, он вытянут,
    и его центр смещён от центра кадра. Целевой элемент компактный,
    посередине и не касается краёв.
    """
    best, best_score = None, -1.0
    for a in atoms:
        if a.width < 6 or a.height < 6:
            continue
        fill = matting.coverage(a)
        squareness = min(a.width, a.height) / max(a.width, a.height)
        cx, cy = a.info.get("center", (0.5, 0.5))
        centred = 1.0 - min(1.0, (abs(cx - 0.5) + abs(cy - 0.5)))
        score = (fill * (0.35 + 0.65 * squareness) * (0.5 + 0.5 * centred)
                 * (a.width * a.height) ** 0.25)
        if a.info.get("touches_edge"):
            score *= 0.45
        if score > best_score:
            best, best_score = a, score

    # Победитель может сам оказаться сросшимся: звезда, стоящая вплотную к
    # канту рамки, приходит одним куском. Признак — сильная вытянутость, и
    # тогда стоит попробовать разделить его ещё раз. Глубину ограничиваем:
    # если элемент действительно вытянут по своей природе (лента, полоса
    # прогресса), дробить его дальше нечего и не нужно.
    if best is not None and _depth < 2 and max(best.size) > 2.2 * min(best.size):
        inner = split_atoms(best, min_area_frac=0.02)
        if len(inner) > 1:
            deeper = pick_atom(inner, _depth + 1)
            if deeper is not None and max(deeper.size) < max(best.size):
                return deeper
    return best


def extract_local(
    img: Image.Image,
    *,
    atomize: bool = False,
    lassoed: bool = False,
) -> tuple[Image.Image, list[Image.Image]]:
    """Локальное извлечение: матирование, затем при нужде — на атомы.

    lassoed — границу уже задал человек контуром. Тогда матирование
    пропускается: заливка пошла бы от края силуэта и съела сам элемент,
    ведь снаружи всё прозрачно, а «фоном» назначился бы цвет самого края.
    """
    img = img.convert("RGBA")
    if lassoed:
        matted = img
    else:
        matted = auto_matte(img)
        cov = matting.coverage(matted)
        if not (COVERAGE_MIN < cov < COVERAGE_MAX):
            log.warning("Матирование подозрительное: покрытие %.2f — "
                        "оставляю кроп как есть", cov)
            matted = img

    atoms = split_atoms(matted)
    if atomize:
        one = pick_atom(atoms)
        if one is not None:
            return matting.trim(one), atoms
    return matting.trim(matted), atoms


async def extract_with_model(
    reg: Registry,
    preset: Preset,
    *,
    title: str,
    sample: bytes,
    screen: bytes | None = None,
    region: dict | None = None,
    atomize: bool = False,
    hollow: bool = False,
    hint: str = "",
) -> Image.Image:
    """Просит image-edit модель изъять элемент, ничего не перерисовывая.

    Формулировка тут решает всё. «Нарисуй такой же элемент» даёт похожий,
    но другой объект — это и ломало предыдущий подход. «Возьми ровно эти
    пиксели и убери фон» модель понимает как задачу редактирования, и
    результат совпадает с оригиналом.
    """
    variant = resolve_variant(preset, "extract")
    prompt = engine.render(
        "part_extract",
        title=title,
        chroma=matting.CHROMA_NAME,
        region=region,
        has_screen=screen is not None,
        atomize=atomize,
        hollow=hollow,
        extra=hint,
    )

    refs = [r for r in ([screen, sample] if screen else [sample]) if r]
    img = Image.open(BytesIO(sample))
    # Размер просим тот же, что у образца: пропорции элемента должны выжить.
    resp = await reg.image(variant, prompt, width=img.width, height=img.height,
                           references=refs)
    if not resp.images:
        raise RuntimeError(f"{title}: модель не вернула изображение")

    out = Image.open(BytesIO(resp.images[0])).convert("RGBA")
    # Если модель отдала альфу сама (background: transparent поддержан),
    # хромакея нет и срезать нечего — вмешательство только испортит края.
    if matting.coverage(out) > 0.98:
        out = matting.remove_background(out, key=matting.detect_key(out))

    cov = matting.coverage(out)
    if not (COVERAGE_MIN < cov < COVERAGE_MAX):
        raise RuntimeError(
            f"{title}: после вырезания фона осталось {cov:.0%} — "
            "модель вернула сплошное полотно либо пустоту")

    if atomize:
        one = pick_atom(split_atoms(out))
        if one is not None:
            out = one
    out = matting.trim(out)

    # Последний барьер: похоже ли вернувшееся на то, что просили. Модель
    # умеет вернуть красивую, но постороннюю картинку — зелёную звезду
    # вместо жёлтой, тёмный прямоугольник вместо рамки. Такое лучше не
    # класть в библиотеку вовсе: вызывающий откатится к вырезу из скрина.
    reason = looks_wrong(out, Image.open(BytesIO(sample)).convert("RGBA"))
    if reason:
        raise RuntimeError(f"{title}: {reason}")
    return out


def dominant_hues(img: Image.Image, top: int = 3) -> list[int]:
    """Основные оттенки непрозрачной части, в градусах круга."""
    import colorsys

    small = img.convert("RGBA")
    small.thumbnail((64, 64), Image.NEAREST)
    px = small.load()
    buckets: dict[int, int] = {}
    for y in range(small.height):
        for x in range(small.width):
            r, g, b, a = px[x, y]
            if a < 128:
                continue
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s < 0.15 or v < 0.15:
                continue        # серое и чёрное оттенка не имеют
            key = int(h * 360) // 30 * 30
            buckets[key] = buckets.get(key, 0) + 1
    return [k for k, _ in sorted(buckets.items(), key=lambda kv: -kv[1])[:top]]


def looks_wrong(made: Image.Image, sample: Image.Image) -> str:
    """Проверяет, что модель вернула тот же элемент, а не свой вариант.

    Сравниваем не пиксели — размер и кадрирование законно меняются, — а
    цвет и пропорции. Если из жёлтой звезды получилась зелёная, это уже
    не извлечение, а рисование заново.
    """
    made_h, sample_h = dominant_hues(made), dominant_hues(sample)
    if made_h and sample_h:
        near = any(min(abs(a - b), 360 - abs(a - b)) <= 45
                   for a in made_h for b in sample_h)
        if not near:
            return (f"цвет не совпал с образцом "
                    f"(вернулось {made_h}, ожидалось {sample_h}) — "
                    "похоже, элемент нарисован заново")

    if made.width > 4 and made.height > 4:
        ar_made = made.width / made.height
        ar_src = max(1, sample.width) / max(1, sample.height)
        # Разброс в три раза — это уже другой объект, а не тот же в другом
        # кадрировании.
        if ar_made / ar_src > 3 or ar_src / ar_made > 3:
            return (f"пропорции разъехались ({ar_made:.2f} против "
                    f"{ar_src:.2f}) — вернулся не тот элемент")
    return ""


# Элементы, которые в скине лежат группой: в библиотеку нужен один экземпляр.
GROUPED = {"stars", "button"}

# Элементы-кольца: внутри них лежат другие элементы, и модель по умолчанию
# вырезает их целиком вместе с начинкой. Рамка карточки приходит с плашкой
# имени, звездой и бейджем — потому что визуально это выглядит как один
# объект. Требование пустой середины ставится отдельным абзацем.
HOLLOW = {"frame", "progress"}


async def extract_pieces(
    reg: Registry,
    preset: Preset,
    pieces: dict[str, dict[str, bytes]],
    *,
    mode: str = "local",
    screen: bytes | None = None,
    boxes: dict[str, dict] | None = None,
    lassoed: set[str] | None = None,
    hints: dict[str, str] | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> dict[str, dict[str, bytes]]:
    """Прогоняет нарезанные куски через извлечение.

    mode: "raw" — как вырезано, "local" — матирование без сети,
    "model" — image-edit модель, с откатом на локальное при отказе.

    lassoed — типы, границу которых уже задали контуром: локальное
    матирование к ним не применяется.
    """
    if mode == "raw":
        return pieces

    from app import parts as parts_lib

    out: dict[str, dict[str, bytes]] = {}
    for kind, assets in pieces.items():
        out[kind] = {}
        for name, blob in assets.items():
            src = Image.open(BytesIO(blob)).convert("RGBA")
            atomize = kind in GROUPED
            meta = parts_lib.PART_KINDS.get(kind, {})
            # В промпт уходит английское имя. С русским «Название пака»
            # модель рисовала ровно эту надпись — она видит незнакомую
            # строку в англоязычном тексте и считает её тем, что надо
            # изобразить.
            title = meta.get("en") or kind
            result: Image.Image | None = None

            if mode == "model":
                try:
                    result = await extract_with_model(
                        reg, preset, title=title, sample=blob, screen=screen,
                        region=(boxes or {}).get(kind), atomize=atomize,
                        hollow=kind in HOLLOW, hint=(hints or {}).get(kind, ""))
                except Exception as e:  # noqa: BLE001
                    log.warning("Модель не изъяла %s: %s", kind, e)
                    if on_event:
                        on_event("warn", {"kind": kind,
                                          "text": f"{title}: {e} — беру локальный вырез"})

            if result is None:
                result, _ = extract_local(
                    src, atomize=atomize, lassoed=kind in (lassoed or set()))

            buf = BytesIO()
            result.save(buf, format="PNG")
            out[kind][name] = buf.getvalue()
            if on_event:
                on_event("extract", {"kind": kind, "asset": name,
                                     "w": result.width, "h": result.height,
                                     "mode": mode})
    return out
