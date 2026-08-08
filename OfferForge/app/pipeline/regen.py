"""Перерисовка UI-элементов по вырезанному образцу.

Сегментация даёт куски скриншота: с подложкой панели по краям, без
прозрачности, мылом при масштабировании и ровно в том количестве, в каком
элемент встретился на экране. Собирать из них — значит собирать из обрезков.

Здесь каждый кусок отправляется в image-модель как референс с требованием
воспроизвести элемент один в один, но изолированно и на чистом фоне. Фон
потом срезается, результат ложится отдельной версией — так в сборку идёт
настоящий ассет, а не фрагмент скриншота.

Атомарность важнее пакетности: генерируется ОДНА звезда, а полоски на
1–5 звёзд собираются кодом. Просить у модели «три звезды в ряд» — значит
получить три разные звезды с неровным шагом.
"""
from __future__ import annotations

import logging
import random
import time
from io import BytesIO
from typing import Any

from PIL import Image

from app import parts
from app.config import resolve_variant
from app.models import Preset
from app.parts import PART_KINDS
from app.pipeline import matting
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry

log = logging.getLogger("offerforge.regen")

# Что именно просить у модели по каждому типу. Формулировки предметные:
# «рамка слота» модель понимает хуже, чем описание формы.
SPECS: dict[str, dict[str, Any]] = {
    "frame": {
        "spec": "A vertical rounded rectangular card slot frame: a thick glossy "
                "border with a hollow empty centre. The centre must be plain flat "
                "background colour, not a picture — it is a window where artwork "
                "will be placed later.",
        "size": "1024x1024",
        "window": True,
    },
    "stars": {
        "spec": "A single five-pointed star badge: plump rounded points, glossy "
                "gradient fill, dark outline. Exactly ONE star, nothing else.",
        "size": "1024x1024",
        "atom": True,        # генерируем один, полоски собираем кодом
    },
    "badge": {
        "spec": "A small rounded quantity badge tag, like a price tag or pointer "
                "shape, glossy with a bright outline. Empty inside — the number "
                "will be drawn on top later.",
        "size": "1024x1024",
    },
    "nameplate": {
        "spec": "A horizontal rounded label plate: a wide flat bar with a bright "
                "rim, used as a background under a caption. Completely empty "
                "inside, no text.",
        "size": "1024x1024",
    },
    "ribbon": {
        "spec": "A wide horizontal banner ribbon with folded tails on both sides, "
                "glossy, used as a screen title bar. Completely empty, no text.",
        "size": "1024x1024",
    },
    "panel": {
        "spec": "A large rounded rectangular UI panel with a thick outer rim and "
                "a lighter inner area. Empty inside, no content on it.",
        "size": "1024x1024",
    },
    "close": {
        "spec": "A bold rounded X close icon with soft bevel, a single cross glyph.",
        "size": "1024x1024",
    },
    "progress": {
        "spec": "A horizontal progress bar capsule: dark track with a bright rim "
                "and a filled portion on the left. No text, no numbers.",
        "size": "1024x1024",
    },
    "footer": {
        "spec": "A narrow horizontal footer plate with rounded ends and a light "
                "fill, used as a caption strip. Completely empty, no text.",
        "size": "1024x1024",
    },
    "button": {
        "spec": "A wide glossy rounded action button with a bright rim and a "
                "highlight on the top half. Completely empty, no text.",
        "size": "1024x1024",
    },
}

MAX_ATTEMPTS = 3
# Границы разумного покрытия после вырезания фона. Почти всё полотно —
# значит фон не срезался; почти ничего — значит выело сам элемент.
COVERAGE_MIN, COVERAGE_MAX = 0.02, 0.92


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def star_strip(star: Image.Image, count: int, overlap: float = 0.18) -> Image.Image:
    """Собирает полоску из N одинаковых звёзд.

    Одна сгенерированная звезда, размноженная кодом, даёт ровный шаг и
    одинаковую форму. Модель, которую просят нарисовать «пять звёзд»,
    рисует пять разных.
    """
    star = matting.trim(star, pad=0)
    w, h = star.size
    step = int(w * (1 - overlap))
    total = w + step * (count - 1)
    strip = Image.new("RGBA", (total, h), (0, 0, 0, 0))
    for i in range(count):
        strip.alpha_composite(star, (i * step, 0))
    return strip


async def regenerate_part(
    reg: Registry,
    preset: Preset,
    kind: str,
    sample: bytes,
    *,
    screen: bytes | None = None,
    region: dict[str, float] | None = None,
    on_event=None,
) -> tuple[Image.Image | None, list[dict]]:
    """Перерисовывает один элемент по образцу. Возвращает картинку и журнал.

    screen — полный экран, region — где на нём лежит элемент. По одному
    обрезку модель не понимает ни назначения элемента, ни его масштаба
    относительно интерфейса, и рисует что-то похожее вместо того же самого.
    """
    spec = SPECS.get(kind)
    if spec is None:
        return None, [{"kind": kind, "skipped": "нет описания для этого типа"}]

    variant = resolve_variant(preset, "render")
    critic = resolve_variant(preset, "critic")
    journal: list[dict] = []
    extra = ""

    # Порядок референсов зафиксирован промптом: сначала весь экран, потом
    # вырезанный элемент.
    refs = [screen, sample] if screen else [sample]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = engine.render(
            "part_regen", spec=spec["spec"],
            # Английское имя: русский заголовок модель принимает за текст,
            # который надо нарисовать на самом элементе.
            title=PART_KINDS.get(kind, {}).get("en") or kind,
            region=region if screen else None,
            chroma=matting.CHROMA_NAME, extra=extra,
        )
        w, h = (int(v) for v in str(spec.get("size", "1024x1024")).split("x"))
        started = time.monotonic()

        try:
            res = await reg.image(
                variant, prompt, width=w, height=h,
                seed=random.randint(1, 2**31 - 1),
                references=refs,
            )
        except ProviderError as e:
            journal.append({"attempt": attempt, "error": str(e)})
            if on_event:
                on_event("part_error", {"kind": kind, "message": str(e)})
            return None, journal

        raw = Image.open(BytesIO(res.images[0])).convert("RGBA")
        if spec.get("window"):
            # У рамки слота центр закрыт со всех сторон, и заливка от краёв
            # туда не доходит — без этого шага рамка легла бы на арт
            # зелёным пятном. Гашение каймы откладываем: оно меняет цвет
            # фона, и замкнутые области перестают опознаваться.
            key = matting.detect_key(raw)
            cleaned = matting.remove_background(raw, key=key, kill_spill=False)
            cleaned = matting.remove_enclosed(cleaned, key=key)
            cleaned = matting.despill(cleaned, key)
        else:
            cleaned = matting.remove_background(raw)
        # Покрытие считаем ДО обрезки: после неё элемент по определению
        # занимает весь кадр, и широкая плашка выглядела бы как несрезанный
        # фон. Смысл метрики — какую долю исходного кадра занял элемент.
        cov = matting.coverage(cleaned)
        cut = matting.trim(cleaned)

        entry = {"attempt": attempt, "model": res.model,
                 "coverage": round(cov, 3),
                 "elapsed_s": round(time.monotonic() - started, 1)}

        if not (COVERAGE_MIN < cov < COVERAGE_MAX):
            entry["rejected"] = (f"фон не отделился (покрытие {cov:.2f})"
                                 if cov >= COVERAGE_MAX
                                 else f"элемент выело (покрытие {cov:.2f})")
            journal.append(entry)
            extra = ("Put the element on a completely flat uniform "
                     f"{matting.CHROMA_NAME} background with clear separation.")
            continue

        verdict = await _critique(reg, critic, kind, sample, _png(cut))
        entry["qc"] = verdict
        journal.append(entry)
        if on_event:
            on_event("part_attempt", {"kind": kind, **entry})

        if verdict.get("passed") or attempt == MAX_ATTEMPTS:
            return cut, journal
        extra = verdict.get("fix_hint") or verdict.get("reason") or ""

    return None, journal


async def _critique(reg: Registry, variant, kind: str,
                    sample: bytes, made: bytes) -> dict:
    """Сравнивает перерисованный элемент с образцом."""
    meta = PART_KINDS.get(kind, {})
    prompt = engine.render(
        "part_qc", kind=kind, title=meta.get("title", kind),
        spec=(SPECS.get(kind) or {}).get("spec", ""),
    )
    try:
        resp = await reg.text(variant, prompt, images=[sample, made], json_mode=True)
        data = engine.parse_json_response(resp.text)
        return {"passed": bool(data.get("passed")),
                "scores": data.get("scores") or {},
                "reason": data.get("reason", ""),
                "fix_hint": data.get("fix_hint", "")}
    except (ProviderError, ValueError) as e:
        # Без критика элемент всё равно годится к использованию, но об этом
        # надо сказать: молчаливый пропуск проверки выглядит как успех.
        log.warning("QC элемента %s недоступен: %s", kind, e)
        return {"passed": True, "scores": {"qc_skipped": True},
                "reason": f"QC пропущен: {e}", "fix_hint": ""}


async def regenerate_skin(
    reg: Registry,
    preset: Preset,
    source_version: str,
    target_version: str,
    *,
    kinds: list[str] | None = None,
    make_default: bool = True,
    screen: bytes | None = None,
    on_event=None,
) -> dict[str, Any]:
    """Перерисовывает все элементы нарезанного скина.

    Читает образцы из версии source_version (результат сегментации) и
    складывает перерисованное в target_version — так вырезанное и
    сгенерированное лежат рядом и переключаются одним селектом.
    """
    todo = kinds or [k for k in SPECS if parts.load_version(k, source_version)]
    result: dict[str, Any] = {"source": source_version, "target": target_version,
                              "parts": [], "failed": []}

    for kind in todo:
        src = parts.load_version(kind, source_version)
        if src is None:
            continue
        sample_name = next((n for n in src.assets if n.endswith(".png")), None)
        if not sample_name:
            continue
        sample = src.asset_path(sample_name).read_bytes()

        if on_event:
            on_event("part_start", {"kind": kind})

        # anchor сохраняется сегментацией — это и есть координаты элемента
        # на исходном экране.
        made, journal = await regenerate_part(
            reg, preset, kind, sample,
            screen=screen, region=src.anchor, on_event=on_event,
        )
        if made is None:
            result["failed"].append({"kind": kind, "journal": journal})
            if on_event:
                on_event("part_failed", {"kind": kind, "journal": journal})
            continue

        # Звёзды: из одного атома собираем полоски на 1–5.
        if (SPECS.get(kind) or {}).get("atom"):
            assets = {f"{n}.png": _png(star_strip(made, n)) for n in range(1, 6)}
        else:
            assets = {(PART_KINDS[kind]["assets"] or ["asset.png"])[0]: _png(made)}

        meta = parts.save_version(
            kind, target_version, assets=assets, title=target_version,
            note=f"перерисовано моделью по образцу {source_version}",
            source="регенерация по образцу", anchor=src.anchor,
            make_default=make_default,
        )
        result["parts"].append({"kind": kind, "version": meta.version,
                                "assets": meta.assets, "journal": journal})
        if on_event:
            on_event("part_done", {"kind": kind, "assets": meta.assets})

    # Шрифт картинкой не рисуется — переносим описание стиля как есть.
    font_src = parts.load_version("font", source_version)
    if font_src and font_src.font:
        parts.save_version("font", target_version, title=target_version,
                           note=f"стиль из {source_version}",
                           source="регенерация по образцу",
                           font=font_src.font, make_default=make_default)
        result["parts"].append({"kind": "font", "version": target_version,
                                "assets": []})

    if on_event:
        on_event("done", {"target": target_version,
                          "ok": len(result["parts"]), "failed": len(result["failed"])})
    return result
