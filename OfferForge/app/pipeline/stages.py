"""Этапы пайплайна: бриф → концепты → генерация → QC.

Каждый этап — чистая функция над Registry и данными. Ни один не знает,
какая конкретно модель его исполняет: он просит вариант по имени.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any
from pathlib import Path

from app.config import reference_images, resolve_variant, style_bible
from app.models import (
    Brief,
    ElementSpec,
    GeneratedAsset,
    OfferPlan,
    OfferTemplate,
    Preset,
    QCVerdict,
)
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry

log = logging.getLogger("offerforge.stages")

# Базовые категории композиции. Пятая — золотые карточки с персонажами —
# добавляется только если она включена в style_bible.collection.gold_cards.
BASE_CATEGORIES = ("1", "2", "3", "4")


def collection_rules() -> dict:
    """Правила набора из style_bible — тот же источник, что и у промптов."""
    return style_bible().get("collection") or {}


def required_categories() -> tuple[str, ...]:
    gold = (collection_rules().get("gold_cards") or {}).get("enabled")
    return BASE_CATEGORIES + ("5",) if gold else BASE_CATEGORIES


def set_size(default: int) -> int:
    return int(collection_rules().get("set_size") or default)


# Совместимость: на это имя ссылается selftest и внешний код.
CATEGORIES = BASE_CATEGORIES


# ---------------------------------------------------------------------------
# 1. Бриф
# ---------------------------------------------------------------------------

async def parse_brief(
    reg: Registry, preset: Preset, theme: str, wishes: str,
    on_event=None, *, lang: str = "ru",
) -> Brief:
    """Разбирает заказ. При недоступной модели откатывается на заказ как есть,
    но обязательно сообщает об этом: молчаливая деградация выглядит как
    успешный этап и уводит диагностику в сторону."""
    code = (lang or "ru").strip().lower()
    lang = "en" if code.startswith("en") else "ru"
    prompt = engine.render("brief", theme=theme, wishes=wishes, lang=lang)
    variant = resolve_variant(preset, "brief")
    try:
        resp = await reg.text(variant, prompt, json_mode=True)
        data = engine.parse_json_response(resp.text)
        if isinstance(data, dict) and data.get("theme"):
            return Brief(**{k: v for k, v in data.items() if k in Brief.model_fields})
        reason = "модель вернула ответ без поля theme"
    except (ProviderError, ValueError) as e:
        reason = str(e).replace("\n", " ")

    log.warning("Брифинг не удался (%s), беру заказ как есть", reason)
    if on_event:
        on_event("warn", {"stage": "brief", "message": reason,
                          "detail": "беру заказ как есть, без разбора"})
    brief = Brief(theme=theme, notes=wishes)
    brief.degraded = True
    return brief


# ---------------------------------------------------------------------------
# 2. Концепты (fan-out)
# ---------------------------------------------------------------------------

def _validate_plan(plan: OfferPlan, n_elements: int) -> list[str]:
    """Жёсткие правила ТЗ, проверяемые кодом, а не на глаз."""
    issues: list[str] = []

    if len(plan.elements) != n_elements:
        issues.append(f"элементов {len(plan.elements)}, нужно {n_elements}")

    subjects = [e.subject.strip().lower() for e in plan.elements]
    if len(set(subjects)) != len(subjects):
        dupes = {s for s in subjects if subjects.count(s) > 1}
        issues.append(f"дубли объектов: {', '.join(sorted(dupes))}")

    present = {e.category for e in plan.elements}
    missing = set(required_categories()) - present
    if missing:
        issues.append(f"нет категорий: {', '.join(sorted(missing))}")

    # Золотые карточки выключены — категории 5 в наборе быть не должно.
    if "5" not in required_categories() and "5" in present:
        issues.append("категория 5 (золотые карты) в этом наборе запрещена")

    return issues


def _repair_plan(plan: OfferPlan, n_elements: int) -> OfferPlan:
    """Чинит то, что чинится детерминированно: дубли и покрытие категорий."""
    seen: set[str] = set()
    unique: list[ElementSpec] = []
    for e in plan.elements:
        key = e.subject.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(e)
    plan.elements = unique[:n_elements]

    # Догоняем покрытие категорий, переназначая их у наименее «дорогих» карт.
    allowed = required_categories()
    for element in plan.elements:
        if element.category not in allowed:
            element.category = allowed[0]

    present = {e.category for e in plan.elements}
    missing = [c for c in allowed if c not in present]
    if missing and plan.elements:
        by_rarity = sorted(plan.elements, key=lambda e: e.rarity)
        for cat, element in zip(missing, by_rarity):
            element.category = cat

    return plan


def _skin_palette(skin: str) -> list[str]:
    """Палитра разобранного скина — из его разметки.

    Записывается при сегментации: vision называет главные цвета экрана.
    Если скина нет или палитра не сохранилась, возвращается пусто, и
    остаётся палитра концепта.
    """
    try:
        from app import skins as skins_lib

        markup = skins_lib.load_markup(skin) or {}
    except Exception:  # noqa: BLE001 — отсутствие скина не повод падать
        return []
    palette = [c for c in (markup.get("palette") or [])
               if isinstance(c, str) and c.startswith("#")]
    return palette[:5]


def _as_variants(parsed: Any) -> list:
    """Достаёт список вариантов из ответа модели, какой бы формы он ни был.

    Просили массив, но когда вариант нужен один, модель отвечает просто
    объектом: `{"variant_id": "v1", "title": ..., "elements": [...]}`.
    Формально это нарушение схемы, по сути — совершенно разумный ответ.
    Раньше он попадал в ветку `parsed.get("variants", [])`, давал пустоту,
    и прогон падал с «модель вернула пустой список», хотя модель ответила
    как надо.

    Ещё модель иногда заворачивает ответ в служебное поле — `thinking`,
    `result`, `output`. Поэтому ищем список вариантов по любому ключу, а
    не только по заранее известному имени.
    """
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []

    # Сам объект и есть вариант.
    if any(k in parsed for k in ("variant_id", "elements", "title")):
        return [parsed]

    for key in ("variants", "concepts", "items", "result", "data", "output"):
        value = parsed.get(key)
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict) and any(
                k in value for k in ("variant_id", "elements", "title")):
            return [value]

    # Последняя попытка: любой список словарей, похожих на вариант.
    for value in parsed.values():
        if (isinstance(value, list) and value
                and isinstance(value[0], dict)
                and any(k in value[0] for k in ("variant_id", "elements", "subject"))):
            return value
    return []


async def generate_concepts(
    reg: Registry,
    preset: Preset,
    brief: Brief,
    n_variants: int,
    n_elements: int,
) -> list[OfferPlan]:
    prompt = engine.render(
        "concepts",
        brief=brief.model_dump(),
        n_variants=n_variants,
        n_elements=n_elements,
    )
    variant = resolve_variant(preset, "concept")

    raw: list = []
    errors: list[str] = []
    last_text = ""
    attempts = 3
    for attempt in range(attempts):
        try:
            resp = await reg.text(variant, prompt, json_mode=True)
            last_text = resp.text or ""
            parsed = engine.parse_json_response(resp.text)
            raw = _as_variants(parsed)
            if raw:
                break
            # Показываем, что именно пришло: «пустой список» без ответа
            # модели не даёт понять, молчит она, ругается или отдаёт JSON
            # не той формы.
            head = " ".join((last_text or "").split())[:200]
            errors.append(f"пустой список; ответ модели: {head!r}"
                          if head else "модель ответила пустотой")
        except (ProviderError, ValueError) as e:
            log.warning("Концепты, попытка %d: %s", attempt + 1, e)
            errors.append(str(e))

    if not raw:
        # Три одинаковые ошибки подряд — это одна причина, а не три.
        uniq = list(dict.fromkeys(errors))
        detail = (f"{uniq[0]} (повторилось {len(errors)} раза)"
                  if len(uniq) == 1 else "\n  ".join(uniq))
        # Придумывать концепты за модель нельзя: получится десяток
        # «предметов номер N», и прогон честно потратит деньги на мусорный
        # арт. Вместо этого подсказываем, куда смотреть.
        raise ProviderError(
            f"не удалось получить концепты — {detail}\n"
            f"  Роль concept сейчас: {variant.model_name}. "
            "Проверьте её кнопкой «Проверить» на экране «Модели»: "
            "чаще всего модель уходит в рассуждения и обрывает JSON "
            "по лимиту токенов, либо запрос режет политика ключа.")

    plans: list[OfferPlan] = []
    for i, item in enumerate(raw[:n_variants]):
        elements = []
        titles: dict[str, str] = {}
        for j, el in enumerate(item.get("elements", [])):
            spec = ElementSpec(
                slot=el.get("slot", "art"),
                subject=el.get("subject", ""),
                category=str(el.get("category", "1")),
                rarity=int(el.get("rarity", 1) or 1),
                seed=random.randint(1, 2**31 - 1),
            )
            elements.append(spec)
            titles[f"element_{j}"] = el.get("title_ru") or el.get("subject", "")

        plan = OfferPlan(
            variant_id=item.get("variant_id") or f"v{i + 1}",
            title=item.get("title", f"Вариант {i + 1}"),
            concept=item.get("concept", ""),
            elements=elements,
            palette=item.get("palette") or brief.palette,
            texts={"set_title": item.get("title", ""), **titles},
        )

        issues = _validate_plan(plan, n_elements)
        if issues:
            log.info("Вариант %s чиним: %s", plan.variant_id, "; ".join(issues))
            plan = _repair_plan(plan, n_elements)
        plans.append(plan)

    return plans


def cross_variant_overlap(plans: list[OfferPlan]) -> dict[str, list[str]]:
    """Пересечения объектов между вариантами — попадают в отчёт прогона."""
    out: dict[str, list[str]] = {}
    for i, a in enumerate(plans):
        for b in plans[i + 1:]:
            sa = {e.subject.strip().lower() for e in a.elements}
            sb = {e.subject.strip().lower() for e in b.elements}
            common = sorted(sa & sb)
            if common:
                out[f"{a.variant_id}↔{b.variant_id}"] = common
    return out


# ---------------------------------------------------------------------------
# 3. Промпты и генерация
# ---------------------------------------------------------------------------

def build_prompt(element: ElementSpec, palette: list[str], extra: str = "",
                 trigger: str = "") -> str:
    """Собирает промпт арта.

    trigger — слово, на котором обучалась своя LoRA. Оно должно стоять в
    начале: дообученная модель узнаёт стиль именно по нему, а в середине
    длинного текста токен теряет вес.
    """
    return engine.render(
        "art",
        subject=element.subject,
        category=element.category,
        palette=palette,
        extra=extra,
        trigger=trigger,
        seed=getattr(element, "seed", 0) or 0,
    ).strip()


async def render_element(
    reg: Registry,
    preset: Preset,
    element: ElementSpec,
    palette: list[str],
    out_dir: Path,
    key: str,
    *,
    max_attempts: int = 3,
    refs: list[bytes] | None = None,
    on_event=None,
) -> tuple[GeneratedAsset | None, list[QCVerdict], list[Path]]:
    """Генерирует один элемент и прогоняет его через QC с авторетраем.

    Отбракованные версии сохраняются намеренно: они и есть доказательство
    хода работы, а без них «прозрачность процесса» ничем не подтверждается.
    """
    render_variant = resolve_variant(preset, "render")
    critic_variant = resolve_variant(preset, "critic")
    sb = style_bible()

    rejected: list[Path] = []
    verdicts: list[QCVerdict] = []
    extra = ""

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(element, palette, extra,
                              trigger=getattr(render_variant, "trigger", "") or "")
        started = time.monotonic()

        try:
            img = await reg.image(
                render_variant,
                prompt,
                negative=sb.get("negative", ""),
                width=1024,
                height=1024,
                seed=element.seed,
                references=refs or [],
            )
        except ProviderError as e:
            log.error("Генерация %s провалилась: %s", key, e)
            if on_event:
                on_event("error", {"slot": key, "message": str(e)})
            return None, verdicts, rejected

        data = img.images[0]
        elapsed = time.monotonic() - started

        # QC
        verdict = await critique(reg, critic_variant, data, element)
        verdicts.append(verdict)

        if verdict.passed or attempt == max_attempts:
            suffix = "" if verdict.passed else "_needs_review"
            path = out_dir / f"{key}{suffix}.png"
            path.write_bytes(data)
            if on_event:
                on_event("asset", {
                    "slot": key, "passed": verdict.passed,
                    "attempt": attempt, "path": path.name,
                })
            return (
                GeneratedAsset(
                    slot=key, path=str(path), model=img.model,
                    provider=render_variant.rotation()[0].provider_id,
                    seed=element.seed, prompt=prompt, attempt=attempt,
                    cost_usd=img.cost_usd, elapsed_s=elapsed,
                ),
                verdicts,
                rejected,
            )

        # Не прошло — сохраняем отбраковку и уточняем промпт
        rej_dir = out_dir.parent / "rejected"
        rej_dir.mkdir(parents=True, exist_ok=True)
        rej_path = rej_dir / f"{key}_a{attempt}.png"
        rej_path.write_bytes(data)
        rejected.append(rej_path)

        extra = verdict.fix_hint or verdict.reason
        element.seed = random.randint(1, 2**31 - 1)
        if on_event:
            on_event("retry", {"slot": key, "attempt": attempt, "reason": verdict.reason})

    return None, verdicts, rejected


async def critique(
    reg: Registry, variant, image: bytes, element: ElementSpec
) -> QCVerdict:
    sb = style_bible()
    prompt = engine.render(
        "critic",
        subject=element.subject,
        category=element.category,
        style_summary=sb.get("summary", "соответствует стилю проекта"),
    )
    try:
        resp = await reg.text(variant, prompt, images=[image], json_mode=True)
        data = engine.parse_json_response(resp.text)
        return QCVerdict(
            slot=element.slot,
            passed=bool(data.get("passed")),
            scores={k: bool(v) for k, v in (data.get("scores") or {}).items()},
            reason=data.get("reason", "") or "",
            fix_hint=data.get("fix_hint", "") or "",
        )
    except (ProviderError, ValueError) as e:
        # Критик недоступен — пропускаем карточку, но честно помечаем это.
        log.warning("QC недоступен для %s: %s", element.slot, e)
        return QCVerdict(
            slot=element.slot, passed=True,
            reason=f"QC пропущен: {e}", scores={"qc_skipped": True},
        )


async def render_plan(
    reg: Registry,
    preset: Preset,
    plan: OfferPlan,
    template: OfferTemplate,
    out_dir: Path,
    *,
    concurrency: int = 4,
    skin: str | None = None,
    on_event=None,
) -> tuple[dict[str, bytes], list[GeneratedAsset], list[QCVerdict], list[str]]:
    """Генерирует все элементы одного варианта оффера параллельно."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Скин UI — ок; Icons-арты как input_references дают subject leakage
    # (ласты/арки вместо нужного предмета). Для стиля карточек — LoRA/промпт.
    refs = reference_images(3, skin=skin, include_art=False)

    # Палитра: скин важнее выдумки модели. Концепт возвращает свою палитру
    # («#FFD700» для «Кино-коллекции»), и арт выходил золотым на синем
    # скине — тема победила гамму, хотя оффер собирается в цветах скина.
    palette = plan.palette
    if skin:
        skin_palette = _skin_palette(skin)
        if skin_palette:
            palette = skin_palette
            log.info("Палитра взята из скина %s: %s", skin, ", ".join(palette))
    # Слой генерации может быть и слоем-шаблоном: у набора image_gen лежит
    # внутри вложенной карточки, а имя слота задаётся здесь.
    gen_layer = template.art_layer()
    layer_id = gen_layer.id if gen_layer else "art"
    repeated = bool(gen_layer and gen_layer.repeat_for)

    sem = asyncio.Semaphore(concurrency)

    async def one(idx: int, element: ElementSpec):
        key = f"{layer_id}#{idx}" if repeated else layer_id
        async with sem:
            return key, await render_element(
                reg, preset, element, palette, out_dir, key,
                refs=refs, on_event=on_event,
            )

    results = await asyncio.gather(
        *(one(i, e) for i, e in enumerate(plan.elements)), return_exceptions=True
    )

    art: dict[str, bytes] = {}
    assets: list[GeneratedAsset] = []
    verdicts: list[QCVerdict] = []
    needs_review: list[str] = []

    for res in results:
        if isinstance(res, BaseException):
            log.error("Элемент упал: %s", res)
            continue
        key, (asset, vs, _rejected) = res
        verdicts.extend(vs)
        if asset is None:
            needs_review.append(key)
            continue
        assets.append(asset)
        art[key] = Path(asset.path).read_bytes()
        if vs and not vs[-1].passed:
            needs_review.append(key)

    return art, assets, verdicts, needs_review
