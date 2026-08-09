"""Движок промптов.

Промпты — внешние .j2-файлы, а не строки в коде: их правят без пересборки,
диффы читаемы, и каждая версия попадает в артефакты прогона. Плюс набор
декораторов — функций, доступных прямо в шаблоне.

Идея повторяет prompts + decorators из SkyrimNet, только на Jinja2.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from app.config import PROMPTS_DIR, style_bible

_DECORATORS: dict[str, Callable[..., Any]] = {}


def decorator(name: str):
    def wrap(fn: Callable[..., Any]):
        _DECORATORS[name] = fn
        return fn
    return wrap


# ---------------------------------------------------------------------------
# Встроенные декораторы
# ---------------------------------------------------------------------------

@decorator("style_rules")
def _style_rules() -> str:
    """Общие стилевые правила из style_bible.json."""
    sb = style_bible()
    parts = sb.get("rules") or []
    return "\n".join(f"- {p}" for p in parts)


@decorator("category_rule")
def _category_rule(category: str) -> str:
    """Правило композиции для категории 1..5 (из ТЗ) — жёсткий блок для art.j2."""
    sb = style_bible()
    cats = sb.get("categories") or {}
    key = str(category)
    c = cats.get(key) or {}
    if not c:
        return ""
    title = c.get("title_en") or c.get("title_ru") or f"category {key}"
    lines = [f"COMPOSITION CATEGORY {key} — {title}:"]
    if c.get("object"):
        lines.append(f"- Object: {c['object']}.")
    if c.get("background"):
        lines.append(f"- Background: {c['background']}.")
    if c.get("forbid"):
        lines.append(f"- Do NOT: {c['forbid']}.")
    lines.append(
        "Follow THIS category exactly for staging and background. "
        "Do not simplify every card into a floating icon on a blank studio void "
        "unless this is category 1."
    )
    return "\n".join(lines)


@decorator("negative_prompt")
def _negative() -> str:
    sb = style_bible()
    return sb.get("negative", "")


@decorator("palette_words")
def _palette_words(palette: list[str] | None) -> str:
    return ", ".join(palette or []) or "vivid saturated colors"


@decorator("as_json")
def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


# --- правила коллекции (единственный источник — style_bible.collection) ----

def _collection() -> dict:
    return style_bible().get("collection") or {}


@decorator("collection_structure")
def _collection_structure() -> str:
    """Как устроена коллекция: сколько наборов, сколько карт в наборе."""
    return _collection().get("structure", "")


@decorator("collection_rules")
def _collection_rules() -> str:
    """Правила подбора объектов в набор."""
    return "\n".join(f"- {r}" for r in _collection().get("rules", []))


@decorator("category_table")
def _category_table() -> str:
    """Таблица категорий композиции из ТЗ (для collection_fill / brief)."""
    cats = style_bible().get("categories") or {}
    gold_on = bool((_collection().get("gold_cards") or {}).get("enabled"))
    keys = ("1", "2", "3", "4", "5") if gold_on else ("1", "2", "3", "4")
    blocks: list[str] = []
    for key in keys:
        c = cats.get(key) or {}
        if not c:
            continue
        title = c.get("title_ru") or c.get("title_en") or f"Категория {key}"
        lines = [f"Категория {key} — {title}"]
        if c.get("object"):
            lines.append(f"  объект: {c['object']}")
        if c.get("background"):
            lines.append(f"  фон: {c['background']}")
        if c.get("forbid"):
            lines.append(f"  нельзя: {c['forbid']}")
        blocks.append("\n".join(lines))
    header = (
        "Категории композиции (из ТЗ). В каждом обычном наборе должны быть "
        "категории 1–4; категория 5 — только золотые карточки, если они включены."
    )
    return header + "\n\n" + "\n\n".join(blocks)


@decorator("rarity_distribution")
def _rarity_distribution() -> str:
    """Раскладка редкостей внутри набора, например 1,1,2,2,3,3,4,4,5,5."""
    return ",".join(str(v) for v in _collection().get("rarity_distribution", []))


@decorator("category_distribution")
def _category_distribution() -> str:
    """Сколько карточек какой категории должно быть в наборе."""
    dist = _collection().get("category_distribution") or {}
    return ", ".join(f"категория {k} — {v} шт" for k, v in dist.items())


@decorator("gold_policy")
def _gold_policy() -> str:
    """Разрешены ли золотые карточки и по каким правилам."""
    gold = _collection().get("gold_cards") or {}
    if not gold.get("enabled"):
        return ("Золотые карточки в этом наборе НЕ используются. "
                "Никаких персонажей и сюжетных сценок — только предметы.")
    return "Золотые карточки разрешены:\n" + "\n".join(
        f"- {r}" for r in gold.get("rules", [])
    )


@decorator("variant_strategy")
def _variant_strategy() -> str:
    """Как разводить 3–5 вариантов одного набора (из ТЗ)."""
    rules = _collection().get("variant_strategy") or []
    return "\n".join(f"- {r}" for r in rules) if rules else (
        "- Варианты отличаются творческим углом, не палитрой.\n"
        "- Пересечение объектов между вариантами — не больше двух."
    )


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------

def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(_DECORATORS)
    return env


def render(template_name: str, **ctx: Any) -> str:
    """Рендерит промпт по имени файла (без расширения)."""
    env = _env()
    try:
        tpl = env.get_template(f"{template_name}.j2")
    except TemplateNotFound as e:
        raise FileNotFoundError(f"Промпт '{template_name}.j2' не найден в {PROMPTS_DIR}") from e
    return tpl.render(**ctx)


def available_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.j2"))


def parse_json_response(text: str) -> Any:
    """Достаёт JSON из ответа модели, терпя ```-обёртки и болтовню вокруг."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Вырезаем JSON из болтовни вокруг. Пробуем ту скобку, что встретилась
    # раньше: иначе в ответе вида «вот результат: [{...}] готово» первым
    # найдётся внутренний объект и массив потеряется.
    candidates = [(s.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if s.find(o) != -1]
    for _, opener, closer in sorted(candidates):
        start, end = s.find(opener), s.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Не удалось разобрать JSON из ответа модели: {text[:200]}")
