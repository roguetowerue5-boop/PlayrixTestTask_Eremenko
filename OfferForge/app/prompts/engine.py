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
    """Правило композиции для категории 1..5 (из ТЗ)."""
    sb = style_bible()
    cats = sb.get("categories") or {}
    c = cats.get(str(category)) or {}
    if not c:
        return ""
    # Раздельными строками: слитый текст модель читает как одно длинное
    # требование и теряет половину.
    bits = []
    if c.get("object"):
        bits.append(f"Object: {c['object']}.")
    if c.get("background"):
        bits.append(f"Background: {c['background']}.")
    return "\n".join(bits)


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
    """Описание всех четырёх категорий композиции одним блоком."""
    cats = style_bible().get("categories") or {}
    lines = []
    for key in ("1", "2", "3", "4"):
        c = cats.get(key) or {}
        if c:
            lines.append(f"{key} — {c.get('object', '')}; фон: {c.get('background', '')}")
    return "\n".join(lines)


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
