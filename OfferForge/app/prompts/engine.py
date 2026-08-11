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


@decorator("style_lock")
def _style_lock() -> str:
    """Тот же STYLE LOCK, что в ComfyUI Icons Style A Positive."""
    sb = style_bible()
    lock = (sb.get("style_lock") or "").strip()
    if lock:
        return lock
    parts = sb.get("rules") or []
    return ", ".join(parts)


@decorator("category_rule")
def _category_rule(category: str) -> str:
    """Короткое правило композиции 1..5 для art.j2 — без повторов."""
    sb = style_bible()
    cats = sb.get("categories") or {}
    key = str(category)
    c = cats.get(key) or {}
    if not c:
        return ""
    title = c.get("title_en") or c.get("title_ru") or key
    bits = [f"Cat {key} ({title})"]
    if c.get("object"):
        bits.append(c["object"])
    if c.get("background"):
        bits.append("BG: " + c["background"])
    if c.get("forbid"):
        bits.append("Avoid: " + c["forbid"])
    return ". ".join(bits) + "."


@decorator("category_rule_short")
def _category_rule_short(category: str) -> str:
    """Сжатое правило категории — меньше шума, сильнее subject."""
    key = str(category or "1")
    short = {
        "1": "Cat 1: object floats mid-air, flat solid or soft-gradient backdrop, one object only",
        "2": "Cat 2: object on a simple flat surface, plain vertical backdrop, one object only",
        "3": "Cat 3: object on a realistic surface (wood/stone/fabric), simple vertical backdrop, one object only",
        "4": "Cat 4: object in a realistic environment, stays the clear central focus, one hero only",
        "5": "Cat 5: one or two characters in a simple narrative beat, cat-4 style environment",
    }
    return short.get(key, short["1"])


def _pick_option(options: list[str], *, key: str, salt: str) -> str:
    """Стабильный выбор одного варианта по ключу (subject+seed+salt)."""
    if not options:
        return ""
    raw = f"{salt}|{key}".encode("utf-8", errors="ignore")
    idx = sum(raw) % len(options)
    return options[idx]


@decorator("lighting_pick")
def _lighting_pick(subject: str = "", seed: int | str = 0) -> str:
    """Опциональный pick; art.j2 Style A обычно не вызывает."""
    block = style_bible().get("lighting") or {}
    options = list(block.get("options") or [])
    if not options:
        return "soft studio light"
    return _pick_option(options[:5], key=f"{subject}|{seed}", salt="light")


@decorator("support_pick")
def _support_pick(category: str = "1", subject: str = "", seed: int | str = 0) -> str:
    """Опциональный pick; категория задаёт композицию в category_rule."""
    block = style_bible().get("supports") or {}
    key = str(category or "1")
    if key == "1":
        return block.get("floating") or "hanging in the air, no surface under it"
    if key == "5":
        return "characters integrated in the scene, remain central"
    pool_key = {
        "2": "options_cat2",
        "3": "options_cat3",
        "4": "options_cat4",
    }.get(key, "options_cat2")
    options = list(block.get(pool_key) or block.get("options") or [])
    if not options:
        return "on a simple surface with soft contact shadow"
    return _pick_option(options, key=f"{subject}|{seed}|{key}", salt="support")


@decorator("lighting_rule")
def _lighting_rule(subject: str = "", seed: int | str = 0) -> str:
    """Одна короткая фраза света — без меню вариантов в промпте."""
    chosen = _lighting_pick(subject, seed)
    return f"light: {chosen}"


@decorator("support_rule")
def _support_rule(
    category: str = "1", subject: str = "", seed: int | str = 0,
) -> str:
    """Одна короткая фраза опоры — без меню вариантов в промпте."""
    chosen = _support_pick(category, subject, seed)
    return f"support: {chosen}"


@decorator("negative_prompt")
def _negative() -> str:
    sb = style_bible()
    return sb.get("negative", "")


@decorator("palette_words")
def _palette_words(palette: list[str] | None) -> str:
    """Палитра — мягкая подсказка цвета, не жёсткий запрет других оттенков."""
    colors = ", ".join(palette or []) or "clear readable colors"
    return (
        f"{colors} "
        "(accent hints only — keep the subject's own material colors; "
        "do not force candy-pink recolor of the hero)"
    )


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
    """Composition category table for collection_fill / brief."""
    cats = style_bible().get("categories") or {}
    gold_on = bool((_collection().get("gold_cards") or {}).get("enabled"))
    keys = ("1", "2", "3", "4", "5") if gold_on else ("1", "2", "3", "4")
    blocks: list[str] = []
    for key in keys:
        c = cats.get(key) or {}
        if not c:
            continue
        title = c.get("title_en") or c.get("title_ru") or f"Category {key}"
        lines = [f"Category {key} — {title}"]
        if c.get("object"):
            lines.append(f"  object: {c['object']}")
        if c.get("background"):
            lines.append(f"  background: {c['background']}")
        if c.get("forbid"):
            lines.append(f"  forbid: {c['forbid']}")
        blocks.append("\n".join(lines))
    header = (
        "Composition categories. Every normal set must include categories 1–4; "
        "category 5 is gold cards only when enabled."
    )
    return header + "\n\n" + "\n\n".join(blocks)


@decorator("rarity_distribution")
def _rarity_distribution() -> str:
    """Rarity layout inside a set, e.g. 1,1,2,2,3,3,4,4,5,5."""
    return ",".join(str(v) for v in _collection().get("rarity_distribution", []))


@decorator("category_distribution")
def _category_distribution() -> str:
    """How many cards per composition category in a set."""
    dist = _collection().get("category_distribution") or {}
    return ", ".join(f"category {k}: {v}" for k, v in dist.items())


@decorator("gold_policy")
def _gold_policy() -> str:
    """Whether gold cards are allowed and under which rules."""
    gold = _collection().get("gold_cards") or {}
    if not gold.get("enabled"):
        return (
            "Gold cards are NOT used in this set. "
            "No characters or story scenes — objects only."
        )
    return "Gold cards are allowed:\n" + "\n".join(
        f"- {r}" for r in gold.get("rules", [])
    )


@decorator("variant_strategy")
def _variant_strategy() -> str:
    """How to diverge 1–6 variants of one set."""
    rules = _collection().get("variant_strategy") or []
    return "\n".join(f"- {r}" for r in rules) if rules else (
        "- Variants differ by creative angle, not palette.\n"
        "- Object overlap between variants — at most two."
    )


@decorator("output_lang_rules")
def _output_lang_rules(lang: str = "en") -> str:
    """Force generated prose/labels into one language; subjects stay English."""
    code = (lang or "en").strip().lower()
    if code.startswith("ru"):
        return (
            "OUTPUT LANGUAGE: Russian ONLY.\n"
            "- All prose, display names, angles, concepts, notes: Russian.\n"
            "- Image-model subjects: ALWAYS short English.\n"
            "- Do not mix Russian and English in display names or prose."
        )
    return (
        "OUTPUT LANGUAGE: English ONLY.\n"
        "- All prose, display names, angles, concepts, notes: English.\n"
        "- Image-model subjects: short English.\n"
        "- Do not use Russian anywhere in the response "
        "(except unavoidable proper nouns already in the title)."
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
