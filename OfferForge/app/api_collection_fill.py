"""Наполнение коллекции: концепты + арт через OpenRouter.

Экран «Создание наполнения коллекции»: название + описание → 10 карточек
(роль concept + render из пресета, промпты collection_fill.j2 и art.j2).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import re
import time
import zipfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    RUNS_DIR,
    is_offline,
    list_presets,
    load_preset,
    providers_config,
    resolve_variant,
    style_bible,
)
from app import settings as app_settings
from app.models import Brief, ElementSpec
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry
from app.pipeline.stages import build_prompt, parse_brief

log = logging.getLogger("offerforge.collection_fill")
router = APIRouter(prefix="/api/collection/fill", tags=["collection-fill"])

FILL_RUNS = RUNS_DIR / "collection-fill"
N_SLOTS = 10

# Fallback только если в bible нет style_lock (стиль в art.j2 через style_lock,
# сюда тематический mood — без повтора Style A).
LORA_STYLE_EXTRA = ""

_STYLE_MARKERS = (
    "playrix-like casual mobile game icon",
    "playrix icons look",
    "keep a playrix icons look",
)


def _art_extra_default() -> str:
    sb = style_bible()
    return (sb.get("art_extra_default") or "").strip()


def _strip_style_echo(text: str) -> str:
    """Убирает куски, которые уже есть в style_lock (чтобы extra не дублировал)."""
    raw = (text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    # Целиком дефолтный Style A hint — не нужен рядом со style_lock.
    if low.startswith("keep a playrix icons look") and "composition category" in low:
        return ""
    if _STYLE_MARKERS[0] in low and "single hero" in low and len(raw) < 400:
        # Похоже на полный style_lock, вставленный в extra.
        return ""
    return raw


# Чистим только «сцену/свет» в subject, не сами объекты (camera, binoculars, stand-as-part).
_LIGHTING_BAN = re.compile(
    r"(?i)\b("
    r"projector\s*beam|volumetric\s+(?:light|beam|shaft|fog)|"
    r"god\s*rays?|chiaroscuro|rim\s*light|stage\s*light(?:ing)?|"
    r"harsh\s+(?:overhead|top)\s+light|dramatic\s+lighting|"
    r"high[- ]contrast(?:\s+lighting)?|overhead\s+spot(?:light)?|"
    r"light\s+cone|light\s+shaft|light\s+puddle|"
    r"casting\b[^,]{0,40}(?:shadows?|light)|"
    r"(?:sharp\s+)?directional\s+shadows?|"
    r"(?:smoky\s+)?(?:noir\s+)?atmosphere|"
    r"прожектор|конус\s+света|объ[её]мн(?:ый|ое)\s+свет"
    r")\b[^.\n;]*[.\n;]?"
)
_STAND_BAN = re.compile(
    r"(?i)\b("
    r"museum\s+pedestal|ornate\s+pedestal|golden\s+pedestal|"
    r"display\s+plinth|on\s+a\s+(?:museum\s+)?pedestal|"
    r"пьедестал|подставк[ауие]\s+(?:музей|витрин)"
    r")\b[^.\n;]*[.\n;]?"
)
_SUBJECT_JUNK = re.compile(
    r"(?i)\b("
    r"no\s+characters|no\s+people|no\s+text|no\s+logo|"
    r"three[- ]quarter\s+view|collectible\s+icon"
    r")\b[^,\n;]*"
)


def _sanitize_theme_notes(text: str) -> str:
    """Mood/palette only: вырезает предписания света и подставок из extra."""
    raw = _strip_style_echo(text)
    if not raw:
        return ""
    cleaned = _LIGHTING_BAN.sub(" ", raw)
    cleaned = _STAND_BAN.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip(" \n,;.-")
    return cleaned


def _sanitize_subject(text: str) -> str:
    """Subject для Flux: только объект. Свет/стол/атмосферу режет art.j2 отдельно."""
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = _LIGHTING_BAN.sub(" ", raw)
    cleaned = _STAND_BAN.sub(" ", cleaned)
    cleaned = _SUBJECT_JUNK.sub(" ", cleaned)
    cleaned = re.sub(r"[,;]\s*[,;]+", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip(" ,;.-")
    # Не раздувать subject: Flux сильнее слушает начало.
    words = cleaned.split()
    if len(words) > 14:
        cleaned = " ".join(words[:14])
    return cleaned


def _with_lora_extra(extra: str, trigger: str) -> str:
    """Extra для LoRA — только тематические notes, без Style A и без light/stand."""
    return _sanitize_theme_notes(extra)


def _brief_art_extra(brief: Brief | None, art_extra: str = "") -> str:
    """Склеивает доп. к арту + mood/avoid из brief. Стиль/свет не дублируем."""
    bits: list[str] = []
    text = _sanitize_theme_notes(art_extra)
    if text:
        bits.append(text)
    if brief:
        if brief.mood:
            mood = _sanitize_theme_notes(", ".join(brief.mood))
            if mood:
                bits.append("Mood: " + mood)
        if brief.must_avoid:
            bits.append("Avoid depicting: " + ", ".join(brief.must_avoid))
        if brief.era:
            bits.append(f"Era: {brief.era}")
    return "\n".join(bits)


def _norm_lang(lang: str | None) -> str:
    code = (lang or "ru").strip().lower()
    return "en" if code.startswith("en") else "ru"


class FillRequest(BaseModel):
    title: str
    description: str = ""
    art_extra: str = ""  # доп. указания ко всем артам (уходит в art.j2 как extra)
    preset_id: str | None = None
    concurrency: int = Field(3, ge=1, le=8)
    n_variants: int = Field(3, ge=1, le=6)
    # Какой вариант рисовать после придумывания (0-based).
    variant_index: int = Field(0, ge=0, le=5)
    lang: str = "ru"  # "ru" | "en" — язык генерируемого текста


class PlanRequest(BaseModel):
    """Только brief + concept: 1–6 вариантов × 10 объектов без картинок."""
    title: str
    description: str = ""
    preset_id: str | None = None
    n_variants: int = Field(3, ge=1, le=6)
    lang: str = "ru"


class ManualElementIn(BaseModel):
    """Один слот ручного плана."""
    subject: str = ""
    name: str = ""
    category: str = "1"
    rarity: int = Field(1, ge=1, le=5)


class ManualPlanRequest(BaseModel):
    """План без AI: пользователь задаёт до 10 предметов сам."""
    title: str
    description: str = ""
    preset_id: str | None = None
    concept: str = ""
    angle: str = ""
    palette: list[str] = Field(default_factory=list)
    elements: list[ManualElementIn] = Field(default_factory=list)


class SelectVariantRequest(BaseModel):
    run_id: str
    variant_index: int = Field(0, ge=0, le=5)


class OneCardRequest(BaseModel):
    """Арт одного слота по уже сохранённому плану (или с ручным subject)."""
    run_id: str
    slot: int = Field(..., ge=1, le=N_SLOTS)
    preset_id: str | None = None
    art_extra: str = ""
    # Если заданы — перекрывают план (ручная правка subject перед генерацией).
    subject: str = ""
    name: str = ""
    category: str = ""
    rarity: int | None = Field(None, ge=1, le=5)
    # Сколько картинок сгенерировать по одному тексту (разные seed).
    n_images: int = Field(1, ge=1, le=4)


class RegenRequest(BaseModel):
    title: str
    description: str = ""
    preset_id: str | None = None
    run_id: str
    slot: int = Field(..., ge=1, le=N_SLOTS)
    subject: str
    name: str = ""
    category: str = "1"
    rarity: int = Field(1, ge=1, le=5)
    palette: list[str] = Field(default_factory=list)
    extra: str = ""  # доп. указания только для этой перегенерации
    n_images: int = Field(1, ge=1, le=4)


class UpscaleRequest(BaseModel):
    """Апскейл уже сгенерированной картинки слота (fal ESRGAN)."""
    run_id: str
    slot: int = Field(..., ge=1, le=N_SLOTS)
    scale: float = Field(2.0, ge=1.0, le=4.0)


class HardResetRequest(BaseModel):
    """Сбросить арт одного слота: файлы, мета генерации, доп. к арту."""
    run_id: str
    slot: int = Field(..., ge=1, le=N_SLOTS)
    name: str | None = None
    subject: str | None = None
    category: int | None = Field(None, ge=1, le=5)
    rarity: int | None = Field(None, ge=1, le=5)


class SuggestRequest(BaseModel):
    """🎲 — сгенерировать текст для поля описания, наполнения или доп. к арту."""
    kind: str  # "description" | "filling" | "art_extra"
    title: str
    description: str = ""  # для filling / art_extra — контекст стиля
    hint: str = ""         # текущий текст поля (переписать)
    preset_id: str | None = None
    lang: str = "ru"  # "ru" | "en"


class ParseDescriptionRequest(BaseModel):
    """Разбор описания через OpenRouter: сколько карточек и какие тексты."""
    title: str
    description: str
    preset_id: str | None = None
    lang: str = "ru"


def _safe_run_id(run_id: str) -> str:
    if "/" in run_id or "\\" in run_id or ".." in run_id or not run_id:
        raise HTTPException(400, "Недопустимый run_id")
    return run_id


@router.get("/meta")
async def fill_meta() -> dict:
    return {
        "slots": N_SLOTS,
        "variants_min": 1,
        "variants_max": 6,
        "presets": list_presets(),
        "prompts": [
            "brief", "collection_fill", "art",
            "suggest_description", "suggest_filling", "suggest_art_extra",
            "parse_description",
        ],
        "offline": is_offline(),
        "description_markup": (
            "At the end of filling add:\n"
            "[[n=10]]\n"
            "1. Name | english subject for model\n"
            "2. …\n"
            "Also allowed: ---cards:10--- instead of [[n=10]]."
        ),
        "langs": ["ru", "en"],
        "default_lang": "ru",
    }


@router.post("/suggest")
async def fill_suggest(req: SuggestRequest) -> dict:
    """Сгенерировать описание, наполнение или доп. указания к арту."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")
    kind = (req.kind or "").strip().lower()
    if kind not in ("description", "filling", "art_extra"):
        raise HTTPException(400, "kind: description, filling или art_extra")

    description = (req.description or "").strip()
    if kind in ("filling", "art_extra") and not description:
        raise HTTPException(
            400,
            "Сначала задай описание коллекции (цвет и стиль)",
        )

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    lang = _norm_lang(req.lang)
    if kind == "description":
        prompt = engine.render(
            "suggest_description",
            title=title,
            hint=(req.hint or "").strip(),
            lang=lang,
        )
    elif kind == "filling":
        prompt = engine.render(
            "suggest_filling",
            title=title,
            description=description,
            hint=(req.hint or "").strip(),
            lang=lang,
        )
    else:
        # Art mood is always English.
        prompt = engine.render(
            "suggest_art_extra",
            title=title,
            description=description,
            hint=(req.hint or "").strip(),
            lang="en",
        )

    # brief — дешёвая текстовая роль; чуть выше temperature через default fallback.
    variant = resolve_variant(preset, "brief")
    reg = Registry(providers_config(), offline=is_offline())
    try:
        resp = await reg.text(variant, prompt, json_mode=False)
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    text = (resp.text or "").strip()
    # Срезать markdown-обёртки, если модель всё же обернула.
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().lower().startswith("text"):
            text = text.lstrip()[4:]
        text = text.strip()
    if kind == "filling":
        text = _strip_filling_preamble(text)
    if not text:
        raise HTTPException(502, "Модель вернула пустой текст")

    return {
        "kind": kind,
        "text": text,
        "model": getattr(resp, "model", None) or variant.model_name,
        "cost_usd": float(getattr(resp, "cost_usd", 0) or 0),
    }


def _strip_filling_preamble(text: str) -> str:
    """Оставляет только [[n=…]] и нумерованный список — без «Набор 1: …»."""
    raw = (text or "").strip()
    if not raw:
        return raw
    m = re.search(r"\[\[\s*n\s*=\s*\d+\s*\]\]", raw, flags=re.IGNORECASE)
    if m:
        raw = raw[m.start():].strip()
    else:
        m2 = re.search(r"(?m)^\s*1\.\s+\S", raw)
        if m2:
            raw = "[[n=10]]\n" + raw[m2.start():].strip()
    lines = raw.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            if out and out[-1] != "":
                continue
            continue
        if i == 0 or (not out and re.match(r"\[\[\s*n\s*=", s, re.I)):
            out.append(s)
            continue
        if re.match(r"^\d{1,2}\.\s+\S", s):
            out.append(s)
            continue
        # Обрываем на первом абзаце после списка.
        if out and re.match(r"^\d{1,2}\.\s+\S", out[-1] or ""):
            break
    return "\n".join(out).strip()


@router.post("/parse-description")
async def fill_parse_description(req: ParseDescriptionRequest) -> dict:
    """OpenRouter: сколько карточек в описании и какие тексты (до генерации)."""
    title = (req.title or "").strip()
    description = (req.description or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")
    if not description:
        raise HTTPException(400, "Укажи описание коллекции с разметкой карточек")

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    prompt = engine.render(
        "parse_description",
        title=title,
        description=description,
        lang=_norm_lang(req.lang),
    )
    variant = resolve_variant(preset, "brief")
    reg = Registry(providers_config(), offline=is_offline())
    try:
        resp = await reg.text(variant, prompt, json_mode=True)
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    try:
        parsed = engine.parse_json_response(resp.text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Не разобрать ответ парсера: {e}") from e
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise HTTPException(502, "Парсер вернул не объект")

    raw_cards = parsed.get("cards") or []
    cards: list[dict] = []
    for i, item in enumerate(raw_cards[:N_SLOTS]):
        if not isinstance(item, dict):
            continue
        subject = (item.get("subject") or item.get("text") or "").strip()
        name = (item.get("name") or item.get("title_ru") or "").strip() or subject
        if not subject and not name:
            continue
        if not subject:
            subject = name
        cat = str(item.get("category") or "").strip() or None
        if cat not in ("1", "2", "3", "4", "5", None):
            cat = None
        rarity = item.get("rarity")
        try:
            rarity = int(rarity) if rarity is not None else None
        except (TypeError, ValueError):
            rarity = None
        if rarity is not None:
            rarity = max(1, min(5, rarity))
        cards.append({
            "slot": i + 1,
            "name": name,
            "subject": subject,
            "category": cat,
            "rarity": rarity,
        })

    count = len(cards)
    declared = parsed.get("declared_n")
    try:
        declared_n = int(declared) if declared is not None else None
    except (TypeError, ValueError):
        declared_n = None
    if declared_n is not None:
        declared_n = max(0, min(N_SLOTS, declared_n))

    notes = (parsed.get("notes") or "").strip()
    mismatch = (
        declared_n is not None and declared_n != count
    )

    return {
        "declared_n": declared_n,
        "count": count,
        "cards": cards,
        "notes": notes,
        "mismatch": mismatch,
        "model": getattr(resp, "model", None) or variant.model_name,
        "cost_usd": float(getattr(resp, "cost_usd", 0) or 0),
    }


def _parse_fill_plan(parsed, title: str) -> dict:
    """Нормализует один вариант к плану с 10 элементами."""
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise ValueError("ожидался JSON-объект с elements")
    elements = parsed.get("elements") or []
    if len(elements) < N_SLOTS:
        raise ValueError(f"в варианте {len(elements)} объектов, нужно {N_SLOTS}")
    elements = elements[:N_SLOTS]
    palette = parsed.get("palette") or ["#1a4d8c", "#f2c14e", "#ffffff"]
    return {
        "id": parsed.get("id") or parsed.get("variant_id") or "",
        "title": parsed.get("title") or title,
        "angle": (parsed.get("angle") or "").strip(),
        "concept": parsed.get("concept") or "",
        "palette": palette,
        "elements": elements,
    }


def _active_flat(doc: dict) -> dict:
    """План активного варианта + служебные поля (brief, variants, active)."""
    variants = doc.get("variants")
    if not isinstance(variants, list) or not variants:
        # Старый формат: один план без variants.
        return doc
    idx = int(doc.get("active") or 0)
    idx = max(0, min(idx, len(variants) - 1))
    v = variants[idx] if isinstance(variants[idx], dict) else {}
    return {
        **doc,
        "active": idx,
        "title": v.get("title") or doc.get("title") or "",
        "concept": v.get("concept") or "",
        "angle": v.get("angle") or "",
        "palette": v.get("palette") or doc.get("palette") or [],
        "elements": v.get("elements") or [],
        "variant_id": v.get("id") or f"v{idx + 1}",
    }


def _variants_public(doc: dict) -> list[dict]:
    """Карточки вариантов для UI — с текстом наполнения для переноса."""
    out = []
    for i, v in enumerate(doc.get("variants") or []):
        if not isinstance(v, dict):
            continue
        els = v.get("elements") or []
        lines: list[str] = []
        subjects: list[str] = []
        for el in els[:10]:
            if not isinstance(el, dict):
                continue
            name = (el.get("title_ru") or el.get("subject") or "").strip()
            subject = (el.get("subject") or name).strip()
            subjects.append(name or subject)
            if subject:
                lines.append(f"{len(lines) + 1}. {name or subject} | {subject}")
        filling = ("[[n=10]]\n" + "\n".join(lines)) if lines else ""
        out.append({
            "index": i,
            "id": v.get("id") or f"v{i + 1}",
            "title": v.get("title") or "",
            "angle": v.get("angle") or "",
            "concept": v.get("concept") or "",
            "palette": v.get("palette") or [],
            "n_elements": len(els),
            "subjects": subjects,
            "filling": filling,
        })
    return out


async def _invent_variants(
    reg: Registry,
    preset,
    title: str,
    description: str,
    brief: Brief | None = None,
    *,
    n_variants: int = 3,
    lang: str = "ru",
) -> dict:
    """Придумывает 1–6 вариантов набора. Возвращает plan.json-документ."""
    from app.pipeline.stages import _as_variants

    n_variants = max(1, min(6, int(n_variants or 3)))
    lang = _norm_lang(lang)
    prompt = engine.render(
        "collection_fill",
        title=title.strip(),
        description=(description or "").strip() or "(no extra description)",
        brief=brief.model_dump() if brief else None,
        n_elements=N_SLOTS,
        n_variants=n_variants,
        lang=lang,
    )
    variant = resolve_variant(preset, "concept")
    errors: list[str] = []
    last_text = ""
    for attempt in range(3):
        try:
            resp = await reg.text(variant, prompt, json_mode=True)
            last_text = resp.text or ""
            parsed = engine.parse_json_response(resp.text)
            raw = _as_variants(parsed)
            if not raw:
                raise ValueError("пустой список вариантов")
            variants: list[dict] = []
            for i, item in enumerate(raw):
                plan = _parse_fill_plan(item, title)
                plan["id"] = plan["id"] or f"v{i + 1}"
                if brief and brief.palette and not plan.get("palette"):
                    plan["palette"] = list(brief.palette)
                variants.append(plan)
            if len(variants) < n_variants:
                raise ValueError(
                    f"модель вернула {len(variants)} вариант(ов), нужно {n_variants}"
                )
            variants = variants[:n_variants]
            cost = float(getattr(resp, "cost_usd", 0) or 0)
            return {
                "title": title.strip(),
                "active": 0,
                "variants": variants,
                "cost_usd": cost,
                # Плоские поля активного (v1) — совместимость со старым UI/кодом.
                **{k: variants[0][k] for k in (
                    "concept", "palette", "elements", "angle", "id"
                ) if k in variants[0]},
            }
        except (ProviderError, ValueError, TypeError) as e:
            log.warning("Наполнение вариантов, попытка %d: %s", attempt + 1, e)
            errors.append(str(e))
    head = " ".join((last_text or "").split())[:200]
    detail = errors[-1] if errors else "неизвестная ошибка"
    raise ProviderError(
        f"не удалось придумать варианты набора — {detail}"
        + (f"; ответ: {head!r}" if head else "")
        + f". Роль concept: {variant.model_name}."
    )


# Обратная совместимость для внутренних вызовов, ожидающих один план.
async def _invent_items(
    reg: Registry,
    preset,
    title: str,
    description: str,
    brief: Brief | None = None,
    *,
    n_variants: int = 3,
    lang: str = "ru",
) -> dict:
    return await _invent_variants(
        reg, preset, title, description, brief,
        n_variants=n_variants, lang=lang,
    )


def _cards_from_plan(plan: dict) -> list[dict]:
    cards = []
    for i, el in enumerate(plan.get("elements") or []):
        slot = i + 1
        name = (el.get("title_ru") or el.get("subject") or f"Item {slot}").strip()
        subject = (el.get("subject") or name).strip()
        cards.append({
            "slot": slot,
            "name": name,
            "subject": subject,
            "category": str(el.get("category") or "1"),
            "rarity": max(1, min(5, int(el.get("rarity") or 1))),
            "file": None,
            "url": None,
        })
    while len(cards) < N_SLOTS:
        s = len(cards) + 1
        cards.append({
            "slot": s, "name": "", "subject": "", "category": "1", "rarity": 1,
            "file": None, "url": None,
        })
    return cards[:N_SLOTS]


def _load_plan(out_dir: Path) -> dict:
    path = out_dir / "plan.json"
    if not path.is_file():
        raise HTTPException(404, "План прогона не найден — сначала «Придумать объекты»")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Не прочитать plan.json: {e}") from e


def _save_meta_card(out_dir: Path, card: dict, *, plan: dict | None = None,
                    title: str = "", description: str = "",
                    preset_id: str | None = None) -> None:
    meta_path = out_dir / "meta.json"
    data: dict
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    if plan:
        data.setdefault("title", plan.get("title") or title)
        data.setdefault("concept", plan.get("concept") or "")
        data.setdefault("palette", plan.get("palette") or [])
    if title:
        data["title"] = title
    if description:
        data["description"] = description
    if preset_id:
        data["preset_id"] = preset_id
    data.setdefault("run_id", out_dir.name)
    cards = data.get("cards") or []
    for i, c in enumerate(cards):
        if c.get("slot") == card["slot"]:
            cards[i] = {**c, **card}
            break
    else:
        cards.append(card)
    data["cards"] = sorted(cards, key=lambda c: c.get("slot", 0))
    cost = sum(float(c.get("cost_usd") or 0) for c in data["cards"])
    if plan and plan.get("cost_usd"):
        cost += float(plan["cost_usd"] or 0)
    data["cost_usd"] = round(cost, 6)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _card_request_fields(meta: dict) -> dict:
    """Поля для «детального просмотра промпта» на карточке."""
    keys = (
        "prompt", "negative", "merged_extra", "extra", "trigger",
        "model", "seed", "width", "height", "subject", "category",
        "cost_usd", "elapsed_s",
    )
    return {k: meta[k] for k in keys if k in meta and meta[k] not in (None, "")}


async def _render_one(
    reg: Registry,
    preset,
    *,
    subject: str,
    category: str,
    palette: list[str],
    out_path: Path,
    seed: int | None = None,
    extra: str = "",
) -> dict:
    clean_subject = _sanitize_subject(subject) or (subject or "").strip()
    element = ElementSpec(
        slot="art",
        subject=clean_subject,
        category=str(category or "1"),
        rarity=1,
        seed=seed or random.randint(1, 2**31 - 1),
    )
    render_variant = resolve_variant(preset, "render")
    trigger = getattr(render_variant, "trigger", "") or ""
    merged_extra = _with_lora_extra(extra, trigger)
    sb = style_bible()
    prompt = build_prompt(
        element,
        palette,
        extra=merged_extra,
        trigger=trigger,
    )
    if clean_subject != (subject or "").strip():
        log.info(
            "Subject sanitized: %r → %r",
            (subject or "")[:160],
            clean_subject[:160],
        )
    if merged_extra:
        log.info("Арт с пожеланиями: slot=%s extra=%r", out_path.name, merged_extra[:240])
    # Не прикладываем Icons как input_references: Flux/Qwen копируют
    # предметы из рефов (ласты, арки…) вместо subject. Стиль — из промпта /
    # LoRA; эталоны Icons остаются в style/art-refs для обучения, не для OR.
    refs: list[bytes] = []
    started = time.monotonic()
    try:
        img = await reg.image(
            render_variant,
            prompt,
            negative=sb.get("negative", ""),
            width=1024,
            height=1024,
            seed=element.seed,
            references=refs,
        )
    except ProviderError as e:
        raise _routing_broken(preset, e) from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img.images[0])
    return {
        "model": img.model,
        "seed": element.seed,
        "prompt": prompt,
        "negative": sb.get("negative", ""),
        "width": 1024,
        "height": 1024,
        "trigger": trigger,
        "subject": subject,
        "category": str(category or "1"),
        "cost_usd": img.cost_usd,
        "elapsed_s": round(time.monotonic() - started, 2),
        "file": out_path.name,
        "extra": (extra or "").strip(),
        "merged_extra": merged_extra,
    }


@router.post("")
async def fill_collection(req: FillRequest):
    """SSE: концепты → 10 артов. События: status, item, card, done, fatal."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = FILL_RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(kind: str, payload: dict) -> None:
        queue.put_nowait(json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str))

    async def worker() -> None:
        cards: list[dict] = []
        try:
            reg = Registry(providers_config(), offline=is_offline())
            emit("status", {"message": "разбираю бриф (промпт brief)…", "run_id": run_id})
            brief = await parse_brief(
                reg, preset, title, req.description or "", lang=_norm_lang(req.lang),
            )
            brief_cost = 0.0  # parse_brief не возвращает cost — ок
            emit("status", {
                "message": (
                    f"бриф: {brief.theme}"
                    + (" · упрощённый" if brief.degraded else "")
                ),
                "run_id": run_id,
                "brief": brief.model_dump(),
            })

            n_variants = max(1, min(6, int(req.n_variants or 3)))
            emit("status", {
                "message": f"придумываю {n_variants} варианта набора (промпт collection_fill)…",
                "run_id": run_id,
            })
            plan_doc = await _invent_variants(
                reg, preset, title, req.description, brief=brief,
                n_variants=n_variants, lang=_norm_lang(req.lang),
            )
            active_idx = max(0, min(int(req.variant_index or 0), len(plan_doc["variants"]) - 1))
            plan_doc["active"] = active_idx
            plan = _active_flat(plan_doc)
            plan_cost = float(plan_doc.get("cost_usd") or 0) + brief_cost
            plan_doc["cost_usd"] = plan_cost
            if brief.palette:
                for v in plan_doc["variants"]:
                    if not v.get("palette"):
                        v["palette"] = list(brief.palette)
                if not plan.get("palette"):
                    plan["palette"] = list(brief.palette)
            (out_dir / "plan.json").write_text(
                json.dumps({**plan_doc, "brief": brief.model_dump()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            emit("variants", {
                "run_id": run_id,
                "active": active_idx,
                "variants": _variants_public(plan_doc),
                "cost_usd": plan_cost,
                "brief": brief.model_dump(),
            })
            emit("plan", {
                "run_id": run_id,
                "title": plan["title"],
                "concept": plan["concept"],
                "angle": plan.get("angle") or "",
                "palette": plan["palette"],
                "active": active_idx,
                "variants": _variants_public(plan_doc),
                "cost_usd": plan_cost,
                "brief": brief.model_dump(),
            })

            sem = asyncio.Semaphore(req.concurrency)
            palette = plan["palette"]
            total_cost = plan_cost
            art_extra_base = _brief_art_extra(brief, req.art_extra)
            emit("status", {
                "message": (
                    f"рисую вариант {active_idx + 1}/{len(plan_doc['variants'])}"
                    + (f" — {plan.get('angle')}" if plan.get("angle") else "")
                ),
                "run_id": run_id,
            })

            async def one(i: int, el: dict) -> dict:
                async with sem:
                    slot = i + 1
                    name = (el.get("title_ru") or el.get("subject") or f"Item {slot}").strip()
                    subject = (el.get("subject") or name).strip()
                    category = str(el.get("category") or "1")
                    rarity = max(1, min(5, int(el.get("rarity") or 1)))
                    emit("status", {
                        "message": f"рисую слот {slot}/10 — {name}",
                        "slot": slot,
                    })
                    path = out_dir / f"v{active_idx + 1}" / f"slot_{slot:02d}.png"
                    try:
                        meta = await _render_one(
                            reg, preset,
                            subject=subject,
                            category=category,
                            palette=palette,
                            out_path=path,
                            extra=art_extra_base,
                        )
                    except ProviderError as e:
                        emit("card_error", {"slot": slot, "message": str(e)})
                        return {
                            "slot": slot,
                            "name": name,
                            "subject": subject,
                            "category": category,
                            "rarity": rarity,
                            "error": str(e),
                            "file": None,
                            "url": None,
                        }
                    rel = f"collection-fill/{run_id}/v{active_idx + 1}/{path.name}"
                    card = {
                        "slot": slot,
                        "name": name,
                        "subject": subject,
                        "category": category,
                        "rarity": rarity,
                        "file": rel,
                        "url": f"/runs/{rel}",
                        **_card_request_fields(meta),
                    }
                    emit("card", card)
                    return card

            tasks = [one(i, el) for i, el in enumerate(plan["elements"])]
            cards = await asyncio.gather(*tasks)
            cards = sorted(cards, key=lambda c: c["slot"])
            total_cost += sum(float(c.get("cost_usd") or 0) for c in cards)
            meta = {
                "run_id": run_id,
                "title": plan["title"],
                "description": req.description,
                "concept": plan["concept"],
                "angle": plan.get("angle") or "",
                "palette": plan["palette"],
                "preset_id": preset.id,
                "active": active_idx,
                "variants": _variants_public(plan_doc),
                "cards": cards,
                "cost_usd": round(total_cost, 6),
            }
            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            ok = sum(1 for c in cards if c.get("url"))
            emit("done", {
                "run_id": run_id,
                "title": plan["title"],
                "concept": plan["concept"],
                "angle": plan.get("angle") or "",
                "palette": plan["palette"],
                "active": active_idx,
                "variants": _variants_public(plan_doc),
                "cards": cards,
                "ok": ok,
                "total": N_SLOTS,
                "cost_usd": round(total_cost, 6),
            })
        except Exception as e:  # noqa: BLE001
            log.exception("Наполнение коллекции упало")
            emit("fatal", {"message": str(e), "run_id": run_id})
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())

    async def stream():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/plan")
async def fill_plan_only(req: PlanRequest) -> dict:
    """Brief + concept: 1–6 вариантов × 10 слотов без генерации картинок."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")
    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = FILL_RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry(providers_config(), offline=is_offline())
    n_variants = max(1, min(6, int(req.n_variants or 3)))
    try:
        brief = await parse_brief(
            reg, preset, title, req.description or "", lang=_norm_lang(req.lang),
        )
        plan_doc = await _invent_variants(
            reg, preset, title, req.description, brief=brief,
            n_variants=n_variants, lang=_norm_lang(req.lang),
        )
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    if brief.palette:
        for v in plan_doc["variants"]:
            if not v.get("palette"):
                v["palette"] = list(brief.palette)
    plan_doc["active"] = 0
    plan = _active_flat(plan_doc)
    plan_cost = float(plan_doc.get("cost_usd") or 0)
    plan_doc["cost_usd"] = plan_cost
    payload = {**plan_doc, "brief": brief.model_dump(), "cost_usd": plan_cost}
    (out_dir / "plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    cards = _cards_from_plan(plan)
    variants_pub = _variants_public(plan_doc)
    meta = {
        "run_id": run_id,
        "title": plan["title"],
        "description": req.description,
        "concept": plan["concept"],
        "angle": plan.get("angle") or "",
        "palette": plan["palette"],
        "preset_id": preset.id,
        "active": 0,
        "variants": variants_pub,
        "cards": cards,
        "cost_usd": round(plan_cost, 6),
        "mode": "plan",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "title": plan["title"],
        "concept": plan["concept"],
        "angle": plan.get("angle") or "",
        "palette": plan["palette"],
        "active": 0,
        "variants": variants_pub,
        "brief": brief.model_dump(),
        "cards": cards,
        "cost_usd": round(plan_cost, 6),
    }


_DEFAULT_CATS = ["1", "1", "1", "2", "2", "2", "3", "3", "4", "4"]
_DEFAULT_RARITY = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def _is_fal_lora_only(preset) -> bool:
    """True, если render-ротация — только fal_lora (FLUX.1 Dev + LoRA)."""
    try:
        variant = resolve_variant(preset, "render")
        ids = [mp.provider_id for mp in variant.rotation()]
    except Exception:  # noqa: BLE001
        return False
    return bool(ids) and all(pid == "fal_lora" for pid in ids)


def _routing_broken(preset, err: BaseException) -> ProviderError:
    detail = str(err).strip() or type(err).__name__
    if _is_fal_lora_only(preset):
        return ProviderError(
            "Routing оборвался: не удалось обратиться к FLUX.1 Dev + LoRA (fal_lora). "
            f"Проверь FAL_KEY и fal_lora.enabled. Детали: {detail}"
        )
    return err if isinstance(err, ProviderError) else ProviderError(detail)


def _normalize_manual_elements(raw: list[ManualElementIn]) -> list[dict]:
    """Ровно N_SLOTS элементов; пустые слоты добиваем дефолтами ТЗ."""
    elements: list[dict] = []
    for i in range(N_SLOTS):
        if i >= len(raw):
            elements.append({
                "subject": "",
                "title_ru": f"Item {i + 1}",
                "category": _DEFAULT_CATS[i],
                "rarity": _DEFAULT_RARITY[i],
            })
            continue
        src = raw[i]
        subject = (src.subject or "").strip()
        name = (src.name or "").strip() or subject or f"Item {i + 1}"
        cat = str(src.category or "").strip() or _DEFAULT_CATS[i]
        if cat not in ("1", "2", "3", "4", "5"):
            cat = _DEFAULT_CATS[i]
        rarity = int(src.rarity) if src.rarity is not None else _DEFAULT_RARITY[i]
        rarity = max(1, min(5, rarity))
        elements.append({
            "subject": subject,
            "title_ru": name,
            "category": cat,
            "rarity": rarity,
        })
    return elements


@router.post("/plan/manual")
async def fill_plan_manual(req: ManualPlanRequest) -> dict:
    """Создать прогон с объектами от пользователя (без brief/concept AI)."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")
    preset_id = req.preset_id
    try:
        # Пресет нужен только чтобы UI/мета знали, чем потом рисовать.
        preset = load_preset(preset_id) if preset_id else load_preset(None)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    elements = _normalize_manual_elements(list(req.elements or []))
    filled = sum(1 for el in elements if (el.get("subject") or "").strip())
    # Пустые слоты ок: пользователь допишет subject в UI перед генерацией.
    palette = [c for c in (req.palette or []) if isinstance(c, str) and c.strip()]
    if len(palette) < 3:
        palette = ["#1a4d8c", "#f2c14e", "#ffffff"]

    concept = (req.concept or "").strip() or (req.description or "").strip()[:160] or title
    angle = (req.angle or "").strip() or "ручной список объектов"

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = FILL_RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    variant = {
        "id": "v1",
        "title": title,
        "concept": concept,
        "angle": angle,
        "palette": palette,
        "elements": elements,
    }
    plan_doc = {
        "title": title,
        "active": 0,
        "variants": [variant],
        "cost_usd": 0,
        "mode": "manual",
        "concept": concept,
        "angle": angle,
        "palette": palette,
        "elements": elements,
        "id": "v1",
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan_doc, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    cards = _cards_from_plan(plan_doc)
    variants_pub = _variants_public(plan_doc)
    meta = {
        "run_id": run_id,
        "title": title,
        "description": req.description or "",
        "concept": concept,
        "angle": angle,
        "palette": palette,
        "preset_id": preset.id,
        "active": 0,
        "variants": variants_pub,
        "cards": cards,
        "cost_usd": 0,
        "mode": "manual",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "title": title,
        "concept": concept,
        "angle": angle,
        "palette": palette,
        "active": 0,
        "variants": variants_pub,
        "cards": cards,
        "cost_usd": 0,
        "mode": "manual",
    }


@router.post("/select-variant")
async def fill_select_variant(req: SelectVariantRequest) -> dict:
    """Переключить активный вариант набора (без новой генерации объектов)."""
    run_id = _safe_run_id(req.run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")
    doc = _load_plan(out_dir)
    variants = doc.get("variants") or []
    if not variants:
        raise HTTPException(400, "В этом прогоне нет нескольких вариантов")
    idx = int(req.variant_index)
    if idx < 0 or idx >= len(variants):
        raise HTTPException(400, f"variant_index 0..{len(variants) - 1}")
    doc["active"] = idx
    plan = _active_flat(doc)
    (out_dir / "plan.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    cards = _cards_from_plan(plan)
    # Подтянуть уже нарисованные арты этого варианта, если есть.
    vdir = out_dir / f"v{idx + 1}"
    for c in cards:
        # Новый путь: vN/slot_XX.png; старый — slot_XX.png только для active==0.
        cand = [
            vdir / f"slot_{c['slot']:02d}.png",
            out_dir / f"slot_{c['slot']:02d}.png" if idx == 0 else None,
        ]
        for path in cand:
            if path and path.is_file():
                rel = f"collection-fill/{run_id}/{path.relative_to(out_dir).as_posix()}"
                c["file"] = rel
                c["url"] = f"/runs/{rel}"
                break
    meta_path = out_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta.update({
        "run_id": run_id,
        "title": plan["title"],
        "concept": plan["concept"],
        "angle": plan.get("angle") or "",
        "palette": plan["palette"],
        "active": idx,
        "variants": _variants_public(doc),
        "cards": cards,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "active": idx,
        "title": plan["title"],
        "concept": plan["concept"],
        "angle": plan.get("angle") or "",
        "palette": plan["palette"],
        "variants": _variants_public(doc),
        "cards": cards,
    }


@router.post("/one")
async def fill_one_card(req: OneCardRequest) -> dict:
    """Сгенерировать 1–4 арта одного слота по одному тексту (разные seed)."""
    run_id = _safe_run_id(req.run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    plan = _active_flat(_load_plan(out_dir))
    elements = plan.get("elements") or []
    if req.slot > len(elements):
        raise HTTPException(400, f"В плане нет слота {req.slot}")
    el = elements[req.slot - 1] or {}

    name = (req.name or el.get("title_ru") or el.get("subject") or f"Item {req.slot}").strip()
    subject = (req.subject or el.get("subject") or name).strip()
    if not subject:
        raise HTTPException(400, "Нужен subject для слота")
    category = str(req.category or el.get("category") or "1")
    rarity = req.rarity if req.rarity is not None else max(1, min(5, int(el.get("rarity") or 1)))
    palette = plan.get("palette") or ["#1a4d8c", "#f2c14e", "#ffffff"]
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname
    vdir.mkdir(parents=True, exist_ok=True)
    canonical = vdir / f"slot_{req.slot:02d}.png"

    brief = None
    raw_brief = plan.get("brief")
    if isinstance(raw_brief, dict):
        try:
            brief = Brief(**{k: v for k, v in raw_brief.items() if k in Brief.model_fields})
        except Exception:  # noqa: BLE001
            brief = None
    extra = _brief_art_extra(brief, req.art_extra)
    n_images = max(1, min(4, int(req.n_images or 1)))

    reg = Registry(providers_config(), offline=is_offline())

    async def _one_img(i: int) -> dict:
        stamp = int(time.time() * 1000) + i
        path = vdir / f"slot_{req.slot:02d}_v{stamp}.png"
        seed = random.randint(1, 2**31 - 1)
        meta = await _render_one(
            reg, preset,
            subject=subject,
            category=category,
            palette=palette,
            out_path=path,
            seed=seed,
            extra=extra,
        )
        rel = f"collection-fill/{run_id}/{vname}/{path.name}"
        return {
            "file": rel,
            "url": f"/runs/{rel}?t={stamp}",
            "version": stamp,
            "seed": meta.get("seed", seed),
            "prompt": meta.get("prompt"),
            "negative": meta.get("negative"),
            "model": meta.get("model"),
            "cost_usd": float(meta.get("cost_usd") or 0),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "trigger": meta.get("trigger"),
            "merged_extra": meta.get("merged_extra"),
        }

    try:
        versions = list(await asyncio.gather(*[_one_img(i) for i in range(n_images)]))
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    primary = versions[0]
    try:
        primary_path = RUNS_DIR / Path(primary["file"])
        if primary_path.is_file():
            canonical.write_bytes(primary_path.read_bytes())
    except OSError:
        pass

    total_cost = sum(float(v.get("cost_usd") or 0) for v in versions)
    card = {
        "slot": req.slot,
        "name": name,
        "subject": subject,
        "category": category,
        "rarity": rarity,
        "file": primary["file"],
        "url": primary["url"],
        "version": primary["version"],
        "versions": versions,
        "extra": (req.art_extra or "").strip(),
        "prompt": primary.get("prompt"),
        "negative": primary.get("negative"),
        "model": primary.get("model"),
        "seed": primary.get("seed"),
        "width": primary.get("width"),
        "height": primary.get("height"),
        "trigger": primary.get("trigger"),
        "merged_extra": primary.get("merged_extra"),
        "cost_usd": round(total_cost, 6),
    }
    _save_meta_card(
        out_dir, card, plan=plan, preset_id=preset.id,
    )
    return card


@router.post("/hard-reset")
async def hard_reset_card(req: HardResetRequest) -> dict:
    """Hard reset слота: удалить PNG/версии, очистить мету генерации.

    Subject / name / category / rarity сохраняются.
    Следующая генерация идёт с пустым extra и без привязки к старым файлам.
    (В LoRA-пайплайне предыдущие PNG и так не шлются как ref — refs=[].)
    """
    run_id = _safe_run_id(req.run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")

    plan_doc = _load_plan(out_dir)
    plan = _active_flat(plan_doc)
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname

    removed: list[str] = []
    if vdir.is_dir():
        for p in sorted(vdir.glob(f"slot_{req.slot:02d}*")):
            if p.is_file():
                try:
                    p.unlink()
                    removed.append(p.name)
                except OSError:
                    pass

    meta_path = out_dir / "meta.json"
    doc: dict = {}
    if meta_path.is_file():
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = {}
    cards = doc.setdefault("cards", [])
    while len(cards) < N_SLOTS:
        cards.append({})
    idx = req.slot - 1
    prev = cards[idx] if isinstance(cards[idx], dict) else {}
    name = (req.name or "").strip() or prev.get("name") or f"Slot {req.slot}"
    subject = (req.subject if req.subject is not None else prev.get("subject") or "")
    if isinstance(subject, str):
        subject = subject.strip()
    else:
        subject = ""
    category = int(req.category if req.category is not None else prev.get("category") or 1)
    rarity = int(req.rarity if req.rarity is not None else prev.get("rarity") or 1)
    kept = {
        "slot": req.slot,
        "name": name,
        "subject": subject,
        "category": max(1, min(5, category)),
        "rarity": max(1, min(5, rarity)),
        "extra": "",
        "hard_reset_at": int(time.time()),
        "hard_reset_removed": removed,
    }
    cards[idx] = kept
    doc["cards"] = cards
    doc["updated_at"] = int(time.time())
    meta_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    return {
        "ok": True,
        "run_id": run_id,
        "slot": req.slot,
        "variant": vname,
        "removed": removed,
        "card": {
            **kept,
            "url": None,
            "file": None,
            "prompt": None,
            "seed": None,
            "model": None,
            "error": None,
            "upscaled": False,
        },
    }


@router.post("/regenerate")
async def regenerate_card(req: RegenRequest) -> dict:
    """Перегенерировать арт одного слота в уже существующем прогоне."""
    run_id = _safe_run_id(req.run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    palette = req.palette or ["#1a4d8c", "#f2c14e", "#ffffff"]
    subject = (req.subject or "").strip()
    if not subject:
        raise HTTPException(400, "Нужен subject для перегенерации")

    plan_doc = _load_plan(out_dir)
    plan = _active_flat(plan_doc)
    if not req.palette and plan.get("palette"):
        palette = plan["palette"]
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname
    vdir.mkdir(parents=True, exist_ok=True)
    canonical = vdir / f"slot_{req.slot:02d}.png"
    n_images = max(1, min(4, int(req.n_images or 1)))
    extra = _brief_art_extra(None, req.extra)
    category = str(req.category or "1")
    reg = Registry(providers_config(), offline=is_offline())

    async def _one_img(i: int) -> dict:
        stamp = int(time.time() * 1000) + i
        path = vdir / f"slot_{req.slot:02d}_v{stamp}.png"
        seed = random.randint(1, 2**31 - 1)
        meta = await _render_one(
            reg, preset,
            subject=subject,
            category=category,
            palette=palette,
            out_path=path,
            seed=seed,
            extra=extra,
        )
        rel = f"collection-fill/{run_id}/{vname}/{path.name}"
        return {
            "file": rel,
            "url": f"/runs/{rel}?t={stamp}",
            "version": stamp,
            "seed": meta.get("seed", seed),
            "prompt": meta.get("prompt"),
            "negative": meta.get("negative"),
            "model": meta.get("model"),
            "cost_usd": float(meta.get("cost_usd") or 0),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "trigger": meta.get("trigger"),
            "merged_extra": meta.get("merged_extra"),
        }

    try:
        versions = list(await asyncio.gather(*[_one_img(i) for i in range(n_images)]))
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    primary = versions[0]
    try:
        primary_path = RUNS_DIR / Path(primary["file"])
        if primary_path.is_file():
            canonical.write_bytes(primary_path.read_bytes())
    except OSError:
        pass

    total_cost = sum(float(v.get("cost_usd") or 0) for v in versions)
    card = {
        "slot": req.slot,
        "name": (req.name or subject).strip(),
        "subject": subject,
        "category": category,
        "rarity": req.rarity,
        "file": primary["file"],
        "url": primary["url"],
        "version": primary["version"],
        "versions": versions,
        "extra": (req.extra or "").strip(),
        "prompt": primary.get("prompt"),
        "negative": primary.get("negative"),
        "model": primary.get("model"),
        "seed": primary.get("seed"),
        "width": primary.get("width"),
        "height": primary.get("height"),
        "trigger": primary.get("trigger"),
        "merged_extra": primary.get("merged_extra"),
        "cost_usd": round(total_cost, 6),
    }

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            cards = data.get("cards") or []
            for i, c in enumerate(cards):
                if c.get("slot") == req.slot:
                    cards[i] = {**c, **card}
                    break
            else:
                cards.append(card)
            data["cards"] = sorted(cards, key=lambda c: c.get("slot", 0))
            meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass

    return card


def _resolve_slot_image(out_dir: Path, slot: int, meta_card: dict | None = None) -> Path:
    """Находит актуальный PNG слота (versioned file или canonical)."""
    candidates: list[Path] = []
    if meta_card and meta_card.get("file"):
        rel = str(meta_card["file"]).replace("\\", "/")
        if rel.startswith("collection-fill/"):
            rel = rel.split("/", 1)[1]
        parts = rel.split("/")
        if parts and parts[0] == out_dir.name:
            candidates.append(out_dir.joinpath(*parts[1:]))
        else:
            candidates.append(FILL_RUNS.joinpath(*parts))
            if len(parts) > 1:
                candidates.append(out_dir.joinpath(*parts[1:]))

    plan: dict = {}
    plan_path = out_dir / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plan = {}
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname
    if vdir.is_dir():
        vers = sorted(
            vdir.glob(f"slot_{slot:02d}_v*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(vers[:1])
        ups = sorted(
            vdir.glob(f"slot_{slot:02d}_up*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Берём исходник до апскейла: versioned/canonical, не предыдущий up.
        candidates.extend([
            vdir / f"slot_{slot:02d}.png",
            out_dir / f"slot_{slot:02d}.png",
        ])
        # Если есть только up — как fallback в конце.
        candidates.extend(ups[:1])
    else:
        candidates.append(out_dir / f"slot_{slot:02d}.png")

    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    raise HTTPException(404, f"Нет картинки для слота {slot}")


async def _fal_esrgan_upscale(image_bytes: bytes, *, scale: float = 2.0) -> tuple[bytes, float]:
    """Апскейл через fal-ai/esrgan. Возвращает (png_bytes, cost_usd)."""
    key = app_settings.resolve_key("fal_lora", "FAL_KEY")
    if not key:
        raise HTTPException(
            400,
            "Нет FAL_KEY — сохрани ключ на экране LoRA Trainer",
        )
    if is_offline():
        raise HTTPException(503, "Оффлайн-режим: апскейл недоступен")

    data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    # scale должен быть в допустимом диапазоне fal (1..8).
    scale = max(1.0, min(8.0, float(scale)))
    payload = {
        "image_url": data_uri,
        "scale": scale,
        "model": "RealESRGAN_x4plus",
        "output_format": "png",
    }
    cost = 0.008
    url = "https://fal.run/fal-ai/esrgan"
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Key {key}",
                    "Content-Type": "application/json",
                },
            )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Routing оборвался (fal upscale): {e}") from e

    if r.status_code >= 400:
        raise HTTPException(
            502,
            f"Routing оборвался (fal upscale HTTP {r.status_code}): {r.text[:300]}",
        )
    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"fal upscale: ответ не JSON — {e}") from e

    img_url = None
    if isinstance(data.get("image"), dict):
        img_url = data["image"].get("url")
    if not img_url and isinstance(data.get("images"), list) and data["images"]:
        first = data["images"][0]
        img_url = first.get("url") if isinstance(first, dict) else first
    if not img_url:
        raise HTTPException(502, "fal upscale: в ответе нет image.url")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            ir = await client.get(img_url)
            ir.raise_for_status()
            out = ir.content
    except httpx.HTTPError as e:
        raise HTTPException(502, f"fal upscale: не скачать результат — {e}") from e

    try:
        from app.billing import record_spend
        record_spend(cost, "fal_upscale")
    except Exception:  # noqa: BLE001
        pass
    return out, cost


UPSCALE_TARGET_PX = 2048


def _image_size(png_bytes: bytes) -> tuple[int, int]:
    from PIL import Image
    import io as _io
    with Image.open(_io.BytesIO(png_bytes)) as im:
        return int(im.width), int(im.height)


def _force_square_png(png_bytes: bytes, size: int = UPSCALE_TARGET_PX) -> bytes:
    """Гарантирует ровно size×size PNG (LANCZOS)."""
    from PIL import Image
    import io as _io
    with Image.open(_io.BytesIO(png_bytes)) as im:
        im = im.convert("RGBA")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _resolve_upscale_source(out_dir: Path, slot: int, meta_card: dict | None = None) -> Path:
    """Для апскейла берём базовый арт (не предыдущий *_up*), чтобы всегда 1024→2048."""
    plan: dict = {}
    plan_path = out_dir / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plan = {}
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname
    candidates: list[Path] = []
    if vdir.is_dir():
        vers = sorted(
            vdir.glob(f"slot_{slot:02d}_v*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(vers[:1])
        candidates.append(vdir / f"slot_{slot:02d}.png")
    candidates.append(out_dir / f"slot_{slot:02d}.png")
    # meta.file только если это не предыдущий апскейл
    if meta_card and meta_card.get("file"):
        rel = str(meta_card["file"]).replace("\\", "/")
        if "_up" not in Path(rel).name:
            if rel.startswith("collection-fill/"):
                rel = rel.split("/", 1)[1]
            parts = rel.split("/")
            if parts and parts[0] == out_dir.name:
                candidates.append(out_dir.joinpath(*parts[1:]))
            else:
                candidates.append(FILL_RUNS.joinpath(*parts))
    for path in candidates:
        if path.is_file():
            return path
    # fallback: любой актуальный файл слота (в т.ч. up)
    return _resolve_slot_image(out_dir, slot, meta_card)


@router.post("/upscale")
async def upscale_card(req: UpscaleRequest) -> dict:
    """Улучшить визуал: Real-ESRGAN → всегда 2048×2048."""
    run_id = _safe_run_id(req.run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")

    meta_path = out_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    cards = meta.get("cards") or []
    card_prev = next((c for c in cards if c.get("slot") == req.slot), None) or {}

    src = _resolve_upscale_source(out_dir, req.slot, card_prev)
    raw = src.read_bytes()
    try:
        w0, h0 = _image_size(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Не прочитать исходник: {e}") from e
    side = max(w0, h0) or 1024
    # Целим 2048: 1024→×2, уже 2048→×1 (только enhance/нормализация), меньше→больше.
    target = UPSCALE_TARGET_PX
    scale = max(1.0, min(8.0, target / float(side)))
    # Если клиент явно просил другой scale — игнорируем в пользу фиксированного 2048.
    started = time.monotonic()
    try:
        if abs(scale - 1.0) < 0.05 and w0 == target and h0 == target:
            # Уже 2048: лёгкий pass ×1.25 → downsample обратно к 2048 (свежий enhance).
            up_bytes, cost = await _fal_esrgan_upscale(raw, scale=1.25)
        else:
            up_bytes, cost = await _fal_esrgan_upscale(raw, scale=scale)
        up_bytes = _force_square_png(up_bytes, target)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Routing оборвался (upscale): {e}") from e

    plan: dict = {}
    plan_path = out_dir / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plan = {}
    active = int(plan.get("active") or 0)
    vname = f"v{active + 1}"
    vdir = out_dir / vname
    vdir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    out_path = vdir / f"slot_{req.slot:02d}_up{stamp}.png"
    canonical = vdir / f"slot_{req.slot:02d}.png"
    out_path.write_bytes(up_bytes)
    try:
        canonical.write_bytes(up_bytes)
    except OSError:
        pass

    rel = f"collection-fill/{run_id}/{vname}/{out_path.name}"
    card = {
        **card_prev,
        "slot": req.slot,
        "file": rel,
        "url": f"/runs/{rel}?t={stamp}",
        "version": stamp,
        "upscaled": True,
        "upscale_scale": round(target / float(side), 3),
        "upscale_model": "fal-ai/esrgan",
        "width": target,
        "height": target,
        "model": (card_prev.get("model") or "") + "+esrgan",
        "cost_usd": cost,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    updated = False
    for i, c in enumerate(cards):
        if c.get("slot") == req.slot:
            cards[i] = {**c, **card}
            updated = True
            break
    if not updated:
        cards.append(card)
    meta["cards"] = sorted(cards, key=lambda c: c.get("slot", 0))
    meta["run_id"] = run_id
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


def _safe_export_stem(name: str, slot: int) -> str:
    raw = (name or "").strip() or f"slot_{slot:02d}"
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    cleaned = cleaned[:80] or f"slot_{slot:02d}"
    return cleaned


def _export_filename(slot: int, name: str | None = None) -> str:
    return f"{slot:02d}_{_safe_export_stem(name or '', slot)}.png"


def _meta_cards(out_dir: Path) -> list[dict]:
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        return []
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    cards = data.get("cards") or []
    return [c for c in cards if isinstance(c, dict)]


@router.get("/export/slot/{slot}")
async def export_slot_image(
    slot: int,
    run_id: str = Query(...),
) -> FileResponse:
    """Скачать PNG одного слота с понятным именем файла."""
    if slot < 1 or slot > N_SLOTS:
        raise HTTPException(400, f"slot должен быть 1…{N_SLOTS}")
    run_id = _safe_run_id(run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")
    cards = _meta_cards(out_dir)
    card = next((c for c in cards if int(c.get("slot") or 0) == slot), {}) or {}
    src = _resolve_slot_image(out_dir, slot, card)
    fname = _export_filename(slot, str(card.get("name") or ""))
    return FileResponse(
        src,
        media_type="image/png",
        filename=fname,
        content_disposition_type="attachment",
    )


@router.get("/export/pack")
async def export_pack_zip(run_id: str = Query(...)) -> StreamingResponse:
    """ZIP всех доступных PNG активного варианта прогона."""
    run_id = _safe_run_id(run_id)
    out_dir = FILL_RUNS / run_id
    if not out_dir.is_dir():
        raise HTTPException(404, f"Прогон {run_id} не найден")

    cards = _meta_cards(out_dir)
    by_slot = {int(c.get("slot") or 0): c for c in cards}
    buf = io.BytesIO()
    written = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for slot in range(1, N_SLOTS + 1):
            card = by_slot.get(slot) or {}
            try:
                src = _resolve_slot_image(out_dir, slot, card)
            except HTTPException:
                continue
            fname = _export_filename(slot, str(card.get("name") or ""))
            zf.writestr(fname, src.read_bytes())
            written += 1
    if written == 0:
        raise HTTPException(404, "Нет картинок для выгрузки")

    buf.seek(0)
    zip_name = f"{run_id}_pack.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_name}"',
        },
    )
