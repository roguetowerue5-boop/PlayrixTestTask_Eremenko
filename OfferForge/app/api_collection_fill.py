"""Наполнение коллекции: концепты + арт через OpenRouter.

Экран «Создание наполнения коллекции»: название + описание → 10 карточек
(роль concept + render из пресета, промпты collection_fill.j2 и art.j2).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
from app.models import Brief, ElementSpec
from app.prompts import engine
from app.providers.base import ProviderError
from app.providers.registry import Registry
from app.pipeline.stages import build_prompt, parse_brief

log = logging.getLogger("offerforge.collection_fill")
router = APIRouter(prefix="/api/collection/fill", tags=["collection-fill"])

FILL_RUNS = RUNS_DIR / "collection-fill"
N_SLOTS = 10

# Жёсткий стиль для пресетов с trigger (LoRA) — тот же STYLE LOCK, что в ComfyUI.
# Не форсируем «plain background / no full scene»: фон задаёт category_rule.
# Fallback, если в style_bible нет art_extra_default / style_lock.
LORA_STYLE_EXTRA = (
    "Playrix-like casual mobile game icon, glossy stylized 3D render, "
    "single hero object centered in frame filling most of the image, "
    "smooth rounded chunky shapes with soft beveled edges, "
    "high-gloss plastic and clay materials with crisp specular highlights, "
    "no surface noise, hyper-saturated candy colors, "
    "bright soft studio lighting with key light from above-left, "
    "gentle ambient fill, soft contact shadow when grounded, "
    "clean polished match-3 collectible icon, high-contrast clear composition, "
    "no text, no logo. "
    "Background and staging MUST follow the card composition category "
    "(1 floating / 2 simple surface / 3 realistic surface / 4 full environment) — "
    "do NOT force a blank studio void on every card. "
    "Avoid: photorealistic, photograph, cinematic still, film grain, muted colors, "
    "anime, flat vector, text, logo, watermark, UI, multiple competing objects."
)

_STYLE_MARKER = "playrix-like casual mobile game icon"


def _art_extra_default() -> str:
    sb = style_bible()
    return (
        (sb.get("art_extra_default") or sb.get("style_lock") or LORA_STYLE_EXTRA)
        or ""
    ).strip()


def _with_lora_extra(extra: str, trigger: str) -> str:
    """Для LoRA-пресетов гарантируем Style A lock; без дубля, если уже в extra."""
    text = (extra or "").strip()
    if not (trigger or "").strip():
        return text
    if _STYLE_MARKER in text.lower():
        return text
    lock = _art_extra_default()
    if not lock:
        return text
    return f"{lock}\n\n{text}".strip() if text else lock


def _brief_art_extra(brief: Brief | None, art_extra: str = "") -> str:
    """Склеивает доп. к арту + mood/avoid из brief. Пустое → Comfy Style A default."""
    bits: list[str] = []
    text = (art_extra or "").strip()
    if not text:
        text = _art_extra_default()
    if text:
        bits.append(text)
    if brief:
        if brief.mood:
            bits.append("Mood: " + ", ".join(brief.mood))
        if brief.must_avoid:
            bits.append("Avoid depicting: " + ", ".join(brief.must_avoid))
        if brief.era:
            bits.append(f"Era: {brief.era}")
    return "\n".join(bits)


class FillRequest(BaseModel):
    title: str
    description: str = ""
    art_extra: str = ""  # доп. указания ко всем артам (уходит в art.j2 как extra)
    preset_id: str | None = None
    concurrency: int = Field(3, ge=1, le=8)
    n_variants: int = Field(3, ge=3, le=5)
    # Какой вариант рисовать после придумывания (0-based).
    variant_index: int = Field(0, ge=0, le=4)


class PlanRequest(BaseModel):
    """Только brief + concept: 3–5 вариантов × 10 объектов без картинок."""
    title: str
    description: str = ""
    preset_id: str | None = None
    n_variants: int = Field(3, ge=3, le=5)


class SelectVariantRequest(BaseModel):
    run_id: str
    variant_index: int = Field(0, ge=0, le=4)


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


class SuggestRequest(BaseModel):
    """🎲 — сгенерировать текст для поля описания или доп. к арту."""
    kind: str  # "description" | "art_extra"
    title: str
    description: str = ""  # для art_extra — контекст набора
    hint: str = ""         # текущий текст поля (переписать)
    preset_id: str | None = None


def _safe_run_id(run_id: str) -> str:
    if "/" in run_id or "\\" in run_id or ".." in run_id or not run_id:
        raise HTTPException(400, "Недопустимый run_id")
    return run_id


@router.get("/meta")
async def fill_meta() -> dict:
    return {
        "slots": N_SLOTS,
        "variants_min": 3,
        "variants_max": 5,
        "presets": list_presets(),
        "prompts": [
            "brief", "collection_fill", "art",
            "suggest_description", "suggest_art_extra",
        ],
        "offline": is_offline(),
    }


@router.post("/suggest")
async def fill_suggest(req: SuggestRequest) -> dict:
    """Сгенерировать описание коллекции или доп. указания к арту."""
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(400, "Укажи название коллекции")
    kind = (req.kind or "").strip().lower()
    if kind not in ("description", "art_extra"):
        raise HTTPException(400, "kind: description или art_extra")

    try:
        preset = load_preset(req.preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e

    if kind == "description":
        prompt = engine.render(
            "suggest_description",
            title=title,
            hint=(req.hint or "").strip(),
        )
    else:
        prompt = engine.render(
            "suggest_art_extra",
            title=title,
            description=(req.description or "").strip(),
            hint=(req.hint or "").strip(),
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
    if not text:
        raise HTTPException(502, "Модель вернула пустой текст")

    return {
        "kind": kind,
        "text": text,
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
    """Краткие карточки вариантов для UI (без тяжёлых полей)."""
    out = []
    for i, v in enumerate(doc.get("variants") or []):
        if not isinstance(v, dict):
            continue
        els = v.get("elements") or []
        out.append({
            "index": i,
            "id": v.get("id") or f"v{i + 1}",
            "title": v.get("title") or "",
            "angle": v.get("angle") or "",
            "concept": v.get("concept") or "",
            "palette": v.get("palette") or [],
            "n_elements": len(els),
            "subjects": [
                (el.get("title_ru") or el.get("subject") or "")
                for el in els[:10]
            ],
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
) -> dict:
    """Придумывает 3–5 вариантов набора. Возвращает plan.json-документ."""
    from app.pipeline.stages import _as_variants

    n_variants = max(3, min(5, int(n_variants or 3)))
    prompt = engine.render(
        "collection_fill",
        title=title.strip(),
        description=(description or "").strip() or "(без дополнительного описания)",
        brief=brief.model_dump() if brief else None,
        n_elements=N_SLOTS,
        n_variants=n_variants,
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
) -> dict:
    return await _invent_variants(
        reg, preset, title, description, brief, n_variants=n_variants,
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
    element = ElementSpec(
        slot="art",
        subject=subject,
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
    if merged_extra:
        log.info("Арт с пожеланиями: slot=%s extra=%r", out_path.name, merged_extra[:240])
    # Не прикладываем Icons как input_references: Flux/Qwen копируют
    # предметы из рефов (ласты, арки…) вместо subject. Стиль — из промпта /
    # LoRA; эталоны Icons остаются в style/art-refs для обучения, не для OR.
    refs: list[bytes] = []
    started = time.monotonic()
    img = await reg.image(
        render_variant,
        prompt,
        negative=sb.get("negative", ""),
        width=1024,
        height=1024,
        seed=element.seed,
        references=refs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(img.images[0])
    return {
        "model": img.model,
        "seed": element.seed,
        "prompt": prompt,
        "cost_usd": img.cost_usd,
        "elapsed_s": round(time.monotonic() - started, 2),
        "file": out_path.name,
        "extra": (extra or "").strip(),
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
            brief = await parse_brief(reg, preset, title, req.description or "")
            brief_cost = 0.0  # parse_brief не возвращает cost — ок
            emit("status", {
                "message": (
                    f"бриф: {brief.theme}"
                    + (" · упрощённый" if brief.degraded else "")
                ),
                "run_id": run_id,
                "brief": brief.model_dump(),
            })

            n_variants = max(3, min(5, int(req.n_variants or 3)))
            emit("status", {
                "message": f"придумываю {n_variants} варианта набора (промпт collection_fill)…",
                "run_id": run_id,
            })
            plan_doc = await _invent_variants(
                reg, preset, title, req.description, brief=brief,
                n_variants=n_variants,
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
                        "prompt": meta.get("prompt", ""),
                        **{k: meta[k] for k in ("model", "seed", "cost_usd", "elapsed_s")},
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
    """Brief + concept: 3–5 вариантов × 10 слотов без генерации картинок."""
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
    n_variants = max(3, min(5, int(req.n_variants or 3)))
    try:
        brief = await parse_brief(reg, preset, title, req.description or "")
        plan_doc = await _invent_variants(
            reg, preset, title, req.description, brief=brief, n_variants=n_variants,
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
    """Сгенерировать арт одного слота по плану прогона."""
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
    path = out_dir / vname / f"slot_{req.slot:02d}.png"

    brief = None
    raw_brief = plan.get("brief")
    if isinstance(raw_brief, dict):
        try:
            brief = Brief(**{k: v for k, v in raw_brief.items() if k in Brief.model_fields})
        except Exception:  # noqa: BLE001
            brief = None
    extra = _brief_art_extra(brief, req.art_extra)

    stamp = int(time.time())
    vdir = out_dir / vname
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / f"slot_{req.slot:02d}_v{stamp}.png"
    canonical = vdir / f"slot_{req.slot:02d}.png"
    reg = Registry(providers_config(), offline=is_offline())
    try:
        meta = await _render_one(
            reg, preset,
            subject=subject,
            category=category,
            palette=palette,
            out_path=path,
            seed=random.randint(1, 2**31 - 1),
            extra=extra,
        )
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    try:
        canonical.write_bytes(path.read_bytes())
    except OSError:
        pass

    rel = f"collection-fill/{run_id}/{vname}/{path.name}"
    card = {
        "slot": req.slot,
        "name": name,
        "subject": subject,
        "category": category,
        "rarity": rarity,
        "file": rel,
        "url": f"/runs/{rel}?t={stamp}",
        "version": stamp,
        "prompt": meta.get("prompt", ""),
        "extra": (req.art_extra or "").strip(),
        **{k: meta[k] for k in ("model", "seed", "cost_usd", "elapsed_s") if k in meta},
    }
    _save_meta_card(
        out_dir, card, plan=plan, preset_id=preset.id,
    )
    return card


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

    # Новая версия файла — иначе «Перенести в сборку» и кеш браузера
    # цепляются к slot_XX.png первой генерации.
    stamp = int(time.time())
    path = vdir / f"slot_{req.slot:02d}_v{stamp}.png"
    canonical = vdir / f"slot_{req.slot:02d}.png"
    reg = Registry(providers_config(), offline=is_offline())
    try:
        meta = await _render_one(
            reg, preset,
            subject=subject,
            category=req.category,
            palette=palette,
            out_path=path,
            seed=random.randint(1, 2**31 - 1),
            extra=_brief_art_extra(None, req.extra),
        )
    except ProviderError as e:
        raise HTTPException(502, str(e)) from e

    try:
        canonical.write_bytes(path.read_bytes())
    except OSError:
        pass

    rel = f"collection-fill/{run_id}/{vname}/{path.name}"
    card = {
        "slot": req.slot,
        "name": (req.name or subject).strip(),
        "subject": subject,
        "category": str(req.category or "1"),
        "rarity": req.rarity,
        "file": rel,
        "url": f"/runs/{rel}?t={stamp}",
        "version": stamp,
        "prompt": meta.get("prompt", ""),
        "extra": (req.extra or "").strip(),
        **{k: meta[k] for k in ("model", "seed", "cost_usd", "elapsed_s") if k in meta},
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
