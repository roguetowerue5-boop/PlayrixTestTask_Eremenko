"""Сборка экрана из готового арта и вырезанных UI-элементов.

Генерация и сборка разведены намеренно. Картинки стоят денег и минут,
сборка — миллисекунды и ноль. Поэтому подобрать скин, поменять шаблон или
переверстать экран можно сколько угодно раз на уже оплаченном арте.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import parts as parts_lib
from app.config import RUNS_DIR, list_templates, load_template
from app.models import ElementSpec, OfferPlan
from app.pipeline.compose import Compositor, contact_sheet

log = logging.getLogger("offerforge.assemble")
router = APIRouter(prefix="/api/assemble", tags=["assemble"])


def _load_run(run_id: str) -> dict:
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(400, "Недопустимый идентификатор прогона")
    path = RUNS_DIR / run_id / "run_report.json"
    if not path.exists():
        raise HTTPException(404, f"Прогон {run_id} не найден")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/sources")
async def sources() -> dict:
    """Что доступно для сборки: прогоны с артом, шаблоны, версии элементов."""
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            report = d / "run_report.json"
            if not (d.is_dir() and report.exists()):
                continue
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            offers = data.get("offers") or []
            art_count = sum(len(o.get("assets") or []) for o in offers)
            if not art_count:
                continue
            runs.append({
                "run_id": data.get("run_id", d.name),
                "theme": (data.get("brief") or {}).get("theme", ""),
                "template_id": data.get("template_id", ""),
                "variants": [
                    {"variant_id": o["plan"]["variant_id"],
                     "title": o["plan"].get("title", ""),
                     "elements": len(o["plan"].get("elements") or []),
                     "assets": len(o.get("assets") or [])}
                    for o in offers
                ],
                "images": art_count,
            })
    return {
        "runs": runs[:50],
        "templates": list_templates(),
        "parts": parts_lib.list_kinds(),
        "defaults": parts_lib.defaults(),
    }


class AssembleRequest(BaseModel):
    run_id: str
    variant_id: str | None = None       # None — собрать все варианты прогона
    template_id: str = "offer_screen"
    parts: dict[str, str] | None = None
    set_title: str | None = None


@router.post("")
async def assemble(req: AssembleRequest) -> dict:
    """Пересобирает экран из уже сгенерированного арта.

    Новых запросов к моделям не делается вообще — только композитинг.
    """
    data = _load_run(req.run_id)
    try:
        template = load_template(req.template_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    part_versions = {**parts_lib.defaults(), **(req.parts or {})}
    compositor = Compositor(template, part_versions=part_versions)
    run_dir = RUNS_DIR / req.run_id
    out_dir = run_dir / "assembled"
    out_dir.mkdir(parents=True, exist_ok=True)

    offers = data.get("offers") or []
    if req.variant_id:
        offers = [o for o in offers if o["plan"]["variant_id"] == req.variant_id]
        if not offers:
            raise HTTPException(404, f"Вариант {req.variant_id} не найден в прогоне")

    results, images, missing_total = [], [], 0
    for offer in offers:
        raw_plan = offer["plan"]
        plan = OfferPlan(
            variant_id=raw_plan["variant_id"],
            title=raw_plan.get("title", ""),
            concept=raw_plan.get("concept", ""),
            palette=raw_plan.get("palette") or [],
            texts={**(raw_plan.get("texts") or {}),
                   **({"set_title": req.set_title} if req.set_title else {})},
            elements=[ElementSpec(**e) for e in raw_plan.get("elements", [])],
            parts=part_versions,
        )

        # Арт берём с диска: он уже оплачен и лежит в прогоне.
        art, missing = {}, 0
        gen_layer = template.art_layer()
        layer_id = gen_layer.id if gen_layer else "art"
        for asset in offer.get("assets") or []:
            path = Path(asset["path"])
            if not path.exists():
                missing += 1
                continue
            # Слот в прогоне назывался по шаблону того прогона ("cards#3"),
            # а собираем мы, возможно, другим шаблоном — переносим по индексу.
            slot = asset["slot"]
            idx = slot.split("#")[-1] if "#" in slot else None
            key = f"{layer_id}#{idx}" if idx is not None else layer_id
            art[key] = path.read_bytes()

        missing_total += missing
        composite = compositor.render(plan, art)
        out_path = out_dir / f"{plan.variant_id}_{req.template_id}.png"
        composite.save(out_path)
        images.append(composite)
        results.append({
            "variant_id": plan.variant_id,
            "file": f"assembled/{out_path.name}",
            "elements": len(plan.elements),
            "art_used": len(art),
            "art_missing": missing,
        })

    sheet_name = None
    if len(images) > 1:
        sheet = contact_sheet(images, cols=2)
        sheet_path = out_dir / f"contact_{req.template_id}.png"
        sheet.save(sheet_path)
        sheet_name = f"assembled/{sheet_path.name}"

    return {
        "run_id": req.run_id,
        "template_id": req.template_id,
        "parts": part_versions,
        "results": results,
        "contact_sheet": sheet_name,
        "art_missing": missing_total,
    }
