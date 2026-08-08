"""Оркестратор прогона.

Один вызов run_offer_generation() проходит все этапы и складывает в
runs/<run_id>/ всё, что нужно для проверки: бриф, концепты, промпты,
сгенерированные и отбракованные картинки, вердикты QC, композиты и отчёт.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import (
    RUNS_DIR,
    is_offline,
    load_preset,
    load_template,
    providers_config,
)
from app import parts as parts_lib
from app.models import Brief, OfferResult, RunReport
from app.pipeline import stages
from app.pipeline.compose import Compositor, contact_sheet
from app.providers.registry import Registry

log = logging.getLogger("offerforge.run")

EventFn = Callable[[str, dict], None]


def _noop(kind: str, payload: dict) -> None:  # pragma: no cover
    pass


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


async def run_offer_generation(
    *,
    theme: str,
    wishes: str = "",
    template_id: str = "card_set",
    preset_id: str | None = None,
    n_variants: int = 4,
    n_elements: int | None = None,
    concurrency: int = 4,
    parts: dict[str, str] | None = None,
    skin: str | None = None,
    on_event: EventFn = _noop,
) -> RunReport:
    started = time.monotonic()
    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    on_event("run", {"run_id": run_id})

    preset = load_preset(preset_id)
    template = load_template(template_id)
    offline = is_offline() or preset.id == "offline"
    reg = Registry(providers_config(), offline=offline)

    if n_elements is None:
        n_elements = template.element_count(default=1)

    on_event("stage", {"name": "brief", "status": "start"})
    brief = await stages.parse_brief(reg, preset, theme, wishes, on_event=on_event)
    (run_dir / "brief.json").write_text(
        brief.model_dump_json(indent=2), encoding="utf-8"
    )
    on_event("stage", {"name": "brief", "status": "done",
                       "degraded": brief.degraded, "data": brief.model_dump()})

    on_event("stage", {"name": "concepts", "status": "start"})
    plans = await stages.generate_concepts(reg, preset, brief, n_variants, n_elements)
    (run_dir / "concepts.json").write_text(
        json.dumps([p.model_dump() for p in plans], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlap = stages.cross_variant_overlap(plans)
    on_event("stage", {
        "name": "concepts", "status": "done",
        "data": [{"id": p.variant_id, "title": p.title, "concept": p.concept} for p in plans],
        "overlap": overlap,
    })

    # Версии составляющих фиксируются на весь прогон: сборка должна
    # быть детерминированной, иначе варианты нельзя сравнивать.
    part_versions = {**parts_lib.defaults(), **(parts or {})}
    compositor = Compositor(template, part_versions=part_versions)
    offers: list[OfferResult] = []
    composites = []
    total_cost = 0.0
    total_images = 0
    total_rejected = 0

    for plan in plans:
        plan.parts = part_versions
        on_event("stage", {"name": "render", "status": "start", "variant": plan.variant_id})
        variant_dir = run_dir / plan.variant_id / "art"

        art, assets, verdicts, needs_review = await stages.render_plan(
            reg, preset, plan, template, variant_dir,
            concurrency=concurrency, skin=skin, on_event=on_event,
        )

        composite = compositor.render(plan, art)
        comp_path = run_dir / f"{plan.variant_id}_composite.png"
        composite.save(comp_path)
        composites.append(composite)

        rejected_dir = run_dir / plan.variant_id / "rejected"
        rejected_count = len(list(rejected_dir.glob("*.png"))) if rejected_dir.exists() else 0

        total_cost += sum(a.cost_usd for a in assets)
        total_images += len(assets) + rejected_count
        total_rejected += rejected_count

        offers.append(OfferResult(
            plan=plan, assets=assets, verdicts=verdicts,
            composite_path=str(comp_path), needs_review=needs_review,
        ))
        on_event("stage", {
            "name": "render", "status": "done", "variant": plan.variant_id,
            "composite": comp_path.name, "needs_review": needs_review,
        })

    sheet = contact_sheet(composites, cols=2 if len(composites) <= 4 else 3)
    sheet_path = run_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    report = RunReport(
        run_id=run_id,
        template_id=template.id,
        preset_id=preset.id,
        brief=brief,
        offers=offers,
        total_cost_usd=round(total_cost, 4),
        total_images=total_images,
        rejected_images=total_rejected,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    (run_dir / "run_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    _write_markdown_report(run_dir, report, overlap, preset.id, offline)

    on_event("done", {
        "run_id": run_id,
        "contact_sheet": sheet_path.name,
        "cost": report.total_cost_usd,
        "images": report.total_images,
        "rejected": report.rejected_images,
        "elapsed": report.elapsed_s,
    })
    return report


def _write_markdown_report(
    run_dir: Path, report: RunReport, overlap: dict, preset_id: str, offline: bool
) -> None:
    """Человекочитаемый отчёт — то, что уходит на доску вместе с картинками."""
    passed = sum(
        1 for o in report.offers for v in o.verdicts if v.passed
    )
    total_v = sum(len(o.verdicts) for o in report.offers)
    first_pass = sum(
        1 for o in report.offers for a in o.assets if a.attempt == 1
    )
    total_a = sum(len(o.assets) for o in report.offers)

    lines = [
        f"# Прогон {report.run_id}",
        "",
        f"- Шаблон оффера: `{report.template_id}`",
        f"- Составляющие: " + (", ".join(f"{k}={v}" for k, v in
          sorted((report.offers[0].plan.parts if report.offers else {}).items()))
          or "по умолчанию"),
        f"- Пресет моделей: `{preset_id}`" + (" (оффлайн, mock)" if offline else ""),
        f"- Тема: {report.brief.theme}",
        f"- Время: {report.elapsed_s} с",
        f"- Изображений сгенерировано: {report.total_images} "
        f"(отбраковано {report.rejected_images})",
        f"- Стоимость: ${report.total_cost_usd}",
        "",
        "## Автономность",
        "",
        f"- Прошло QC с первой попытки: {first_pass} из {total_a}",
        f"- Вердиктов пройдено: {passed} из {total_v}",
        f"- Требуют ручного просмотра: "
        f"{sum(len(o.needs_review) for o in report.offers)}",
        "",
        "## Бриф",
        "",
        "```json",
        report.brief.model_dump_json(indent=2),
        "```",
        "",
        "## Варианты",
        "",
    ]

    for o in report.offers:
        lines += [
            f"### {o.plan.variant_id} — {o.plan.title}",
            "",
            f"{o.plan.concept}",
            "",
            f"![{o.plan.variant_id}]({Path(o.composite_path).name})" if o.composite_path else "",
            "",
            "| # | Объект | Кат. | Ред. | Попыток | Модель |",
            "|---|--------|------|------|---------|--------|",
        ]
        by_slot = {a.slot: a for a in o.assets}
        for i, el in enumerate(o.plan.elements):
            a = by_slot.get(f"cards#{i}") or by_slot.get(f"art#{i}")
            lines.append(
                f"| {i + 1} | {el.subject} | {el.category} | {el.rarity} | "
                f"{a.attempt if a else '—'} | {a.model if a else 'не сгенерировано'} |"
            )
        if o.needs_review:
            lines += ["", f"**Требуют просмотра:** {', '.join(o.needs_review)}"]
        lines.append("")

    if overlap:
        lines += ["## Пересечения объектов между вариантами", ""]
        for pair, items in overlap.items():
            lines.append(f"- `{pair}`: {', '.join(items)}")
        lines.append("")

    lines += [
        "## Артефакты",
        "",
        "- `brief.json` — разобранный заказ",
        "- `concepts.json` — все варианты наполнения",
        "- `<variant>/art/` — принятые изображения",
        "- `<variant>/rejected/` — отбраковка QC с номером попытки",
        "- `<variant>_composite.png` — собранный оффер",
        "- `contact_sheet.png` — все варианты на одном листе",
        "",
    ]

    (run_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")
