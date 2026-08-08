#!/usr/bin/env python
"""Проверка боевого тракта с настоящим ключом.

selftest.py проверяет код на заглушках: он зелёный даже когда ключ отозван
и ни один запрос не уходит. Этот скрипт — наоборот, про живые вызовы.
Он идёт по тем же шагам, что интерфейс, и на каждом говорит, что именно
пришло от модели.

Запуск:
    .venv\\Scripts\\python tools\\healthcheck.py
    .venv\\Scripts\\python tools\\healthcheck.py --skin ..\\Framing\\Orange.png
    .venv\\Scripts\\python tools\\healthcheck.py --preset balanced --skip-art
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.config import load_preset, providers_config, resolve_variant  # noqa: E402
from app.providers.base import Capability  # noqa: E402
from app.providers.registry import Registry  # noqa: E402

OK, BAD, WARN = "  ok  ", " ПЛОХО", " ! "
_fails = 0


def say(mark: str, title: str, detail: str = "") -> None:
    global _fails
    if mark is BAD:
        _fails += 1
    print(f"{mark} {title}" + (f"\n         {detail}" if detail else ""))


def head(text: str) -> None:
    print(f"\n=== {text}")


async def check_key(reg: Registry) -> bool:
    head("1. Ключ и провайдеры")
    alive = False
    for pid, p in reg.providers.items():
        if not p.is_configured():
            say(WARN, f"{pid}: ключа нет", "провайдер пропущен")
            continue
        status = None
        if hasattr(p, "key_status"):
            try:
                status = await p.key_status()
            except Exception as e:  # noqa: BLE001
                say(BAD, f"{pid}: ключ не принят", str(e)[:200])
                continue
        alive = True
        if status:
            limit = status.get("limit")
            used = status.get("usage")
            say(OK, f"{pid}: ключ живой",
                f"израсходовано {used}, лимит {limit if limit is not None else 'без лимита'}")
        else:
            say(OK, f"{pid}: настроен")
    if not alive:
        say(BAD, "живых провайдеров нет",
            "впишите ключ в config/secrets.json или переменную OPENROUTER_API_KEY")
    return alive


async def check_roles(reg: Registry, preset) -> None:
    head("2. Роли пресета")
    for role in ("brief", "concept", "critic"):
        variant = resolve_variant(preset, role)
        t = time.time()
        try:
            resp = await reg.text(
                variant,
                'Ответь строго JSON: {"ok": true}',
                json_mode=True,
            )
            body = " ".join((resp.text or "").split())[:120]
            if not body:
                say(BAD, f"{role} ({variant.model_name}) молчит",
                    "пустой ответ — обычно модель ушла в рассуждения "
                    "и обрыв по лимиту токенов")
            else:
                say(OK, f"{role} ({variant.model_name}) отвечает",
                    f"{time.time() - t:.1f}с · {body}")
        except Exception as e:  # noqa: BLE001
            say(BAD, f"{role} ({variant.model_name}) упала", str(e)[:300])


async def check_images(reg: Registry, preset) -> None:
    head("3. Картинки")
    for role in ("render",):
        variant = resolve_variant(preset, role)
        t = time.time()
        try:
            resp = await reg.image(variant, "a plain red circle on white",
                                   width=512, height=512)
            if not resp.images:
                say(BAD, f"{role} ({variant.model_name}) вернула пусто")
                continue
            im = Image.open(BytesIO(resp.images[0]))
            say(OK, f"{role} ({variant.model_name}) рисует",
                f"{time.time() - t:.1f}с · {im.width}x{im.height} · "
                f"${resp.cost_usd:.4f}")
        except Exception as e:  # noqa: BLE001
            say(BAD, f"{role} ({variant.model_name}) упала", str(e)[:300])


async def check_segment(reg: Registry, preset, skin: Path) -> None:
    head(f"4. Сегментация: {skin.name}")
    from app.pipeline import segment

    raw = skin.read_bytes()
    t = time.time()
    try:
        markup = await segment.analyze(reg, preset, raw)
    except Exception as e:  # noqa: BLE001
        say(BAD, "разметка не получена", str(e)[:300])
        return

    kinds = [p.get("kind") for p in markup.get("parts", [])]
    say(OK, f"разметка за {time.time() - t:.1f}с", f"нашлось: {', '.join(kinds)}")

    slots = markup.get("slots") or {}
    if segment.first_cell(slots):
        say(OK, "сетка карточек найдена",
            f"{slots.get('cols')}×{slots.get('rows')}")
    else:
        say(BAD, "сетка карточек не найдена",
            "без неё второй проход по карточке не делается")

    with_point = [p["kind"] for p in markup.get("parts", []) if p.get("at")]
    if with_point:
        say(OK, "точки для автолассо есть", ", ".join(with_point))
    else:
        say(WARN, "модель не дала ни одной точки",
            "вырезание пойдёт по боксам, границы будут грубее")

    problems = segment.validate_markup(markup)
    if problems:
        say(WARN, "валидация ругается", "; ".join(problems[:5]))
    else:
        say(OK, "разметка проходит валидацию")

    head("5. Вырезание и сверка сборкой")
    t = time.time()
    try:
        result = await segment.segment_skin(
            reg, preset, raw, "healthcheck", markup=markup,
            extract_mode="model", make_default=False, save_references=False)
    except Exception as e:  # noqa: BLE001
        say(BAD, "нарезка упала", str(e)[:300])
        return

    say(OK, f"вырезано за {time.time() - t:.1f}с",
        ", ".join(s["kind"] for s in result["saved"]))

    checks = result.get("checks") or []
    if not checks:
        say(WARN, "сверка не выполнялась")
    else:
        last = checks[-1]
        detail = f"среднее расхождение {last['mean_diff']}/255"
        if last.get("repaired"):
            detail += " · починено: " + ", ".join(last["repaired"])
        if last["ok"]:
            say(OK, "сборка сошлась", detail)
        else:
            say(BAD, "сборка не сошлась", detail)
            for row in last["parts"]:
                if not row["ok"]:
                    print(f"           {row['kind']}: {'; '.join(row['troubles'])}")

    # За собой прибираем: healthcheck не должен засорять библиотеку.
    from app import parts, skins

    for item in result["saved"]:
        try:
            parts.delete_version(item["kind"], "healthcheck")
        except Exception:  # noqa: BLE001
            pass
    try:
        skins.delete_skin("healthcheck")
    except Exception:  # noqa: BLE001
        pass


async def check_art(reg: Registry, preset) -> None:
    head("6. Генерация арта (короткий прогон)")
    from app.pipeline import stages

    try:
        brief = await stages.parse_brief(reg, preset, "Кино-коллекция", "")
        say(OK, "бриф разобран",
            "с оговоркой — модель была недоступна" if getattr(brief, "degraded", False)
            else getattr(brief, "theme", ""))
    except Exception as e:  # noqa: BLE001
        say(BAD, "бриф не разобран", str(e)[:300])
        return

    t = time.time()
    try:
        plans = await stages.generate_concepts(reg, preset, brief, 1, 10)
        say(OK, f"концепты за {time.time() - t:.1f}с",
            f"{len(plans)} вариант, элементов: {len(plans[0].elements)}")
        print("           " + ", ".join(e.subject for e in plans[0].elements[:5]) + " …")
    except Exception as e:  # noqa: BLE001
        say(BAD, "концепты не получены", str(e)[:400])


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--skin", type=Path, default=Path("../Framing/Orange.png"))
    ap.add_argument("--skip-art", action="store_true")
    ap.add_argument("--skip-segment", action="store_true")
    args = ap.parse_args()

    preset = load_preset(args.preset)
    reg = Registry(providers_config(), offline=False)
    print(f"Пресет: {preset.id} ({preset.name})")

    if not await check_key(reg):
        print("\nДальше идти незачем: без ключа всё остальное упадёт.")
        return 1

    await check_roles(reg, preset)
    await check_images(reg, preset)

    skin = (Path(__file__).resolve().parent.parent / args.skin).resolve()
    if args.skip_segment:
        pass
    elif not skin.exists():
        say(WARN, f"скин не найден: {skin}", "сегментацию пропускаю")
    else:
        await check_segment(reg, preset, skin)

    if not args.skip_art:
        await check_art(reg, preset)

    print(f"\n{'=' * 60}")
    if _fails:
        print(f"Провалов: {_fails}. Смотрите строки «ПЛОХО» — там причина.")
    else:
        print("Всё живое. Если интерфейс всё равно даёт мусор — дело в качестве "
              "разметки, а не в связи с моделями.")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
