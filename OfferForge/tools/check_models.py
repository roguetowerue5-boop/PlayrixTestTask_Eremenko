"""Сверка моделей из пресетов с живым каталогом OpenRouter.

    python tools/check_models.py

Идентификаторы моделей протухают: провайдер переименовал, снял с публикации,
сменил префикс. Ловится это иначе только на середине прогона, когда половина
картинок уже оплачена. Скрипт проверяет каждую модель во всех пресетах:

  - существует ли она вообще;
  - умеет ли то, что от неё требует вариант (vision для critic,
    генерация картинок для render, приём референсов);
  - сколько стоит.

Каталог openrouter.ai/api/v1/models публичный — ключ не нужен.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CATALOG_URL = "https://openrouter.ai/api/v1/models"
TOKENS_PER_IMAGE = 1290  # типовой тайл 1024×1024 — для моделей с потокенной ценой

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# Модели китайских лабораторий на OpenRouter чаще прочих упираются в
# ограничения по политике данных и отдают HTTP 403 "Access denied by
# security policy" — причём группой, а не поодиночке. Ротация целиком из
# этой группы падает вся сразу, что и наблюдалось на практике.
#
# Для остальных вендоров такой корреляции нет: Anthropic и Google
# независимы, и одновременная блокировка обоих — не тот случай, о котором
# стоит предупреждать.
RESTRICTED_PRONE = (
    "qwen", "deepseek", "minimax", "z-ai", "glm", "moonshot", "stepfun",
    "tencent", "kwaipilot", "bytedance", "inclusionai", "ling", "baidu",
)


def vendor_of(model_id: str) -> str:
    return model_id.split("/", 1)[0].lower()


def is_restriction_prone(model_id: str) -> bool:
    low = model_id.lower()
    return any(m in low for m in RESTRICTED_PRONE)


def fetch_catalog() -> dict[str, dict]:
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(CATALOG_URL)
        r.raise_for_status()
        return {m["id"]: m for m in r.json()["data"]}


def price_text(m: dict) -> str:
    p = m.get("pricing") or {}
    try:
        inp, out = float(p.get("prompt") or 0) * 1e6, float(p.get("completion") or 0) * 1e6
    except (TypeError, ValueError):
        return "?"
    if not (inp or out):
        return "бесплатно"
    return f"${inp:.3f}/${out:.3f} за 1M"


def price_image(m: dict) -> str:
    p = m.get("pricing") or {}
    try:
        flat = float(p.get("image") or 0)
        if flat:
            return f"${flat:.4f}/img"
        tok = float(p.get("image_output") or p.get("image_token") or 0)
        if tok:
            return f"≈${tok * TOKENS_PER_IMAGE:.4f}/img"
    except (TypeError, ValueError):
        pass
    return "?"


def check() -> int:
    try:
        catalog = fetch_catalog()
    except httpx.HTTPError as e:
        print(f"{RED}Не удалось получить каталог: {e}{OFF}")
        return 2
    print(f"Каталог OpenRouter: {len(catalog)} моделей\n")

    problems: list[str] = []
    warnings: list[str] = []
    presets_dir = ROOT / "model-presets" / "presets"

    for path in sorted(presets_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        variants = data.get("variants") or {}

        # Оффлайн-пресет ходит в mock, сверять там нечего.
        if all(
            mp.get("provider_id") == "mock"
            for v in variants.values()
            for mp in v.get("model_params", [])
        ):
            print(f"{DIM}{path.name}: mock-пресет, пропуск{OFF}")
            continue

        print(f"{path.name}")
        for variant, cfg in variants.items():
            rotation = cfg.get("model_params", [])

            # Две проверки устойчивости ротации. Обе — предупреждения, а не
            # ошибки: конфиг рабочий, просто хрупкий.
            names = [mp["name"] for mp in rotation]
            if len(rotation) > 1:
                if len({vendor_of(n) for n in names}) == 1:
                    warnings.append(f"{path.name}/{variant}: вся ротация от одного "
                                    f"вендора ({vendor_of(names[0])})")
                    print(f"  {YELLOW}!{OFF}  {variant:8} {DIM}вся ротация от одного "
                          f"вендора — фолбэк не спасёт при отказе провайдера{OFF}")
                elif all(is_restriction_prone(n) for n in names):
                    warnings.append(f"{path.name}/{variant}: вся ротация из моделей, "
                                    f"часто закрытых политикой")
                    print(f"  {YELLOW}!{OFF}  {variant:8} {DIM}все модели ротации "
                          f"часто закрыты политикой провайдера — добавь одну "
                          f"из другой группы{OFF}")

            for mp in rotation:
                mid = mp["name"]
                is_image = mp.get("provider_id") == "openrouter_images"
                model = catalog.get(mid)

                if model is None:
                    near = [k for k in catalog if k.rsplit("/", 1)[-1] == mid.rsplit("/", 1)[-1]]
                    hint = f"  похоже на: {', '.join(near[:3])}" if near else ""
                    print(f"  {RED}✗{OFF} {variant:8} {mid:40} НЕТ В КАТАЛОГЕ{hint}")
                    problems.append(f"{path.name}/{variant}: {mid} не существует")
                    continue

                arch = model.get("architecture") or {}
                ins = arch.get("input_modalities") or []
                outs = arch.get("output_modalities") or []
                notes, fail = [], False

                if is_image:
                    if "image" not in outs:
                        notes.append("НЕ ГЕНЕРИРУЕТ КАРТИНКИ")
                        fail = True
                    if mp.get("n_reference_images") and "image" not in ins:
                        notes.append("не принимает референсы")
                        fail = True
                    if "seed" not in (model.get("supported_parameters") or []):
                        notes.append("без seed")
                    notes.append(price_image(model))
                else:
                    if variant == "critic" and "image" not in ins:
                        notes.append("НЕТ VISION — критик не соберётся")
                        fail = True
                    notes.append(price_text(model))

                mark = f"{RED}✗{OFF}" if fail else f"{GREEN}ok{OFF}"
                colour = RED if fail else (YELLOW if "без seed" in notes else DIM)
                print(f"  {mark} {variant:8} {mid:40} {colour}{' · '.join(notes)}{OFF}")
                if fail:
                    problems.append(f"{path.name}/{variant}: {mid} — {notes[0]}")
        print()

    rec_path = ROOT / "model-presets" / "recommended-models.yaml"
    if rec_path.exists():
        rec = yaml.safe_load(rec_path.read_text(encoding="utf-8")) or {}
        missing = [
            (var, it["id"] if isinstance(it, dict) else it)
            for var, items in rec.items()
            for it in items
            if (it["id"] if isinstance(it, dict) else it) not in catalog
        ]
        if missing:
            print("recommended-models.yaml")
            for var, mid in missing:
                print(f"  {RED}✗{OFF} {var:8} {mid} НЕТ В КАТАЛОГЕ")
                problems.append(f"recommended/{var}: {mid} не существует")
        else:
            print(f"recommended-models.yaml — {GREEN}все модели на месте{OFF}")

    print()
    if warnings:
        print(f"{YELLOW}Предупреждения ({len(warnings)}) — работать будет, но хрупко:{OFF}")
        for w in warnings:
            print(f"  · {w}")
        print()

    if problems:
        print(f"{RED}Проблем: {len(problems)}{OFF}")
        for p in problems:
            print(f"  · {p}")
        return 1

    print(f"{GREEN}Все модели существуют и умеют то, что от них требуется.{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(check())
