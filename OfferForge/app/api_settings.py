"""Эндпоинты экранов настроек: Провайдеры, Модели, Промпты.

Вынесены отдельным роутером, чтобы main.py оставался про генерацию.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import settings
from app.config import (
    is_offline,
    list_presets,
    load_preset,
    preset_index,
    providers_config,
    recommended_models,
)
from app.models import Capability, VariantConfig
from app.providers.base import ProviderError, TextRequest
from app.providers.registry import Registry

log = logging.getLogger("offerforge.settings")
router = APIRouter(prefix="/api/settings", tags=["settings"])

# Роли, которые реально дергает текущий UI (наполнение + 🎲 в сборке).
# copy / extract убраны — старые пайплайны при необходимости берут default.
VARIANT_INFO = {
    "default": ("По умолчанию", "Фолбэк, если роль не описана в пресете явно.", "text"),
    "brief": ("Разбор заказа", "Свободное описание → структура брифа перед наполнением.", "text"),
    "concept": ("Концепты", "Придумывает 10 объектов для карточек коллекции.", "text"),
    "critic": ("Имя с картинки", "Кнопка 🎲 в сборке: читает арт и предлагает название. Нужна vision.", "vision"),
    "render": ("Генерация арта", "Рисует картинки карточек коллекции.", "image"),
}


# ---------------------------------------------------------------------------
# Провайдеры
# ---------------------------------------------------------------------------

class ProviderPatch(BaseModel):
    endpoint: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool | None = None
    auth_header: str | None = None
    auth_scheme: str | None = None
    capabilities: list[str] | None = None


class ProviderCreate(BaseModel):
    id: str
    kind: str = "openai_compatible"
    endpoint: str = ""
    base_url: str = ""
    api_key: str = ""
    capabilities: list[str] = ["text"]


@router.get("/providers")
async def get_providers() -> dict:
    return {
        "providers": settings.list_providers(),
        "builtin": sorted(settings.BUILTIN_PROVIDERS),
        "kinds": [
            {"id": "openai_compatible", "name": "OpenAI-совместимый (текст, vision)"},
            {"id": "openrouter_images", "name": "OpenRouter Images (генерация)"},
            {"id": "custom_rest", "name": "Кастомный REST (описывается в providers.yaml)"},
        ],
        "offline": is_offline(),
    }


@router.patch("/providers/{provider_id}")
async def patch_provider(provider_id: str, patch: ProviderPatch) -> dict:
    try:
        return settings.update_provider(
            provider_id, patch.model_dump(exclude_unset=True)
        )
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/providers")
async def post_provider(body: ProviderCreate) -> dict:
    try:
        return settings.create_provider(body.id, body.model_dump(exclude={"id"}))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/providers/{provider_id}")
async def remove_provider(provider_id: str) -> dict:
    try:
        settings.delete_provider(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.delete("/providers/{provider_id}/key")
async def clear_provider_key(provider_id: str) -> dict:
    """Удалить только API-ключ, провайдера оставить."""
    try:
        return settings.clear_provider_key(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/providers/{provider_id}/test")
async def test_provider(provider_id: str) -> dict:
    """Живая проверка: ходим настоящим запросом, а не проверяем непустоту поля."""
    cfg = (providers_config().get("providers") or {}).get(provider_id)
    if cfg is None:
        raise HTTPException(404, f"Провайдер '{provider_id}' не найден")

    reg = Registry(providers_config(), offline=False)
    provider = reg.providers.get(provider_id)
    if provider is None:
        return {"ok": False, "message": "Провайдер выключен или собран с ошибкой"}
    if not provider.is_configured():
        return {"ok": False, "message": "Не задан API-ключ"}

    started = time.monotonic()
    notes: list[str] = []

    # Статус ключа — первым делом. Он отвечает на вопрос «ключ вообще
    # рабочий и что ему разрешено», а список моделей на это ответить не может.
    status = None
    if hasattr(provider, "key_status"):
        status = await provider.key_status()
        if status and status.get("error"):
            return {"ok": False, "message": f"ключ отклонён: {status['error']}",
                    "elapsed_ms": int((time.monotonic() - started) * 1000)}
        if status:
            limit, left = status.get("limit"), status.get("limit_remaining")
            if left is not None:
                notes.append(f"остаток ${left}")
            elif limit is not None:
                notes.append(f"лимит ${limit}")
            if status.get("is_free_tier"):
                notes.append("free tier")
            if status.get("label"):
                notes.append(str(status["label"])[:40])

    # Затем — настоящий запрос к модели. Раньше здесь дёргался список
    # моделей, но /models публичный: он отвечал и без доступа, поэтому
    # провайдер всегда показывался рабочим.
    caps = provider.capabilities
    try:
        if Capability.IMAGE in caps and Capability.TEXT not in caps:
            models = await provider.list_models(Capability.IMAGE)
            notes.append(f"моделей в каталоге: {len(models)}" if models
                         else "каталог доступен")
            notes.append("реальный запрос проверяется кнопкой у варианта render")
        else:
            # DeepSeek V4 / MiniMax по умолчанию включают reasoning и съедают
            # крошечный бюджет «Проверить» (раньше max_tokens=5 → finish=length
            # после 1 токена). Без effort:none кнопка врёт «обрезан лимит».
            probe = _pick_probe_model(provider)
            res = await provider.generate_text(
                probe,
                TextRequest(
                    prompt="Reply with exactly one word: pong",
                    params={
                        "max_tokens": 64,
                        "temperature": 0,
                        "reasoning": {"effort": "none"},
                    },
                ),
                30,
            )
            notes.append(f"модель {probe} ответила: {res.text.strip()[:30] or 'пусто'}")
        return {"ok": True, "message": " · ".join(notes),
                "elapsed_ms": int((time.monotonic() - started) * 1000)}
    except Exception as e:  # noqa: BLE001 - показываем текст ошибки как есть
        prefix = (" · ".join(notes) + " · ") if notes else ""
        return {"ok": False, "message": prefix + str(e).replace("\n", " ")[:300],
                "elapsed_ms": int((time.monotonic() - started) * 1000)}


def _pick_probe_model(provider) -> str:  # noqa: ANN001
    """Модель для пробного запроса — берём первую из активного пресета,
    чтобы проверять ровно то, чем пользуются, а не абстрактную модель."""
    try:
        preset = load_preset(preset_index().get("default_preset"))
        for name in ("default", "brief"):
            cfg = preset.variants.get(name)
            if cfg:
                for mp in cfg.rotation():
                    if mp.provider_id == provider.id:
                        return mp.name
    except Exception:  # noqa: BLE001
        pass
    return "openai/gpt-5.6-luna"


# ---------------------------------------------------------------------------
# Модели по вариантам
# ---------------------------------------------------------------------------

@router.get("/variants")
async def get_variants(preset_id: str | None = None) -> dict:
    preset = load_preset(preset_id)
    provs = settings.list_providers()

    variants = []
    for name, cfg in preset.variants.items():
        title, doc, cap = VARIANT_INFO.get(name, (name, "", "text"))
        variants.append({
            "name": name,
            "title": title,
            "doc": doc,
            "capability": cap,
            "timeout": cfg.timeout,
            "rotation": [m.model_dump(exclude_none=True) for m in cfg.rotation()],
            "default_params": cfg.default_params,
        })

    order = list(VARIANT_INFO)
    variants.sort(key=lambda v: (order.index(v["name"]) if v["name"] in order else 99, v["name"]))

    return {
        "preset": {"id": preset.id, "name": preset.name},
        "presets": list_presets(),
        "variants": variants,
        "providers": [
            {"id": p["id"], "capabilities": p["capabilities"], "has_key": p["has_key"]}
            for p in provs if p["enabled"]
        ],
        "recommended": recommended_models(),
    }


class RotationEntry(BaseModel):
    name: str
    provider_id: str = "openrouter"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # Без этого поля настройка рассуждений терялась бы при любом
    # сохранении варианта из интерфейса.
    reasoning: dict | None = None
    size: str | None = None
    n_reference_images: int | None = None


class VariantPatch(BaseModel):
    rotation: list[RotationEntry]
    timeout: int | None = None


@router.put("/variants/{preset_id}/{variant}")
async def put_variant(preset_id: str, variant: str, body: VariantPatch) -> dict:
    if not body.rotation:
        raise HTTPException(400, "В ротации должна остаться хотя бы одна модель")
    try:
        raw = settings.read_preset_raw(preset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e

    variants = raw.setdefault("variants", {})
    entry = variants.setdefault(variant, {})
    rotation = [m.model_dump(exclude_none=True) for m in body.rotation]

    entry["model_name"] = rotation[0]["name"]
    entry["model_params"] = rotation
    entry.setdefault("default_params", {})
    first = rotation[0]
    for field in ("temperature", "max_tokens", "top_p"):
        if field in first:
            entry["default_params"][field] = first[field]
    if body.timeout:
        entry["timeout"] = body.timeout

    settings.write_preset(preset_id, variants)
    return {"ok": True, "variant": variant, "rotation": rotation}


class PresetSaveAs(BaseModel):
    id: str
    name: str
    note: str = ""
    from_preset: str | None = None


@router.post("/presets")
async def save_preset(body: PresetSaveAs) -> dict:
    source = settings.read_preset_raw(body.from_preset or list_presets()[0]["id"])
    try:
        return settings.save_preset_as(
            body.id, body.name, source.get("variants", {}), body.note
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/presets/{preset_id}")
async def remove_preset(preset_id: str) -> dict:
    try:
        settings.delete_preset(preset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.get("/presets/{preset_id}/export")
async def export_preset(preset_id: str) -> dict:
    try:
        return settings.read_preset_raw(preset_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


class PresetImport(BaseModel):
    id: str
    name: str
    variants: dict
    note: str = "импортирован"


@router.post("/presets/import")
async def import_preset(body: PresetImport) -> dict:
    if "default" not in body.variants:
        raise HTTPException(400, "В пресете обязателен вариант 'default'")
    try:
        return settings.save_preset_as(body.id, body.name, body.variants, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/variants/{preset_id}/{variant}/test")
async def test_variant(preset_id: str, variant: str) -> dict:
    """Прогоняет вариант ровно тем маршрутом, каким пойдёт пайплайн."""
    preset = load_preset(preset_id)
    cfg: VariantConfig | None = preset.variants.get(variant)
    if cfg is None:
        raise HTTPException(404, f"Вариант '{variant}' не найден")

    reg = Registry(providers_config(), offline=is_offline())
    _, _, cap = VARIANT_INFO.get(variant, (variant, "", "text"))
    results = []

    for mp in cfg.rotation():
        started = time.monotonic()
        single = VariantConfig(
            model_name=mp.name, model_params=[mp],
            default_params=cfg.default_params, timeout=min(cfg.timeout, 90),
        )
        # Проверяем теми же параметрами, какими пойдёт боевой запрос: brief,
        # concept и critic вызываются с json_mode, и если модель его не
        # принимает, тест обязан это показать, а не отчитаться зелёным.
        json_mode = variant in {"brief", "concept", "critic"}
        try:
            if cap == "image":
                res = await reg.image(single, "a simple red apple, plain background",
                                      width=512, height=512)
                detail = f"{len(res.images)} изображение"
            elif cap == "vision":
                res = await reg.text(
                    single,
                    'Что на картинке? Ответь JSON вида {"object": "одно слово"}.',
                    images=[_probe_png()], json_mode=json_mode,
                )
                detail = res.text.strip().replace("\n", " ")[:60]
            else:
                res = await reg.text(
                    single,
                    'Ответь JSON вида {"ok": true}.' if json_mode
                    else "Ответь одним словом: работает?",
                    json_mode=json_mode,
                )
                detail = res.text.strip().replace("\n", " ")[:60]
            results.append({
                "model": mp.name, "provider": mp.provider_id, "ok": True,
                "detail": detail, "elapsed_ms": int((time.monotonic() - started) * 1000),
            })
        except (ProviderError, Exception) as e:  # noqa: BLE001
            results.append({
                "model": mp.name, "provider": mp.provider_id, "ok": False,
                "detail": str(e).replace("\n", " ")[:200],
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            })

    ok = sum(1 for r in results if r["ok"])
    return {"variant": variant, "results": results,
            "summary": f"ответили {ok} из {len(results)}", "all_ok": ok == len(results)}


def _probe_png() -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse([64, 64, 192, 192], fill=(210, 40, 40))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

@router.get("/prompts")
async def get_prompts() -> dict:
    return {
        "files": settings.list_prompt_files(),
        "decorators": settings.decorator_reference(),
    }


@router.get("/prompts/{name}")
async def get_prompt(name: str) -> dict:
    try:
        content = settings.read_prompt(name)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {
        "name": name,
        "content": content,
        "validation": settings.validate_prompt(content),
        "backups": settings.list_backups(name),
    }


class PromptBody(BaseModel):
    content: str


@router.put("/prompts/{name}")
async def put_prompt(name: str, body: PromptBody) -> dict:
    validation = settings.validate_prompt(body.content)
    if not validation["valid"]:
        raise HTTPException(400, f"Ошибка синтаксиса — {validation['error']}")
    try:
        result = settings.write_prompt(name, body.content)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {**result, "validation": validation,
            "backups": settings.list_backups(name)}


@router.post("/prompts/{name}/validate")
async def check_prompt(name: str, body: PromptBody) -> dict:
    return settings.validate_prompt(body.content)


@router.post("/prompts/{name}/restore")
async def restore_prompt(name: str, backup: str) -> dict:
    try:
        content = settings.restore_backup(name, backup)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    return {"name": name, "content": content, "backups": settings.list_backups(name)}
