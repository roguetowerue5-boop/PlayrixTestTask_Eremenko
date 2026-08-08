"""Реестр провайдеров и разрешение вариантов.

Вариант — это роль в пайплайне ("critic", "render"), а не конкретная модель.
Код просит роль; какая модель и у какого провайдера её исполнит, решает
пресет. Ротация внутри варианта работает как цепочка фолбэка: упала первая —
идём ко второй, и так до конца списка.

Схема заимствована из model-presets SkyrimNet: там ровно так же variants →
model_params[] → provider_id, и новый пресет добавляется файлом, без кода.
"""
from __future__ import annotations

import logging

from app.models import Capability, ModelParam, VariantConfig
from app.providers.base import (
    ImageRequest,
    ImageResponse,
    Provider,
    ProviderError,
    TextRequest,
    TextResponse,
)
from app.providers.rest import MockProvider, build_provider

log = logging.getLogger("offerforge.registry")


def _is_blank(text: str | None, json_mode: bool) -> bool:
    """Ответ формально пришёл, но пользы в нём ноль.

    Модели под нагрузкой отдают `{}`, `{"": [""]}`, пустую строку или один
    открывающий брейс. Провайдер такое отказом не считает, и без этой
    проверки ротация не переключалась — мы повторяли запрос к той же
    модели, вместо того чтобы взять следующую в цепочке.
    """
    body = (text or "").strip()
    if not body:
        return True
    if not json_mode:
        return False

    import json as _json

    stripped = body.strip("`").removeprefix("json").strip()
    try:
        parsed = _json.loads(stripped)
    except ValueError:
        # Не разобралось — пусть решает вызывающий, там может быть JSON
        # внутри прозы. Но огрызок вида "{" полезным быть не может.
        return len(stripped) < 3

    if parsed in (None, {}, []):
        return True
    if isinstance(parsed, dict):
        # Ключи есть, а смысла нет: {"": [""]} и подобное.
        return not any(
            k and v not in (None, "", [], {}, [""])
            for k, v in parsed.items()
        )
    return False


def _summarize(errors: list[str]) -> str:
    """Схлопывает одинаковые ошибки ротации.

    Когда у всех моделей один провайдер и кончились кредиты, текст ошибки
    повторяется столько раз, сколько моделей в ротации. Читать это невозможно,
    а причина одна.
    """
    uniq = list(dict.fromkeys(errors))
    if len(uniq) == 1:
        return uniq[0]
    return "все модели варианта упали:\n  " + "\n  ".join(uniq)


class Registry:
    def __init__(self, providers_cfg: dict, offline: bool = False):
        self.offline = offline
        self.providers: dict[str, Provider] = {}
        for pid, cfg in (providers_cfg.get("providers") or {}).items():
            if not cfg.get("enabled", True):
                continue
            try:
                self.providers[pid] = build_provider(pid, cfg)
            except ValueError as e:
                log.warning("Пропускаю провайдера: %s", e)
        self.providers["mock"] = MockProvider()

    # -- состояние -----------------------------------------------------------

    def available(self) -> dict[str, dict]:
        out = {}
        for pid, p in self.providers.items():
            configured = getattr(p, "is_configured", lambda: True)()
            out[pid] = {
                "configured": configured,
                "capabilities": sorted(c.value for c in p.capabilities),
            }
        return out

    def _resolve(self, mp: ModelParam) -> Provider | None:
        if self.offline:
            return self.providers["mock"]
        p = self.providers.get(mp.provider_id)
        if p is None:
            log.warning("Провайдер %s не найден", mp.provider_id)
            return None
        if not getattr(p, "is_configured", lambda: True)():
            log.warning("Провайдер %s без ключа — пропуск", mp.provider_id)
            return None
        return p

    @staticmethod
    def _merge(variant: VariantConfig, mp: ModelParam) -> dict:
        params = dict(variant.default_params)
        for field in ("temperature", "max_tokens", "top_p", "reasoning"):
            val = getattr(mp, field, None)
            if val is not None:
                params[field] = val
        return params

    # -- вызовы --------------------------------------------------------------

    async def text(
        self,
        variant: VariantConfig,
        prompt: str,
        *,
        system: str | None = None,
        images: list[bytes] | None = None,
        json_mode: bool = False,
    ) -> TextResponse:
        cap = Capability.VISION if images else Capability.TEXT
        errors: list[str] = []

        for mp in variant.rotation():
            provider = self._resolve(mp)
            if provider is None:
                errors.append(f"{mp.provider_id}: недоступен")
                continue
            if not provider.supports(cap) and provider.id != "mock":
                errors.append(f"{mp.provider_id}: нет {cap.value}")
                continue
            req = TextRequest(
                prompt=prompt,
                system=system,
                images=images or [],
                params=self._merge(variant, mp),
                json_mode=json_mode,
            )
            try:
                resp = await provider.generate_text(mp.name, req, variant.timeout)
            except ProviderError as e:
                log.warning("Ротация: %s", e)
                errors.append(str(e))
                continue

            # Пустой ответ — тоже отказ, просто без исключения. Модель
            # отвечает `{}` или `{"": [""]}`, формально успешно, и ротация
            # не срабатывала: мы трижды спрашивали ту же модель вместо того,
            # чтобы перейти к следующей в цепочке.
            if _is_blank(resp.text, json_mode):
                short = " ".join((resp.text or "").split())[:80]
                log.warning("Ротация: %s вернула пустой ответ %r", mp.name, short)
                errors.append(f"{mp.name}: пустой ответ {short!r}")
                continue

            if resp.cost_usd:
                try:
                    from app.billing import record_spend
                    record_spend(resp.cost_usd, mp.provider_id)
                except Exception:  # noqa: BLE001
                    pass
            return resp

        raise ProviderError(_summarize(errors))

    async def image(
        self,
        variant: VariantConfig,
        prompt: str,
        *,
        negative: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        references: list[bytes] | None = None,
    ) -> ImageResponse:
        errors: list[str] = []

        for mp in variant.rotation():
            provider = self._resolve(mp)
            if provider is None:
                errors.append(f"{mp.provider_id}: недоступен")
                continue
            if not provider.supports(Capability.IMAGE) and provider.id != "mock":
                errors.append(f"{mp.provider_id}: нет image")
                continue

            refs = list(references or [])
            if mp.n_reference_images is not None:
                refs = refs[: mp.n_reference_images]
            # OpenRouter: flux ≤4, gemini-image ≤1. Дублируем clamp здесь,
            # чтобы пресеты с устаревшим n_reference_images=6 не роняли ротацию.
            from app.providers.rest import OpenRouterImageProvider
            if provider.id in ("openrouter_images", "openrouter"):
                refs = refs[: OpenRouterImageProvider._ref_cap(mp.name)]
            w, h = width, height
            if mp.size and "x" in mp.size:
                try:
                    w, h = (int(v) for v in mp.size.split("x", 1))
                except ValueError:
                    pass

            # default_params применялись только к тексту, из-за чего
            # background: transparent и прочие настройки картинок молча
            # никуда не уходили. Параметры конкретной модели важнее общих.
            params = dict(variant.default_params)
            params.update({k: v for k, v in (mp.params or {}).items()
                           if v is not None})

            req = ImageRequest(
                prompt=prompt,
                negative=negative,
                width=w,
                height=h,
                seed=seed,
                reference_images=refs,
                params=params,
            )
            try:
                resp = await provider.generate_image(mp.name, req, variant.timeout)
                if resp.cost_usd:
                    try:
                        from app.billing import record_spend
                        record_spend(resp.cost_usd, mp.provider_id)
                    except Exception:  # noqa: BLE001
                        pass
                return resp
            except ProviderError as e:
                log.warning("Ротация: %s", e)
                errors.append(str(e))

        raise ProviderError(_summarize(errors))

    async def list_models(self, cap: Capability) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for pid, p in self.providers.items():
            if pid == "mock" or not p.supports(cap):
                continue
            models = await p.list_models(cap)
            if models:
                out[pid] = models
        return out
