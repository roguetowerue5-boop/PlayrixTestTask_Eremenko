"""Провайдеры поверх HTTP.

Три вида, все описываются в providers.yaml без единой строки кода:

  openai_compatible — /chat/completions в формате OpenAI (текст + vision).
  openrouter_images — POST /images унифицированного Image API OpenRouter.
  custom_rest       — произвольный эндпоинт: тело запроса и путь к картинкам
                      в ответе задаются прямо в YAML. Сюда подключается любой
                      локальный ComfyUI/A1111/внутренний сервис.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx

from app.models import Capability
from app.providers.base import (
    ImageRequest,
    ImageResponse,
    Provider,
    ProviderError,
    TextRequest,
    TextResponse,
    decode_images,
    extract_path,
    substitute,
)


log = logging.getLogger("offerforge.providers")

# Признаки того, что модель не переварила response_format, а не сломалась
# по существу. У разных провайдеров формулировки свои.
_JSON_MODE_MARKERS = (
    "response_format",
    "json_object",
    "json mode",
    "json_schema",
    "structured output",
)


def _is_json_mode_error(err: Exception) -> bool:
    text = str(err).lower()
    return "http 400" in text and any(m in text for m in _JSON_MODE_MARKERS)


def _is_reasoning_error(err: Exception) -> bool:
    """Модель не поняла параметр reasoning — не повод падать."""
    text = str(err).lower()
    return "http 400" in text and "reasoning" in text


def _append_json_hint(content: Any) -> Any:
    """Просим JSON словами, раз параметром не вышло."""
    hint = "\n\nВерни ТОЛЬКО валидный JSON, без пояснений и без markdown-обёртки."
    if isinstance(content, str):
        return content + hint
    if isinstance(content, list):
        out = list(content)
        for i, part in enumerate(out):
            if isinstance(part, dict) and part.get("type") == "text":
                out[i] = {**part, "text": part.get("text", "") + hint}
                return out
        out.append({"type": "text", "text": hint})
        return out
    return content


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# Разметка типового экрана коллекции: панель, лента, прогресс, сетка 5×2.
# Координаты сняты с реального скина и проверены вырезкой, поэтому годятся
# и как ответ mock-провайдера, и как запасной вариант, если vision вернул
# мусор — лучше отработать по типовой раскладке, чем не отработать вовсе.
FALLBACK_SCREEN_MARKUP: dict[str, Any] = {
    "parts": [
        {"kind": "panel", "box": {"x": 0.171, "y": 0.018, "w": 0.663, "h": 0.965}},
        {"kind": "ribbon", "box": {"x": 0.171, "y": 0.018, "w": 0.663, "h": 0.150}},
        {"kind": "progress", "box": {"x": 0.335, "y": 0.175, "w": 0.340, "h": 0.090}},
        {"kind": "close", "box": {"x": 0.762, "y": 0.087, "w": 0.028, "h": 0.048}},
        {"kind": "footer", "box": {"x": 0.392, "y": 0.880, "w": 0.216, "h": 0.070}},
        {"kind": "frame", "box": {"x": 0.254, "y": 0.278, "w": 0.081, "h": 0.268}},
        {"kind": "stars", "box": {"x": 0.276, "y": 0.258, "w": 0.040, "h": 0.048},
         "stars_count": 1},
        {"kind": "badge", "box": {"x": 0.348, "y": 0.328, "w": 0.040, "h": 0.058}},
        {"kind": "nameplate", "box": {"x": 0.254, "y": 0.488, "w": 0.081, "h": 0.048}},
        {"kind": "art", "box": {"x": 0.259, "y": 0.292, "w": 0.071, "h": 0.180}},
    ],
    "slots": {"cols": 5, "rows": 2,
              "box": {"x": 0.245, "y": 0.255, "w": 0.515, "h": 0.560}},
    "font": {"size_rel": 0.016, "color": "#FFF3D0", "stroke_color": "#6B3410",
             "stroke_width": 3, "uppercase": True},
    "palette": ["#F07C1E", "#C74A0B", "#FFD24A"],
    "summary": "экран коллекции: панель с лентой-заголовком, прогресс-бар и сетка 5×2",
}


class HTTPProvider(Provider):
    def __init__(self, pid: str, cfg: dict[str, Any]):
        self.id = pid
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.capabilities = {Capability(c) for c in cfg.get("capabilities", [])}
        self._key_env = cfg.get("api_key_env")
        self.extra_headers: dict[str, str] = cfg.get("headers", {}) or {}

    @property
    def api_key(self) -> str | None:
        # Сначала ключ, введённый в интерфейсе, потом переменная окружения.
        # Импорт локальный: app.settings тянет app.config, а тот — этот модуль.
        from app.settings import resolve_key

        return resolve_key(self.id, self._key_env)

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        key = self.api_key
        if key:
            scheme = self.cfg.get("auth_scheme", "Bearer")
            header = self.cfg.get("auth_header", "Authorization")
            h[header] = f"{scheme} {key}".strip()
        return h

    def is_configured(self) -> bool:
        """Провайдер без ключа считается недоступным.

        Локальные сервисы (A1111, ComfyUI) ключа не требуют — у них нет ни
        api_key_env, ни введённого ключа, и они остаются доступными.
        """
        if self._key_env or self.cfg.get("requires_key"):
            return bool(self.api_key)
        return True

    @staticmethod
    def _explain(status: int, body: str) -> str:
        """Достаёт человеческую причину из ответа.

        OpenRouter кладёт её в {"error": {"message": ...}}, и без разбора
        пользователь видит обрезанный JSON вместо «модели нет» или
        «кончились кредиты».
        """
        detail = body[:300]
        try:
            data = json.loads(body)
            err = data.get("error")
            if isinstance(err, dict):
                detail = err.get("message") or detail
                meta = err.get("metadata") or {}
                # В metadata обычно и лежит суть: какой провайдер отказал
                # и что именно он ответил.
                extras = [
                    str(meta[k])[:140]
                    for k in ("provider_name", "reasons", "raw", "flagged_input")
                    if meta.get(k)
                ]
                if extras:
                    detail = f"{detail} [{'; '.join(extras)}]"
            elif isinstance(err, str):
                detail = err
        except (ValueError, AttributeError):
            pass

        hint = {
            401: "ключ не принят или отозван",
            402: "недостаточно кредитов на балансе OpenRouter",
            # 403 приходит и когда закрыта модель, и когда ограничен сам
            # аккаунт. Если её отдают модели разных вендоров подряд —
            # дело почти наверняка в ключе, а не в моделях.
            403: "запрос отклонён политикой — см. Settings → Privacy и "
                 "ограничения ключа на openrouter.ai",
            404: "модель не найдена — проверь идентификатор",
            429: "превышен лимит запросов",
        }.get(status)
        return f"HTTP {status}" + (f" ({hint})" if hint else "") + f" — {detail}"

    async def _post(self, url: str, payload: dict, timeout: int) -> dict:
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers())
            except httpx.HTTPError as e:
                raise ProviderError(f"{self.id}: сеть — {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"{self.id}: {self._explain(r.status_code, r.text)}")
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderError(f"{self.id}: ответ не JSON — {r.text[:200]}") from e

        # Некоторые провайдеры отдают ошибку с кодом 200 в теле ответа.
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            raise ProviderError(f"{self.id}: {self._explain(200, r.text)}")
        return data


class OpenAICompatProvider(HTTPProvider):
    """Текст и vision через /chat/completions."""

    async def generate_text(self, model: str, req: TextRequest, timeout: int) -> TextResponse:
        content: list[dict[str, Any]] | str
        if req.images:
            content = [{"type": "text", "text": req.prompt}]
            for img in req.images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_b64(img)}"},
                })
        else:
            content = req.prompt

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update({k: v for k, v in req.params.items() if v is not None})
        if req.json_mode:
            payload["response_format"] = {"type": "json_object"}

        endpoint = self.cfg.get("endpoints", {}).get("chat", "/chat/completions")
        url = f"{self.base_url}{endpoint}"
        try:
            data = await self._post(url, payload, timeout)
        except ProviderError as e:
            # Оба параметра — вспомогательные, и ни один не стоит падения:
            # response_format дублируется просьбой в тексте, а reasoning
            # поддерживают не все модели.
            retry = False
            if req.json_mode and _is_json_mode_error(e):
                log.info("%s: модель не приняла response_format, повтор без него", self.id)
                payload.pop("response_format", None)
                payload["messages"] = [
                    *messages[:-1],
                    {"role": "user", "content": _append_json_hint(content)},
                ]
                retry = True
            if "reasoning" in payload and _is_reasoning_error(e):
                log.info("%s: модель не приняла reasoning, повтор без него", self.id)
                payload.pop("reasoning", None)
                retry = True
            if not retry:
                raise
            data = await self._post(url, payload, timeout)

        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content")
        except (KeyError, IndexError) as e:
            raise ProviderError(f"{self.id}: неожиданная форма ответа") from e

        finish = choice.get("finish_reason") or choice.get("native_finish_reason")
        usage = data.get("usage") or {}

        # Обрыв по лимиту токенов выглядит как битый JSON, и без явной
        # проверки диагностика уходит в «модель ответила не тем».
        if finish == "length":
            out = usage.get("completion_tokens")
            raise ProviderError(
                f"{self.id}: ответ обрезан по лимиту токенов"
                + (f" (сгенерировано {out})" if out else "")
                + " — увеличь max_tokens у варианта"
            )

        # Reasoning-модели тратят бюджет на рассуждения. Если он весь ушёл
        # туда, content приходит пустым, и причина совершенно неочевидна.
        if not text:
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            spent = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if reasoning or spent:
                raise ProviderError(
                    f"{self.id}: модель израсходовала бюджет на рассуждения"
                    + (f" ({spent} токенов)" if spent else "")
                    + ", до ответа не дошло — увеличь max_tokens или возьми "
                      "модель без reasoning"
                )
            raise ProviderError(
                f"{self.id}: пустой ответ (finish_reason={finish or 'нет'})"
            )

        return TextResponse(
            text=text,
            model=model,
            cost_usd=float(usage.get("cost", 0.0) or 0.0),
            raw=data,
        )

    async def key_status(self) -> dict[str, Any] | None:
        """Состояние ключа: лимиты, остаток, ограничения.

        У OpenRouter это GET /key. Нужен именно он, а не /models: /models
        публичный и отвечает даже без ключа, поэтому проверкой доступа
        служить не может.
        """
        path = self.cfg.get("endpoints", {}).get("key")
        if path is None and "openrouter.ai" in self.base_url:
            path = "/key"
        if not path:
            return None
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{self.base_url}{path}", headers=self.headers())
        except httpx.HTTPError as e:
            return {"error": f"сеть — {e}"}
        if r.status_code >= 400:
            return {"error": self._explain(r.status_code, r.text)}
        try:
            return r.json().get("data") or r.json()
        except ValueError:
            return None

    async def list_models(self, cap: Capability) -> list[str]:
        if not self.supports(cap) or not self.is_configured():
            return []
        url = f"{self.base_url}{self.cfg.get('endpoints', {}).get('models', '/models')}"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(url, headers=self.headers())
                r.raise_for_status()
                data = r.json()
        except Exception:  # noqa: BLE001 - список моделей необязателен
            return []
        out = []
        for m in data.get("data", []):
            mods = (m.get("architecture") or {}).get("output_modalities") or []
            if cap is Capability.IMAGE and "image" not in mods:
                continue
            out.append(m.get("id", ""))
        return [m for m in out if m]


class OpenRouterImageProvider(HTTPProvider):
    """Унифицированный Image API OpenRouter: POST /images."""

    @staticmethod
    def _ref_cap(model: str) -> int:
        """Жёсткие лимиты OpenRouter по вендорам (иначе HTTP 400 на input_references)."""
        m = (model or "").lower()
        if "gemini" in m:
            return 1
        if m.startswith("google/") and "image" in m:
            return 1
        if "black-forest-labs" in m or "flux" in m:
            return 4
        if "qwen" in m or "alibaba" in m:
            return 4
        if "krea" in m or "seedream" in m or "bytedance" in m:
            return 4
        return 4

    async def generate_image(self, model: str, req: ImageRequest, timeout: int) -> ImageResponse:
        # У Image API нет отдельного negative_prompt, поэтому запреты
        # дописываются в текст. Без этого весь негатив-список собирался,
        # но никуда не уходил, и модель исправно рисовала то, что запрещено.
        prompt = req.prompt
        if req.negative:
            prompt = f"{prompt}\n\nAvoid completely: {req.negative}"

        refs = list(req.reference_images or [])
        cap = self._ref_cap(model)
        if len(refs) > cap:
            log.info("%s: обрезаю референсы %d → %d для %s", self.id, len(refs), cap, model)
            refs = refs[:cap]

        endpoint = self.cfg.get("endpoints", {}).get("images", "/images")
        url = f"{self.base_url}{endpoint}"

        # Если вендор всё же отвергнет count — пробуем текущий → 1 → 0.
        attempt_caps: list[int] = []
        for c in (len(refs), 1, 0):
            if c not in attempt_caps:
                attempt_caps.append(c)

        last_err: ProviderError | None = None
        for attempt_n in attempt_caps:
            use_refs = refs[:attempt_n]
            payload: dict[str, Any] = {"model": model, "prompt": prompt}
            if req.seed is not None:
                payload["seed"] = req.seed
            if req.width and req.height:
                payload["size"] = f"{req.width}x{req.height}"
            if use_refs:
                # Поле именно input_references и именно такой формы.
                payload["input_references"] = [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_b64(b)}"}}
                    for b in use_refs
                ]
            payload.update({k: v for k, v in req.params.items() if v is not None})
            try:
                data = await self._post(url, payload, timeout)
            except ProviderError as e:
                msg = str(e).lower()
                if "input_references" in msg or (
                    "reference" in msg and "parameter" in msg
                ):
                    log.warning("%s: %s — повтор с меньшим числом референсов", self.id, e)
                    last_err = e
                    continue
                raise

            images: list[bytes] = []
            for path in ("data[*].b64_json", "data[*].image_url.url", "images[*]"):
                images = decode_images(extract_path(data, path), "base64")
                if images:
                    break
            if not images:
                raise ProviderError(f"{self.id}: в ответе нет изображений")

            return ImageResponse(
                images=images,
                model=model,
                cost_usd=float(data.get("usage", {}).get("cost", 0.0) or 0.0),
                raw={k: v for k, v in data.items() if k != "data"},
            )

        raise last_err or ProviderError(f"{self.id}: не удалось сгенерировать изображение")


class CustomRESTProvider(HTTPProvider):
    """Полностью декларативный провайдер.

    Тело запроса и путь к картинкам берутся из YAML, поэтому подключение
    внутреннего или локального сервиса — это правка конфига, а не кода.
    """

    async def generate_image(self, model: str, req: ImageRequest, timeout: int) -> ImageResponse:
        spec = self.cfg.get("image") or {}
        if not spec:
            raise ProviderError(f"{self.id}: секция image не описана в providers.yaml")

        ctx = {
            "model": model,
            "prompt": req.prompt,
            "negative": req.negative,
            "width": req.width,
            "height": req.height,
            "seed": req.seed,
            "reference_images": [_b64(b) for b in req.reference_images],
            **req.params,
        }
        payload = substitute(spec.get("body", {}), ctx)
        url = f"{self.base_url}{spec.get('endpoint', '')}"
        data = await self._post(url, payload, timeout)

        resp_spec = spec.get("response", {})
        values = extract_path(data, resp_spec.get("images", "images[*]"))
        encoding = resp_spec.get("encoding", "base64")
        if encoding == "url":
            # Часть сервисов (fal.ai и подобные) кладёт картинку в хранилище
            # и отдаёт ссылку. Без докачки такой ответ выглядел бы как
            # «картинок нет», хотя генерация прошла и уже оплачена.
            images = await self._fetch_images(
                [v for v in values if isinstance(v, str)], timeout)
        else:
            images = decode_images(values, encoding)
        if not images:
            raise ProviderError(f"{self.id}: по пути {resp_spec.get('images')} картинок нет")

        return ImageResponse(
            images=images,
            model=model,
            cost_usd=float(spec.get("cost_per_image", 0.0)) * len(images),
        )

    async def _fetch_images(self, urls: list[str], timeout: int) -> list[bytes]:
        """Докачивает картинки по ссылкам из ответа провайдера."""
        out: list[bytes] = []
        async with httpx.AsyncClient(timeout=timeout) as client:
            for u in urls:
                if not u.startswith(("http://", "https://")):
                    continue
                try:
                    r = await client.get(u)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    raise ProviderError(
                        f"{self.id}: картинка сгенерирована, но не скачалась "
                        f"({u[:80]}): {e}") from e
                out.append(r.content)
        return out

    async def generate_text(self, model: str, req: TextRequest, timeout: int) -> TextResponse:
        spec = self.cfg.get("text") or {}
        if not spec:
            raise ProviderError(f"{self.id}: секция text не описана в providers.yaml")
        ctx = {"model": model, "prompt": req.prompt, "system": req.system or "", **req.params}
        payload = substitute(spec.get("body", {}), ctx)
        data = await self._post(f"{self.base_url}{spec.get('endpoint', '')}", payload, timeout)
        got = extract_path(data, spec.get("response", {}).get("text", "text"))
        if not got:
            raise ProviderError(f"{self.id}: текст не найден в ответе")
        return TextResponse(text=str(got[0]), model=model)


class MockProvider(Provider):
    """Оффлайн-провайдер для dry-run и тестов.

    Рисует плашку с хешем промпта вместо картинки и отдаёт заглушку JSON.
    Позволяет прогнать весь пайплайн и проверить сборку без единого запроса.
    """

    def __init__(self) -> None:
        self.id = "mock"
        self.capabilities = {Capability.TEXT, Capability.VISION, Capability.IMAGE}

    def is_configured(self) -> bool:
        return True

    @staticmethod
    def _mock_extract(model: str, req: ImageRequest) -> ImageResponse:
        """Имитация image-edit: образец на хромакее.

        Настоящая модель убирает фон вокруг элемента. Мок вместо этого
        обводит присланный образец рамкой хромакея — форма и пиксели
        сохраняются, а вокруг появляется то, что должен срезать матинг.
        Так оффлайн проходит весь тракт, включая проверку покрытия.
        """
        from io import BytesIO

        from PIL import Image

        sample = req.reference_images[-1] if req.reference_images else None
        if not sample:
            return ImageResponse(images=[], model=model or "mock-image")

        src = Image.open(BytesIO(sample)).convert("RGB")
        pad = max(6, min(src.width, src.height) // 10)
        out = Image.new("RGB", (src.width + pad * 2, src.height + pad * 2),
                        (0, 255, 0))
        out.paste(src, (pad, pad))
        buf = BytesIO()
        out.save(buf, format="PNG")
        time.sleep(0.01)
        return ImageResponse(images=[buf.getvalue()], model=model or "mock-image")

    @staticmethod
    def _mock_ui_asset(model: str, req: ImageRequest) -> ImageResponse:
        """UI-элемент на хромакее — заглушка для оффлайн-проверки регенерации."""
        import math
        from io import BytesIO

        from PIL import Image, ImageDraw

        w, h = req.width, req.height
        img = Image.new("RGB", (w, h), (0, 255, 0))
        d = ImageDraw.Draw(img)
        cx, cy = w // 2, h // 2
        fill, rim = (240, 124, 30), (255, 236, 190)
        p = req.prompt

        if "five-pointed star" in p:
            r = min(w, h) * 0.34
            pts = []
            for k in range(10):
                ang = math.pi / 2 + k * math.pi / 5
                rad = r if k % 2 == 0 else r * 0.45
                pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
            d.polygon(pts, fill=(255, 200, 60), outline=(176, 112, 12))
        elif "card slot frame" in p:
            d.rounded_rectangle([w * .22, h * .10, w * .78, h * .90],
                                radius=int(w * .05), outline=(200, 215, 240),
                                width=int(w * .04))
        elif "banner ribbon" in p or "progress bar" in p or "footer plate" in p \
                or "label plate" in p or "action button" in p:
            d.rounded_rectangle([w * .08, h * .38, w * .92, h * .62],
                                radius=int(h * .08), fill=fill, outline=rim,
                                width=max(2, int(h * .02)))
        elif "X close icon" in p:
            t = int(min(w, h) * .10)
            d.line([(cx - w * .18, cy - h * .18), (cx + w * .18, cy + h * .18)],
                   fill=(170, 40, 20), width=t)
            d.line([(cx - w * .18, cy + h * .18), (cx + w * .18, cy - h * .18)],
                   fill=(170, 40, 20), width=t)
        elif "UI panel" in p:
            d.rounded_rectangle([w * .10, h * .10, w * .90, h * .90],
                                radius=int(w * .06), fill=fill, outline=rim,
                                width=max(3, int(w * .02)))
        else:   # бейдж и всё остальное
            d.rounded_rectangle([w * .30, h * .36, w * .70, h * .64],
                                radius=int(h * .08), fill=(255, 200, 60),
                                outline=(176, 112, 12), width=max(2, int(h * .02)))

        buf = BytesIO()
        img.save(buf, format="PNG")
        return ImageResponse(images=[buf.getvalue()], model=model or "mock-image")

    # Правдоподобные заглушки: определяем этап по маркеру в промпте, чтобы
    # оффлайн-прогон доходил до конца и реально собирал композит.
    _SUBJECTS = [
        "vintage film projector", "director clapperboard", "popcorn bucket",
        "spotlight lamp", "film reel stack", "velvet cinema seat",
        "golden award statuette", "ticket booth sign", "boom microphone",
        "old movie camera",
    ]

    async def generate_text(self, model: str, req: TextRequest, timeout: int) -> TextResponse:
        import json

        p = req.prompt

        # Поштучный поиск элемента: маркер — первая строка segment_one.j2.
        # Отвечаем боксом в середине зоны и точкой на нём. Зона у каждого
        # типа своя, поэтому один и тот же относительный бокс даёт разные
        # координаты на экране — этого хватает, чтобы оффлайн проверил
        # склейку координат и нарезку.
        if "Find EXACTLY ONE thing" in p:
            # Отвечаем внутри подсказанной зоны: координаты теперь в долях
            # всего кадра, и ответ по центру кадра отбраковывался бы
            # проверкой «найденное лежит вне зоны» — как и должно быть.
            import re

            m = re.search(
                r"x from ([\d.]+) to ([\d.]+),\s*\n?\s*y from ([\d.]+) to ([\d.]+)",
                p)
            if m:
                x0, x1, y0, y1 = (float(v) for v in m.groups())
            else:
                x0, y0, x1, y1 = 0.2, 0.2, 0.8, 0.8
            # Занимаем середину зоны, оставляя поля: так есть что обрезать
            # по альфе и видно, что бокс не равен зоне.
            bx = x0 + (x1 - x0) * 0.2
            by = y0 + (y1 - y0) * 0.2
            bw = max(0.01, (x1 - x0) * 0.6)
            bh = max(0.01, (y1 - y0) * 0.6)
            return TextResponse(
                text=json.dumps({"found": True,
                                 "box": {"x": round(bx, 4), "y": round(by, 4),
                                         "w": round(bw, 4), "h": round(bh, 4)},
                                 "at": [round(bx + bw / 2, 4),
                                        round(by + bh / 2, 4)],
                                 "count": 3}),
                model=model,
            )

        if '"slots"' in p and '"parts"' in p:
            # Сегментация UI-скина. Боксы взяты с реального макета экрана
            # коллекции, чтобы оффлайн-прогон резал осмысленно.
            return TextResponse(
                text=json.dumps(FALLBACK_SCREEN_MARKUP, ensure_ascii=False),
                model=model,
            )

        if '"parts"' in p and "box" in p:
            # Разбор карточки: отдаём правдоподобную разметку, чтобы
            # оффлайн-прогон реально нарезал элементы и собрал версию.
            return TextResponse(text=json.dumps({
                "parts": [
                    {"kind": "frame", "box": {"x": 0, "y": 0, "w": 1, "h": 1},
                     "note": "рамка целиком"},
                    {"kind": "stars", "box": {"x": 0.25, "y": 0.04, "w": 0.5, "h": 0.1}},
                    {"kind": "nameplate", "box": {"x": 0.11, "y": 0.82, "w": 0.78, "h": 0.12}},
                    {"kind": "badge", "box": {"x": 0.75, "y": 0.03, "w": 0.2, "h": 0.14}},
                    {"kind": "button", "box": {"x": 0.15, "y": 0.9, "w": 0.7, "h": 0.08}},
                    {"kind": "art", "box": {"x": 0.07, "y": 0.13, "w": 0.86, "h": 0.65}},
                ],
                "font": {"size_rel": 0.055, "color": "#FFF3D0",
                         "stroke_color": "#5A2E00", "stroke_width": 5,
                         "uppercase": True, "note": "оффлайн-заглушка"},
                "palette": ["#3B1F5E", "#160C2C"],
                "summary": "оффлайн-заглушка разбора",
            }, ensure_ascii=False), model=model)

        if req.images or '"passed"' in p:
            # Критик: каждая седьмая карточка «не проходит» — так видно,
            # что петля ретраев действительно работает. Хеш берём
            # устойчивый: встроенный hash() рандомизирован между запусками,
            # и тесты из-за этого плавали.
            import hashlib

            digest = hashlib.md5(p.encode("utf-8")).digest()
            fail = digest[0] % 7 == 0
            return TextResponse(
                text=json.dumps({
                    "passed": not fail,
                    "scores": {"subject_correct": True, "no_text": not fail},
                    "reason": "" if not fail else "в кадре видна надпись",
                    "fix_hint": "" if not fail else "absolutely no text or lettering",
                }),
                model=model,
            )

        if "variant_id" in p:
            n_var = 4
            for token in p.split():
                if token.isdigit() and 1 <= int(token) <= 8:
                    n_var = int(token)
                    break
            variants = []
            for v in range(n_var):
                elements = []
                for i in range(10):
                    subj = self._SUBJECTS[(v * 3 + i) % len(self._SUBJECTS)]
                    elements.append({
                        "slot": "art",
                        "subject": f"{subj} v{v + 1}",
                        "category": str((i % 4) + 1),
                        "rarity": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5][i],
                        "title_ru": subj.split()[0].upper(),
                    })
                variants.append({
                    "variant_id": f"v{v + 1}",
                    "title": f"MOCK SET {v + 1}",
                    "concept": f"оффлайн-заглушка, вариант {v + 1}",
                    "palette": [["#3B1F5E", "#160C2C"], ["#5E1F2B", "#2C0C10"],
                                ["#1F4B5E", "#0C222C"], ["#5E4B1F", "#2C220C"]][v % 4],
                    "elements": elements,
                })
            return TextResponse(text=json.dumps(variants, ensure_ascii=False), model=model)

        if '"theme"' in p:
            return TextResponse(
                text=json.dumps({
                    "theme": "mock theme", "genre": None, "mood": ["mock"],
                    "palette": ["#3B1F5E", "#160C2C"], "era": None,
                    "must_include": [], "must_avoid": [], "notes": "оффлайн-заглушка",
                }, ensure_ascii=False),
                model=model,
            )

        return TextResponse(text="{}", model=model)

    async def generate_image(self, model: str, req: ImageRequest, timeout: int) -> ImageResponse:
        from io import BytesIO

        from PIL import Image, ImageDraw

        # Перерисовка UI-элемента: рисуем фигуру на хромакее, чтобы оффлайн
        # проверялась и вырезка фона, и сборка полосок звёзд. Маркер — первая
        # строка part_regen.j2: она не меняется при правках формулировок,
        # в отличие от произвольной фразы внутри текста.
        if "recreating ONE user-interface element" in req.prompt:
            return self._mock_ui_asset(model, req)

        # Изъятие элемента: возвращаем присланный образец, положив его на
        # хромакей. Ровно то, что должна делать настоящая image-edit модель,
        # — так оффлайн проверяется весь тракт: матирование, отбор атома,
        # запись версии. Маркер — первая строка part_extract.j2.
        if "background-removal edit" in req.prompt:
            return self._mock_extract(model, req)

        seed = req.seed or abs(hash(req.prompt))
        hue = (seed % 360) / 360.0
        img = Image.new("RGB", (req.width, req.height))
        d = ImageDraw.Draw(img)
        import colorsys

        r1, g1, b1 = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.55, 0.95)]
        r2, g2, b2 = [int(c * 255) for c in colorsys.hsv_to_rgb((hue + 0.12) % 1, 0.75, 0.55)]
        for y in range(req.height):
            t = y / max(req.height - 1, 1)
            d.line(
                [(0, y), (req.width, y)],
                fill=(int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t)),
            )
        cx, cy = req.width // 2, req.height // 2
        rad = min(req.width, req.height) // 3
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(255, 255, 255, 60))
        label = req.prompt[:40]
        d.text((16, req.height - 28), label, fill="white")

        buf = BytesIO()
        img.save(buf, format="PNG")
        time.sleep(0.01)
        return ImageResponse(images=[buf.getvalue()], model=model or "mock-image")


KINDS = {
    "openai_compatible": OpenAICompatProvider,
    "openrouter_images": OpenRouterImageProvider,
    "custom_rest": CustomRESTProvider,
}


def build_provider(pid: str, cfg: dict[str, Any]) -> Provider:
    kind = cfg.get("kind", "openai_compatible")
    if kind not in KINDS:
        raise ValueError(f"Провайдер {pid}: неизвестный kind={kind}")
    return KINDS[kind](pid, cfg)
