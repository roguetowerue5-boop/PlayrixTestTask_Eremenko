"""Абстракция провайдера.

Пайплайн никогда не знает, куда именно уходит запрос. Он просит вариант
("critic", "render"), реестр отдаёт провайдера и модель. Это позволяет
подключить любой кастомный API правкой YAML, без изменения кода.
"""
from __future__ import annotations

import abc
import base64
import re
from dataclasses import dataclass, field
from typing import Any

from app.models import Capability


@dataclass
class TextRequest:
    prompt: str
    system: str | None = None
    images: list[bytes] = field(default_factory=list)   # для vision
    params: dict[str, Any] = field(default_factory=dict)
    json_mode: bool = False


@dataclass
class TextResponse:
    text: str
    model: str
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRequest:
    prompt: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    reference_images: list[bytes] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    images: list[bytes]
    model: str
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Ошибка провайдера. Реестр ловит её и переходит к следующему в ротации."""


class Provider(abc.ABC):
    id: str
    capabilities: set[Capability]

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    async def generate_text(self, model: str, req: TextRequest, timeout: int) -> TextResponse:
        raise ProviderError(f"{self.id}: text не поддерживается")

    async def generate_image(self, model: str, req: ImageRequest, timeout: int) -> ImageResponse:
        raise ProviderError(f"{self.id}: image не поддерживается")

    async def list_models(self, cap: Capability) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Утилиты для декларативных (YAML-описанных) провайдеров
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def substitute(node: Any, ctx: dict[str, Any]) -> Any:
    """Подставляет ${name} из ctx в произвольную JSON-структуру.

    Если строка состоит ровно из одного плейсхолдера — подставляется
    значение с сохранением типа (int останется int, None выкинет ключ).
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            sv = substitute(v, ctx)
            if sv is not None:
                out[k] = sv
        return out
    if isinstance(node, list):
        return [substitute(v, ctx) for v in node]
    if isinstance(node, str):
        m = _PLACEHOLDER.fullmatch(node.strip())
        if m:
            return ctx.get(m.group(1))
        # Внутри строки отсутствующее значение — пустая строка, а не "None".
        def _one(mm: "re.Match[str]") -> str:
            v = ctx.get(mm.group(1))
            return "" if v is None else str(v)

        return _PLACEHOLDER.sub(_one, node)
    return node


def extract_path(data: Any, path: str) -> list[Any]:
    """Мини-извлекатель по пути вида `data[*].b64` или `images[0]`.

    Полноценный jsonpath тут был бы лишней зависимостью: нужны только
    точки, индексы и звёздочка.
    """
    if not path:
        return []
    current: list[Any] = [data]
    for token in path.replace("$.", "").split("."):
        if not token:
            continue
        key, _, idx = token.partition("[")
        nxt: list[Any] = []
        for item in current:
            if key:
                if not isinstance(item, dict) or key not in item:
                    continue
                item = item[key]
            if idx:
                sel = idx.rstrip("]")
                if sel == "*":
                    if isinstance(item, list):
                        nxt.extend(item)
                    continue
                try:
                    if isinstance(item, list):
                        nxt.append(item[int(sel)])
                except (ValueError, IndexError):
                    continue
            else:
                nxt.append(item)
        current = nxt
    return current


def decode_images(values: list[Any], encoding: str) -> list[bytes]:
    """Приводит извлечённые значения к байтам картинок."""
    out: list[bytes] = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v)
        elif isinstance(v, str):
            if encoding == "base64":
                payload = v.split(",", 1)[-1] if v.startswith("data:") else v
                try:
                    out.append(base64.b64decode(payload))
                except Exception:  # noqa: BLE001 - битый ответ провайдера
                    continue
    return out
