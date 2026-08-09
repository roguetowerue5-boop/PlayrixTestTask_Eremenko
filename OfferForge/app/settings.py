"""Настройки, редактируемые из интерфейса.

Три вещи, которые правятся без перезапуска и без текстового редактора:
провайдеры (адрес + ключ), модели по вариантам и промпты.

Ключи не лежат в providers.yaml: конфиг провайдеров ездит в git, ключи —
нет. Они хранятся в config/secrets.json, который в .gitignore. Переменные
окружения продолжают работать как запасной путь для CI.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.config import (
    PRESETS_DIR,
    PROMPTS_DIR,
    ROOT,
    invalidate_caches,
    preset_index,
    providers_config,
)

CONFIG_DIR = ROOT / "config"
SECRETS_FILE = CONFIG_DIR / "secrets.json"
BACKUPS_DIR = ROOT / "prompts" / ".backups"
PROVIDERS_FILE = ROOT / "providers.yaml"

MAX_BACKUPS = 20


# safe_dump не умеет сохранять комментарии, а без них формат кастомного
# провайдера перестаёт быть очевидным. Поэтому шапка пишется заново при
# каждом сохранении из интерфейса.
PROVIDERS_HEADER = """\
# Реестр провайдеров. Правится из интерфейса (AI → Провайдеры) или руками.
# Добавить свой API = добавить сюда блок, кода не трогать.
#
# kind:
#   openai_compatible — /chat/completions в формате OpenAI (текст и vision)
#   openrouter_images — унифицированный Image API OpenRouter (POST /images)
#   custom_rest       — произвольный эндпоинт: тело запроса и путь к картинкам
#                       в ответе задаются прямо здесь, в секции image/text
#
# Ключи здесь НЕ хранятся — они лежат в config/secrets.json (в .gitignore).
# api_key_env оставлен как запасной путь для CI и переменных окружения.
#
# Пример секции для custom_rest:
#   image:
#     endpoint: /v1/generate
#     cost_per_image: 0.02
#     body:                      # ${prompt} ${negative} ${width} ${height}
#       text: "${prompt}"        # ${seed} ${model} ${reference_images}
#       resolution: { width: "${width}", height: "${height}" }
#     response:
#       images: "result.artifacts[*].image_b64"
#       encoding: base64

"""

PRESET_HEADER = """\
# Пресет моделей. Правится из интерфейса (AI → Модели) или руками.
# Вариант — это роль в пайплайне, а не модель: brief, concept, copy,
# critic (нужен vision), render (генерация картинок).
# model_params — цепочка фолбэка: упала первая, берётся следующая.

"""


def _dump_yaml(path: Path, data: dict, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Ключи
# ---------------------------------------------------------------------------

def load_secrets() -> dict[str, str]:
    if not SECRETS_FILE.exists():
        return {}
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_secret(provider_id: str, key: str | None) -> None:
    """Пустое значение удаляет ключ, а не пишет пустую строку."""
    data = load_secrets()
    if key:
        data[provider_id] = key
    else:
        data.pop(provider_id, None)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        os.chmod(SECRETS_FILE, 0o600)
    except OSError:
        pass  # на Windows права работают иначе, это не повод падать


def resolve_key(provider_id: str, api_key_env: str | None) -> str | None:
    """Сначала то, что ввели в интерфейсе, потом переменная окружения."""
    stored = load_secrets().get(provider_id)
    if stored:
        return stored
    return os.getenv(api_key_env) if api_key_env else None


def mask(key: str | None) -> str:
    if not key:
        return ""
    return f"{key[:6]}…{key[-4:]}" if len(key) > 14 else "…" * len(key)


# ---------------------------------------------------------------------------
# Провайдеры
# ---------------------------------------------------------------------------

def list_providers() -> list[dict[str, Any]]:
    """Провайдеры для экрана настроек — с замаскированным ключом."""
    out = []
    for pid, cfg in (providers_config().get("providers") or {}).items():
        key = resolve_key(pid, cfg.get("api_key_env"))
        out.append({
            "id": pid,
            "kind": cfg.get("kind", "openai_compatible"),
            "base_url": cfg.get("base_url", ""),
            "endpoint": _main_endpoint(cfg),
            "capabilities": cfg.get("capabilities", []),
            "enabled": cfg.get("enabled", True),
            "api_key_env": cfg.get("api_key_env"),
            "has_key": bool(key),
            "key_masked": mask(key),
            "key_from_env": bool(
                not load_secrets().get(pid) and cfg.get("api_key_env") and key
            ),
            "auth_header": cfg.get("auth_header", "Authorization"),
            "auth_scheme": cfg.get("auth_scheme", "Bearer"),
        })
    return out


def _main_endpoint(cfg: dict) -> str:
    """Полный URL основного эндпоинта — то, что показывается в поле адреса."""
    base = (cfg.get("base_url") or "").rstrip("/")
    kind = cfg.get("kind", "openai_compatible")
    if kind == "openai_compatible":
        return base + cfg.get("endpoints", {}).get("chat", "/chat/completions")
    if kind == "openrouter_images":
        return base + cfg.get("endpoints", {}).get("images", "/images")
    return base + (cfg.get("image") or cfg.get("text") or {}).get("endpoint", "")


def _split_endpoint(cfg: dict, full_url: str) -> None:
    """Раскладывает введённый URL обратно на base_url и путь эндпоинта."""
    kind = cfg.get("kind", "openai_compatible")
    known = {
        "openai_compatible": ("endpoints", "chat", "/chat/completions"),
        "openrouter_images": ("endpoints", "images", "/images"),
    }
    url = full_url.strip().rstrip("/")

    if kind in known:
        section, key, default = known[kind]
        path = cfg.get(section, {}).get(key, default)
        if path and url.endswith(path):
            cfg["base_url"] = url[: -len(path)]
        else:
            # Путь ввели нестандартный — запоминаем его как есть.
            for marker in ("/chat/completions", "/images", "/v1"):
                idx = url.find(marker)
                if idx != -1:
                    cfg["base_url"] = url[:idx] + ("/v1" if marker != "/v1" else "/v1")
                    cfg.setdefault(section, {})[key] = url[idx:].replace("/v1", "", 1) or path
                    return
            cfg["base_url"] = url
        return

    section = "image" if cfg.get("image") else "text"
    path = (cfg.get(section) or {}).get("endpoint", "")
    if path and url.endswith(path):
        cfg["base_url"] = url[: -len(path)]
    else:
        cfg["base_url"] = url


def update_provider(provider_id: str, patch: dict[str, Any]) -> dict:
    raw = providers_config()
    providers = raw.setdefault("providers", {})
    if provider_id not in providers:
        raise KeyError(f"Провайдер '{provider_id}' не найден")

    cfg = providers[provider_id]

    if "endpoint" in patch and patch["endpoint"]:
        _split_endpoint(cfg, patch["endpoint"])
    if "base_url" in patch and patch["base_url"]:
        cfg["base_url"] = patch["base_url"].rstrip("/")
    for field in ("enabled", "auth_header", "auth_scheme", "capabilities", "kind"):
        if field in patch:
            cfg[field] = patch[field]

    _dump_yaml(PROVIDERS_FILE, raw, PROVIDERS_HEADER)

    if "api_key" in patch:
        save_secret(provider_id, (patch["api_key"] or "").strip() or None)

    invalidate_caches()
    return next(p for p in list_providers() if p["id"] == provider_id)


def create_provider(provider_id: str, cfg: dict[str, Any]) -> dict:
    raw = providers_config()
    providers = raw.setdefault("providers", {})
    if provider_id in providers:
        raise ValueError(f"Провайдер '{provider_id}' уже существует")

    kind = cfg.get("kind", "openai_compatible")
    base = (cfg.get("base_url") or cfg.get("endpoint") or "")
    is_local = any(h in base for h in ("127.0.0.1", "localhost", "0.0.0.0"))

    entry: dict[str, Any] = {
        "kind": kind,
        "base_url": (cfg.get("base_url") or "").rstrip("/"),
        "capabilities": cfg.get("capabilities") or ["text"],
        "enabled": True,
        # Локальным сервисам ключ не нужен, внешним — нужен, и без него
        # провайдер честно показывается ненастроенным.
        "requires_key": not is_local,
    }
    if kind == "openai_compatible":
        entry["endpoints"] = {"chat": "/chat/completions", "models": "/models"}
    elif kind == "openrouter_images":
        entry["endpoints"] = {"images": "/images"}
    else:
        entry["image"] = {
            "endpoint": "/generate",
            "body": {"prompt": "${prompt}"},
            "response": {"images": "images[*]", "encoding": "base64"},
        }

    if cfg.get("endpoint"):
        _split_endpoint(entry, cfg["endpoint"])

    providers[provider_id] = entry
    _dump_yaml(PROVIDERS_FILE, raw, PROVIDERS_HEADER)

    if cfg.get("api_key"):
        save_secret(provider_id, cfg["api_key"])

    invalidate_caches()
    return next(p for p in list_providers() if p["id"] == provider_id)


def clear_provider_key(provider_id: str) -> dict:
    """Убирает только ключ — сам провайдер остаётся в реестре."""
    raw = providers_config()
    if provider_id not in (raw.get("providers") or {}):
        raise KeyError(f"Провайдер '{provider_id}' не найден")
    save_secret(provider_id, None)
    invalidate_caches()
    return next(p for p in list_providers() if p["id"] == provider_id)


# Встроенные провайдеры: удаление ломает пресеты. Ключ чистится отдельно.
BUILTIN_PROVIDERS = frozenset({
    "openrouter", "openrouter_images", "fal_lora",
    "local_a1111", "internal_art_api",
})


def delete_provider(provider_id: str) -> None:
    raw = providers_config()
    if provider_id not in (raw.get("providers") or {}):
        raise KeyError(f"Провайдер '{provider_id}' не найден")
    if provider_id in BUILTIN_PROVIDERS:
        raise ValueError(
            f"'{provider_id}' — встроенный провайдер, его нельзя удалить. "
            "Чтобы убрать ключ — нажми «Очистить ключ»."
        )
    del raw["providers"][provider_id]
    _dump_yaml(PROVIDERS_FILE, raw, PROVIDERS_HEADER)
    save_secret(provider_id, None)
    invalidate_caches()


# ---------------------------------------------------------------------------
# Пресеты моделей
# ---------------------------------------------------------------------------

def _preset_entry(preset_id: str) -> dict | None:
    return next((e for e in preset_index().get("presets", []) if e["id"] == preset_id), None)


def read_preset_raw(preset_id: str) -> dict:
    entry = _preset_entry(preset_id)
    if not entry:
        raise KeyError(f"Пресет '{preset_id}' не найден")
    path = PRESETS_DIR / entry["file"]
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_preset(preset_id: str, variants: dict[str, Any]) -> None:
    entry = _preset_entry(preset_id)
    if not entry:
        raise KeyError(f"Пресет '{preset_id}' не найден")
    _dump_yaml(PRESETS_DIR / entry["file"], {"variants": variants}, PRESET_HEADER)
    invalidate_caches()


def save_preset_as(preset_id: str, name: str, variants: dict[str, Any],
                   note: str = "") -> dict:
    """Сохраняет как новый пресет и регистрирует его в манифесте.

    Ровно та же механика, что в SkyrimNet: файл в presets/ плюс строка в
    index.yaml, кода это не касается.
    """
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in preset_id.lower())
    if not slug:
        raise ValueError("Пустой идентификатор пресета")

    index = preset_index()
    presets = index.setdefault("presets", [])
    filename = f"presets/{slug}.yaml"

    _dump_yaml(PRESETS_DIR / filename, {"variants": variants}, PRESET_HEADER)

    existing = next((e for e in presets if e["id"] == slug), None)
    if existing:
        existing.update({"name": name or slug, "file": filename, "note": note})
    else:
        presets.append({"id": slug, "name": name or slug, "file": filename, "note": note})

    _dump_yaml(PRESETS_DIR / "index.yaml", index)
    invalidate_caches()
    return {"id": slug, "name": name or slug, "file": filename, "note": note}


def delete_preset(preset_id: str) -> None:
    index = preset_index()
    presets = index.get("presets", [])
    entry = next((e for e in presets if e["id"] == preset_id), None)
    if not entry:
        raise KeyError(f"Пресет '{preset_id}' не найден")
    if len(presets) <= 1:
        raise ValueError("Нельзя удалить последний пресет")

    # Сначала снимаем регистрацию в манифесте — именно она определяет, что
    # видно в интерфейсе. Файл может не удалиться (открыт, права, сетевой
    # диск), и это не повод оставлять пресет в списке.
    index["presets"] = [e for e in presets if e["id"] != preset_id]
    _dump_yaml(PRESETS_DIR / "index.yaml", index)
    invalidate_caches()

    path = PRESETS_DIR / entry["file"]
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            import logging

            logging.getLogger("offerforge.settings").warning(
                "Пресет %s снят с регистрации, но файл %s не удалён: %s",
                preset_id, path.name, e,
            )


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

def list_prompt_files() -> list[dict]:
    # Только актуальные шаблоны текущего UI (наполнение + 🎲).
    used_by = {
        "brief": "наполнение: разбор заказа",
        "collection_fill": "наполнение: 10 объектов",
        "art": "наполнение / regen: картинка",
        "name_from_art": "сборка: 🎲 имя с арта",
        "suggest_description": "наполнение: 🎲 описание",
        "suggest_art_extra": "наполнение: 🎲 доп. к арту",
    }
    out = []
    for p in sorted(PROMPTS_DIR.glob("*.j2")):
        text = p.read_text(encoding="utf-8")
        out.append({
            "name": p.stem,
            "file": p.name,
            "bytes": p.stat().st_size,
            "lines": text.count("\n") + 1,
            "backups": len(list(BACKUPS_DIR.glob(f"{p.stem}.*.j2"))) if BACKUPS_DIR.exists() else 0,
            "status": "active" if p.stem in used_by else "unused",
            "used_by": used_by.get(p.stem, ""),
        })
    return out


def read_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.j2"
    if not path.exists():
        raise KeyError(f"Промпт '{name}' не найден")
    return path.read_text(encoding="utf-8")


def write_prompt(name: str, content: str) -> dict:
    """Пишет промпт, предварительно сняв бэкап предыдущей версии."""
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("Недопустимое имя промпта")

    path = PROMPTS_DIR / f"{name}.j2"
    created = not path.exists()

    if not created:
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, BACKUPS_DIR / f"{name}.{stamp}.j2")
        old = sorted(BACKUPS_DIR.glob(f"{name}.*.j2"))
        for stale in old[:-MAX_BACKUPS]:
            stale.unlink()

    path.write_text(content, encoding="utf-8")
    return {"name": name, "created": created, "bytes": path.stat().st_size}


def validate_prompt(content: str) -> dict:
    """Проверяет синтаксис Jinja2 до сохранения — иначе поломка всплывёт
    только на середине прогона."""
    from jinja2 import Environment, TemplateSyntaxError

    from app.prompts.engine import _DECORATORS

    env = Environment()
    env.globals.update(_DECORATORS)
    try:
        parsed = env.parse(content)
    except TemplateSyntaxError as e:
        return {"valid": False, "error": f"строка {e.lineno}: {e.message}"}

    from jinja2 import meta, nodes

    # find_undeclared_variables не видит декораторы: они зарегистрированы
    # как globals, то есть формально объявлены. Поэтому вызовы ищем по AST.
    used = {
        node.node.name
        for node in parsed.find_all(nodes.Call)
        if isinstance(node.node, nodes.Name) and node.node.name in _DECORATORS
    }
    unknown = {
        node.node.name
        for node in parsed.find_all(nodes.Call)
        if isinstance(node.node, nodes.Name) and node.node.name not in _DECORATORS
    }

    return {
        "valid": True,
        "variables": sorted(meta.find_undeclared_variables(parsed) - set(_DECORATORS) - unknown),
        "decorators": sorted(used),
        "unknown_calls": sorted(unknown),
    }


def list_backups(name: str) -> list[dict]:
    if not BACKUPS_DIR.exists():
        return []
    out = []
    for p in sorted(BACKUPS_DIR.glob(f"{name}.*.j2"), reverse=True):
        stamp = p.name[len(name) + 1: -3]
        out.append({"file": p.name, "stamp": stamp, "bytes": p.stat().st_size})
    return out


def restore_backup(name: str, backup_file: str) -> str:
    if "/" in backup_file or "\\" in backup_file or ".." in backup_file:
        raise ValueError("Недопустимое имя бэкапа")
    src = BACKUPS_DIR / backup_file
    if not src.exists():
        raise KeyError(f"Бэкап '{backup_file}' не найден")
    content = src.read_text(encoding="utf-8")
    write_prompt(name, content)     # текущая версия тоже уйдёт в бэкап
    return content


def decorator_reference() -> list[dict]:
    """Справка: только декораторы, которые вызываются из активных .j2."""
    from app.prompts.engine import _DECORATORS

    used: set[str] = set()
    for path in sorted(PROMPTS_DIR.glob("*.j2")):
        text = path.read_text(encoding="utf-8")
        for deco in _DECORATORS:
            if f"{deco}(" in text:
                used.add(deco)

    return [
        {"name": name, "doc": (fn.__doc__ or "").strip().split("\n")[0]}
        for name, fn in sorted(_DECORATORS.items())
        if name in used
    ]
