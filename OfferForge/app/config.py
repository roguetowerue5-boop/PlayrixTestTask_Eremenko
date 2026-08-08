"""Загрузка конфигурации: провайдеры, пресеты, шаблоны офферов.

Всё, что можно вынести из кода, вынесено в YAML. Новый пресет, новый
провайдер или новый шаблон оффера добавляются файлом — как в SkyrimNet,
где index.yaml перечисляет пресеты и код об этом ничего не знает.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

from app.models import OfferTemplate, Preset, VariantConfig

ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = ROOT / "model-presets"
TEMPLATES_DIR = ROOT / "offer-templates"
PROMPTS_DIR = ROOT / "prompts"
STYLE_DIR = ROOT / "style"
RUNS_DIR = ROOT / "runs"

load_dotenv(ROOT / ".env")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Провайдеры
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def providers_config() -> dict:
    return _read_yaml(ROOT / "providers.yaml")


def invalidate_caches() -> None:
    """Сбрасывает кеши после правки конфигов из интерфейса.

    Без этого изменённый провайдер или пресет подхватились бы только
    после перезапуска сервера.
    """
    for fn in (providers_config, preset_index, recommended_models, style_bible):
        fn.cache_clear()


# ---------------------------------------------------------------------------
# Пресеты и варианты
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def preset_index() -> dict:
    return _read_yaml(PRESETS_DIR / "index.yaml")


def list_presets() -> list[dict]:
    return preset_index().get("presets", [])


def load_preset(preset_id: str | None = None) -> Preset:
    index = preset_index()
    entries = index.get("presets", [])
    if not entries:
        raise FileNotFoundError("model-presets/index.yaml пуст или не найден")

    entry = next((e for e in entries if e["id"] == preset_id), None) or entries[0]
    raw = _read_yaml(PRESETS_DIR / entry["file"])

    variants = {
        name: VariantConfig(**cfg) for name, cfg in (raw.get("variants") or {}).items()
    }
    if "default" not in variants:
        raise ValueError(f"Пресет {entry['id']}: обязателен вариант 'default'")

    return Preset(id=entry["id"], name=entry.get("name", entry["id"]), variants=variants)


def resolve_variant(preset: Preset, name: str) -> VariantConfig:
    """Неизвестный вариант наследует default — как в SkyrimNet."""
    return preset.variants.get(name) or preset.variants["default"]


@lru_cache(maxsize=1)
def recommended_models() -> dict:
    rel = preset_index().get("recommended_models", "recommended-models.yaml")
    return _read_yaml(PRESETS_DIR / rel)


# ---------------------------------------------------------------------------
# Шаблоны офферов
# ---------------------------------------------------------------------------

def list_templates() -> list[dict]:
    out = []
    for p in sorted(TEMPLATES_DIR.glob("*.yaml")):
        raw = _read_yaml(p)
        if raw:
            out.append({"id": raw.get("id", p.stem), "name": raw.get("name", p.stem), "file": p.name})
    return out


def load_template(template_id: str) -> OfferTemplate:
    for p in TEMPLATES_DIR.glob("*.yaml"):
        raw = _read_yaml(p)
        if raw.get("id") == template_id or p.stem == template_id:
            return OfferTemplate(**raw)
    raise FileNotFoundError(f"Шаблон оффера '{template_id}' не найден в {TEMPLATES_DIR}")


# ---------------------------------------------------------------------------
# Стиль
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def style_bible() -> dict:
    import json

    p = STYLE_DIR / "style_bible.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def art_references(limit: int = 3) -> list[bytes]:
    """Эталонные арты карточек — образцы того, как должна выглядеть картинка.

    Куски скина показывают палитру и материалы интерфейса, но не отвечают
    на главный вопрос: как выглядит сам арт внутри карточки. Пара готовых
    карточек из игры отвечает на него точнее любого описания словами.

    Папка наполняется из tools/build_lora_dataset.py --art-refs.
    """
    d = STYLE_DIR / "art-refs"
    if not d.is_dir():
        return []
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return [p.read_bytes() for p in files[:limit]]


def reference_images(
    limit: int = 3,
    skin: str | None = None,
    *,
    include_art: bool = True,
) -> list[bytes]:
    """Референсы стиля, которые прикладываются к каждой генерации.

    Порядок: сначала эталонные арты (подача картинки), потом элементы скина
    (палитра UI). include_art=False — только скин: Icons-арты как
    input_references часто копируются моделью вместо subject.
    """
    refs_dir = STYLE_DIR / "refs"

    art: list[bytes] = []
    if include_art:
        art = art_references(min(3, max(1, limit // 2)))
    room = limit - len(art)

    skin_refs: list[bytes] = []
    if room > 0 and skin:
        skin_dir = refs_dir / skin
        if skin_dir.is_dir():
            files = sorted(p for p in skin_dir.iterdir()
                           if p.suffix.lower() in IMG_EXTS)
            skin_refs = [p.read_bytes() for p in files[:room]]

    if room > 0 and not skin_refs and refs_dir.exists():
        files = sorted(p for p in refs_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        skin_refs = [p.read_bytes() for p in files[:room]]

    return art + skin_refs


def available_skins() -> list[str]:
    """Скины, для которых нарезаны референсы."""
    refs_dir = STYLE_DIR / "refs"
    if not refs_dir.exists():
        return []
    return sorted(d.name for d in refs_dir.iterdir()
                  if d.is_dir() and any(d.glob("*.png")))


@lru_cache(maxsize=1)
def segment_targets() -> list[dict]:
    """Что и как искать в скине — по записи на элемент.

    Список правится файлом segment-targets.yaml, а не кодом: набор
    элементов у каждого скина свой, и добавление ещё одного не должно
    требовать правки Python.
    """
    raw = _read_yaml(ROOT / "segment-targets.yaml")
    out: list[dict] = []
    for item in raw.get("targets") or []:
        kind = (item.get("kind") or "").strip()
        if not kind:
            continue
        zone = item.get("zone") or [0.0, 0.0, 1.0, 1.0]
        if len(zone) != 4:
            zone = [0.0, 0.0, 1.0, 1.0]
        out.append({
            "kind": kind,
            "title": item.get("title") or kind,
            "en": item.get("en") or kind,
            "hint": (item.get("hint") or "").strip(),
            "level": "cell" if item.get("level") == "cell" else "screen",
            "zone": [float(v) for v in zone],
            "counted": bool(item.get("counted")),
            "hollow": bool(item.get("hollow")),
        })
    return out


def save_segment_targets(targets: list[dict]) -> None:
    """Перезаписывает список целей, сохраняя пояснения в начале файла."""
    path = ROOT / "segment-targets.yaml"
    header: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("targets:"):
                break
            header.append(line)
    # Зону пишем одной строкой: четыре числа в столбик читаются плохо, а
    # править этот файл руками — обычное дело.
    class _Flow(list):
        pass

    yaml.add_representer(_Flow, lambda d, data: d.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True))
    rows = [{**t_, "zone": _Flow(t_["zone"])} for t_ in targets]
    body = yaml.dump({"targets": rows}, allow_unicode=True,
                     sort_keys=False, default_flow_style=False)
    path.write_text("\n".join(header).rstrip() + "\n\n" + body, encoding="utf-8")
    segment_targets.cache_clear()


def is_offline() -> bool:
    return os.getenv("OFFERFORGE_OFFLINE", "").lower() in {"1", "true", "yes"}
