"""Библиотека составляющих оффера.

Карточка собирается из готовых элементов, а не рисуется целиком. Нейросети
отдаётся только арт предмета; звёзды, рамка, кнопки, бейдж, плашка названия
и шрифт берутся из библиотеки — поэтому они одинаковы на всех карточках
по построению и не галлюцинируют.

Каждый тип составляющей может иметь несколько версий. Версия выбирается
в момент генерации, то есть сборка полностью детерминирована: один и тот же
набор версий даёт один и тот же результат.

    style/parts/
      index.yaml                 реестр типов и версий
      frame/classic/part.yaml    метаданные версии
      frame/classic/frame.png    сам ассет
      stars/classic/3.png        варианты по количеству звёзд
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config import STYLE_DIR, _read_yaml

PARTS_DIR = STYLE_DIR / "parts"
INDEX_FILE = PARTS_DIR / "index.yaml"

# Типы составляющих. Список из разбора реальной карточки: именно эти
# элементы нельзя доверять генератору — он их искажает.
PART_KINDS: dict[str, dict[str, Any]] = {
    "frame": {
        "title": "Рамка",
        "en": "card slot frame",
        "note": "Обрамление карточки с прозрачным окном под арт.",
        "assets": ["frame.png"],
    },
    "stars": {
        "title": "Звёзды",
        "en": "rarity star",
        "note": "Индикатор редкости, отдельный файл на каждое количество.",
        "assets": [f"{n}.png" for n in range(1, 6)],
    },
    "button": {
        "title": "Кнопки",
        "en": "UI button",
        "note": "Кнопка покупки и вспомогательные элементы управления.",
        "assets": ["buy.png"],
    },
    "badge": {
        "title": "Цифра с плюсом",
        "en": "quantity badge",
        "note": "Бейдж количества вида «+5» в углу карточки.",
        "assets": ["badge.png"],
    },
    "nameplate": {
        "title": "Название пака",
        "en": "name plate",
        "note": "Плашка-подложка под текст названия.",
        "assets": ["plate.png"],
    },
    "panel": {
        "title": "Панель набора",
        "en": "background panel",
        "note": "Фоновая панель экрана, на которой лежат все слоты карточек.",
        "assets": ["panel.png"],
    },
    "ribbon": {
        "title": "Лента заголовка",
        "en": "title ribbon banner",
        "note": "Лента с названием коллекции в верхней части экрана.",
        "assets": ["ribbon.png"],
    },
    "progress": {
        "title": "Прогресс-бар",
        "en": "progress bar",
        "note": "Полоса прогресса собранных карточек со счётчиком.",
        "assets": ["progress.png"],
    },
    "close": {
        "title": "Кнопка закрытия",
        "en": "close button",
        "note": "Крестик в правом верхнем углу панели.",
        "assets": ["close.png"],
    },
    "footer": {
        "title": "Футер набора",
        "en": "footer plate",
        "note": "Нижняя плашка с номером сета вида SET 4/10.",
        "assets": ["footer.png"],
    },
    "font": {
        "title": "Шрифт",
        "en": "font style",
        "note": "Начертание и стиль надписи: цвет, обводка, тень.",
        "assets": [],           # шрифт описывается метаданными, а не картинкой
    },
}

# Элементы полного экрана оффера — то, что ищет сегментация в игровом скине.
# Порядок примерно сверху вниз, в этом же виде уходит в промпт.
# close, footer и button сюда не входят намеренно: крестик нарисован на
# ленте заголовка, плашка «SET 4/10» впечатана в фоновую панель, а кнопка —
# это та же плашка имени. Отдельными элементами они дублировали то, что и
# так приезжает вместе с носителем.
SCREEN_KINDS = (
    "panel", "ribbon", "progress",
    "frame", "stars", "badge", "nameplate",
)


class FontStyle(BaseModel):
    file: str | None = None          # ttf внутри папки версии
    size: int = 40
    color: str = "#FFF3D0"
    stroke_color: str | None = "#5A2E00"
    stroke_width: int = 5
    uppercase: bool = True
    shadow_offset: tuple[int, int] | None = (0, 3)


class PartVersion(BaseModel):
    """Одна версия составляющей."""
    kind: str
    version: str
    title: str = ""
    note: str = ""
    source: str = ""                 # откуда взята: «разбор карточки», «вручную»
    created: str = ""
    assets: list[str] = Field(default_factory=list)
    # Относительные координаты в исходной карточке (0..1) — подсказка, куда
    # элемент ставился в оригинале. Помогает собрать шаблон без гадания.
    anchor: dict[str, float] | None = None
    font: FontStyle | None = None

    @property
    def dir(self) -> Path:
        return PARTS_DIR / self.kind / self.version

    def asset_path(self, name: str) -> Path:
        return self.dir / name


def _index() -> dict:
    return _read_yaml(INDEX_FILE) or {"version": 1, "parts": {}}


def _write_index(data: dict) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(".yaml.tmp")
    header = (
        "# Реестр составляющих оффера. Заполняется разбором эталонной карточки\n"
        "# (экран «Составляющие») или вручную.\n"
        "# Версия выбирается в момент генерации — сборка детерминирована.\n\n"
    )
    with tmp.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    tmp.replace(INDEX_FILE)


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------

def list_kinds() -> list[dict]:
    """Типы составляющих со всеми версиями — для интерфейса."""
    idx = _index().get("parts") or {}
    out = []
    for kind, meta in PART_KINDS.items():
        versions = []
        for ver in (idx.get(kind) or {}):
            v = load_version(kind, ver)
            if v:
                versions.append({
                    "version": v.version,
                    "title": v.title or v.version,
                    "note": v.note,
                    "source": v.source,
                    "created": v.created,
                    "assets": v.assets,
                    "has_font": v.font is not None,
                })
        out.append({
            "kind": kind,
            "title": meta["title"],
            "note": meta["note"],
            "expected_assets": meta["assets"],
            "versions": sorted(versions, key=lambda x: x["created"], reverse=True),
        })
    return out


def load_version(kind: str, version: str) -> PartVersion | None:
    path = PARTS_DIR / kind / version / "part.yaml"
    if not path.exists():
        return None
    raw = _read_yaml(path)
    if not raw:
        return None
    raw.setdefault("kind", kind)
    raw.setdefault("version", version)
    if raw.get("font"):
        raw["font"] = FontStyle(**raw["font"])
    return PartVersion(**raw)


def resolve(kind: str, version: str | None) -> PartVersion | None:
    """Версия по имени; без имени — та, что помечена по умолчанию."""
    idx = _index().get("parts") or {}
    available = idx.get(kind) or {}
    if not available:
        return None
    if version and version in available:
        return load_version(kind, version)
    default = (_index().get("defaults") or {}).get(kind)
    if default and default in available:
        return load_version(kind, default)
    return load_version(kind, sorted(available)[0])


def defaults() -> dict[str, str]:
    return _index().get("defaults") or {}


def set_default(kind: str, version: str) -> None:
    data = _index()
    data.setdefault("defaults", {})[kind] = version
    _write_index(data)


# ---------------------------------------------------------------------------
# Запись
# ---------------------------------------------------------------------------

def save_version(
    kind: str,
    version: str,
    *,
    assets: dict[str, bytes] | None = None,
    title: str = "",
    note: str = "",
    source: str = "вручную",
    anchor: dict[str, float] | None = None,
    font: FontStyle | dict | None = None,
    make_default: bool = False,
) -> PartVersion:
    if kind not in PART_KINDS:
        raise ValueError(f"Неизвестный тип составляющей: {kind}")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in version.lower())
    if not slug:
        raise ValueError("Пустое имя версии")

    vdir = PARTS_DIR / kind / slug
    vdir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name, data in (assets or {}).items():
        safe = Path(name).name
        (vdir / safe).write_bytes(data)
        written.append(safe)

    if isinstance(font, dict):
        font = FontStyle(**font)

    meta = PartVersion(
        kind=kind, version=slug,
        title=title or slug, note=note, source=source,
        created=datetime.now().isoformat(timespec="seconds"),
        assets=written or [p.name for p in vdir.iterdir() if p.suffix != ".yaml"],
        anchor=anchor, font=font,
    )
    payload = json.loads(meta.model_dump_json(exclude={"kind", "version"}))
    with (vdir / "part.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    data = _index()
    data.setdefault("parts", {}).setdefault(kind, {})[slug] = {
        "title": meta.title, "created": meta.created, "source": source,
    }
    if make_default or not (data.get("defaults") or {}).get(kind):
        data.setdefault("defaults", {})[kind] = slug
    _write_index(data)
    return meta


def delete_version(kind: str, version: str) -> None:
    data = _index()
    versions = (data.get("parts") or {}).get(kind) or {}
    if version not in versions:
        raise KeyError(f"Версия {kind}/{version} не найдена")

    # Сначала снимаем регистрацию: именно она определяет, что видно.
    del versions[version]
    if (data.get("defaults") or {}).get(kind) == version:
        data["defaults"].pop(kind, None)
        if versions:
            data["defaults"][kind] = sorted(versions)[0]
    _write_index(data)

    vdir = PARTS_DIR / kind / version
    if vdir.exists():
        try:
            shutil.rmtree(vdir)
        except OSError:
            pass    # файл занят или права — регистрация уже снята, это главное
