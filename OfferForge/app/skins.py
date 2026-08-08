"""Хранилище исходных скинов и ручной разметки.

Исходник нужен дважды после загрузки: как холст для разметки курсором и как
референс при перерисовке элементов. Держать его только в памяти запроса
нельзя — разметка и регенерация происходят в разные заходы.

    style/skins/<версия>/
      screen.png     исходный экран как загрузили
      markup.json    разметка: боксы элементов в долях
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import STYLE_DIR

SKINS_DIR = STYLE_DIR / "skins"


def skin_dir(version: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in version.lower())
    return SKINS_DIR / safe


def save_screen(version: str, image: bytes) -> Path:
    d = skin_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "screen.png"
    path.write_bytes(image)
    return path


def load_screen(version: str) -> bytes | None:
    path = skin_dir(version) / "screen.png"
    return path.read_bytes() if path.exists() else None


def has_screen(version: str) -> bool:
    return (skin_dir(version) / "screen.png").exists()


def save_markup(version: str, markup: dict[str, Any], *, source: str = "vision") -> None:
    d = skin_dir(version)
    d.mkdir(parents=True, exist_ok=True)
    payload = {**markup, "_source": source,
               "_saved": datetime.now().isoformat(timespec="seconds")}
    (d / "markup.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_markup(version: str) -> dict[str, Any] | None:
    path = skin_dir(version) / "markup.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_skins() -> list[dict[str, Any]]:
    """Загруженные скины с признаком наличия разметки."""
    if not SKINS_DIR.exists():
        return []
    out = []
    for d in sorted(SKINS_DIR.iterdir()):
        if not (d.is_dir() and (d / "screen.png").exists()):
            continue
        markup = load_markup(d.name)
        out.append({
            "version": d.name,
            "has_markup": markup is not None,
            "markup_source": (markup or {}).get("_source", ""),
            "parts": len((markup or {}).get("parts") or []),
            "size_kb": (d / "screen.png").stat().st_size // 1024,
        })
    return out


def delete_skin(version: str) -> None:
    import shutil

    d = skin_dir(version)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
