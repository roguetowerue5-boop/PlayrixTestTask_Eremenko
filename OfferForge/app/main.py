"""Локальный сервер OfferForge.

Аккаунтов нет: инструмент работает на машине продюсера, ключи лежат в .env.
Прогресс отдаётся через SSE — прогон идёт минутами, и молчащий спиннер
на таком отрезке бесполезен.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    ROOT,
    available_skins,
    RUNS_DIR,
    is_offline,
    list_presets,
    list_templates,
    load_preset,
    preset_index,
    providers_config,
    recommended_models,
    style_bible,
)
from app.api_assemble import router as assemble_router
from app.api_settings import router as settings_router
from app import parts as parts_lib
from app.models import Capability
from app.pipeline.run import run_offer_generation
from app.providers.registry import Registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("offerforge")

app = FastAPI(title="OfferForge", version="1.0")

STATIC_DIR = Path(__file__).parent / "static"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
app.include_router(settings_router)
app.include_router(assemble_router)

try:
    from app.api_collection_fill import router as collection_fill_router

    app.include_router(collection_fill_router)
except (ImportError, RuntimeError) as e:  # pragma: no cover
    log.error("API наполнения коллекции не подключено: %s", e)

try:
    from app.api_lora import router as lora_router

    app.include_router(lora_router)
except (ImportError, RuntimeError) as e:  # pragma: no cover
    log.error("API LoRA Trainer не подключено: %s", e)

# Коллекция грузит картинки слотов через multipart — та же зависимость, что
# у «Составляющих». Без пакета экран выключаем, остальное приложение живёт.
COLLECTION_AVAILABLE = True
COLLECTION_ERROR = ""
try:
    from app.api_collection import router as collection_router

    app.include_router(collection_router)
except (ImportError, RuntimeError) as e:  # pragma: no cover
    COLLECTION_AVAILABLE = False
    COLLECTION_ERROR = str(e)
    log.error(
        "Экран «Собрать коллекцию» выключен: %s\n"
        "    Лечится командой: pip install -r requirements.txt", e,
    )

# Экран «Составляющие» требует python-multipart (загрузка файла). Если
# зависимость не доставлена, теряется один экран — но сервер обязан
# подняться и сказать, чего не хватает. Иначе непоставленный пакет
# выглядит как «приложение не запускается», без единой подсказки.
PARTS_AVAILABLE = True
PARTS_ERROR = ""
try:
    from app.api_markup import router as markup_router
    from app.api_parts import router as parts_router

    app.include_router(parts_router)
    app.include_router(markup_router)
# FastAPI на отсутствующий python-multipart отвечает RuntimeError
# («Form data requires "python-multipart" to be installed»), а не ImportError —
# ловить надо оба, иначе сервер молча падает целиком.
except (ImportError, RuntimeError) as e:  # pragma: no cover - зависит от окружения
    PARTS_AVAILABLE = False
    PARTS_ERROR = str(e)
    log.error(
        "Экраны «Составляющие» и «Разметка» выключены: %s\n"
        "    Лечится командой: pip install -r requirements.txt", e,
    )


# ---------------------------------------------------------------------------
# Метаданные для интерфейса
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/billing")
async def get_billing() -> dict:
    """Локальные траты OfferForge + остаток на OpenRouter / fal."""
    from app.billing import billing_snapshot, load_spend, record_spend, sum_runs_cost

    spend = load_spend()
    runs = sum_runs_cost()
    # Первый запуск после обновления — подтянуть историю из runs в ledger.
    # После «Очистить бюджет» bootstrap не делаем.
    if (
        not spend.get("skip_runs_bootstrap")
        and float(spend.get("total_usd") or 0) < runs * 0.5
        and runs > 0
    ):
        gap = round(runs - float(spend.get("total_usd") or 0), 6)
        if gap > 0:
            record_spend(gap, "runs_bootstrap")
    return await billing_snapshot()


@app.delete("/api/billing")
async def reset_billing() -> dict:
    """Обнулить локальный счётчик «Потрачено суммарно»."""
    from app.billing import billing_snapshot, clear_spend

    clear_spend()
    return await billing_snapshot()


@app.get("/api/config")
async def get_config() -> dict:
    reg = Registry(providers_config(), offline=is_offline())
    presets = list_presets()
    variants: list[str] = []
    if presets:
        variants = sorted(load_preset(presets[0]["id"]).variants.keys())
    sb = style_bible()
    style_lock = (sb.get("style_lock") or "").strip()
    art_extra_default = (sb.get("art_extra_default") or style_lock).strip()
    return {
        "presets": presets,
        "default_preset": preset_index().get("default_preset"),
        "templates": list_templates(),
        "providers": reg.available(),
        "variants": variants,
        "recommended": recommended_models(),
        "offline": is_offline(),
        "parts": parts_lib.list_kinds(),
        "part_defaults": parts_lib.defaults(),
        "parts_available": PARTS_AVAILABLE,
        "parts_error": PARTS_ERROR,
        "skins": available_skins(),
        "style_lock": style_lock,
        "art_extra_default": art_extra_default,
    }


@app.get("/api/preset/{preset_id}")
async def get_preset(preset_id: str) -> dict:
    # load_preset намеренно откатывается на первый пресет при неизвестном id
    # (так пайплайн не падает от опечатки в конфиге). Для API это неверно:
    # клиент должен узнать, что такого пресета нет.
    if preset_id not in {p["id"] for p in list_presets()}:
        raise HTTPException(404, f"Пресет '{preset_id}' не найден")
    try:
        preset = load_preset(preset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    return {
        "id": preset.id,
        "name": preset.name,
        "variants": {
            name: {
                "model_name": v.model_name,
                "timeout": v.timeout,
                "rotation": [m.model_dump(exclude_none=True) for m in v.rotation()],
            }
            for name, v in preset.variants.items()
        },
    }


@app.get("/api/models")
async def get_models(capability: str = "image") -> dict:
    """Живой список моделей у провайдеров — вместо захардкоженного."""
    try:
        cap = Capability(capability)
    except ValueError as e:
        raise HTTPException(400, f"Неизвестная capability: {capability}") from e
    reg = Registry(providers_config(), offline=is_offline())
    return await reg.list_models(cap)


@app.get("/api/runs")
async def get_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        report = d / "run_report.json"
        if d.is_dir() and report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            out.append({
                "run_id": data.get("run_id", d.name),
                "theme": (data.get("brief") or {}).get("theme", ""),
                "variants": len(data.get("offers", [])),
                "cost": data.get("total_cost_usd", 0),
                "images": data.get("total_images", 0),
            })
    return out[:50]


@app.get("/api/run/{run_id}/report")
async def get_report(run_id: str):
    p = RUNS_DIR / run_id / "run_report.md"
    if not p.exists():
        raise HTTPException(404, "Отчёт не найден")
    return FileResponse(p, media_type="text/markdown")


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    theme: str
    wishes: str = ""
    template_id: str = "card_set"
    preset_id: str | None = None
    n_variants: int = 4
    concurrency: int = 4
    # Версии составляющих: {"frame": "classic", ...}. Пусто — из библиотеки.
    parts: dict[str, str] | None = None
    # Скин, из которого берутся референсы стиля для генерации арта.
    skin: str | None = None


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Прогон с потоковым прогрессом (SSE)."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(kind: str, payload: dict) -> None:
        msg = json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str)
        # Пайплайн целиком асинхронный, поэтому кладём в очередь напрямую:
        # call_soon_threadsafe только планирует колбэк, и финальное событие
        # "done" успевало проиграть гонку сентинелу None. Потоковый путь
        # оставлен на случай, если этап однажды уедет в executor.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop.call_soon_threadsafe(queue.put_nowait, msg)
        else:
            queue.put_nowait(msg)

    async def worker() -> None:
        try:
            await run_offer_generation(
                theme=req.theme,
                wishes=req.wishes,
                template_id=req.template_id,
                preset_id=req.preset_id,
                n_variants=req.n_variants,
                concurrency=req.concurrency,
                parts=req.parts,
                skin=req.skin,
                on_event=emit,
            )
        except Exception as e:  # noqa: BLE001 - любую ошибку показываем в UI
            log.exception("Прогон упал")
            emit("fatal", {"message": str(e)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())

    async def stream():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health() -> dict:
    reg = Registry(providers_config(), offline=is_offline())
    providers = reg.available()
    ready = [p for p, v in providers.items() if v["configured"] and p != "mock"]
    return {
        "ok": True,
        "offline": is_offline(),
        "providers_ready": ready,
        "root": str(ROOT),
    }
