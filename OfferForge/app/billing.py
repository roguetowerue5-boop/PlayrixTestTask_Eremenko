"""Учёт локальных трат OfferForge + балансы подключённых API."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import ROOT, RUNS_DIR
from app import settings as app_settings

log = logging.getLogger("offerforge.billing")

CONFIG_DIR = ROOT / "config"
SPEND_FILE = CONFIG_DIR / "spend.json"
_lock = threading.Lock()


def _empty() -> dict[str, Any]:
    return {
        "total_usd": 0.0,
        "by_provider": {},
        "updated_at": None,
    }


def load_spend() -> dict[str, Any]:
    if not SPEND_FILE.exists():
        return _empty()
    try:
        data = json.loads(SPEND_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty()
    data.setdefault("total_usd", 0.0)
    data.setdefault("by_provider", {})
    return data


def _save(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SPEND_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_spend(usd: float, provider_id: str = "unknown") -> float:
    """Добавляет стоимость вызова. Возвращает новый суммарный total."""
    amount = float(usd or 0)
    if amount <= 0:
        return load_spend().get("total_usd", 0.0)
    with _lock:
        data = load_spend()
        data["total_usd"] = round(float(data.get("total_usd") or 0) + amount, 6)
        by = data.setdefault("by_provider", {})
        pid = provider_id or "unknown"
        by[pid] = round(float(by.get(pid) or 0) + amount, 6)
        _save(data)
        return data["total_usd"]


def sum_runs_cost() -> float:
    """Сумма cost_usd по meta.json в runs/ (наполнение + коллекции)."""
    total = 0.0
    if not RUNS_DIR.is_dir():
        return 0.0
    for meta in RUNS_DIR.rglob("meta.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("cost_usd", "total_cost_usd"):
            v = data.get(key)
            if v is not None:
                try:
                    total += float(v)
                except (TypeError, ValueError):
                    pass
                break
        else:
            for card in data.get("cards") or []:
                try:
                    total += float(card.get("cost_usd") or 0)
                except (TypeError, ValueError):
                    pass
    return round(total, 6)


async def _openrouter_balance(key: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://offerforge.local",
        "X-Title": "OfferForge",
    }
    out: dict[str, Any] = {"provider": "openrouter", "ok": False}
    async with httpx.AsyncClient(timeout=20) as client:
        # /key работает с обычным API-ключом
        try:
            r = await client.get("https://openrouter.ai/api/v1/key", headers=headers)
            if r.status_code < 400:
                data = (r.json() or {}).get("data") or {}
                usage = data.get("usage")
                left = data.get("limit_remaining")
                out.update({
                    "ok": True,
                    "label": data.get("label"),
                    "usage_usd": float(usage) if usage is not None else None,
                    "usage_daily_usd": data.get("usage_daily"),
                    "usage_monthly_usd": data.get("usage_monthly"),
                    "limit_usd": data.get("limit"),
                    "limit_remaining_usd": float(left) if left is not None else None,
                })
            else:
                out["error"] = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            out["error"] = str(e)

        # /credits — баланс аккаунта (иногда нужен management key)
        try:
            r = await client.get(
                "https://openrouter.ai/api/v1/credits", headers=headers
            )
            if r.status_code < 400:
                data = (r.json() or {}).get("data") or {}
                credits = data.get("total_credits")
                used = data.get("total_usage")
                if credits is not None and used is not None:
                    out["account_credits_usd"] = float(credits)
                    out["account_usage_usd"] = float(used)
                    out["account_remaining_usd"] = round(
                        float(credits) - float(used), 4
                    )
                    out["ok"] = True
        except httpx.HTTPError:
            pass
    return out


async def _fal_balance(key: str) -> dict[str, Any]:
    out: dict[str, Any] = {"provider": "fal", "ok": False}
    headers = {"Authorization": f"Key {key}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://api.fal.ai/v1/account/billing",
                headers=headers,
                params={"expand": "credits"},
            )
        if r.status_code >= 400:
            out["error"] = f"HTTP {r.status_code}"
            return out
        data = r.json() or {}
        credits = data.get("credits") or {}
        bal = credits.get("current_balance")
        out.update({
            "ok": True,
            "username": data.get("username"),
            "remaining_usd": float(bal) if bal is not None else None,
            "currency": credits.get("currency") or "USD",
        })
    except httpx.HTTPError as e:
        out["error"] = str(e)
    except (ValueError, TypeError) as e:
        out["error"] = str(e)
    return out


async def billing_snapshot() -> dict[str, Any]:
    spend = load_spend()
    local_total = float(spend.get("total_usd") or 0)
    runs_total = sum_runs_cost()
    # Берём max: ledger мог отстать от runs или наоборот.
    spent_usd = round(max(local_total, runs_total), 4)

    accounts: list[dict[str, Any]] = []
    or_key = (
        app_settings.resolve_key("openrouter", "OPENROUTER_API_KEY")
        or app_settings.resolve_key("openrouter_images", "OPENROUTER_API_KEY")
    )
    if or_key:
        accounts.append(await _openrouter_balance(or_key))

    fal_key = app_settings.resolve_key("fal_lora", "FAL_KEY")
    if fal_key:
        accounts.append(await _fal_balance(fal_key))

    remaining_parts = []
    for a in accounts:
        if not a.get("ok"):
            continue
        if a.get("account_remaining_usd") is not None:
            remaining_parts.append(float(a["account_remaining_usd"]))
        elif a.get("limit_remaining_usd") is not None:
            remaining_parts.append(float(a["limit_remaining_usd"]))
        elif a.get("remaining_usd") is not None:
            remaining_parts.append(float(a["remaining_usd"]))

    return {
        "spent_usd": spent_usd,
        "ledger_usd": round(local_total, 4),
        "runs_usd": runs_total,
        "by_provider": spend.get("by_provider") or {},
        "accounts": accounts,
        "remaining_usd": (
            round(sum(remaining_parts), 4) if remaining_parts else None
        ),
        "updated_at": spend.get("updated_at"),
    }
