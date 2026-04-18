"""
ExnessAPI/history.py — Trade history and P&L reporting via MT5.

Provides cached daily/weekly/monthly profit calculations, formatted
output, and a full history summary dictionary for Telegram reporting.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Literal
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

from log_config import get_log_path
from mt5_client import MT5_LOCK, ensure_mt5

# =============================================================================
# Global Constants
# =============================================================================
_TZ_NAME = "Asia/Dushanbe"
_CACHE_TTL_SEC = 5.0
_CONN_TTL_SEC = 1.0
_PRINT_THROTTLE_SEC = 60.0

_HISTORY_LOG_PATH = get_log_path("history.log")


# =============================================================================
# Module State
# =============================================================================
_cache: Dict[str, Any] | None = None
_cache_time: datetime | None = None
_cache_mono: float | None = None

_conn_ok: bool = False
_conn_mono: float = 0.0
_last_trade_disabled_print_mono: float = 0.0

_TZ: ZoneInfo | None = None


# =============================================================================
# Logging Setup
# =============================================================================
log_history = logging.getLogger("history")


def _ensure_rotating_handler(logger: logging.Logger, path: Path, level: int) -> None:
    """Attach a rotating file handler if one for this path doesn't already exist."""
    logger.setLevel(int(level))
    logger.propagate = False

    target = str(path.resolve())

    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            try:
                existing = str(Path(getattr(handler, "baseFilename", "")).resolve())
                if existing == target:
                    handler.setLevel(int(level))
                    return
            except Exception:
                continue

    file_handler = RotatingFileHandler(
        filename=str(path),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(int(level))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        )
    )

    logger.addHandler(file_handler)


_ensure_rotating_handler(log_history, _HISTORY_LOG_PATH, logging.ERROR)


# =============================================================================
# Public API
# =============================================================================
def get_profit_period(period: Literal["day", "week", "month"]) -> dict:
    """Return profit/loss/net for the specified period."""
    if not _connect():
        return _empty_profit_dict()

    now = _local_now()

    if period == "day":
        from_date = _day_start_local(now)
    elif period == "week":
        from_date = _day_start_local(now - timedelta(days=int(now.weekday())))
    elif period == "month":
        try:
            from_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            from_date = datetime(now.year, now.month, 1)
    else:
        return _empty_profit_dict()

    try:
        with MT5_LOCK:
            deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(now))
    except Exception as exc:
        log_history.error("history_deals_get failed: %s", exc)
        return _empty_profit_dict()

    total_profit, total_loss = _calculate_profit_loss(deals)
    net = round(total_profit + total_loss, 2)

    return {"profit": total_profit, "loss": total_loss, "net": net}


def get_day_profit() -> float:
    """Return today's total profit."""
    return float(get_profit_period("day")["profit"])


def get_day_loss() -> float:
    """Return today's total loss."""
    return float(get_profit_period("day")["loss"])


def get_day_net() -> float:
    """Return today's net P&L."""
    return float(get_profit_period("day")["net"])


def get_week_profit() -> float:
    """Return this week's total profit."""
    return float(get_profit_period("week")["profit"])


def get_week_loss() -> float:
    """Return this week's total loss."""
    return float(get_profit_period("week")["loss"])


def get_week_net() -> float:
    """Return this week's net P&L."""
    return float(get_profit_period("week")["net"])


def get_month_profit() -> float:
    """Return this month's total profit."""
    return float(get_profit_period("month")["profit"])


def get_month_loss() -> float:
    """Return this month's total loss."""
    return float(get_profit_period("month")["loss"])


def get_month_net() -> float:
    """Return this month's net P&L."""
    return float(get_profit_period("month")["net"])


def format_usdt(amount: float, show_plus: bool = False) -> str:
    """Format a dollar amount with sign prefix and USDT suffix."""
    try:
        amt = float(amount)
    except Exception:
        return "0.00 USDT"

    if amt == 0.0:
        return "0.00 USDT"

    prefix = ""
    if show_plus and amt > 0:
        prefix = "+"
    elif amt < 0:
        prefix = "-"

    return f"{prefix}{abs(amt):,.2f} USDT".replace(",", " ")


def view_all_history_dict(force_refresh: bool = False) -> Dict[str, Any]:
    """Return a cached summary dict of today's trading activity."""
    global _cache, _cache_time, _cache_mono

    now = _local_now()
    now_mono = time.monotonic()

    need_refresh = (
        force_refresh
        or _cache is None
        or _cache_time is None
        or _cache_mono is None
        or (now_mono - float(_cache_mono)) > _CACHE_TTL_SEC
    )

    if not need_refresh:
        return _cache  # type: ignore

    if not _connect():
        summary = {
            "date": now.strftime("%Y-%m-%d"),
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": 0.0,
            "records": [],
        }

        _cache = summary
        _cache_time = now
        _cache_mono = now_mono
        return summary

    try:
        with MT5_LOCK:
            deals = mt5.history_deals_get(
                _naive_local(_day_start_local(now)),
                _naive_local(now),
            )
            positions = mt5.positions_get() or []
            acc = mt5.account_info()

    except Exception as exc:
        log_history.error("MT5 data fetch failed: %s", exc)
        deals, positions, acc = [], [], None

    balance = _safe_float(getattr(acc, "balance", 0.0)) if acc else 0.0

    total_profit, total_loss = _calculate_profit_loss(deals)
    net = round(total_profit + total_loss, 2)

    unrealized = sum(_safe_float(getattr(p, "profit", 0.0)) for p in positions)

    summary = {
        "date": now.strftime("%Y-%m-%d"),
        "wins": 0,
        "losses": 0,
        "total_closed": 0,
        "total_open": len(positions),
        "unrealized_pnl": round(unrealized, 2),
        "profit": total_profit,
        "loss": abs(total_loss),
        "net": net,
        "balance": round(balance, 2),
        "records": [],
    }

    _cache = summary
    _cache_time = now
    _cache_mono = now_mono

    return summary


# =============================================================================
# Private Helpers
# =============================================================================
def _safe_float(value: Any) -> float:
    """Convert to float with safe fallback."""
    try:
        return float(value or 0.0)
    except Exception as exc:
        log_history.debug("safe_float failed: %s", exc)
        return 0.0


def _local_now() -> datetime:
    """Return current time in the configured local timezone."""
    global _TZ

    if _TZ is None:
        try:
            _TZ = ZoneInfo(_TZ_NAME)
        except Exception as exc:
            log_history.warning("ZoneInfo load failed: %s", exc)
            _TZ = None

    try:
        return datetime.now(_TZ) if _TZ else datetime.now()
    except Exception as exc:
        log_history.error("datetime.now failed: %s", exc)
        return datetime.now()


def _day_start_local(now: datetime) -> datetime:
    """Return midnight of the given date."""
    try:
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        return datetime(now.year, now.month, now.day)


def _naive_local(dt: datetime) -> datetime:
    """Strip timezone info for MT5 API compatibility."""
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def _real_print(msg: str) -> None:
    """Print to the original stdout, bypassing any redirectors."""
    out = getattr(sys, "__stdout__", None)
    if out is None:
        return

    try:
        out.write(str(msg) + "\n")
        out.flush()
    except Exception:
        pass


def _empty_profit_dict() -> dict:
    """Return a zeroed profit/loss/net dictionary."""
    return {"profit": 0.0, "loss": 0.0, "net": 0.0}


def _connect() -> bool:
    """Establish and cache MT5 connectivity status with throttling."""
    global _conn_ok, _conn_mono, _last_trade_disabled_print_mono

    now_mono = time.monotonic()

    if (now_mono - _conn_mono) < _CONN_TTL_SEC:
        return bool(_conn_ok)

    ok = False

    try:
        ensure_mt5()

        with MT5_LOCK:
            term = mt5.terminal_info()
            ok = bool(
                term
                and getattr(term, "trade_allowed", False)
                and getattr(term, "connected", True)
            )
    except Exception as exc:
        log_history.error("MT5 connection failed: %s", exc)
        ok = False

    if not ok and (now_mono - _last_trade_disabled_print_mono) >= _PRINT_THROTTLE_SEC:
        _last_trade_disabled_print_mono = now_mono
        _real_print("АвтоТорговля дар MT5 хомӯш аст! Лутфан онро фаъол кунед.")

    _conn_ok = bool(ok)
    _conn_mono = now_mono

    return _conn_ok


def _calculate_profit_loss(deals) -> tuple[float, float]:
    """Compute total profit and total loss from closed deals."""
    total_profit = 0.0
    total_loss = 0.0

    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)

    for deal in deals or []:
        try:
            if getattr(deal, "entry", None) != entry_out:
                continue

            profit = _safe_float(getattr(deal, "profit", 0.0))

            if profit > 0.0:
                total_profit += profit
            elif profit < 0.0:
                total_loss += profit

        except Exception as exc:
            log_history.debug("Deal parsing failed: %s", exc)
            continue

    return round(total_profit, 2), round(total_loss, 2)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "format_usdt",
    "get_day_loss",
    "get_day_net",
    "get_day_profit",
    "get_month_loss",
    "get_month_net",
    "get_month_profit",
    "get_profit_period",
    "get_week_loss",
    "get_week_net",
    "get_week_profit",
    "view_all_history_dict",
]
