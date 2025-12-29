from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np

from mt5_client import MT5_LOCK, ensure_mt5

SLTPUnits = Literal["points", "usd", "price"]

# =============================================================================
# Logging (ERROR by default, with rotation)
# =============================================================================
_LOG_DIR = (Path(__file__).resolve().parent.parent / "Logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_ORDERS_LOG_PATH = _LOG_DIR / "orders.log"


def _ensure_rotating_handler(logger: logging.Logger, path: Path, level: int) -> None:
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            try:
                if Path(getattr(h, "baseFilename", "")).resolve() == path.resolve():
                    h.setLevel(level)
                    return
            except Exception:
                continue

    h = RotatingFileHandler(
        filename=str(path),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,             # keep 5 backups
        encoding="utf-8",
        delay=True,
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(h)


log_orders = logging.getLogger("orders")
_ensure_rotating_handler(log_orders, _ORDERS_LOG_PATH, logging.ERROR)

# =============================================================================
# MT5 connectivity (throttled checks)
# =============================================================================
@dataclass
class _MT5HealthCache:
    ts: float = 0.0
    ok: bool = False
    trade_allowed: bool = False


_MT5_CACHE = _MT5HealthCache()


def _mt5_health_cached(*, require_trade_allowed: bool = False, ttl_sec: float = 1.0) -> bool:
    """
    Fast MT5 health check with TTL to avoid spamming terminal_info/account_info.
    """
    now = time.time()
    if (now - _MT5_CACHE.ts) < float(ttl_sec):
        return bool(_MT5_CACHE.ok and (not require_trade_allowed or _MT5_CACHE.trade_allowed))

    try:
        ensure_mt5()
        with MT5_LOCK:
            term = mt5.terminal_info()
            acc = mt5.account_info()

        ok = bool(term and getattr(term, "connected", False) and acc)
        trade_allowed = bool(term and getattr(term, "trade_allowed", False))
        _MT5_CACHE.ts = now
        _MT5_CACHE.ok = ok
        _MT5_CACHE.trade_allowed = trade_allowed
        return bool(ok and (not require_trade_allowed or trade_allowed))
    except Exception as exc:
        _MT5_CACHE.ts = now
        _MT5_CACHE.ok = False
        _MT5_CACHE.trade_allowed = False
        log_orders.error("MT5 health check failed: %s | last_error=%s", exc, mt5.last_error())
        return False


def _ensure_mt5_connected(*, require_trade_allowed: bool = False) -> None:
    """
    Ensures MT5 is connected (and optionally trade_allowed=True).
    Raises RuntimeError if not healthy.
    """
    ok = _mt5_health_cached(require_trade_allowed=require_trade_allowed, ttl_sec=0.5)
    if not ok:
        with MT5_LOCK:
            term = mt5.terminal_info()
        raise RuntimeError(
            f"MT5 not ready | connected={bool(term and getattr(term, 'connected', False))} "
            f"trade_allowed={bool(term and getattr(term, 'trade_allowed', False))}"
        )


# =============================================================================
# Public helpers
# =============================================================================
def enable_trading() -> None:
    """
    Validates that AutoTrading is enabled in MT5 terminal.
    """
    _ensure_mt5_connected(require_trade_allowed=True)


def get_balance() -> float:
    """
    Returns account balance (0.0 on failure).
    """
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            acc = mt5.account_info()
        if acc is None:
            log_orders.error("get_balance failed | last_error=%s", mt5.last_error())
            return 0.0
        bal = float(getattr(acc, "balance", 0.0) or 0.0)
        return float(bal) if np.isfinite(bal) else 0.0
    except Exception as exc:
        log_orders.error("get_balance error: %s | tb=%s", exc, _safe_last_error())
        return 0.0


def get_positions_summary(symbol: Optional[str] = None) -> float:
    """
    Returns total unrealized P/L (profit) for open positions. Optional symbol filter.
    """
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            positions = mt5.positions_get(symbol=symbol) if symbol else (mt5.positions_get() or [])
        total = 0.0
        for p in positions or []:
            pr = float(getattr(p, "profit", 0.0) or 0.0)
            if np.isfinite(pr):
                total += pr
        return float(total)
    except Exception as exc:
        log_orders.error("get_positions_summary error: %s | last_error=%s", exc, mt5.last_error())
        return 0.0


_positions_cache: Dict[str, Any] = {"data": [], "ts": 0.0}


def get_order_by_index(index: int) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Returns open position by 0-based index and total positions count.
    Cached for 1 second to support Telegram pagination.
    """
    global _positions_cache
    try:
        now = time.time()
        if now - float(_positions_cache.get("ts", 0.0)) < 1.0 and _positions_cache.get("data"):
            positions = _positions_cache["data"]
        else:
            _ensure_mt5_connected()
            with MT5_LOCK:
                positions = mt5.positions_get() or []
            _positions_cache = {"data": positions, "ts": now}

        total = len(positions)
        if total <= 0:
            return None, 0

        i = max(0, min(int(index), total - 1))
        pos = positions[i]
        return (
            {
                "ticket": int(getattr(pos, "ticket", 0) or 0),
                "symbol": str(getattr(pos, "symbol", "")),
                "type": "BUY" if int(getattr(pos, "type", 0) or 0) == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": float(getattr(pos, "volume", 0.0) or 0.0),
                "price": float(getattr(pos, "price_open", 0.0) or 0.0),
                "profit": float(getattr(pos, "profit", 0.0) or 0.0),
            },
            total,
        )
    except Exception as exc:
        log_orders.error("get_order_by_index error: %s | last_error=%s", exc, mt5.last_error())
        return None, 0


def close_order(ticket: int, *, deviation: int = 50, magic: int = 123456, retries: int = 3) -> bool:
    """
    Closes a specific open position by ticket.
    Returns True on success.
    """
    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []
        pos = next((p for p in positions if int(getattr(p, "ticket", -1)) == int(ticket)), None)
        if not pos:
            log_orders.error("close_order: position not found ticket=%s", ticket)
            return False

        symbol = str(getattr(pos, "symbol", ""))
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)

        with MT5_LOCK:
            _ = mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            log_orders.error("close_order: tick_missing symbol=%s ticket=%s", symbol, ticket)
            return False

        close_type = mt5.ORDER_TYPE_SELL if ptype == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = float(tick.bid if ptype == mt5.POSITION_TYPE_BUY else tick.ask)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": int(close_type),
            "position": int(ticket),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": "manual_close",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        for _ in range(max(1, int(retries))):
            with MT5_LOCK:
                res = mt5.order_send(req)
            if res and int(getattr(res, "retcode", -1)) in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                return True
            time.sleep(0.35)

        log_orders.error(
            "close_order failed ticket=%s symbol=%s ret=%s last_error=%s",
            ticket,
            symbol,
            getattr(res, "retcode", None) if "res" in locals() else None,
            mt5.last_error(),
        )
        return False
    except Exception as exc:
        log_orders.error("close_order error ticket=%s: %s | last_error=%s", ticket, exc, mt5.last_error())
        return False


def close_all_position(*, deviation: int = 50, magic: int = 123456) -> Dict[str, Any]:
    """
    Closes all open positions and cancels all pending orders.

    Returns:
      {
        "ok": bool,
        "closed": int,
        "canceled": int,
        "errors": [str, ...],
        "last_error": tuple
      }
    """
    out: Dict[str, Any] = {"ok": False, "closed": 0, "canceled": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        # 1) Close positions
        with MT5_LOCK:
            positions = mt5.positions_get() or []

        for pos in positions:
            try:
                ticket = int(getattr(pos, "ticket", 0) or 0)
                symbol = str(getattr(pos, "symbol", ""))
                volume = float(getattr(pos, "volume", 0.0) or 0.0)
                ptype = int(getattr(pos, "type", 0) or 0)

                with MT5_LOCK:
                    _ = mt5.symbol_select(symbol, True)
                    tick = mt5.symbol_info_tick(symbol)

                if tick is None:
                    out["errors"].append(f"{ticket}: tick_missing")
                    continue

                close_type = mt5.ORDER_TYPE_SELL if ptype == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = float(tick.bid if ptype == mt5.POSITION_TYPE_BUY else tick.ask)

                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(volume),
                    "type": int(close_type),
                    "position": int(ticket),
                    "price": float(price),
                    "deviation": int(deviation),
                    "magic": int(magic),
                    "comment": "close_all",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                ok = False
                last_ret = None
                for _ in range(5):
                    with MT5_LOCK:
                        res = mt5.order_send(req)
                    last_ret = int(getattr(res, "retcode", -1)) if res else None
                    if res and last_ret in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                        ok = True
                        break
                    time.sleep(0.35)

                if ok:
                    out["closed"] += 1
                else:
                    out["errors"].append(f"{ticket}: close_failed retcode={last_ret}")

            except Exception as exc_pos:
                log_orders.error("close_all_position: close position error: %s | last_error=%s", exc_pos, mt5.last_error())
                out["errors"].append("pos_close_exception")

        # 2) Cancel pending orders
        with MT5_LOCK:
            pending = mt5.orders_get() or []

        for o in pending:
            try:
                oticket = int(getattr(o, "ticket", 0) or 0)
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": oticket}

                ok = False
                last_ret = None
                for _ in range(3):
                    with MT5_LOCK:
                        res = mt5.order_send(req)
                    last_ret = int(getattr(res, "retcode", -1)) if res else None
                    if res and last_ret == mt5.TRADE_RETCODE_DONE:
                        ok = True
                        break
                    time.sleep(0.25)

                if ok:
                    out["canceled"] += 1
                else:
                    out["errors"].append(f"{oticket}: cancel_failed retcode={last_ret}")

            except Exception as exc_ord:
                log_orders.error("close_all_position: cancel order error: %s | last_error=%s", exc_ord, mt5.last_error())
                out["errors"].append("order_cancel_exception")

        out["last_error"] = mt5.last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = mt5.last_error()
        out["errors"].append(str(exc))
        log_orders.error("close_all_position global error: %s | last_error=%s", exc, mt5.last_error())
        return out


def _safe_last_error() -> str:
    try:
        return str(mt5.last_error())
    except Exception:
        return "n/a"


__all__ = [
    "enable_trading",
    "get_balance",
    "close_all_position",
    "get_positions_summary",
    "get_order_by_index",
    "close_order",
]
