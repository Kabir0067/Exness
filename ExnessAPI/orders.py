from __future__ import annotations

"""ExnessAPI/orders.py (PRODUCTION-GRADE)

Ин модул танҳо як кор мекунад: идоракунии позицияҳо дар MT5.

Чӣ чизҳо дар ин ҷо ҳастанд:
  1) Health-check + cache (то MT5-ро 100 бор дар як сония напурсад)
  2) Helpers: баланс, PnL (open positions), pagination (get_order_by_index)
  3) Close: close_order / close_all_position
  4) SL/TP дар USD: set_takeprofit_all_positions_usd / set_stoploss_all_positions_usd

Мавридҳои муҳим:
- Барои ҳисобкунии TP/SL аз USD мо `trade_tick_value` ва `trade_tick_size`-ро истифода мебарем.
  Агар брокер ин параметрҳоро 0 диҳад → ҳисоб имконнопазир мешавад ва позиция skip мешавад.
- Ҳамаи даъватҳои MT5 дар MT5_LOCK иҷро мешаванд.
- Логҳо ба `Logs/orders.log` (rotating) меравад.
"""

import logging
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np

from mt5_client import MT5_LOCK, ensure_mt5
from log_config import LOG_DIR as LOG_ROOT, get_log_path



# =============================================================================
# Types / constants
# =============================================================================
SLTPUnits = Literal["points", "usd", "price"]

# Default knobs (бо хоҳиш метавонед аз ҷойи дигар override кунед)
DEFAULT_DEVIATION = 50
DEFAULT_MAGIC = 123456
DEFAULT_RETRIES = 3

# Positions cache барои Telegram pagination
_POS_CACHE_TTL_SEC = 1.0

# MT5 health cache
_MT5_HEALTH_TTL_SEC = 0.5



# =============================================================================
# Logging (ERROR-only, rotating)
# =============================================================================
_ORDERS_LOG_PATH = get_log_path("orders.log")

def _ensure_rotating_handler(logger: logging.Logger, path: Path, level: int) -> None:
    """Attach RotatingFileHandler only once (no duplicates)."""

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
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(h)

log_orders = logging.getLogger("orders")
_ensure_rotating_handler(log_orders, _ORDERS_LOG_PATH, logging.ERROR)



# =============================================================================
# MT5 health (cached)
# =============================================================================
@dataclass
class _MT5HealthCache:
    ts: float = 0.0
    ok: bool = False
    trade_allowed: bool = False


_MT5_CACHE = _MT5HealthCache()


def _safe_last_error() -> str:
    try:
        return str(mt5.last_error())
    except Exception:
        return "n/a"


def _mt5_health_cached(*, require_trade_allowed: bool = False, ttl_sec: float = 1.0) -> bool:
    """Health-check бо кеш.

    require_trade_allowed=True → AutoTrading бояд фаъол бошад.
    ttl_sec → дар ин муддат натиҷа cached мемонад.
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
        log_orders.error("MT5 health check failed: %s | last_error=%s", exc, _safe_last_error())
        return False

def _ensure_mt5_connected(*, require_trade_allowed: bool = False) -> None:
    """Raises RuntimeError if MT5 is not ready."""

    ok = _mt5_health_cached(require_trade_allowed=require_trade_allowed, ttl_sec=_MT5_HEALTH_TTL_SEC)
    if ok:
        return

    # Extra context for debugging
    with MT5_LOCK:
        term = mt5.terminal_info()

    raise RuntimeError(
        "MT5 not ready | "
        f"connected={bool(term and getattr(term, 'connected', False))} "
        f"trade_allowed={bool(term and getattr(term, 'trade_allowed', False))}"
    )



# =============================================================================
# Public helpers
# =============================================================================
def enable_trading() -> None:
    """Validates that AutoTrading is enabled in MT5 terminal."""

    _ensure_mt5_connected(require_trade_allowed=True)

def get_balance() -> float:
    """Returns account balance (0.0 on failure)."""

    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            acc = mt5.account_info()

        if acc is None:
            log_orders.error("get_balance failed | last_error=%s", _safe_last_error())
            return 0.0

        bal = float(getattr(acc, "balance", 0.0) or 0.0)
        return float(bal) if np.isfinite(bal) else 0.0

    except Exception as exc:
        log_orders.error("get_balance error: %s | last_error=%s", exc, _safe_last_error())
        return 0.0

def get_positions_summary(symbol: Optional[str] = None) -> float:
    """Returns total unrealized P/L (profit) for open positions.

    symbol=None → ҳамаи позицияҳо
    symbol='XAUUSDm' → танҳо барои ҳамин symbol
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
        log_orders.error("get_positions_summary error: %s | last_error=%s", exc, _safe_last_error())
        return 0.0



# =============================================================================
# Positions cache (Telegram pagination)
# =============================================================================
_positions_cache: Dict[str, Any] = {"data": [], "ts": 0.0}

def get_order_by_index(index: int) -> Tuple[Optional[Dict[str, Any]], int]:
    """Returns open position by 0-based index and total positions count.

    Cached for 1 second to support Telegram pagination.
    """

    global _positions_cache

    try:
        now = time.time()
        if now - float(_positions_cache.get("ts", 0.0)) < _POS_CACHE_TTL_SEC and _positions_cache.get("data"):
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
        log_orders.error("get_order_by_index error: %s | last_error=%s", exc, _safe_last_error())
        return None, 0

def get_all_open_positions() -> List[Any]:
    """Returns list of all open positions (mt5.positions_get())."""

    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            positions = mt5.positions_get() or []
        return list(positions)

    except Exception as exc:
        log_orders.error("get_all_open_positions error: %s | last_error=%s", exc, _safe_last_error())
        return []


# =============================================================================
# Close helpers
# =============================================================================
def close_order(
    ticket: int,
    *,
    deviation: int = DEFAULT_DEVIATION,
    magic: int = DEFAULT_MAGIC,
    retries: int = DEFAULT_RETRIES,
) -> bool:
    """Closes a specific open position by ticket.

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

        res = None
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
            getattr(res, "retcode", None) if res else None,
            _safe_last_error(),
        )
        return False

    except Exception as exc:
        log_orders.error("close_order error ticket=%s: %s | last_error=%s", ticket, exc, _safe_last_error())
        return False

def close_all_position(*, deviation: int = DEFAULT_DEVIATION, magic: int = DEFAULT_MAGIC) -> Dict[str, Any]:
    """Closes all open positions and cancels all pending orders.

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

                ok_one = False
                last_ret: Optional[int] = None
                res = None
                for _ in range(5):
                    with MT5_LOCK:
                        res = mt5.order_send(req)
                    last_ret = int(getattr(res, "retcode", -1)) if res else None
                    if res and last_ret in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                        ok_one = True
                        break
                    time.sleep(0.35)

                if ok_one:
                    out["closed"] += 1
                else:
                    out["errors"].append(f"{ticket}: close_failed retcode={last_ret}")

            except Exception as exc_pos:
                log_orders.error("close_all_position: close position error: %s | last_error=%s", exc_pos, _safe_last_error())
                out["errors"].append("pos_close_exception")

        # 2) Cancel pending orders
        with MT5_LOCK:
            pending = mt5.orders_get() or []

        for o in pending:
            try:
                oticket = int(getattr(o, "ticket", 0) or 0)
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": oticket}

                ok_one = False
                last_ret: Optional[int] = None
                res = None
                for _ in range(3):
                    with MT5_LOCK:
                        res = mt5.order_send(req)
                    last_ret = int(getattr(res, "retcode", -1)) if res else None
                    if res and last_ret == mt5.TRADE_RETCODE_DONE:
                        ok_one = True
                        break
                    time.sleep(0.25)

                if ok_one:
                    out["canceled"] += 1
                else:
                    out["errors"].append(f"{oticket}: cancel_failed retcode={last_ret}")

            except Exception as exc_ord:
                log_orders.error("close_all_position: cancel order error: %s | last_error=%s", exc_ord, _safe_last_error())
                out["errors"].append("order_cancel_exception")

        out["last_error"] = mt5.last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = mt5.last_error()
        out["errors"].append(str(exc))
        log_orders.error("close_all_position global error: %s | last_error=%s", exc, _safe_last_error())
        return out



# =============================================================================
# USD -> TP/SL price conversion helpers
# =============================================================================
def _usd_to_tp_price_for_position(pos: Any, usd_profit: float) -> Optional[float]:
    """Converts desired profit in USD to TP price for a position.

    Формула:
      profit_per_tick_for_volume = trade_tick_value * volume
      ticks_needed = usd_profit / profit_per_tick_for_volume
      price_delta = ticks_needed * trade_tick_size

    BUY  → TP = open_price + price_delta
    SELL → TP = open_price - price_delta
    """

    try:
        symbol = str(getattr(pos, "symbol", ""))
        vol = float(getattr(pos, "volume", 0.0) or 0.0)
        open_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)

        if not symbol or vol <= 0.0 or usd_profit <= 0.0 or open_price <= 0.0:
            return None

        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
        if info is None:
            return None

        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        digits = int(getattr(info, "digits", 5) or 5)

        if tick_value <= 0.0 or tick_size <= 0.0:
            return None

        ticks_needed = float(usd_profit) / (tick_value * vol)
        if not np.isfinite(ticks_needed) or ticks_needed <= 0:
            return None

        price_delta = ticks_needed * tick_size
        tp = open_price + price_delta if ptype == mt5.POSITION_TYPE_BUY else open_price - price_delta

        tp = round(float(tp), digits)
        if tp <= 0:
            return None

        return tp

    except Exception as exc:
        log_orders.error("_usd_to_tp_price_for_position error: %s | last_error=%s", exc, _safe_last_error())
        return None

def _usd_to_sl_price_for_position(pos: Any, usd_loss: float) -> Optional[float]:
    """Converts desired loss in USD to SL price for a position.

    BUY  → SL = open_price - delta
    SELL → SL = open_price + delta
    """

    try:
        symbol = str(getattr(pos, "symbol", ""))
        vol = float(getattr(pos, "volume", 0.0) or 0.0)
        open_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)

        if not symbol or vol <= 0.0 or usd_loss <= 0.0 or open_price <= 0.0:
            return None

        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
        if info is None:
            return None

        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        digits = int(getattr(info, "digits", 5) or 5)

        if tick_value <= 0.0 or tick_size <= 0.0:
            return None

        ticks_needed = float(usd_loss) / (tick_value * vol)
        if not np.isfinite(ticks_needed) or ticks_needed <= 0:
            return None

        price_delta = ticks_needed * tick_size
        sl = open_price - price_delta if ptype == mt5.POSITION_TYPE_BUY else open_price + price_delta

        sl = round(float(sl), digits)
        if sl <= 0:
            return None

        # Direction sanity
        if ptype == mt5.POSITION_TYPE_BUY and not (sl < open_price):
            return None
        if ptype == mt5.POSITION_TYPE_SELL and not (sl > open_price):
            return None

        return sl

    except Exception as exc:
        log_orders.error("_usd_to_sl_price_for_position error: %s | last_error=%s", exc, _safe_last_error())
        return None

def _enforce_stop_distance_for_sl(symbol: str, ptype: int, sl_price: float) -> float:
    """Enforces broker minimum stop distance (trade_stops_level).

    BUY:  SL <= bid - min_dist
    SELL: SL >= ask + min_dist

    Агар stop distance 0 бошад → unchanged.
    Агар tick/info набошад → unchanged.
    """

    try:
        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)

        if not info or not tick:
            return float(sl_price)

        point = float(getattr(info, "point", 0.0) or 0.0)
        digits = int(getattr(info, "digits", 5) or 5)
        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)

        if point <= 0.0 or stops_level <= 0:
            return round(float(sl_price), digits)

        min_dist = float(stops_level) * point

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0.0 or ask <= 0.0:
            return round(float(sl_price), digits)

        if ptype == mt5.POSITION_TYPE_BUY:
            max_allowed = bid - min_dist
            sl_adj = min(float(sl_price), float(max_allowed))
        else:
            min_allowed = ask + min_dist
            sl_adj = max(float(sl_price), float(min_allowed))

        return round(float(sl_adj), digits)

    except Exception:
        return float(sl_price)



# =============================================================================
# Public: set TP/SL for ALL positions by USD
# =============================================================================
def set_takeprofit_all_positions_usd(
    usd_profit: float,
    *,
    deviation: int = DEFAULT_DEVIATION,
    magic: int = DEFAULT_MAGIC,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """Sets TP for ALL open positions by desired profit in USD (same usd_profit per position).

    SL нигоҳ дошта мешавад (ҳамон SL мемонад).
    """

    out: Dict[str, Any] = {"ok": False, "total": 0, "updated": 0, "skipped": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        out["total"] = len(positions)
        if not positions:
            out["ok"] = True
            out["last_error"] = mt5.last_error()
            return out

        for pos in positions:
            ticket = int(getattr(pos, "ticket", 0) or 0)
            symbol = str(getattr(pos, "symbol", ""))

            cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)

            tp_price = _usd_to_tp_price_for_position(pos, float(usd_profit))
            if tp_price is None:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: tp_calc_failed")
                continue

            if cur_tp > 0 and abs(cur_tp - tp_price) < 1e-10:
                out["skipped"] += 1
                continue

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(cur_sl) if cur_sl > 0 else 0.0,
                "tp": float(tp_price),
                "deviation": int(deviation),
                "magic": int(magic),
                "comment": f"tp_all_usd_{usd_profit}",
            }

            ok_one = False
            last_ret: Optional[int] = None
            res = None
            for _ in range(max(1, int(retries))):
                with MT5_LOCK:
                    res = mt5.order_send(req)
                last_ret = int(getattr(res, "retcode", -1)) if res else None
                if res and last_ret == mt5.TRADE_RETCODE_DONE:
                    ok_one = True
                    break
                time.sleep(0.25)

            if ok_one:
                out["updated"] += 1
            else:
                out["errors"].append(f"{ticket}:{symbol}: update_failed retcode={last_ret}")

        out["last_error"] = mt5.last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = mt5.last_error()
        out["errors"].append(str(exc))
        log_orders.error("set_takeprofit_all_positions_usd error: %s | last_error=%s", exc, _safe_last_error())
        return out

def set_stoploss_all_positions_usd(
    usd_loss: float,
    *,
    deviation: int = DEFAULT_DEVIATION,
    magic: int = DEFAULT_MAGIC,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """Sets SL for ALL open positions by desired loss in USD (same usd_loss per position).

    TP нигоҳ дошта мешавад (ҳамон TP мемонад).
    """

    out: Dict[str, Any] = {"ok": False, "total": 0, "updated": 0, "skipped": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        out["total"] = len(positions)
        if not positions:
            out["ok"] = True
            out["last_error"] = mt5.last_error()
            return out

        for pos in positions:
            ticket = int(getattr(pos, "ticket", 0) or 0)
            symbol = str(getattr(pos, "symbol", ""))
            ptype = int(getattr(pos, "type", 0) or 0)

            cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)

            sl_price = _usd_to_sl_price_for_position(pos, float(usd_loss))
            if sl_price is None:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: sl_calc_failed")
                continue

            # Broker constraints (min stop distance)
            sl_price = _enforce_stop_distance_for_sl(symbol, ptype, float(sl_price))
            if sl_price <= 0:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: sl_invalid_after_enforce")
                continue

            if cur_sl > 0 and abs(cur_sl - sl_price) < 1e-10:
                out["skipped"] += 1
                continue

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "symbol": symbol,
                "sl": float(sl_price),
                "tp": float(cur_tp) if cur_tp > 0 else 0.0,
                "deviation": int(deviation),
                "magic": int(magic),
                "comment": f"sl_all_usd_{usd_loss}",
            }

            ok_one = False
            last_ret: Optional[int] = None
            res = None
            for _ in range(max(1, int(retries))):
                with MT5_LOCK:
                    res = mt5.order_send(req)
                last_ret = int(getattr(res, "retcode", -1)) if res else None
                if res and last_ret == mt5.TRADE_RETCODE_DONE:
                    ok_one = True
                    break
                time.sleep(0.25)

            if ok_one:
                out["updated"] += 1
            else:
                out["errors"].append(f"{ticket}:{symbol}: update_failed retcode={last_ret}")

        out["last_error"] = mt5.last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = mt5.last_error()
        out["errors"].append(str(exc))
        log_orders.error("set_stoploss_all_positions_usd error: %s | last_error=%s", exc, _safe_last_error())
        return out



__all__ = [
    "enable_trading",
    "get_balance",
    "get_positions_summary",
    "get_order_by_index",
    "get_all_open_positions",
    "close_order",
    "close_all_position",
    "set_takeprofit_all_positions_usd",
    "set_stoploss_all_positions_usd",
]
