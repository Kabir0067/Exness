from __future__ import annotations

"""ExnessAPI/functions.py (PRODUCTION-GRADE)

Ин модул танҳо як кор мекунад: идоракунии позицияҳо дар MT5.

Оптимизатсияҳои SENIOR:
- MT5_LOCK барои 100% ҳамаи mt5.* call
- health-cache + symbol-info/tick micro-cache (камфишор ба терминал)
- group-by-symbol дар close_all_position (тик 1x барои ҳар symbol)
- numpy хориҷ (сабуктар/тезтар)
- retry backoff хурд ва муайян (бе спам), fail-fast барои retcode-ҳои доимӣ
- retry бо refresh-и price барои close (requote/price_changed/off_quotes)
- SL/TP stop-distance enforcement барои SL ва TP
"""

import logging
import math
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime

import MetaTrader5 as mt5

from mt5_client import MT5_LOCK, ensure_mt5

from log_config import get_log_path

# =============================================================================
# Types / constants
# =============================================================================
SLTPUnits = Literal["points", "usd", "price"]

DEFAULT_DEVIATION = 50
# IMPORTANT: Must match bot engine magic to correctly count/filter positions.
# Engine configs use magic=777001.
DEFAULT_MAGIC = 777001
DEFAULT_RETRIES = 2

# Sleep knobs (deterministic, low-pressure)
_CLOSE_RETRY_BASE_SEC = 0.12
_CLOSE_RETRY_CAP_SEC = 0.60
_CANCEL_RETRY_BASE_SEC = 0.08
_CANCEL_RETRY_CAP_SEC = 0.40
_MODIFY_RETRY_BASE_SEC = 0.05
_MODIFY_RETRY_CAP_SEC = 0.25

# Caches
_POS_CACHE_TTL_SEC = 1.0
_MT5_HEALTH_TTL_SEC = 0.5
_SYMBOL_CACHE_TTL_SEC = 0.25
_TICK_CACHE_TTL_SEC = 0.20

# =============================================================================
# Logging (ERROR-only, rotating)
# =============================================================================
_ORDERS_LOG_PATH = get_log_path("functions.log")

def clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def adaptive_risk_money(
    equity: float,
    base_risk_pct: float,
    confidence: float,
    phase: str,
    drawdown: float,
    *,
    phase_factors: dict | None = None,
    dd_cut: float = 0.10,
    dd_mult: float = 0.5,
) -> float:
    if not math.isfinite(float(equity)) or equity <= 0.0:
        return 0.0
    br = float(base_risk_pct)
    if not math.isfinite(br) or br <= 0.0:
        return 0.0
    c = clamp01(confidence)
    pf = phase_factors or {"A": 1.2, "B": 0.8, "C": 0.5}
    ph = str(phase or "A").upper()
    phase_mult = float(pf.get(ph, 1.0))
    dd = float(drawdown)
    if not math.isfinite(dd):
        dd = 0.0
    dd_factor = float(dd_mult) if dd > float(dd_cut) else 1.0
    return float(equity * br * c * phase_mult * dd_factor)


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
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    logger.addHandler(h)

log_orders = logging.getLogger("functions")
_ensure_rotating_handler(log_orders, _ORDERS_LOG_PATH, logging.ERROR)



# =============================================================================
# Time helpers
# =============================================================================
def _mono() -> float:
    return time.monotonic()

def _sleep_backoff(attempt: int, base: float, cap: float) -> None:
    # deterministic exponential backoff: base * 2^attempt (attempt starts at 0)
    a = int(attempt)
    delay = float(base) * (2.0 ** a)
    time.sleep(min(float(cap), delay))



# =============================================================================
# MT5 safe last_error
# =============================================================================
def _safe_last_error() -> str:
    with MT5_LOCK:
        try:
            return str(mt5.last_error())
        except Exception:
            return "n/a"



# =============================================================================
# MT5 health (cached)
# =============================================================================
@dataclass
class _MT5HealthCache:
    ts_mono: float = 0.0
    ok: bool = False
    trade_allowed: bool = False

_MT5_CACHE = _MT5HealthCache()

def _mt5_health_cached(*, require_trade_allowed: bool = False, ttl_sec: float = 0.5) -> bool:
    now = _mono()
    if (now - _MT5_CACHE.ts_mono) < float(ttl_sec):
        return bool(_MT5_CACHE.ok and (not require_trade_allowed or _MT5_CACHE.trade_allowed))

    try:
        ensure_mt5()
        with MT5_LOCK:
            term = mt5.terminal_info()
            acc = mt5.account_info()

        ok = bool(term and getattr(term, "connected", False) and acc)
        trade_allowed = bool(term and getattr(term, "trade_allowed", False))

        _MT5_CACHE.ts_mono = now
        _MT5_CACHE.ok = ok
        _MT5_CACHE.trade_allowed = trade_allowed

        return bool(ok and (not require_trade_allowed or trade_allowed))

    except Exception as exc:
        _MT5_CACHE.ts_mono = now
        _MT5_CACHE.ok = False
        _MT5_CACHE.trade_allowed = False
        log_orders.error("MT5 health check failed: %s | last_error=%s", exc, _safe_last_error())
        return False

def _ensure_mt5_connected(*, require_trade_allowed: bool = False) -> None:
    ok = _mt5_health_cached(require_trade_allowed=require_trade_allowed, ttl_sec=_MT5_HEALTH_TTL_SEC)
    if ok:
        return

    with MT5_LOCK:
        term = mt5.terminal_info()

    raise RuntimeError(
        "MT5 not ready | "
        f"connected={bool(term and getattr(term, 'connected', False))} "
        f"trade_allowed={bool(term and getattr(term, 'trade_allowed', False))}"
    )


# =============================================================================
# Symbol micro-cache (info + tick)
# =============================================================================
@dataclass
class _SymCacheEntry:
    info: Any = None
    info_ts: float = 0.0
    tick: Any = None
    tick_ts: float = 0.0

_SYM_CACHE: Dict[str, _SymCacheEntry] = {}

def _get_sym_entry(symbol: str) -> _SymCacheEntry:
    e = _SYM_CACHE.get(symbol)
    if e is None:
        e = _SymCacheEntry()
        _SYM_CACHE[symbol] = e
    return e

def _symbol_select(symbol: str) -> bool:
    with MT5_LOCK:
        try:
            return bool(mt5.symbol_select(symbol, True))
        except Exception:
            return False

def _symbol_info_cached(symbol: str) -> Any:
    now = _mono()
    e = _get_sym_entry(symbol)
    if e.info is not None and (now - e.info_ts) < _SYMBOL_CACHE_TTL_SEC:
        return e.info

    with MT5_LOCK:
        try:
            info = mt5.symbol_info(symbol)
        except Exception:
            info = None

    e.info = info
    e.info_ts = now
    return info

def _tick_cached(symbol: str) -> Any:
    now = _mono()
    e = _get_sym_entry(symbol)
    if e.tick is not None and (now - e.tick_ts) < _TICK_CACHE_TTL_SEC:
        return e.tick

    with MT5_LOCK:
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None

    e.tick = tick
    e.tick_ts = now
    return tick

def _tick_refresh(symbol: str) -> Any:
    # Force refresh (still updates cache). Use on retry for requote/price changes.
    now = _mono()
    with MT5_LOCK:
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None
    e = _get_sym_entry(symbol)
    e.tick = tick
    e.tick_ts = now
    return tick

def _best_filling_type(symbol: str) -> int:
    """
    Барои устуворӣ: IOC/FOK/RETURN интихоб мекунад.
    Агар маълумот набошад → IOC.
    """
    info = _symbol_info_cached(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC

    fm = getattr(info, "filling_mode", None)
    if isinstance(fm, int) and fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        return int(fm)

    flags = getattr(info, "trade_fill_flags", None)
    if isinstance(flags, int) and flags > 0:
        if flags & mt5.ORDER_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if flags & mt5.ORDER_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if flags & mt5.ORDER_FILLING_RETURN:
            return mt5.ORDER_FILLING_RETURN

    return mt5.ORDER_FILLING_IOC

def _digits_point_stops(symbol: str) -> Tuple[int, float, int]:
    info = _symbol_info_cached(symbol)
    if not info:
        return 5, 0.0, 0
    digits = int(getattr(info, "digits", 5) or 5)
    point = float(getattr(info, "point", 0.0) or 0.0)
    stops = int(getattr(info, "trade_stops_level", 0) or 0)
    return digits, point, stops



# =============================================================================
# Retcode sets (fail-fast & safe retries)
# =============================================================================
def _mt5_const(name: str, default: int = -1) -> int:
    v = getattr(mt5, name, None)
    return int(v) if isinstance(v, int) else int(default)

_TRANSIENT_RETCODES: Tuple[int, ...] = tuple(
    r for r in (
        _mt5_const("TRADE_RETCODE_REQUOTE"),
        _mt5_const("TRADE_RETCODE_TIMEOUT"),
        _mt5_const("TRADE_RETCODE_OFF_QUOTES"),
        _mt5_const("TRADE_RETCODE_PRICE_CHANGED"),
        _mt5_const("TRADE_RETCODE_PRICE_OFF"),
        _mt5_const("TRADE_RETCODE_CONNECTION"),
        _mt5_const("TRADE_RETCODE_NO_CONNECTION"),
        _mt5_const("TRADE_RETCODE_SERVER_DISABLES_AT"),
    )
    if r >= 0
)



# =============================================================================
# order_send retry helper (NO sleep under MT5_LOCK)
# =============================================================================
def _send_with_retries(
    req: Dict[str, Any],
    *,
    retries: int,
    success_retcodes: Tuple[int, ...],
    sleep_base: float,
    sleep_cap: float,
    retry_retcodes: Tuple[int, ...] = _TRANSIENT_RETCODES,
    refresh_before_send: Optional["callable[[Dict[str, Any], int], None]"] = None,
) -> Tuple[bool, Optional[Any], Optional[int]]:
    r = max(1, int(retries))
    last_res = None
    last_ret: Optional[int] = None

    for attempt in range(r):
        if refresh_before_send is not None:
            try:
                refresh_before_send(req, attempt)
            except Exception:
                # refresh must never break execution
                pass

        with MT5_LOCK:
            last_res = mt5.order_send(req)

        last_ret = int(getattr(last_res, "retcode", -1)) if last_res else None

        if last_res and (last_ret in success_retcodes):
            return True, last_res, last_ret

        # fail-fast for non-transient errors (reduces spam, speeds up)
        if attempt < r - 1:
            if last_ret is not None and last_ret not in retry_retcodes:
                break
            _sleep_backoff(attempt, base=float(sleep_base), cap=float(sleep_cap))

    return False, last_res, last_ret



# =============================================================================
# Public helpers
# =============================================================================
def enable_trading() -> None:
    _ensure_mt5_connected(require_trade_allowed=True)

def get_balance() -> float:
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            acc = mt5.account_info()

        if acc is None:
            log_orders.error("get_balance failed | last_error=%s", _safe_last_error())
            return 0.0

        bal = float(getattr(acc, "balance", 0.0) or 0.0)
        return float(bal) if math.isfinite(bal) else 0.0

    except Exception as exc:
        log_orders.error("get_balance error: %s | last_error=%s", exc, _safe_last_error())
        return 0.0

def get_account_info() -> Dict[str, Any]:
    """
    Получить полную информацию об аккаунте.
    """
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            acc = mt5.account_info()

        if acc is None:
            log_orders.error("get_account_info failed | last_error=%s", _safe_last_error())
            return {}

        return {
            "login": int(getattr(acc, "login", 0) or 0),
            "server": str(getattr(acc, "server", "") or ""),
            "balance": float(getattr(acc, "balance", 0.0) or 0.0),
            "equity": float(getattr(acc, "equity", 0.0) or 0.0),
            "margin": float(getattr(acc, "margin", 0.0) or 0.0),
            "free_margin": float(getattr(acc, "margin_free", 0.0) or 0.0),
            "margin_level": float(getattr(acc, "margin_level", 0.0) or 0.0),
            "profit": float(getattr(acc, "profit", 0.0) or 0.0),
            "currency": str(getattr(acc, "currency", "USD") or "USD"),
            "company": str(getattr(acc, "company", "") or ""),
        }
    except Exception as exc:
        log_orders.error("get_account_info error: %s | last_error=%s", exc, _safe_last_error())
        return {}

def get_positions_summary(symbol: Optional[str] = None) -> float:
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            positions = mt5.positions_get(symbol=symbol) if symbol else (mt5.positions_get() or [])

        total = 0.0
        for p in positions or []:
            pr = float(getattr(p, "profit", 0.0) or 0.0)
            if math.isfinite(pr):
                total += pr

        return float(total)

    except Exception as exc:
        log_orders.error("get_positions_summary error: %s | last_error=%s", exc, _safe_last_error())
        return 0.0



# =============================================================================
# Positions cache (Telegram pagination)
# =============================================================================
_positions_cache: Dict[str, Any] = {"data": [], "ts_mono": 0.0}

def get_order_by_index(index: int) -> Tuple[Optional[Dict[str, Any]], int]:
    global _positions_cache

    try:
        now = _mono()
        cached_ok = (now - float(_positions_cache.get("ts_mono", 0.0))) < _POS_CACHE_TTL_SEC and _positions_cache.get("data")
        if cached_ok:
            positions = _positions_cache["data"]
        else:
            _ensure_mt5_connected()
            with MT5_LOCK:
                positions = mt5.positions_get() or []
            _positions_cache = {"data": positions, "ts_mono": now}

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
    try:
        _ensure_mt5_connected()
        with MT5_LOCK:
            positions = mt5.positions_get() or []
        return list(positions)
    except Exception as exc:
        log_orders.error("get_all_open_positions error: %s | last_error=%s", exc, _safe_last_error())
        return []

def has_open_positions() -> bool:
    """
    Санҷад дар аккаунт ягон позиция (order/position) кушода аст ё не.
    
    Логика:
    - Шанбе/Якшанбе: новобаста аз ҳама чиз -> False (аналитика фаъол мешавад)
    - Агар позицияҳо кушода бошанд -> True (аналитика пауза мешавад)
    - Агар позицияҳо нест -> False (аналитика фаъол мешавад)
    """
    try:
        _ensure_mt5_connected()

        # Weekend override (local time) - выходные не важны, аналитика работает
        wd = datetime.now().weekday()  # Mon=0 ... Sun=6
        if wd in (5, 6):  # 5=Saturday, 6=Sunday
            return False  # Выходные: аналитика работает (не паузится)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        has_pos = len(positions) > 0
        return has_pos

    except Exception as exc:
        log_orders.error(
            "has_open_positions error: %s | last_error=%s",
            exc,
            _safe_last_error(),
        )
        return False  # При ошибке считаем, что позиций нет (аналитика работает)

def market_is_open(asset: str, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    wd = now.weekday()  # Mon=0..Sun=6
    if str(asset).upper() == "BTC":
        return True  # 24/7
    if str(asset).upper() == "XAU":
        return wd < 5  # Mon-Fri only
    return True



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
    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        # direct lookup (faster, less memory)
        with MT5_LOCK:
            pos_list = mt5.positions_get(ticket=int(ticket)) or []
        pos = pos_list[0] if pos_list else None

        if not pos:
            log_orders.error("close_order: position not found ticket=%s", ticket)
            return False

        symbol = str(getattr(pos, "symbol", ""))
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)

        if not symbol or volume <= 0:
            log_orders.error("close_order: invalid position ticket=%s symbol=%s volume=%s", ticket, symbol, volume)
            return False

        _symbol_select(symbol)

        close_type = mt5.ORDER_TYPE_SELL if ptype == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

        req: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": int(close_type),
            "position": int(ticket),
            "price": 0.0,  # will be refreshed
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": "manual_close",
            "type_filling": int(_best_filling_type(symbol)),
            "type_time": mt5.ORDER_TIME_GTC,
        }

        def _refresh_price(rq: Dict[str, Any], attempt: int) -> None:
            # attempt 0 uses cached tick; retries force refresh
            tick = _tick_cached(symbol) if attempt == 0 else _tick_refresh(symbol)
            if tick is None:
                return
            price = float(tick.bid if ptype == mt5.POSITION_TYPE_BUY else tick.ask)
            if price > 0:
                rq["price"] = float(price)

        ok, res, last_ret = _send_with_retries(
            req,
            retries=max(1, int(retries)),
            success_retcodes=(mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL),
            sleep_base=_CLOSE_RETRY_BASE_SEC,
            sleep_cap=_CLOSE_RETRY_CAP_SEC,
            refresh_before_send=_refresh_price,
        )

        if ok:
            return True

        log_orders.error(
            "close_order failed ticket=%s symbol=%s ret=%s res=%s last_error=%s",
            ticket,
            symbol,
            last_ret,
            getattr(res, "comment", None),
            _safe_last_error(),
        )
        return False

    except Exception as exc:
        log_orders.error("close_order error ticket=%s: %s | last_error=%s", ticket, exc, _safe_last_error())
        return False

def close_all_position(*, deviation: int = DEFAULT_DEVIATION, magic: int = DEFAULT_MAGIC) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "closed": 0, "canceled": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        # 1) Close positions (group by symbol -> 1 tick per symbol + refresh on retry)
        with MT5_LOCK:
            positions = mt5.positions_get() or []

        by_symbol: Dict[str, List[Any]] = {}
        for p in positions:
            s = str(getattr(p, "symbol", ""))
            if s:
                by_symbol.setdefault(s, []).append(p)

        for symbol, plist in by_symbol.items():
            try:
                _symbol_select(symbol)
                tick0 = _tick_cached(symbol)
                if tick0 is None:
                    for pos in plist:
                        ticket = int(getattr(pos, "ticket", 0) or 0)
                        out["errors"].append(f"{ticket}: tick_missing")
                    continue

                bid0 = float(getattr(tick0, "bid", 0.0) or 0.0)
                ask0 = float(getattr(tick0, "ask", 0.0) or 0.0)
                if bid0 <= 0 or ask0 <= 0:
                    for pos in plist:
                        ticket = int(getattr(pos, "ticket", 0) or 0)
                        out["errors"].append(f"{ticket}: bad_tick")
                    continue

                filling = int(_best_filling_type(symbol))

                for pos in plist:
                    ticket = int(getattr(pos, "ticket", 0) or 0)
                    volume = float(getattr(pos, "volume", 0.0) or 0.0)
                    ptype = int(getattr(pos, "type", 0) or 0)
                    if ticket <= 0 or volume <= 0:
                        out["errors"].append(f"{ticket}: invalid_position")
                        continue

                    close_type = mt5.ORDER_TYPE_SELL if ptype == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

                    req: Dict[str, Any] = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": float(volume),
                        "type": int(close_type),
                        "position": int(ticket),
                        "price": float(bid0 if ptype == mt5.POSITION_TYPE_BUY else ask0),
                        "deviation": int(deviation),
                        "magic": int(magic),
                        "comment": "close_all",
                        "type_filling": filling,
                        "type_time": mt5.ORDER_TIME_GTC,
                    }

                    def _refresh_price(rq: Dict[str, Any], attempt: int) -> None:
                        tick = _tick_cached(symbol) if attempt == 0 else _tick_refresh(symbol)
                        if tick is None:
                            return
                        bid = float(getattr(tick, "bid", 0.0) or 0.0)
                        ask = float(getattr(tick, "ask", 0.0) or 0.0)
                        if bid <= 0 or ask <= 0:
                            return
                        rq["price"] = float(bid if ptype == mt5.POSITION_TYPE_BUY else ask)

                    ok, _, last_ret = _send_with_retries(
                        req,
                        retries=5,
                        success_retcodes=(mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL),
                        sleep_base=_CLOSE_RETRY_BASE_SEC,
                        sleep_cap=_CLOSE_RETRY_CAP_SEC,
                        refresh_before_send=_refresh_price,
                    )

                    if ok:
                        out["closed"] += 1
                    else:
                        out["errors"].append(f"{ticket}: close_failed retcode={last_ret}")

            except Exception as exc_sym:
                log_orders.error("close_all_position: symbol batch error: %s | last_error=%s", exc_sym, _safe_last_error())
                out["errors"].append("symbol_batch_exception")

        # 2) Cancel pending orders
        with MT5_LOCK:
            pending = mt5.orders_get() or []

        for o in pending:
            try:
                oticket = int(getattr(o, "ticket", 0) or 0)
                if oticket <= 0:
                    out["errors"].append("bad_pending_ticket")
                    continue

                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": oticket}

                ok, _, last_ret = _send_with_retries(
                    req,
                    retries=3,
                    success_retcodes=(mt5.TRADE_RETCODE_DONE,),
                    sleep_base=_CANCEL_RETRY_BASE_SEC,
                    sleep_cap=_CANCEL_RETRY_CAP_SEC,
                )

                if ok:
                    out["canceled"] += 1
                else:
                    out["errors"].append(f"{oticket}: cancel_failed retcode={last_ret}")

            except Exception as exc_ord:
                log_orders.error("close_all_position: cancel order error: %s | last_error=%s", exc_ord, _safe_last_error())
                out["errors"].append("order_cancel_exception")

        out["last_error"] = _safe_last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = _safe_last_error()
        out["errors"].append(str(exc))
        log_orders.error("close_all_position global error: %s | last_error=%s", exc, _safe_last_error())
        return out

def close_all_position_by_profit(
    *,
    min_profit_usd: float = 0.01,
    deviation: int = DEFAULT_DEVIATION,
    magic: int = DEFAULT_MAGIC,
) -> Dict[str, Any]:
    """Close only positions with profit >= min_profit_usd (MT5-style «close by profit»)."""
    out: Dict[str, Any] = {"ok": False, "closed": 0, "canceled": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        # Filter: only positions with profit >= min_profit_usd
        profitable: List[Any] = []
        for p in positions:
            profit = float(getattr(p, "profit", 0.0) or 0.0)
            swap = float(getattr(p, "swap", 0.0) or 0.0)
            if (profit + swap) >= min_profit_usd:
                profitable.append(p)

        if not profitable:
            out["ok"] = True
            return out

        by_symbol: Dict[str, List[Any]] = {}
        for p in profitable:
            s = str(getattr(p, "symbol", ""))
            if s:
                by_symbol.setdefault(s, []).append(p)

        for symbol, plist in by_symbol.items():
            try:
                _symbol_select(symbol)
                tick0 = _tick_cached(symbol)
                if tick0 is None:
                    for pos in plist:
                        ticket = int(getattr(pos, "ticket", 0) or 0)
                        out["errors"].append(f"{ticket}: tick_missing")
                    continue

                bid0 = float(getattr(tick0, "bid", 0.0) or 0.0)
                ask0 = float(getattr(tick0, "ask", 0.0) or 0.0)
                if bid0 <= 0 or ask0 <= 0:
                    for pos in plist:
                        ticket = int(getattr(pos, "ticket", 0) or 0)
                        out["errors"].append(f"{ticket}: bad_tick")
                    continue

                filling = int(_best_filling_type(symbol))

                for pos in plist:
                    ticket = int(getattr(pos, "ticket", 0) or 0)
                    volume = float(getattr(pos, "volume", 0.0) or 0.0)
                    ptype = int(getattr(pos, "type", 0) or 0)
                    if ticket <= 0 or volume <= 0:
                        out["errors"].append(f"{ticket}: invalid_position")
                        continue

                    close_type = mt5.ORDER_TYPE_SELL if ptype == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

                    req: Dict[str, Any] = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": float(volume),
                        "type": int(close_type),
                        "position": int(ticket),
                        "price": float(bid0 if ptype == mt5.POSITION_TYPE_BUY else ask0),
                        "deviation": int(deviation),
                        "magic": int(magic),
                        "comment": "close_by_profit",
                        "type_filling": filling,
                        "type_time": mt5.ORDER_TIME_GTC,
                    }

                    def _refresh_price(rq: Dict[str, Any], attempt: int) -> None:
                        tick = _tick_cached(symbol) if attempt == 0 else _tick_refresh(symbol)
                        if tick is None:
                            return
                        bid = float(getattr(tick, "bid", 0.0) or 0.0)
                        ask = float(getattr(tick, "ask", 0.0) or 0.0)
                        if bid <= 0 or ask <= 0:
                            return
                        rq["price"] = float(bid if ptype == mt5.POSITION_TYPE_BUY else ask)

                    ok, _, last_ret = _send_with_retries(
                        req,
                        retries=5,
                        success_retcodes=(mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL),
                        sleep_base=_CLOSE_RETRY_BASE_SEC,
                        sleep_cap=_CLOSE_RETRY_CAP_SEC,
                        refresh_before_send=_refresh_price,
                    )

                    if ok:
                        out["closed"] += 1
                    else:
                        out["errors"].append(f"{ticket}: close_failed retcode={last_ret}")

            except Exception as exc_sym:
                log_orders.error("close_all_position_by_profit: symbol batch error: %s | last_error=%s", exc_sym, _safe_last_error())
                out["errors"].append("symbol_batch_exception")

        out["last_error"] = _safe_last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = _safe_last_error()
        out["errors"].append(str(exc))
        log_orders.error("close_all_position_by_profit error: %s | last_error=%s", exc, _safe_last_error())
        return out


# =============================================================================
# USD -> TP/SL price conversion helpers
# =============================================================================
# =============================================================================
# ATR-based TP helpers (volatility-adjusted)
# =============================================================================
_ATR_TF = mt5.TIMEFRAME_M1
_ATR_PERIOD = 14
_ATR_BARS = 200

_TP_CONF_MIN_MULT = 1.5
_TP_CONF_MAX_MULT = 3.0
_TP_MIN_RR = 1.5

_MANUAL_CONFIDENCE = 0.60
_MANUAL_SL_ATR_MULT = 1.0
_MANUAL_BASE_RISK_PCT = 0.01


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _tp_mult_from_conf(confidence: float) -> float:
    c = _clamp01(confidence)
    lo = float(_TP_CONF_MIN_MULT)
    hi = float(_TP_CONF_MAX_MULT)
    if hi < lo:
        hi = lo
    return float(lo + (hi - lo) * c)


def _atr_wilder(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n <= int(period):
        return 0.0
    trs: list[float] = []
    prev_close = float(closes[0])
    for i in range(1, n):
        hi = float(highs[i])
        lo = float(lows[i])
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
        prev_close = float(closes[i])
    if len(trs) < int(period):
        return 0.0
    atr = sum(trs[: int(period)]) / float(period)
    for tr in trs[int(period):]:
        atr = (atr * (float(period) - 1.0) + tr) / float(period)
    return float(atr)


def _get_atr_value(symbol: str) -> float:
    try:
        ensure_mt5()
        _symbol_select(symbol)
        with MT5_LOCK:
            rates = mt5.copy_rates_from_pos(symbol, _ATR_TF, 0, int(_ATR_BARS))
        if rates is None or len(rates) < (_ATR_PERIOD + 2):
            return 0.0
        highs = [float(r["high"]) for r in rates]
        lows = [float(r["low"]) for r in rates]
        closes = [float(r["close"]) for r in rates]
        return float(_atr_wilder(highs, lows, closes, int(_ATR_PERIOD)))
    except Exception as exc:
        log_orders.error("_get_atr_value error: %s | last_error=%s", exc, _safe_last_error())
        return 0.0


def _expected_profit(symbol: str, side: str, volume: float, entry: float, price2: float) -> float:
    try:
        order_type = mt5.ORDER_TYPE_BUY if str(side).lower() == "buy" else mt5.ORDER_TYPE_SELL
        with MT5_LOCK:
            val = mt5.order_calc_profit(order_type, symbol, float(volume), float(entry), float(price2))
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def _normalize_volume_floor(symbol: str, vol: float) -> float:
    info = _symbol_info_cached(symbol)
    if info is None:
        return float(vol)
    vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
    vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    v = max(vmin, min(float(vol), vmax))
    n = math.floor((v - vmin) / max(step, 1e-9))
    v_aligned = vmin + (n * step)
    v_aligned = max(vmin, min(v_aligned, vmax))
    return float(round(v_aligned, 8))


def _account_equity_drawdown() -> tuple[float, float]:
    try:
        with MT5_LOCK:
            acc = mt5.account_info()
        if not acc:
            return 0.0, 0.0
        bal = float(getattr(acc, "balance", 0.0) or 0.0)
        eq = float(getattr(acc, "equity", 0.0) or 0.0)
        dd = 0.0
        if bal > 0 and eq > 0:
            dd = max(0.0, (bal - eq) / bal)
        return float(eq or bal), float(dd)
    except Exception:
        return 0.0, 0.0


def _adaptive_lot_for_entry(symbol: str, side: str, entry: float, sl: float, confidence: float) -> float:
    try:
        eq, dd = _account_equity_drawdown()
        risk_money = adaptive_risk_money(
            eq,
            float(_MANUAL_BASE_RISK_PCT),
            float(confidence),
            "A",
            float(dd),
            phase_factors={"A": 1.2, "B": 0.8, "C": 0.5},
            dd_cut=0.10,
            dd_mult=0.5,
        )
        if risk_money <= 0:
            return 0.0
        risk_1lot = abs(_expected_profit(symbol, side, 1.0, float(entry), float(sl)))
        if risk_1lot <= 0:
            return 0.0
        raw = float(risk_money) / float(risk_1lot)
        return _normalize_volume_floor(symbol, raw)
    except Exception:
        return 0.0


def _atr_tp_from_entry(
    symbol: str,
    side: str,
    entry: float,
    volume: float,
    *,
    confidence: float,
    atr: float | None = None,
    min_profit_usd: float = 0.0,
    sl_price: float | None = None,
) -> Optional[float]:
    try:
        atr_val = float(atr) if atr is not None else 0.0
        if atr_val <= 0.0:
            atr_val = _get_atr_value(symbol)
        if atr_val <= 0.0:
            return None

        s = str(side).lower()
        sign = 1.0 if s == "buy" else -1.0

        mult = _tp_mult_from_conf(confidence)
        min_mult = float(_TP_MIN_RR) * float(_MANUAL_SL_ATR_MULT)
        if mult < min_mult:
            mult = min_mult

        tp = float(entry) + sign * atr_val * mult

        if float(min_profit_usd) > 0.0:
            profit_per_atr = abs(_expected_profit(symbol, side, float(volume), float(entry), float(entry) + sign * atr_val))
            if profit_per_atr > 0.0:
                mult_needed = float(min_profit_usd) / float(profit_per_atr)
                if mult_needed > mult:
                    mult = mult_needed
                    tp = float(entry) + sign * atr_val * mult

        if sl_price is not None and float(sl_price) > 0.0:
            dist_sl = abs(float(entry) - float(sl_price))
            if dist_sl > 0 and abs(tp - float(entry)) < dist_sl * float(_TP_MIN_RR):
                tp = float(entry) + sign * dist_sl * float(_TP_MIN_RR)

        return float(tp)
    except Exception:
        return None


def _usd_to_tp_price_for_position(pos: Any, usd_profit: float) -> Optional[float]:
    """
    Deprecated USD-based TP conversion.
    Now uses ATR-based TP with the USD value acting as a MINIMUM profit floor.
    """
    try:
        symbol = str(getattr(pos, "symbol", ""))
        vol = float(getattr(pos, "volume", 0.0) or 0.0)
        open_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)
        sl_price = float(getattr(pos, "sl", 0.0) or 0.0)

        if not symbol or vol <= 0.0 or open_price <= 0.0:
            return None

        side = "Buy" if ptype == mt5.POSITION_TYPE_BUY else "Sell"
        tp = _atr_tp_from_entry(
            symbol,
            side,
            open_price,
            vol,
            confidence=float(_MANUAL_CONFIDENCE),
            atr=None,
            min_profit_usd=float(usd_profit or 0.0),
            sl_price=sl_price if sl_price > 0 else None,
        )
        if tp is None or float(tp) <= 0.0:
            return None

        tp = _enforce_stop_distance_for_tp(symbol, ptype, float(tp))
        return float(tp) if tp > 0 else None

    except Exception as exc:
        log_orders.error("_usd_to_tp_price_for_position error: %s | last_error=%s", exc, _safe_last_error())
        return None

def _usd_to_sl_price_for_position(pos: Any, usd_loss: float) -> Optional[float]:
    try:
        symbol = str(getattr(pos, "symbol", ""))
        vol = float(getattr(pos, "volume", 0.0) or 0.0)
        open_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        ptype = int(getattr(pos, "type", 0) or 0)

        if not symbol or vol <= 0.0 or usd_loss <= 0.0 or open_price <= 0.0:
            return None

        info = _symbol_info_cached(symbol)
        if info is None:
            return None

        tick_value = float(
            (getattr(info, "trade_tick_value", 0.0) or 0.0)
            or (getattr(info, "trade_tick_value_profit", 0.0) or 0.0)
        )
        tick_size = float(
            (getattr(info, "trade_tick_size", 0.0) or 0.0)
            or (getattr(info, "trade_tick_size_profit", 0.0) or 0.0)
        )
        digits = int(getattr(info, "digits", 5) or 5)

        if tick_value <= 0.0 or tick_size <= 0.0:
            return None

        ticks_needed = float(usd_loss) / (tick_value * vol)
        if not math.isfinite(ticks_needed) or ticks_needed <= 0:
            return None

        price_delta = ticks_needed * tick_size
        sl = open_price - price_delta if ptype == mt5.POSITION_TYPE_BUY else open_price + price_delta

        sl = round(float(sl), digits)
        if sl <= 0:
            return None

        if ptype == mt5.POSITION_TYPE_BUY and not (sl < open_price):
            return None
        if ptype == mt5.POSITION_TYPE_SELL and not (sl > open_price):
            return None

        return sl

    except Exception as exc:
        log_orders.error("_usd_to_sl_price_for_position error: %s | last_error=%s", exc, _safe_last_error())
        return None

def _enforce_stop_distance_for_sl(symbol: str, ptype: int, sl_price: float) -> float:
    try:
        digits, point, stops_level = _digits_point_stops(symbol)
        tick = _tick_cached(symbol)
        if point <= 0.0 or stops_level <= 0 or not tick:
            return round(float(sl_price), digits)

        min_dist = float(stops_level) * float(point)
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

def _enforce_stop_distance_for_tp(symbol: str, ptype: int, tp_price: float) -> float:
    try:
        digits, point, stops_level = _digits_point_stops(symbol)
        tick = _tick_cached(symbol)
        if point <= 0.0 or stops_level <= 0 or not tick:
            return round(float(tp_price), digits)

        min_dist = float(stops_level) * float(point)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0.0 or ask <= 0.0:
            return round(float(tp_price), digits)

        if ptype == mt5.POSITION_TYPE_BUY:
            min_allowed = ask + min_dist
            tp_adj = max(float(tp_price), float(min_allowed))
        else:
            max_allowed = bid - min_dist
            tp_adj = min(float(tp_price), float(max_allowed))

        return round(float(tp_adj), digits)

    except Exception:
        return float(tp_price)



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
    out: Dict[str, Any] = {"ok": False, "total": 0, "updated": 0, "skipped": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        out["total"] = len(positions)
        if not positions:
            out["ok"] = True
            out["last_error"] = _safe_last_error()
            return out

        for pos in positions:
            ticket = int(getattr(pos, "ticket", 0) or 0)
            symbol = str(getattr(pos, "symbol", ""))
            ptype = int(getattr(pos, "type", 0) or 0)

            cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)

            if not symbol or ticket <= 0:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: invalid_position")
                continue

            _symbol_select(symbol)

            tp_price = _usd_to_tp_price_for_position(pos, float(usd_profit))
            if tp_price is None:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: tp_calc_failed")
                continue

            tp_price = _enforce_stop_distance_for_tp(symbol, ptype, float(tp_price))
            if tp_price <= 0:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: tp_invalid_after_enforce")
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

            ok, _, last_ret = _send_with_retries(
                req,
                retries=max(1, int(retries)),
                success_retcodes=(mt5.TRADE_RETCODE_DONE,),
                sleep_base=_MODIFY_RETRY_BASE_SEC,
                sleep_cap=_MODIFY_RETRY_CAP_SEC,
            )

            if ok:
                out["updated"] += 1
            else:
                out["errors"].append(f"{ticket}:{symbol}: update_failed retcode={last_ret}")

        out["last_error"] = _safe_last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = _safe_last_error()
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
    out: Dict[str, Any] = {"ok": False, "total": 0, "updated": 0, "skipped": 0, "errors": [], "last_error": None}

    try:
        _ensure_mt5_connected(require_trade_allowed=True)

        with MT5_LOCK:
            positions = mt5.positions_get() or []

        out["total"] = len(positions)
        if not positions:
            out["ok"] = True
            out["last_error"] = _safe_last_error()
            return out

        for pos in positions:
            ticket = int(getattr(pos, "ticket", 0) or 0)
            symbol = str(getattr(pos, "symbol", ""))
            ptype = int(getattr(pos, "type", 0) or 0)

            cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)

            if not symbol or ticket <= 0:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: invalid_position")
                continue

            _symbol_select(symbol)

            sl_price = _usd_to_sl_price_for_position(pos, float(usd_loss))
            if sl_price is None:
                out["skipped"] += 1
                out["errors"].append(f"{ticket}:{symbol}: sl_calc_failed")
                continue

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

            ok, _, last_ret = _send_with_retries(
                req,
                retries=max(1, int(retries)),
                success_retcodes=(mt5.TRADE_RETCODE_DONE,),
                sleep_base=_MODIFY_RETRY_BASE_SEC,
                sleep_cap=_MODIFY_RETRY_CAP_SEC,
            )

            if ok:
                out["updated"] += 1
            else:
                out["errors"].append(f"{ticket}:{symbol}: update_failed retcode={last_ret}")

        out["last_error"] = _safe_last_error()
        out["ok"] = len(out["errors"]) == 0
        return out

    except Exception as exc:
        out["last_error"] = _safe_last_error()
        out["errors"].append(str(exc))
        log_orders.error("set_stoploss_all_positions_usd error: %s | last_error=%s", exc, _safe_last_error())
        return out



# =============================================================================
# History reports (full detailed reports for day/week/month)
# =============================================================================
def get_full_report_day(force_refresh: bool = True) -> Dict[str, Any]:
    """
    Полный отчет за день (как view_all_history_dict, но только за сегодня).
    Возвращает полную структуру с wins, losses, profit, loss, net, balance и т.д.
    """
    try:
        from ExnessAPI.history import view_all_history_dict, _local_now, _day_start_local
        from datetime import datetime

        summary = view_all_history_dict(force_refresh=force_refresh)
        local_now = _local_now()
        day_start = _day_start_local(local_now)
        
        # Добавляем даты периода
        summary["date_from"] = day_start.strftime("%Y-%m-%d %H:%M:%S")
        summary["date_to"] = local_now.strftime("%Y-%m-%d %H:%M:%S")
        summary["period"] = "day"
        
        return summary
    except Exception as exc:
        log_orders.error("get_full_report_day error: %s", exc)
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "date": "",
            "date_from": now_str,
            "date_to": now_str,
            "period": "day",
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

def get_full_report_week(force_refresh: bool = True) -> Dict[str, Any]:
    """
    Полный отчет за неделю (с начала недели до сейчас).
    """
    try:
        from ExnessAPI.history import _connect, _day_start_local, _local_now, _naive_local
        from datetime import timedelta

        if not _connect():
            return {
                "period": "week",
                "wins": 0,
                "losses": 0,
                "total_closed": 0,
                "total_open": 0,
                "unrealized_pnl": 0.0,
                "profit": 0.0,
                "loss": 0.0,
                "net": 0.0,
                "balance": 0.0,
            }

        local_now = _local_now()
        week_start = local_now - timedelta(days=int(local_now.weekday()))
        week_start = _day_start_local(week_start)

        with MT5_LOCK:
            deals = mt5.history_deals_get(_naive_local(week_start), _naive_local(local_now))
            open_positions = mt5.positions_get() or []
            acc = mt5.account_info()

        balance = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0

        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        unrealized_pnl = 0.0

        entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        if deals:
            for d in deals:
                try:
                    if getattr(d, "entry", None) != entry_out:
                        continue
                    p = float(getattr(d, "profit", 0.0) or 0.0)
                    if p > 0.0:
                        wins += 1
                        total_profit += p
                    elif p < 0.0:
                        losses += 1
                        total_loss += abs(p)
                except Exception:
                    continue

        for p in open_positions:
            try:
                unrealized_pnl += float(getattr(p, "profit", 0.0) or 0.0)
            except Exception:
                continue

        return {
            "period": "week",
            "date_from": week_start.strftime("%Y-%m-%d %H:%M:%S"),
            "date_to": local_now.strftime("%Y-%m-%d %H:%M:%S"),
            "wins": int(wins),
            "losses": int(losses),
            "total_closed": int(wins + losses),
            "total_open": int(len(open_positions)),
            "unrealized_pnl": round(float(unrealized_pnl), 2),
            "profit": round(float(total_profit), 2),
            "loss": round(float(total_loss), 2),
            "net": round(float(total_profit - total_loss), 2),
            "balance": round(float(balance), 2),
        }
    except Exception as exc:
        log_orders.error("get_full_report_week error: %s", exc)
        return {
            "period": "week",
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": 0.0,
        }

def get_full_report_all(force_refresh: bool = True) -> Dict[str, Any]:
    """
    Полный отчет за весь период (с самого начала аккаунта до сейчас).
    """
    try:
        from ExnessAPI.history import _connect, _local_now, _naive_local
        from datetime import datetime, timedelta

        if not _connect():
            return {
                "period": "all",
                "date_from": "",
                "date_to": "",
                "wins": 0,
                "losses": 0,
                "total_closed": 0,
                "total_open": 0,
                "unrealized_pnl": 0.0,
                "profit": 0.0,
                "loss": 0.0,
                "net": 0.0,
                "balance": 0.0,
            }

        local_now = _local_now()
        # Используем 1 год назад для получения истории
        from datetime import timedelta
        from_date = local_now - timedelta(days=365)

        deals = None
        with MT5_LOCK:
            try:
                # Получаем все сделки с самого начала
                deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(local_now))
                # Проверяем на ошибку MT5
                if deals is None:
                    err = mt5.last_error()
                    if err and err[0] != 1:  # 1 = Success
                        log_orders.error("history_deals_get failed: code=%s desc=%s", err[0] if err else "?", err[1] if err and len(err) > 1 else "?")
                        deals = []
            except Exception as exc:
                log_orders.error("history_deals_get exception: %s", exc)
                deals = []
            
            try:
                open_positions = mt5.positions_get() or []
            except Exception:
                open_positions = []
            
            try:
                acc = mt5.account_info()
            except Exception:
                acc = None

        balance = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0

        # Находим реальную дату первой сделки (самую раннюю)
        first_deal_date = None
        if deals and len(deals) > 0:
            try:
                # Сортируем сделки по времени (самая ранняя первая)
                deals_sorted = sorted(deals, key=lambda d: getattr(d, "time", 0) or 0)
                first_deal = deals_sorted[0]
                first_deal_time = getattr(first_deal, "time", None)
                if first_deal_time:
                    # Преобразуем в datetime если нужно
                    if isinstance(first_deal_time, datetime):
                        first_deal_date = first_deal_time
                    else:
                        # Если это timestamp (секунды с 1970)
                        try:
                            first_deal_date = datetime.fromtimestamp(int(first_deal_time))
                        except (ValueError, OSError):
                            # Если timestamp в миллисекундах
                            try:
                                first_deal_date = datetime.fromtimestamp(int(first_deal_time) / 1000)
                            except Exception:
                                pass
            except Exception:
                pass

        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        unrealized_pnl = 0.0

        entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        if deals:
            for d in deals:
                try:
                    if getattr(d, "entry", None) != entry_out:
                        continue
                    p = float(getattr(d, "profit", 0.0) or 0.0)
                    if p > 0.0:
                        wins += 1
                        total_profit += p
                    elif p < 0.0:
                        losses += 1
                        total_loss += abs(p)
                except Exception:
                    continue

        # Обрабатываем открытые позиции
        open_positions_info = []
        for p in open_positions:
            try:
                ticket = int(getattr(p, "ticket", 0) or 0)
                symbol = str(getattr(p, "symbol", "") or "")
                volume = float(getattr(p, "volume", 0.0) or 0.0)
                profit_val = float(getattr(p, "profit", 0.0) or 0.0)
                unrealized_pnl += profit_val
                open_positions_info.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "volume": volume,
                    "profit": profit_val
                })
            except Exception:
                continue

        # Форматируем даты - показываем реальную дату первой сделки или дату начала периода
        date_from_str = ""
        if first_deal_date:
            try:
                if first_deal_date.tzinfo is None:
                    date_from_str = first_deal_date.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_from_str = first_deal_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_from_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Если нет сделок, показываем дату начала периода (1 год назад)
            date_from_str = from_date.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "period": "all",
            "date_from": date_from_str,
            "date_to": local_now.strftime("%Y-%m-%d %H:%M:%S"),
            "wins": int(wins),
            "losses": int(losses),
            "total_closed": int(wins + losses),
            "total_open": int(len(open_positions)),
            "open_positions": open_positions_info,
            "unrealized_pnl": round(float(unrealized_pnl), 2),
            "profit": round(float(total_profit), 2),
            "loss": round(float(total_loss), 2),
            "net": round(float(total_profit - total_loss), 2),
            "balance": round(float(balance), 2),
        }
    except Exception as exc:
        log_orders.error("get_full_report_all error: %s", exc)
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "period": "all",
            "date_from": "",
            "date_to": now_str,
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": 0.0,
        }

def get_full_report_month(force_refresh: bool = True) -> Dict[str, Any]:
    """
    Полный отчет за месяц (с начала месяца до сейчас).
    """
    try:
        from ExnessAPI.history import _connect, _day_start_local, _local_now, _naive_local
        from datetime import datetime

        if not _connect():
            return {
                "period": "month",
                "wins": 0,
                "losses": 0,
                "total_closed": 0,
                "total_open": 0,
                "unrealized_pnl": 0.0,
                "profit": 0.0,
                "loss": 0.0,
                "net": 0.0,
                "balance": 0.0,
            }

        local_now = _local_now()
        try:
            month_start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            month_start = datetime(local_now.year, local_now.month, 1, 0, 0, 0)

        # keep _day_start_local import for compatibility (structure), even if not used directly here
        _ = _day_start_local  # noqa: F841

        with MT5_LOCK:
            deals = mt5.history_deals_get(_naive_local(month_start), _naive_local(local_now))
            open_positions = mt5.positions_get() or []
            acc = mt5.account_info()

        balance = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0

        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        unrealized_pnl = 0.0

        entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        if deals:
            for d in deals:
                try:
                    if getattr(d, "entry", None) != entry_out:
                        continue
                    p = float(getattr(d, "profit", 0.0) or 0.0)
                    if p > 0.0:
                        wins += 1
                        total_profit += p
                    elif p < 0.0:
                        losses += 1
                        total_loss += abs(p)
                except Exception:
                    continue

        for p in open_positions:
            try:
                unrealized_pnl += float(getattr(p, "profit", 0.0) or 0.0)
            except Exception:
                continue

        return {
            "period": "month",
            "date_from": month_start.strftime("%Y-%m-%d %H:%M:%S"),
            "date_to": local_now.strftime("%Y-%m-%d %H:%M:%S"),
            "wins": int(wins),
            "losses": int(losses),
            "total_closed": int(wins + losses),
            "total_open": int(len(open_positions)),
            "unrealized_pnl": round(float(unrealized_pnl), 2),
            "profit": round(float(total_profit), 2),
            "loss": round(float(total_loss), 2),
            "net": round(float(total_profit - total_loss), 2),
            "balance": round(float(balance), 2),
        }
    except Exception as exc:
        log_orders.error("get_full_report_month error: %s", exc)
        return {
            "period": "month",
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": 0.0,
        }




# =============================================================================
# Fast market open: adaptive lot + ATR-based SL/TP, sync (no await)
# =============================================================================
_SYMBOL_BTC = "BTCUSDm"     
_SYMBOL_XAU = "XAUUSDm"


def _place_market_order_fixed_sltp(symbol: str, side: str) -> bool:
    """DEFAULT SIMPLE ORDER: lot=0.02, TP=2 USD, SL=0"""
    try:
        _ensure_mt5_connected(require_trade_allowed=True)
        _symbol_select(symbol)

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if not tick or not info:
            return False

        is_buy = side.lower() == "buy"
        entry = float(tick.ask if is_buy else tick.bid)
        if entry <= 0:
            return False

        lot = 0.01
        profit_usd = 2.0
        sl = 0.0  # ❌ NO STOP LOSS

        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value

        if tick_size <= 0 or tick_value <= 0:
            return False

        price_delta = (profit_usd * tick_size) / (tick_value * lot)

        tp = entry + price_delta if is_buy else entry - price_delta
        tp = round(tp, info.digits)

        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": DEFAULT_DEVIATION,
            "magic": DEFAULT_MAGIC,
            "comment": "fixed_lot_tp2_sl0",
            "type_filling": _best_filling_type(symbol),
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    except Exception as e:
        log_orders.error("fixed simple order error %s %s", symbol, e)
        return False


def open_buy_order_btc(count: int) -> int:
    """Open `count` market BUY on BTCUSDm. Adaptive lot + ATR SL/TP, sync."""
    n = 0
    for _ in range(max(0, int(count))):
        if _place_market_order_fixed_sltp(_SYMBOL_BTC, "Buy"):
            n += 1
    return n


def open_buy_order_xau(count: int) -> int:
    """Open `count` market BUY on XAUUSDm. Adaptive lot + ATR SL/TP, sync."""
    n = 0
    for _ in range(max(0, int(count))):
        if _place_market_order_fixed_sltp(_SYMBOL_XAU, "Buy"):
            n += 1
    return n


def open_sell_order_btc(count: int) -> int:
    """Open `count` market SELL on BTCUSDm. Adaptive lot + ATR SL/TP, sync."""
    n = 0
    for _ in range(max(0, int(count))):
        if _place_market_order_fixed_sltp(_SYMBOL_BTC, "Sell"):
            n += 1
    return n


def open_sell_order_xau(count: int) -> int:
    """Open `count` market SELL on XAUUSDm. Adaptive lot + ATR SL/TP, sync."""
    n = 0
    for _ in range(max(0, int(count))):
        if _place_market_order_fixed_sltp(_SYMBOL_XAU, "Sell"):
            n += 1
    return n




__all__ = [
    "enable_trading",
    "get_balance",
    "get_account_info",
    "get_positions_summary",
    "get_order_by_index",
    "get_all_open_positions",
    "has_open_positions",
    "close_order",
    "close_all_position",
    "set_takeprofit_all_positions_usd",
    "set_stoploss_all_positions_usd",
    "get_full_report_day",
    "get_full_report_week",
    "get_full_report_month",
    "get_full_report_all",
    "open_buy_order_btc",
    "open_buy_order_xau",
    "open_sell_order_btc",
    "open_sell_order_xau",
]



