# ExnessAPI/order_execution.py  (PRODUCTION-GRADE / ULTRA-FAST / SLTP-ROBUST)
from __future__ import annotations

import logging
import math
import os
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional

import MetaTrader5 as mt5

from mt5_client import MT5_LOCK, ensure_mt5
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# =============================================================================
# Logging
#   For maximum speed, you can disable health logs:
#     set ORDER_EXEC_HEALTH_LOG=0
# =============================================================================

LOG_DIR = LOG_ROOT


def _rotating_file_logger(
    name: str,
    filename: str,
    level: int,
    default_max_bytes: int = 5_242_880,  # 5MB
    default_backups: int = 5,
) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        fh = RotatingFileHandler(
            filename=str(get_log_path(filename)),
            maxBytes=default_max_bytes,
            backupCount=default_backups,
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
        log.addHandler(fh)

    return log


log_err = _rotating_file_logger("order_execution", "order_execution.log", logging.ERROR)

# Health logger disabled (only errors to order_execution.log)
_HEALTH_ENABLED = False
log_health = logging.getLogger("order_execution.health")
log_health.setLevel(logging.CRITICAL)
log_health.propagate = False

# =============================================================================
# Models
# =============================================================================


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    signal: str  # "Buy" or "Sell"
    lot: float
    sl: float
    tp: float
    price: float  # snapshot price (for logs / fallback)
    order_id: str
    signal_id: str
    enqueue_time: float
    deviation: int = 50
    magic: int = 987654
    comment: str = "scalp"


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    signal_id: str
    ok: bool
    reason: str
    sent_ts: float
    fill_ts: float
    req_price: float
    exec_price: float
    volume: float
    slippage: float
    retcode: int = 0


@dataclass
class _SymbolMeta:
    symbol: str
    digits: int
    point: float
    filling: int
    v_step: float
    v_min: float
    v_max: float
    v_decimals: int
    ts: float


# =============================================================================
# Executor
# =============================================================================


class OrderExecutor:
    """Ultra-fast MT5 market order executor.

    Key properties:
    - Zero sleep retry loop for transient retcodes
    - Symbol metadata cache
    - Robust SL/TP attachment (position ticket resolve: result.position/order/deal + bounded polling)
    """

    def __init__(
        self,
        *,
        normalize_volume_fn: Optional[Callable[[float, Any], float]] = None,
        sanitize_stops_fn: Optional[Callable[[str, float, float, float], tuple[float, float]]] = None,
        normalize_price_fn: Optional[Callable[[float], float]] = None,
        close_on_sltp_fail: Optional[bool] = None,
        auto_ensure_mt5: bool = False,
        symbol_cache_ttl: float = 60.0,
    ) -> None:
        self.normalize_volume_fn = normalize_volume_fn
        self.sanitize_stops_fn = sanitize_stops_fn
        self.normalize_price_fn = normalize_price_fn
        self.auto_ensure_mt5 = bool(auto_ensure_mt5)
        self.symbol_cache_ttl = float(symbol_cache_ttl)

        env_default = True
        self.close_on_sltp_fail = bool(env_default if close_on_sltp_fail is None else close_on_sltp_fail)

        self._meta: dict[str, _SymbolMeta] = {}

    # -------------------- Public API --------------------

    def send_market_order(
        self,
        request: OrderRequest,
        *,
        max_attempts: int = 2,
        telemetry_hooks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> OrderResult:
        hooks = telemetry_hooks or {}

        side = str(request.signal)
        if side not in ("Buy", "Sell"):
            return self._fail(request, time.time(), "bad_side", 0)

        if self.auto_ensure_mt5:
            try:
                ensure_mt5()
            except Exception:
                return self._fail(request, time.time(), "mt5_not_ready", 0)

        planned_sl = float(request.sl) if float(request.sl) > 0 else 0.0
        planned_tp = float(request.tp) if float(request.tp) > 0 else 0.0

        # MT5 retcodes
        RC_DONE = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        RC_DONE_PARTIAL = int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        RC_REQUOTE = int(getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004))
        RC_TIMEOUT = int(getattr(mt5, "TRADE_RETCODE_TIMEOUT", 10012))
        RC_PRICE_CHANGED = int(getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020))
        RC_OFF_QUOTES = int(getattr(mt5, "TRADE_RETCODE_OFF_QUOTES", 10021))
        RC_INVALID_STOPS = int(getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016))
        RC_INVALID_PRICE = int(getattr(mt5, "TRADE_RETCODE_INVALID_PRICE", 10015))

        last_reason = "unknown"
        last_retcode = 0
        sent_ts = time.time()

        meta = self._get_symbol_meta(request.symbol)
        if meta is None:
            return self._fail(request, sent_ts, "symbol_meta_none", 0)

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL

        # Compute volume outside lock
        vol = self._normalize_volume(float(request.lot), meta)

        # INSTANT RETRY LOOP (Zero Sleep)
        for attempt in range(1, int(max_attempts) + 1):
            try:
                with MT5_LOCK:
                    tick = mt5.symbol_info_tick(request.symbol)
                    le_tick = mt5.last_error()
                if tick is None:
                    last_reason = f"tick_none_{le_tick}"
                    continue

                req_price = float(tick.ask) if side == "Buy" else float(tick.bid)
                if req_price <= 0.0:
                    last_reason = "bad_req_price"
                    continue

                if self.normalize_price_fn:
                    try:
                        req_price = float(self.normalize_price_fn(req_price))
                    except Exception:
                        pass

                sl_s, tp_s = self._sanitize_stops(side, req_price, planned_sl, planned_tp)

                send_ts = time.time()
                sent_ts = send_ts

                if _HEALTH_ENABLED:
                    log_health.info(
                        "ORDER_SEND | attempt=%s side=%s symbol=%s lot=%.4f req=%.5f sl=%.5f tp=%.5f dev=%s",
                        attempt,
                        side,
                        request.symbol,
                        vol,
                        req_price,
                        sl_s,
                        tp_s,
                        int(request.deviation),
                    )

                result = self._send_deal(
                    symbol=request.symbol,
                    order_type=int(order_type),
                    price=float(req_price),
                    volume=float(vol),
                    sl=float(sl_s),
                    tp=float(tp_s),
                    filling=int(meta.filling),
                    request=request,
                )

                if result is None:
                    last_reason = "order_send_none"
                    continue

                last_retcode = int(getattr(result, "retcode", 0) or 0)

                # If broker rejects stops/price: immediate retry without stops
                used_stops = True
                if last_retcode in (RC_INVALID_STOPS, RC_INVALID_PRICE) and (planned_sl > 0.0 or planned_tp > 0.0):
                    used_stops = False
                    if _HEALTH_ENABLED:
                        log_health.info("ORDER_RETRY_NO_STOPS | retcode=%s", last_retcode)

                    result2 = self._send_deal(
                        symbol=request.symbol,
                        order_type=int(order_type),
                        price=float(req_price),
                        volume=float(vol),
                        sl=0.0,
                        tp=0.0,
                        filling=int(meta.filling),
                        request=request,
                    )
                    if result2 is None:
                        last_reason = "order_send_none_2"
                        continue
                    result = result2
                    last_retcode = int(getattr(result2, "retcode", 0) or 0)

                if last_retcode in (RC_DONE, RC_DONE_PARTIAL):
                    fill_ts = time.time()
                    exec_price = float(getattr(result, "price", req_price) or req_price)
                    exec_vol = float(getattr(result, "volume", vol) or vol)
                    slippage = abs(exec_price - req_price)

                    # Candidate tickets from MT5 result (may be 0 depending on broker/terminal)
                    pos_ticket = int(getattr(result, "position", 0) or 0)
                    ord_ticket = int(getattr(result, "order", 0) or 0)
                    deal_ticket = int(getattr(result, "deal", 0) or 0)

                    # Ensure SL/TP (robust resolve, bounded polling, no sleep)
                    if planned_sl > 0.0 or planned_tp > 0.0:
                        attached = True
                        if (not used_stops) or (not self._position_has_any_sltp(request.symbol, side, int(request.magic))):
                            attached = self._ensure_sltp_fast(
                                symbol=request.symbol,
                                side=side,
                                exec_price=exec_price,
                                volume=exec_vol,
                                planned_sl=planned_sl,
                                planned_tp=planned_tp,
                                magic=int(request.magic),
                                ticket=pos_ticket if pos_ticket > 0 else None,
                                candidate_tickets=(pos_ticket, ord_ticket, deal_ticket),
                                sent_ts=float(sent_ts),
                                rc_done=RC_DONE,
                                rc_done_partial=RC_DONE_PARTIAL,
                            )
                        if (not attached) and self.close_on_sltp_fail:
                            self._fail_safe_close(request.symbol, side, exec_vol, int(request.magic))

                    # Minimal positions count (optional)
                    positions_open = 1
                    try:
                        with MT5_LOCK:
                            positions_open = len(mt5.positions_get(symbol=request.symbol) or [])
                    except Exception:
                        positions_open = 1

                    self._safe_hook(
                        hooks,
                        "record_trade",
                        exec_price,
                        float(request.sl),
                        exec_vol,
                        float(request.tp),
                        int(positions_open),
                    )
                    self._safe_hook(
                        hooks,
                        "record_execution_metrics",
                        request.order_id,
                        side,
                        float(request.enqueue_time),
                        float(sent_ts),
                        float(fill_ts),
                        float(req_price),
                        float(exec_price),
                        float(slippage),
                    )

                    if _HEALTH_ENABLED:
                        log_health.info(
                            "ORDER_FILLED | order_id=%s side=%s symbol=%s ret=%s req=%.5f exec=%.5f vol=%.4f slip=%.5f pos=%s ord=%s deal=%s",
                            request.order_id,
                            side,
                            request.symbol,
                            last_retcode,
                            req_price,
                            exec_price,
                            exec_vol,
                            slippage,
                            pos_ticket,
                            ord_ticket,
                            deal_ticket,
                        )

                    return OrderResult(
                        order_id=request.order_id,
                        signal_id=request.signal_id,
                        ok=True,
                        reason="filled",
                        sent_ts=float(sent_ts),
                        fill_ts=float(fill_ts),
                        req_price=float(req_price),
                        exec_price=float(exec_price),
                        volume=float(exec_vol),
                        slippage=float(slippage),
                        retcode=int(last_retcode),
                    )

                if last_retcode in (RC_REQUOTE, RC_TIMEOUT, RC_PRICE_CHANGED, RC_OFF_QUOTES):
                    last_reason = f"retry_retcode_{last_retcode}"
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                log_err.error(
                    "Order process error | attempt=%s err=%s tb=%s",
                    attempt,
                    exc,
                    traceback.format_exc(),
                )
                continue

        self._safe_hook(hooks, "record_execution_failure", request.order_id, float(request.enqueue_time), float(sent_ts), last_reason)
        log_err.error(
            "ORDER_FAILED | order_id=%s side=%s symbol=%s reason=%s retcode=%s",
            request.order_id,
            side,
            request.symbol,
            last_reason,
            last_retcode,
        )
        return self._fail(request, sent_ts, last_reason, last_retcode)

    # -------------------- Helpers --------------------

    @staticmethod
    def _safe_hook(hooks: dict[str, Callable[..., Any]], name: str, *args: Any) -> None:
        fn = hooks.get(name)
        if not fn:
            return
        try:
            fn(*args)
        except Exception:
            return

    def _fail(self, request: OrderRequest, sent_ts: float, reason: str, retcode: int) -> OrderResult:
        return OrderResult(
            order_id=request.order_id,
            signal_id=request.signal_id,
            ok=False,
            reason=str(reason),
            sent_ts=float(sent_ts),
            fill_ts=float(sent_ts),
            req_price=float(request.price),
            exec_price=0.0,
            volume=float(request.lot),
            slippage=0.0,
            retcode=int(retcode),
        )

    def _get_symbol_meta(self, symbol: str) -> Optional[_SymbolMeta]:
        now = time.time()
        m = self._meta.get(symbol)
        if m and (now - float(m.ts) <= float(self.symbol_cache_ttl)):
            return m

        try:
            with MT5_LOCK:
                info = mt5.symbol_info(symbol)
                if info is None:
                    return None
                # Ensure selected
                try:
                    if hasattr(info, "visible") and not bool(getattr(info, "visible", True)):
                        mt5.symbol_select(symbol, True)
                except Exception:
                    pass

            digits = int(getattr(info, "digits", 0) or 0)
            point = float(getattr(info, "point", 0.0) or 0.0) or (10 ** (-digits) if digits > 0 else 0.0)

            filling = getattr(info, "filling_mode", None)
            if filling not in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
                filling = mt5.ORDER_FILLING_IOC

            v_step = float(getattr(info, "volume_step", 0.01) or 0.01)
            v_min = float(getattr(info, "volume_min", v_step) or v_step)
            v_max = float(getattr(info, "volume_max", v_min) or v_min)

            if v_step <= 0:
                v_step = 0.01

            v_decimals = 0
            if v_step < 1.0:
                try:
                    v_decimals = max(0, min(8, int(round(-math.log10(v_step)))))
                except Exception:
                    v_decimals = 2

            m2 = _SymbolMeta(
                symbol=str(symbol),
                digits=digits,
                point=point,
                filling=int(filling),
                v_step=float(v_step),
                v_min=float(v_min),
                v_max=float(v_max),
                v_decimals=int(v_decimals),
                ts=float(now),
            )
            self._meta[symbol] = m2
            return m2

        except Exception as exc:
            log_err.error("symbol_meta_failed | symbol=%s err=%s tb=%s", symbol, exc, traceback.format_exc())
            return None

    def _normalize_volume(self, vol: float, meta: _SymbolMeta) -> float:
        if self.normalize_volume_fn:
            try:
                return float(self.normalize_volume_fn(vol, meta))
            except Exception:
                pass

        vol = float(vol)
        vmin, vmax, step = float(meta.v_min), float(meta.v_max), float(meta.v_step)
        vol = max(vmin, min(vmax, vol))
        k = math.floor(vol / step)
        v = round(k * step, int(meta.v_decimals))
        return float(v if v >= vmin else vmin)

    def _sanitize_stops(self, side: str, req_price: float, sl: float, tp: float) -> tuple[float, float]:
        if self.sanitize_stops_fn:
            try:
                sl2, tp2 = self.sanitize_stops_fn(side, req_price, sl, tp)
                return float(sl2 or 0.0), float(tp2 or 0.0)
            except Exception:
                return float(sl or 0.0), float(tp or 0.0)
        return float(sl or 0.0), float(tp or 0.0)

    @staticmethod
    def _send_deal(
        *,
        symbol: str,
        order_type: int,
        price: float,
        volume: float,
        sl: float,
        tp: float,
        filling: int,
        request: OrderRequest,
    ) -> Any:
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": str(symbol),
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "sl": float(sl) if float(sl) > 0 else 0.0,
            "tp": float(tp) if float(tp) > 0 else 0.0,
            "deviation": int(request.deviation),
            "magic": int(request.magic),
            "comment": str(request.comment),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": int(filling),
        }
        with MT5_LOCK:
            r = mt5.order_send(req)
            _ = mt5.last_error()
        return r

    def _position_has_any_sltp(self, symbol: str, side: str, magic: int) -> bool:
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                if int(getattr(p, "magic", 0) or 0) != int(magic):
                    continue
                if int(getattr(p, "type", -1) or -1) != int(want_type):
                    continue
                cur_sl = float(getattr(p, "sl", 0.0) or 0.0)
                cur_tp = float(getattr(p, "tp", 0.0) or 0.0)
                if cur_sl > 0.0 or cur_tp > 0.0:
                    return True
            return False
        except Exception:
            return False

    # -------------------- SLTP robust attach --------------------

    def _ensure_sltp_fast(
        self,
        *,
        symbol: str,
        side: str,
        exec_price: float,
        volume: float,
        planned_sl: float,
        planned_tp: float,
        magic: int,
        ticket: Optional[int] = None,
        candidate_tickets: tuple[int, ...] = (),
        sent_ts: float = 0.0,
        rc_done: int,
        rc_done_partial: int,
    ) -> bool:
        if planned_sl <= 0.0 and planned_tp <= 0.0:
            return True

        sl2, tp2 = self._sanitize_stops(side, exec_price, planned_sl, planned_tp)
        if sl2 <= 0.0 and tp2 <= 0.0:
            return False

        resolved = self._resolve_position_ticket(
            symbol=symbol,
            side=side,
            magic=magic,
            exec_price=exec_price,
            volume=volume,
            ticket=ticket,
            candidate_tickets=candidate_tickets,
            sent_ts=sent_ts,
            poll_ms=280,  # bounded window for terminal update (no sleep)
        )

        if resolved == 0:
            # No open position (netting/offset/instant close) => SLTP not applicable
            if _HEALTH_ENABLED:
                log_health.info("SLTP_SKIP_NO_POSITION | symbol=%s side=%s", symbol, side)
            return True

        if resolved is None:
            with MT5_LOCK:
                le = mt5.last_error()
            log_err.error(
                "SLTP: position_not_found | symbol=%s side=%s vol=%.4f candidates=%s last_error=%s",
                symbol,
                side,
                float(volume),
                candidate_tickets,
                le,
            )
            return False

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(resolved),
            "symbol": str(symbol),
            "sl": float(sl2) if float(sl2) > 0 else 0.0,
            "tp": float(tp2) if float(tp2) > 0 else 0.0,
            "magic": int(magic),
            "comment": "sltp_fast",
        }

        for i in range(1, 4):
            with MT5_LOCK:
                r = mt5.order_send(req)
                le = mt5.last_error()
            rc = int(getattr(r, "retcode", 0) or 0) if r is not None else 0

            if _HEALTH_ENABLED:
                log_health.info(
                    "SLTP_SET | try=%s ticket=%s rc=%s sl=%.5f tp=%.5f last_error=%s",
                    i,
                    int(resolved),
                    rc,
                    float(sl2),
                    float(tp2),
                    le,
                )

            if r is not None and rc in (rc_done, rc_done_partial):
                return True

        log_err.error("SLTP_SET_FAILED | ticket=%s symbol=%s", int(resolved), symbol)
        return False

    def _resolve_position_ticket(
        self,
        *,
        symbol: str,
        side: str,
        magic: int,
        exec_price: float,
        volume: float,
        ticket: Optional[int],
        candidate_tickets: tuple[int, ...],
        sent_ts: float,
        poll_ms: int = 250,
    ) -> Optional[int]:
        want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL

        def _pos_exists_by_ticket(tk: int) -> bool:
            if tk <= 0:
                return False
            try:
                with MT5_LOCK:
                    try:
                        p = mt5.positions_get(ticket=int(tk)) or []
                    except TypeError:
                        p = None
                    if p:
                        px = p[0]
                        return str(getattr(px, "symbol", "")) == str(symbol)
            except Exception:
                return False
            return False

        # A) direct candidates first
        direct = [int(ticket or 0)] + [int(x or 0) for x in candidate_tickets]
        for tk in direct:
            if tk > 0 and _pos_exists_by_ticket(tk):
                return tk

        # B) bounded polling by elapsed time (no sleep)
        deadline = time.perf_counter() + (float(poll_ms) / 1000.0)
        best_ticket: int = 0
        best_score = None
        saw_any = False

        while time.perf_counter() < deadline:
            try:
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []
            except Exception:
                positions = []

            if positions:
                saw_any = True
            else:
                continue

            for p in positions:
                try:
                    if int(getattr(p, "type", -1) or -1) != int(want_type):
                        continue

                    tk = int(getattr(p, "ticket", 0) or 0)
                    if tk <= 0:
                        continue

                    pmagic = int(getattr(p, "magic", 0) or 0)
                    pv = float(getattr(p, "volume", 0.0) or 0.0)
                    popen = float(getattr(p, "price_open", 0.0) or 0.0)

                    tmsc = getattr(p, "time_msc", None)
                    pt = float(tmsc) if tmsc is not None else float(getattr(p, "time", 0.0) or 0.0) * 1000.0

                    magic_penalty = 0 if pmagic == int(magic) else 1  # prefer matching magic
                    dv = abs(pv - float(volume))
                    dp = abs(popen - float(exec_price)) if popen > 0 else 0.0

                    # Prefer: magic match, newest, closest volume, closest open price
                    score = (magic_penalty, -pt, dv, dp)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_ticket = tk
                except Exception:
                    continue

            if best_ticket > 0:
                return int(best_ticket)

        # If we saw positions but couldn't select a ticket => inconsistent state
        if saw_any:
            return None

        # Never saw any open position => netting/offset/instant close
        return 0

    # -------------------- legacy fast finder (kept) --------------------

    @staticmethod
    def _find_position_ticket_fast(*, symbol: str, side: str, volume: float, magic: int) -> Optional[int]:
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL

            for _ in range(12):
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []

                best_tk = 0
                best_key = None

                for p in positions:
                    try:
                        if int(getattr(p, "magic", 0) or 0) != int(magic):
                            continue
                        if int(getattr(p, "type", -1) or -1) != int(want_type):
                            continue
                        pv = float(getattr(p, "volume", 0.0) or 0.0)
                        pt = float(getattr(p, "time", 0.0) or 0.0)
                        tk = int(getattr(p, "ticket", 0) or 0)
                        if tk <= 0 or pv <= 0.0:
                            continue

                        dv = abs(pv - float(volume))
                        key = (-pt, dv)  # newest, then closest volume
                        if best_key is None or key < best_key:
                            best_key = key
                            best_tk = tk
                    except Exception:
                        continue

                if best_tk > 0:
                    return int(best_tk)

            return None

        except Exception:
            return None

    @staticmethod
    def _fail_safe_close(symbol: str, side: str, volume: float, magic: int) -> None:
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol) or []

            for p in positions:
                try:
                    if int(getattr(p, "magic", 0) or 0) != int(magic):
                        continue
                    if int(getattr(p, "type", -1) or -1) != int(want_type):
                        continue
                    ticket = int(getattr(p, "ticket", 0) or 0)
                    pv = float(getattr(p, "volume", 0.0) or 0.0)
                    if ticket <= 0 or pv <= 0.0:
                        continue

                    with MT5_LOCK:
                        tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue

                    price = float(tick.bid) if side == "Buy" else float(tick.ask)
                    close_type = mt5.ORDER_TYPE_SELL if side == "Buy" else mt5.ORDER_TYPE_BUY
                    req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": str(symbol),
                        "position": int(ticket),
                        "volume": float(pv),
                        "type": int(close_type),
                        "price": float(price),
                        "deviation": 50,
                        "magic": int(magic),
                        "comment": "failsafe_close_sltp",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    with MT5_LOCK:
                        r = mt5.order_send(req)
                        le = mt5.last_error()
                    log_err.error(
                        "FAILSAFE_CLOSE | ticket=%s rc=%s last_error=%s",
                        ticket,
                        int(getattr(r, "retcode", 0) or 0),
                        le,
                    )
                except Exception:
                    continue
        except Exception:
            return


__all__ = ["OrderRequest", "OrderResult", "OrderExecutor"]
