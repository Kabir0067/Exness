from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional

import MetaTrader5 as mt5

from mt5_client import MT5_LOCK, ensure_mt5
from log_config import get_log_path

# =============================================================================
# Logging (ERROR-only)
# =============================================================================

def _rotating_file_logger(
    name: str,
    filename: str,
    level: int,
    default_max_bytes: int = 5_242_880,  
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

# Health logs OFF by default for speed
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
    stops_level: int
    freeze_level: int
    ts: float


# =============================================================================
# Executor
# =============================================================================

class OrderExecutor:
    """
    Ultra-fast MT5 market order executor with SLTP-ROBUST attachment.

    Main guarantees:
    - No sleep under MT5_LOCK
    - SL/TP will be re-attached using CURRENT tick constraints (stops_level + freeze_level)
    - Position ticket resolve works in netting/hedging (bounded polling with micro-sleep)
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

        # Default: close position if SLTP cannot be attached (fail-safe)
        self.close_on_sltp_fail = bool(True if close_on_sltp_fail is None else close_on_sltp_fail)

        self._meta: dict[str, _SymbolMeta] = {}

        # SLTP resolve/attach tuning (fast + reliable)
        self._sltp_poll_deadline_ms = 650  # enough for terminal update under load
        self._sltp_poll_sleep_ms_seq = (5, 10, 20, 40, 80, 80, 80)  # total <= ~315ms typical
        self._sltp_send_retries = 4

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

        # MT5 retcodes (defaults if missing)
        RC_DONE = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        RC_DONE_PARTIAL = int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        RC_REQUOTE = int(getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004))
        RC_TIMEOUT = int(getattr(mt5, "TRADE_RETCODE_TIMEOUT", 10012))
        RC_PRICE_CHANGED = int(getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10020))
        RC_OFF_QUOTES = int(getattr(mt5, "TRADE_RETCODE_OFF_QUOTES", 10021))
        RC_INVALID_STOPS = int(getattr(mt5, "TRADE_RETCODE_INVALID_STOPS", 10016))
        RC_INVALID_PRICE = int(getattr(mt5, "TRADE_RETCODE_INVALID_PRICE", 10015))
        RC_TRADE_CONTEXT_BUSY = int(getattr(mt5, "TRADE_RETCODE_TRADE_CONTEXT_BUSY", 10027))
        RC_NO_CHANGES = int(getattr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025))

        last_reason = "unknown"
        last_retcode = 0
        sent_ts = time.time()

        meta = self._get_symbol_meta(request.symbol)
        if meta is None:
            return self._fail(request, sent_ts, "symbol_meta_none", 0)

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL

        # Compute volume outside lock
        vol = self._normalize_volume(float(request.lot), meta)

        # INSTANT RETRY LOOP (no big sleeps)
        for attempt in range(1, int(max_attempts) + 1):
            try:
                with MT5_LOCK:
                    tick = mt5.symbol_info_tick(request.symbol)

                if tick is None:
                    last_reason = "tick_none"
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

                # IMPORTANT:
                # When sending DEAL, try with planned stops first (fast path).
                # If broker rejects stops => send without stops, then robust attach.
                sl_s, tp_s = self._sanitize_stops(side, req_price, planned_sl, planned_tp)

                sent_ts = time.time()

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

                used_stops = True
                if last_retcode in (RC_INVALID_STOPS, RC_INVALID_PRICE) and (planned_sl > 0.0 or planned_tp > 0.0):
                    used_stops = False
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

                    pos_ticket = int(getattr(result, "position", 0) or 0)
                    ord_ticket = int(getattr(result, "order", 0) or 0)
                    deal_ticket = int(getattr(result, "deal", 0) or 0)

                    # ALWAYS verify SL/TP after fill if planned, because broker may ignore stops on market orders.
                    if planned_sl > 0.0 or planned_tp > 0.0:
                        ok_attach = self._ensure_sltp_robust(
                            symbol=request.symbol,
                            side=side,
                            planned_sl=planned_sl,
                            planned_tp=planned_tp,
                            magic=int(request.magic),
                            ticket_hint=pos_ticket if pos_ticket > 0 else None,
                            candidate_tickets=(pos_ticket, ord_ticket, deal_ticket),
                            sent_ts=float(sent_ts),
                            meta=meta,
                            rc_done=RC_DONE,
                            rc_done_partial=RC_DONE_PARTIAL,
                            rc_no_changes=RC_NO_CHANGES,
                            rc_invalid_stops=RC_INVALID_STOPS,
                            rc_trade_context_busy=RC_TRADE_CONTEXT_BUSY,
                        )
                        if (not ok_attach) and self.close_on_sltp_fail:
                            self._fail_safe_close(request.symbol, side, exec_vol, int(request.magic))

                    # Optional telemetry
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

                # transient retcodes -> retry quickly
                if last_retcode in (RC_REQUOTE, RC_TIMEOUT, RC_PRICE_CHANGED, RC_OFF_QUOTES, RC_TRADE_CONTEXT_BUSY):
                    last_reason = f"retry_retcode_{last_retcode}"
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                log_err.error("ORDER_EXCEPTION | attempt=%s err=%s tb=%s", attempt, exc, traceback.format_exc())
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
                try:
                    mt5.symbol_select(symbol, True)
                except Exception:
                    pass

            digits = int(getattr(info, "digits", 0) or 0)
            point = float(getattr(info, "point", 0.0) or 0.0)
            if point <= 0.0 and digits > 0:
                point = 10 ** (-digits)

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

            stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
            freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)

            m2 = _SymbolMeta(
                symbol=str(symbol),
                digits=digits,
                point=float(point),
                filling=int(filling),
                v_step=float(v_step),
                v_min=float(v_min),
                v_max=float(v_max),
                v_decimals=int(v_decimals),
                stops_level=int(stops_level),
                freeze_level=int(freeze_level),
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
        # External sanitizer (optional)
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
            return mt5.order_send(req)

    # =============================================================================
    # SLTP ROBUST
    # =============================================================================

    def _ensure_sltp_robust(
        self,
        *,
        symbol: str,
        side: str,
        planned_sl: float,
        planned_tp: float,
        magic: int,
        ticket_hint: Optional[int],
        candidate_tickets: tuple[int, ...],
        sent_ts: float,
        meta: _SymbolMeta,
        rc_done: int,
        rc_done_partial: int,
        rc_no_changes: int,
        rc_invalid_stops: int,
        rc_trade_context_busy: int,
    ) -> bool:
        if planned_sl <= 0.0 and planned_tp <= 0.0:
            return True

        # 1) Resolve position ticket reliably (netting/hedging)
        resolved = self._resolve_position_ticket_robust(
            symbol=symbol,
            side=side,
            magic=magic,
            ticket_hint=ticket_hint,
            candidate_tickets=candidate_tickets,
            sent_ts=sent_ts,
            deadline_ms=int(self._sltp_poll_deadline_ms),
        )

        # 0 => no position found (netting offset / instant close) => SLTP not applicable
        if resolved == 0:
            return True

        if resolved is None:
            log_err.error(
                "SLTP_RESOLVE_FAILED | symbol=%s side=%s magic=%s candidates=%s",
                symbol, side, int(magic), candidate_tickets
            )
            return False

        # 2) If already has SL/TP, verify (some brokers ignore on DEAL)
        cur_sl, cur_tp = self._get_position_sltp(resolved)
        if self._sltp_is_good(cur_sl, cur_tp, planned_sl, planned_tp, meta.point):
            return True

        # 3) Build SL/TP using CURRENT tick constraints (stops_level + freeze_level)
        sl2, tp2, ctx = self._sanitize_sltp_by_market_constraints(
            symbol=symbol,
            side=side,
            desired_sl=planned_sl,
            desired_tp=planned_tp,
            meta=meta,
        )
        if sl2 <= 0.0 and tp2 <= 0.0:
            log_err.error(
                "SLTP_SANITIZE_FAILED | ticket=%s symbol=%s side=%s desired_sl=%.5f desired_tp=%.5f ctx=%s",
                int(resolved), symbol, side, float(planned_sl), float(planned_tp), ctx
            )
            return False

        # 4) Send SLTP with retries (retcode-aware)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(resolved),
            "symbol": str(symbol),
            "sl": float(sl2) if float(sl2) > 0 else 0.0,
            "tp": float(tp2) if float(tp2) > 0 else 0.0,
            "magic": int(magic),
            "comment": "sltp_attach",
        }

        last_rc = 0
        last_le = ""
        for i in range(1, int(self._sltp_send_retries) + 1):
            with MT5_LOCK:
                r = mt5.order_send(req)
                last_le = str(mt5.last_error())
            last_rc = int(getattr(r, "retcode", 0) or 0) if r is not None else 0

            if r is not None and last_rc in (rc_done, rc_done_partial, rc_no_changes):
                return True

            # if invalid stops -> re-sanitize from fresh tick and retry
            if last_rc in (rc_invalid_stops, rc_trade_context_busy):
                sl2, tp2, ctx = self._sanitize_sltp_by_market_constraints(
                    symbol=symbol, side=side, desired_sl=planned_sl, desired_tp=planned_tp, meta=meta
                )
                req["sl"] = float(sl2) if sl2 > 0 else 0.0
                req["tp"] = float(tp2) if tp2 > 0 else 0.0

            # micro-backoff (NO lock)
            sleep_ms = 10 if i == 1 else 20 if i == 2 else 40
            time.sleep(sleep_ms / 1000.0)

        log_err.error(
            "SLTP_SET_FAILED | ticket=%s symbol=%s side=%s rc=%s last_error=%s sl=%.5f tp=%.5f stops=%s freeze=%s",
            int(resolved),
            symbol,
            side,
            int(last_rc),
            last_le,
            float(req["sl"] or 0.0),
            float(req["tp"] or 0.0),
            int(meta.stops_level),
            int(meta.freeze_level),
        )
        return False

    def _get_position_sltp(self, ticket: int) -> tuple[float, float]:
        try:
            with MT5_LOCK:
                try:
                    p = mt5.positions_get(ticket=int(ticket)) or []
                except TypeError:
                    p = []
            if not p:
                return 0.0, 0.0
            px = p[0]
            return float(getattr(px, "sl", 0.0) or 0.0), float(getattr(px, "tp", 0.0) or 0.0)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _sltp_is_good(cur_sl: float, cur_tp: float, planned_sl: float, planned_tp: float, point: float) -> bool:
        # Accept if current exists and close enough (1 point)
        eps = max(float(point), 1e-10)
        ok_sl = (planned_sl <= 0.0) or (cur_sl > 0.0 and abs(cur_sl - planned_sl) <= eps)
        ok_tp = (planned_tp <= 0.0) or (cur_tp > 0.0 and abs(cur_tp - planned_tp) <= eps)
        return bool(ok_sl and ok_tp)

    def _sanitize_sltp_by_market_constraints(
        self,
        *,
        symbol: str,
        side: str,
        desired_sl: float,
        desired_tp: float,
        meta: _SymbolMeta,
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Sanitizes SL/TP against:
        - current tick bid/ask
        - trade_stops_level and trade_freeze_level
        - correct direction (BUY/SELL)
        """
        ctx: dict[str, Any] = {"symbol": symbol}

        with MT5_LOCK:
            tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            ctx["reason"] = "tick_none"
            return 0.0, 0.0, ctx

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0.0 or ask <= 0.0:
            ctx["reason"] = "bad_tick"
            return 0.0, 0.0, ctx

        point = float(meta.point) if float(meta.point) > 0 else 0.00001
        digits = int(meta.digits) if int(meta.digits) > 0 else 5

        # Minimum distance = max(stops_level, freeze_level) * point + safety margin
        min_level = max(int(meta.stops_level), int(meta.freeze_level))
        min_dist = float(min_level) * point
        safety = 2.0 * point  # small safety
        dist = min_dist + safety

        sl = float(desired_sl) if float(desired_sl) > 0 else 0.0
        tp = float(desired_tp) if float(desired_tp) > 0 else 0.0

        if side == "Buy":
            # For BUY: SL must be <= bid - dist, TP must be >= bid + dist
            if sl > 0.0:
                sl = min(sl, bid - dist)
            if tp > 0.0:
                tp = max(tp, bid + dist)
            # direction sanity
            if sl > 0.0 and not (sl < bid):
                sl = 0.0
            if tp > 0.0 and not (tp > bid):
                tp = 0.0
        else:
            # SELL: SL >= ask + dist, TP <= ask - dist
            if sl > 0.0:
                sl = max(sl, ask + dist)
            if tp > 0.0:
                tp = min(tp, ask - dist)
            if sl > 0.0 and not (sl > ask):
                sl = 0.0
            if tp > 0.0 and not (tp < ask):
                tp = 0.0

        # rounding
        if sl > 0.0:
            sl = round(float(sl), digits)
        if tp > 0.0:
            tp = round(float(tp), digits)

        ctx.update(
            {
                "bid": bid,
                "ask": ask,
                "digits": digits,
                "point": point,
                "stops_level": int(meta.stops_level),
                "freeze_level": int(meta.freeze_level),
                "dist": dist,
            }
        )
        return float(sl), float(tp), ctx

    def _resolve_position_ticket_robust(
        self,
        *,
        symbol: str,
        side: str,
        magic: int,
        ticket_hint: Optional[int],
        candidate_tickets: tuple[int, ...],
        sent_ts: float,
        deadline_ms: int,
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
                        p = []
                if not p:
                    return False
                px = p[0]
                return str(getattr(px, "symbol", "")) == str(symbol)
            except Exception:
                return False

        # A) Direct candidates (fast)
        direct = [int(ticket_hint or 0)] + [int(x or 0) for x in candidate_tickets]
        for tk in direct:
            if tk > 0 and _pos_exists_by_ticket(tk):
                return tk

        # B) Poll positions_get(symbol) with micro-sleep (reliable + low CPU)
        deadline = time.perf_counter() + (float(deadline_ms) / 1000.0)
        best_ticket = 0
        best_score = None
        saw_any = False

        sleep_seq = list(self._sltp_poll_sleep_ms_seq)

        while time.perf_counter() < deadline:
            try:
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []
            except Exception:
                positions = []

            if positions:
                saw_any = True

            # Prefer:
            # 1) same type
            # 2) magic match
            # 3) newest time_msc
            for p in positions:
                try:
                    if int(getattr(p, "type", -1) or -1) != int(want_type):
                        continue

                    tk = int(getattr(p, "ticket", 0) or 0)
                    if tk <= 0:
                        continue

                    pmagic = int(getattr(p, "magic", 0) or 0)
                    tmsc = getattr(p, "time_msc", None)
                    pt = int(tmsc) if tmsc is not None else int(float(getattr(p, "time", 0.0) or 0.0) * 1000.0)

                    magic_penalty = 0 if pmagic == int(magic) else 1
                    score = (magic_penalty, -pt)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_ticket = tk
                except Exception:
                    continue

            if best_ticket > 0:
                return int(best_ticket)

            # no position yet -> wait a bit (low CPU)
            if sleep_seq:
                time.sleep(sleep_seq.pop(0) / 1000.0)
            else:
                time.sleep(0.08)

        # If never saw position -> netting offset / instant close
        if not saw_any:
            return 0

        # Saw positions but couldn't pick => inconsistent
        return None

    # =============================================================================
    # Fail-safe close if SLTP attach fails
    # =============================================================================

    @staticmethod
    def _fail_safe_close(symbol: str, side: str, volume: float, magic: int) -> None:
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=symbol) or []

            for p in positions:
                try:
                    if int(getattr(p, "type", -1) or -1) != int(want_type):
                        continue
                    if int(getattr(p, "magic", 0) or 0) != int(magic):
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
