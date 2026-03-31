from __future__ import annotations

import logging
import math
import random
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from typing import Any, Callable, Optional

import MetaTrader5 as mt5

from log_config import get_log_path
from mt5_client import MT5_LOCK, ensure_mt5

# =============================================================================
# Logging (ERROR-only, correct handler dedupe)
# =============================================================================


def _rotating_file_logger(
    name: str,
    filename: str,
    level: int,
    *,
    max_bytes: int = 5_242_880,
    backups: int = 5,
) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(int(level))
    lg.propagate = False

    target = str(get_log_path(filename))
    has_target = False
    for h in lg.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == target:
            has_target = True
            h.setLevel(int(level))
            break

    if not has_target:
        fh = RotatingFileHandler(
            filename=target,
            maxBytes=int(max_bytes),
            backupCount=int(backups),
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(int(level))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
        lg.addHandler(fh)

    return lg


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
    cfg: Any = None  # for Dry Run access


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
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class _Retcodes:
    invalid: int
    done: int
    done_partial: int
    requote: int
    timeout: int
    price_changed: int
    off_quotes: int
    invalid_fill: int
    invalid_stops: int
    invalid_price: int
    trade_context_busy: int
    no_changes: int


def _mt5_const(name: str, default: int) -> int:
    try:
        return int(getattr(mt5, name, default) or default)
    except Exception:
        return int(default)


_RC = _Retcodes(
    invalid=_mt5_const("TRADE_RETCODE_INVALID", 10013),
    done=_mt5_const("TRADE_RETCODE_DONE", 10009),
    done_partial=_mt5_const("TRADE_RETCODE_DONE_PARTIAL", 10010),
    requote=_mt5_const("TRADE_RETCODE_REQUOTE", 10004),
    timeout=_mt5_const("TRADE_RETCODE_TIMEOUT", 10012),
    price_changed=_mt5_const("TRADE_RETCODE_PRICE_CHANGED", 10020),
    off_quotes=_mt5_const("TRADE_RETCODE_OFF_QUOTES", 10021),
    invalid_fill=_mt5_const("TRADE_RETCODE_INVALID_FILL", 10030),
    invalid_stops=_mt5_const("TRADE_RETCODE_INVALID_STOPS", 10016),
    invalid_price=_mt5_const("TRADE_RETCODE_INVALID_PRICE", 10015),
    trade_context_busy=_mt5_const("TRADE_RETCODE_TRADE_CONTEXT_BUSY", -1),
    no_changes=_mt5_const("TRADE_RETCODE_NO_CHANGES", 10025),
)

_RC_CONNECTION = _mt5_const("TRADE_RETCODE_CONNECTION", 10031)
_RC_NO_CONNECTION = _mt5_const("TRADE_RETCODE_NO_CONNECTION", 10032)
_RC_TOO_MANY_REQUESTS = _mt5_const("TRADE_RETCODE_TOO_MANY_REQUESTS", 10024)
_RC_CLIENT_DISABLES_AT = _mt5_const("TRADE_RETCODE_CLIENT_DISABLES_AT", 10027)
_RC_SERVER_DISABLES_AT = _mt5_const("TRADE_RETCODE_SERVER_DISABLES_AT", 10026)
_RETRYABLE_RETCODES = frozenset(
    {
        _RC.requote,
        _RC.timeout,
        _RC.price_changed,
        _RC.off_quotes,
        _RC.trade_context_busy,
        _RC_CONNECTION,
        _RC_NO_CONNECTION,
        _RC_TOO_MANY_REQUESTS,
        _RC_SERVER_DISABLES_AT,
    }
)


def _ordered_fillings(preferred: Any) -> tuple[int, ...]:
    """
    Return unique filling modes ordered by preference.
    Some brokers expose `symbol_info().filling_mode` as a bitmask (for example 3 => FOK|IOC)
    while `order_send` expects a single `ORDER_FILLING_*` value.
    """
    fok = _mt5_const("ORDER_FILLING_FOK", 0)
    ioc = _mt5_const("ORDER_FILLING_IOC", 1)
    ret = _mt5_const("ORDER_FILLING_RETURN", 2)

    def _unique(*vals: Any) -> tuple[int, ...]:
        out: list[int] = []
        seen: set[int] = set()
        for val in vals:
            try:
                iv = int(val)
            except Exception:
                continue
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
        return tuple(out)

    try:
        pref = int(preferred)
    except Exception:
        pref = None

    direct = (ioc, fok, ret)
    if pref in (fok, ioc, ret):
        return _unique(pref, *direct)

    if isinstance(pref, int) and pref > 2:
        # Common MetaTrader pattern: 1 => FOK, 2 => IOC, 4 => RETURN/BOC-style flag.
        if pref & 2:
            return _unique(ioc, fok, ret)
        if pref & 1:
            return _unique(fok, ioc, ret)
        if pref & 4:
            return _unique(ret, ioc, fok)

    return _unique(*direct)


# =============================================================================
# Executor
# =============================================================================

class OrderExecutor:
    """
    Ultra-fast MT5 market order executor with robust SL/TP attachment.

    Guarantees:
    - No sleep while holding MT5_LOCK
    - SL/TP re-attached using CURRENT tick constraints (stops_level + freeze_level)
    - Ticket resolution works in netting/hedging (bounded polling w/ micro-sleep)
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

        # SLTP tuning (fast + reliable)
        self._sltp_poll_deadline_ms = 650
        self._sltp_poll_sleep_ms_seq = (5, 10, 20, 40, 80, 80, 80)
        self._sltp_send_retries = 4
        self._transport_repair_cooldown_sec = 0.35
        self._last_transport_repair_ts = 0.0

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

        meta = self._get_symbol_meta(request.symbol)
        if meta is None:
            return self._fail(request, time.time(), "symbol_meta_none", 0)

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        vol = self._normalize_volume(float(request.lot), meta)

        last_reason = "unknown"
        last_retcode = 0
        last_err_text = ""
        sent_ts = time.time()
        max_attempts = max(2, int(max_attempts))

        # Instant retry loop (no long sleeps)
        for attempt in range(1, int(max_attempts) + 1):
            try:
                tick, tick_err_text = self._call_mt5_with_last_error("symbol_info_tick", request.symbol)

                if tick is None:
                    last_reason = "tick_none"
                    last_err_text = tick_err_text
                    self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                    continue

                req_price = float(tick.ask) if side == "Buy" else float(tick.bid)
                if req_price <= 0.0:
                    last_reason = "bad_req_price"
                    self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                    continue

                if self.normalize_price_fn:
                    try:
                        req_price = float(self.normalize_price_fn(req_price))
                    except Exception:
                        pass

                sl_s, tp_s = self._sanitize_stops(side, req_price, planned_sl, planned_tp)
                sent_ts = time.time()

                # DRY RUN
                if request.cfg and getattr(request.cfg, "dry_run", False):
                    try:
                        print("\n[DRY RUN] Order BLOCKED by Simulation Mode.")
                        print(f"          Symbol: {request.symbol}")
                        print(f"          Side:   {side}")
                        print(f"          Vol:    {vol}")
                        print(f"          Price:  {req_price}")
                        print(f"          SL:     {sl_s}")
                        print(f"          TP:     {tp_s}")
                    except Exception:
                        pass
                    return OrderResult(
                        order_id=request.order_id,
                        signal_id=request.signal_id,
                        ok=True,
                        reason="dry_run_success",
                        sent_ts=float(sent_ts),
                        fill_ts=time.time(),
                        req_price=float(req_price),
                        exec_price=float(req_price),
                        volume=float(vol),
                        slippage=0.0,
                        retcode=_RC.done,
                    )

                result, last_err_text = self._send_deal(
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
                    self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                    continue

                last_retcode = int(getattr(result, "retcode", 0) or 0)
                result_comment = str(getattr(result, "comment", "") or "").strip()
                if result_comment:
                    if not last_err_text or last_err_text == "(1, 'Success')":
                        last_err_text = result_comment
                    elif result_comment.lower() not in last_err_text.lower():
                        last_err_text = f"{last_err_text} | comment={result_comment}"

                # If broker rejects stops, send without stops and attach robustly
                if last_retcode in (_RC.invalid_stops, _RC.invalid_price) and (planned_sl > 0.0 or planned_tp > 0.0):
                    result2, last_err_text = self._send_deal(
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
                        self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                        continue
                    result = result2
                    last_retcode = int(getattr(result2, "retcode", 0) or 0)
                    result2_comment = str(getattr(result2, "comment", "") or "").strip()
                    if result2_comment:
                        if not last_err_text or last_err_text == "(1, 'Success')":
                            last_err_text = result2_comment
                        elif result2_comment.lower() not in last_err_text.lower():
                            last_err_text = f"{last_err_text} | comment={result2_comment}"

                if last_retcode in (_RC.done, _RC.done_partial):
                    fill_ts = time.time()
                    exec_price = float(getattr(result, "price", req_price) or req_price)
                    exec_vol = float(getattr(result, "volume", vol) or vol)
                    slippage = abs(exec_price - req_price)
                    target_sl, target_tp = self._rebase_stops_for_fill(
                        side,
                        req_price,
                        exec_price,
                        planned_sl,
                        planned_tp,
                    )

                    pos_ticket = int(getattr(result, "position", 0) or 0)
                    ord_ticket = int(getattr(result, "order", 0) or 0)
                    deal_ticket = int(getattr(result, "deal", 0) or 0)

                    # Verify SL/TP post-fill (brokers may ignore stops on market deals)
                    if target_sl > 0.0 or target_tp > 0.0:
                        ok_attach = self._ensure_sltp_robust(
                            symbol=request.symbol,
                            side=side,
                            planned_sl=target_sl,
                            planned_tp=target_tp,
                            magic=int(request.magic),
                            ticket_hint=pos_ticket if pos_ticket > 0 else None,
                            candidate_tickets=(pos_ticket, ord_ticket, deal_ticket),
                            sent_ts=float(sent_ts),
                            meta=meta,
                        )
                        if (not ok_attach) and self.close_on_sltp_fail:
                            self._fail_safe_close(request.symbol, side, exec_vol, int(request.magic))

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
                        order_ticket=int(ord_ticket),
                        deal_ticket=int(deal_ticket),
                        position_ticket=int(pos_ticket),
                    )

                if last_retcode == _RC_CLIENT_DISABLES_AT:
                    last_reason = "client_autotrading_disabled"
                    break

                # transient -> retry quickly
                if self._is_retryable_retcode(last_retcode, last_err_text):
                    last_reason = f"retry_retcode_{last_retcode}"
                    self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                last_err_text = str(exc)
                self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                log_err.error("ORDER_EXCEPTION | attempt=%s err=%s tb=%s", attempt, exc, traceback.format_exc())
                continue

        recovered = self._probe_recovered_fill(
            symbol=request.symbol,
            side=side,
            volume=float(vol),
            magic=int(request.magic),
            sent_ts=float(sent_ts),
        )
        if recovered is not None:
            rec_fill_ts, rec_price, rec_vol, rec_deal_ticket = recovered
            self._safe_hook(
                hooks,
                "record_execution_metrics",
                request.order_id,
                side,
                float(request.enqueue_time),
                float(sent_ts),
                float(rec_fill_ts),
                float(request.price),
                float(rec_price),
                float(abs(float(rec_price) - float(request.price))),
            )
            return OrderResult(
                order_id=request.order_id,
                signal_id=request.signal_id,
                ok=True,
                reason="recovered_after_transport",
                sent_ts=float(sent_ts),
                fill_ts=float(rec_fill_ts),
                req_price=float(request.price),
                exec_price=float(rec_price),
                volume=float(rec_vol),
                slippage=float(abs(float(rec_price) - float(request.price))),
                retcode=int(last_retcode),
                order_ticket=0,
                deal_ticket=int(rec_deal_ticket),
                position_ticket=0,
            )

        self._safe_hook(
            hooks,
            "record_execution_failure",
            request.order_id,
            float(request.enqueue_time),
            float(sent_ts),
            last_reason,
        )
        log_err.error(
            "ORDER_FAILED | order_id=%s side=%s symbol=%s reason=%s retcode=%s last_error=%s",
            request.order_id,
            side,
            request.symbol,
            last_reason,
            last_retcode,
            last_err_text or "-",
        )
        return self._fail(request, sent_ts, last_reason, last_retcode)

    # -------------------- Small helpers --------------------

    @staticmethod
    def _safe_hook(hooks: dict[str, Callable[..., Any]], name: str, *args: Any) -> None:
        fn = hooks.get(name)
        if not fn:
            return
        try:
            fn(*args)
        except Exception:
            return

    @staticmethod
    def _normalize_mt5_call(
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        call_args = tuple(args)
        call_kwargs = dict(kwargs)
        name = str(method_name)

        # MT5 bindings are inconsistent about positional arguments across methods.
        # Keep order requests positional for compatibility with wrappers/monkeypatches.
        if name in ("order_send", "order_check"):
            if len(call_args) == 1 and not call_kwargs and isinstance(call_args[0], dict):
                return (call_args[0],), {}
            if not call_args and set(call_kwargs) == {"request"} and isinstance(call_kwargs["request"], dict):
                return (call_kwargs["request"],), {}
        if name == "positions_get" and len(call_args) == 1 and not call_kwargs and isinstance(call_args[0], str):
            return (), {"symbol": call_args[0]}

        return call_args, call_kwargs

    @classmethod
    def _call_mt5(cls, method_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            raise AttributeError(f"mt5.{method_name} not callable")

        call_args, call_kwargs = cls._normalize_mt5_call(str(method_name), tuple(args), dict(kwargs))
        with MT5_LOCK:
            if str(method_name) in ("order_send", "order_check") and len(call_args) == 1 and not call_kwargs:
                return fn(call_args[0])
            return fn(*call_args, **call_kwargs)

    @classmethod
    def _call_mt5_with_last_error(cls, method_name: str, *args: Any, **kwargs: Any) -> tuple[Any, str]:
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            raise AttributeError(f"mt5.{method_name} not callable")

        call_args, call_kwargs = cls._normalize_mt5_call(str(method_name), tuple(args), dict(kwargs))
        with MT5_LOCK:
            if str(method_name) in ("order_send", "order_check") and len(call_args) == 1 and not call_kwargs:
                result = fn(call_args[0])
            else:
                result = fn(*call_args, **call_kwargs)
            try:
                last_error = str(mt5.last_error())
            except Exception:
                last_error = ""
        return result, last_error

    def _is_retryable_retcode(self, retcode: int, last_error_text: str = "") -> bool:
        if int(retcode) in _RETRYABLE_RETCODES:
            return True
        msg = str(last_error_text or "").lower()
        if "timeout" in msg:
            return True
        if "ipc" in msg or "-10005" in msg:
            return True
        if "no connection" in msg or "connection" in msg:
            return True
        return False

    def _maybe_repair_transport(self, reason: str, retcode: int, last_error_text: str) -> None:
        if not self._is_retryable_retcode(int(retcode or 0), str(last_error_text or "") + " " + str(reason or "")):
            return
        now = time.monotonic()
        if (now - self._last_transport_repair_ts) < float(self._transport_repair_cooldown_sec):
            return
        self._last_transport_repair_ts = now
        try:
            ensure_mt5()
        except Exception:
            return

    def _probe_recovered_fill(
        self,
        *,
        symbol: str,
        side: str,
        volume: float,
        magic: int,
        sent_ts: float,
    ) -> Optional[tuple[float, float, float, int]]:
        """
        Best-effort recovery probe after ambiguous transport failure.
        Returns (fill_ts, price, volume, deal_ticket) if a matching fill is found.
        """
        try:
            now_dt = datetime.now()
            start_dt = datetime.fromtimestamp(max(0.0, float(sent_ts) - 5.0))
            with MT5_LOCK:
                deals = mt5.history_deals_get(start_dt, now_dt) or []
        except Exception:
            return None

        if not deals:
            return None

        want_type = mt5.ORDER_TYPE_BUY if str(side) == "Buy" else mt5.ORDER_TYPE_SELL
        entry_in = int(getattr(mt5, "DEAL_ENTRY_IN", 0) or 0)
        best: Optional[tuple[float, float, float, int]] = None
        best_score: Optional[tuple[float, float]] = None

        for d in deals:
            try:
                if str(getattr(d, "symbol", "")) != str(symbol):
                    continue
                if int(getattr(d, "magic", 0) or 0) != int(magic):
                    continue
                if int(getattr(d, "entry", 0) or 0) != entry_in:
                    continue
                if int(getattr(d, "type", -1) or -1) != int(want_type):
                    continue

                d_time = float(getattr(d, "time", 0.0) or 0.0)
                if d_time <= 0.0:
                    continue
                if d_time < (float(sent_ts) - 5.0):
                    continue

                d_vol = float(getattr(d, "volume", 0.0) or 0.0)
                if d_vol <= 0.0:
                    continue

                # choose nearest volume + most recent deal
                vol_delta = abs(float(d_vol) - float(volume))
                recency = -float(d_time)
                score = (float(vol_delta), recency)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (
                        float(d_time),
                        float(getattr(d, "price", 0.0) or 0.0),
                        float(d_vol),
                        int(getattr(d, "ticket", 0) or 0),
                    )
            except Exception:
                continue

        return best

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

    # =============================================================================
    # Symbol meta cache
    # =============================================================================

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

            filling = _ordered_fillings(getattr(info, "filling_mode", None))[0]

            v_step = float(getattr(info, "volume_step", 0.01) or 0.01)
            if v_step <= 0.0:
                v_step = 0.01
            v_min = float(getattr(info, "volume_min", v_step) or v_step)
            v_max = float(getattr(info, "volume_max", v_min) or v_min)

            # CRITICAL FIX: correct decimals for steps like 0.25 (log10 is wrong for non-powers of 10)
            try:
                v_decimals = max(0, min(8, -Decimal(str(v_step)).normalize().as_tuple().exponent))
            except Exception:
                v_decimals = 2

            stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
            freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)

            m2 = _SymbolMeta(
                symbol=str(symbol),
                digits=int(digits),
                point=float(point if point > 0 else 0.00001),
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

        vmin, vmax, step = float(meta.v_min), float(meta.v_max), float(meta.v_step)
        vol = max(vmin, min(vmax, float(vol)))
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
    def _rebase_stops_for_fill(
        side: str,
        requested_price: float,
        filled_price: float,
        planned_sl: float,
        planned_tp: float,
    ) -> tuple[float, float]:
        """
        Preserve the original stop/target distance after market-order slippage.

        The strategy still exits via SL/TP; this only rebases the absolute levels
        so a slipped fill does not silently widen risk.
        """
        side_s = str(side or "")
        req = float(requested_price or 0.0)
        fill = float(filled_price or 0.0)
        sl = float(planned_sl or 0.0)
        tp = float(planned_tp or 0.0)
        if req <= 0.0 or fill <= 0.0 or side_s not in ("Buy", "Sell"):
            return sl, tp

        if side_s == "Buy":
            sl_dist = max(0.0, req - sl) if sl > 0.0 else 0.0
            tp_dist = max(0.0, tp - req) if tp > 0.0 else 0.0
            sl2 = (fill - sl_dist) if sl_dist > 0.0 else 0.0
            tp2 = (fill + tp_dist) if tp_dist > 0.0 else 0.0
        else:
            sl_dist = max(0.0, sl - req) if sl > 0.0 else 0.0
            tp_dist = max(0.0, req - tp) if tp > 0.0 else 0.0
            sl2 = (fill + sl_dist) if sl_dist > 0.0 else 0.0
            tp2 = (fill - tp_dist) if tp_dist > 0.0 else 0.0

        return float(sl2 if sl2 > 0.0 else 0.0), float(tp2 if tp2 > 0.0 else 0.0)

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
    ) -> tuple[Any, str]:
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
        }

        fill_candidates = _ordered_fillings(filling)
        last_result = None
        last_err_text = ""
        for idx, fill_mode in enumerate(fill_candidates):
            req["type_filling"] = int(fill_mode)
            result, last_err_text = OrderExecutor._call_mt5_with_last_error("order_send", req)
            last_result = result

            if result is None:
                break

            retcode = int(getattr(result, "retcode", 0) or 0)
            if retcode not in (_RC.invalid, _RC.invalid_fill):
                return result, last_err_text

            if idx == len(fill_candidates) - 1:
                return result, last_err_text

        return last_result, last_err_text

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
    ) -> bool:
        if planned_sl <= 0.0 and planned_tp <= 0.0:
            return True

        resolved = self._resolve_position_ticket_robust(
            symbol=symbol,
            side=side,
            magic=magic,
            ticket_hint=ticket_hint,
            candidate_tickets=candidate_tickets,
            sent_ts=sent_ts,
            deadline_ms=int(self._sltp_poll_deadline_ms),
        )

        # 0 => no position found (netting offset / instant close)
        if resolved == 0:
            return True

        if resolved is None:
            log_err.error(
                "SLTP_RESOLVE_FAILED | symbol=%s side=%s magic=%s candidates=%s",
                symbol,
                side,
                int(magic),
                candidate_tickets,
            )
            return False

        cur_sl, cur_tp = self._get_position_sltp(resolved)
        if self._sltp_is_good(cur_sl, cur_tp, planned_sl, planned_tp, meta.point):
            return True

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
                int(resolved),
                symbol,
                side,
                float(planned_sl),
                float(planned_tp),
                ctx,
            )
            return False

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
            r, last_le = self._call_mt5_with_last_error("order_send", req)
            last_rc = int(getattr(r, "retcode", 0) or 0) if r is not None else 0

            if r is not None and last_rc in (_RC.done, _RC.done_partial, _RC.no_changes):
                return True

            if last_rc in (_RC.invalid_stops, _RC.trade_context_busy):
                expand_mult = float(i) * 2.0
                sl2, tp2, ctx = self._sanitize_sltp_by_market_constraints(
                    symbol=symbol,
                    side=side,
                    desired_sl=planned_sl,
                    desired_tp=planned_tp,
                    meta=meta,
                    expand_dist_mult=expand_mult,
                )
                req["sl"] = float(sl2) if sl2 > 0 else 0.0
                req["tp"] = float(tp2) if tp2 > 0 else 0.0
            elif self._is_retryable_retcode(last_rc, last_le):
                self._maybe_repair_transport("sltp_retry", last_rc, last_le)

            time.sleep((10 if i == 1 else 20 if i == 2 else 40) / 1000.0)

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
            try:
                p = self._call_mt5("positions_get", ticket=int(ticket)) or []
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
        expand_dist_mult: float = 1.0,
    ) -> tuple[float, float, dict[str, Any]]:
        ctx: dict[str, Any] = {"symbol": symbol}

        tick = self._call_mt5("symbol_info_tick", symbol)

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

        min_level = max(int(meta.stops_level), int(meta.freeze_level))
        base_safety = 5.0 * point
        min_dist_abs = (float(min_level) * point) + (base_safety * float(expand_dist_mult))

        sl = float(desired_sl) if float(desired_sl) > 0 else 0.0
        tp = float(desired_tp) if float(desired_tp) > 0 else 0.0

        if side == "Buy":
            if sl > 0.0:
                sl = min(sl, bid - min_dist_abs)
            if tp > 0.0:
                tp = max(tp, bid + min_dist_abs)
            if sl > 0.0 and not (sl < bid):
                sl = 0.0
            if tp > 0.0 and not (tp > bid):
                tp = 0.0
        else:
            if sl > 0.0:
                sl = max(sl, ask + min_dist_abs)
            if tp > 0.0:
                tp = min(tp, ask - min_dist_abs)
            if sl > 0.0 and not (sl > ask):
                sl = 0.0
            if tp > 0.0 and not (tp < ask):
                tp = 0.0

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
                "dist_required": float(min_dist_abs),
                "expand_mult": float(expand_dist_mult),
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
                try:
                    p = self._call_mt5("positions_get", ticket=int(tk)) or []
                except TypeError:
                    p = []
                return bool(p and str(getattr(p[0], "symbol", "")) == str(symbol))
            except Exception:
                return False

        direct = [int(ticket_hint or 0)] + [int(x or 0) for x in candidate_tickets]
        for tk in direct:
            if tk > 0 and _pos_exists_by_ticket(tk):
                return tk

        deadline = time.perf_counter() + (float(deadline_ms) / 1000.0)
        best_ticket = 0
        best_score = None
        saw_any = False
        sleep_seq = list(self._sltp_poll_sleep_ms_seq)

        while time.perf_counter() < deadline:
            try:
                positions = self._call_mt5("positions_get", symbol=symbol) or []
            except Exception:
                positions = []

            if positions:
                saw_any = True

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

            time.sleep((sleep_seq.pop(0) if sleep_seq else 80) / 1000.0)

        if not saw_any:
            return 0
        return None

    # =============================================================================
    # Fail-safe close if SLTP attach fails
    # =============================================================================

    @staticmethod
    def _fail_safe_close(symbol: str, side: str, volume: float, magic: int) -> None:
        try:
            want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            positions = OrderExecutor._call_mt5("positions_get", symbol=symbol) or []

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

                    tick = OrderExecutor._call_mt5("symbol_info_tick", symbol)
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

                    r, le = OrderExecutor._call_mt5_with_last_error("order_send", req)

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

@dataclass(frozen=True)
class BrokerSimulationConfig:
    starting_balance: float = 1_000.0
    leverage: int = 200
    latency_ms_min: int = 50
    latency_ms_max: int = 500
    slippage_pips_min: int = 1
    slippage_pips_max: int = 5
    requote_probability: float = 0.08
    disconnect_duration_sec: float = 10.0
    spread_points_xau: float = 25.0
    spread_points_btc: float = 150.0
    seed: int = 42


class BrokerEnvironmentSimulator:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    TRADE_RETCODE_DONE = _RC.done
    TRADE_RETCODE_REQUOTE = _RC.requote
    TRADE_RETCODE_CONNECTION = _RC_CONNECTION
    TRADE_RETCODE_NO_CONNECTION = _RC_NO_CONNECTION
    ORDER_FILLING_IOC = _mt5_const("ORDER_FILLING_IOC", 1)

    def __init__(self, cfg: Optional[BrokerSimulationConfig] = None) -> None:
        self.cfg = cfg or BrokerSimulationConfig()
        self._rand = random.Random(int(self.cfg.seed))
        self._lock = threading.RLock()
        self._disconnect_until: float = 0.0
        self._hidden_fill_budget: int = 0
        self._next_ticket: int = 100_000
        self._balance: float = float(self.cfg.starting_balance)
        self._equity: float = float(self.cfg.starting_balance)
        self._positions: dict[int, dict[str, Any]] = {}
        self._deals: list[Any] = []
        self._symbols: dict[str, dict[str, Any]] = {
            "XAUUSDm": {
                "symbol": "XAUUSDm",
                "digits": 2,
                "point": 0.01,
                "pip_size": 0.01,
                "contract_size": 100.0,
                "spread_points": float(self.cfg.spread_points_xau),
                "mid": 2350.00,
                "volume_step": 0.01,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "trade_stops_level": 0,
                "trade_freeze_level": 0,
            },
            "BTCUSDm": {
                "symbol": "BTCUSDm",
                "digits": 2,
                "point": 0.01,
                "pip_size": 0.01,
                "contract_size": 1.0,
                "spread_points": float(self.cfg.spread_points_btc),
                "mid": 68000.00,
                "volume_step": 0.01,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "trade_stops_level": 0,
                "trade_freeze_level": 0,
            },
        }
        self._aliases: dict[str, str] = {
            "XAU": "XAUUSDm",
            "XAUUSD": "XAUUSDm",
            "XAUUSDM": "XAUUSDm",
            "BTC": "BTCUSDm",
            "BTCUSD": "BTCUSDm",
            "BTCUSDM": "BTCUSDm",
        }
        for symbol in list(self._symbols):
            self._refresh_tick_locked(symbol)

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def connected(self) -> bool:
        return time.time() >= float(self._disconnect_until or 0.0)

    def start_disconnect(self, duration_sec: Optional[float] = None, *, hidden_fill_next: bool = False) -> None:
        duration = float(duration_sec or self.cfg.disconnect_duration_sec)
        with self._lock:
            self._disconnect_until = max(float(self._disconnect_until or 0.0), time.time() + max(0.1, duration))
            if hidden_fill_next:
                self._hidden_fill_budget += 1

    def stop_disconnect(self) -> None:
        with self._lock:
            self._disconnect_until = 0.0

    def _resolve_symbol(self, symbol: str) -> str:
        raw = str(symbol or "").strip()
        if raw in self._symbols:
            return raw
        key = raw.upper().rstrip(".")
        return self._aliases.get(key, raw or "XAUUSDm")

    def _refresh_tick_locked(self, symbol: str) -> None:
        meta = self._symbols[self._resolve_symbol(symbol)]
        point = float(meta["point"])
        spread_points = max(1.0, float(meta["spread_points"]))
        mid = float(meta["mid"])
        half_spread = point * spread_points * 0.5
        meta["bid"] = round(mid - half_spread, int(meta["digits"]))
        meta["ask"] = round(mid + half_spread, int(meta["digits"]))

    def _mark_to_market_locked(self) -> None:
        unrealized = 0.0
        for pos in self._positions.values():
            symbol = self._resolve_symbol(pos["symbol"])
            meta = self._symbols[symbol]
            bid = float(meta["bid"])
            ask = float(meta["ask"])
            exit_price = bid if int(pos["type"]) == self.POSITION_TYPE_BUY else ask
            direction = 1.0 if int(pos["type"]) == self.POSITION_TYPE_BUY else -1.0
            pnl = (float(exit_price) - float(pos["price_open"])) * direction
            pnl *= float(pos["volume"]) * float(meta["contract_size"])
            pos["profit"] = float(pnl)
            unrealized += float(pnl)
        self._equity = float(self._balance + unrealized)

    def _next_id_locked(self) -> int:
        self._next_ticket += 1
        return int(self._next_ticket)

    def _entry_deal_locked(self, pos: dict[str, Any]) -> None:
        self._deals.append(
            SimpleNamespace(
                symbol=str(pos["symbol"]),
                magic=int(pos["magic"]),
                entry=int(self.DEAL_ENTRY_IN),
                type=int(self.ORDER_TYPE_BUY if int(pos["type"]) == self.POSITION_TYPE_BUY else self.ORDER_TYPE_SELL),
                time=float(pos["time"]),
                price=float(pos["price_open"]),
                volume=float(pos["volume"]),
                profit=0.0,
                commission=0.0,
                swap=0.0,
                fee=0.0,
                ticket=int(self._next_id_locked()),
            )
        )

    def _close_position_locked(self, ticket: int, exit_price: float, now_ts: float, reason: str) -> None:
        pos = self._positions.pop(int(ticket), None)
        if pos is None:
            return
        symbol = self._resolve_symbol(pos["symbol"])
        meta = self._symbols[symbol]
        direction = 1.0 if int(pos["type"]) == self.POSITION_TYPE_BUY else -1.0
        pnl = (float(exit_price) - float(pos["price_open"])) * direction
        pnl *= float(pos["volume"]) * float(meta["contract_size"])
        self._balance += float(pnl)
        self._deals.append(
            SimpleNamespace(
                symbol=str(pos["symbol"]),
                magic=int(pos["magic"]),
                entry=int(self.DEAL_ENTRY_OUT),
                type=int(self.ORDER_TYPE_BUY if int(pos["type"]) == self.POSITION_TYPE_BUY else self.ORDER_TYPE_SELL),
                time=float(now_ts),
                price=float(exit_price),
                volume=float(pos["volume"]),
                profit=float(pnl),
                commission=0.0,
                swap=0.0,
                fee=0.0,
                reason=str(reason),
                ticket=int(self._next_id_locked()),
            )
        )

    def _check_tp_sl_locked(self, now_ts: float) -> None:
        to_close: list[tuple[int, float, str]] = []
        for ticket, pos in list(self._positions.items()):
            symbol = self._resolve_symbol(pos["symbol"])
            meta = self._symbols[symbol]
            bid = float(meta["bid"])
            ask = float(meta["ask"])
            sl = float(pos.get("sl", 0.0) or 0.0)
            tp = float(pos.get("tp", 0.0) or 0.0)
            if int(pos["type"]) == self.POSITION_TYPE_BUY:
                if sl > 0.0 and bid <= sl:
                    to_close.append((int(ticket), float(sl), "sl_hit"))
                elif tp > 0.0 and bid >= tp:
                    to_close.append((int(ticket), float(tp), "tp_hit"))
            else:
                if sl > 0.0 and ask >= sl:
                    to_close.append((int(ticket), float(sl), "sl_hit"))
                elif tp > 0.0 and ask <= tp:
                    to_close.append((int(ticket), float(tp), "tp_hit"))
        for ticket, exit_price, reason in to_close:
            self._close_position_locked(ticket, exit_price, now_ts, reason)
        self._mark_to_market_locked()

    def update_market(self, symbol: str, mid_price: float, *, ts: Optional[float] = None) -> None:
        with self._lock:
            key = self._resolve_symbol(symbol)
            if key not in self._symbols:
                return
            self._symbols[key]["mid"] = float(mid_price)
            self._refresh_tick_locked(key)
            self._check_tp_sl_locked(float(ts or time.time()))

    def terminal_info(self) -> Any:
        if not self.connected():
            return None
        return SimpleNamespace(connected=True, trade_allowed=True)

    def account_info(self) -> Any:
        if not self.connected():
            return None
        with self._lock:
            self._mark_to_market_locked()
            return SimpleNamespace(
                balance=float(self._balance),
                equity=float(self._equity),
                margin_free=float(self._equity),
                leverage=int(self.cfg.leverage),
                trade_allowed=True,
                login=0,
                server="SIMULATOR",
            )

    def symbol_info_tick(self, symbol: str) -> Any:
        if not self.connected():
            return None
        with self._lock:
            key = self._resolve_symbol(symbol)
            meta = self._symbols.get(key)
            if meta is None:
                return None
            return SimpleNamespace(
                bid=float(meta["bid"]),
                ask=float(meta["ask"]),
                time=int(time.time()),
                time_msc=int(time.time() * 1000.0),
            )

    def symbol_info(self, symbol: str) -> Any:
        if not self.connected():
            return None
        with self._lock:
            key = self._resolve_symbol(symbol)
            meta = self._symbols.get(key)
            if meta is None:
                return None
            return SimpleNamespace(
                name=str(meta["symbol"]),
                point=float(meta["point"]),
                digits=int(meta["digits"]),
                filling_mode=int(self.ORDER_FILLING_IOC),
                volume_step=float(meta["volume_step"]),
                volume_min=float(meta["volume_min"]),
                volume_max=float(meta["volume_max"]),
                trade_stops_level=int(meta["trade_stops_level"]),
                trade_freeze_level=int(meta["trade_freeze_level"]),
                trade_contract_size=float(meta["contract_size"]),
            )

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None) -> Any:
        if not self.connected():
            return None
        with self._lock:
            self._mark_to_market_locked()
            out: list[Any] = []
            key = self._resolve_symbol(symbol) if symbol else ""
            for pos_ticket, pos in self._positions.items():
                if ticket is not None and int(pos_ticket) != int(ticket):
                    continue
                if key and self._resolve_symbol(pos["symbol"]) != key:
                    continue
                out.append(
                    SimpleNamespace(
                        symbol=str(pos["symbol"]),
                        type=int(pos["type"]),
                        magic=int(pos["magic"]),
                        volume=float(pos["volume"]),
                        price_open=float(pos["price_open"]),
                        ticket=int(pos_ticket),
                        time=float(pos["time"]),
                        profit=float(pos.get("profit", 0.0) or 0.0),
                        swap=0.0,
                        sl=float(pos.get("sl", 0.0) or 0.0),
                        tp=float(pos.get("tp", 0.0) or 0.0),
                    )
                )
            return out

    def history_deals_get(self, start: Any, end: Any) -> Any:
        if not self.connected():
            return None
        try:
            start_ts = float(getattr(start, "timestamp", lambda: float(start))())
        except Exception:
            try:
                start_ts = float(start.timestamp())
            except Exception:
                start_ts = 0.0
        try:
            end_ts = float(getattr(end, "timestamp", lambda: float(end))())
        except Exception:
            try:
                end_ts = float(end.timestamp())
            except Exception:
                end_ts = time.time()
        with self._lock:
            return [d for d in self._deals if start_ts <= float(getattr(d, "time", 0.0) or 0.0) <= end_ts]

    @staticmethod
    def last_error() -> tuple[int, str]:
        return (1, "Success")

    def async_call(self, method_name: str, *args: Any, timeout: Optional[float] = None, default: Any = None, **kwargs: Any) -> Any:
        try:
            fn = getattr(self, str(method_name))
            if not callable(fn):
                return default
            value = fn(*args, **kwargs)
            return default if value is None else value
        except Exception:
            return default

    def _pip_size_locked(self, symbol: str) -> float:
        meta = self._symbols[self._resolve_symbol(symbol)]
        return float(meta["pip_size"])

    def _open_position_locked(self, request: OrderRequest, exec_price: float, now_ts: float) -> dict[str, Any]:
        symbol = self._resolve_symbol(request.symbol)
        side = str(request.signal)
        pos_type = self.POSITION_TYPE_BUY if side == "Buy" else self.POSITION_TYPE_SELL
        ticket = self._next_id_locked()
        sl_val, tp_val = OrderExecutor._rebase_stops_for_fill(
            side,
            float(request.price),
            float(exec_price),
            float(request.sl),
            float(request.tp),
        )
        pos = {
            "symbol": symbol,
            "type": int(pos_type),
            "magic": int(request.magic),
            "volume": float(request.lot),
            "price_open": float(exec_price),
            "ticket": int(ticket),
            "time": float(now_ts),
            "sl": float(sl_val),
            "tp": float(tp_val),
            "order_id": str(request.order_id),
            "signal_id": str(request.signal_id),
            "profit": 0.0,
        }
        self._positions[int(ticket)] = pos
        self._entry_deal_locked(pos)
        self._mark_to_market_locked()
        return pos

    def close_all_positions(self, reason: str = "flatten") -> dict[str, Any]:
        with self._lock:
            closed = 0
            for ticket, pos in list(self._positions.items()):
                symbol = self._resolve_symbol(pos["symbol"])
                meta = self._symbols[symbol]
                exit_price = (
                    float(meta["bid"])
                    if int(pos["type"]) == self.POSITION_TYPE_BUY
                    else float(meta["ask"])
                )
                self._close_position_locked(int(ticket), float(exit_price), time.time(), str(reason))
                closed += 1
            self._mark_to_market_locked()
        return {
            "ok": True,
            "closed": int(closed),
            "canceled": 0,
            "errors": [],
        }

    def execute_order(
        self,
        request: OrderRequest,
        *,
        telemetry_hooks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> OrderResult:
        hooks = telemetry_hooks or {}
        side = str(request.signal)
        sent_ts = time.time()
        with self._lock:
            disconnect_active = not self.connected()
            hidden_fill = disconnect_active and self._hidden_fill_budget > 0
            if hidden_fill:
                self._hidden_fill_budget -= 1
        tick = self.symbol_info_tick(request.symbol)
        if tick is None and not hidden_fill:
            reason = f"retry_retcode_{self.TRADE_RETCODE_NO_CONNECTION}:sim_no_connection"
            fn = hooks.get("record_execution_failure")
            if callable(fn):
                fn(str(request.order_id), float(request.enqueue_time), float(sent_ts), reason)
            return OrderResult(
                order_id=str(request.order_id),
                signal_id=str(request.signal_id),
                ok=False,
                reason=reason,
                sent_ts=float(sent_ts),
                fill_ts=float(sent_ts),
                req_price=float(request.price),
                exec_price=float(request.price),
                volume=float(request.lot),
                slippage=0.0,
                retcode=int(self.TRADE_RETCODE_NO_CONNECTION),
            )
        if tick is None and hidden_fill:
            with self._lock:
                meta = self._symbols[self._resolve_symbol(request.symbol)]
                req_price = float(meta["ask"]) if side == "Buy" else float(meta["bid"])
                slippage_pips = self._rand.randint(int(self.cfg.slippage_pips_min), int(self.cfg.slippage_pips_max))
                slip = float(slippage_pips) * self._pip_size_locked(request.symbol)
                exec_price = req_price + slip if side == "Buy" else req_price - slip
                pos = self._open_position_locked(request, float(exec_price), float(sent_ts))
            reason = "order_send_none:sim_disconnect_hidden_fill"
            fn = hooks.get("record_execution_failure")
            if callable(fn):
                fn(str(request.order_id), float(request.enqueue_time), float(sent_ts), reason)
            return OrderResult(
                order_id=str(request.order_id),
                signal_id=str(request.signal_id),
                ok=False,
                reason=reason,
                sent_ts=float(sent_ts),
                fill_ts=float(sent_ts),
                req_price=float(req_price),
                exec_price=float(exec_price),
                volume=float(request.lot),
                slippage=float(abs(exec_price - req_price)),
                retcode=int(self.TRADE_RETCODE_CONNECTION),
                position_ticket=int(pos["ticket"]),
            )

        latency_ms = self._rand.randint(int(self.cfg.latency_ms_min), int(self.cfg.latency_ms_max))
        time.sleep(max(0.0, float(latency_ms) / 1000.0))
        sent_ts = float(sent_ts)
        fill_ts = time.time()
        req_price = float(tick.ask) if side == "Buy" else float(tick.bid)

        if disconnect_active:
            if hidden_fill:
                slippage_pips = self._rand.randint(int(self.cfg.slippage_pips_min), int(self.cfg.slippage_pips_max))
                slip = float(slippage_pips) * self._pip_size_locked(request.symbol)
                exec_price = req_price + slip if side == "Buy" else req_price - slip
                with self._lock:
                    pos = self._open_position_locked(request, float(exec_price), float(fill_ts))
                reason = "order_send_none:sim_disconnect_hidden_fill"
                fn = hooks.get("record_execution_failure")
                if callable(fn):
                    fn(str(request.order_id), float(request.enqueue_time), float(sent_ts), reason)
                return OrderResult(
                    order_id=str(request.order_id),
                    signal_id=str(request.signal_id),
                    ok=False,
                    reason=reason,
                    sent_ts=float(sent_ts),
                    fill_ts=float(fill_ts),
                    req_price=float(req_price),
                    exec_price=float(exec_price),
                    volume=float(request.lot),
                    slippage=float(abs(exec_price - req_price)),
                    retcode=int(self.TRADE_RETCODE_CONNECTION),
                    position_ticket=int(pos["ticket"]),
                )
            reason = f"retry_retcode_{self.TRADE_RETCODE_CONNECTION}:sim_disconnect"
            fn = hooks.get("record_execution_failure")
            if callable(fn):
                fn(str(request.order_id), float(request.enqueue_time), float(sent_ts), reason)
            return OrderResult(
                order_id=str(request.order_id),
                signal_id=str(request.signal_id),
                ok=False,
                reason=reason,
                sent_ts=float(sent_ts),
                fill_ts=float(fill_ts),
                req_price=float(req_price),
                exec_price=float(req_price),
                volume=float(request.lot),
                slippage=0.0,
                retcode=int(self.TRADE_RETCODE_CONNECTION),
            )

        if self._rand.random() < float(self.cfg.requote_probability):
            reason = f"retry_retcode_{self.TRADE_RETCODE_REQUOTE}:sim_requote"
            fn = hooks.get("record_execution_failure")
            if callable(fn):
                fn(str(request.order_id), float(request.enqueue_time), float(sent_ts), reason)
            return OrderResult(
                order_id=str(request.order_id),
                signal_id=str(request.signal_id),
                ok=False,
                reason=reason,
                sent_ts=float(sent_ts),
                fill_ts=float(fill_ts),
                req_price=float(req_price),
                exec_price=float(req_price),
                volume=float(request.lot),
                slippage=0.0,
                retcode=int(self.TRADE_RETCODE_REQUOTE),
            )

        slippage_pips = self._rand.randint(int(self.cfg.slippage_pips_min), int(self.cfg.slippage_pips_max))
        slip = float(slippage_pips) * self._pip_size_locked(request.symbol)
        exec_price = req_price + slip if side == "Buy" else req_price - slip
        with self._lock:
            pos = self._open_position_locked(request, float(exec_price), float(fill_ts))
        fn = hooks.get("record_execution_metrics")
        if callable(fn):
            fn(
                str(request.order_id),
                side,
                float(request.enqueue_time),
                float(sent_ts),
                float(fill_ts),
                float(req_price),
                float(exec_price),
                float(abs(exec_price - req_price)),
            )
        return OrderResult(
            order_id=str(request.order_id),
            signal_id=str(request.signal_id),
            ok=True,
            reason="sim_filled",
            sent_ts=float(sent_ts),
            fill_ts=float(fill_ts),
            req_price=float(req_price),
            exec_price=float(exec_price),
            volume=float(request.lot),
            slippage=float(abs(exec_price - req_price)),
            retcode=int(self.TRADE_RETCODE_DONE),
            position_ticket=int(pos["ticket"]),
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._mark_to_market_locked()
            return {
                "balance": round(float(self._balance), 2),
                "equity": round(float(self._equity), 2),
                "open_positions": len(self._positions),
                "deals": len(self._deals),
                "connected": self.connected(),
            }


class SimulatedOrderExecutor:
    def __init__(self, simulator: BrokerEnvironmentSimulator) -> None:
        self.simulator = simulator

    def send_market_order(
        self,
        request: OrderRequest,
        *,
        max_attempts: int = 2,
        telemetry_hooks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> OrderResult:
        _ = max_attempts
        return self.simulator.execute_order(request, telemetry_hooks=telemetry_hooks)


__all__ = [
    "OrderRequest",
    "OrderResult",
    "OrderExecutor",
    "BrokerSimulationConfig",
    "BrokerEnvironmentSimulator",
    "SimulatedOrderExecutor",
]
