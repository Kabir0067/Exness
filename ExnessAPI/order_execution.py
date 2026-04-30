#!/usr/bin/env python3
"""
Institutional-Grade MT5 Market Order Executor with Robust SL/TP Attachment.

Ultra-low-latency, zero-downtime order execution engine for high-frequency
scalping and institutional trading. Guarantees:
- No blocking sleeps while holding the global MT5 lock
- SL/TP re-attachment using live tick + broker stops/freeze levels
- Automatic fill recovery after transient transport failures
- Idempotent handler setup and comprehensive error telemetry

All public interfaces are preserved exactly for seamless integration.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional, cast

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
    """Create or reuse a rotating file logger with exact target deduplication."""
    lg = logging.getLogger(name)
    lg.setLevel(int(level))
    lg.propagate = False

    target = str(get_log_path(filename))
    for h in lg.handlers:
        if (
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == target
        ):
            h.setLevel(int(level))
            return lg

    fh = RotatingFileHandler(
        filename=target,
        maxBytes=int(max_bytes),
        backupCount=int(backups),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(int(level))
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
        )
    )
    lg.addHandler(fh)
    return lg


log_err = _rotating_file_logger("order_execution", "order_execution.log", logging.ERROR)

# Health logs OFF by default for maximum speed (enable only during debugging)
_HEALTH_ENABLED = False
log_health = logging.getLogger("order_execution.health")
log_health.setLevel(logging.CRITICAL)
log_health.propagate = False


# =============================================================================
# Models (memory-optimized with slots)
# =============================================================================


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Immutable order request DTO carrying strategy signal + risk parameters."""

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


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Immutable execution result with full telemetry for risk / analytics."""

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


@dataclass(frozen=True, slots=True)
class _SymbolMeta:
    """Cached symbol trading constraints (refreshed every symbol_cache_ttl)."""

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


@dataclass(frozen=True, slots=True)
class _Retcodes:
    """Broker return-code constants (resolved at import for speed)."""

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
    """Resolve MT5 constant with safe fallback (used at module import)."""
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

_RETRYABLE_RETCODES: frozenset[int] = frozenset(
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
    Return unique filling modes ordered by preference (FOK > IOC > RETURN).

    Handles both direct constants and broker bitmasks transparently.
    Extracted helper guarantees deduplication and stable ordering.
    """
    fok = _mt5_const("ORDER_FILLING_FOK", 0)
    ioc = _mt5_const("ORDER_FILLING_IOC", 1)
    ret = _mt5_const("ORDER_FILLING_RETURN", 2)

    def _unique(*vals: Any) -> tuple[int, ...]:
        """Deduplicate while preserving order (micro-optimized for hot path)."""
        out: list[int] = []
        seen: set[int] = set()
        for val in vals:
            try:
                iv = int(val)
            except Exception:
                continue
            if iv not in seen:
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

    _BIT_FOK = 1
    _BIT_IOC = 2
    _BIT_RETURN = 4

    if isinstance(pref, int) and pref > 2:
        if pref & _BIT_IOC:
            return _unique(ioc, fok, ret)
        if pref & _BIT_FOK:
            return _unique(fok, ioc, ret)
        if pref & _BIT_RETURN:
            return _unique(ret, ioc, fok)

    return _unique(*direct)


# =============================================================================
# Private helpers (zero duplication, defensive, documented)
# =============================================================================


def _merge_comment_into_err_text(existing: str, new_comment: str) -> str:
    """
    Merge broker comment into accumulated error text without redundancy.

    Treats "(1, 'Success')" as empty. Used after every order_send to capture
    broker diagnostics while keeping logs concise.
    """
    comment = str(new_comment or "").strip()
    if not comment:
        return existing
    existing_s = str(existing or "")
    if not existing_s or existing_s == "(1, 'Success')":
        return comment
    if comment.lower() in existing_s.lower():
        return existing_s
    return f"{existing_s} | comment={comment}"


def _sltp_retry_delay_ms(attempt: int) -> float:
    """
    SL/TP retry back-off (ms): 1→10, 2→20, 3+→40.

    Extracted for readability and single-point tuning. Never blocks the
    critical MT5_LOCK section.
    """
    if attempt == 1:
        return 10.0
    if attempt == 2:
        return 20.0
    return 40.0


# =============================================================================
# Executor
# =============================================================================


class OrderExecutor:
    """
    Ultra-fast MT5 market order executor with robust SL/TP attachment.

    Guarantees:
    - Zero sleep while holding MT5_LOCK (critical for HFT)
    - SL/TP re-attached using CURRENT tick + broker stops_level/freeze_level
    - Ticket resolution works in both netting and hedging modes
    - Automatic recovery probe after transient transport errors
    - Dry-run mode for strategy backtesting without broker risk
    """

    def __init__(
        self,
        *,
        normalize_volume_fn: Optional[Callable[[float, Any], float]] = None,
        sanitize_stops_fn: Optional[
            Callable[[str, float, float, float], tuple[float, float]]
        ] = None,
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

        # Fail-safe: close position if SL/TP attachment ultimately fails
        self.close_on_sltp_fail: bool = (
            True if close_on_sltp_fail is None else bool(close_on_sltp_fail)
        )

        self._meta: dict[str, _SymbolMeta] = {}

        # SLTP tuning (tuned for <1 s total latency even on slow brokers)
        self._sltp_poll_deadline_ms = 650
        self._sltp_poll_sleep_ms_seq = (5, 10, 20, 40, 80, 80, 80)
        self._sltp_send_retries = 4
        self._transport_repair_cooldown_sec = 0.35
        self._last_transport_repair_ts = 0.0

    # -------------------- Public API (SIGNATURES UNCHANGED) --------------------

    def send_market_order(
        self,
        request: OrderRequest,
        *,
        max_attempts: int = 2,
        telemetry_hooks: Optional[dict[str, Callable[..., Any]]] = None,
    ) -> OrderResult:
        """
        Send a market order with automatic SL/TP attachment and recovery.

        This is the primary hot-path entry point. All internal methods are
        optimized to minimize time spent holding the global MT5 lock.
        """
        hooks = telemetry_hooks or {}

        side = str(request.signal)
        if side not in ("Buy", "Sell"):
            log_err.error(
                "ORDER_BAD_SIDE | order_id=%s symbol=%s signal=%s",
                request.order_id,
                request.symbol,
                request.signal,
            )
            return self._fail(request, time.time(), "bad_side", 0)

        if self.auto_ensure_mt5:
            try:
                ensure_mt5()
            except Exception as exc:
                log_err.error(
                    "MT5_NOT_READY | order_id=%s err=%s",
                    request.order_id,
                    exc,
                )
                return self._fail(request, time.time(), "mt5_not_ready", 0)

        # Pre-compute once outside retry loop (performance)
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

        effective_attempts = max(2, int(max_attempts))

        for attempt in range(1, effective_attempts + 1):
            try:
                tick, tick_err_text = self._call_mt5_with_last_error(
                    "symbol_info_tick", request.symbol
                )

                if tick is None:
                    last_reason = "tick_none"
                    last_err_text = tick_err_text
                    log_err.error(
                        "ORDER_TICK_NONE | order_id=%s attempt=%d symbol=%s err=%s",
                        request.order_id,
                        attempt,
                        request.symbol,
                        tick_err_text,
                    )
                    self._maybe_repair_transport(
                        last_reason, last_retcode, last_err_text
                    )
                    continue

                req_price = float(tick.ask) if side == "Buy" else float(tick.bid)
                if req_price <= 0.0:
                    last_reason = "bad_req_price"
                    log_err.error(
                        "ORDER_BAD_REQ_PRICE | order_id=%s attempt=%d symbol=%s price=%.5f",
                        request.order_id,
                        attempt,
                        request.symbol,
                        req_price,
                    )
                    self._maybe_repair_transport(
                        last_reason, last_retcode, last_err_text
                    )
                    continue

                if self.normalize_price_fn:
                    try:
                        req_price = float(self.normalize_price_fn(req_price))
                    except Exception:
                        pass  # preserve raw price on callback failure

                sl_s, tp_s = self._sanitize_stops(
                    side, req_price, planned_sl, planned_tp
                )
                sent_ts = time.time()

                # Dry-run simulation (zero broker risk, full telemetry)
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
                    self._maybe_repair_transport(
                        last_reason, last_retcode, last_err_text
                    )
                    continue

                last_retcode = int(getattr(result, "retcode", 0) or 0)
                last_err_text = _merge_comment_into_err_text(
                    last_err_text,
                    str(getattr(result, "comment", "") or ""),
                )

                # Broker rejected stops — retry market order without stops, then attach
                if last_retcode in (_RC.invalid_stops, _RC.invalid_price) and (
                    planned_sl > 0.0 or planned_tp > 0.0
                ):
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
                        self._maybe_repair_transport(
                            last_reason, last_retcode, last_err_text
                        )
                        continue
                    result = result2
                    last_retcode = int(getattr(result2, "retcode", 0) or 0)
                    last_err_text = _merge_comment_into_err_text(
                        last_err_text,
                        str(getattr(result2, "comment", "") or ""),
                    )

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
                            self._fail_safe_close(
                                request.symbol, side, exec_vol, int(request.magic)
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
                    break  # hard failure — retrying will not help

                if self._is_retryable_retcode(last_retcode, last_err_text):
                    last_reason = f"retry_retcode_{last_retcode}"
                    self._maybe_repair_transport(
                        last_reason, last_retcode, last_err_text
                    )
                    continue

                last_reason = f"fail_retcode_{last_retcode}"
                break

            except Exception as exc:
                last_reason = f"exception_{type(exc).__name__}"
                last_err_text = str(exc)
                self._maybe_repair_transport(last_reason, last_retcode, last_err_text)
                log_err.error(
                    "ORDER_EXCEPTION | order_id=%s attempt=%d err=%s tb=%s",
                    request.order_id,
                    attempt,
                    exc,
                    traceback.format_exc(),
                )
                continue

        # Post-loop recovery probe (best-effort after transport ambiguity)
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

    # -------------------- Small helpers (inlined where beneficial) --------------------

    @staticmethod
    def _safe_hook(hooks: dict[str, Callable[..., Any]], name: str, *args: Any) -> None:
        """Invoke optional telemetry hook without ever raising."""
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
        """Normalize MT5 call arguments for cross-version / wrapper compatibility."""
        call_args = tuple(args)
        call_kwargs = dict(kwargs)
        name = str(method_name)

        if name in ("order_send", "order_check"):
            if (
                len(call_args) == 1
                and not call_kwargs
                and isinstance(call_args[0], dict)
            ):
                return (call_args[0],), {}
            if (
                not call_args
                and set(call_kwargs) == {"request"}
                and isinstance(call_kwargs["request"], dict)
            ):
                return (call_kwargs["request"],), {}

        if (
            name == "positions_get"
            and len(call_args) == 1
            and not call_kwargs
            and isinstance(call_args[0], str)
        ):
            return (), {"symbol": call_args[0]}

        return call_args, call_kwargs

    @classmethod
    def _invoke_mt5(
        cls,
        method_name: str,
        *args: Any,
        capture_last_error: bool = False,
        **kwargs: Any,
    ) -> Any | tuple[Any, str]:
        """
        Single source of truth for all MT5 calls (eliminates duplication).

        Always acquires MT5_LOCK. Optionally captures mt5.last_error() while
        still inside the lock to avoid race conditions on error state.
        """
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            raise AttributeError(f"mt5.{method_name} not callable")

        call_args, call_kwargs = cls._normalize_mt5_call(
            str(method_name), tuple(args), dict(kwargs)
        )

        with MT5_LOCK:
            if (
                str(method_name) in ("order_send", "order_check")
                and len(call_args) == 1
                and not call_kwargs
            ):
                result = fn(call_args[0])
            else:
                result = fn(*call_args, **call_kwargs)

            last_error = ""
            if capture_last_error:
                try:
                    last_error = str(mt5.last_error())
                except Exception:
                    last_error = ""

        if capture_last_error:
            return result, last_error
        return result

    @classmethod
    def _call_mt5(cls, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """MT5 call without last_error capture (used for read-only queries)."""
        return cls._invoke_mt5(method_name, *args, capture_last_error=False, **kwargs)

    @classmethod
    def _call_mt5_with_last_error(
        cls, method_name: str, *args: Any, **kwargs: Any
    ) -> tuple[Any, str]:
        """MT5 call that also returns last_error() captured under lock."""
        res = cls._invoke_mt5(
            method_name, *args, capture_last_error=True, **kwargs
        )
        # mypy / runtime safety: _invoke_mt5 guarantees tuple when capture=True
        return cast(tuple[Any, str], res)

    def _is_retryable_retcode(self, retcode: int, last_error_text: str = "") -> bool:
        """Return True for transient errors that justify an immediate retry."""
        if int(retcode) in _RETRYABLE_RETCODES:
            return True
        msg = str(last_error_text or "").lower()
        return (
            "timeout" in msg
            or "ipc" in msg
            or "-10005" in msg
            or "no connection" in msg
            or "connection" in msg
        )

    def _maybe_repair_transport(
        self, reason: str, retcode: int, last_error_text: str
    ) -> None:
        """Best-effort MT5 reconnection with cooldown (non-blocking)."""
        combined = str(last_error_text or "") + " " + str(reason or "")
        if not self._is_retryable_retcode(int(retcode or 0), combined):
            return
        now = time.monotonic()
        if (now - self._last_transport_repair_ts) < float(
            self._transport_repair_cooldown_sec
        ):
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
        Best-effort recovery after ambiguous transport failure.

        Scans recent deal history for a matching fill that arrived despite
        the error. Returns (fill_ts, price, volume, deal_ticket) or None.
        """
        try:
            now_dt = datetime.now()
            start_dt = datetime.fromtimestamp(max(0.0, float(sent_ts) - 5.0))
            deals = self._call_mt5("history_deals_get", start_dt, now_dt) or []
        except Exception as exc:
            log_err.error(
                "PROBE_RECOVERED_FILL_FAILED | symbol=%s side=%s err=%s",
                symbol,
                side,
                exc,
            )
            return None

        if not deals:
            return None

        want_type = mt5.ORDER_TYPE_BUY if str(side) == "Buy" else mt5.ORDER_TYPE_SELL
        entry_in = int(getattr(mt5, "DEAL_ENTRY_IN", 0) or 0)
        best: Optional[tuple[float, float, float, int]] = None
        best_score: Optional[tuple[float, float]] = None

        for d in deals:
            try:
                if (
                    str(getattr(d, "symbol", "")) != str(symbol)
                    or int(getattr(d, "magic", 0) or 0) != int(magic)
                    or int(getattr(d, "entry", 0) or 0) != entry_in
                    or int(getattr(d, "type", -1) or -1) != int(want_type)
                ):
                    continue

                d_time = float(getattr(d, "time", 0.0) or 0.0)
                if d_time <= 0.0 or d_time < (float(sent_ts) - 5.0):
                    continue

                d_vol = float(getattr(d, "volume", 0.0) or 0.0)
                if d_vol <= 0.0:
                    continue

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
            except Exception as exc:
                log_err.error(
                    "PROBE_DEAL_PARSE_ERROR | symbol=%s err=%s",
                    symbol,
                    exc,
                )
                continue

        return best

    def _fail(
        self, request: OrderRequest, sent_ts: float, reason: str, retcode: int
    ) -> OrderResult:
        """Construct a failed OrderResult (used on all terminal error paths)."""
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
    # Symbol meta cache (TTL + defensive refresh)
    # =============================================================================

    def _get_symbol_meta(self, symbol: str) -> Optional[_SymbolMeta]:
        """Return cached or freshly fetched symbol meta (thread-safe via MT5_LOCK)."""
        now = time.time()
        m = self._meta.get(symbol)
        if m and (now - float(m.ts) <= float(self.symbol_cache_ttl)):
            return m

        try:
            info = self._call_mt5("symbol_info", symbol)
            if info is None:
                return None

            try:
                self._call_mt5("symbol_select", symbol, True)
            except Exception as exc:
                log_err.error(
                    "SYMBOL_SELECT_FAILED | symbol=%s err=%s",
                    symbol,
                    exc,
                )

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

            try:
                v_decimals = max(
                    0,
                    min(8, -Decimal(str(v_step)).normalize().as_tuple().exponent),
                )
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
            log_err.error(
                "SYMBOL_META_FAILED | symbol=%s err=%s tb=%s",
                symbol,
                exc,
                traceback.format_exc(),
            )
            return None

    def _normalize_volume(self, vol: float, meta: _SymbolMeta) -> float:
        """Normalize volume to broker step/min/max with exact decimal handling."""
        if self.normalize_volume_fn:
            try:
                return float(self.normalize_volume_fn(vol, meta))
            except Exception:
                pass

        vmin = float(meta.v_min)
        vmax = float(meta.v_max)
        step = float(meta.v_step)
        vol = max(vmin, min(vmax, float(vol)))
        k = math.floor(vol / step)
        v = round(k * step, int(meta.v_decimals))
        return float(v if v >= vmin else vmin)

    def _sanitize_stops(
        self, side: str, req_price: float, sl: float, tp: float
    ) -> tuple[float, float]:
        """Apply optional external stop sanitizer (fallback to raw values)."""
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
        Rebase absolute SL/TP levels after market slippage while preserving risk distance.

        Prevents silent risk widening when the fill price differs from the
        requested price.
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
        """Send market deal trying all supported filling modes in preference order."""
        req: dict[str, Any] = {
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
        last_result: Any = None
        last_err_text = ""

        for idx, fill_mode in enumerate(fill_candidates):
            req["type_filling"] = int(fill_mode)
            result, last_err_text = OrderExecutor._call_mt5_with_last_error(
                "order_send", req
            )
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
    # SLTP ROBUST ATTACHMENT (core reliability logic)
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
        """Ensure SL/TP are correctly attached to the live position (retry + sanitize)."""
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

        if resolved == 0:  # no position (netting offset / instant close)
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
                "SLTP_SANITIZE_FAILED | ticket=%s symbol=%s side=%s "
                "desired_sl=%.5f desired_tp=%.5f ctx=%s",
                int(resolved),
                symbol,
                side,
                float(planned_sl),
                float(planned_tp),
                ctx,
            )
            return False

        req: dict[str, Any] = {
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

            if r is not None and last_rc in (
                _RC.done,
                _RC.done_partial,
                _RC.no_changes,
            ):
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

            time.sleep(_sltp_retry_delay_ms(i) / 1000.0)

        log_err.error(
            "SLTP_SET_FAILED | ticket=%s symbol=%s side=%s rc=%s "
            "last_error=%s sl=%.5f tp=%.5f stops=%s freeze=%s",
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
        """Return (sl, tp) for a position ticket (defensive, never raises)."""
        try:
            p = self._call_mt5("positions_get", ticket=int(ticket)) or []
            if not p:
                return 0.0, 0.0
            px = p[0]
            return (
                float(getattr(px, "sl", 0.0) or 0.0),
                float(getattr(px, "tp", 0.0) or 0.0),
            )
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _sltp_is_good(
        cur_sl: float,
        cur_tp: float,
        planned_sl: float,
        planned_tp: float,
        point: float,
    ) -> bool:
        """Return True if current SL/TP are within one point of planned levels."""
        eps = max(float(point), 1e-10)
        ok_sl = (planned_sl <= 0.0) or (
            cur_sl > 0.0 and abs(cur_sl - planned_sl) <= eps
        )
        ok_tp = (planned_tp <= 0.0) or (
            cur_tp > 0.0 and abs(cur_tp - planned_tp) <= eps
        )
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
        """Clamp desired SL/TP to current tick + broker stops/freeze levels."""
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
        min_dist_abs = (float(min_level) * point) + (
            base_safety * float(expand_dist_mult)
        )

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
        """
        Resolve live position ticket with bounded polling (netting + hedging safe).

        Returns ticket, 0 (no position), or None (timeout).
        """
        want_type = mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL

        def _pos_exists_by_ticket(tk: int) -> bool:
            if tk <= 0:
                return False
            try:
                p = self._call_mt5("positions_get", ticket=int(tk)) or []
                return bool(p and str(getattr(p[0], "symbol", "")) == str(symbol))
            except Exception:
                return False

        # Fast path — direct ticket lookup before any polling
        for tk in [int(ticket_hint or 0)] + [int(x or 0) for x in candidate_tickets]:
            if tk > 0 and _pos_exists_by_ticket(tk):
                return tk

        deadline = time.perf_counter() + (float(deadline_ms) / 1000.0)
        best_ticket = 0
        best_score: Optional[tuple[int, int]] = None
        saw_any = False
        sleep_seq = list(self._sltp_poll_sleep_ms_seq)
        sleep_idx = 0

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
                    pt = (
                        int(tmsc)
                        if tmsc is not None
                        else int(float(getattr(p, "time", 0.0) or 0.0) * 1000.0)
                    )
                    magic_penalty = 0 if pmagic == int(magic) else 1
                    score = (magic_penalty, -pt)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_ticket = tk
                except Exception:
                    continue

            if best_ticket > 0:
                return int(best_ticket)

            delay = sleep_seq[sleep_idx] if sleep_idx < len(sleep_seq) else 80
            sleep_idx += 1
            time.sleep(delay / 1000.0)

        if saw_any and best_ticket == 0:
            log_err.error(
                "RESOLVE_TICKET_TIMEOUT | symbol=%s side=%s magic=%s deadline_ms=%d",
                symbol,
                side,
                int(magic),
                int(deadline_ms),
            )

        return 0 if not saw_any else None

    # =============================================================================
    # Fail-safe close (last-resort risk containment)
    # =============================================================================

    @staticmethod
    def _fail_safe_close(symbol: str, side: str, volume: float, magic: int) -> None:
        """Close the position if SL/TP attachment failed (defensive containment)."""
        try:
            want_type = (
                mt5.POSITION_TYPE_BUY if side == "Buy" else mt5.POSITION_TYPE_SELL
            )
            positions = OrderExecutor._call_mt5("positions_get", symbol=symbol) or []

            for p in positions:
                try:
                    if (
                        int(getattr(p, "type", -1) or -1) != int(want_type)
                        or int(getattr(p, "magic", 0) or 0) != int(magic)
                    ):
                        continue

                    ticket = int(getattr(p, "ticket", 0) or 0)
                    pv = float(getattr(p, "volume", 0.0) or 0.0)
                    if ticket <= 0 or pv <= 0.0:
                        continue

                    tick = OrderExecutor._call_mt5("symbol_info_tick", symbol)
                    if tick is None:
                        log_err.error(
                            "FAILSAFE_CLOSE_TICK_NONE | ticket=%s symbol=%s",
                            ticket,
                            symbol,
                        )
                        continue

                    price = float(tick.bid) if side == "Buy" else float(tick.ask)
                    close_type = (
                        mt5.ORDER_TYPE_SELL if side == "Buy" else mt5.ORDER_TYPE_BUY
                    )

                    req: dict[str, Any] = {
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
                except Exception as exc:
                    log_err.error(
                        "FAILSAFE_CLOSE_POSITION_ERROR | symbol=%s err=%s",
                        symbol,
                        exc,
                    )
                    continue
        except Exception:
            return


__all__ = [
    "OrderRequest",
    "OrderResult",
    "OrderExecutor",
]