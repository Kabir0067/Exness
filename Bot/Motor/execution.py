"""
Execution workers and MT5 order-dispatch helpers.

Processes execution intents, integrates with MT5, and records
order outcomes for the engine runtime.
"""

from __future__ import annotations

import builtins
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Imports
# =============================================================================
import MetaTrader5 as mt5

from ExnessAPI.functions import close_all_position
from ExnessAPI.order_execution import OrderExecutor, OrderRequest
from ExnessAPI.order_execution import OrderResult as ExecOrderResult
from ExnessAPI.order_execution import log_health as exec_health_log
from mt5_client import MT5_LOCK, mt5_async_call

from .models import (
    AssetCandidate,
    ExecutionResult,
    OrderIntent,
    log_err,
    log_health,
    signal_age_sec,
    tf_seconds,
)

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine

# =============================================================================
# Classes
# =============================================================================


def _signal_max_age_sec(cfg: Any, *, source: str, timeframe: str) -> float:
    base_cfg = float(getattr(cfg, "signal_max_age_sec", 75.0) or 75.0)
    bridge_cfg = float(
        getattr(cfg, "ml_bridge_max_age_sec", min(base_cfg, 30.0)) or min(base_cfg, 30.0)
    )
    tf_sec = float(tf_seconds(timeframe) or 0.0)
    if tf_sec > 0.0:
        base_cfg = max(base_cfg, tf_sec * 1.25)
        bridge_cfg = min(base_cfg, max(bridge_cfg, tf_sec * 0.50))
    return bridge_cfg if str(source or "").strip().lower() == "ml_bridge" else base_cfg


def _signal_stale_reason(
    cfg: Any,
    *,
    source: str,
    bar_key: str,
    timeframe: str,
    created_ts: float,
    now_ts: float,
) -> str:
    age_sec = signal_age_sec(bar_key, now_ts=now_ts, created_ts=created_ts)
    if age_sec is None:
        return ""
    max_age_sec = _signal_max_age_sec(cfg, source=source, timeframe=timeframe)
    if max_age_sec > 0.0 and age_sec > max_age_sec:
        return (
            f"stale_signal:age={age_sec:.1f}s>max={max_age_sec:.1f}s"
            f"|tf={timeframe or '-'}|source={source or '-'}"
        )
    return ""


class ExecutionWorker(threading.Thread):
    def __init__(
        self,
        order_queue: "queue.Queue[OrderIntent]",
        result_queue: "queue.Queue[ExecutionResult]",
        dry_run: bool,
        order_notify_cb: Optional[
            Callable[[OrderIntent, ExecutionResult], None]
        ] = None,
        result_fallback_cb: Optional[Callable[[ExecutionResult], None]] = None,
        worker_label: str = "exec",
    ) -> None:
        super().__init__(
            daemon=True, name=f"ExecutionWorker-{str(worker_label or 'exec')}"
        )
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.dry_run = bool(dry_run)
        self.order_notify_cb = order_notify_cb
        self.result_fallback_cb = result_fallback_cb
        self.worker_label = str(worker_label or "exec")
        # IMPORTANT: do not shadow threading.Thread._stop
        self._stop_evt = threading.Event()
        self._executor: Optional[Any] = None

    def stop(self) -> None:
        self._stop_evt.set()

    def _get_hook(
        self, intent: OrderIntent
    ) -> Optional[Callable[[ExecOrderResult], None]]:
        rm = intent.risk_manager
        if rm and hasattr(rm, "execution_hook"):
            fn = getattr(rm, "execution_hook", None)
            if callable(fn):
                return fn  # type: ignore[return-value]
        return None

    def _get_telemetry_hooks(
        self, intent: OrderIntent
    ) -> Optional[dict[str, Callable[..., None]]]:
        rm = intent.risk_manager
        if not rm:
            return None

        hooks: dict[str, Callable[..., None]] = {}
        rec_metrics = getattr(rm, "record_execution_metrics", None)
        if callable(rec_metrics):
            hooks["record_execution_metrics"] = rec_metrics

        rec_failure = getattr(rm, "record_execution_failure", None)
        if callable(rec_failure):
            hooks["record_execution_failure"] = rec_failure

        return hooks or None

    def _build_executor(self) -> Any:
        if self._executor is None:
            self._executor = OrderExecutor(
                auto_ensure_mt5=(not self.dry_run),
                symbol_cache_ttl=30.0,
            )
        return self._executor

    def _resolve_position_ticket(
        self,
        intent: OrderIntent,
        result: ExecOrderResult,
        sent_ts: float,
    ) -> int:
        """Resolve a broker position ticket when the immediate response omits it."""
        try:
            position_ticket = int(getattr(result, "position_ticket", 0) or 0)
        except Exception:
            position_ticket = 0
        if position_ticket > 0:
            return position_ticket

        try:
            deal_ticket = int(getattr(result, "deal_ticket", 0) or 0)
        except Exception:
            deal_ticket = 0

        symbol = str(getattr(intent, "symbol", "") or "")
        if not symbol:
            return 0

        try:
            magic = int(getattr(intent.cfg, "magic", 777001) or 777001)
        except Exception:
            magic = 777001

        want_type = (
            mt5.POSITION_TYPE_BUY
            if str(getattr(intent, "signal", "") or "").lower().startswith("b")
            else mt5.POSITION_TYPE_SELL
        )

        for attempt in range(4):
            if attempt > 0:
                time.sleep(min(0.25 * float(attempt), 0.75))

            if deal_ticket > 0:
                try:
                    start_dt = datetime.fromtimestamp(max(0.0, float(sent_ts) - 10.0))
                    end_dt = datetime.fromtimestamp(time.time() + 2.0)
                    with MT5_LOCK:
                        deals = mt5.history_deals_get(start_dt, end_dt) or []
                    for deal in deals:
                        try:
                            if int(getattr(deal, "ticket", 0) or 0) != deal_ticket:
                                continue
                            position_id = int(getattr(deal, "position_id", 0) or 0)
                            if position_id > 0:
                                return position_id
                        except Exception:
                            continue
                except Exception:
                    pass

            try:
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []
            except Exception:
                positions = []

            best_ticket = 0
            best_time = 0.0
            for position in positions:
                try:
                    if int(getattr(position, "magic", 0) or 0) != int(magic):
                        continue
                    if int(getattr(position, "type", -1) or -1) != int(want_type):
                        continue
                    pos_time = float(getattr(position, "time", 0.0) or 0.0)
                    if pos_time + 5.0 < float(sent_ts):
                        continue
                    ticket = int(getattr(position, "ticket", 0) or 0)
                    if ticket > 0 and pos_time >= best_time:
                        best_time = pos_time
                        best_ticket = ticket
                except Exception:
                    continue

            if best_ticket > 0:
                return int(best_ticket)

        return 0

    def _process(self, intent: OrderIntent) -> ExecutionResult:
        sent = time.time()
        try:
            dispatch_lot = float(intent.lot)
            rm = intent.risk_manager
            gate_reason = "ok"
            stale_reason = _signal_stale_reason(
                intent.cfg,
                source=str(getattr(intent, "source", "") or ""),
                bar_key=str(getattr(intent, "bar_key", "") or ""),
                timeframe=str(getattr(intent, "timeframe", "") or ""),
                created_ts=float(getattr(intent, "created_ts", 0.0) or 0.0),
                now_ts=sent,
            )
            if stale_reason:
                if rm and hasattr(rm, "record_execution_failure"):
                    try:
                        rm.record_execution_failure(
                            str(intent.order_id),
                            float(intent.enqueue_time),
                            sent,
                            stale_reason,
                        )
                    except Exception as exc:  # DEFENSIVE CODE
                        log_err.error(
                            "EXEC_STALE_RECORD_FAILURE_ERROR | order_id=%s err=%s",
                            intent.order_id,
                            exc,
                        )
                return ExecutionResult(
                    order_id=str(intent.order_id),
                    signal_id=str(intent.signal_id),
                    ok=False,
                    reason=stale_reason,
                    sent_ts=sent,
                    fill_ts=sent,
                    req_price=float(intent.price),
                    exec_price=float(intent.price),
                    volume=0.0,
                    slippage=0.0,
                    retcode=-4,
                )

            if rm and hasattr(rm, "pre_order_gate"):
                gate_fn = getattr(rm, "pre_order_gate", None)
                if callable(gate_fn):
                    try:
                        allowed, gated_lot, gate_reason = gate_fn(
                            side=str(intent.signal),
                            confidence=float(intent.confidence),
                            lot=float(dispatch_lot),
                            entry_price=float(intent.price),
                            sl=float(intent.sl),
                            tp=float(intent.tp),
                            signal_id=str(intent.signal_id),
                            stage="dispatch",
                            base_lot=float(
                                getattr(intent, "base_lot", intent.lot) or intent.lot
                            ),
                            phase_snapshot=str(
                                getattr(intent, "phase_snapshot", "") or ""
                            ),
                        )
                    except TypeError:
                        allowed, gated_lot, gate_reason = gate_fn(  # type: ignore[misc]
                            side=str(intent.signal),
                            confidence=float(intent.confidence),
                            lot=float(dispatch_lot),
                            entry_price=float(intent.price),
                            sl=float(intent.sl),
                            tp=float(intent.tp),
                        )
                    if not allowed:
                        reason = f"dispatch_gate:{gate_reason}"
                        if rm and hasattr(rm, "record_execution_failure"):
                            try:
                                rm.record_execution_failure(
                                    str(intent.order_id),
                                    float(intent.enqueue_time),
                                    sent,
                                    reason,
                                )
                            except Exception as exc:  # DEFENSIVE CODE
                                log_err.error(
                                    "EXEC_GATE_RECORD_FAILURE_ERROR | order_id=%s err=%s",
                                    intent.order_id,
                                    exc,
                                )
                        return ExecutionResult(
                            order_id=str(intent.order_id),
                            signal_id=str(intent.signal_id),
                            ok=False,
                            reason=reason,
                            sent_ts=sent,
                            fill_ts=sent,
                            req_price=float(intent.price),
                            exec_price=float(intent.price),
                            volume=0.0,
                            slippage=0.0,
                            retcode=-2,
                        )
                    dispatch_lot = float(gated_lot or dispatch_lot)

                    # ─── SECTION 10 — AUTO-DEGRADE multiplier ────────────
                    # Apply the current risk-governor multiplier (derived
                    # from PSI/KS drift + CUSUM edge-monitor) to the lot
                    # size. This is the institutional single-application
                    # site: the governor is observational, execution
                    # enforces. Defensive: any failure leaves dispatch_lot
                    # untouched.
                    try:
                        from core.stability_monitor import get_default_governor

                        _gov_state = get_default_governor().snapshot()
                        _gov_mult = float(
                            getattr(_gov_state, "last_multiplier", 1.0) or 1.0
                        )
                        # Clamp to [0, 1] — governor can only REDUCE size.
                        _gov_mult = max(0.0, min(1.0, _gov_mult))
                        if _gov_mult < 1.0:
                            dispatch_lot = float(dispatch_lot) * _gov_mult
                            log_health.info(
                                "AUTO_DEGRADE_APPLIED | order_id=%s mult=%.4f"
                                " reason=%s lot_after=%.4f",
                                intent.order_id,
                                _gov_mult,
                                str(
                                    getattr(_gov_state, "last_reason", "")
                                    or ""
                                ),
                                dispatch_lot,
                            )
                    except Exception as _gov_exc:
                        log_err.error(
                            "AUTO_DEGRADE_ERROR | order_id=%s err=%s",
                            intent.order_id,
                            _gov_exc,
                        )

                    if dispatch_lot <= 0.0:
                        return ExecutionResult(
                            order_id=str(intent.order_id),
                            signal_id=str(intent.signal_id),
                            ok=False,
                            reason="dispatch_gate:lot_nonpositive",
                            sent_ts=sent,
                            fill_ts=sent,
                            req_price=float(intent.price),
                            exec_price=float(intent.price),
                            volume=0.0,
                            slippage=0.0,
                            retcode=-2,
                        )

            # ─── Idempotency gate (WAL) ─────────────────────────────────
            # Refuse to send if this idempotency_key is already in-flight or
            # already confirmed in the journal. Embed the key's tail into
            # the broker comment so that reconciliation after a crash can
            # match history deals back to this logical order.
            _idem_journal = None
            _idem_key = str(
                getattr(intent, "idempotency_key", "") or intent.order_id
            )
            _comment_base = str(
                getattr(intent.cfg, "comment", "portfolio") or "portfolio"
            )
            _order_comment = _comment_base
            try:
                from core.idempotency import (
                    STATUS_PENDING,
                    get_default_journal,
                    idem_key_to_comment,
                )

                _idem_journal = get_default_journal()
                _existing = _idem_journal.get(_idem_key)
                _queue_owned_pending = bool(
                    _existing is not None and _existing.status == STATUS_PENDING
                )
                if not _queue_owned_pending:
                    _ok_to_send, _idem_reason = _idem_journal.should_send(_idem_key)
                    if not _ok_to_send:
                        return ExecutionResult(
                            order_id=str(intent.order_id),
                            signal_id=str(intent.signal_id),
                            ok=False,
                            reason=f"idempotency_refuse:{_idem_reason}",
                            sent_ts=sent,
                            fill_ts=sent,
                            req_price=float(intent.price),
                            exec_price=float(intent.price),
                            volume=0.0,
                            slippage=0.0,
                            retcode=-3,
                        )
                _order_comment = idem_key_to_comment(_idem_key, base=_comment_base)
            except Exception as _idem_exc:
                log_err.error(
                    "EXEC_IDEMPOTENCY_WARN | order_id=%s err=%s",
                    intent.order_id,
                    _idem_exc,
                )

            req = OrderRequest(
                symbol=str(intent.symbol),
                signal=str(intent.signal),
                lot=float(dispatch_lot),
                sl=float(intent.sl),
                tp=float(intent.tp),
                price=float(intent.price),
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                enqueue_time=float(intent.enqueue_time),
                magic=int(getattr(intent.cfg, "magic", 777001) or 777001),
                comment=str(_order_comment),
                cfg=intent.cfg,  # Pass Config for Dry Run
            )
            ex = self._build_executor()
            hooks = self._get_telemetry_hooks(intent)
            r: ExecOrderResult = ex.send_market_order(req, telemetry_hooks=hooks)

            # ─── Record outcome in WAL ──────────────────────────────────
            if _idem_journal is not None:
                try:
                    _retcode = int(getattr(r, "retcode", 0) or 0)
                    _ok = bool(getattr(r, "ok", False))
                    _pos_tk = self._resolve_position_ticket(intent, r, sent) if _ok else 0
                    _deal_tk = int(getattr(r, "deal_ticket", 0) or 0)
                    if _ok:
                        _idem_journal.record_sent(
                            _idem_key,
                            retcode=_retcode,
                            order_ticket=int(getattr(r, "order_ticket", 0) or 0),
                            deal_ticket=int(_deal_tk),
                            position_ticket=int(_pos_tk),
                            reason="ok",
                        )
                        if _pos_tk > 0 or _deal_tk > 0:
                            _idem_journal.record_confirmed(
                                _idem_key,
                                position_ticket=int(_pos_tk),
                                deal_ticket=int(_deal_tk),
                                reason=(
                                    "position_ticket_resolved"
                                    if _pos_tk > 0
                                    else "deal_ticket_confirmed"
                                ),
                            )
                    else:
                        _idem_journal.record_failed(
                            _idem_key,
                            reason=str(getattr(r, "reason", "") or "broker_fail"),
                            retcode=_retcode,
                        )
                except Exception as _idem_rec_exc:  # DEFENSIVE
                    log_err.error(
                        "EXEC_IDEMPOTENCY_RECORD_FAIL | order_id=%s err=%s",
                        intent.order_id,
                        _idem_rec_exc,
                    )

            filled = float(getattr(r, "fill_ts", 0.0) or time.time())
            ok = bool(getattr(r, "ok", False))
            reason = str(getattr(r, "reason", "") or "")
            retcode = int(getattr(r, "retcode", 0) or 0)
            position_ticket = self._resolve_position_ticket(intent, r, sent) if ok else 0

            return ExecutionResult(
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                ok=ok,
                reason=reason,
                sent_ts=float(getattr(r, "sent_ts", sent) or sent),
                fill_ts=filled,
                req_price=float(getattr(r, "req_price", intent.price) or intent.price),
                exec_price=float(
                    getattr(r, "exec_price", intent.price) or intent.price
                ),
                volume=float(getattr(r, "volume", intent.lot) or intent.lot),
                slippage=float(getattr(r, "slippage", 0.0) or 0.0),
                retcode=retcode,
                order_ticket=int(getattr(r, "order_ticket", 0) or 0),
                deal_ticket=int(getattr(r, "deal_ticket", 0) or 0),
                position_ticket=int(position_ticket),
            )
        except Exception as exc:
            log_err.error(
                "execution worker error: %s | tb=%s", exc, traceback.format_exc()
            )
            return ExecutionResult(
                order_id=str(intent.order_id),
                signal_id=str(intent.signal_id),
                ok=False,
                reason=f"exception:{exc}",
                sent_ts=sent,
                fill_ts=sent,
                req_price=float(intent.price),
                exec_price=float(intent.price),
                volume=float(intent.lot),
                slippage=0.0,
                retcode=-1,
                order_ticket=0,
                deal_ticket=0,
                position_ticket=0,
            )

    def _fallback_risk_update(self, intent: OrderIntent, res: ExecutionResult) -> None:
        """If result_queue is unavailable/full, ensure RM still sees the result."""
        rm = intent.risk_manager
        if rm and hasattr(rm, "on_execution_result"):
            try:
                rm.on_execution_result(res)
            except Exception as exc:  # DEFENSIVE CODE
                log_err.error(
                    "EXEC_FALLBACK_RISK_UPDATE_ERROR | order_id=%s err=%s",
                    intent.order_id,
                    exc,
                )

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                intent = self.order_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                res = self._process(intent)

                # Try to publish result to engine
                published = False
                try:
                    self.result_queue.put(res, timeout=0.25)
                    published = True
                except Exception as exc:  # DEFENSIVE CODE
                    published = False
                    log_err.error(
                        "EXEC_RESULT_QUEUE_PUT_FAILED | order_id=%s worker=%s err=%s",
                        res.order_id,
                        self.worker_label,
                        exc,
                    )

                # If not published, resolve directly so pending broker-sync state
                # does not leak and later turn into a false timeout.
                if not published:
                    resolved = False
                    if self.result_fallback_cb:
                        try:
                            self.result_fallback_cb(res)
                            resolved = True
                        except Exception as exc:
                            log_err.error(
                                "execution fallback resolve error: %s | tb=%s",
                                exc,
                                traceback.format_exc(),
                            )
                    if not resolved:
                        self._fallback_risk_update(intent, res)

                # External notify bridge (telegram, etc.)
                if self.order_notify_cb:
                    try:
                        self.order_notify_cb(intent, res)
                    except Exception as exc:  # DEFENSIVE CODE
                        log_err.error(
                            "EXEC_NOTIFY_CB_ERROR | order_id=%s err=%s",
                            res.order_id,
                            exc,
                        )

                # Optional executor health log
                try:
                    exec_health_log.info(
                        "EXEC_HEALTH | worker=%s order_id=%s ok=%s reason=%s",
                        self.worker_label,
                        res.order_id,
                        res.ok,
                        res.reason,
                    )
                except Exception as exc:  # DEFENSIVE CODE
                    log_err.error(
                        "EXEC_HEALTH_LOG_ERROR | worker=%s order_id=%s err=%s",
                        self.worker_label,
                        res.order_id,
                        exc,
                    )

            finally:
                try:
                    self.order_queue.task_done()
                except Exception:
                    pass


# --- Execution manager -----------------------------------------------------
class ExecutionManager:
    """Owns enqueue-time order gating and queue dispatch semantics."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine
        self._account_cache_lock = threading.Lock()
        self._account_cache: Dict[str, float] = {
            "ts": 0.0,
            "equity": 0.0,
            "leverage": 100.0,
        }

    def _cached_account_probe(self) -> Tuple[float, float]:
        now = time.time()
        with self._account_cache_lock:
            cached_ts = float(self._account_cache.get("ts", 0.0) or 0.0)
            cached_equity = float(self._account_cache.get("equity", 0.0) or 0.0)
            cached_leverage = float(self._account_cache.get("leverage", 100.0) or 100.0)
        if cached_ts > 0.0 and (now - cached_ts) <= 1.5 and cached_equity > 0.0:
            return cached_equity, max(1.0, cached_leverage)

        acc_info = mt5_async_call("account_info", timeout=0.25, default=None)
        if acc_info is not None:
            equity = float(getattr(acc_info, "equity", 0.0) or 0.0)
            leverage = float(getattr(acc_info, "leverage", 100.0) or 100.0)
            if equity > 0.0:
                with self._account_cache_lock:
                    self._account_cache["ts"] = float(now)
                    self._account_cache["equity"] = float(equity)
                    self._account_cache["leverage"] = float(max(1.0, leverage))
                return equity, max(1.0, leverage)

        return cached_equity, max(1.0, cached_leverage)

    def enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        e = self._e
        now = time.time()

        if bool(getattr(e, "_manual_stop", False)):
            log_health.info("ENQUEUE_SKIP | asset=%s reason=manual_stop", cand.asset)
            return False, None

        if cand.asset == "XAU":
            risk = e._xau.risk if e._xau else None
            cfg = e._xau_cfg
        else:
            risk = e._btc.risk if e._btc else None
            cfg = e._btc_cfg

        try:
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                fn()
        except Exception as exc:  # DEFENSIVE CODE
            log_err.error(
                "ENQUEUE_PHASE_UPDATE_ERROR | asset=%s err=%s", cand.asset, exc
            )

        try:
            hard_stop_attr = getattr(risk, "requires_hard_stop", False)
            hard_stop_active = bool(
                hard_stop_attr() if callable(hard_stop_attr) else hard_stop_attr
            )
            if hard_stop_active:
                log_health.info("ENQUEUE_SKIP | asset=%s reason=hard_stop", cand.asset)
                return False, None
        except Exception as exc:  # DEFENSIVE CODE
            log_err.error(
                "ENQUEUE_HARD_STOP_CHECK_ERROR | asset=%s err=%s", cand.asset, exc
            )

        phase = e._get_phase(risk)
        if phase == "C":
            log_health.info(
                "PHASE_C_TRADE_PAUSE | asset=%s signal=%s conf=%.2f lot=%.4f sl=%.2f tp=%.2f",
                cand.asset,
                cand.signal,
                cand.confidence,
                cand.lot,
                cand.sl,
                cand.tp,
            )
            return False, None
        last_close_ts = e._last_trade_close_ts.get(cand.asset, 0.0)
        time_since_close = now - last_close_ts
        if time_since_close < e._trade_cooldown_sec and last_close_ts > 0:
            e._cooldown_blocked_count[cand.asset] = (
                e._cooldown_blocked_count.get(cand.asset, 0) + 1
            )
            if e._cooldown_blocked_count[cand.asset] % 10 == 1:
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=trade_cooldown remaining=%.0fs total_blocked=%d",
                    cand.asset,
                    e._trade_cooldown_sec - time_since_close,
                    e._cooldown_blocked_count[cand.asset],
                )
            return False, None

        open_positions = e._asset_open_positions(cand.asset)
        max_positions = e._asset_max_positions(cand.asset)
        if max_positions > 0 and open_positions >= max_positions:
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=max_positions open=%d max=%d",
                cand.asset,
                open_positions,
                max_positions,
            )
            return False, None

        equity, leverage = self._cached_account_probe()

        c_size = float(getattr(cfg.symbol_params, "contract_size", 100.0))
        proposed_margin = (
            float(cand.close if hasattr(cand, "close") else 0)
            * float(cand.lot)
            * c_size
        ) / float(leverage or 1)

        pre_tick = mt5_async_call(
            "symbol_info_tick",
            cand.symbol,
            timeout=0.35,
            default=None,
        )
        pre_info = mt5_async_call(
            "symbol_info",
            cand.symbol,
            timeout=0.5,
            default=None,
        )
        pre_price = float(
            pre_tick.ask
            if (pre_tick is not None and cand.signal == "Buy")
            else (pre_tick.bid if pre_tick is not None else 0.0)
        )
        if proposed_margin <= 0.0 and pre_price > 0.0:
            proposed_margin = (pre_price * float(cand.lot) * c_size) / float(
                leverage or 1
            )
        live_contract_size = float(getattr(pre_info, "trade_contract_size", 0.0) or 0.0)
        if live_contract_size <= 0.0:
            live_contract_size = float(
                getattr(cfg.symbol_params, "contract_size", 1.0) or 1.0
            )
        proposed_risk_amount = max(
            0.0,
            abs(pre_price - float(cand.sl or 0.0))
            * float(cand.lot)
            * live_contract_size,
        )

        allowed, adj_lot, block_reason = e._portfolio_risk.check_before_order(
            asset=cand.asset,
            side=cand.signal,
            equity=equity,
            proposed_lot=float(cand.lot),
            proposed_margin=proposed_margin,
            proposed_risk_amount=proposed_risk_amount,
            asset_exposure_factor=float(
                getattr(cfg, "max_asset_exposure_factor", 0.0) or 0.0
            ),
            asset_risk_factor=float(
                getattr(
                    cfg,
                    "max_asset_risk_per_asset_pct",
                    max(0.0, float(getattr(cfg, "max_risk_per_trade", 0.0) or 0.0))
                    * 2.0,
                )
                or 0.0
            ),
        )
        if not allowed:
            log_health.warning(
                "PORTFOLIO_BLOCK | asset=%s reason=%s", cand.asset, block_reason
            )
            return False, None

        if adj_lot < cand.lot:
            log_health.info(
                "PORTFOLIO_REDUCE | asset=%s lot %.2f -> %.2f (Correlation Guard)",
                cand.asset,
                cand.lot,
                adj_lot,
            )
            lot_override = adj_lot

        last_id, last_sig = e._edge_last_trade.get(cand.asset, ("", "Neutral"))
        if (
            cand.signal in ("Buy", "Sell")
            and last_id == str(cand.signal_id)
            and last_sig == str(cand.signal)
        ):
            last_log = e._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=edge_same_signal_id signal=%s signal_id=%s",
                    cand.asset,
                    cand.signal,
                    cand.signal_id,
                )
                e._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        if e._is_duplicate(
            cand.asset,
            cand.signal_id,
            now,
            max_orders=int(order_count),
            order_index=int(order_index),
        ):
            last_log = e._last_log_id.get(cand.asset)
            if last_log != str(cand.signal_id):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=duplicate signal_id=%s",
                    cand.asset,
                    cand.signal_id,
                )
                e._last_log_id[cand.asset] = str(cand.signal_id)
            return False, None

        stale_reason = _signal_stale_reason(
            cfg,
            source=str(getattr(cand, "source", "") or ""),
            bar_key=str(getattr(cand, "bar_key", "") or ""),
            timeframe=str(getattr(cand, "timeframe", "") or ""),
            created_ts=float(getattr(cand, "created_ts", 0.0) or 0.0),
            now_ts=now,
        )
        if stale_reason:
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=%s signal_id=%s",
                cand.asset,
                stale_reason,
                cand.signal_id,
            )
            return False, None

        if e._live_trading_pause_reason(force=False):
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=live_evidence_pause_late",
                cand.asset,
            )
            return False, None

        tick = (
            pre_tick
            if pre_tick is not None
            else mt5_async_call(
                "symbol_info_tick",
                cand.symbol,
                timeout=0.35,
                default=None,
            )
        )
        info = (
            pre_info
            if pre_info is not None
            else mt5_async_call(
                "symbol_info",
                cand.symbol,
                timeout=0.5,
                default=None,
            )
        )
        if tick is None:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=tick_missing", cand.asset)
            return False, None
        if info is None:
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=symbol_info_missing", cand.asset
            )
            return False, None

        price = float(tick.ask if cand.signal == "Buy" else tick.bid)
        if price <= 0:
            log_health.info("ENQUEUE_SKIP | asset=%s reason=bad_price", cand.asset)
            return False, None

        current_spread_points = (
            float(tick.ask - tick.bid) / float(info.point) if info.point else 99999.0
        )
        max_spread_pts = max(
            float(getattr(cfg, "max_spread_points", 0) or 0),
            float(getattr(cfg, "exec_max_spread_points", 0) or 0),
        )
        if max_spread_pts <= 0:
            max_spread_pts = 2500.0 if cand.asset == "BTC" else 350.0

        if current_spread_points > max_spread_pts:
            sp_key = f"_spread_skip_ts_{cand.asset}"
            sp_last = getattr(e, sp_key, 0.0)
            if (now - sp_last) >= 30.0:
                setattr(e, sp_key, now)
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=spread_too_high spread=%s > max=%s",
                    cand.asset,
                    int(current_spread_points),
                    int(max_spread_pts),
                )
            return False, None

        digits = int(getattr(info, "digits", 0) or 0) if info else 0
        sl_val = float(cand.sl)
        tp_val = float(cand.tp)
        if digits > 0:
            try:
                price = float(builtins.round(price, digits))
                sl_val = float(builtins.round(sl_val, digits))
                tp_val = float(builtins.round(tp_val, digits))
            except Exception as exc:  # DEFENSIVE CODE
                log_err.error("PRICE_ROUNDING_ERROR | asset=%s err=%s", cand.asset, exc)

        if cand.signal == "Buy":
            if not (sl_val < price < tp_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Buy price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None
        else:
            if not (tp_val < price < sl_val):
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=bad_sl_tp_relation side=Sell price=%.5f sl=%.5f tp=%.5f",
                    cand.asset,
                    price,
                    sl_val,
                    tp_val,
                )
                return False, None

        lot_val = float(lot_override) if lot_override is not None else float(cand.lot)
        lot_val = e._apply_asset_lot_cap(cand.asset, lot_val, "enqueue")
        if lot_val <= 0:
            log_health.info(
                "ENQUEUE_SKIP | asset=%s reason=lot_nonpositive", cand.asset
            )
            return False, None

        base_lot_val = float(lot_val)
        if phase == "B":
            orig = lot_val
            lot_val = max(0.01, lot_val * 0.5)
            log_health.info(
                "PHASE_B_LOT_REDUCE | asset=%s original_lot=%.4f reduced_lot=%.4f",
                cand.asset,
                float(orig),
                float(lot_val),
            )

        gate_reason = "ok"
        if risk and hasattr(risk, "pre_order_gate"):
            gate_fn = getattr(risk, "pre_order_gate", None)
            if callable(gate_fn):
                try:
                    allowed, gated_lot, gate_reason = gate_fn(
                        side=cand.signal,
                        confidence=float(cand.confidence),
                        lot=float(lot_val),
                        entry_price=float(price),
                        sl=float(sl_val),
                        tp=float(tp_val),
                        signal_id=str(cand.signal_id),
                        stage="enqueue",
                        base_lot=float(base_lot_val),
                        phase_snapshot=str(phase),
                    )
                except TypeError:
                    allowed, gated_lot, gate_reason = gate_fn(  # type: ignore[misc]
                        side=cand.signal,
                        confidence=float(cand.confidence),
                        lot=float(lot_val),
                        entry_price=float(price),
                        sl=float(sl_val),
                        tp=float(tp_val),
                    )
                except Exception as exc:
                    log_err.error(
                        "ENQUEUE_GATE_EXCEPTION | asset=%s err=%s", cand.asset, exc
                    )
                    return False, None

                if not allowed:
                    log_health.info(
                        "ENQUEUE_SKIP | asset=%s reason=gate_blocked gate_reason=%s",
                        cand.asset,
                        gate_reason,
                    )
                    return False, None
                lot_val = float(gated_lot or lot_val)
                lot_val = e._apply_asset_lot_cap(cand.asset, lot_val, "enqueue_gate")
                if lot_val <= 0.0:
                    log_health.info(
                        "ENQUEUE_SKIP | asset=%s reason=gate_lot_nonpositive",
                        cand.asset,
                    )
                    return False, None

        order_id = e._next_order_id(cand.asset)
        # Build a globally-unique idempotency key (UUID-suffixed) and register
        # a PENDING entry in the write-ahead log BEFORE enqueueing. This is
        # the institutional guarantee that NO duplicate order can be sent
        # under any retry / crash / restart scenario.
        try:
            from core.idempotency import (
                new_idempotency_key,
                get_default_journal,
            )

            _idem_journal = get_default_journal()
            idem = new_idempotency_key(
                cand.symbol, str(cand.signal_id), str(cand.signal)
            )
            _ok_to_send, _idem_reason = _idem_journal.should_send(idem)
            if not _ok_to_send:
                log_health.info(
                    "ENQUEUE_SKIP | asset=%s reason=idempotency_refuse:%s",
                    cand.asset,
                    _idem_reason,
                )
                return False, None
            _idem_journal.record_pending(
                idem,
                symbol=str(cand.symbol),
                side=str(cand.signal),
                volume=float(lot_val),
                price=float(price),
                sl=float(sl_val),
                tp=float(tp_val),
                magic=0,
            )
        except Exception as _idem_exc:  # defensive: never break enqueue
            idem = f"{cand.symbol}:{cand.signal_id}:{cand.signal}"
            log_health.warning(
                "ENQUEUE_IDEMPOTENCY_WARN | asset=%s err=%s",
                cand.asset,
                _idem_exc,
            )

        intent = OrderIntent(
            asset=cand.asset,
            symbol=cand.symbol,
            signal=cand.signal,
            confidence=float(cand.confidence),
            lot=float(lot_val),
            sl=float(sl_val),
            tp=float(tp_val),
            price=float(price),
            enqueue_time=float(now),
            order_id=str(order_id),
            signal_id=str(cand.signal_id),
            idempotency_key=str(idem),
            risk_manager=risk,
            cfg=cfg,
            base_lot=float(base_lot_val),
            phase_snapshot=str(phase),
            bar_key=str(getattr(cand, "bar_key", "") or ""),
            timeframe=str(getattr(cand, "timeframe", "") or ""),
            created_ts=float(getattr(cand, "created_ts", 0.0) or now),
            source=str(getattr(cand, "source", "") or "pipeline"),
        )

        with e._order_state_lock:
            e._order_rm_by_id[str(order_id)] = risk
        e._register_pending_order(intent)

        try:
            e._order_q.put(intent, timeout=0.15)
        except queue.Full:
            with e._order_state_lock:
                e._order_rm_by_id.pop(str(order_id), None)
            e._clear_pending_order(str(order_id))
            if risk and hasattr(risk, "record_execution_failure"):
                try:
                    risk.record_execution_failure(
                        order_id, now, now, "queue_backlog_drop"
                    )
                except Exception as exc:  # DEFENSIVE CODE
                    log_err.error(
                        "ENQUEUE_RECORD_FAILURE_ERROR | order_id=%s err=%s",
                        order_id,
                        exc,
                )
            log_health.warning(
                "ENQUEUE_FAIL | asset=%s reason=queue_full order_id=%s",
                cand.asset,
                order_id,
            )
            return False, None

        if risk and hasattr(risk, "on_position_opened"):
            try:
                # Mark the symbol as occupied as soon as the order enters the
                # live dispatch path so a second opposite order cannot race in
                # before fill reconciliation arrives.
                risk.on_position_opened()
            except Exception as exc:  # DEFENSIVE CODE
                log_err.error(
                    "POSITION_OPEN_MARK_ERROR | asset=%s order_id=%s err=%s",
                    cand.asset,
                    order_id,
                    exc,
                )

        e._mark_seen(cand.asset, cand.signal_id, now)
        e._last_selected_asset = cand.asset
        e._edge_last_trade[cand.asset] = (str(cand.signal_id), str(cand.signal))

        if risk:
            if hasattr(risk, "register_signal_emitted"):
                try:
                    risk.register_signal_emitted()
                except Exception as exc:  # DEFENSIVE CODE
                    log_err.error(
                        "REGISTER_SIGNAL_EMITTED_ERROR | asset=%s err=%s",
                        cand.asset,
                        exc,
                    )
            if hasattr(risk, "track_signal_survival"):
                try:
                    risk.track_signal_survival(
                        order_id,
                        cand.signal,
                        price,
                        sl_val,
                        tp_val,
                        now,
                        cand.confidence,
                    )
                except Exception as exc:  # DEFENSIVE CODE
                    log_err.error(
                        "TRACK_SIGNAL_SURVIVAL_ERROR | asset=%s err=%s", cand.asset, exc
                    )

        return True, str(order_id)

    def effective_min_conf(self, c: AssetCandidate) -> float:
        e = self._e
        cfg = e._xau_cfg if c.asset == "XAU" else e._btc_cfg
        base = float(getattr(cfg, "min_confidence_signal", 0.55) or 0.55)

        env_global: float = 0.50
        env_asset: float = float(env_global)
        floor = max(0.35, min(0.90, env_asset))

        effective_min = max(floor, base)
        rs = tuple(str(r) for r in (c.reasons or ()))

        if any(r.startswith("early_momentum") for r in rs):
            effective_min = max(floor, effective_min - 0.08)

        return float(max(0.0, min(1.0, effective_min)))

    def is_asset_blocked(self, asset: str) -> bool:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        return asset_u in e._blocked_assets

    def log_blocked_asset_skip(self, asset: str, stage: str) -> None:
        e = self._e
        key = f"{str(asset).upper()}:{stage}"
        now = time.time()
        last = float(e._last_skip_log_ts.get(key, 0.0))
        if now - last < 30.0:
            return
        e._last_skip_log_ts[key] = now
        log_health.info(
            "ASSET_BLOCKED_SKIP | asset=%s stage=%s blocked_assets=%s",
            str(asset).upper(),
            stage,
            ",".join(sorted(set(e._blocked_assets))),
        )

    def candidate_is_tradeable(self, c: AssetCandidate) -> bool:
        if self.is_asset_blocked(c.asset):
            self.log_blocked_asset_skip(c.asset, "candidate_tradeable")
            return False
        if c.signal not in ("Buy", "Sell"):
            return False
        if c.blocked:
            return False
        if c.lot <= 0.0:
            return False
        if c.sl <= 0.0 or c.tp <= 0.0:
            return False

        conf_val = float(c.confidence)
        return bool(conf_val >= self.effective_min_conf(c))

    def asset_open_positions(self, asset: str) -> int:
        e = self._e
        pipe = e._xau if str(asset).upper() == "XAU" else e._btc
        if pipe is None:
            return 0
        try:
            return int(pipe.open_positions())
        except Exception as exc:  # DEFENSIVE CODE
            log_err.error("ASSET_OPEN_POSITIONS_ERROR | asset=%s err=%s", asset, exc)
            return 0

    def asset_max_positions(self, asset: str) -> int:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        return max(1, int(e._max_open_positions.get(asset_u, 1) or 1))

    def asset_lot_cap(self, asset: str) -> float:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        cfg = e._xau_cfg if asset_u == "XAU" else e._btc_cfg
        broker_max = float(
            getattr(getattr(cfg, "symbol_params", None), "lot_max", 0.0) or 0.0
        )
        cap = float(
            e._max_lot_cap.get(asset_u, broker_max if broker_max > 0.0 else 0.0) or 0.0
        )
        if broker_max > 0.0:
            if cap <= 0.0:
                cap = broker_max
            else:
                cap = min(cap, broker_max)
        return max(0.0, cap)

    def apply_asset_lot_cap(self, asset: str, lot: float, stage: str) -> float:
        lot_val = float(lot)
        cap = self.asset_lot_cap(asset)
        if cap > 0.0 and lot_val > cap:
            log_health.warning(
                "LOT_CAP_APPLIED | asset=%s stage=%s lot=%.4f cap=%.4f",
                str(asset).upper(),
                stage,
                lot_val,
                cap,
            )
            return float(cap)
        return lot_val

    def orders_for_candidate(self, cand: AssetCandidate) -> int:
        pipe = self._e._xau if cand.asset == "XAU" else self._e._btc
        if not pipe or not pipe.risk:
            return 1

        phase = self._e._get_phase(pipe.risk)
        if phase == "C":
            return 0

        try:
            hard_stop_attr = getattr(pipe.risk, "requires_hard_stop", False)
            hard_stop_active = bool(
                hard_stop_attr() if callable(hard_stop_attr) else hard_stop_attr
            )
            if hard_stop_active:
                return 0
        except Exception:
            pass

        return 1

    @staticmethod
    def min_lot(risk: Any) -> float:
        try:
            if risk and hasattr(risk, "_symbol_meta"):
                risk._symbol_meta()  # type: ignore[attr-defined]
            return float(getattr(risk, "_vol_min", 0.01) or 0.01)
        except Exception:
            return 0.01

    def split_lot(self, lot: float, parts: int, risk: Any, cfg: Any) -> List[float]:
        lot = float(lot)
        if int(parts) <= 1:
            return [lot]

        min_lot = self.min_lot(risk)
        split_enabled = bool(getattr(cfg, "multi_order_split_lot", True))
        if not split_enabled:
            if min_lot <= 0 or lot < min_lot:
                return [lot]
            return [lot for _ in range(int(parts))]

        if min_lot <= 0 or (lot / float(parts)) < min_lot:
            return [lot]

        base = lot / float(parts)
        lots = [base for _ in range(int(parts))]
        lots[-1] = float(lot) - (base * float(int(parts) - 1))
        return lots

    def execute_candidates(self, candidates: List[AssetCandidate]) -> None:
        e = self._e
        for selected in candidates:
            if not self.candidate_is_tradeable(selected):
                log_health.info(
                    "FSM_ORDER_SKIP | asset=%s signal=%s conf=%.3f blocked=%s",
                    selected.asset,
                    selected.signal,
                    selected.confidence,
                    selected.blocked,
                )
                continue

            if e._manual_stop:
                log_health.info(
                    "FSM_ORDER_SKIP | reason=manual_stop asset=%s", selected.asset
                )
                continue

            open_positions = self.asset_open_positions(selected.asset)
            max_positions = self.asset_max_positions(selected.asset)
            if max_positions > 0 and open_positions >= max_positions:
                log_health.info(
                    "FSM_ORDER_SKIP | asset=%s reason=max_positions open=%d max=%d",
                    selected.asset,
                    open_positions,
                    max_positions,
                )
                continue

            order_count = self.orders_for_candidate(selected)
            if int(order_count) <= 0:
                continue

            risk = (
                e._xau.risk
                if selected.asset == "XAU" and e._xau
                else (e._btc.risk if e._btc else None)
            )
            cfg = e._xau_cfg if selected.asset == "XAU" else e._btc_cfg
            lots = self.split_lot(float(selected.lot), order_count, risk, cfg)
            signal_enqueued = False

            for idx, lot_val in enumerate(lots):
                ok, oid = self.enqueue_order(
                    selected,
                    order_index=int(idx),
                    order_count=int(order_count),
                    lot_override=float(lot_val),
                )
                if ok:
                    if not signal_enqueued:
                        emit_ts = time.time()
                        e._record_signal_emit(selected.asset, emit_ts)

                        last_notified_id, last_notified_sig = e._edge_last_notified.get(
                            selected.asset,
                            ("", "Neutral"),
                        )
                        current_sig_id = str(selected.signal_id)
                        current_sig = str(selected.signal)
                        if e._signal_notifier and not (
                            last_notified_id == current_sig_id
                            and last_notified_sig == current_sig
                        ):
                            try:
                                e._signal_notifier(
                                    selected.asset,
                                    {
                                        "type": "LIVE_ENQUEUED",
                                        "asset": selected.asset,
                                        "signal": selected.signal,
                                        "confidence": float(selected.confidence),
                                        "reasons": list(selected.reasons or ()),
                                        "signal_id": current_sig_id,
                                        "order_id": str(oid or ""),
                                        "lot": float(lot_val),
                                        "phase": (
                                            e._get_phase(risk)
                                            if risk is not None
                                            else ""
                                        ),
                                        "blocked": False,
                                    },
                                )
                            except Exception as exc:
                                log_err.error(
                                    "SIGNAL_NOTIFY_ERROR | asset=%s signal_id=%s err=%s",
                                    selected.asset,
                                    current_sig_id,
                                    exc,
                                )
                        e._edge_last_notified[selected.asset] = (
                            current_sig_id,
                            current_sig,
                        )
                        signal_enqueued = True

                    log_health.info(
                        "FSM_ORDER_ENQUEUED | asset=%s signal=%s conf=%.3f lot=%.4f order_id=%s",
                        selected.asset,
                        selected.signal,
                        selected.confidence,
                        float(lot_val),
                        oid,
                    )

    def verification_step(self) -> None:
        e = self._e
        while True:
            try:
                r = e._result_q.get_nowait()
            except queue.Empty:
                break

            try:
                e._resolve_order_result(r)
            except Exception as exc:  # DEFENSIVE CODE
                log_err.error(
                    "VERIFICATION_RESOLVE_ERROR | order_id=%s err=%s",
                    getattr(r, "order_id", "unknown"),
                    exc,
                )
            finally:
                try:
                    e._result_q.task_done()
                except Exception:
                    pass

        e._sync_pending_orders()

        if e._xau:
            e._xau.reconcile_positions()
        if e._btc:
            e._btc.reconcile_positions()
        e._last_reconcile_ts = time.time()
        e._refresh_open_position_protection()

        e._update_portfolio_risk_state()

        open_xau = e._xau.open_positions() if e._xau else 0
        open_btc = e._btc.open_positions() if e._btc else 0
        e._track_position_closures(open_xau, open_btc)
        e._active_asset = e._select_active_asset(open_xau, open_btc)
        e._heartbeat(open_xau, open_btc)


# --- Order sync manager ----------------------------------------------------
class OrderSyncManager:
    """Owns pending-order state tracking and broker reconciliation."""

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    def register_pending_order(self, intent: OrderIntent) -> None:
        e = self._e
        with e._order_state_lock:
            e._pending_order_meta[str(intent.order_id)] = {
                "asset": str(intent.asset),
                "symbol": str(intent.symbol),
                "signal": str(intent.signal),
                "signal_id": str(intent.signal_id),
                "confidence": float(intent.confidence),
                "lot": float(intent.lot),
                "sl": float(intent.sl),
                "tp": float(intent.tp),
                "req_price": float(intent.price),
                "enqueue_ts": float(intent.enqueue_time),
                "magic": int(getattr(intent.cfg, "magic", 777001) or 777001),
                "risk": intent.risk_manager,
            }

    def clear_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._e._order_state_lock:
            return self._e._pending_order_meta.pop(str(order_id), None)

    @staticmethod
    def _should_defer_result(
        r: ExecutionResult, meta: Optional[Dict[str, Any]]
    ) -> bool:
        if not isinstance(meta, dict) or bool(r.ok):
            return False
        reason = str(r.reason or "").lower()
        return (
            reason.startswith("order_send_none")
            or reason.startswith("retry_retcode_")
            or reason.startswith("exception_")
        )

    @staticmethod
    def _build_intent_snapshot(order_id: str, meta: Dict[str, Any]) -> Any:
        return SimpleNamespace(
            asset=str(meta.get("asset", "") or ""),
            symbol=str(meta.get("symbol", "") or ""),
            signal=str(meta.get("signal", "") or ""),
            confidence=float(meta.get("confidence", 0.0) or 0.0),
            lot=float(meta.get("lot", 0.0) or 0.0),
            sl=float(meta.get("sl", 0.0) or 0.0),
            tp=float(meta.get("tp", 0.0) or 0.0),
            price=float(meta.get("req_price", 0.0) or 0.0),
            enqueue_time=float(meta.get("enqueue_ts", 0.0) or 0.0),
            order_id=str(order_id),
            signal_id=str(meta.get("signal_id", "") or ""),
        )

    @staticmethod
    def _call_mt5(method_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(mt5, str(method_name), None)
        if not callable(fn):
            raise AttributeError(f"mt5.{method_name} not callable")
        with MT5_LOCK:
            return fn(*args, **kwargs)

    def resolve_order_result(self, r: ExecutionResult) -> None:
        e = self._e
        order_id = str(r.order_id)
        with e._order_state_lock:
            meta = e._pending_order_meta.get(order_id)
        if self._should_defer_result(r, meta):
            log_health.info(
                "ORDER_SYNC_WAIT | order_id=%s reason=%s retcode=%s",
                order_id,
                r.reason,
                r.retcode,
            )
            return

        e._record_execution_telemetry(r)
        with e._order_state_lock:
            rm = e._order_rm_by_id.pop(order_id, None)
            meta = e._pending_order_meta.pop(order_id, None)
        if rm is None and isinstance(meta, dict):
            rm = meta.get("risk")
        if rm is None:
            asset_hint = str((meta or {}).get("asset", "") or "").upper().strip()
            if asset_hint == "XAU" and e._xau is not None:
                rm = e._xau.risk
            elif asset_hint == "BTC" and e._btc is not None:
                rm = e._btc.risk
            else:
                log_health.warning(
                    "ORDER_SYNC_ORPHAN_RESULT | order_id=%s ok=%s reason=%s",
                    order_id,
                    r.ok,
                    r.reason,
                )
                return
        if rm and hasattr(rm, "on_execution_result"):
            try:
                rm.on_execution_result(r)
            except Exception as exc:
                log_err.error(
                    "ORDER_SYNC_RISK_UPDATE_ERROR | order_id=%s err=%s", order_id, exc
                )
        if (
            isinstance(meta, dict)
            and str(r.reason or "").startswith("sync_")
            and getattr(e, "_order_notifier", None)
        ):
            try:
                e._order_notifier(self._build_intent_snapshot(order_id, meta), r)
            except Exception:
                pass

    def probe_pending_order_state(
        self,
        order_id: str,
        meta: Dict[str, Any],
        now_ts: float,
    ) -> Optional[ExecutionResult]:
        e = self._e
        try:
            symbol = str(meta.get("symbol", "") or "")
            if not symbol:
                return None
            signal = str(meta.get("signal", "") or "")
            if signal not in ("Buy", "Sell"):
                return None
            want_pos_type = (
                mt5.POSITION_TYPE_BUY if signal == "Buy" else mt5.POSITION_TYPE_SELL
            )
            want_deal_type = (
                mt5.ORDER_TYPE_BUY if signal == "Buy" else mt5.ORDER_TYPE_SELL
            )
            enqueue_ts = float(meta.get("enqueue_ts", 0.0) or 0.0)
            req_price = float(meta.get("req_price", 0.0) or 0.0)
            req_lot = float(meta.get("lot", 0.0) or 0.0)
            magic = int(meta.get("magic", 777001) or 777001)
            signal_id = str(meta.get("signal_id", "") or "")

            positions = self._call_mt5("positions_get", symbol=symbol) or []

            for p in positions:
                try:
                    if int(getattr(p, "type", -1) or -1) != int(want_pos_type):
                        continue
                    if int(getattr(p, "magic", 0) or 0) != int(magic):
                        continue
                    p_vol = float(getattr(p, "volume", 0.0) or 0.0)
                    p_price = float(getattr(p, "price_open", 0.0) or req_price)
                    p_ticket = int(getattr(p, "ticket", 0) or 0)
                    p_time = float(getattr(p, "time", 0.0) or now_ts)
                    if p_time + 2.0 < enqueue_ts:
                        continue
                    if req_lot > 0.0 and p_vol <= 0.0:
                        continue
                    return ExecutionResult(
                        order_id=str(order_id),
                        signal_id=signal_id,
                        ok=True,
                        reason="sync_recovered_position",
                        sent_ts=float(enqueue_ts or now_ts),
                        fill_ts=float(p_time if p_time > 0 else now_ts),
                        req_price=float(req_price),
                        exec_price=float(p_price if p_price > 0 else req_price),
                        volume=float(p_vol if p_vol > 0 else req_lot),
                        slippage=float(
                            abs((p_price if p_price > 0 else req_price) - req_price)
                        ),
                        retcode=0,
                        order_ticket=0,
                        deal_ticket=0,
                        position_ticket=int(p_ticket),
                    )
                except Exception:
                    continue

            if (now_ts - enqueue_ts) < e._order_sync_timeout_sec:
                return None

            start_dt = datetime.fromtimestamp(max(0.0, enqueue_ts - 5.0))
            end_dt = datetime.fromtimestamp(now_ts + 1.0)
            deals = self._call_mt5("history_deals_get", start_dt, end_dt) or []
            entry_in = int(getattr(mt5, "DEAL_ENTRY_IN", 0) or 0)
            best: Optional[Tuple[float, float, float, int]] = None
            best_score: Optional[Tuple[float, float]] = None
            for d in deals:
                try:
                    if str(getattr(d, "symbol", "")) != symbol:
                        continue
                    if int(getattr(d, "magic", 0) or 0) != int(magic):
                        continue
                    if int(getattr(d, "entry", 0) or 0) != entry_in:
                        continue
                    if int(getattr(d, "type", -1) or -1) != int(want_deal_type):
                        continue
                    d_time = float(getattr(d, "time", 0.0) or 0.0)
                    if d_time + 2.0 < enqueue_ts:
                        continue
                    d_vol = float(getattr(d, "volume", 0.0) or 0.0)
                    d_price = float(getattr(d, "price", 0.0) or req_price)
                    d_ticket = int(getattr(d, "ticket", 0) or 0)
                    vol_delta = abs(d_vol - req_lot)
                    score = (float(vol_delta), -float(d_time))
                    if best_score is None or score < best_score:
                        best_score = score
                        best = (d_time, d_price, d_vol, d_ticket)
                except Exception:
                    continue

            if best is not None:
                b_time, b_price, b_vol, b_ticket = best
                return ExecutionResult(
                    order_id=str(order_id),
                    signal_id=signal_id,
                    ok=True,
                    reason="sync_recovered_deal",
                    sent_ts=float(enqueue_ts or now_ts),
                    fill_ts=float(b_time if b_time > 0 else now_ts),
                    req_price=float(req_price),
                    exec_price=float(b_price if b_price > 0 else req_price),
                    volume=float(b_vol if b_vol > 0 else req_lot),
                    slippage=float(
                        abs((b_price if b_price > 0 else req_price) - req_price)
                    ),
                    retcode=0,
                    order_ticket=0,
                    deal_ticket=int(b_ticket),
                    position_ticket=0,
                )

            return ExecutionResult(
                order_id=str(order_id),
                signal_id=signal_id,
                ok=False,
                reason="sync_timeout_no_broker_state",
                sent_ts=float(enqueue_ts or now_ts),
                fill_ts=float(now_ts),
                req_price=float(req_price),
                exec_price=float(req_price),
                volume=float(req_lot),
                slippage=0.0,
                retcode=-3,
                order_ticket=0,
                deal_ticket=0,
                position_ticket=0,
            )
        except Exception as exc:
            log_err.error("ORDER_SYNC_PROBE_ERROR | order_id=%s err=%s", order_id, exc)
            return None

    def sync_pending_orders(self) -> None:
        e = self._e
        if e.dry_run:
            return
        now_ts = time.time()
        with e._order_state_lock:
            if not e._pending_order_meta:
                return
            if (now_ts - e._last_order_sync_ts) < e._order_sync_interval_sec:
                return
            e._last_order_sync_ts = now_ts
            pending_items = list(e._pending_order_meta.items())[:64]

        for order_id, meta in pending_items:
            enqueue_ts = float(meta.get("enqueue_ts", 0.0) or 0.0)
            if enqueue_ts > 0.0 and (now_ts - enqueue_ts) < e._order_sync_grace_sec:
                continue
            sync_res = self.probe_pending_order_state(order_id, meta, now_ts)
            if sync_res is None:
                continue
            log_health.info(
                "ORDER_SYNC_RESOLVED | order_id=%s ok=%s reason=%s",
                order_id,
                sync_res.ok,
                sync_res.reason,
            )
            try:
                self.resolve_order_result(sync_res)
            except Exception as exc:
                log_err.error(
                    "ORDER_SYNC_RESOLVE_ERROR | order_id=%s err=%s", order_id, exc
                )


# --- Engine mixin extracted from engine.py --------------------------------
class EngineExecutionMixin:
    def _restart_exec_worker(self) -> None:
        self._stop_exec_workers(timeout=4.0)

        workers: List[ExecutionWorker] = []
        for idx in range(int(self._exec_parallelism)):
            worker = ExecutionWorker(
                self._order_q,
                self._result_q,
                self.dry_run,
                self._order_notify_bridge,
                self._direct_order_result_bridge,
                worker_label=f"{idx + 1}",
            )
            worker.start()
            workers.append(worker)

        self._exec_worker = workers[0] if workers else None
        self._exec_aux_workers = workers[1:] if len(workers) > 1 else []
        log_health.info("EXEC_WORKERS_STARTED | count=%d", len(workers))

    def _iter_exec_workers(self) -> List[ExecutionWorker]:
        workers: List[ExecutionWorker] = []
        if self._exec_worker is not None:
            workers.append(self._exec_worker)
        workers.extend(w for w in self._exec_aux_workers if w is not None)
        return workers

    def _stop_exec_workers(self, timeout: float = 4.0) -> None:
        workers = self._iter_exec_workers()
        for worker in workers:
            try:
                worker.stop()
            except Exception:
                pass
        for worker in workers:
            try:
                worker.join(timeout=timeout)
            except Exception:
                pass
        self._exec_worker = None
        self._exec_aux_workers = []

    def _total_exec_queue_size(self) -> int:
        try:
            return int(self._order_q.qsize())
        except Exception:
            return 0

    def _order_notify_bridge(self, intent: OrderIntent, res: ExecutionResult) -> None:
        if self._order_notifier:
            try:
                self._order_notifier(intent, res)
            except Exception:
                pass

    def _direct_order_result_bridge(self, res: ExecutionResult) -> None:
        self._resolve_order_result(res)

    @staticmethod
    def _drain_queue(q: "queue.Queue[Any]", limit: int = 10_000) -> int:
        n = 0
        try:
            while n < int(limit):
                _ = q.get_nowait()
                try:
                    q.task_done()
                except Exception:
                    pass
                n += 1
        except Exception:
            pass
        return n

    def _register_pending_order(self, intent: OrderIntent) -> None:
        self._order_sync_manager.register_pending_order(intent)

    def _clear_pending_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self._order_sync_manager.clear_pending_order(order_id)

    def _resolve_order_result(self, r: ExecutionResult) -> None:
        self._order_sync_manager.resolve_order_result(r)

    def _probe_pending_order_state(
        self, order_id: str, meta: Dict[str, Any], now_ts: float
    ) -> Optional[ExecutionResult]:
        return self._order_sync_manager.probe_pending_order_state(
            order_id, meta, now_ts
        )

    def _sync_pending_orders(self) -> None:
        self._order_sync_manager.sync_pending_orders()

    def _next_order_id(self, asset: str) -> str:
        with self._lock:
            self._order_counter += 1
            counter = self._order_counter
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"PORD_{asset}_{ts}_{counter}_{os.getpid()}"

    def _effective_min_conf(self, c: AssetCandidate) -> float:
        return self._execution_manager.effective_min_conf(c)

    def _is_asset_blocked(self, asset: str) -> bool:
        return self._execution_manager.is_asset_blocked(asset)

    def _log_blocked_asset_skip(self, asset: str, stage: str) -> None:
        self._execution_manager.log_blocked_asset_skip(asset, stage)

    def _candidate_is_tradeable(self, c: AssetCandidate) -> bool:
        return self._execution_manager.candidate_is_tradeable(c)

    def _asset_open_positions(self, asset: str) -> int:
        return self._execution_manager.asset_open_positions(asset)

    def _asset_max_positions(self, asset: str) -> int:
        return self._execution_manager.asset_max_positions(asset)

    def _asset_lot_cap(self, asset: str) -> float:
        return self._execution_manager.asset_lot_cap(asset)

    def _apply_asset_lot_cap(self, asset: str, lot: float, stage: str) -> float:
        return self._execution_manager.apply_asset_lot_cap(asset, lot, stage)

    def _enqueue_order(
        self,
        cand: AssetCandidate,
        *,
        order_index: int = 0,
        order_count: int = 1,
        lot_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        return self._execution_manager.enqueue_order(
            cand,
            order_index=order_index,
            order_count=order_count,
            lot_override=lot_override,
        )

    def _orders_for_candidate(self, cand: AssetCandidate) -> int:
        return self._execution_manager.orders_for_candidate(cand)

    @staticmethod
    def _min_lot(risk: Any) -> float:
        return ExecutionManager.min_lot(risk)

    def _split_lot(self, lot: float, parts: int, risk: Any, cfg: Any) -> list[float]:
        return self._execution_manager.split_lot(lot, parts, risk, cfg)

    def _report_cycle_latency_ms(self, latency_ms: float) -> None:
        try:
            lat = float(latency_ms)
        except Exception:
            return
        if lat < 0.0:
            return
        self._mark_runtime_alive()
        self._fsm_cycle_ms_hist.append(lat)
        if lat > (self._live_max_p95_latency_ms * 1.75):
            now = time.time()
            if (now - self._last_cycle_spike_log_ts) >= 5.0:
                self._last_cycle_spike_log_ts = now
                log_health.warning(
                    "FSM_CYCLE_SPIKE | latency_ms=%.1f threshold_ms=%.1f",
                    lat,
                    self._live_max_p95_latency_ms * 1.75,
                )

    def _record_execution_telemetry(self, r: ExecutionResult) -> None:
        try:
            sent = float(getattr(r, "sent_ts", 0.0) or 0.0)
            fill = float(getattr(r, "fill_ts", 0.0) or 0.0)
            slip = abs(float(getattr(r, "slippage", 0.0) or 0.0))
            if fill > sent > 0.0:
                self._exec_latency_ms_hist.append((fill - sent) * 1000.0)
            if slip >= 0.0:
                self._exec_slippage_hist.append(slip)
        except Exception:
            return

    @staticmethod
    def _estimate_position_atr(risk: Any, position: Any) -> float:
        cfg = getattr(risk, "cfg", None)
        entry = float(getattr(position, "price_open", 0.0) or 0.0)
        sl_price = float(getattr(position, "sl", 0.0) or 0.0)
        tp_price = float(getattr(position, "tp", 0.0) or 0.0)
        estimates: List[float] = []

        try:
            cached_atr = float(getattr(risk, "_cached_atr", 0.0) or 0.0)
        except Exception:
            cached_atr = 0.0
        if cached_atr > 0.0:
            estimates.append(cached_atr)

        sl_mult = max(float(getattr(cfg, "atr_sl_multiplier", 0.0) or 0.0), 0.01)
        if entry > 0.0 and sl_price > 0.0:
            sl_dist = abs(entry - sl_price)
            if sl_dist > 0.0:
                estimates.append(sl_dist / sl_mult)

        tp_min = float(getattr(cfg, "atr_tp_min_multiplier", 0.0) or 0.0)
        tp_max = float(getattr(cfg, "atr_tp_max_multiplier", 0.0) or 0.0)
        tp_candidates = [v for v in (tp_min, tp_max) if v > 0.0]
        tp_mult = min(tp_candidates) if tp_candidates else 0.0
        if entry > 0.0 and tp_price > 0.0 and tp_mult > 0.0:
            tp_dist = abs(tp_price - entry)
            if tp_dist > 0.0:
                estimates.append(tp_dist / tp_mult)

        if estimates:
            return float(max(estimates))
        if entry > 0.0:
            return float(max(entry * 0.001, 0.00001))
        return 0.0

    def _refresh_open_position_protection(self) -> None:
        if self.dry_run or mt5 is None:
            return

        active_tickets: set[int] = set()
        for asset, pipe in (("XAU", self._xau), ("BTC", self._btc)):
            if pipe is None:
                continue
            risk = getattr(pipe, "risk", None)
            cfg = getattr(risk, "cfg", None) if risk is not None else None
            symbol = str(getattr(pipe, "symbol", "") or "")
            if risk is None or cfg is None or not symbol:
                continue

            try:
                with MT5_LOCK:
                    positions = mt5.positions_get(symbol=symbol) or []
            except Exception as exc:
                log_err.error(
                    "POSITION_PROTECT_FETCH_ERROR | asset=%s symbol=%s err=%s",
                    asset,
                    symbol,
                    exc,
                )
                continue

            magic = int(getattr(cfg, "magic", 0) or 0)
            filtered_positions = []
            for position in positions:
                try:
                    if magic > 0 and int(getattr(position, "magic", 0) or 0) != magic:
                        continue
                    filtered_positions.append(position)
                except Exception:
                    continue

            for position in filtered_positions:
                ticket = 0
                try:
                    ticket = int(getattr(position, "ticket", 0) or 0)
                    if ticket <= 0:
                        continue
                    active_tickets.add(ticket)
                    self._maybe_update_position_protection(asset, pipe, position)
                except Exception as exc:
                    log_err.error(
                        "POSITION_PROTECT_ERROR | asset=%s ticket=%s err=%s",
                        asset,
                        ticket,
                        exc,
                    )

        with self._lock:
            stale_tickets = [
                int(ticket)
                for ticket in self._position_protection_last_ts_by_ticket.keys()
                if int(ticket) not in active_tickets
            ]
            for ticket in stale_tickets:
                self._position_protection_last_ts_by_ticket.pop(int(ticket), None)
                self._position_protection_last_sl_by_ticket.pop(int(ticket), None)

    def _maybe_update_position_protection(
        self,
        asset: str,
        pipe: Any,
        position: Any,
    ) -> None:
        risk = getattr(pipe, "risk", None)
        cfg = getattr(risk, "cfg", None) if risk is not None else None
        if risk is None or cfg is None or mt5 is None:
            return

        ticket = int(getattr(position, "ticket", 0) or 0)
        if ticket <= 0:
            return

        now = time.time()
        with self._lock:
            last_ts = float(
                self._position_protection_last_ts_by_ticket.get(ticket, 0.0) or 0.0
            )
        if (now - last_ts) < self._position_protection_interval_sec:
            return

        buy_type = int(getattr(mt5, "POSITION_TYPE_BUY", 0) or 0)
        sell_type = int(getattr(mt5, "POSITION_TYPE_SELL", 1) or 1)
        pos_type = int(getattr(position, "type", -1) or -1)
        if pos_type == buy_type:
            side = "Buy"
        elif pos_type == sell_type:
            side = "Sell"
        else:
            return

        symbol = str(getattr(position, "symbol", "") or getattr(pipe, "symbol", ""))
        entry_price = float(getattr(position, "price_open", 0.0) or 0.0)
        sl_price = float(getattr(position, "sl", 0.0) or 0.0)
        tp_price = float(getattr(position, "tp", 0.0) or 0.0)
        if not symbol or entry_price <= 0.0 or sl_price <= 0.0 or tp_price <= 0.0:
            return

        try:
            with MT5_LOCK:
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
        except Exception as exc:
            log_err.error(
                "POSITION_PROTECT_SNAPSHOT_ERROR | asset=%s ticket=%s err=%s",
                asset,
                ticket,
                exc,
            )
            return
        if info is None or tick is None:
            return

        point = float(getattr(info, "point", 0.0) or 0.0)
        digits = int(getattr(info, "digits", 0) or 0)
        if point <= 0.0 and digits > 0:
            point = 10.0 ** (-digits)
        price_tol = max(point, abs(entry_price) * 1e-8, 1e-10)
        min_buffer = max(
            price_tol,
            float(getattr(info, "trade_stops_level", 0.0) or 0.0) * max(point, 0.0),
        )

        if side == "Buy":
            market_price = float(getattr(tick, "bid", 0.0) or 0.0)
        else:
            market_price = float(getattr(tick, "ask", 0.0) or 0.0)
        if market_price <= 0.0:
            market_price = float(getattr(position, "price_current", 0.0) or entry_price)
        if market_price <= 0.0:
            return

        atr_est = self._estimate_position_atr(risk, position)
        if atr_est <= 0.0:
            return

        try:
            new_sl = risk.check_trailing_stop(
                current_price=float(market_price),
                entry_price=float(entry_price),
                sl_price=float(sl_price),
                tp_price=float(tp_price),
                side=str(side),
                atr=float(atr_est),
            )
        except Exception as exc:
            log_err.error(
                "POSITION_PROTECT_CALC_ERROR | asset=%s ticket=%s err=%s",
                asset,
                ticket,
                exc,
            )
            return

        if new_sl is None:
            return

        target_sl = float(new_sl)
        if side == "Buy":
            target_sl = min(target_sl, market_price - min_buffer)
            if digits > 0:
                target_sl = float(builtins.round(target_sl, digits))
            if not (sl_price + price_tol < target_sl < market_price - price_tol):
                return
        else:
            target_sl = max(target_sl, market_price + min_buffer)
            if digits > 0:
                target_sl = float(builtins.round(target_sl, digits))
            if not (market_price + price_tol < target_sl < sl_price - price_tol):
                return

        with self._lock:
            last_sent_sl = self._position_protection_last_sl_by_ticket.get(ticket)
        if last_sent_sl is not None and abs(float(last_sent_sl) - target_sl) <= price_tol:
            return

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": symbol,
            "sl": float(target_sl),
            "tp": float(tp_price),
            "deviation": 50,
            "magic": int(getattr(cfg, "magic", 0) or 0),
            "comment": f"protect_{str(asset).lower()}",
        }

        try:
            with MT5_LOCK:
                result = mt5.order_send(request)
        except Exception as exc:
            with self._lock:
                self._position_protection_last_ts_by_ticket[ticket] = now
            log_err.error(
                "POSITION_PROTECT_SEND_ERROR | asset=%s ticket=%s err=%s",
                asset,
                ticket,
                exc,
            )
            return

        retcode = int(getattr(result, "retcode", 0) or 0) if result is not None else 0
        done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009) or 10009)
        no_changes = int(getattr(mt5, "TRADE_RETCODE_NO_CHANGES", 10025) or 10025)
        with self._lock:
            self._position_protection_last_ts_by_ticket[ticket] = now
        if retcode in {done, no_changes}:
            mode = (
                "breakeven"
                if abs(target_sl - entry_price) <= max(price_tol * 2.0, 1e-10)
                else "trail"
            )
            with self._lock:
                self._position_protection_last_sl_by_ticket[ticket] = float(target_sl)
            log_health.info(
                "POSITION_PROTECT_UPDATE | asset=%s ticket=%s mode=%s old_sl=%.5f new_sl=%.5f price=%.5f",
                asset,
                ticket,
                mode,
                sl_price,
                target_sl,
                market_price,
            )
            return

        log_health.warning(
            "POSITION_PROTECT_REJECTED | asset=%s ticket=%s retcode=%s old_sl=%.5f new_sl=%.5f",
            asset,
            ticket,
            retcode,
            sl_price,
            target_sl,
        )

    def close_all(self) -> bool:
        """Best-effort close all open positions and pending orders."""
        try:
            res = close_all_position()
        except Exception as exc:
            log_err.error("ENGINE_CLOSE_ALL_EXCEPTION | err=%s", exc)
            return False

        ok = bool(res.get("ok", False))
        closed = int(res.get("closed", 0) or 0)
        canceled = int(res.get("canceled", 0) or 0)
        errors = list(res.get("errors") or [])

        if ok:
            log_health.info(
                "ENGINE_CLOSE_ALL_DONE | closed=%d canceled=%d",
                closed,
                canceled,
            )
        else:
            preview = " | ".join(str(e) for e in errors[:3]) if errors else "unknown"
            log_err.error(
                "ENGINE_CLOSE_ALL_FAILED | closed=%d canceled=%d errors=%s",
                closed,
                canceled,
                preview,
            )
        return ok

    def _emergency_flatten(self, reason: str) -> bool:
        """
        Force-close all strategy symbols on critical halt paths.

        This is intentionally fail-safe and is called from kill-switch / HALT
        transitions to prevent orphaned positions.
        """
        if self.dry_run:
            log_health.warning(
                "EMERGENCY_FLATTEN_SKIP | reason=%s mode=dry_run", reason
            )
            return True

        reason_s = str(reason or "unspecified")
        now = time.time()
        with self._lock:
            if (
                now - self._last_flatten_ts
            ) < self._flatten_cooldown_sec and reason_s == self._last_flatten_reason:
                return False
            self._last_flatten_ts = now
            self._last_flatten_reason = reason_s

        log_err.critical(
            "EMERGENCY_FLATTEN_START | reason=%s retries=%d",
            reason_s,
            self._flatten_retries,
        )
        ok = False
        for attempt in range(1, self._flatten_retries + 1):
            try:
                ok = bool(self.close_all())
            except Exception as exc:
                ok = False
                log_err.error(
                    "EMERGENCY_FLATTEN_EXCEPTION | attempt=%d reason=%s err=%s",
                    attempt,
                    reason_s,
                    exc,
                )

            if ok:
                log_health.critical(
                    "EMERGENCY_FLATTEN_DONE | reason=%s attempt=%d", reason_s, attempt
                )
                return True

            time.sleep(self._flatten_backoff_sec * float(attempt))

        log_err.critical("EMERGENCY_FLATTEN_FAILED | reason=%s", reason_s)
        return False


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "EngineExecutionMixin",
    "ExecutionManager",
    "ExecutionWorker",
    "OrderSyncManager",
]
