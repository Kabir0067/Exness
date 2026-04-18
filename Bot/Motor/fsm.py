"""
Bot/Motor/fsm.py - Deterministic FSM orchestrator.

Ин модул як мошини ҳолатҳои ягонаро (FSM) нишон медиҳад,
ки тартиби кори системаро (sync -> infer -> risk -> execute) таъмин мекунад.
"""

from __future__ import annotations

import math as _math
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import MetaTrader5 as mt5

from Backtest.engine import BTC_BACKTEST_CONFIG, XAU_BACKTEST_CONFIG, BacktestEngine
from core.ml_router import MLSignal, fetch_ml_payloads
from core.portfolio_risk import AssetExposure
from mt5_client import MT5_LOCK, mt5_status

from .inference import InferenceEngine
from .models import (
    _DIAG_ENABLED,
    _DIAG_EVERY_SEC,
    AssetCandidate,
    EngineCycleContext,
    EngineState,
    PortfolioStatus,
    StepDecision,
    log_diag,
    log_err,
    log_health,
    safe_json_dumps,
)
from .pipeline import UTCScheduler

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


# ---------------------------------------------------------------------------


class IFSMEnginePorts(Protocol):
    """Hexagonal runtime ports for deterministic engine orchestration."""

    def is_running(self) -> bool: ...

    def cycle_floor_sec(self) -> float: ...

    def report_cycle_latency_ms(self, latency_ms: float) -> None: ...

    def step_boot(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_data_sync(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_ml_inference(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_risk_calc(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_execution_queue(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_verification(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision: ...

    def step_halt(self, ctx: EngineCycleContext) -> None: ...

    def on_exception(
        self,
        state: EngineState,
        ctx: EngineCycleContext,
        exc: Exception,
    ) -> StepDecision: ...


# --- Runtime ---------------------------------------------------------------


class DeterministicFSMRunner:
    """
    Deterministic single-threaded FSM runtime.
    The runner owns no trading logic; it only controls state transitions,
    dispatches stage handlers through ports, and enforces cycle pacing.
    """

    def __init__(self, ports: IFSMEnginePorts) -> None:
        self._ports = ports
        self._dispatch: dict[
            EngineState,
            Callable[[EngineState, EngineCycleContext], StepDecision],
        ] = {
            EngineState.BOOT: self._ports.step_boot,
            EngineState.DATA_SYNC: self._ports.step_data_sync,
            EngineState.ML_INFERENCE: self._ports.step_ml_inference,
            EngineState.RISK_CALC: self._ports.step_risk_calc,
            EngineState.EXECUTION_QUEUE: self._ports.step_execution_queue,
            EngineState.VERIFICATION: self._ports.step_verification,
        }

    def run(
        self,
        *,
        initial_state: EngineState = EngineState.BOOT,
        initial_ctx: EngineCycleContext | None = None,
    ) -> None:
        state = initial_state
        ctx = initial_ctx or EngineCycleContext()

        while self._ports.is_running():
            t0 = time.monotonic()
            skip_sleep = False

            try:
                if state == EngineState.HALT:
                    self._ports.step_halt(ctx)
                    break

                handler = self._dispatch.get(state)
                if handler is None:
                    prev_state = state
                    state = EngineState.HALT
                    ctx.halt_reason = f"unknown_state:{prev_state}"
                    skip_sleep = True
                else:
                    decision = handler(state, ctx)
                    state, ctx, skip_sleep = (
                        decision.next_state,
                        decision.ctx,
                        decision.skip_sleep,
                    )
            except Exception as exc:
                decision = self._ports.on_exception(state, ctx, exc)
                state, ctx, skip_sleep = (
                    decision.next_state,
                    decision.ctx,
                    decision.skip_sleep,
                )

            if skip_sleep:
                continue

            dt = time.monotonic() - t0

            # DEFENSIVE CODE: non-critical reporting must never crash the cycle
            try:
                self._ports.report_cycle_latency_ms(float(dt * 1000.0))
            except Exception:
                pass  # original behavior preserved for stability

            # PERFORMANCE: avoid repeated max() calls
            min_cycle = max(0.02, float(self._ports.cycle_floor_sec()))
            sleep_duration = max(0.02, min_cycle - dt)
            time.sleep(sleep_duration)


# --- Stage services --------------------------------------------------------


class EngineFSMStageServices(IFSMEnginePorts):
    """
    Stateful adapters from FSM runtime ports to concrete engine behavior.
    This class intentionally keeps stage logic explicit and deterministic while
    preserving existing trading semantics.
    """

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine
        self._asset_priority = ("XAU", "BTC")
        self._ml_signal_log_ttl_sec = max(
            5.0,
            float(os.getenv("FSM_SIGNAL_LOG_TTL_SEC", "30.0") or 30.0),
        )
        self._ml_signal_log_cache: Dict[str, Tuple[str, float]] = {}

    def _asset_pipe(self, asset: str) -> Any:
        asset_u = str(asset or "").upper().strip()
        if asset_u == "XAU":
            return self._e._xau
        if asset_u == "BTC":
            return self._e._btc
        return None

    def _refresh_market_validation(self, asset: str) -> Tuple[bool, str]:
        pipe = self._asset_pipe(asset)
        if pipe is None:
            return False, "pipeline_missing"
        validate_fn = getattr(pipe, "validate_market_data", None)
        if not callable(validate_fn):
            return False, "market_validate_missing"
        try:
            if bool(validate_fn()):
                return True, "ok"
            return False, str(getattr(pipe, "last_market_reason", "") or "market_invalid")
        except Exception as exc:
            return False, f"market_validate_exception:{exc}"

    def _log_ml_signal(self, asset: str, sig: MLSignal) -> None:
        asset_key = str(asset).upper().strip()
        conf = round(float(sig.confidence or 0.0), 4)
        signature = "|".join(
            (
                str(sig.signal),
                f"{conf:.4f}",
                str(sig.provider),
                str(sig.model),
                str(sig.reason),
            )
        )
        now = time.time()
        last_signature, last_ts = self._ml_signal_log_cache.get(asset_key, ("", 0.0))
        if signature == last_signature and (now - last_ts) < self._ml_signal_log_ttl_sec:
            return
        self._ml_signal_log_cache[asset_key] = (signature, now)
        log_health.info(
            "FSM_ML_SIGNAL | asset=%s signal=%s conf=%.3f provider=%s model=%s reason=%s",
            asset,
            sig.signal,
            sig.confidence,
            sig.provider,
            sig.model,
            sig.reason,
        )

    def _apply_risk_halt(self, halt_reason: str) -> None:
        """# SAFE IMPROVEMENT: Extracted duplicated risk-halt logic (used in data_sync + execution_queue)."""
        with self._e._lock:
            self._e._manual_stop = True
            self._e._portfolio_stop_reason = str(halt_reason)

        self._e._drain_queue(self._e._order_q)
        self._e._drain_queue(self._e._result_q)

        with self._e._order_state_lock:
            self._e._order_rm_by_id.clear()
            self._e._pending_order_meta.clear()

    def is_running(self) -> bool:
        return bool(self._e._run.is_set())

    def cycle_floor_sec(self) -> float:
        return max(float(self._e._poll_fast), float(self._e._min_cycle_sec))

    def report_cycle_latency_ms(self, latency_ms: float) -> None:
        self._e._report_cycle_latency_ms(float(latency_ms))

    def step_boot(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        self._e._touch_runtime_progress()

        if self._e.manual_stop_active():
            log_health.info("FSM_MONITORING_BOOT | reason=manual_stop_active")

        if not self._e.dry_run and not (
            self._e._model_loaded and self._e._backtest_passed
        ):
            with self._e._lock:
                self._e._manual_stop = True
            log_health.warning(
                "FSM_MODEL_GATE_MONITORING_ONLY | reason=gatekeeper_failed execution_disabled=True analytics_alive=True"
            )

        if not self._e._check_mt5_health():
            ctx.halt_reason = "mt5_unhealthy"
            nxt = self._e._transition_state(state, EngineState.HALT, "mt5_unhealthy")
            return StepDecision(next_state=nxt, ctx=ctx)

        nxt = self._e._transition_state(state, EngineState.DATA_SYNC, "boot_ok")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_data_sync(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision:
        self._e._touch_runtime_progress()

        if not self._e._check_mt5_health():
            ctx.halt_reason = "mt5_disconnected"
            nxt = self._e._transition_state(state, EngineState.HALT, "mt5_disconnected")
            return StepDecision(next_state=nxt, ctx=ctx)

        self._e._check_hard_stop_file()

        if self._e.manual_stop_active():
            log_health.info(
                "FSM_MONITORING_CYCLE | reason=manual_stop_active execution_disabled=True"
            )

        # --- risk halt check (extracted) ---
        halt_reason = ""
        halt_fn = getattr(self._e, "_live_risk_halt_reason", None)
        if callable(halt_fn):
            try:
                halt_reason = str(halt_fn() or "")
            except Exception:
                halt_reason = ""

        if halt_reason:
            self._apply_risk_halt(halt_reason)
            log_health.warning(
                "FSM_RISK_MONITORING_ONLY | reason=%s execution_disabled=True analytics_alive=True",
                halt_reason,
            )

        if self._e._live_trading_pause_reason(force=False):
            log_health.warning(
                "FSM_LIVE_EVIDENCE_PAUSE | execution_disabled=True analytics_alive=True"
            )

        payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
        active_assets = self._ordered_active_assets(UTCScheduler.get_active_assets())
        data_sync_ok = True
        data_sync_reason = ""

        for asset in active_assets:
            self._e._touch_runtime_progress()
            if self._e._is_asset_blocked(asset):
                self._e._log_blocked_asset_skip(asset, "data_sync")
                continue

            market_ok, market_reason = self._refresh_market_validation(asset)
            if not market_ok:
                data_sync_ok = False
                data_sync_reason = f"{asset}_market_data_failed:{market_reason}"
                break

            asset_payloads = fetch_ml_payloads(asset)
            self._e._touch_runtime_progress()
            ok, reason = self._e._validate_data_sync_payloads(asset, asset_payloads)
            if not ok:
                data_sync_ok = False
                data_sync_reason = f"{asset}_data_sync_failed:{reason}"
                break
            payloads[asset] = asset_payloads

        if data_sync_ok and active_assets and not payloads:
            data_sync_ok = False
            data_sync_reason = "all_active_assets_blocked_by_gate"

        if data_sync_ok:
            nxt = self._e._transition_state(
                state, EngineState.ML_INFERENCE, "data_sync_ok"
            )
            return StepDecision(
                next_state=nxt, ctx=EngineCycleContext(payloads=payloads)
            )

        now = time.time()
        if (
            data_sync_reason != self._e._last_data_sync_fail_reason
            or (now - self._e._last_data_sync_fail_ts) > 5.0
        ):
            log_health.warning("DATA_SYNC_WAIT | reason=%s", data_sync_reason)
            self._e._last_data_sync_fail_reason = data_sync_reason
            self._e._last_data_sync_fail_ts = now

        return StepDecision(next_state=state, ctx=ctx)

    def _emit_analysis_signal(self, asset: str, sig: MLSignal) -> None:
        asset_u = str(asset).upper().strip()
        cfg = self._e._xau_cfg if asset_u == "XAU" else self._e._btc_cfg
        manual_watch_only = bool(getattr(self._e, "_user_manual_stop", False))
        live_ml_notify = bool(getattr(cfg, "analysis_notify_live_ml", True))
        sig_name = str(sig.signal or "").upper().strip()
        if not manual_watch_only:
            if not live_ml_notify:
                return
            if sig_name not in ("STRONG BUY", "STRONG SELL"):
                return

        side = str(sig.side if sig.side in ("Buy", "Sell") else sig.signal)
        if side not in ("Buy", "Sell"):
            return

        frame: Dict[str, Any] = {}
        try:
            frame = self._e._extract_payload_frame(sig.scalp_payload)
            if not frame:
                frame = self._e._extract_payload_frame(sig.intraday_payload)
        except Exception:
            frame = {}

        raw_bar = frame.get("ts_bar") or frame.get("t_close") or 0
        try:
            bar_key = str(int(float(raw_bar or 0)))
        except Exception:
            bar_key = str(int(time.time() // 60))

        if bar_key == "0":
            bar_key = str(int(time.time() // 60))

        signal_id = f"MLSIG_{asset_u}_{bar_key}_{side}"

        last_id, last_sig = self._e._analysis_signal_last_notified.get(
            asset_u, ("", "Neutral")
        )
        if last_id == signal_id and last_sig == side:
            return

        self._e._analysis_signal_last_notified[asset_u] = (signal_id, side)
        now_ts = time.time()
        try:
            self._e._record_signal_emit(asset_u, now_ts)
        except Exception:
            pass

        notifier = getattr(self._e, "_signal_notifier", None)
        if not notifier:
            return

        blocked_reason = ""
        if self._e.manual_stop_active():
            blocked_reason = str(
                self._e._unsafe_account_reason
                or self._e._portfolio_stop_reason
                or "manual_stop_active"
            )

        reasons = [
            f"ml_reason:{sig.reason}",
            f"ml_provider:{sig.provider}",
            f"ml_model:{sig.model}",
            "analysis_cycle",
        ]

        try:
            notifier(
                asset_u,
                {
                    "type": "ML_SIGNAL",
                    "asset": asset_u,
                    "signal": side,
                    "confidence": float(sig.confidence),
                    "reasons": reasons,
                    "signal_id": signal_id,
                    "lot": 0.0,
                    "phase": "MONITORING" if blocked_reason else "ANALYSIS",
                    "blocked": bool(blocked_reason),
                    "blocked_reason": blocked_reason,
                    "analysis_only": True,
                    "provider": str(sig.provider),
                    "model": str(sig.model),
                },
            )
        except Exception as exc:
            log_err.error(
                "ANALYSIS_SIGNAL_NOTIFY_ERROR | asset=%s signal_id=%s err=%s",
                asset_u,
                signal_id,
                exc,
            )

    def step_ml_inference(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision:
        self._e._touch_runtime_progress()
        signals: Dict[str, MLSignal] = {}

        for asset, payload in ctx.payloads.items():
            self._e._touch_runtime_progress()
            if self._e._is_asset_blocked(asset):
                self._e._log_blocked_asset_skip(asset, "ml_inference")
                continue

            sig = None
            has_catboost = asset in self._e._catboost_payloads

            if has_catboost:
                sig = self._e._infer_catboost(asset, payload)
                self._e._touch_runtime_progress()
                if sig is None:
                    # PATCH-4: LLM fallback disabled in live path (preserved)
                    log_health.info(
                        "CATBOOST_FAIL_HOLD | asset=%s reason=catboost_returned_none_llm_fallback_disabled",
                        asset,
                    )
                    sig = MLSignal(
                        asset=asset,
                        signal="HOLD",
                        side="Neutral",
                        confidence=0.0,
                        reason="catboost_inference_failed_llm_disabled",
                        provider="catboost",
                        model="catboost_trained",
                        entry=None,
                        stop_loss=None,
                        take_profit=None,
                        scalp_payload=payload.get("scalp"),
                        intraday_payload=payload.get("intraday"),
                    )
            else:
                # SAFE IMPROVEMENT: simplified else (was redundant "elif not has_catboost")
                sig = MLSignal(
                    asset=asset,
                    signal="HOLD",
                    side="Neutral",
                    confidence=0.0,
                    reason="catboost_payload_missing_no_fallback",
                    provider="catboost",
                    model="catboost_missing",
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    scalp_payload=payload.get("scalp"),
                    intraday_payload=payload.get("intraday"),
                )

            self._log_ml_signal(asset, sig)
            # NOTE (truthful-signal fix):
            # The Telegram "analysis signal" notifier used to fire HERE — right
            # after the ML inference but BEFORE the pipeline confluence check.
            # That produced the "СИГНАЛИ ТАҲЛИЛӢ" messages for signals that
            # were later silently rejected by the pipeline (e.g. tm=-1.00,
            # baseline_warmup, struct<thr, mr-contradiction). From the user's
            # perspective this looked like a stream of false/ghost signals.
            #
            # Fix: defer notification to `step_risk_calc`, and emit ONLY when
            # `_build_ml_candidate` returns a non-None candidate — i.e. the
            # pipeline confluence gate has already accepted the trade. See
            # `step_risk_calc` below.
            signals[asset] = sig

        ctx.ml_signals = signals
        nxt = self._e._transition_state(state, EngineState.RISK_CALC, "ml_complete")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_risk_calc(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision:
        self._e._touch_runtime_progress()
        candidates: List[AssetCandidate] = []

        for asset in ("XAU", "BTC"):
            self._e._touch_runtime_progress()
            sig = ctx.ml_signals.get(asset)
            if sig is None:
                continue
            cand = self._e._build_ml_candidate(asset, sig)
            if cand is not None:
                # ─── Truthful-signal notifier ───────────────────────────────
                # Pipeline confluence accepted the ML signal. Only NOW do we
                # notify Telegram with the "analysis signal" (the user will
                # subsequently see LIVE_ENQUEUED / order opened / order failed
                # updates as the execution layer drives the intent forward).
                # If the pipeline rejects the signal, no message is sent at
                # all — this is the definition of a confirmed (non-ghost)
                # signal.
                try:
                    self._emit_analysis_signal(asset, sig)
                except Exception as _emit_exc:
                    log_err.error(
                        "ANALYSIS_SIGNAL_EMIT_ERROR | asset=%s err=%s",
                        asset,
                        _emit_exc,
                    )
                candidates.append(cand)

        ctx.candidates = candidates
        self._e._last_cand_xau = next((c for c in candidates if c.asset == "XAU"), None)
        self._e._last_cand_btc = next((c for c in candidates if c.asset == "BTC"), None)

        nxt = self._e._transition_state(
            state, EngineState.EXECUTION_QUEUE, "risk_complete"
        )
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_execution_queue(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision:
        self._e._touch_runtime_progress()

        if self._e._retraining_mode:
            now = time.time()
            if now - self._e._last_retraining_log_ts > 10.0:
                log_health.info(
                    "RETRAINING_PAUSE | skipping new orders during retraining"
                )
                self._e._last_retraining_log_ts = now
            nxt = self._e._transition_state(
                state, EngineState.VERIFICATION, "retraining_pause"
            )
            return StepDecision(next_state=nxt, ctx=ctx)

        # --- risk halt check (extracted) ---
        halt_reason = ""
        halt_fn = getattr(self._e, "_live_risk_halt_reason", None)
        if callable(halt_fn):
            try:
                halt_reason = str(halt_fn() or "")
            except Exception:
                halt_reason = ""

        if halt_reason:
            self._apply_risk_halt(halt_reason)
            log_health.warning(
                "FSM_ORDER_RISK_BLOCK | reason=%s execution_disabled=True analytics_alive=True",
                halt_reason,
            )
            nxt = self._e._transition_state(
                state, EngineState.VERIFICATION, "risk_block_monitoring"
            )
            return StepDecision(next_state=nxt, ctx=ctx, skip_sleep=True)

        if self._e._live_trading_pause_reason(force=False):
            nxt = self._e._transition_state(
                state, EngineState.VERIFICATION, "live_evidence_pause"
            )
            return StepDecision(next_state=nxt, ctx=ctx)

        self._e._execute_candidates(ctx.candidates)
        self._e._touch_runtime_progress()

        nxt = self._e._transition_state(
            state, EngineState.VERIFICATION, "queue_complete"
        )
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_verification(
        self, state: EngineState, ctx: EngineCycleContext
    ) -> StepDecision:
        self._e._touch_runtime_progress()
        self._e._verification_step()
        self._e._touch_runtime_progress()

        nxt = self._e._transition_state(
            state, EngineState.DATA_SYNC, "verification_complete"
        )
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_halt(self, ctx: EngineCycleContext) -> None:
        self._e._halt_fsm(ctx.halt_reason or "unspecified")

    def on_exception(
        self,
        state: EngineState,
        ctx: EngineCycleContext,
        exc: Exception,
    ) -> StepDecision:
        log_err.error(
            "FSM_LOOP_EXCEPTION | err=%s | tb=%s", exc, traceback.format_exc()
        )
        ctx.halt_reason = f"fsm_exception:{exc}"
        return StepDecision(next_state=EngineState.HALT, ctx=ctx, skip_sleep=True)

    def _ordered_active_assets(self, assets: List[str]) -> List[str]:
        """Return active assets in deterministic execution order."""
        unique = {str(a).strip().upper() for a in assets if str(a).strip()}
        ordered: List[str] = [a for a in self._asset_priority if a in unique]
        remaining = sorted(a for a in unique if a not in set(self._asset_priority))
        ordered.extend(remaining)
        return ordered


# --- Engine mixin extracted from engine.py --------------------------------


RECOVERABLE_HALT_REASONS = frozenset({"mt5_unhealthy", "mt5_disconnected"})

MONITORING_ONLY_HALT_REASONS = frozenset(
    {
        "manual_stop",
        "manual_stop_active",
        "manual_stop_triggered",
    }
)


class EngineRuntimeMixin:
    def _reference_win_rate(self) -> float:
        vals: List[float] = []
        for asset in ("XAU", "BTC"):
            payload = self._catboost_payloads.get(asset)
            if not isinstance(payload, dict):
                continue
            cal = payload.get("alpha_calibration", {})
            if not isinstance(cal, dict):
                continue
            wr = float(cal.get("holdout_direction_accuracy_active", 0.0) or 0.0)
            if wr > 0.0:
                vals.append(wr)
        if not vals:
            return 0.0
        return float(min(vals))

    def _live_trade_win_rate(self) -> Tuple[float, int]:
        profits: List[float] = []
        for pipe in (self._xau, self._btc):
            rm = getattr(pipe, "risk", None) if pipe is not None else None
            fn = getattr(rm, "_recent_closed_trade_profits", None)
            if not callable(fn):
                continue
            try:
                p = fn(60)
                if isinstance(p, list):
                    profits.extend(float(x) for x in p if x is not None)
            except Exception:
                continue
        if not profits:
            return 0.0, 0
        wins = sum(1 for p in profits if float(p) > 0.0)
        n = int(len(profits))
        return float(wins / max(1, n)), n

    def _account_snapshot_reason(
        self, *, balance: float, equity: float, now: float
    ) -> str:
        try:
            bal = float(balance)
            eq = float(equity)
        except Exception:
            return "account_values_non_numeric"
        if not (bal > 0.0 and eq > 0.0):
            return f"account_values_invalid:balance={bal:.2f}|equity={eq:.2f}"
        prev_bal = float(self._last_account_snapshot.get("balance", 0.0) or 0.0)
        prev_eq = float(self._last_account_snapshot.get("equity", 0.0) or 0.0)
        prev_ts = float(self._last_account_snapshot.get("updated_ts", 0.0) or 0.0)
        factor = max(2.0, float(self._account_anomaly_factor or 50.0))
        if prev_ts > 0.0 and prev_bal > 0.0:
            bal_jump = max((bal / prev_bal), (prev_bal / bal))
            if bal_jump >= factor:
                return f"balance_jump:{prev_bal:.2f}->{bal:.2f}"
        if prev_ts > 0.0 and prev_eq > 0.0:
            eq_jump = max((eq / prev_eq), (prev_eq / eq))
            if eq_jump >= factor:
                return f"equity_jump:{prev_eq:.2f}->{eq:.2f}"
        self._last_account_snapshot = {
            "balance": bal,
            "equity": eq,
            "updated_ts": float(now),
        }
        return ""

    def _stable_account_snapshot(
        self, acc: Any, now: float
    ) -> Tuple[float, float, str]:
        bal = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0
        eq = float(getattr(acc, "equity", 0.0) or 0.0) if acc else 0.0
        reason = self._account_snapshot_reason(balance=bal, equity=eq, now=now)
        if not reason:
            return bal, eq, ""
        prev_bal = float(self._last_account_snapshot.get("balance", 0.0) or 0.0)
        prev_eq = float(self._last_account_snapshot.get("equity", 0.0) or 0.0)
        if prev_bal > 0.0 and prev_eq > 0.0:
            return prev_bal, prev_eq, ""
        return bal, eq, reason

    def _live_drawdown_limit(self) -> float:
        dd_candidates: List[float] = []
        for cfg in (getattr(self, "_xau_cfg", None), getattr(self, "_btc_cfg", None)):
            if cfg is None:
                continue
            try:
                candidate = float(getattr(cfg, "max_drawdown", 0.0) or 0.0)
            except Exception:
                candidate = 0.0
            if candidate > 0.0:
                dd_candidates.append(candidate)
        return float(min(dd_candidates)) if dd_candidates else 0.0

    def _live_open_positions_total(self) -> int:
        total = 0
        pipes = (
            ("XAU", getattr(self, "_xau", None)),
            ("BTC", getattr(self, "_btc", None)),
        )
        with self._lock:
            open_pos_snap = dict(self._current_open_positions)
        for asset, pipe in pipes:
            if pipe is None:
                total += max(0, int(open_pos_snap.get(asset, 0) or 0))
                continue
            try:
                total += max(0, int(pipe.open_positions()))
            except Exception:
                total += max(0, int(open_pos_snap.get(asset, 0) or 0))
        return int(total)

    def _live_risk_halt_reason(self, snapshot: Optional[Dict[str, Any]] = None) -> str:
        if self.dry_run or self._manual_stop:
            return ""
        snap = dict(snapshot or self._update_live_evidence(force=False))
        if str(snap.get("status", "") or "").upper() != "DEGRADED":
            return ""
        dd_limit = self._live_drawdown_limit()
        try:
            dd_pct = float(snap.get("dd_pct", 0.0) or 0.0)
        except Exception:
            dd_pct = 0.0
        if dd_limit <= 0.0 or dd_pct <= dd_limit:
            return ""
        open_total = self._live_open_positions_total()
        if open_total <= 0:
            return ""
        return f"live_drawdown_breach:{dd_pct:.3f}>{dd_limit:.3f}|open_positions:{open_total}"

    def _update_live_evidence(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (not force) and (
            (now - self._last_live_monitor_ts) < self._live_monitor_interval_sec
        ):
            return dict(self._live_evidence_state)

        self._last_live_monitor_ts = now

        runtime_age_sec = (
            max(0.0, now - float(self._loop_started_ts or 0.0))
            if self._loop_started_ts > 0
            else 0.0
        )
        latency_samples = max(
            len(self._exec_latency_ms_hist),
            len(self._fsm_cycle_ms_hist),
        )
        slippage_samples = len(self._exec_slippage_hist)

        telemetry_runtime_ready = (
            self._loop_started_ts > 0
            and runtime_age_sec >= self._live_evidence_grace_sec
        )
        latency_gate_ready = (
            telemetry_runtime_ready
            and latency_samples >= self._live_evidence_min_latency_samples
        )
        slippage_gate_ready = (
            telemetry_runtime_ready
            and slippage_samples >= self._live_evidence_min_latency_samples
        )

        lat_p95 = max(
            self._p95(self._exec_latency_ms_hist),
            self._p95(self._fsm_cycle_ms_hist),
        )
        slip_p95 = self._p95(self._exec_slippage_hist)

        risk_lat_candidates: List[float] = []
        risk_slip_candidates: List[float] = []
        for pipe in (self._xau, self._btc):
            rm = getattr(pipe, "risk", None) if pipe is not None else None
            if rm is None:
                continue
            try:
                if hasattr(rm, "exec_p95_latency"):
                    risk_lat_candidates.append(float(rm.exec_p95_latency() or 0.0))
                if hasattr(rm, "exec_p95_slippage"):
                    risk_slip_candidates.append(float(rm.exec_p95_slippage() or 0.0))
            except Exception:
                continue

        if risk_lat_candidates:
            lat_p95 = max(lat_p95, max(risk_lat_candidates))
        if risk_slip_candidates:
            slip_p95 = max(slip_p95, max(risk_slip_candidates))

        live_wr, wr_samples = self._live_trade_win_rate()
        ref_wr = self._reference_win_rate()
        drift = float(live_wr - ref_wr) if (live_wr > 0.0 and ref_wr > 0.0) else 0.0

        bal = 0.0
        eq = 0.0
        dd_pct = 0.0
        dd_limit = self._live_drawdown_limit()

        if not self.dry_run:
            try:
                with MT5_LOCK:
                    acc = mt5.account_info()
            except Exception:
                acc = None
            bal, eq, account_reason = self._stable_account_snapshot(acc, now)
            if bal > 0.0:
                dd_pct = max(0.0, (bal - eq) / bal)
        else:
            account_reason = ""

        reasons: List[str] = []
        if account_reason:
            reasons.append(account_reason)

        with self._lock:
            open_positions_snap = dict(self._current_open_positions)
        for asset, open_now in open_positions_snap.items():
            max_pos = int(self._max_open_positions.get(asset, 0) or 0)
            if max_pos > 0 and int(open_now) > max_pos:
                reasons.append(f"open_positions:{asset}:{int(open_now)}>{max_pos}")

        if latency_gate_ready and lat_p95 > self._live_max_p95_latency_ms:
            reasons.append(
                f"latency_p95:{lat_p95:.1f}>{self._live_max_p95_latency_ms:.1f}"
            )
        if slippage_gate_ready and slip_p95 > self._live_max_p95_slippage_points:
            reasons.append(
                f"slippage_p95:{slip_p95:.2f}>{self._live_max_p95_slippage_points:.2f}"
            )
        if wr_samples >= 8 and ref_wr > 0.0 and drift < -self._live_max_winrate_drift:
            reasons.append(f"winrate_drift:{drift:+.3f}")
        if dd_limit > 0.0 and dd_pct > dd_limit:
            reasons.append(f"dd_pct:{dd_pct:.3f}>{dd_limit:.3f}")

        status = "HEALTHY" if not reasons else "DEGRADED"
        if self._manual_stop:
            status = "MONITORING_ONLY"
        elif not self._mt5_ready and not self.dry_run:
            status = "MT5_OFFLINE"

        self._prune_signal_window(self._signal_emit_ts_global, now)

        # Statistical credibility: Wilson score confidence interval for win rate.
        _n = max(1, int(wr_samples))
        _p = float(live_wr)
        _z = 1.96  # 95% CI
        _denom = 1 + (_z * _z / _n)
        _centre = (_p + _z * _z / (2 * _n)) / _denom
        _margin = (_z / _denom) * _math.sqrt(
            (_p * (1 - _p) / _n) + (_z * _z / (4 * _n * _n))
        )
        wr_ci_lo = max(0.0, _centre - _margin)
        wr_ci_hi = min(1.0, _centre + _margin)

        snapshot = {
            "status": status,
            "reason": "|".join(reasons) if reasons else "ok",
            "updated_ts": now,
            "latency_p95_ms": float(lat_p95),
            "slippage_p95_points": float(slip_p95),
            "win_rate_live": float(live_wr),
            "win_rate_reference": float(ref_wr),
            "win_rate_drift": float(drift),
            "closed_trade_samples": int(wr_samples),
            "signals_24h": int(len(self._signal_emit_ts_global)),
            "dd_pct": float(dd_pct),
            "balance": float(bal),
            "equity": float(eq),
            "runtime_age_sec": round(runtime_age_sec, 1),
            "latency_samples": int(latency_samples),
            "slippage_samples": int(slippage_samples),
            "telemetry_warmup": bool(not (latency_gate_ready and slippage_gate_ready)),
            "wr_ci_95_lo": round(wr_ci_lo, 4),
            "wr_ci_95_hi": round(wr_ci_hi, 4),
        }

        self._live_evidence_state = snapshot

        # PATCH-2: Zero-signal alarm
        _ZERO_SIGNAL_ALARM_SEC = float(
            os.getenv("ZERO_SIGNAL_ALARM_SEC", "14400") or 14400
        )
        if (
            snapshot["signals_24h"] == 0
            and self._loop_started_ts > 0
            and (now - self._loop_started_ts) > _ZERO_SIGNAL_ALARM_SEC
            and not self._manual_stop
            and not self.dry_run
        ):
            reasons.append("zero_signals_extended")
            if status == "HEALTHY":
                status = "DEGRADED"
                snapshot["status"] = status
                snapshot["reason"] = "|".join(reasons) if reasons else "ok"

        # PATCH-3: Low-sample confidence flag
        _MIN_TRADES_RELIABLE = 100
        if snapshot["closed_trade_samples"] < _MIN_TRADES_RELIABLE:
            snapshot["win_rate_confidence"] = "LOW_SAMPLE"
        else:
            snapshot["win_rate_confidence"] = "RELIABLE"

        level = log_health.warning if reasons else log_health.info
        level(
            "LIVE_EVIDENCE | status=%s reason=%s latency_p95=%.1f slippage_p95=%.2f "
            "win_rate_live=%.3f win_rate_ref=%.3f drift=%+.3f dd_pct=%.3f "
            "closed_trades=%d signals_24h=%d wr_confidence=%s",
            snapshot["status"],
            snapshot["reason"],
            snapshot["latency_p95_ms"],
            snapshot["slippage_p95_points"],
            snapshot["win_rate_live"],
            snapshot["win_rate_reference"],
            snapshot["win_rate_drift"],
            snapshot["dd_pct"],
            snapshot["closed_trade_samples"],
            snapshot["signals_24h"],
            snapshot.get("win_rate_confidence", "UNKNOWN"),
        )
        return dict(snapshot)

    def _live_trading_pause_reason(self, *, force: bool = False) -> str:
        if self.dry_run:
            return ""
        snapshot = self._update_live_evidence(force=force)
        if str(snapshot.get("status", "") or "").upper() != "DEGRADED":
            return ""
        reason = str(snapshot.get("reason", "degraded") or "degraded")
        now = time.time()
        if (
            reason != self._last_live_pause_reason
            or (now - self._last_live_pause_log_ts) >= 5.0
        ):
            self._last_live_pause_reason = reason
            self._last_live_pause_log_ts = now
            log_health.warning("LIVE_TRADE_PAUSE | reason=%s", reason)
        return reason

    def _heartbeat(self, open_xau: int, open_btc: int) -> None:
        try:
            self._mark_runtime_alive()
            with self._lock:
                self._current_open_positions["XAU"] = int(open_xau)
                self._current_open_positions["BTC"] = int(open_btc)
            now = time.time()
            with MT5_LOCK:
                acc = mt5.account_info()
            bal, eq, _ = self._stable_account_snapshot(acc, now)
            dd = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0
            pnl = float(getattr(acc, "profit", 0.0) or 0.0) if acc else 0.0
            try:
                p1 = self._xau.risk if self._xau else None
                p2 = self._btc.risk if self._btc else None
                v1 = float(getattr(p1, "today_pnl", 0.0) or 0.0) if p1 else 0.0
                v2 = float(getattr(p2, "today_pnl", 0.0) or 0.0) if p2 else 0.0
                if (p1 and hasattr(p1, "today_pnl")) or (
                    p2 and hasattr(p2, "today_pnl")
                ):
                    pnl = float(v1 + v2)
            except Exception:
                pass

            last_sig_x = self._xau.last_signal if self._xau else "?"
            last_sig_b = self._btc.last_signal if self._btc else "?"
            st = PortfolioStatus(
                connected=bool(self._mt5_ready),
                trading=bool(self._run.is_set()),
                manual_stop=bool(self._manual_stop),
                active_asset=str(self._active_asset),
                balance=bal,
                equity=eq,
                dd_pct=float(dd),
                today_pnl=float(pnl),
                open_trades_total=int(open_xau + open_btc),
                open_trades_xau=int(open_xau),
                open_trades_btc=int(open_btc),
                last_signal_xau=str(last_sig_x),
                last_signal_btc=str(last_sig_b),
                last_selected_asset=str(self._last_selected_asset),
                exec_queue_size=int(self._total_exec_queue_size()),
                last_reconcile_ts=float(self._last_reconcile_ts),
            )
            live_evidence = self._update_live_evidence(force=False)
            if _DIAG_ENABLED and (time.time() - self._diag_last_ts) >= float(
                _DIAG_EVERY_SEC
            ):
                self._diag_last_ts = time.time()
                payload = {
                    "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "engine": st,
                    "mt5": mt5_status(),
                    "live_evidence": live_evidence,
                    "runtime": self.runtime_watchdog_snapshot(),
                    "account_state": dict(self._unsafe_account_snapshot),
                }
                try:
                    log_diag.info(safe_json_dumps(payload))
                except Exception:
                    pass
        except Exception as exc:
            log_err.error("heartbeat error: %s | tb=%s", exc, traceback.format_exc())

    def _track_position_closures(self, open_xau: int, open_btc: int) -> None:
        now = time.time()
        for asset, open_now in (("XAU", int(open_xau)), ("BTC", int(open_btc))):
            prev = int(self._last_open_positions.get(asset, 0))
            if prev > open_now:
                self._last_trade_close_ts[asset] = now
                log_health.info(
                    "TRADE_CLOSE_DETECTED | asset=%s prev_open=%d open=%d cooldown=%.0fs",
                    asset,
                    prev,
                    open_now,
                    self._trade_cooldown_sec,
                )
            self._last_open_positions[asset] = open_now

    def status(self) -> PortfolioStatus:
        runtime_snapshot = self.runtime_watchdog_snapshot()
        controller = self.controller_snapshot()
        trading = bool(runtime_snapshot.get("trading_ok", False))
        if self.dry_run:
            connected = True
            with self._lock:
                bal = float(self._last_account_snapshot.get("balance", 0.0) or 0.0)
                eq = float(self._last_account_snapshot.get("equity", 0.0) or 0.0)
            if bal <= 0.0:
                bal = 10000.0
            if eq <= 0.0:
                eq = bal
            pnl = 0.0
            dd = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0
        else:
            try:
                with MT5_LOCK:
                    term = mt5.terminal_info()
                    acc = mt5.account_info()
            except Exception:
                term = None
                acc = None
            identity_ok, _identity_reason = self._mt5_identity_ok(acc)
            connected = bool(
                self._mt5_ready
                and term
                and getattr(term, "connected", False)
                and identity_ok
            )
            bal, eq, _ = self._stable_account_snapshot(acc, time.time())
            pnl = float(getattr(acc, "profit", 0.0) or 0.0) if acc else 0.0
            dd = max(0.0, (bal - eq) / bal) if bal > 0 else 0.0
        open_xau = self._xau.open_positions() if self._xau else 0
        open_btc = self._btc.open_positions() if self._btc else 0
        return PortfolioStatus(
            connected=connected,
            trading=trading,
            manual_stop=bool(self._manual_stop),
            active_asset=str(self._active_asset),
            balance=bal,
            equity=eq,
            dd_pct=float(dd),
            today_pnl=float(pnl),
            open_trades_total=int(open_xau + open_btc),
            open_trades_xau=int(open_xau),
            open_trades_btc=int(open_btc),
            last_signal_xau=str(self._xau.last_signal if self._xau else "Neutral"),
            last_signal_btc=str(self._btc.last_signal if self._btc else "Neutral"),
            last_selected_asset=str(self._last_selected_asset),
            exec_queue_size=int(self._total_exec_queue_size()),
            last_reconcile_ts=float(self._last_reconcile_ts),
            controller_state=str(
                controller.get("controller_state", "stopped") or "stopped"
            ),
            risk_halt_reason=str(controller.get("risk_halt_reason", "") or ""),
            gate_reason=str(controller.get("gate_reason", "") or ""),
            chaos_state=str(controller.get("chaos_state", "unknown") or "unknown"),
            blocked_assets=tuple(controller.get("blocked_assets", ())),
        )

    def _update_portfolio_risk_state(self) -> None:
        try:
            if self.dry_run:
                from types import SimpleNamespace

                acc = SimpleNamespace(equity=10000.0, leverage=200)
                positions = []
            else:
                with MT5_LOCK:
                    acc = mt5.account_info()
                    positions = mt5.positions_get()
            if not acc:
                return
            equity = float(getattr(acc, "equity", 0.0))
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            account_state = self._inspect_live_account_positions(positions)
            unsafe_reason = str(account_state.get("reason", "") or "")
            should_quarantine = False
            should_log_unsafe = False
            with self._lock:
                prev_unsafe_reason = str(self._unsafe_account_reason or "")
                self._unsafe_account_snapshot = dict(account_state)
                self._unsafe_account_reason = unsafe_reason
                if unsafe_reason:
                    should_log_unsafe = unsafe_reason != prev_unsafe_reason
                    if not self._manual_stop:
                        self._manual_stop = True
                        should_quarantine = True
            self._portfolio_risk.set_daily_start_equity(equity, date_str)
            hard_stopped, stop_reason = self._portfolio_risk.evaluate_equity(equity)
            exposures = {
                "XAU": AssetExposure(symbol="XAUUSDm"),
                "BTC": AssetExposure(symbol="BTCUSDm"),
            }

            def _get_asset(sym: str):
                s = sym.upper()
                if "BTC" in s:
                    return "BTC"
                if "XAU" in s or "GOLD" in s:
                    return "XAU"
                return None

            if positions:
                for p in positions:
                    asset = _get_asset(p.symbol)
                    if asset and asset in exposures:
                        exp = exposures[asset]
                        exp.symbol = p.symbol
                        vol = p.volume
                        p_type = p.type
                        exp.position_count += 1
                        margin_used = float(getattr(p, "margin", 0.0) or 0.0)
                        if margin_used <= 0.0:
                            price_ref = float(
                                getattr(p, "price_current", 0.0)
                                or getattr(p, "price_open", 0.0)
                                or 0.0
                            )
                            lev = float(getattr(acc, "leverage", 0.0) or 0.0)
                            if price_ref > 0.0 and lev > 0.0:
                                cfg = self._xau_cfg if asset == "XAU" else self._btc_cfg
                                contract_size = float(
                                    getattr(
                                        getattr(cfg, "symbol_params", None),
                                        "contract_size",
                                        1.0,
                                    )
                                    or 1.0
                                )
                                margin_used = (
                                    price_ref * float(vol) * contract_size
                                ) / max(lev, 1.0)
                        exp.margin_used += max(0.0, float(margin_used))
                        exp.unrealized_pnl += p.profit
                        sl_price = float(getattr(p, "sl", 0.0) or 0.0)
                        price_ref = float(
                            getattr(p, "price_current", 0.0)
                            or getattr(p, "price_open", 0.0)
                            or 0.0
                        )
                        if sl_price > 0.0 and price_ref > 0.0:
                            cfg = self._xau_cfg if asset == "XAU" else self._btc_cfg
                            contract_size = float(
                                getattr(
                                    getattr(cfg, "symbol_params", None),
                                    "contract_size",
                                    1.0,
                                )
                                or 1.0
                            )
                            exp.open_risk += max(
                                0.0,
                                abs(price_ref - sl_price) * float(vol) * contract_size,
                            )
                        signed_vol = vol if p_type == 0 else -vol
                        exp.volume += signed_vol
            for asset, exp in exposures.items():
                side = "Neutral"
                if exp.volume > 0.000001:
                    side = "Buy"
                elif exp.volume < -0.000001:
                    side = "Sell"
                exp.side = side
                exp.volume = abs(exp.volume)
                self._portfolio_risk.update_exposure(asset, exp)
            if unsafe_reason:
                if should_log_unsafe:
                    issue_sample = list(account_state.get("position_issues", []) or [])[
                        :5
                    ]
                    grace_sample = list(account_state.get("grace_positions", []) or [])[
                        :5
                    ]
                    log_err.critical(
                        "UNSAFE_ACCOUNT_STATE_RUNTIME | reason=%s issues=%s grace=%s",
                        unsafe_reason,
                        safe_json_dumps(issue_sample) if issue_sample else "-",
                        safe_json_dumps(grace_sample) if grace_sample else "-",
                    )
                if should_quarantine:
                    self._drain_queue(self._order_q)
                    self._drain_queue(self._result_q)
                    with self._order_state_lock:
                        self._order_rm_by_id.clear()
                        self._pending_order_meta.clear()
            if hard_stopped:
                should_flatten = False
                with self._lock:
                    if not self._portfolio_stop_triggered or str(
                        self._portfolio_stop_reason or ""
                    ) != str(stop_reason or ""):
                        self._portfolio_stop_triggered = True
                        self._portfolio_stop_reason = str(stop_reason or "")
                        self._manual_stop = True
                        should_flatten = True
                if should_flatten:
                    log_err.critical(
                        "PORTFOLIO_RUNTIME_HARD_STOP | reason=%s equity=%.2f",
                        stop_reason,
                        equity,
                    )
                    self._drain_queue(self._order_q)
                    self._drain_queue(self._result_q)
                    with self._order_state_lock:
                        self._order_rm_by_id.clear()
                        self._pending_order_meta.clear()
                    self._emergency_flatten(f"portfolio_hard_stop:{stop_reason}")
            else:
                with self._lock:
                    self._portfolio_stop_triggered = False
                    self._portfolio_stop_reason = ""
        except Exception as exc:
            log_err.error("portfolio update error: %s", exc)

    def _mark_runtime_alive(self) -> None:
        now = time.time()
        with self._lock:
            self._last_runtime_heartbeat_ts = now
        # ─── External watchdog heartbeat (Section 10) ───────────────────
        # Emit a JSON heartbeat to disk so that scripts/watchdog.py can
        # detect and force-kill the process if the engine loop stalls.
        # Defensive: writes are O(1) amortised and wrapped so that a
        # broken heartbeat file never stops the trading loop.
        try:
            from core.stability_monitor import get_default_heartbeat

            get_default_heartbeat().beat(status="alive", note="engine_loop")
        except Exception:
            pass

    def _touch_runtime_progress(self) -> None:
        self._mark_runtime_alive()

    def runtime_watchdog_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            running = bool(self._run.is_set())
            manual_stop = bool(self._manual_stop)
            loop_thread = self._loop_thread
            loop_started_ts = float(self._loop_started_ts or 0.0)
            last_heartbeat_ts = float(self._last_runtime_heartbeat_ts or 0.0)
            completed_cycles = int(len(self._fsm_cycle_ms_hist))
        loop_alive = bool(loop_thread and loop_thread.is_alive())
        now = time.time()
        last_activity_ts = (
            last_heartbeat_ts if last_heartbeat_ts > 0.0 else loop_started_ts
        )
        heartbeat_age_sec = (
            max(0.0, now - last_activity_ts) if last_activity_ts > 0.0 else 0.0
        )
        starting = bool(
            running
            and loop_alive
            and last_heartbeat_ts <= 0.0
            and loop_started_ts > 0.0
            and (now - loop_started_ts) <= self._runtime_start_grace_sec
        )
        bootstrapping = bool(
            running
            and loop_alive
            and completed_cycles <= 0
            and loop_started_ts > 0.0
            and (now - loop_started_ts) <= self._runtime_first_cycle_grace_sec
        )
        stale = bool(
            running
            and loop_alive
            and not starting
            and not bootstrapping
            and last_activity_ts > 0.0
            and heartbeat_age_sec > self._runtime_stall_timeout_sec
        )
        thread_dead = bool(running and loop_started_ts > 0.0 and not loop_alive)
        trading_ok = bool(running and loop_alive and not stale)
        return {
            "running": running,
            "loop_alive": loop_alive,
            "starting": starting,
            "bootstrapping": bootstrapping,
            "stale": stale,
            "thread_dead": thread_dead,
            "heartbeat_age_sec": float(heartbeat_age_sec),
            "last_heartbeat_ts": float(last_heartbeat_ts),
            "loop_started_ts": float(loop_started_ts),
            "manual_stop": manual_stop,
            "trading_ok": trading_ok,
        }

    def unsafe_account_state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._unsafe_account_snapshot)

    def _log_growth_snapshot(self) -> Dict[str, Any]:
        candidate_roots = [Path("logs"), Path("Logs")]
        roots: List[Path] = []
        seen_roots: set[str] = set()
        for root in candidate_roots:
            try:
                if not root.exists():
                    continue
                root_key = str(root.resolve()).lower()
            except Exception:
                root_key = str(root).lower()
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            roots.append(root)
        total_bytes = 0
        oversized_files: List[str] = []
        tracked_files = 0
        seen_files: set[str] = set()
        for root in roots:
            for pattern in ("*.log", "*.jsonl", "*.csv"):
                for path in root.rglob(pattern):
                    try:
                        file_key = str(path.resolve()).lower()
                    except Exception:
                        file_key = str(path).lower()
                    if file_key in seen_files:
                        continue
                    seen_files.add(file_key)
                    try:
                        size = int(path.stat().st_size)
                    except Exception:
                        continue
                    tracked_files += 1
                    total_bytes += size
                    if size > self._log_growth_limit_bytes:
                        oversized_files.append(f"{path}:{size}")
        return {
            "tracked_files": tracked_files,
            "total_bytes": total_bytes,
            "oversized_files": tuple(sorted(oversized_files)),
            "limit_bytes": int(self._log_growth_limit_bytes),
        }

    def manage_runtime_housekeeping(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (
            not force
            and (now - self._last_housekeeping_ts) < self._housekeeping_interval_sec
        ):
            return {}
        self._last_housekeeping_ts = now
        snapshot = self._log_growth_snapshot()
        oversized = tuple(snapshot.get("oversized_files", ()))
        if oversized:
            log_health.warning(
                "LOG_GROWTH_PRESSURE | count=%d limit=%d",
                len(oversized),
                int(snapshot.get("limit_bytes", 0) or 0),
            )
        return snapshot

    def run_chaos_audit(self, *, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        cached_ts = float(self._last_chaos_audit.get("timestamp", 0.0) or 0.0)
        if self._last_chaos_audit and not force and (now - cached_ts) < 5.0:
            return dict(self._last_chaos_audit)
        unsafe = self.unsafe_account_state_snapshot()
        housekeeping = self.manage_runtime_housekeeping(force=force)

        def _p95(values: Deque[float]) -> float:
            arr = sorted(float(v) for v in values if float(v) >= 0.0)
            if not arr:
                return 0.0
            idx = min(len(arr) - 1, max(0, int(round(0.95 * (len(arr) - 1)))))
            return float(arr[idx])

        def _pipe_metrics(pipe: Any, default_symbol: str) -> Dict[str, Any]:
            if pipe is None:
                return {
                    "market_ok": False,
                    "reason": "pipeline_missing",
                    "rows": 0,
                    "tick_age": float("inf"),
                    "symbol": default_symbol,
                    "last_signal": "Neutral",
                    "signal_ts": 0.0,
                }
            validate_fn = getattr(pipe, "validate_market_data", None)
            if callable(validate_fn):
                try:
                    validate_fn()
                except Exception:
                    pass
            return {
                "market_ok": bool(getattr(pipe, "last_market_ok", False)),
                "reason": str(getattr(pipe, "last_market_reason", "") or ""),
                "rows": int(getattr(pipe, "last_market_rows", 0) or 0),
                "tick_age": float(
                    getattr(pipe, "last_tick_age_sec", 0.0)
                    or getattr(pipe, "last_bar_age_sec", 0.0)
                    or 0.0
                ),
                "symbol": str(
                    getattr(pipe, "symbol", default_symbol) or default_symbol
                ),
                "last_signal": str(
                    getattr(pipe, "last_signal", "Neutral") or "Neutral"
                ),
                "signal_ts": float(getattr(pipe, "last_signal_ts", 0.0) or 0.0),
            }

        xau_metrics = _pipe_metrics(self._xau, "XAUUSDm")
        btc_metrics = _pipe_metrics(self._btc, "BTCUSDm")
        recent_signal_ts = [
            float(ts) for ts in self._signal_emit_ts_global if (now - float(ts)) <= 60.0
        ]
        simultaneous_dual_trigger = bool(
            xau_metrics["last_signal"] != "Neutral"
            and btc_metrics["last_signal"] != "Neutral"
            and abs(float(xau_metrics["signal_ts"]) - float(btc_metrics["signal_ts"]))
            <= 2.0
        )

        scenarios: Dict[str, Dict[str, Any]] = {}

        def _record(
            name: str, passed: bool, detail: str, *, recoverable: bool = True
        ) -> None:
            scenarios[name] = {
                "passed": bool(passed),
                "recoverable": bool(recoverable),
                "detail": str(detail or ""),
            }

        _record(
            "MT5 disconnect",
            bool(self.dry_run or self._mt5_ready),
            "ready" if (self.dry_run or self._mt5_ready) else "mt5_not_ready",
        )
        _record(
            "internet delay",
            _p95(self._exec_latency_ms_hist)
            <= float(self._live_max_p95_latency_ms or 0.0),
            f"p95_ms={_p95(self._exec_latency_ms_hist):.1f}/{self._live_max_p95_latency_ms:.1f}",
        )
        uptime_sec = max(0.0, now - float(self._boot_ts or now))
        feed_warmup_sec = max(
            60.0,
            float(getattr(self._xau_cfg, "market_min_bar_age_sec", 30.0) or 30.0),
            float(getattr(self._btc_cfg, "market_min_bar_age_sec", 30.0) or 30.0),
        )
        feed_warmup_active = uptime_sec < feed_warmup_sec

        stale_feed_passed = bool(xau_metrics["market_ok"] and btc_metrics["market_ok"])
        stale_feed_detail = f"xau={xau_metrics['reason']} btc={btc_metrics['reason']}"
        if feed_warmup_active and not stale_feed_passed:
            stale_feed_passed = True
            stale_feed_detail = (
                f"warmup:{uptime_sec:.1f}/{feed_warmup_sec:.1f}s {stale_feed_detail}"
            )
        _record(
            "stale feed",
            stale_feed_passed,
            stale_feed_detail,
        )

        missing_bars_passed = bool(xau_metrics["rows"] >= 30 and btc_metrics["rows"] >= 30)
        missing_bars_detail = (
            f"xau_rows={xau_metrics['rows']} btc_rows={btc_metrics['rows']}"
        )
        if feed_warmup_active and not missing_bars_passed:
            missing_bars_passed = True
            missing_bars_detail = (
                f"warmup:{uptime_sec:.1f}/{feed_warmup_sec:.1f}s {missing_bars_detail}"
            )
        _record(
            "missing bars",
            missing_bars_passed,
            missing_bars_detail,
        )
        _record(
            "spread spike",
            _p95(self._exec_slippage_hist)
            <= float(self._live_max_p95_slippage_points or 0.0),
            f"p95_slippage={_p95(self._exec_slippage_hist):.2f}/{self._live_max_p95_slippage_points:.2f}",
        )
        gap_reason = f"{getattr(self._xau, 'risk', None).hard_stop_reason if self._xau else ''}|{getattr(self._btc, 'risk', None).hard_stop_reason if self._btc else ''}"
        _record(
            "sudden gap",
            "gap" not in gap_reason.lower(),
            gap_reason or "clear",
        )
        _record(
            "slow model response",
            _p95(self._fsm_cycle_ms_hist)
            <= max(2.0 * self._live_max_p95_latency_ms, 2500.0),
            f"p95_cycle_ms={_p95(self._fsm_cycle_ms_hist):.1f}",
        )
        _record(
            "duplicate signal storm",
            len(recent_signal_ts) <= max(20, int(self._target_signals_per_24h_max * 2)),
            f"signals_last_min={len(recent_signal_ts)}",
        )
        _record(
            "simultaneous XAU + BTC triggers",
            not simultaneous_dual_trigger,
            "dual_trigger_detected" if simultaneous_dual_trigger else "clear",
        )
        cfg_valid = bool(
            getattr(getattr(self._xau_cfg, "symbol_params", None), "contract_size", 0.0)
            and getattr(
                getattr(self._btc_cfg, "symbol_params", None), "contract_size", 0.0
            )
        )
        _record(
            "corrupted config",
            cfg_valid,
            "config_ok" if cfg_valid else "invalid_contract_size",
            recoverable=False,
        )
        symbol_info_valid = bool(
            xau_metrics["symbol"]
            and btc_metrics["symbol"]
            and int(
                getattr(getattr(self._xau_cfg, "symbol_params", None), "digits", 0) or 0
            )
            > 0
            and int(
                getattr(getattr(self._btc_cfg, "symbol_params", None), "digits", 0) or 0
            )
            > 0
        )
        _record(
            "invalid symbol info",
            symbol_info_valid,
            "symbol_info_ok" if symbol_info_valid else "symbol_info_invalid",
            recoverable=False,
        )
        restart_ready = bool(
            callable(getattr(self, "_recover_all", None))
            and callable(getattr(self, "_restart_exec_worker", None))
        )
        _record(
            "process restart mid-position",
            restart_ready,
            "restart_hooks_ready" if restart_ready else "restart_hooks_missing",
        )
        log_growth_ok = not bool(tuple(housekeeping.get("oversized_files", ())))
        _record(
            "log file growth",
            log_growth_ok,
            "within_limit" if log_growth_ok else "oversized_logs_detected",
        )
        terminal_restart_ready = bool(
            callable(getattr(self, "_emergency_flatten", None)) and restart_ready
        )
        _record(
            "terminal restart during open trade",
            terminal_restart_ready,
            "recovery_ready" if terminal_restart_ready else "recovery_missing",
        )
        if unsafe.get("reason"):
            _record(
                "invalid live account state",
                False,
                str(unsafe.get("reason") or ""),
                recoverable=False,
            )
        overall_passed = bool(
            all(item.get("passed", False) for item in scenarios.values())
        )
        snapshot = {
            "timestamp": now,
            "overall_passed": overall_passed,
            "scenarios": scenarios,
        }
        self._last_chaos_audit = dict(snapshot)
        return snapshot

    def controller_snapshot(self) -> Dict[str, Any]:
        runtime = self.runtime_watchdog_snapshot()
        chaos = self.run_chaos_audit(force=False)
        risk_halt_reason = str(
            self._portfolio_stop_reason or self._unsafe_account_reason or ""
        )
        if bool(runtime.get("trading_ok", False)) and not bool(self._manual_stop):
            controller_state = "running"
        elif bool(self._manual_stop):
            controller_state = "monitoring"
        elif bool(runtime.get("starting", False)):
            controller_state = "starting"
        else:
            controller_state = "stopped"
        return {
            "controller_state": controller_state,
            "runtime": runtime,
            "risk_halt_reason": risk_halt_reason,
            "gate_reason": str(self._gate_last_reason or ""),
            "blocked_assets": tuple(sorted(set(self._blocked_assets or []))),
            "chaos_state": (
                "pass" if bool(chaos.get("overall_passed", False)) else "warn"
            ),
            "chaos_audit": chaos,
            "uptime_sec": max(0.0, time.time() - float(self._boot_ts or time.time())),
        }

    def _preflight_live_account_state(self) -> str:
        if self.dry_run:
            return ""
        snapshot = self._inspect_live_account_positions()
        reason = str(snapshot.get("reason", "") or "")
        with self._lock:
            self._unsafe_account_snapshot = dict(snapshot)
            self._unsafe_account_reason = reason
        return reason

    def _inspect_live_account_positions(
        self, positions: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "reason": "",
                "total_positions": 0,
                "symbol_counts": {},
                "foreign_symbols": [],
                "unmanaged_positions": 0,
                "zero_protection_positions": 0,
                "magic_counts": {},
                "position_issues": [],
                "grace_positions": [],
            }
        if positions is None:
            try:
                with MT5_LOCK:
                    positions = mt5.positions_get() or []
            except Exception as exc:
                return {
                    "reason": f"positions_probe_failed:{exc}",
                    "total_positions": 0,
                    "symbol_counts": {},
                    "foreign_symbols": [],
                    "unmanaged_positions": 0,
                    "zero_protection_positions": 0,
                    "magic_counts": {},
                    "position_issues": [],
                    "grace_positions": [],
                }
        total_positions = 0
        symbol_counts: Dict[str, int] = {}
        foreign_symbols: List[str] = []
        magic_counts: Dict[int, int] = {}
        position_issues: List[Dict[str, Any]] = []
        grace_positions: List[Dict[str, Any]] = []
        unmanaged_positions = 0
        zero_protection_positions = 0
        now_ts = time.time()
        grace_sec = max(
            0.0, float(getattr(self, "_unsafe_position_grace_sec", 20.0) or 20.0)
        )
        known_symbols = {
            str(getattr(self._xau, "symbol", "XAUUSDm") or "XAUUSDm").upper(): "XAU",
            str(getattr(self._btc, "symbol", "BTCUSDm") or "BTCUSDm").upper(): "BTC",
        }
        expected_magics = {
            int(getattr(self._xau_cfg, "magic", 777001) or 777001),
            int(getattr(self._btc_cfg, "magic", 777001) or 777001),
            777001,
        }

        def _position_age_sec(pos: Any) -> Optional[float]:
            raw = None
            for attr in ("time_msc", "time_update_msc", "time", "time_update"):
                raw = getattr(pos, attr, None)
                if raw:
                    break
            try:
                epoch = float(raw or 0.0)
            except Exception:
                return None
            if epoch <= 0.0:
                return None
            if epoch > 10_000_000_000.0:
                epoch /= 1000.0
            return max(0.0, now_ts - epoch)

        for pos in positions or []:
            try:
                ticket = int(getattr(pos, "ticket", 0) or 0)
                if ticket <= 0:
                    continue
                total_positions += 1
                symbol = str(getattr(pos, "symbol", "") or "").strip().upper()
                asset = str(known_symbols.get(symbol, "") or "")
                if symbol:
                    symbol_counts[symbol] = int(symbol_counts.get(symbol, 0) or 0) + 1
                magic = int(getattr(pos, "magic", 0) or 0)
                magic_counts[magic] = int(magic_counts.get(magic, 0) or 0) + 1
                sl = float(getattr(pos, "sl", 0.0) or 0.0)
                tp = float(getattr(pos, "tp", 0.0) or 0.0)
                is_foreign_symbol = bool(symbol and symbol not in known_symbols)
                is_expected_magic = bool(magic in expected_magics)
                missing_protection = bool(sl <= 0.0 and tp <= 0.0)
                age_sec = _position_age_sec(pos)
                in_grace = bool(age_sec is not None and age_sec <= grace_sec)
                issues: List[str] = []
                if is_foreign_symbol:
                    issues.append("foreign_symbol")
                if not is_expected_magic:
                    issues.append("unmanaged_magic")
                if missing_protection:
                    if is_expected_magic:
                        issues.append("zero_protection")
                    elif not is_foreign_symbol:
                        issues.append("zero_protection_unmanaged")
                issue_info = {
                    "ticket": ticket,
                    "symbol": symbol,
                    "asset": asset,
                    "magic": magic,
                    "sl": sl,
                    "tp": tp,
                    "age_sec": (
                        round(float(age_sec), 3) if age_sec is not None else None
                    ),
                    "issues": list(issues),
                }
                if issues and in_grace:
                    grace_positions.append(issue_info)
                    continue
                if issues:
                    position_issues.append(issue_info)
                if is_foreign_symbol and symbol not in foreign_symbols:
                    foreign_symbols.append(symbol)
                if not is_expected_magic:
                    unmanaged_positions += 1
                if missing_protection and is_expected_magic:
                    zero_protection_positions += 1
            except Exception:
                continue
        reasons: List[str] = []
        allowed_total = int(
            sum(max(0, int(v or 0)) for v in self._max_open_positions.values())
        )
        if allowed_total > 0 and total_positions > allowed_total:
            reasons.append(f"total_positions:{total_positions}>{allowed_total}")
        for symbol, asset in known_symbols.items():
            max_allowed = int(self._max_open_positions.get(asset, 0) or 0)
            current = int(symbol_counts.get(symbol, 0) or 0)
            if max_allowed > 0 and current > max_allowed:
                reasons.append(f"{asset}_positions:{current}>{max_allowed}")
        if foreign_symbols:
            reasons.append(f"foreign_symbols:{','.join(sorted(foreign_symbols))}")
        if unmanaged_positions > 0:
            reasons.append(f"unmanaged_positions:{unmanaged_positions}")
        if zero_protection_positions > 0:
            reasons.append(f"zero_protection_positions:{zero_protection_positions}")
        top_magics = dict(
            sorted(
                magic_counts.items(), key=lambda item: (-int(item[1]), int(item[0]))
            )[:5]
        )
        return {
            "reason": "|".join(reasons),
            "total_positions": int(total_positions),
            "symbol_counts": dict(symbol_counts),
            "foreign_symbols": list(sorted(foreign_symbols)),
            "unmanaged_positions": int(unmanaged_positions),
            "zero_protection_positions": int(zero_protection_positions),
            "magic_counts": top_magics,
            "position_issues": list(position_issues),
            "grace_positions": list(grace_positions),
        }

    def _check_backtest_integration(self) -> None:
        if self._backtest_linkage_verified:
            return
        try:
            if BacktestEngine is None:
                raise ImportError("Backtest module not found")
            _ = BacktestEngine(
                "XAU",
                model_version=XAU_BACKTEST_CONFIG.model_version,
                run_cfg=XAU_BACKTEST_CONFIG,
            )
            _ = BacktestEngine(
                "BTC",
                model_version=BTC_BACKTEST_CONFIG.model_version,
                run_cfg=BTC_BACKTEST_CONFIG,
            )
            self._backtest_linkage_verified = True
            log_health.info(
                "Backtest Engine Integration: VERIFIED (XAU + BTC configs loaded)"
            )
        except Exception as exc:
            log_err.error("Backtest Engine Integration FAILED: %s", exc)
            if not self.dry_run:
                print(f"WARNING: Backtest Engine Linkage Error: {exc}")

    def _transition_state(
        self, current: EngineState, nxt: EngineState, reason: str
    ) -> EngineState:
        now = time.time()
        ttl = max(
            2.0,
            float(os.getenv("FSM_TRANSITION_LOG_TTL_SEC", "15.0") or 15.0),
        )
        signature = f"{current.value}->{nxt.value}:{reason}"
        cache = dict(getattr(self, "_fsm_transition_log_cache", {}) or {})
        last_ts = float(cache.get(signature, 0.0) or 0.0)
        if last_ts <= 0.0 or (now - last_ts) >= ttl:
            log_health.info(
                "FSM_TRANSITION | %s -> %s | reason=%s",
                current.value,
                nxt.value,
                reason,
            )
            cache[signature] = now
            if len(cache) > 32:
                oldest = sorted(cache.items(), key=lambda item: item[1])[:8]
                for old_key, _ in oldest:
                    cache.pop(old_key, None)
            self._fsm_transition_log_cache = cache
        return nxt

    def _resolve_halt_reason(self, reason: str) -> str:
        reason_s = str(reason or "unspecified").strip() or "unspecified"
        if reason_s != "live_risk_breach":
            return reason_s
        try:
            snapshot = self._update_live_evidence(force=True)
        except Exception:
            snapshot = None
        try:
            resolved = str(self._live_risk_halt_reason(snapshot=snapshot) or "").strip()
        except Exception:
            resolved = ""
        if resolved:
            log_health.warning(
                "FSM_HALT_REASON_RESOLVED | raw=%s resolved=%s",
                reason_s,
                resolved,
            )
            return resolved
        return reason_s

    def _halt_fsm(self, reason: str) -> None:
        reason_s = self._resolve_halt_reason(reason)
        recoverable = reason_s.lower() in RECOVERABLE_HALT_REASONS
        monitoring_only = reason_s.lower() in MONITORING_ONLY_HALT_REASONS
        if monitoring_only:
            log_health.warning("FSM_HALT | reason=%s | mode=monitoring_only", reason_s)
        else:
            log_err.critical("FSM_HALT | reason=%s", reason_s)
        with self._lock:
            if not (recoverable or monitoring_only):
                self._manual_stop = True
            self._run.clear()
        self._drain_queue(self._order_q)
        self._drain_queue(self._result_q)
        with self._order_state_lock:
            self._order_rm_by_id.clear()
            self._pending_order_meta.clear()
        if recoverable:
            log_health.warning(
                "FSM_HALT_RECOVERABLE | reason=%s | attempting_auto_restart", reason_s
            )

            def _auto_restart():
                _MAX_RESTART_ATTEMPTS = 3
                for attempt in range(1, _MAX_RESTART_ATTEMPTS + 1):
                    delay = 10.0 * attempt
                    log_health.info(
                        "FSM_AUTO_RESTART_ATTEMPT | reason=%s attempt=%d/%d delay=%.0fs",
                        reason_s,
                        attempt,
                        _MAX_RESTART_ATTEMPTS,
                        delay,
                    )
                    time.sleep(delay)
                    try:
                        ok = self._recover_all()
                        if ok:
                            log_health.info(
                                "FSM_AUTO_RESTART_OK | reason=%s attempt=%d",
                                reason_s,
                                attempt,
                            )
                            self.start()
                            return
                        log_err.error(
                            "FSM_AUTO_RESTART_FAILED | reason=%s attempt=%d/%d | _recover_all=False",
                            reason_s,
                            attempt,
                            _MAX_RESTART_ATTEMPTS,
                        )
                    except Exception as exc:
                        log_err.error(
                            "FSM_AUTO_RESTART_EXCEPTION | reason=%s attempt=%d/%d err=%s",
                            reason_s,
                            attempt,
                            _MAX_RESTART_ATTEMPTS,
                            exc,
                        )
                log_err.critical(
                    "FSM_AUTO_RESTART_EXHAUSTED | reason=%s | all %d attempts failed — manual intervention required",
                    reason_s,
                    _MAX_RESTART_ATTEMPTS,
                )

            t = threading.Thread(
                target=_auto_restart, name="fsm.auto_restart", daemon=True
            )
            t.start()
            return
        if monitoring_only:
            log_health.info(
                "FSM_HALT_MONITORING_ONLY | reason=%s | auto_restart_allowed=False",
                reason_s,
            )
            return
        self._emergency_flatten(f"fsm_halt:{reason_s}")

    def _validate_payload_timestamp(
        self, asset: str, payload: Dict[str, Any], tf: str
    ) -> Tuple[bool, str]:
        return self._inference_engine.validate_payload_timestamp(asset, payload, tf)

    def _validate_data_sync_payloads(
        self, asset: str, payloads: Dict[str, Optional[Dict[str, Any]]]
    ) -> Tuple[bool, str]:
        return self._inference_engine.validate_data_sync_payloads(asset, payloads)

    @staticmethod
    def _extract_payload_frame(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return InferenceEngine.extract_payload_frame(payload)

    def _probabilistic_levels(
        self,
        *,
        side: str,
        entry: float,
        base_sl: float,
        base_tp: float,
        confidence: float,
        frame: Dict[str, Any],
    ) -> Tuple[float, float]:
        return self._inference_engine.probabilistic_levels(
            side=side,
            entry=entry,
            base_sl=base_sl,
            base_tp=base_tp,
            confidence=confidence,
            frame=frame,
        )

    def _build_ml_candidate(
        self, asset: str, sig: MLSignal
    ) -> Optional[AssetCandidate]:
        return self._inference_engine.build_ml_candidate(asset, sig)

    def _execute_candidates(self, candidates: List[AssetCandidate]) -> None:
        self._execution_manager.execute_candidates(candidates)

    def _verification_step(self) -> None:
        self._execution_manager.verification_step()

    def _loop(self) -> None:
        current_thread = threading.current_thread()
        self._mark_runtime_alive()
        if not self._xau or not self._btc:
            log_err.error("Portfolio engine loop start failed: pipelines not built")
            self._run.clear()
            with self._lock:
                if self._loop_thread is current_thread:
                    self._loop_thread = None
            return
        unexpected_exit = False
        try:
            self._fsm_runner.run(
                initial_state=EngineState.BOOT,
                initial_ctx=EngineCycleContext(),
            )
            with self._lock:
                unexpected_exit = bool(self._run.is_set())
                if unexpected_exit:
                    self._run.clear()
        except Exception as exc:
            with self._lock:
                self._run.clear()
            log_err.error(
                "ENGINE_LOOP_FATAL | err=%s | tb=%s", exc, traceback.format_exc()
            )
        finally:
            with self._lock:
                if self._loop_thread is current_thread:
                    self._loop_thread = None
            if unexpected_exit:
                log_err.error(
                    "ENGINE_LOOP_EXIT_UNEXPECTED | manual_stop=%s mt5_ready=%s",
                    self._manual_stop,
                    self._mt5_ready,
                )

    def _p95(self, values: Deque[float]) -> float:
        """# SAFE IMPROVEMENT: Private helper extracted from run_chaos_audit (used internally)."""
        arr = sorted(float(v) for v in values if float(v) >= 0.0)
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, max(0, int(round(0.95 * (len(arr) - 1)))))
        return float(arr[idx])


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "DeterministicFSMRunner",
    "EngineFSMStageServices",
    "EngineRuntimeMixin",
    "IFSMEnginePorts",
    "MONITORING_ONLY_HALT_REASONS",
    "RECOVERABLE_HALT_REASONS",
]
