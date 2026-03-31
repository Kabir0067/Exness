from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.ml_router import MLSignal, fetch_ml_payloads, infer_from_payloads

from .fsm_ports import IFSMEnginePorts
from .fsm_types import EngineCycleContext, EngineState, StepDecision
from .logging_setup import log_err, log_health
from .models import AssetCandidate
from .scheduler import UTCScheduler

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


class EngineFSMStageServices(IFSMEnginePorts):
    """
    Stateful adapters from FSM runtime ports to concrete engine behavior.

    This class intentionally keeps stage logic explicit and deterministic while
    preserving existing trading semantics.
    """

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine
        self._asset_priority = ("XAU", "BTC")

    def is_running(self) -> bool:
        return bool(self._e._run.is_set())

    def cycle_floor_sec(self) -> float:
        return max(float(self._e._poll_fast), float(self._e._min_cycle_sec))

    def report_cycle_latency_ms(self, latency_ms: float) -> None:
        self._e._report_cycle_latency_ms(float(latency_ms))

    def step_boot(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        if self._e.manual_stop_active():
            ctx.halt_reason = "manual_stop_active"
            nxt = self._e._transition_state(state, EngineState.HALT, "manual_stop")
            return StepDecision(next_state=nxt, ctx=ctx, skip_sleep=True)

        if not self._e.dry_run and not (self._e._model_loaded and self._e._backtest_passed):
            ctx.halt_reason = "gatekeeper_failed"
            nxt = self._e._transition_state(state, EngineState.HALT, "gatekeeper_failed")
            return StepDecision(next_state=nxt, ctx=ctx)

        if not self._e._check_mt5_health():
            ctx.halt_reason = "mt5_unhealthy"
            nxt = self._e._transition_state(state, EngineState.HALT, "mt5_unhealthy")
            return StepDecision(next_state=nxt, ctx=ctx)

        nxt = self._e._transition_state(state, EngineState.DATA_SYNC, "boot_ok")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_data_sync(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        if not self._e._check_mt5_health():
            ctx.halt_reason = "mt5_disconnected"
            nxt = self._e._transition_state(state, EngineState.HALT, "mt5_disconnected")
            return StepDecision(next_state=nxt, ctx=ctx)

        self._e._check_hard_stop_file()
        if self._e.manual_stop_active():
            ctx.halt_reason = "manual_stop_triggered"
            nxt = self._e._transition_state(state, EngineState.HALT, "manual_stop_triggered")
            return StepDecision(next_state=nxt, ctx=ctx, skip_sleep=True)

        halt_reason = ""
        halt_fn = getattr(self._e, "_live_risk_halt_reason", None)
        if callable(halt_fn):
            try:
                halt_reason = str(halt_fn() or "")
            except Exception:
                halt_reason = ""
        if halt_reason:
            ctx.halt_reason = halt_reason
            nxt = self._e._transition_state(state, EngineState.HALT, "live_risk_breach")
            return StepDecision(next_state=nxt, ctx=ctx, skip_sleep=True)

        if self._e._live_trading_pause_reason(force=False):
            return StepDecision(next_state=state, ctx=EngineCycleContext())

        payloads: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
        active_assets = self._ordered_active_assets(UTCScheduler.get_active_assets())
        data_sync_ok = True
        data_sync_reason = ""
        for asset in active_assets:
            if self._e._is_asset_blocked(asset):
                self._e._log_blocked_asset_skip(asset, "data_sync")
                continue

            asset_payloads = fetch_ml_payloads(asset)
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
            nxt = self._e._transition_state(state, EngineState.ML_INFERENCE, "data_sync_ok")
            return StepDecision(next_state=nxt, ctx=EngineCycleContext(payloads=payloads))

        now = time.time()
        if (
            data_sync_reason != self._e._last_data_sync_fail_reason
            or (now - self._e._last_data_sync_fail_ts) > 5.0
        ):
            log_health.warning("DATA_SYNC_WAIT | reason=%s", data_sync_reason)
            self._e._last_data_sync_fail_reason = data_sync_reason
            self._e._last_data_sync_fail_ts = now
        return StepDecision(next_state=state, ctx=ctx)

    def step_ml_inference(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        signals: Dict[str, MLSignal] = {}
        for asset, payload in ctx.payloads.items():
            if self._e._is_asset_blocked(asset):
                self._e._log_blocked_asset_skip(asset, "ml_inference")
                continue

            sig = None
            has_catboost = asset in self._e._catboost_payloads
            if has_catboost:
                sig = self._e._infer_catboost(asset, payload)
                if sig is None:
                    if self._e._allow_llm_fallback:
                        sig = infer_from_payloads(
                            asset,
                            scalp_payload=payload.get("scalp"),
                            intraday_payload=payload.get("intraday"),
                        )
                    else:
                        sig = MLSignal(
                            asset=asset,
                            signal="HOLD",
                            side="Neutral",
                            confidence=0.0,
                            reason="catboost_inference_failed_no_fallback",
                            provider="catboost",
                            model="catboost_trained",
                            entry=None,
                            stop_loss=None,
                            take_profit=None,
                            scalp_payload=payload.get("scalp"),
                            intraday_payload=payload.get("intraday"),
                        )
            elif self._e._allow_llm_fallback:
                sig = infer_from_payloads(
                    asset,
                    scalp_payload=payload.get("scalp"),
                    intraday_payload=payload.get("intraday"),
                )
            else:
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

            log_health.info(
                "FSM_ML_SIGNAL | asset=%s signal=%s conf=%.3f provider=%s model=%s reason=%s",
                asset,
                sig.signal,
                sig.confidence,
                sig.provider,
                sig.model,
                sig.reason,
            )
            signals[asset] = sig

        ctx.ml_signals = signals
        nxt = self._e._transition_state(state, EngineState.RISK_CALC, "ml_complete")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_risk_calc(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        candidates: List[AssetCandidate] = []
        for asset in ("XAU", "BTC"):
            sig = ctx.ml_signals.get(asset)
            if sig is None:
                continue
            cand = self._e._build_ml_candidate(asset, sig)
            if cand is not None:
                candidates.append(cand)

        ctx.candidates = candidates
        self._e._last_cand_xau = next((c for c in candidates if c.asset == "XAU"), None)
        self._e._last_cand_btc = next((c for c in candidates if c.asset == "BTC"), None)
        nxt = self._e._transition_state(state, EngineState.EXECUTION_QUEUE, "risk_complete")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_execution_queue(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        if self._e._retraining_mode:
            now = time.time()
            if now - self._e._last_retraining_log_ts > 10.0:
                log_health.info("RETRAINING_PAUSE | skipping new orders during retraining")
                self._e._last_retraining_log_ts = now
            nxt = self._e._transition_state(state, EngineState.VERIFICATION, "retraining_pause")
            return StepDecision(next_state=nxt, ctx=ctx)

        halt_reason = ""
        halt_fn = getattr(self._e, "_live_risk_halt_reason", None)
        if callable(halt_fn):
            try:
                halt_reason = str(halt_fn() or "")
            except Exception:
                halt_reason = ""
        if halt_reason:
            ctx.halt_reason = halt_reason
            nxt = self._e._transition_state(state, EngineState.HALT, "live_risk_breach")
            return StepDecision(next_state=nxt, ctx=ctx, skip_sleep=True)

        if self._e._live_trading_pause_reason(force=False):
            nxt = self._e._transition_state(state, EngineState.VERIFICATION, "live_evidence_pause")
            return StepDecision(next_state=nxt, ctx=ctx)

        self._e._execute_candidates(ctx.candidates)
        nxt = self._e._transition_state(state, EngineState.VERIFICATION, "queue_complete")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_verification(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        self._e._verification_step()
        nxt = self._e._transition_state(state, EngineState.DATA_SYNC, "verification_complete")
        return StepDecision(next_state=nxt, ctx=ctx)

    def step_halt(self, ctx: EngineCycleContext) -> None:
        self._e._halt_fsm(ctx.halt_reason or "unspecified")

    def on_exception(
        self,
        state: EngineState,
        ctx: EngineCycleContext,
        exc: Exception,
    ) -> StepDecision:
        log_err.error("FSM_LOOP_EXCEPTION | err=%s | tb=%s", exc, traceback.format_exc())
        ctx.halt_reason = f"fsm_exception:{exc}"
        return StepDecision(next_state=EngineState.HALT, ctx=ctx, skip_sleep=True)

    def _ordered_active_assets(self, assets: List[str]) -> List[str]:
        """Return active assets in deterministic execution order."""
        unique = {str(a).strip().upper() for a in assets if str(a).strip()}
        ordered: List[str] = [a for a in self._asset_priority if a in unique]
        remaining = sorted(a for a in unique if a not in set(self._asset_priority))
        ordered.extend(remaining)
        return ordered
