from __future__ import annotations

import time
from typing import Callable

from .fsm_ports import IFSMEnginePorts
from .fsm_types import EngineCycleContext, EngineState, StepDecision


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
                    state, ctx, skip_sleep = decision.next_state, decision.ctx, decision.skip_sleep
            except Exception as exc:
                decision = self._ports.on_exception(state, ctx, exc)
                state, ctx, skip_sleep = decision.next_state, decision.ctx, decision.skip_sleep

            if skip_sleep:
                continue

            dt = time.monotonic() - t0
            try:
                self._ports.report_cycle_latency_ms(float(dt * 1000.0))
            except Exception:
                pass
            min_cycle = max(0.02, float(self._ports.cycle_floor_sec()))
            time.sleep(max(0.02, min_cycle - dt))
