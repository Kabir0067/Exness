from __future__ import annotations

from typing import Protocol

from .fsm_types import EngineCycleContext, EngineState, StepDecision


class IFSMEnginePorts(Protocol):
    """Hexagonal runtime ports for deterministic engine orchestration."""

    def is_running(self) -> bool:
        ...

    def cycle_floor_sec(self) -> float:
        ...

    def report_cycle_latency_ms(self, latency_ms: float) -> None:
        ...

    def step_boot(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_data_sync(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_ml_inference(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_risk_calc(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_execution_queue(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_verification(self, state: EngineState, ctx: EngineCycleContext) -> StepDecision:
        ...

    def step_halt(self, ctx: EngineCycleContext) -> None:
        ...

    def on_exception(
        self,
        state: EngineState,
        ctx: EngineCycleContext,
        exc: Exception,
    ) -> StepDecision:
        ...
