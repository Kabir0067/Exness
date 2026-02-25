from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

import core.risk_engine as risk_mod
from core.config import BTCSymbolParams, BaseEngineConfig
from core.risk_engine import RiskManager


def _pct_ms(values_sec: list[float], p: float) -> float:
    if not values_sec:
        return 0.0
    values = sorted(v * 1000.0 for v in values_sec)
    idx = max(0, min(len(values) - 1, math.ceil(float(p) * len(values)) - 1))
    return float(values[idx])


def _timer_budget_ms() -> float:
    # Windows monotonic timer can be ~15.6ms; avoid impossible 5ms assertions there.
    res_ms = float(time.get_clock_info("monotonic").resolution) * 1000.0
    return max(5.0, res_ms * 2.0)


@pytest.mark.parametrize("n_threads,n_loops", [(100, 120)])
def test_risk_manager_thread_safety_under_extreme_contention(
    monkeypatch: pytest.MonkeyPatch,
    n_threads: int,
    n_loops: int,
) -> None:
    # Isolate from external MT5 state for deterministic concurrency checks.
    monkeypatch.setattr(risk_mod, "mt5", None)

    sp = BTCSymbolParams()
    cfg = BaseEngineConfig(symbol_params=sp)
    rm = RiskManager(cfg, sp)

    start_equity = Decimal("10000.0")
    with rm._lock:
        rm._acc.balance = float(start_equity)
        rm._acc.equity = float(start_equity)
        rm._acc.margin_free = float(start_equity)
        rm._acc.ts = time.time()
        rm._bot_balance_base = float(start_equity)
        rm._peak_equity = float(start_equity)
        rm._daily_peak = float(start_equity)

    def pnl_worker(seed: int) -> tuple[Decimal, list[float]]:
        rng = random.Random(seed)
        local_sum = Decimal("0")
        lat = []
        for _ in range(n_loops):
            delta = rng.uniform(-0.05, 0.05)
            t0 = time.monotonic()
            rm.update_pnl(delta)
            lat.append(time.monotonic() - t0)
            local_sum += Decimal(str(delta))
        return local_sum, lat

    def guard_eval_worker(seed: int) -> tuple[list[float], list[float]]:
        rng = random.Random(seed)
        guard_lat = []
        eval_lat = []
        for _ in range(n_loops):
            spread_pct = 0.0001 + rng.random() * 0.0001
            t0 = time.monotonic()
            rm.guard_decision(
                spread_pct=spread_pct,
                tick_ok=True,
                tick_reason="",
                ingest_ms=0.5,
                last_bar_age=5.0,
                in_session=True,
                drawdown_exceeded=False,
                latency_cooldown=False,
            )
            guard_lat.append(time.monotonic() - t0)

            t1 = time.monotonic()
            rm.evaluate_account_state()
            eval_lat.append(time.monotonic() - t1)
        return guard_lat, eval_lat

    n_pnl = n_threads // 2
    n_guard = n_threads - n_pnl

    total_delta = Decimal("0")
    all_pnl_lat: list[float] = []
    all_guard_lat: list[float] = []
    all_eval_lat: list[float] = []

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = []
        for i in range(n_pnl):
            futs.append(("pnl", pool.submit(pnl_worker, i + 7)))
        for i in range(n_guard):
            futs.append(("guard", pool.submit(guard_eval_worker, i + 1007)))

        for kind, fut in futs:
            data = fut.result(timeout=90)
            if kind == "pnl":
                dlt, lat = data
                total_delta += dlt
                all_pnl_lat.extend(lat)
            else:
                g_lat, e_lat = data
                all_guard_lat.extend(g_lat)
                all_eval_lat.extend(e_lat)

    with rm._lock:
        final_equity = Decimal(str(rm._acc.equity))
        final_daily_pnl = Decimal(str(rm._daily_pnl))
        base_equity = Decimal(str(rm._bot_balance_base))

    expected_equity = start_equity + total_delta
    assert abs(final_equity - expected_equity) <= Decimal("1e-8")

    # Atomic consistency invariant: daily_pnl == equity - base_equity
    assert abs(final_daily_pnl - (final_equity - base_equity)) <= Decimal("1e-8")

    # Contention latency budget (lock bottleneck detection).
    # Use dynamic floor based on OS timer resolution.
    budget = _timer_budget_ms()
    assert _pct_ms(all_pnl_lat, 0.95) <= budget
    assert _pct_ms(all_guard_lat, 0.95) <= budget
    assert _pct_ms(all_eval_lat, 0.95) <= budget
    assert _pct_ms(all_guard_lat, 0.99) <= 20.0
