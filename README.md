<div align="center">

<h1>QuantCore Pro</h1>

<h3>Institutional-style MetaTrader 5 trading runtime</h3>

<p>
  Built for deterministic orchestration, model admission control, layered risk rails,
  broker-aware execution, and operational visibility.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Runtime-MT5%20%2F%20Exness-0B4F9C?style=flat-square" alt="MT5 Exness" />
  <img src="https://img.shields.io/badge/Assets-XAUUSDm%20%2B%20BTCUSDm-8A6A00?style=flat-square" alt="Assets" />
  <img src="https://img.shields.io/badge/ML-CatBoost-F6C343?style=flat-square" alt="CatBoost" />
  <img src="https://img.shields.io/badge/Execution-Deterministic%20FSM-111111?style=flat-square" alt="Deterministic FSM" />
  <img src="https://img.shields.io/badge/V1-Finalized-0A7F5A?style=flat-square" alt="V1 Finalized" />
  <img src="https://img.shields.io/badge/Verification-116%20tests%20verified-1F9D55?style=flat-square" alt="116 tests verified" />
</p>

</div>

---

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>Release Posture</h3>
      <p>
        <strong>V1 is finalized as a runtime-ready engineering release.</strong>
      </p>
      <p>
        The core runtime, guard rails, verification paths, and documentation
        have been aligned to the system that is actually present in this repository.
      </p>
    </td>
    <td valign="top" width="50%">
      <h3>System Focus</h3>
      <p>
        QuantCore Pro is designed around the same engineering concerns that define
        serious internal execution stacks: deterministic flow, controlled startup,
        risk-first behavior, reconciliation, and observability.
      </p>
    </td>
  </tr>
</table>

## Executive Summary

QuantCore Pro is not presented as a one-file signal bot, a toy MT5 script, or a "black box alpha" marketing page.

It is a production-minded trading runtime for MetaTrader 5 that combines:

- deterministic finite-state execution
- model gatekeeping before live order flow
- layered portfolio and per-asset risk controls
- execution verification and broker reconciliation
- runtime diagnostics, health telemetry, and recovery paths
- dry-run, engine-only, and monitoring-oriented operating modes

The result is a repository that reads and behaves far closer to an institutional execution runtime than to the average retail automation project. That statement describes engineering posture and system design. It does not claim ownership of any hidden bank strategy, proprietary desk alpha, or guaranteed profit profile.

## Why It Feels Like A Serious Internal Runtime

<table>
  <tr>
    <td valign="top" width="33%">
      <h3>Deterministic Core</h3>
      <p>
        The engine is built around a finite-state machine instead of ad hoc
        background behavior. That makes transitions explicit, failures more
        diagnosable, and execution flow easier to reason about under stress.
      </p>
    </td>
    <td valign="top" width="33%">
      <h3>Trade Admission Control</h3>
      <p>
        Models are not blindly trusted. The runtime checks gate state, artifact
        readiness, asset eligibility, and startup conditions before letting live
        order flow proceed.
      </p>
    </td>
    <td valign="top" width="33%">
      <h3>Risk As Infrastructure</h3>
      <p>
        Risk is not a single stop-loss rule. It is embedded across sizing,
        drawdown limits, circuit breakers, portfolio controls, trade pausing,
        and emergency stop behavior.
      </p>
    </td>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <h3>Execution Verification</h3>
      <p>
        Orders are tracked beyond the initial send path. The stack includes
        queueing, deduplication, reconciliation, SL/TP attachment handling,
        and pending-order state resolution.
      </p>
    </td>
    <td valign="top" width="33%">
      <h3>Recovery-Aware Runtime</h3>
      <p>
        Startup, reconnect, and unhealthy runtime paths are supervised. The
        engine is built to degrade, pause, recover, or hold rather than fail
        silently.
      </p>
    </td>
    <td valign="top" width="33%">
      <h3>Operational Visibility</h3>
      <p>
        Health logs, diagnostics snapshots, MT5 events, and execution telemetry
        make the runtime inspectable. That is one of the clearest markers of a
        production-minded system.
      </p>
    </td>
  </tr>
</table>

## V1 Final Status

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>Verified</h3>
      <ul>
        <li>Core runtime compiles cleanly</li>
        <li>Local smoke verifier passes: <code>python test.py</code></li>
        <li>Committed regression verification recorded: <code>116 passed</code></li>
        <li>Dry-run startup path has been verified</li>
        <li>Dry-run engine-only soak check completed cleanly</li>
        <li>Volatility breaker now behaves as a cooldown, not an all-day hard stop</li>
        <li>Irregular bar-gap false positives are blocked in the volatility breaker</li>
        <li>Startup telemetry warmup was added for live evidence evaluation</li>
        <li>Dry-run Telegram noise is disabled by default</li>
      </ul>
    </td>
    <td valign="top" width="50%">
      <h3>Release Meaning</h3>
      <p>
        V1 is complete as an <strong>engineering release</strong>.
      </p>
      <p>
        That means the runtime boots, the control rails are in place, the
        verification layer is present, and the public-facing documentation now
        reflects verified system behavior.
      </p>
      <p>
        Market edge, profitability durability, and long-horizon live alpha still
        require ongoing statistical validation, exactly as they should in any
        serious trading program.
      </p>
    </td>
  </tr>
</table>

The canonical release gate is documented here:

- [V1_FINAL_CHECKLIST.md](V1_FINAL_CHECKLIST.md)

## Runtime Topology

```text
main.py
  -> runtime bootstrap
  -> startup policy and model gate orchestration
  -> engine supervisor
  -> optional Telegram supervisor

Bot/Motor/
  -> deterministic FSM runtime
  -> inference engine
  -> execution manager
  -> order sync manager
  -> portfolio engine health and diagnostics

core/
  -> configuration
  -> feature engineering
  -> signal logic
  -> risk engine
  -> portfolio risk
  -> model gate and model manager

DataFeed/
  -> XAU market data path
  -> BTC market data path
  -> tick, bar, and microstructure payload generation

ExnessAPI/
  -> order placement
  -> position handling
  -> broker interaction utilities

Backtest/
  -> model training support
  -> backtest engine
  -> metrics and validation support
```

## Operating Modes

| Mode | Purpose | Notes |
| --- | --- | --- |
| `python main.py` | Live-oriented startup path | Uses strict runtime and model admission rules |
| `python main.py --dry-run --engine-only` | Controlled runtime verification | Best path for clean engine-only startup checks |
| `DRY_RUN=1 python main.py` | Dry-run from environment | Telegram remains disabled by default unless explicitly enabled |
| `MONITORING_ONLY=1 python main.py` | Observation without trade execution | Useful for supervised runtime inspection |

If Telegram is required during dry-run, enable it explicitly:

```bash
ALLOW_TG_IN_DRY_RUN=1
```

## Verification

These commands represent the V1 verification posture:

```bash
python -m py_compile core/risk_engine.py Bot/Motor/engine.py main.py test.py
python test.py
python main.py --dry-run --engine-only
```

The V1 release record also includes a committed regression verification run with `116 passed`.

## System Characteristics That Are Real In This Repository

- XAUUSDm and BTCUSDm are first-class runtime assets.
- Live-oriented and dry-run execution paths both exist.
- Model gatekeeping is enforced before live order admission.
- Risk controls include hard stops, soft stops, cooldowns, kill-switch behavior, and portfolio exposure controls.
- The volatility circuit breaker is time-bounded and cooldown-based.
- Health and diagnostics are emitted into `Logs/`.
- The runtime includes startup supervision, recovery hooks, and broker-state awareness.
- The repository includes a built-in smoke verifier for local release checks.

## Logs And Observability

The runtime is designed to be inspectable while it is running. Key operational outputs include:

- `Logs/main.log`
- `Logs/portfolio_engine_health.log`
- `Logs/portfolio_engine_diag.jsonl`
- `Logs/mt5.log`
- `Logs/telegram.log`

That visibility matters because serious trading systems are judged not only by how they trade, but by how clearly they reveal their own health, state transitions, and failure modes.

## Repository Layout

```text
main.py
mt5_client.py
log_config.py
test.py

Bot/
Backtest/
core/
DataFeed/
ExnessAPI/
Artifacts/
Logs/
strategies/
```

If the committed regression suite is present in the checkout, it lives under `tests/`.

## Environment

Typical runtime variables:

```ini
EXNESS_LOGIN=...
EXNESS_PASSWORD=...
EXNESS_SERVER=...

TG_TOKEN=...
TG_ADMIN_ID=...
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Positioning

The strongest truthful one-line description of this repository is:

> QuantCore Pro is a production-minded, institutional-style MT5 trading runtime with deterministic orchestration, strict trade admission, layered risk infrastructure, broker-aware recovery, and operational diagnostics.

That is the level at which V1 is finalized.

## Disclaimer

This repository is a trading engineering system.

- It is designed to control risk, not eliminate it.
- Backtest quality and runtime controls improve discipline, but do not guarantee future returns.
- Real-money deployment should still begin with monitoring, staged exposure, and capital discipline.

## Next Stage

V1 handoff is complete.

The next development stage is **V2**.
