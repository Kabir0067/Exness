<div align="center">

# EX Trading System

### Private Quant Runtime for XAUUSDm and BTCUSDm

**Session-Aware · WAL-Protected · Self-Healing · Operator-Visible**

<sub>
Internal system dossier.  
This document describes the runtime as it exists today.                                  
</sub>

</div>

---

## System Identity

This repository is a long-running trading runtime built around two very different
markets:

- `XAUUSDm` as the session-sensitive institutional asset
- `BTCUSDm` as the always-on 24/7 asset

The system is no longer treated as a simple bot. It is structured as a
production controller with explicit safeguards around:

- market-session transitions
- stale-data rejection
- model-gate enforcement
- idempotent order execution
- crash recovery and restart safety
- Telegram-based operational visibility

The design target is not raw aggression. The target is durable execution under
messy real conditions: weekend gaps, transport jitter, MT5 instability, process
restarts, noisy logs, and long unattended uptime.

---

## Current Production Contract

### 1. Session Integrity

`Bot/Motor/pipeline.py` and `UTCScheduler` currently define the live
session boundary behavior.

- `XAU` respects the weekday market-open window.
- `BTC` remains tradable as a 24/7 asset.
- session open / close transitions reset short-term runtime state and trigger
  reconciliation
- reopen caution is enforced through market-feed validation, risk gates, and
  `core/news_blackout.py` rather than a separate persisted session-manager file

This means the runtime will not treat Sunday reopen as a normal continuous feed.
It requires the session transition to settle and then re-validates the market
stack before fresh XAU execution is allowed.

### 2. Data and Gap Defense

The live engine rejects unsafe input before inference or execution:

- stale tick / bar input is blocked in `Bot/Motor/pipeline.py`
- large baseline discontinuities reset the short-term pipeline cache
- closed-session assets are reconciled without being traded
- session transitions reset runtime state and force reconciliation before the
  asset is trusted again

The intent is simple: no stale carry-over and no quiet reuse of pre-gap state.

### 3. Execution Safety

Order execution is protected by `core/idempotency.py`.

- each order receives a unique idempotency key
- a write-ahead log is written before broker submission
- restart reconciliation confirms or fails in-flight orders
- WAL auto-compaction prevents unbounded growth over long uptime

If the process dies between send and broker acknowledgement, recovery happens
from durable state rather than memory assumptions.

### 4. Operator Visibility

`runmain/supervisors.py`, `Bot/bot.py`, and `core/stability_monitor.py`
currently form the operational visibility layer.

- a bounded Telegram notifier queue with retry and backoff
- operator-facing Telegram controls wired directly into the live engine
- execution and runtime notifications bridged through the bot layer
- JSON heartbeats written to disk for watchdog-style health observation

This layer provides visibility and control surfaces around the engine. It does
not replace the engine's own risk and execution safeguards.

### 5. Logging Discipline

The runtime is tuned for production-grade signal rather than debug noise.

- rotating module logs are created under `Logs/`
- root logging defaults to `INFO`
- noisy third-party libraries are clamped down in `log_config.py`
- runtime health is exposed through heartbeat files and structured log events

---

## Runtime Topology

```text
main.py
  -> runmain/bootstrap.py
  -> runmain/gate.py
  -> runmain/supervisors.py
  -> Bot/portfolio_engine.py
       -> Bot/Motor/engine.py
       -> Bot/Motor/fsm.py
       -> Bot/Motor/pipeline.py
       -> Bot/Motor/inference.py
       -> Bot/Motor/execution.py
  -> Bot/bot.py
  -> core/risk_manager.py
  -> core/portfolio_risk.py
  -> core/idempotency.py
  -> core/news_blackout.py
  -> core/stability_monitor.py
```

Key responsibilities:

- `main.py`: production entry point
- `runmain/bootstrap.py`: logging, runtime wiring, process preflight
- `runmain/gate.py`: model-gate checks and retraining readiness
- `Bot/portfolio_engine.py`: compatibility facade for the motor exports
- `Bot/Motor/fsm.py`: deterministic live cycle
- `Bot/Motor/pipeline.py`: market validation, active-asset logic, stale-data control
- `Bot/Motor/execution.py`: queueing, broker submission, result reconciliation
- `Bot/bot.py`: operator-facing Telegram controls and runtime notifications
- `core/risk_manager.py`: live pre-order gate, sizing, hard stops
- `core/portfolio_risk.py`: cross-asset portfolio guardrails
- `core/idempotency.py`: WAL, replay safety, startup reconcile
- `core/news_blackout.py`: event blackout enforcement
- `core/stability_monitor.py`: heartbeat, drift, and stability telemetry

---

## Model State

The repository contains institutional backtest and training artifacts for both
core assets. The current practical reading is:

- `BTC` and `XAU` are gate-verified in the current artifact set
- `XAU` now passes full walk-forward validation in the non-fast path
- `institutional_grade` can still remain `false` when the suspicious-Sharpe
  safeguard is triggered

That distinction is deliberate. A model can pass the live gate while still
failing to earn the higher institutional-grade label. Very high backtest Sharpe
is treated as a risk signal, not as a trophy.

---

## XAU Status Note

The latest hardening pass closed the most important XAU reliability gaps:

- regime-aware training and live inference are aligned
- walk-forward windows pass in the full non-fast path
- weekend session handling is now coordinated through `UTCScheduler` and the
  pipeline transition logic
- Sunday reopen no longer relies on naive weekday-only assumptions
- live admission after reopen is filtered again by data-integrity, risk, and
  blackout checks

This improves survivability. It does not magically certify the model as perfect.
The remaining `institutional_grade=false` state is currently caused by the
protective suspicious-Sharpe block, not by a failed walk-forward audit.

---

## Long-Run Stability Rules

The runtime is designed around a few non-negotiable rules:

1. No order is sent without a durable WAL record first.
2. No XAU trade is admitted during market closure, and reopen conditions must
   pass fresh validation before execution.
3. No stale market payload is allowed to glide through inference silently.
4. No MT5 disconnect should fail quietly without operator visibility.
5. No restart should create duplicate broker intent.

These rules matter more than cosmetic profit curves.

---

## Deployment Note

The live entry point in this worktree is `main.py`.

Environment-specific launchers or checklist files are not currently committed
here, so unattended deployment should be built around the supervisor/runtime
stack in `main.py`, `runmain/bootstrap.py`, and `runmain/supervisors.py`.

---

## Directory Notes

```text
Artifacts/   trained models, reports, validation outputs
Backtest/    model training and institutional validation logic
Bot/         live runtime, motor, Telegram bridge, compatibility facade
core/        risk, blackout, idempotency, config, stability infrastructure
Logs/        rotating runtime logs and heartbeat files
runmain/     bootstrap and supervisor layer
```

---

## Closing View

This codebase is best understood as a machine that must stay coherent under
stress, not merely as a strategy script that looks good in a chart report.

The strongest version of the system is the one that:

- survives weekend gaps cleanly
- restarts without duplicate orders
- stays honest about model uncertainty
- tells the operator the truth quickly
- keeps running without human babysitting

That is the standard this repository is now being shaped around.
