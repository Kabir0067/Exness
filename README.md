<div align="center">

# EX Trading System

### Private Quant Runtime for XAUUSDm and BTCUSDm

**Session-Aware · WAL-Protected · Self-Healing · Guardian-Monitored**

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

`core/session_manager.py` is the session source of truth.

- `XAU` respects the Friday close to Sunday reopen window.
- `BTC` remains tradable as a 24/7 asset.
- the first post-gap window is blocked by `SESSION_POST_GAP_COOLDOWN_SEC`
  with a default of `1200` seconds
- the cooldown state survives hard restarts through a persisted marker file

This means the runtime will not treat Sunday reopen as a normal continuous feed.
It explicitly delays fresh XAU entries until the post-gap noise window expires.

### 2. Data and Gap Defense

The live engine rejects unsafe input before inference or execution:

- stale tick / bar input is blocked in `Bot/Motor/pipeline.py`
- large baseline discontinuities reset the short-term pipeline cache
- closed-session assets are reconciled without being traded
- pending execution state is rechecked while the engine waits out closed sessions

The intent is simple: no stale carry-over and no quiet reuse of pre-gap state.

### 3. Execution Safety

Order execution is protected by `core/idempotency.py`.

- each order receives a unique idempotency key
- a write-ahead log is written before broker submission
- restart reconciliation confirms or fails in-flight orders
- WAL auto-compaction prevents unbounded growth over long uptime

If the process dies between send and broker acknowledgement, recovery happens
from durable state rather than memory assumptions.

### 4. Guardian Visibility

`Bot/guardian.py` acts as the operational watchtower.

- heartbeat every 4 hours
- trade alerts with execution context and P/L snapshot
- emergency alerts for MT5 disconnects
- emergency alerts for slippage breaches
- session close / reopen alerts, including post-gap cooldown notice

This module is observational only. It does not place trades.

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
  -> runmain/supervisors.py
  -> Bot/portfolio_engine.py
       -> Bot/Motor/engine.py
       -> Bot/Motor/fsm.py
       -> Bot/Motor/pipeline.py
       -> Bot/Motor/inference.py
       -> Bot/Motor/execution.py
  -> core/risk_manager.py
  -> core/idempotency.py
  -> core/session_manager.py
  -> core/news_blackout.py
  -> core/stability_monitor.py
  -> Bot/guardian.py
```

Key responsibilities:

- `main.py`: production entry point
- `runmain/bootstrap.py`: logging, runtime wiring, process preflight
- `Bot/Motor/fsm.py`: deterministic live cycle
- `Bot/Motor/pipeline.py`: market validation, active-asset logic, stale-data control
- `Bot/Motor/execution.py`: queueing, broker submission, result reconciliation
- `core/risk_manager.py`: live pre-order gate, sizing, hard stops
- `core/session_manager.py`: session close / reopen / cooldown policy
- `core/idempotency.py`: WAL, replay safety, startup reconcile
- `Bot/guardian.py`: heartbeats and operator-facing alerts

---

## Model State

The repository contains institutional backtest and training artifacts for both
core assets. The current practical reading is:

- `BTC` remains gate-verified
- `XAU` now passes full walk-forward validation in the non-fast path
- `XAU institutional_grade` still remains `false` when Sharpe breaches the
  suspicious overfit cap

That last point is deliberate. A very high backtest Sharpe is treated as a risk
signal, not as a trophy. The system currently prefers stability skepticism over
blind promotion of the model grade.

---

## XAU Status Note

The latest hardening pass closed the most important XAU reliability gaps:

- regime-aware training and live inference are aligned
- walk-forward windows pass in the full non-fast path
- weekend session handling is now coordinated by a single source of truth
- Sunday reopen no longer relies on naive weekday-only logic
- post-gap cooldown is enforced before live order admission

This improves survivability. It does not magically certify the model as perfect.
The remaining `institutional_grade=false` state is currently caused by the
protective suspicious-Sharpe block, not by a failed walk-forward audit.

---

## Long-Run Stability Rules

The runtime is designed around a few non-negotiable rules:

1. No order is sent without a durable WAL record first.
2. No XAU trade is admitted during the weekend closure or immediate reopen shock.
3. No stale market payload is allowed to glide through inference silently.
4. No MT5 disconnect should fail quietly without operator visibility.
5. No restart should create duplicate broker intent.

These rules matter more than cosmetic profit curves.

---

## Linux Deployment Artifacts

Two operational artifacts are included for unattended deployment:

- [start_prod.sh](/C:/Users/Kabir/Desktop/Exness/start_prod.sh)
- [DEPLOYMENT_CHECKLIST.md](/C:/Users/Kabir/Desktop/Exness/DEPLOYMENT_CHECKLIST.md)

`start_prod.sh` is a systemd-friendly launcher with conservative production
defaults. The checklist is intended as the final gate before remote unattended
runtime.

---

## Directory Notes

```text
Artifacts/   trained models, reports, validation outputs
Backtest/    model training and institutional validation logic
Bot/         live runtime, execution, Guardian, Telegram bridge
core/        risk, session, idempotency, config, stability infrastructure
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
