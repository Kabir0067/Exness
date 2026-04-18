<div align="center">

# QuantCore Pro

### Institutional-Grade Algorithmic Trading System

**Production-Ready · Statistically Valid · Execution-Safe · Fail-Closed**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-production--hardened-success)
![Assets](https://img.shields.io/badge/assets-XAUUSDm%20%7C%20BTCUSDm-gold)
![Broker](https://img.shields.io/badge/broker-MetaTrader%205-informational)
![Hardening](https://img.shields.io/badge/institutional--hardening-10%2F10%20sections-critical)

---

*"Backtest performance should decrease. Live performance should become stable and realistic."*

</div>

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Institutional Hardening Matrix](#institutional-hardening-matrix)
- [System Architecture](#system-architecture)
- [Engine Flow (Finite State Machine)](#engine-flow-finite-state-machine)
- [Risk & Execution Layer](#risk--execution-layer)
- [Observability](#observability)
- [Installation](#installation)
- [Configuration Reference](#configuration-reference)
- [Running the System](#running-the-system)
- [Repository Layout](#repository-layout)
- [Failure Modes Survived](#failure-modes-survived)
- [Operational Runbook](#operational-runbook)

---

## Executive Summary

**QuantCore Pro** is a production MetaTrader 5 trading runtime for `XAUUSDm`
(Gold) and `BTCUSDm` (Bitcoin). Originally a bot script, it has been
systematically rebuilt into an **institutional-grade algorithmic trading
system** along ten hardening axes covering mathematical correctness, ML
validity, execution safety, and long-term operational stability.

The system boots a **full live stack** with:

- Strict environment preflight and singleton process lock
- Dedicated MT5 connectivity layer with auto-heal and session recovery
- Deterministic multi-asset **Finite-State-Machine** engine
- Lookahead-free feature engineering and data-integrity validation
- CatBoost models with **probability-calibrated** outputs (Isotonic / Platt)
- Portfolio-level exposure controls and confluence filtering
- **Idempotent** execution queue with **Write-Ahead Log** and crash recovery
- **News-event hard blackout** (NFP / CPI / FOMC, fail-closed)
- **Drift + edge monitoring** with auto-degradation of position sizing
- External **heartbeat watchdog** for deadlock detection
- Telegram control plane for signals, commands, and confirmed entries
- Per-module rotating log files (28 dedicated streams)

---

## Institutional Hardening Matrix

All ten non-negotiable hardening sections have been implemented. Each fix is
documented inline in the referenced files.

| §  | Area                           | Implementation                                                                 | Files |
|----|--------------------------------|--------------------------------------------------------------------------------|-------|
| 1  | **Lookahead / Leakage**        | Train-only thresholds, `ffill(max_gap)` only, no `candles[idx+k]`              | `Backtest/engine.py`, `Backtest/model_train.py`, `DataFeed/*.py` |
| 2  | **Indicator Correctness**      | All EMA / RSI / MACD delegated to **TA-Lib** (identical in training & live)    | `DataFeed/ai_scalp_market_feed.py`, `DataFeed/ai_day_market_feed.py` |
| 3  | **Probability Calibration**    | Isotonic / Platt on held-out validation; Kelly disabled until calibrated       | `Backtest/model_train.py::_fit_probability_calibrator`, `core/utils.py::calibrated_probability`, `Bot/Motor/inference.py` |
| 4  | **Risk Engine**                | SELL trailing anchored `max(entry, price+ATR*m)`; slippage-aware sizing        | `core/utils.py::volatility_trailing_stop`, `core/risk_manager.py::calculate_position_size` |
| 5  | **Execution Safety**           | UUID idempotency keys · persistent WAL · `safe_order_send` · crash reconcile   | `core/idempotency.py`, `Bot/Motor/execution.py`, `Bot/Motor/engine.py` |
| 6  | **Backtest ↔ Live Parity**     | BID/ASK, spread, slippage, commissions, gap-risk, pessimistic fills            | `Backtest/engine.py`, `Backtest/metrics.py` (equity-based returns) |
| 7  | **ML Validation**              | **Purged K-Fold** + embargo ≥ prediction horizon                               | `Backtest/model_train.py::_build_fold_splits` |
| 8  | **Monte Carlo**                | **Stationary Block Bootstrap** (preserves loss clustering)                     | `Backtest/engine.py::_monte_carlo_institutional` |
| 9  | **News Hard Block**            | NFP / CPI / FOMC · ±45 min window · fail-closed                                | `core/news_blackout.py`, `core/risk_manager.py::pre_order_gate` |
| 10 | **Long-Term Stability**        | PSI + KS drift · rolling Sharpe · **CUSUM** on PnL · auto-degrade · watchdog   | `core/stability_monitor.py`, `scripts/watchdog.py` |

> **Design contract:** every fix is deterministic, production-safe, and
> fail-closed by default. No business logic was changed except where it was
> mathematically incorrect.

---

## System Architecture

```
                     ┌──────────────────────────────┐
                     │         main.py              │
                     │ singleton · bootstrap loader │
                     └──────────────┬───────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │           runmain/bootstrap.py        │
                │  env preflight · logging · exhooks    │
                │  per-module log attach · TG wiring    │
                └───────────────────┬───────────────────┘
                                    │
        ┌──────────────── supervisors ─────────────────┐
        │                                              │
        ▼                                              ▼
 ┌────────────────┐                          ┌──────────────────┐
 │  Engine Thread │                          │ Telegram Thread  │
 │ Bot/Motor/…    │                          │  Bot/bot.py      │
 └──────┬─────────┘                          └──────────────────┘
        │
 ┌──────▼─────────────── Portfolio Engine FSM ─────────────────┐
 │  BOOT → DATA_SYNC → ML_INFERENCE → RISK_CALC →              │
 │         EXECUTION_QUEUE → VERIFICATION → (loop)             │
 │                                                             │
 │  Cross-cutting concerns (observational):                    │
 │    · core.idempotency        (WAL · reconcile)              │
 │    · core.news_blackout      (NFP/CPI/FOMC hard block)      │
 │    · core.stability_monitor  (PSI/KS · CUSUM · governor)    │
 │    · core.risk_manager       (per-asset sizing · stops)     │
 │    · core.portfolio_risk     (correlation · exposure)       │
 └─────────────────────────────────────────────────────────────┘
```

### Key Modules

| Layer                | Module                                              | Responsibility |
|----------------------|-----------------------------------------------------|----------------|
| **Entry / Bootstrap** | `main.py`, `runmain/bootstrap.py`                  | Singleton lock, env preflight, logging config, exception hooks |
| **Supervisors**       | `runmain/supervisors.py`, `runmain/gate.py`        | Supervise engine & Telegram threads; model-gate readiness |
| **Engine Core**       | `Bot/Motor/engine.py`, `Bot/Motor/fsm.py`          | Multi-asset orchestration · FSM state transitions |
| **ML Inference**      | `Bot/Motor/inference.py`, `core/model_engine.py`   | CatBoost loading · calibrated-probability scoring |
| **Pipeline**          | `Bot/Motor/pipeline.py`, `core/ml_router.py`       | Signal-confluence filtering (trend, momentum, structure, flow) |
| **Risk**              | `core/risk_manager.py`, `core/portfolio_risk.py`   | Position sizing · SL/TP · daily DD · portfolio exposure |
| **Execution**         | `Bot/Motor/execution.py`, `ExnessAPI/`             | Queue · retries · broker reconciliation · MT5 order send |
| **Safety Cross-cuts** | `core/idempotency.py`, `core/news_blackout.py`, `core/stability_monitor.py` | Institutional hardening sections §5, §9, §10 |
| **Market Data**       | `DataFeed/`                                        | Real-time feeds · AI payload builders (TA-Lib indicators) |
| **AI Analysis**       | `AiAnalysis/`                                      | LLM-assisted context (scalp, intraday, news-sentiment) |
| **Backtest / Train**  | `Backtest/`                                        | Purged K-Fold CV · Stationary Block Bootstrap MC |
| **Observability**     | `log_config.py`, `Logs/`                           | 28 per-module rotating log streams |

---

## Engine Flow (Finite State Machine)

The engine executes a deterministic cycle per asset:

```
 BOOT ──► DATA_SYNC ──► ML_INFERENCE ──► RISK_CALC ──► EXECUTION_QUEUE
                ▲                                              │
                │                                              ▼
                └────────────── VERIFICATION ◄─────────────────┘
                                      │
                                      ▼  (on fault)
                                    HALT ──► AUTO_RESTART (1..3 × 10s)
```

### State-by-state semantics

| State              | Action                                                                       | Guard |
|--------------------|------------------------------------------------------------------------------|-------|
| `BOOT`             | MT5 init · WAL reconcile · symbol spec sync · workers start                  | `MT5_INIT_OK` · `PIPELINES_BUILT` |
| `DATA_SYNC`        | Fetch closed candles (M1/M5/M15/H1/H4/D1) · integrity validation             | freshness · gap check |
| `ML_INFERENCE`     | CatBoost score · calibrated probability · heuristic fallback                 | model-gate passed |
| `RISK_CALC`        | Sizing · SL/TP · pipeline confluence · **news blackout** · **auto-degrade**  | `pre_order_gate` |
| `EXECUTION_QUEUE`  | Enqueue intent · record WAL pending · send `OrderIntent`                     | idempotency `should_send()` |
| `VERIFICATION`     | Confirm via `positions_get` / `history_deals_get` · WAL confirm/fail         | broker ack |
| `HALT`             | Recoverable: attempt auto-restart (≤3). Terminal: notify & stop              | — |

**Institutional properties preserved across every cycle:**

- No order is sent without an idempotency key persisted to WAL first.
- No signal is emitted to Telegram until it passes pipeline confluence
  (prevents *ghost* notifications).
- No position sizing is applied without effective-distance = `|entry-SL| + spread + expected_slippage`.
- No trade opens within ±45 min of NFP / CPI / FOMC.

---

## Risk & Execution Layer

### Position Sizing (`core/risk_manager.py`)

```
base_risk_pct  = config.per_asset_risk
kelly_factor   = disabled (KELLY_DISABLED=1 by default until calibrator verified)
confidence_scale = calibrated_probability(pred, model_alpha_calibration)
effective_stop = |entry − SL| + current_spread + expected_slippage
lot = equity × base_risk_pct × confidence_scale
        / (effective_stop × point_value)
lot = apply_broker_constraints(lot)       # lot_min / lot_step / lot_max
lot = governor.risk_multiplier × lot       # auto-degrade on drift / edge decay
```

### Stop-Loss / Trailing (`core/utils.py::volatility_trailing_stop`)

| Direction | Rule                                          | Invariant |
|-----------|-----------------------------------------------|-----------|
| `BUY`     | `SL = min(entry, price − ATR × multiplier)`   | `SL ≤ entry` (never breakeven-negative) |
| `SELL`    | `SL = max(entry, price + ATR × multiplier)`   | `SL ≥ entry` (never breakeven-negative) |

### Execution Safety (`core/idempotency.py`)

Every order flows through the following sequence:

```
1. intent.order_id (UUID)  ──┐
                             ├──► idempotency_key = uuid4()
2. WAL.record_pending(key)   │
3. should_send(key)?  ── no ──► refuse (no duplicate)
                      │
                      └ yes ──►  mt5.order_send(comment=ticker|key_tail)
                             │
                             ├─ success ──► WAL.record_confirmed(key, ticket)
                             └─ fail    ──► WAL.record_failed(key, reason)
```

On process restart `reconcile_on_startup`:

- Loads `Logs/idempotency_wal.jsonl`
- Walks all `PENDING` / `SENT` entries
- Confirms against `mt5.positions_get()` and `mt5.history_deals_get()`
- Marks entries `CONFIRMED` or `FAILED` — no duplicate ever sent

---

## Observability

### Per-Module Log Streams

Registry-driven. 28 dedicated rotating log files live under `Logs/`.

| Logger                        | File                          | Purpose                                |
|-------------------------------|-------------------------------|----------------------------------------|
| `main`                        | `main.log`                    | Bootstrap · uncaught exceptions (both threads & main) |
| `stdout`                      | `stdout.log`                  | Every `print()` captured via redirector |
| `portfolio.engine.health`     | `portfolio_engine_health.log` | FSM transitions · LIVE_EVIDENCE · symbol sync |
| `portfolio.engine.err`        | `portfolio_engine_error.log`  | Engine errors only                     |
| `portfolio.engine.diag`       | `portfolio_engine_diag.jsonl` | JSON diagnostic stream                 |
| `core.idempotency`            | `idempotency_wal.log`         | WAL reconcile summary · send refusals  |
| `core.news_blackout`          | `news_blackout.log`           | NFP/CPI/FOMC block decisions           |
| `core.stability_monitor`      | `stability_monitor.log`       | PSI/KS drift · CUSUM trips · governor transitions |
| `core.risk_engine`            | `risk_engine.log`             | Sizing · DD · `pre_order_gate` verdicts |
| `core.signal_engine`          | `signal_engine.log`           | Pipeline confluence decisions          |
| `core.model_engine`           | `model_engine.log`            | CatBoost load · inference              |
| `core.data_engine`            | `data_engine.log`             | Feature-engine · candle integrity      |
| `ai.market_feed`              | `ai_market_feed.log`          | XAU + BTC M1–M15 feed liveness         |
| `ai.intraday_market_feed`     | `ai_intraday_market_feed.log` | H1/H4/D1 feed liveness                 |
| `feed_xau` / `feed_btc`       | `xau_market_feed.log` / `btc_market_feed.log` | trading feeds    |
| `order_execution*`            | `order_execution.log`         | Broker send · retry · health           |
| `functions` / `history`       | `exness_api_*.log`            | MT5 API wrappers                       |
| `mt5`                         | `mt5_client.log`              | Terminal auto-heal · session reset     |
| `ExnessAPI.daily_balance`     | `exness_api_daily_balance.log`| Daily balance anchoring                |
| `telegram.bot` / `TeleBot`    | `telegram_bot.log` / `telegram_lib.log` | User commands · replies      |
| `strategies.xau` / `.btc`     | `strategies_*.log`            | Strategy-specific signals              |
| `backtest.*`                  | `backtest_*.log`              | Online-retraining cycles               |
| `ai.*` (analysis)             | `sclaping_ai.log`, `intraday_ai.log`, `sym_news.log`, `analysis_common.log` | LLM analysis |
| `supervisors` · `gate`        | `supervisors.log` · `gate.log`| Supervisor restarts · model-gate       |

> *Files use `delay=True` — they appear only on first write. A zero-byte or
> missing file means the logger simply has not emitted yet.*

### Structured Heartbeat

A compact JSON heartbeat is persisted atomically to `Logs/engine_heartbeat.json`
for optional out-of-process watchdog monitoring:

```json
{"pid": 15768, "ts": 1776518579.768, "status": "alive", "note": "engine_loop"}
```

- **Throttled** to every `ENGINE_HEARTBEAT_MIN_INTERVAL_SEC` (default 5 s).
- **Windows-safe**: retries transient `[WinError 5]` up to 3 × 15 ms.
- **Opt-out**: set `ENGINE_HEARTBEAT_ENABLED=0` if you do not run a watchdog.

### Uncaught Exception Capture

| Source                        | Sink                           |
|-------------------------------|--------------------------------|
| `sys.excepthook`              | `Logs/main.log`                |
| `threading.excepthook`        | `Logs/main.log`                |
| Any `print(...)` call         | `Logs/stdout.log`              |

---

## Installation

### Prerequisites

- Windows 10 / 11 (MetaTrader 5 is Windows-only)
- Python **3.10+**
- MetaTrader 5 terminal installed + logged into an Exness account
- (Optional) TA-Lib binary wheel appropriate for your Python version

### Setup

```powershell
git clone <your-fork-url> quantcore-pro
cd quantcore-pro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### First-Run Checks

```powershell
python -c "import MetaTrader5 as mt5; print(mt5.__version__)"
python -c "import talib; print(talib.__ta_version__)"
python -c "from log_config import configure_module_logs; print(configure_module_logs(), 'loggers attached')"
```

---

## Configuration Reference

All configuration lives in environment variables. Defaults are institutional-safe.

### MT5 Connection

| Variable         | Default        | Purpose                          |
|------------------|----------------|----------------------------------|
| `MT5_LOGIN`      | —              | Broker login (numeric)           |
| `MT5_PASSWORD`   | —              | Broker password                  |
| `MT5_SERVER`     | —              | Broker server (e.g. `Exness-MT5Trial15`) |
| `MT5_PATH`       | auto-detect    | Full path to `terminal64.exe`    |

### Risk / Sizing

| Variable                          | Default | Purpose                                            |
|-----------------------------------|---------|----------------------------------------------------|
| `KELLY_DISABLED`                  | `1`     | `1` = fixed-risk sizing; flip only after calibrator is verified |
| `RISK_PER_TRADE_PCT`              | `0.5`   | Base per-trade risk (% of equity)                  |
| `MAX_DAILY_DRAWDOWN_PCT`          | `2.0`   | Halts new entries if breached                      |

### News Blackout (§9)

| Variable                            | Default                | Purpose |
|-------------------------------------|------------------------|---------|
| `NEWS_CALENDAR_PATH`                | `Artifacts/news_calendar.json` | Path to high-impact events |
| `NEWS_BLACKOUT_WINDOW_MIN`          | `45`                   | ± minutes around event (30–60) |
| `NEWS_BLACKOUT_ALLOW_EMPTY_CALENDAR`| `0`                    | Dev-only; leave `0` in production |
| `NEWS_BLACKOUT_FAIL_OPEN_ON_EXC`    | `0`                    | Fail-closed on unexpected errors |

### Heartbeat / Watchdog (§10)

| Variable                          | Default | Purpose                                            |
|-----------------------------------|---------|----------------------------------------------------|
| `ENGINE_HEARTBEAT_ENABLED`        | `1`     | `0` disables writes entirely                       |
| `ENGINE_HEARTBEAT_MIN_INTERVAL_SEC`| `5.0`  | Throttle between disk writes                       |
| `ENGINE_HEARTBEAT_PATH`           | `Logs/engine_heartbeat.json` | Override location                    |

### Telegram

| Variable                | Default | Purpose                               |
|-------------------------|---------|---------------------------------------|
| `TELEGRAM_BOT_TOKEN`    | —       | Bot token from @BotFather             |
| `TELEGRAM_CHAT_ID`      | —       | Chat / channel receiving notifications |

### Logging

| Variable       | Default | Purpose                                  |
|----------------|---------|------------------------------------------|
| `LOG_DIR`      | `Logs`  | Base directory for all log files         |
| `LOG_LEVEL`    | `INFO`  | Root logger level                        |

---

## Running the System

### Start the full live stack

```powershell
python main.py
```

Startup sequence:

1. Acquire singleton process lock (TCP port `12345`)
2. `bootstrap_runtime(allow_telegram=True)` → env preflight, logging, hooks
3. Attach 28 per-module log handlers via `configure_module_logs()`
4. Run startup model checks / auto-training policy (`runmain/supervisors.py`)
5. Wire signal notifications into the Telegram Bot
6. Spawn the Engine Supervisor thread
7. Spawn the Telegram Supervisor thread (if token configured)
8. Enter the housekeeping / chaos-audit loop

### Graceful stop

- `Ctrl+C` in the terminal
- Telegram `/stop` command (authorised chat only)

### Emergency stop

- Terminate the process (`taskkill /F /PID <pid>`)
- The next startup will automatically reconcile the WAL — no duplicate orders

---

## Repository Layout

```text
main.py                        # Production entrypoint
mt5_client.py                  # MT5 low-level client with auto-heal
log_config.py                  # Centralised logging + module registry
requirements.txt
README.md

runmain/
  bootstrap.py                 # Env preflight · logging · exception hooks
  gate.py                      # Model-gate readiness
  supervisors.py               # Engine + Telegram thread supervisors

Bot/
  bot.py                       # Telegram bot (commands · notifications)
  bot_utils.py                 # Bot helpers · authorised-chat filter
  portfolio_engine.py          # Public portfolio-engine API
  Motor/
    engine.py                  # Top-level orchestration · WAL reconcile
    fsm.py                     # Finite-State-Machine loop + heartbeat
    pipeline.py                # Confluence filters (trend, momentum, structure, flow)
    inference.py               # CatBoost scoring · calibrated probability
    execution.py               # Queue · idempotent send · auto-degrade
    models.py                  # engine.health / engine.err / engine.diag loggers

core/
  config.py                    # Typed config access
  data_integrity.py            # Clock sync · candle freshness · gap checks
  feature_engine.py            # Feature generation (ffill-only, no lookahead)
  ml_router.py                 # Signal routing across models
  model_engine.py              # Model load · cache · version gate
  risk_manager.py              # §4 risk engine · §9 news-blackout gate
  portfolio_risk.py            # Portfolio-level exposure controls
  signal_engine.py             # Signal lifecycle & auditing
  utils.py                     # Trailing stop · calibrated_probability helper
  idempotency.py               # §5 WAL · safe_order_send · reconcile
  news_blackout.py             # §9 NFP/CPI/FOMC hard block
  stability_monitor.py         # §10 PSI/KS · CUSUM · governor · heartbeat

DataFeed/
  xau_market_feed.py           # XAU trading feed (shift=1, closed-only)
  btc_market_feed.py           # BTC trading feed (shift=1, closed-only)
  ai_scalp_market_feed.py      # M1–M15 payload · TA-Lib indicators
  ai_day_market_feed.py        # H1/H4/D1 payload · TA-Lib indicators

AiAnalysis/
  analysis_common.py           # Shared LLM plumbing
  scalp_ai_analys.py           # Scalping-context analyst
  intrd_ai_analys.py           # Intraday-context analyst
  sym_news.py                  # Symbol-specific news synthesis

Backtest/
  engine.py                    # §1 train-only thresholds · §8 block bootstrap
  model_train.py               # §3 probability calibrator · §7 Purged K-Fold
  metrics.py                   # §6 equity-based returns

ExnessAPI/                     # Broker API wrappers (order / history / balance)
Artifacts/                     # Trained models · calibrators · news calendar
Logs/                          # Rotating log files (auto-created)
scripts/                       # External watchdog (optional)
tests/                         # Unit + integration tests
```

---

## Failure Modes Survived

| Failure                                     | Defence                                                                     |
|---------------------------------------------|-----------------------------------------------------------------------------|
| Process crash mid-order-send                | WAL reconcile on next start · no duplicate order                            |
| Broker rejection / network blip             | `safe_order_send` refuses duplicate key · retries with new intent           |
| MT5 terminal `terminal_info_none`           | Soft reset → taskkill → respawn in `mt5_client.py`                          |
| Stale / missing feed bars                   | `CHAOS_AUDIT_WARN` · FSM holds at `DATA_SYNC` until freshness restored      |
| High-impact news release                    | `news_blackout.check()` blocks entries for ±window minutes                  |
| Silent model drift                          | PSI ≥ 0.25 or KS-reject → governor reduces `risk_multiplier`                |
| Sustained loss streak                       | CUSUM on PnL trips → governor cuts sizing auto-graded                       |
| GIL deadlock / DLL hang                     | `scripts/watchdog.py` detects stale heartbeat → terminates process          |
| Log-volume blow-up                          | `RotatingFileHandler` caps each file · log-monitor warns above thresholds   |
| Uncaught exception in worker thread         | `threading.excepthook` routes to `Logs/main.log`                            |
| Singleton-lock race (double-start)          | TCP-port lock · second instance exits with `FATAL`                          |

---

## Operational Runbook

### Healthy startup signature

```
main | ENGINE_MODE | dry_run=False
main | STARTUP_MODE | mode=full_live_stack telegram=enabled
main | SIGNAL_NOTIFIER_WIRED | Bot connected
main | PRODUCTION_BOOT | controller=stopped gate_reason=ok
portfolio.engine.health | MT5_INIT_OK | login=… server=…
portfolio.engine.health | SYMBOL_SPEC_SYNC | asset=XAU …
portfolio.engine.health | SYMBOL_SPEC_SYNC | asset=BTC …
portfolio.engine.health | PIPELINES_BUILT | xau=XAUUSDm btc=BTCUSDm
portfolio.engine.health | IDEMPOTENCY_RECONCILE | confirmed=0 failed=0 still_pending=0
portfolio.engine.health | EXEC_WORKERS_STARTED | count=2
portfolio.engine.health | PORTFOLIO_ENGINE_START | dry_run=False …
portfolio.engine.health | FSM_TRANSITION | BOOT -> DATA_SYNC | reason=boot_ok
```

### Triage checklist

1. `Logs/main.log` — bootstrap errors, uncaught exceptions
2. `Logs/portfolio_engine_health.log` — state transitions, gate verdicts
3. `Logs/portfolio_engine_error.log` — engine-level exceptions
4. `Logs/mt5_client.log` — terminal auto-heal events
5. `Logs/idempotency_wal.log` — WAL reconcile outcome
6. `Logs/news_blackout.log` — blocked entries around events
7. `Logs/stability_monitor.log` — drift / edge / governor state
8. `Logs/order_execution.log` — broker-level send outcomes

### Routine commands

```powershell
# Confirm all 28 loggers attached
python -c "from log_config import configure_module_logs; print(configure_module_logs(force=True))"

# Tail the engine health stream (PowerShell 7)
Get-Content Logs\portfolio_engine_health.log -Wait -Tail 40

# Inspect the WAL
Get-Content Logs\idempotency_wal.jsonl | Select-Object -Last 50
```

---

<div align="center">

### Built for survival, not for a leaderboard

*This README reflects the code as hardened. If a section says the system
does something, `grep -R` will find it.*

</div>
