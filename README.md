# QuantCore Pro

Institutional-style MetaTrader 5 trading runtime for `XAUUSDm` and `BTCUSDm`, built around audit-first execution, deterministic control, portfolio risk firewalls, pessimistic backtesting, model training audits, and a Telegram control center.

## What This Repository Actually Is

This repository is not a single-file MT5 bot and it is not a bare signal script.

Running `python main.py` boots a full production-style runtime with:

- runtime bootstrap and environment preflight
- optional startup model training / model readiness checks
- deterministic multi-asset trading engine
- Phase 1 market-data integrity audit
- Phase 2 feature integrity audit
- live market regime classification
- signal generation with hard pre-trade barriers
- risk engine, portfolio risk manager, and execution integrity gates
- broker send, reconciliation, and runtime health supervision
- Telegram bot control and monitoring without blocking the engine loop
- chaos-audit and recovery-oriented controller snapshots

The current first-class assets in the live runtime are:

- `XAUUSDm`
- `BTCUSDm`

## End-to-End Runtime Flow

```text
python main.py
  -> startup preflight
  -> MT5 / Telegram availability checks
  -> model readiness gate (+ optional auto-train on startup)
  -> MultiAssetTradingEngine
      -> XAU + BTC data feeds
      -> Phase 1 Data Integrity Audit
      -> feature engineering
      -> Phase 2 Feature Integrity Audit
      -> market regime classification
      -> signal engine
      -> risk engine + portfolio firewall
      -> execution manager
      -> MT5 order send + reconciliation
  -> Telegram supervisor (optional)
  -> controller snapshot, chaos audit, and housekeeping loop
```

## Production Entry Point

The system is designed to be started from:

```bash
python main.py
```

Supported CLI flags:

- `python main.py --headless`
- `python main.py --engine-only`
- `python main.py --dry-run`
- `python main.py --dry-run --engine-only`

Environment toggles used by the real runtime:

| Variable | Effect |
| --- | --- |
| `DRY_RUN=1` | Forces simulation mode. |
| `MONITORING_ONLY=1` | Keeps the system in observation mode with trading disabled. |
| `ALLOW_TG_IN_DRY_RUN=1` | Allows Telegram to stay enabled in dry-run mode. |
| `AUTO_TRAIN_ON_STARTUP=1` | Trains/retrains models automatically when startup policy requires it. |
| `AUTO_TRAIN_ON_STARTUP=0` | Skips startup retraining. |
| `ALLOW_MISSING_TG=1` | If Telegram credentials are missing, the system falls back to engine-only startup instead of hard failing. |
| `AUTO_DRY_RUN_ON_MISSING_ENV=1` | If live broker credentials are missing, startup can auto-fall back to dry-run. |

Operational behavior in `main.py`:

- enforces singleton boot
- performs environment preflight
- wires engine notifiers to Telegram when available
- starts engine supervision thread
- starts Telegram supervision thread when allowed
- exposes a boot-time controller report via `_controller_boot_report()`
- periodically calls `manage_runtime_housekeeping()` and `run_chaos_audit()`
- keeps Telegram and engine separated so the bot does not block the trading loop

## Core Architecture

These are the main existing files that carry the system:

- `main.py`: master orchestrator, startup policy, engine and Telegram supervision, controller boot reporting
- `Bot/Motor/engine.py`: `MultiAssetTradingEngine`, live engine state machine, runtime health, controller snapshot, chaos audit, housekeeping
- `Bot/Motor/execution_manager.py`: order dispatch preparation, portfolio pre-checks, proposed risk computation
- `Bot/Motor/models.py`: portfolio status model used by engine and Telegram
- `Bot/bot.py`: Telegram control center, admin-only command layer, notifications, menus, actions
- `Bot/bot_utils.py`: Telegram keyboards, formatting, summaries, helper limits
- `core/data_integrity.py`: Phase 1 market-data validator and detector set
- `core/feature_engine.py`: feature computation, Phase 2 feature validator, MTF integrity checks, regime classifier
- `core/signal_engine.py`: data and feature firewalls before any live signal result is allowed through
- `core/risk_engine.py`: strategy phase logic, kill-switch, execution barrier, SL/TP geometry checks, lot caps
- `core/portfolio_risk.py`: daily-loss hard stop, drawdown hard stop, exposure and open-risk caps
- `core/model_manager.py`: model metadata persistence, including training audit metadata
- `Backtest/engine.py`: pessimistic backtest engine and backtest audit
- `Backtest/model_train.py`: model training pipeline and training-audit report
- `tests/test_data_integrity_audit.py`: regression suite covering the integrated audit layers

## Audit And Safety Layers

### Phase 1: Data Integrity Audit

Implemented in `core/data_integrity.py` and enforced in `core/signal_engine.py`.

The validator covers:

- missing bars
- duplicate bars
- wrong timezone handling and strict UTC normalization
- future leakage in timestamps
- candle order correctness
- stale ticks / tick freshness
- spread anomalies
- weekend and session-hole detection
- symbol digits / point / contract-size consistency
- broker feed inconsistencies
- gap anomalies
- impossible spikes
- strict OHLC integrity assertions

Exact data states before signal generation:

- `data valid`
- `data stale`
- `data incomplete`
- `abnormal spread`
- `market unusable`

If Phase 1 fails, the signal path is blocked before any trade decision is generated.

### Phase 2: Feature Integrity Audit

Implemented in `core/feature_engine.py` and enforced in `core/signal_engine.py`.

The feature firewall validates:

- EMA reference behavior against observed price envelopes
- RSI mathematical bounds
- MACD line / signal / histogram identity
- ATR non-negativity and true-range consistency
- zero NaN / Inf leakage into the decision layer
- warmup sufficiency before features are considered valid
- scaling / normalization boundaries
- strict no-lookahead behavior for `shift()` and rolling calculations
- MTF alignment from fully closed higher-timeframe candles only
- feature lag alignment

Exact feature states before the decision layer:

- `features valid`
- `nan detected`
- `insufficient warmup`
- `lookahead bias detected`
- `mtf alignment error`
- `scaling error`

If Phase 2 fails, `SignalResult` is returned in a blocked neutral state and does not reach live execution.

### Phase 4 And 5: Risk And Execution Audit

Implemented across `core/risk_engine.py`, `core/portfolio_risk.py`, `Bot/Motor/execution_manager.py`, and `Bot/Motor/engine.py`.

Hard protections include:

- max daily loss hard stop
- max drawdown from peak hard stop
- persistent no-trade-after-hard-stop behavior
- separate risk tracking per asset
- total exposure caps
- per-asset exposure caps
- per-asset open-risk caps
- strategy `A / B / C` phase switching
- cooldown enforcement
- kill-switch enforcement
- portfolio exposure checked before order send
- hard lot-size cap
- SL / TP geometry validation
- minimum stop-distance validation
- execution barrier snapshots for diagnostics

Live open-risk is reconciled from real positions by the engine and proposed risk is computed before order admission by the execution manager.

### Phase 6: Backtest Audit

Implemented in `Backtest/engine.py`.

The backtest engine is intentionally pessimistic and now enforces:

- commission included
- spread included
- slippage included
- realistic fill rules
- session restrictions
- minimum stop-distance rules
- no future-data usage
- no bar-close cheating
- walk-forward split
- strict train / test separation
- out-of-sample validation
- next-bar entry logic through `entry_delay_bars`

Backtest metadata includes `backtest_audit`.

### Phase 7: Model Training Audit

Implemented in `Backtest/model_train.py` and persisted through `core/model_manager.py`.

The training audit reports and gates on:

- feature leakage
- target leakage
- class imbalance
- overfitting
- regime dependency
- asset-specific model quality
- calibration quality
- threshold stability
- retraining cadence
- stale model detection
- out-of-sample degradation

Model metadata includes `training_audit`, and the institutional gate now requires the audit to pass.

### Phase 8: Regime Audit

Implemented in `core/feature_engine.py` and consumed by `core/signal_engine.py`.

Live regime labels currently include:

- `trend strong`
- `trend weak`
- `range`
- `breakout`
- `fake breakout`
- `high volatility`
- `low volatility`
- `session open`
- `session dead hours`
- `news hours`
- `weekend BTC regime`

Higher-timeframe signals are derived from closed candles only to prevent repainting.

### Phase 10: Stress Test Audit

Implemented through the controller and chaos hooks in `Bot/Motor/engine.py` and surfaced through `main.py`.

The runtime includes resilience checks and recovery paths for:

- MT5 disconnect
- internet or processing delay
- stale feed conditions
- missing bars
- spread spikes
- sudden gaps
- slow model response
- duplicate signal storms
- simultaneous multi-asset trigger pressure
- corrupted configuration conditions
- invalid symbol information
- process restart while positions are open
- terminal restart during an open trade
- log growth and housekeeping pressure

The engine exposes:

- `status()`
- `runtime_watchdog_snapshot()`
- `manage_runtime_housekeeping()`
- `run_chaos_audit()`
- `controller_snapshot()`

### Phase 11: Main Orchestration And Bot Control

`main.py` is the production entry point and now boots the realistic system end to end.

The bot is not a side gadget. It is the control center for:

- start / stop behavior
- kill-style trading halt
- monitoring and reporting
- order-management actions
- AI menu access
- helper actions
- operational notifications

## Telegram Bot: Full Feature Map

The Telegram control plane is implemented in `Bot/bot.py` and `Bot/bot_utils.py`.

### Access Model

- admin-only control surface
- unauthorized access attempts trigger an alert back to the admin
- Telegram network calls are wrapped with safe handlers
- typing / UI actions are handled without blocking the engine

### Main Telegram Commands

These are the main commands exposed through `bot_commands()`:

| Command | Real behavior |
| --- | --- |
| `/start` | Opens the base control entry and directs the admin to the control panel. |
| `/buttons` | Opens the main reply-keyboard control panel. |
| `/status` | Shows live system state, including controller and chaos fields. |
| `/ai` | Opens the AI market-analysis menu for XAU and BTC modes. |
| `/balance` | Shows balance / equity account state. |
| `/history` | Shows full trading history and account reporting. |
| `/helpers` | Opens the helper panel for TP/SL mass actions and manual helper orders. |

Additional admin helper commands that exist in the bot:

- `/tek_prof`
- `/stop_ls`

These are quick-action command paths for bulk TP / SL management in USD across open positions.

### Main Reply Keyboard Buttons

The main control keyboard includes these exact buttons:

- `🚀 Оғози Тиҷорат`
- `🛑 Қатъи Тиҷорат`
- `❌ Бастани ҳама ордерҳо`
- `💰 Бастани фоидадорҳо`
- `📋 Дидани Ордерҳои Кушода`
- `📈 Фоидаи Имрӯза`
- `📊 Фоидаи Ҳафтаина`
- `💹 Фоидаи Моҳона`
- `💳 Баланс`
- `📊 Хулосаи Позицияҳо`
- `🔍 Санҷиши Муҳаррик`
- `🛠 Санҷиши Пурраи мотор`

### AI Inline Menu

The AI menu contains these exact options:

- `🥇 AI XAU`
- `₿ AI BTC`
- `📈 XAU рӯзона`
- `📉 BTC рӯзона`

These route into the bot's analysis views for scalp and intraday variants of XAU and BTC.

### Helpers Inline Menu

The helper menu contains these actions:

- `TP`
- `SL`
- `BTC Buy`
- `BTC Sell`
- `XAU Buy`
- `XAU Sell`

Important operational rule:

- manual helper orders are restricted to a single safe order only
- `HELPER_ORDER_COUNTS = (1,)`
- bulk helper stacking is intentionally blocked

USD range limits for the mass TP / SL actions:

- TP range: `1..10` USD
- SL range: `1..10` USD

### Order-Management Features Exposed By The Bot

The bot can:

- start trading
- stop trading
- close all open orders
- close profitable positions only
- view open orders
- navigate open orders one by one
- close an individual order from the order viewer
- close the order-view panel
- show daily, weekly, and monthly profit summaries
- show balance and position summaries
- run engine status checks
- run full engine checks

### Notifications The Engine Sends To Telegram

The engine is wired to Telegram for these non-blocking notifications:

- signal notifications
- order updates
- skipped-order notifications
- phase-change notifications
- engine-stop notifications
- daily-start notifications

### What `/status` Surfaces

The status output is meant to be operational, not cosmetic. It includes current trading and engine context such as:

- controller state
- chaos state
- gate reason
- risk halt reason
- balance
- equity
- drawdown
- daily PnL
- open-position counts
- last-signal context
- queue / engine runtime health

## Controller, Runtime Supervision, And Recovery

The live engine in `Bot/Motor/engine.py` maintains a unified controller snapshot consumed by both `main.py` and Telegram.

Key controller outputs include:

- `controller_state`
- `gate_reason`
- `blocked_assets`
- `risk_halt_reason`
- `chaos_state`
- `uptime_sec`

This controller surface is used for:

- startup boot reports
- periodic supervision probes
- Telegram status reporting
- runtime diagnostics
- graceful restart / recovery-aware behavior

## Risk And Execution Details That Matter

This repository now treats risk as infrastructure rather than a single stop-loss rule.

Important protections already present in code:

- no order is allowed through if SL / TP geometry is mathematically invalid
- no order is allowed through if the minimum stop distance is violated
- no order is allowed through if the lot size breaches the hard cap
- no order is allowed through if portfolio or per-asset risk is already above limits
- no new trade is allowed after a portfolio hard stop trips
- phase logic and drawdown protections remain active while sizing adapts
- data-state and feature-state audits block trade creation before execution

## Backtest And Training Reality

This repository does not assume optimistic fills or lenient model approval.

What is actually enforced:

- transaction costs are included in backtests
- entries are delayed to avoid same-bar cheating
- walk-forward and out-of-sample logic are part of the evaluation flow
- training audit failure can block institutional approval
- stale or overfit models are explicitly flagged
- regime dependency and asset-specific weakness are measured

## Repository Layout

```text
main.py
mt5_client.py
log_config.py
README.md
requirements.txt
test.py

core/
  config.py
  data_integrity.py
  feature_engine.py
  signal_engine.py
  risk_engine.py
  portfolio_risk.py
  model_manager.py

Bot/
  bot.py
  bot_utils.py
  Motor/
    engine.py
    execution_manager.py
    models.py

Backtest/
  engine.py
  model_train.py

DataFeed/
ExnessAPI/
Artifacts/
Logs/
tests/
  test_data_integrity_audit.py
```

## Verification

The current integrated regression commands for this repository are:

```bash
python -m py_compile core\data_integrity.py core\feature_engine.py core\portfolio_risk.py core\risk_engine.py core\signal_engine.py core\model_manager.py Bot\Motor\models.py Bot\Motor\execution_manager.py Bot\Motor\engine.py Backtest\engine.py Backtest\model_train.py main.py Bot\bot.py tests\test_data_integrity_audit.py
python -m unittest tests.test_data_integrity_audit -v
```

Latest verified integrated suite on this branch:

- `29` tests passed

Important note:

- `test.py` is not the production regression gate for the trading runtime; the audit-focused test suite is `tests/test_data_integrity_audit.py`

## Summary

This repository now reflects a realistic trading stack with:

- audit-first data admission
- audit-first feature admission
- regime-aware signal generation
- risk-first execution gating
- portfolio hard-stop enforcement
- pessimistic backtest validation
- model training audit gates
- resilience and chaos supervision
- a Telegram control surface that exposes the real runtime rather than hiding it

That is the actual shape of the system currently present in this codebase.
