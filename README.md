# Dual-Asset Scalping Stack (BTCUSDm & XAUUSDm) — Exness MT5

An institutional-grade, fully automated scalping system for **BTCUSDm** (crypto) and **XAUUSDm** (gold) on Exness MetaTrader 5.  
The repository ships two production pipelines—each with market data ingestion, indicator/feature construction, signal generation, phase-aware risk management, asynchronous trade execution, telemetry, and Telegram supervision—behind one reproducible, configurable codebase.

> **Mission:** capture repeatable high-probability opportunities on BTCUSDm and XAUUSDm while enforcing strict risk constraints and daily capital protection through adaptive phase gating.

> **Risk Notice (mandatory):** This software does **not** guarantee returns. Trading is risky; you can lose capital. Use dry-run + backtests before live deployment.

---

## 1) Quick Start

### 1.1 Base environment

| Step | Command / Action |
| --- | --- |
| 1 | Copy `.env.example` (if provided) → `.env` and fill MT5 + Telegram credentials (see **Configuration** below). |
| 2 | Create/activate venv: `python -m venv .venv` then `source .venv/bin/activate` (Linux/macOS) / `.venv\Scripts\activate` (Windows). |
| 3 | Install deps: `pip install -r requirements.txt` |
| 4 | Launch MT5 terminal, log in, turn **AutoTrading ON** |

### 1.2 Running the gold (XAUUSDm) stack

| Step | Command / Action |
| --- | --- |
| 1 | Ensure `.env` matches the XAUUSDm account (credentials, Telegram bot token, timezone). |
| 2 | Start supervisors + bot: `python main.py` |
| 3 | Telegram: `/start` → press **Оғози Тиҷорат** to enable live trading; use **Қатъи Тиҷорат** to stop |

### 1.3 Running the bitcoin (BTCUSDm) stack

The BTC engine is packaged in `Bot/btc_engine.py`. For now it is launched programmatically:

```python
from Bot.btc_engine import engine as btc_engine

btc_engine.start()
# ... trade loop runs in background thread
# To stop gracefully:
btc_engine.stop()
```

> **Note:** The Telegram bot currently controls the gold stack. To expose BTC start/stop via Telegram, wire the BTC engine into `Bot/bot.py` (see roadmap below).

**Tip (restricted networks):** if Telegram is blocked, set a proxy in `Bot/bot.py` or via the `TELEGRAM_PROXY_URL` environment variable.

---

## 2) Dual-Asset Architecture Overview

| Layer | BTC Files | XAU Files | Notes |
| --- | --- | --- | --- |
| Configuration | `config_btc.py` | `config_xau.py` | Each defines tuned `EngineConfig`/`SymbolParams` for its asset. |
| Market Feed | `DataFeed/btc_feed.py` | `DataFeed/market_feed.py` | Both enforce MT5 symbol readiness, cache bars, compute microstructure stats. |
| Feature Engine | `StrategiesBtc/indicators.py` | `StrategiesXau/indicators.py` | Shared indicator philosophy; BTC adds crypto-specific anomaly signals. |
| Risk Management | `StrategiesBtc/risk_management.py` | `StrategiesXau/risk_management.py` | Each enforces asset-specific drawdown, spread, execution breakers. |
| Signal Engine | `StrategiesBtc/signal_engine.py` | `StrategiesXau/signal_engine.py` | Multitimeframe ensemble scoring with adaptive thresholds. |
| Trading Engine | `Bot/btc_engine.py` | `Bot/xauengine.py` (via `Bot/bot.py` & `main.py`) | BTC engine launched manually; XAU engine supervised + Telegram control. |
| Telegram Ops | _(integrate manually)_ | `Bot/bot.py` | BTC hooks can reuse notifier & command flow. |

### 2.1 Executive summary

1. **Common infrastructure** — `mt5_client.py`, `ExnessAPI/*`, logging, and thread orchestration are asset-agnostic.  
2. **Asset-specific tuning** — Config, indicators, risk, and signal logic live in parallel `StrategiesBtc/*` and `StrategiesXau/*` directories.  
3. **Operational split** — Gold stack already wired into Telegram & supervisors; BTC stack is a drop-in engine awaiting integration (roadmap section below).

---

## 3) System Overview

| Layer | File(s) | What it does |
| --- | --- | --- |
| Configuration | `config.py` | Loads `EngineConfig`, `SymbolParams`, risk/session/microstructure parameters. |
| MT5 Access | `mt5_client.py` | MT5 boot + singleton guard (`ensure_mt5()`), global lock (`MT5_LOCK`). |
| Data & Features | `DataFeed/market_feed.py`, `Strategies/indicators.py` | Market feed caching + feature/indicator construction via `Classic_FeatureEngine`. |
| Decisioning | `Strategies/signal_engine.py`, `Strategies/risk_management.py` | Signal compute + risk gating + lot/SL/TP planning and analytics hooks. |
| Execution | `Bot/engine.py`, `ExnessAPI/orders.py` | Queue-based order intents + MT5 execution, fills, and execution QA hooks. |
| Orchestration | `main.py`, `Bot/bot.py`, `ExnessAPI/history.py` | App lifecycle + Telegram control plane + history/status snapshots. |
| Research | _(not bundled)_ | Bring your own backtesting harness; production engine can be reused in research scripts. |

### 2.1 End-to-End Pipeline (10k foot view)

1. **Market feed refresh** — `MarketFeed` polls MT5 rates/ticks, caches bars, and exposes helper metrics (spread %, volatility, bar age).
2. **Feature build** — `Classic_FeatureEngine` derives multi-timeframe indicators (EMA stack, ATR, structure metrics, micro-zones, momentum scores).
3. **Signal compute** — `SignalEngine.compute(execute=True)` fuses indicators, adaptive regime data, and guard rails into a `SignalResult` with reasons, confidence, regime, and (optionally) planned entry/SL/TP/Lot.
4. **Risk gate** — `RiskManager.guard_decision()` runs pre-signal session/latency/drawdown checks; `can_trade()` is the last gate before enqueue, injecting sizing and cooldown logic based on account state and confidence thresholds.
5. **Order queue** — `TradingEngine._enqueue_order()` deduplicates signals, snapshots book price, and pushes an `OrderIntent` into the bounded execution queue.
6. **Execution worker** — `ExecutionWorker` sanitises volume/stops, retries MT5 `order_send`, and posts structured `ExecutionResult` telemetry back to the engine.
7. **Post-trade analytics** — Risk hooks (`record_trade`, `track_signal_survival`, `record_execution_metrics`) update drawdown, daily phase, and survival analytics for monitoring.

### 2.2 Boot sequence & configuration load

1. **Environment ingestion** (`config.py`)
   - `get_config_from_env()` parses `.env`, converts types, injects defaults (timezone, drawdown limits, queue sizes, Telegram proxy, etc.).
   - `EngineConfig` encapsulates runtime knobs; `SymbolParams` resolves symbol name/timeframes.
2. **Main entrypoint** (`main.py`)
   - Sets up structured logging + log rotation.
   - Instantiates `Notifier` (Telegram dispatcher) and supervisors.
   - Registers POSIX signal handlers (`GracefulShutdown`) to coordinate a clean stop.
3. **Engine supervisor** (`run_engine_supervisor`)
   - Repeatedly tries to start the trading engine with exponential backoff until MT5 is ready.
   - Monitors health (`engine.status()` if available) and restarts on disconnection or fatal errors.
4. **Telegram supervisor** (`run_telegram_supervisor`)
   - Ensures bot commands are registered.
   - Runs infinite polling loop with bounded backoff on network failures.

### 2.3 TradingEngine stack build

When the supervisor calls `engine.start()`:
1. **MT5 boot** — `_init_mt5()` ensures terminal connectivity, account/terminal readiness, and logs failures with tracebacks.
2. **Symbol ensure** — `_ensure_symbol()` guarantees the trading symbol is visible/selected before any requests.
3. **Stack wiring** — `_build_stack()` constructs fresh instances of MarketFeed → FeatureEngine → RiskManager → SignalEngine using the same config.
4. **RiskManager warm-up** — validates required config fields, initialises caches, loads persisted daily state, and restores survival logs (if present).
5. **Execution thread** — `_restart_execution_worker()` starts a daemon worker consuming the order queue tied to the newly built risk manager.

### 2.4 Supervisors & background services

| Thread | Owner | Responsibility |
| --- | --- | --- |
| `engine.loop` | TradingEngine | Core trading pipeline (heartbeat, health checks, signal evaluation, queue management). |
| `execution-worker` | TradingEngine | Sends MT5 deals, attaches SL/TP, retries on transient retcodes, records execution metrics. |
| `engine.supervisor` | main.py | Restarts engine on MT5 disconnect/failure with exponential backoff. |
| `telegram.supervisor` | main.py | Keeps Telegram polling alive, throttles network errors. |
| `notifier` | main.py | Serialises Telegram notifications, enforces rate limits, drains queue on shutdown. |

All MT5 operations are guarded by the global `MT5_LOCK` to avoid concurrent terminal access.

### 2.5 Signal analysis pipeline (fine-grained)

1. **Market data ingestion** — MarketFeed fetches rates/ticks, performing TTL caching and staleness checks.
2. **Indicator suite** — Classic_FeatureEngine computes trend, momentum, mean-reversion, volume, structure, anomaly signals per timeframe.
3. **Guard rails** — RiskManager.guard_decision rejects analysis when spread/latency/drawdown/session constraints fail.
4. **Ensemble score** — SignalEngine integrates indicator families into a normalised bias + confidence metric.
5. **Confluence filters** — Squeeze filter, liquidity sweep, divergence, round-number proximity, conformal abstention, and meta gating.
6. **Stability gate** — Debounce + stability threshold prevents oscillating signals.
7. **Planning** — `_finalize()` pulls adaptive parameters, risk plan (entry/sl/tp/lot via RiskManager.plan_order), micro zones, and outputs SignalResult with reasons.

### 2.6 Risk management logic

- **Daily regime** — Tracks daily start balance/equity, peak, and determines phase A/B/C. Enforces hard stops (target hit, daily loss, drawdown from peak).
- **Cooldown orchestration** — Combines analysis cooldown, post-fill cooldown, latency cooldown, and per-hour/per-day signal caps.
- **Sizing** — `calculate_position_size` ensures R/R, broker limits, equity risk, drawdown adjustments, and optional margin percentage caps.
- **Execution breaker** — Monitors execution anomalies via ExecutionQualityMonitor; triggers `exec_breaker` window holding off new trades.
- **Survival analytics** — Maintains per-order state, flushes to JSON/CSV, and detects SL/TP hits.

### 2.7 Execution & reconciliation

1. **Enqueue** — TradingEngine builds `OrderIntent`, checks for duplicates, missing SL/TP, or invalid price, then pushes into queue.
2. **Execution** — Worker normalises volume/stops, attempts MT5 order_send (with optional retry without stops), sets SL/TP via `TRADE_ACTION_SLTP` if needed.
3. **Telemetry** — On fill, worker calls risk hooks to update survival state, account evaluation, execution metrics, and survival tracking.
4. **Results** — Engine drains execution results and passes them to RiskManager for cooldown updates.
5. **Reconcile** — Periodically fetches open positions, triggers `on_reconcile_positions`, updates survival analytics, and raises alarms if exceeding `max_positions`.

### 2.8 Telemetry, notifications, and graceful shutdown

- **Logging** — Dedicated rotating logs per subsystem (`main`, `engine`, `market_feed`, `risk_manager`, `telegram`, etc.).
- **Telegram notifier** — Non-blocking queue that serialises notifications, throttles repeated errors, and acknowledges drops.
- **Heartbeat** — Every `engine_heartbeat_sec` logs health snapshot, updates risk account state, and feeds survival tick updater.
- **Graceful stop** — `GracefulShutdown` sets stop event, supervisors wind down threads, bot polling stops, engine loop exits, execution worker joins, notifier drains and sends final shutdown message.

---

## 3) Live Trading Mechanics

### 3.1 TradingEngine loop (Bot/engine.py) — exact runtime checkpoints

The engine runs a fast polling loop and emits structured health logs.

**Loop cadence (engine defaults):**
- `poll_seconds_fast`: **0.15s**
- `poll_seconds_slow`: **0.9s**
- `engine_heartbeat_sec`: **5s**
- `market_validate_interval_sec`: **2s**
- `reconcile_interval_sec`: **15s**
- `signal_cooldown_sec`: **2s**
- `max_consecutive_errors`: **10**
- `recover_backoff_sec`: **10s**
- `max_exec_queue`: **10** (queue size is internal to the engine; override by extending `EngineConfig` if required)
- `max_positions`: **3**

**Per iteration:**
1. **Heartbeat**  
   Emits current balance/equity/DD, last signal, queue length, latency metrics in `Logs/engine_health.log`.
2. **MT5 health stage**  
   `PIPELINE_STAGE | step=mt5_health ok=...`  
   If MT5 disconnects, engine escalates recovery.
3. **Market validation stage**  
   `PIPELINE_STAGE | step=market_data ok=...`  
   Rejects stale/empty bars (e.g., bar age > 180s, NaNs, invalid prices).
4. **Reconcile (throttled)**  
   Reads open positions; if open positions exceed `cfg.max_positions`, logs:
   `RECONCILE: positions exceed max_expected | open=X max=Y`  
   (This is an alarm; not an auto-close by default.)
5. **Consume execution results**  
   Pulls `ExecutionResult` from result queue and forwards to `RiskManager.on_execution_result(...)` (if implemented).
6. **Signal evaluation**  
   Calls `SignalEngine.compute(execute=False)`  
   Logs a standardized step line:
   `PIPELINE_STEP | iter=... signal=... prev=... same=... blocked=... conf_raw=... reasons=...`
7. **Risk decision**  
   For actionable signals (`Buy`/`Sell`), calls `RiskManager.can_trade(conf, signal)` and logs:
   `RISK_DECISION | signal=... conf=... allowed=... lot=... phase=... reasons=...`
8. **Order enqueue**  
   If allowed, builds an `OrderIntent` and pushes into a bounded queue (FIFO).  
   If queue is full: `ENQUEUE_FAIL | reason=queue_full ...`
9. **Adaptive throttling**  
   Sleeps fast/slow depending on queue size and measured latency.

---

### 3.2 Order idempotency (anti-duplicate protection)

Order enqueue is idempotent via:
- `signal_id` (from `SignalResult.signal_id`, else generated)
- cooldown window: `signal_cooldown_sec` (default **2 seconds**)
- 60-second cleanup window for seen signals (prevents memory growth)
- explicit log on duplicates:
  `ENQUEUE_SKIP | reason=duplicate signal_id=...`

---

### 3.3 ExecutionWorker (asynchronous MT5 execution)

- Dedicated daemon thread consuming the order queue.
- MT5 calls are protected by `MT5_LOCK` to avoid concurrent terminal access.
- Retries up to **3 attempts** with explicit reason/retcode logging.
- Returns a structured `ExecutionResult` to the engine’s result queue.
- Hooks for analytics (if implemented inside RiskManager):
  - `record_execution_metrics(...)`
  - `record_execution_failure(...)`
  - `record_trade(...)`
  - `track_signal_survival(...)`

### 3.4 Analysis cooldown behaviour

- **Cooldown trigger:** `RiskManager.on_position_opened()` is invoked after every fill (see `record_trade` hook) and applies an analysis cooldown of `cfg.cooldown_seconds`. A larger cooldown is applied when the account returns to flat (`analysis_cooldown_sec`, scaled by drawdown severity).
- **What continues during open positions:** by default the system **does not block fresh analysis strictly because positions remain open**. Instead, follow-on trades are gated by:
  - `signal_cooldown_sec` (engine-level duplicate guard),
  - drawdown/daily phase logic in `can_trade()`,
  - the configurable `max_positions` and the scaling rules inside `plan_order` (lot trimming & SL/TP adjustments for multiple concurrent positions).
- **Optional stricter behaviour:** if you need analytics to pause until the book is flat, add a config attribute such as `analysis_wait_for_flat` and wire it in `RiskManager.can_analyze()` (the infrastructure is already structured for this with `on_reconcile_positions`).

### 3.5 Signal survival telemetry

- **Entry:** every queued order calls `RiskManager.track_signal_survival(...)`, creating an in-memory snapshot and writing an `entry` event to `Logs/signal_survival_updates.jsonl`, plus persisting the active map to `Logs/signal_survival_active.json`.
- **Live updates:**
  - `_heartbeat()` streams the latest MT5 tick to `RiskManager.tick_signal_survival(...)`, updating MFE/MAE and marking SL/TP touches.
  - `_reconcile_positions()` invokes `RiskManager.sync_signal_survival(...)`, refreshing prices from open MT5 positions.
- **Exit:** when a position disappears from MT5 reconciliation, `sync_signal_survival` auto-calls `finalize_signal_survival(...)`, appending a row to `Logs/signal_survival_final.csv` (with headers), emitting an `exit` event to JSONL, and pruning the order from the active cache.
- **Artifacts:**
  - `signal_survival_active.json` — atomic snapshot of all live trades for dashboards/monitoring.
  - `signal_survival_updates.jsonl` — append-only stream (entry/update/exit) suitable for replaying timelines.
  - `signal_survival_final.csv` — append-only dataset of completed trades (entry/exit stats, MFE/MAE, SL/TP hits).


Recommended phase model (your A/B/C design):
- **Phase A (normal)**: base thresholds, standard sizing, standard SL/TP multipliers
- **Phase B (after hitting target)**: tighter confidence gating, reduce risk, protect profits
- **Phase C (daily loss / retrace)**: hard stop for the day (pause trading + close exposure)

Recommended hard limits (defaults if you configured them):
- Daily target: `daily_target_pct` (example: **15%**)
- Max daily loss: `max_daily_loss_pct` (example: **10%**)
- Max risk per trade: `max_risk_per_trade` (example: **2%**)
- Max open positions: `max_positions` (default **3**)

**Important:** your log error
`RECONCILE: positions exceed max_expected | open=4 max=3`
means you manually opened more positions than allowed. Engine flags it so RiskManager can react (or you close extras manually).

---

## 5) Configuration

All runtime parameters are loaded through `get_config_from_env()` in `config.py`.

### 5.1 `.env` (template)

Use the exact variable names implemented in `config.py`. Typical keys look like:

```env
# MT5 (Exness)
EXNESS_LOGIN=...
EXNESS_PASSWORD=...
EXNESS_SERVER=...

# Telegram
BOT_TOKEN=...
ADMIN_ID=...

# Runtime
TIMEZONE=Asia/Dushanbe

# Optional
MT5_PATH=...
TELEGRAM_PROXY_URL=socks5://127.0.0.1:9050

# BTC overrides (optional)
BTC_FIXED_VOLUME=0.02
BTC_MAX_POSITIONS=3

# XAU overrides (optional)
XAU_MAX_POSITIONS=3
XAU_DAILY_TARGET_PCT=0.12
```

### 5.2 High-impact tunables

These directly affect stability and execution quality (all live inside `TradingEngine`; expose them via config only if you customize the engine class):

- poll_seconds_fast, poll_seconds_slow
- market_validate_interval_sec, reconcile_interval_sec, engine_heartbeat_sec
- max_exec_queue, max_positions
- signal_cooldown_sec
- max_consecutive_errors, recover_backoff_sec

### 5.3 Key runtime flags & environment vars

- `USE_PROXY` / `TELEGRAM_PROXY_URL` — optional socks proxy for Telegram when running from restricted networks.
- `MT5_PATH` — override default MetaTrader terminal discovery (useful for VPS automation).
- Custom risk guard toggles can be injected through `EngineConfig` (for example, `analysis_wait_for_flat`, `max_spread_points`, `exec_breaker_sec`). Make sure every flag you depend on is surfaced in `.env` and parsed in `config.py`.

---

## 6) Monitoring & Observability

Logs (default `Logs/`):
- `Logs/engine_health.log` (TradingEngine): pipeline stages, loop steps, risk decisions, enqueue events, queue size, latencies
- `Logs/engine.log` (TradingEngine/Worker): error traces, recovery failures, MT5 exceptions
- `Logs/market_feed.log` (MarketFeed): data fetching issues, cache/age violations
- `Logs/telegram.log` (Telegram bot): API errors, retries, proxy events
- `Logs/risk.log` or `Logs/risk_manager.log` (RiskManager): sizing failures, state persistence warnings

Telegram supervision (minimum):
- `/start` + start/stop buttons
- `/status` > engine state (connected/trading/balance/equity/DD/phase/queue)
- `/history` > last orders + daily P/L snapshot

---

## 7) Backtesting & Research

There is no bundled backtesting module. Reuse the production components in your own research harness and only promote parameters after both backtest **and** dry-run confirm stability.

---

## 8) Troubleshooting

| Symptom | Root cause | Fix |
| --- | --- | --- |
| **MT5 init failed** | Terminal closed / wrong creds / AutoTrading OFF | Open MT5, verify login/server, **enable AutoTrading** (Tools → Options → Expert Advisors → Allow Algo Trading) |
| **`mt5_health_failed:trade_not_allowed`** | AutoTrading disabled in terminal | **Critical:** Click the **AutoTrading** button in MT5 toolbar (must be green/enabled). Without this, the system cannot place orders. |
| `PIPELINE_STAGE step=market_data ok=False` | Market closed / stale bars / feed failure | Wait for market; check feed and symbol select |
| **`RECONCILE: positions exceed max_expected`** | Manual trades or external EA exceeded `max_positions` | **Close extra positions immediately** or increase `max_positions` in config. System logs this as ERROR but does not auto-close. |
| `ENQUEUE_FAIL reason=queue_full` | Execution backlog | Investigate MT5 latency, reduce load |
| Many loop errors + recovery | Unstable MT5 / network | Move VPS closer, reduce terminal load, validate MT5 health |
| Engine restarts with `connected=False trading=True` | MT5 disconnected while engine running | Supervisor auto-restarts engine. Ensure stable internet and MT5 server connectivity. |

---

## 9) Repository Map

- Bot/bot.py — Telegram handlers + supervision UI
- Bot/engine.py — TradingEngine + ExecutionWorker + health logging
- DataFeed/market_feed.py — rates/ticks fetch, caching, bar age checks
- Strategies/indicators.py — `Classic_FeatureEngine` (feature/indicator construction)
- Strategies/risk_management.py — `RiskManager` (phases, sizing, SL/TP, analytics hooks)
- Strategies/signal_engine.py — `SignalEngine.compute(...)` + reasons/confidence
- ExnessAPI/orders.py — order helpers, `close_all_position`, etc.
- ExnessAPI/history.py — cached MT5 history queries for Telegram
- config.py — `get_config_from_env()`, `EngineConfig` / `SymbolParams`
- mt5_client.py — `ensure_mt5()`, `MT5_LOCK`, reconnect strategy
- main.py — entrypoint: orchestrates engine + bot
- requirements.txt — Python dependencies

---

## 10) Operational Guidelines (Tajik)

- VPS-ро наздик ба сервери Exness интихоб кунед: латентӣ = сифати fill.
- Пеш аз live, 2–4 ҳафта dry-run кунед ва engine_health + execution metrics ҷамъ кунед.
- Ҳадафҳо ва лимитҳои рӯзона (target/loss/phase) қоидаи ҳатмӣ ҳастанд — онҳоро вайрон накунед.
- **МУҲИМ:** Дар MT5 ҳатман **AutoTrading**-ро фаъол кунед (тугмаи сабз дар toolbar). Бе ин система ордер намефиристад.
- Агар дастӣ тиҷорат кунед ва `max_positions`-ро зиёд намоед, reconcile ERROR мегирад. Позицияҳои зиёдро фавран пӯшед.
- Агар `mt5_health_failed:trade_not_allowed` дидед, MT5-ро кушоед ва AutoTrading-ро фаъол намоед.
- Ҳар тағйирот: backtest → dry-run → live. Бе ин пайдарпайӣ параметрҳоро иваз накунед.

---

## 11) Asset-specific checklist

| Category | BTCUSDm | XAUUSDm |
| --- | --- | --- |
| Primary config | `config_btc.py` | `config_xau.py` |
| Symbol guard | Hard-coded `BTCUSDm` | Hard-coded `XAUUSDm` |
| Market feed | `DataFeed/btc_feed.py` (crypto microstructure gates) | `DataFeed/market_feed.py` (gold sessions) |
| Spread limits | `spread_limit_pct = 0.0050` (0.50%) | `spread_limit_pct = 0.00018` (0.018%) |
| Circuit breaker | `spread_cb_pct = 0.0075` | `spread_cb_pct = 0.0010` |
| Position sizing | Fixed volume 0.01 lots (tunable) | Configurable via `max_risk_per_trade` + margin checks |
| Execution worker | `BtcExecutionWorker` | `ExecutionWorker` |
| Start/stop | Manual via Python shell (see Quick Start) | Telegram `/start` & `/stop` |
| Logging | `Logs/btc_engine*.log` | `Logs/engine*.log` |

---

## 12) Roadmap / next steps

1. **Telegram integration for BTC** — mirror `/start` / `/stop` commands to control `BtcTradingEngine`. Add health/status panels similar to XAU.
2. **Unified launcher** — refactor `main.py` to accept `--asset btc` / `--asset xau` flags, or run both engines concurrently with isolated queues.
3. **Shared notifier** — once BTC is wired into Telegram, reuse existing notifier queue so both engines report fills, risk events, and survival stats.
4. **Backtesting hooks** — export common indicator/risk interfaces for research harnesses across both assets.
5. **Deployment profiles** — document separate `.env` templates per asset (different accounts/lot sizing, Telegram bot IDs).