# XAUUSDm Institutional Scalping Stack

An institutional-grade, fully automated scalping system for **XAUUSDm** on Exness MetaTrader 5. The project bundles the entire live workflow—market data ingestion, ensemble-based signal generation, phase-aware risk management, asynchronous trade execution, telemetry, and Telegram supervision—behind a reproducible, configurable codebase.

> **Mission:** capture repeatable high-probability opportunities on gold while enforcing strict Islamic-compliant risk rules (swap-free assumption, leverage ≥ 1:1000) and daily capital protection through adaptive phase gating.

---

## 1. Quick Start

| Step | Command / Action |
| --- | --- | --- |
| 1 | Create `.env` with MT5 & Telegram credentials (see [Configuration](#41-environment-variables)). |
| 2 | `python -m venv .venv && .\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux). |
| 3 | `pip install -r requirements.txt` (install TA-Lib wheel from `Appprogram/` if required). |
| 4 | Launch MT5 terminal with AutoTrading ON, then run `python main.py`. |
| 5 | Open Telegram, `/start`, press **🚀 Оғози Тиҷорат** to enable live trading. |
| 6 | Use `/status`, `/history`, and buttons for monitoring; **🛑 Қатъи Тиҷорат** closes exposure and pauses the engine. |

> **Tip:** On restricted networks set `TELEGRAM_PROXY_URL` or enable offline mode via `TELEGRAM_OFFLINE=1` (details below).

---

## 2. System Overview

```
┌────────────┐                                        ┌─────────────┐    ┌───────────┐
│  config.py │───┐        ┌────────────────────┐      │ RiskManager │──▶│ Trading   │
└────────────┘   │        │ MarketFeed          │      │ (phases,    │    │ Engine    │
                 ├──────▶ │ + FeatureEngine     │ ───▶ │ sizing, SL/ │    │ (queue +  │
┌────────────┐   │        │ + SignalEngine      │      │ TP, logging)│    │ worker)   │
│ mt5_client │───┘        └────────────────────┘      └────┬────────┘    └────┬──────┘
└────────────┘                                                 │               │
                                                               ▼               ▼
                                                       ┌──────────────┐  ┌──────────────┐
                                                       │ ExnessAPI/   │  │ Telegram Bot │
                                                       │ orders.py    │  │ bot.py       │
                                                       └──────────────┘  └──────────────┘
```

The runtime loop is linear and deterministic:

1. **MarketFeed** pulls fresh candles/ticks, microstructure stats, and latency metrics.
2. **FeatureEngine** vectorizes indicators (EMA stack, ADX, ATR, Bollinger, micro-zones, etc.).
3. **SignalEngine** guards, scores, and annotates potential trades with reasons and confidence.
4. **RiskManager** enforces phase logic (A/B/C), adaptive cooldowns, and Islamic constraints before planning SL/TP/lot.
5. **TradingEngine** enqueues MT5 orders asynchronously, monitors fills, and records execution quality.
6. **Telegram Bot** surfaces health, status, and manual controls; logs persist under `Logs/` per module.

Core modules live under:

| Layer | Key File(s) | Highlights |
| --- | --- | --- |
| Configuration | `config.py` | Dataclasses with validated MT5, risk, session, and microstructure parameters. |
| MT5 Access | `mt5_client.py` | Singleton guard that boots MT5, exposes `ensure_mt5()` and `MT5_LOCK`. |
| Data & Features | `DataFeed/market_feed.py`, `Strategies/feature_engine.py` | Sub-second candle cache, tick stats, micro-zones; TA-Lib indicator ensemble. |
| Decisioning | `Strategies/signal_engine.py`, `Strategies/risk_management.py` | Guard rails, ensemble scoring, phase transitions, Kelly-based sizing, signal survival metrics. |
| Execution | `Bot/engine.py`, `ExnessAPI/orders.py` | Queue-based order pipeline, reconciliation, execution metrics feedback. |
| Orchestration | `main.py`, `Bot/bot.py`, `ExnessAPI/history.py` | Lifecycle management, Telegram control plane, MT5 history snapshots. |
| Research | `Strategies/backtest.py` | Offline replay with expectancy tables, regime-conditioned stats, filter ablation, latency-aware PnL simulation. |

---

## 3. Live Trading Mechanics

### 3.1 Engine Loop (Bot/engine.py)

The trading engine runs a fast polling loop (default 150 ms) with the following checkpoints per iteration:

1. **Heartbeat & Telemetry** – Logs balance, equity, drawdown, last_signal, order queue length. (`engine_health.log`)
2. **MT5 Health Gate** – Reconnects with exponential backoff if terminal disconnected.
3. **Market Validation** – Rejects stale/empty OHLC data; triggers recovery if cache unusable.
4. **Order Reconciliation** – Keeps open positions bounded (`max_positions`), notifies RiskManager.
5. **Signal Evaluation** – Calls `SignalEngine.compute()`, logging repeat/blocked signals with reasons.
6. **Risk Decision** – `RiskManager.can_trade()` logs whether the signal passed phase/risk gates.
7. **Order Enqueue** – Enqueues `OrderIntent` with idempotency check; ExecutionWorker handles send/fill logging.
8. **Adaptive Sleep** – Switches to slower cadence when queue congested or latency elevated.

### 3.2 ExecutionWorker Highlights

- Runs as daemon thread consuming the order queue.
- Supports **dry-run** mode that logs simulated fills without touching MT5.
- Records execution metrics (enqueue→send→fill delays, slippage) back into RiskManager for quality analytics.
- Retries MT5 order placement (IOC/FOK) up to 3 attempts with precise retcode logging.

### 3.3 Risk Phases & Islamic Compliance

| Phase | Trigger | Behaviour |
| --- | --- | --- |
| **Phase A** | Daily return < `daily_target_pct` (default 15 %) | Base confidence threshold, standard SL/TP multipliers. |
| **Phase B** | Daily return ≥ target | Tightens gating to `ultra_confidence_min` (default 0.97), boosts TP for runners. |
| **Phase C** | Drawdown ≥ `max_daily_loss_pct` (default 10 %) or post-target retrace | Hard stop: close positions, pause trading until next session. |

Additional protections:

- Leverage check (`islamic_min_leverage`) and swap-free expectation enforced before sizing.
- Adaptive cooldown adjusts signal cadence during drawdowns or latency breaches.
- Microstructure-aware SL/TP derived from current order book; ATR fallback ensures minimum RR.
- Execution breaker suspends trading if latency/slippage anomalies are detected (see `exec_breaker_until`).

**Risk flow (RiskManager):**

1. `evaluate_account_state()` runs every decision cycle to track balance/equity, daily P/L, drawdown, and peak equity protection.
2. `guard_decision()` screens pre-signal conditions (session window, spread, tick quality, hourly signal limit, latency cooldown, rollover blackout, drawdown tolerances).
3. `can_trade()` is called for actionable Buy/Sell signals and rejects trades if confidence is too low for the current phase, market is closed, cooldowns are active, or drawdown/latency/exec breaker limits are hit.
4. `plan_order()` derives entry/SL/TP/lot using micro-price zones when available, ensures broker stop distances, enforces min risk-reward, and scales down lots when stacking toward `max_positions`.
5. `calculate_position_size()` caps risk by equity (`max_risk_per_trade`), respects margin limits, and rounds to broker volume steps.
6. Post-execution, `record_trade()` and `record_execution_metrics()` feed CSV/JSONL logs (`execution_quality.csv`, `signal_survival_*`) for analytics and adaptive controls.

Signal survival state is persisted atomically; corrupted JSON is auto-sanitised to avoid restart loops. Multi-order scaling tightens SL and extends TP for add-on positions while keeping risk bounded.

### 3.4 Signal Cadence & Order Behaviour

- `max_signals_per_day` (default **30**) caps total signals; RiskManager also limits signals to roughly `⌊30 / 24⌋ = 1` new signal per hour (`_hour_window_count`).
- Manual stop (`engine.manual_stop_active()`) pauses trading without restarting the engine; supervisor honours this idle state.
- Engine order queue (`max_exec_queue`, default **10**) throttles concurrent intents; execution worker runs FIFO with retry-on-requote logic.
- `max_positions` (default **3**) bounds simultaneously open trades. Additional entries scale position size (`scale_factor ≥ 0.3`) and adjust TP/SL to bank profit quicker.
- Signals are debounced (≥150 ms) and require state change to avoid spamming identical intents; duplicates are dropped via `_seen_signals` cache.
- RiskManager’s `_daily_signal_count` and `_exec_breaker_until` ensure that prolonged drawdowns or execution anomalies slow the cadence automatically.

**Practical cadence:** In liquid sessions the engine typically emits **4 – 12 qualified signals per day** depending on volatility and guard conditions. High-vol regimes may approach the theoretical 30-signal cap, while low-vol or drawdown days can deliver only a handful of neutralised signals.

### 3.5 Performance Envelope & Expectations

| Metric | Default Limit / Behaviour | Notes |
| --- | --- | --- |
| Daily target | **+15 %** equity gain (`daily_target_pct`) | Phase B activates once exceeded; profit is protected via peak drawdown guard (3 %). |
| Max daily loss | **−10 %** (`max_daily_loss_pct`) | Breach triggers Phase C hard stop until next session. |
| Signals per day | ≤ **30** hard cap | Hourly limiter + guard rails typically yield 4–12 actionable trades. |
| Orders per signal | 1 primary + up to 2 scaled adds | Adds require positive unrealized P/L and confidence ≥ `ultra_confidence_min`. |
| Concurrent orders | ≤ **3** (`max_positions`) | Queue prevents further enqueues once limit reached. |
| Risk per trade | ≤ **2 %** of equity (`max_risk_per_trade`) | Adjusted down during Phase B / drawdowns; respects broker margin. |
| Order type | Market execution only | Entry price pulled from tick/bid-ask snapshot at send time. |
| SL/TP logic | Micro-zone aware with ATR fallback | Broker minimum distances enforced; min RR maintained via `_rr` check. |
| Execution logging | CSV/JSONL + histograms | Latency/slippage tracked for post-trade QA and breaker triggers. |

> **Reality check:** These limits define the *ceiling*. Live performance depends on market structure, spreads, and guard vetoes; expect variability day-to-day. Dry-run telemetry is the recommended baseline for tuning expectations.

---

## 4. Configuration & Deployment

### 4.1 Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `EXNESS_LOGIN` | ✅ | MT5 account login. |
| `EXNESS_PASSWORD` | ✅ | MT5 password (swap-free account recommended). |
| `EXNESS_SERVER` | ✅ | MT5 server string (e.g., `Exness-MT5Real7`). |
| `BOT_TOKEN` | ✅ | Telegram bot token. |
| `ADMIN_ID` | ✅ | Telegram chat ID allowed to control the bot. |
| `TIMEZONE` | ⛔ (default `Asia/Dushanbe`) | Local timezone used for session checks. |
| `TELEGRAM_PROXY_URL` | Optional | HTTPS/SOCKS proxy for Telegram (e.g., `socks5://127.0.0.1:9050`). |
| `TELEGRAM_OFFLINE` | Optional | Set to `1/true/on` to disable outbound Telegram calls (avoids retry storms behind firewalls). |
| `MT5_PATH` / `MT5_PORTABLE` | Optional | Override terminal location or run in portable mode. |
| `DAILY_TARGET_PCT`, `MAX_DAILY_LOSS_PCT`, etc. | Optional | Override defaults defined in `EngineConfig`. |

> See `config.py` for the full catalog of tunable parameters (indicator periods, ATR multipliers, cooldown timers, microstructure thresholds, etc.). All fields are validated on load; missing required envs halt startup with explicit error messages.

### 4.2 Installation Notes

- **Python**: The project targets Python 3.11+. Ensure MT5 Python API (`MetaTrader5`) is installed in the same environment.
- **TA-Lib**: Use platform-specific wheel located under `Appprogram/` if building from source is problematic.
- **Dependencies**: `requirements.txt` includes `pandas`, `numpy`, `talib`, `pyTelegramBotAPI`, `python-dotenv`, etc.
- **Logs Directory**: Created automatically; per-module log files include `engine.log`, `engine_health.log`, `market_feed.log`, `telegram.log`, etc.

### 4.3 Running Modes (main.py)

| Mode | Command | Purpose |
| --- | --- | --- |
| Default | `python main.py` | Starts both trading engine and Telegram bot with lifecycle notifications. |
| Headless | `python main.py --headless` | Runs only the trading engine (no Telegram). Useful for dry-run or network-limited environments. |
| Engine Only | `python main.py --engine-only` | Same as headless (alias). |

Graceful shutdown traps SIGINT/SIGTERM, stops the engine, and drains the order queue before exit.

### 4.4 Deployment Checklist

1. Deploy on a VPS close to Exness MT5 servers (low latency). Recommended 2 vCPU / 4 GB RAM.
2. Install MT5 terminal, log in with swap-free account, enable algo trading.
3. Run in **dry-run** (set `cfg.dry_run=True` inside config) for 2–4 weeks to gather execution quality metrics (`Logs/execution_quality.csv`, etc.).
4. After validation, switch to live mode and use Telegram for oversight.
5. Monitor `Logs/engine_health.log` for pipeline stages (MT5 health, market data, risk decisions) and respond to anomalies promptly.

---

## 5. Monitoring & Observability

| Log File | Source | Contents |
| --- | --- | --- |
| `Logs/engine_health.log` | `TradingEngine` | Heartbeats, pipeline stage outcomes, risk decisions, enqueue success/failure, queue size. |
| `Logs/engine.log` | `TradingEngine` | Error-level stack traces (MT5 failures, recovery errors). |
| `Logs/market_feed.log` | `MarketFeed` | Data fetching errors (e.g., TTL violations, MT5 outages). |
| `Logs/telegram.log` | `Bot` | Telegram API errors, retry exhaustions (disabled when `TELEGRAM_OFFLINE=1`). |
| `Logs/risk_manager.log` | `RiskManager` | Critical risk/lot sizing errors. |
| CSV outputs | `RiskManager` | `signal_survival_log.csv`, `execution_quality.csv`, etc. capture analytics for post-trade review (populated once live trades occur). |

Telegram commands/buttons mirror these metrics in real time (`/status`, “🧭 Мониторинг”). The bot also emits notifications on engine start/stop and critical exceptions (if online).

---

## 6. Backtesting & Research

The offline framework in `Strategies/backtest.py` reuses the live components to run historical data through the same signal → risk → execution planning pipeline. Key outputs:

- **Expectancy Tables** – Confidence buckets with hit rate, expectancy, sample count.
- **Regime Stats** – Trend vs. range, volatility bands, session splits.
- **Filter Ablation** – Impact of each guard (spread, latency, drawdown) on trade count and PnL.
- **Latency/Slippage Modelling** – Simulated execution quality based on recorded live metrics.

Use it to validate parameter changes before deploying live. Further enhancement (e.g., Monte Carlo, stress tests) can be layered without touching the live engine.

---

## 7. Troubleshooting

| Symptom | Cause | Remedy |
| --- | --- | --- |
| `MT5 init failed` | Terminal not running / incorrect credentials | Verify `.env`, ensure MT5 terminal open with AutoTrading ON. |
| `market_feed get_rates error: DataFrame ambiguous` | Pandas truth-value check (fixed) | Already patched to avoid `df or cached`; update repo if still occurring. |
| `telegram.bot ConnectionResetError 10054` | Network blocks Telegram | Configure `TELEGRAM_PROXY_URL` or set `TELEGRAM_OFFLINE=1` to silence retries. Consider running `--engine-only` on firewalled servers. |
| Engine stalls / duplicate signals | Signal idempotency disabled | Ensure `SignalEngine.signal_id` stable; engine logs `ENQUEUE_SKIP | reason=duplicate`. |
| No trades executed | RiskManager blocking | Check `RISK_DECISION` entries in `engine_health.log` for reasons (confidence, cooldown, phase). |

---

## 8. Roadmap & Contribution

Planned enhancements include:

1. **Extended Backtester** – Integrate spread/commission modelling and walk-forward parameter sweeps.
2. **State Persistence** – Snapshot risk state to resume seamlessly after restarts.
3. **Event Bus / State Manager** – Further decouple modules for multi-symbol expansion.
4. **Analytics Dashboard** – Stream execution quality metrics into a lightweight web UI.

Contributions are welcome. Fork the repo, work off a feature branch, and open a PR referencing observed behaviour/logs. Ensure linting/tests pass and documentation updates accompany behavioural changes.

---

## 9. Appendix

### 9.1 Repository Map

```
.
├─ Bot/
│  ├─ bot.py             # Telegram handlers, proxy/offline control, retries, status formatting
│  └─ engine.py          # TradingEngine + ExecutionWorker + health logging
├─ DataFeed/
│  └─ market_feed.py     # Rates/Tick fetch, caching, microstructure stats
├─ Strategies/
│  ├─ feature_engine.py  # Indicator construction, adaptive parameters
│  ├─ indicators.py      # Low-level indicator helpers
│  ├─ risk_management.py # RiskManager, signal survival, execution quality capture
│  ├─ signal_engine.py   # SignalEngine planner-only logic
│  └─ backtest.py        # Offline backtesting scaffold
├─ ExnessAPI/
│  ├─ orders.py          # Order helpers, close_all_position
│  └─ history.py         # Cached MT5 history queries for Telegram
├─ config.py             # EngineConfig / SymbolParams with env loaders
├─ mt5_client.py         # ensure_mt5(), MT5_LOCK, reconnect strategy
├─ main.py               # CLI entrypoint, orchestrates engine + bot
└─ requirements.txt      # Python dependencies
```

### 9.2 Operational Guidelines (Tajik)

- Ҳадафҳои воқеъбинона нигоҳ доред: 5–8 % дар рӯз ба ҳисоби миёна, вале рӯзҳои бефоида имкон доранд.
- VPS наздик ба сервери Exness интихоб кунед, барои латентии паст.
- Пеш аз маблағгузории воқеӣ **dry-run** (режими симулятсионӣ) гузаронед ва логҳои `Logs/`-ро низоман бигиред.
- Фоидаро мунтазам бароред ва ҳаҷми lot-ро тадриҷан зиёд кунед; «scale-out» ба ҳисобҳои дигар барои кам кардани хавф тавсия мешавад.
- Telegram-ро доимо назорат кунед; дар ҳолати хатогиҳои MT5 ё пайваст, фавран тадбир андешед.

---

## 10. Strengths & Limitations

### Strengths

1. **End-to-end automation** – Unified stack from data feed through execution, with Telegram supervision and health logging.
2. **Deterministic guard rails** – RiskManager enforces Islamic-compliant leverage, drawdown locks, signal/hour caps, latency breakers, and execution analytics.
3. **Adaptive trade planning** – Microstructure-aware SL/TP, ATR fallbacks, and multi-order scaling adjust to volatility in real time.
4. **Operational resilience** – Supervisor avoids restart storms, manual stops persist, and all critical state (signal survival, execution metrics) is crash-safe.
5. **Observability-first** – Rich logs, CSV telemetry, and Telegram summaries enable rapid diagnosis and performance review.

### Limitations / Watch-outs

1. **Market dependency** – Tight spreads and consistent liquidity are required; during news or thin markets, guard rails may block most trades.
2. **Infrastructure sensitivity** – MT5 connectivity, VPS latency, and Telegram reachability directly influence cadence and breaker triggers.
3. **Configuration complexity** – Numerous parameters (cooldowns, ATR multipliers, phase thresholds) demand disciplined change management.
4. **Execution-only focus** – No in-built portfolio hedging or multi-symbol diversification; meant for single-symbol XAUUSDm scalping.
5. **Performance variability** – Theoretical targets (15 % daily) rely on ideal conditions; actual returns should be validated via dry-run/backtests before risking capital.

---

**Operate scientifically. Respect phases, monitor telemetry, and let the automation do the heavy lifting.**