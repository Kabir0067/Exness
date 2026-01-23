# Portfolio Scalping System (XAUUSDm + BTCUSDm)

This project is a **production-grade algorithmic trading system** connected to **MetaTrader 5 (MT5)**.  
It runs **two strategies in parallel**:

- **XAUUSDm** (Gold)
- **BTCUSDm** (Bitcoin)

Each pipeline continuously analyzes the market, generates signals, applies risk rules, and sends orders to MT5.

This README is aligned with the current codebase. Environment variables are used **only for 5 secret values**.

---

## What the System Does

- Runs two pipelines (**XAU + BTC**) concurrently; **both are allowed to open trades**.
- Uses multi-timeframe analysis: **M1 + M5 + M15**  
  If M5/M15 are not available → falls back to **M1**.
- Produces **Signal + SL/TP/Lot** via the risk layer, then executes orders in MT5.
- Includes a **Telegram bot** for supervision and control with **professional reporting**.
- Writes **health + diagnostic logs** into `Logs/` with **standardized logger names**.
- Opens **1–3 orders** depending on signal confidence tiers (lot splitting; **SL is never tightened**).
- **Dynamic confidence calculation** (70-96%) based on signal strength, not fixed values.
- Continues analyzing **even while positions are open** (except in Phase C, where analysis is skipped).
- BTC is **24/7**; XAU is **24/5** (controlled by `market_open_24_5`, closed on weekends).
- **Automatic daily reset** — Phase C resets to Phase A at UTC midnight.

---

## Telegram Bot Features

### Commands

- `/start` — Welcome message and menu
- `/status` — System status (trading state, MT5 connection, balance, equity, open positions)
- `/history` — **Full account history report (1 year)** with open positions details
- `/balance` — Current account balance
- `/buttons` — Show control panel menu

### Buttons Menu

- **📊 Фоидаи Имрӯза** — Daily profit report with full statistics
- **📊 Фоидаи Ҳафтаина** — Weekly profit report with full statistics
- **📊 Фоидаи Моҳона** — Monthly profit report with full statistics
- **📋 Дидани Ордерҳои Кушода** — View open positions (inline navigation)
- **🧹 Бастани Ҳамаи Ордерҳо** — Close all positions
- **🚀 Оғози Тиҷорат** — Start trading engine
- **⛔ Қатъи Тиҷорат** — Stop trading engine

### Professional Reporting

All reports use **compact, professional formatting**:
- Clean separators (`━━━━━━━━━━━━━━━━━━━━━━━━`)
- Compact data display (one line per metric)
- Account information included
- Open positions details (ticket, symbol, volume, P&L)
- Win Rate, Profit Factor, and detailed statistics
- Date ranges for all periods

### Notifications

The system notifies the admin:
- When an order is opened (with full details in professional format).
- When risk regime changes (**A/B/C**) with reason.
- When a new trading day starts (daily reset).
- When **trade-lock (hard stop)** is triggered with reason.

---

## Architecture (Core Modules)

### Portfolio Engine
File: `Bot/portfolio_engine.py`

Responsibilities:
- MT5 connection and pipeline orchestration.
- Market data freshness validation.
- Collecting candidates from each signal engine.
- Executing orders via a dedicated worker thread.
- Health + diagnostic logging.

### Strategy Pipelines
- XAU: `StrategiesXau/`
- BTC: `StrategiesBtc/`

Each pipeline includes:
- `xau_indicators.py` / `btc_indicators.py` — feature/indicator engine
- `xau_signal_engine.py` / `btc_signal_engine.py` — signal logic
- `xau_risk_management.py` / `btc_risk_management.py` — risk control + sizing + regimes

### Market Data Feed
- `DataFeed/xau_market_feed.py`
- `DataFeed/btc_market_feed.py`

### MT5 Connection Layer
File: `mt5_client.py`

Includes:
- Initialize/login with retry + recovery.
- Single-instance lock (one Python engine process at a time).

### Order Execution
File: `ExnessAPI/order_execution.py`

Includes:
- Market order sending with SL/TP.
- Retry logic for selected retcodes.
- Execution telemetry: latency and slippage.

### Telegram Bot
File: `Bot/bot.py`

Capabilities:
- Start/stop engine
- Status/balance/open positions/profit-loss
- **Professional reporting** (daily/weekly/monthly/history)
- Admin-only commands: `/tek_prof` and `/stop_ls`
- Notifications for orders, regimes, trade-lock, and new day reset

### API Functions
File: `ExnessAPI/functions.py`

Key functions:
- `get_account_info()` — Get detailed account information
- `get_full_report_day()` — Daily report with statistics
- `get_full_report_week()` — Weekly report with statistics
- `get_full_report_month()` — Monthly report with statistics
- `get_full_report_all()` — **Full history report (1 year)** with open positions
- `has_open_positions()` — Check if any open positions exist (returns True/False)
- `get_all_open_positions()` — Get list of all open positions
- `close_order()` — Close specific order by ticket
- `close_all_position()` — Close all positions
- `set_takeprofit_all_positions_usd()` — Set TP for all positions (USD-based)
- `set_stoploss_all_positions_usd()` — Set SL for all positions (USD-based)

---

## Indicators

Classic indicators (XAU and BTC):
- EMA (short / mid / long / very-long)
- ATR
- RSI
- ADX
- MACD (signal + histogram)
- Bollinger Bands
- Z-Volume (z-score)

Structural/pattern filters:
- FVG (Fair Value Gap)
- Liquidity sweep
- Order block
- Divergence (RSI/MACD)
- Near round-number filter

---

## Runtime Requirements

- **Windows + MetaTrader 5** (logged in).
- **Python 3.10+**
- MT5 **AutoTrading enabled**.

---

## Setup

### 1) Install dependencies
```powershell
pip install -r requirements.txt
```

### 2) `.env` (only 5 environment variables)

```ini
EXNESS_LOGIN=12345678
EXNESS_PASSWORD=your_password
EXNESS_SERVER=Exness-MT5Real
BOT_TOKEN=your_telegram_bot_token
ADMIN_ID=your_telegram_user_id
```

Only these 5 values are read from ENV.
All other configuration is defined in:

* `config_xau.py`
* `config_btc.py`

---

## Run

```powershell
py main.py
```

Flags:

* `--headless` → run without Telegram (VPS mode)
* `--engine-only` → engine only, no bot

---

## Logs & Monitoring

All logs are stored under `Logs/` with **standardized logger names**:

Key files:

* `Logs/portfolio_engine_health.log`
* `Logs/telegram.log`
* `Logs/main.log`
* `Logs/mt5.log`
* `Logs/order_execution.log`
* `Logs/functions.log` — API functions errors
* `Logs/history.log` — History module errors
* `Logs/risk_xau.log` — XAU risk management errors
* `Logs/risk_btc.log` — BTC risk management errors
* `Logs/indicators_xau.log` — XAU indicators errors
* `Logs/indicators_btc.log` — BTC indicators errors
* `Logs/signal_xau.log` — XAU signal engine errors
* `Logs/signal_btc.log` — BTC signal engine errors
* `Logs/feed_xau.log` — XAU market feed errors
* `Logs/feed_btc.log` — BTC market feed errors

**All error logs use module-specific names** for easy identification.

Common fields:

* `ok_xau` / `ok_btc` → market data fresh?
* `reason_xau` / `reason_btc` → block reason
* `q` → execution queue size

---

## Risk & Safety Control

Implemented in:

* `StrategiesXau/xau_risk_management.py`
* `StrategiesBtc/btc_risk_management.py`

### Phase A/B/C Regimes (Automatic Daily Reset)

The system operates in three risk regimes that automatically reset at the start of each UTC day:

#### **Phase A (Normal Trading)**
- **Activation**: Default state, or when daily return < target
- **Confidence Threshold**: 
  - XAU: 85% (`min_confidence_signal`) — **Updated: 80%**
  - BTC: 80% (`min_confidence_signal`) — **Updated: 75%**
- **Behavior**: Standard trading with normal risk parameters

#### **Phase B (Conservative/Protection Mode)**
- **Activation Conditions**:
  1. Daily profit ≥ `daily_target_pct` (default: 10%) → Profit protection
  2. Daily loss ≤ `-daily_loss_b_pct` (XAU: -2%, BTC: -3%) → Loss protection
- **Confidence Threshold**: 
  - XAU: 90% (`ultra_confidence_min`)
  - BTC: 90% (`ultra_confidence_min`)
- **Behavior**: Only high-confidence signals allowed, reduced risk

#### **Phase C (Hard Stop)**
- **Activation**: Daily loss ≤ `-daily_loss_c_pct` (XAU: -5%, BTC: -6%)
- **Logic Fix**: **Now checks `loss_c` first** — if exceeded, immediately goes to Phase C (skips Phase B)
- **Behavior**: 
  - **Trading completely blocked** (no orders, no signal analysis)
  - `plan_order()` returns `None, None, None, None`
  - `can_trade()` and `can_emit_signal()` return `False`
  - Analysis skipped to save CPU resources
- **Reset**: Automatically resets to Phase A at start of new UTC day

### Daily Target Lock (Profit Protection)

If daily profit:
1. Reaches ≥ `daily_target_pct` (e.g., 10%) → Phase B activated
2. Exceeds target by ≥ 0.5% (e.g., 10.5%) → Peak tracked
3. Falls back to ≤ target (e.g., 10%) → **Hard stop triggered** (Phase C)

This protects profits by locking trading when gains are given back.

### Key Safety Features

Includes:

* **Phase-based confidence thresholds** (A/B/C)
* **Fixed Phase transition logic** — checks `loss_c` first, then `loss_b` (prevents double transitions)
* Signal throttling (hour/day limits)
* Latency + spread breakers
* Execution quality monitor
* Optional session filtering
* Cooldowns after fill or high latency
* **Automatic daily reset** (Phase C → Phase A at UTC midnight)
* **Early Phase C check** in signal engine (skips analysis to save CPU)

Important rules:

* **Hard-stop does NOT stop the engine** — it only **locks trading** (trade-lock).
* **Phase C blocks analysis** — no CPU waste on signals that can't trade.
* Full engine stop happens only via the **"Stop Trading"** control.
* **Daily reset is automatic** — no manual intervention needed.

---

## SL/TP and Lot Sizing

Computed by `RiskManager`:

### Lot Size Calculation

**XAU (Gold)**:
- **Balance-based tiered system**:
  - Balance < 200$: `lot = 0.02`, `TP = 2 USD`
  - Balance 200-299$: `lot = 0.03`, `TP = 3 USD`
  - Balance 300-399$: `lot = 0.04`, `TP = 4 USD`
  - Each +100$ balance: `+0.01 lot`, `+1 USD TP`
- Formula: `lot = 0.02 + (step × 0.01)`, `TP = 2 + (step × 1)`
- Example: 4000$ balance → `lot = 0.41`, `TP = 41 USD`

**BTC (Bitcoin)**:
- Similar balance-based tiered system (see `btc_lot_and_takeprofit()` function)

### SL/TP Calculation

* Initial SL/TP comes from **ATR-based logic** (primary) or micro-zones (fallback).
* Then enforced with:

  * **min distance** (broker stops_level + freeze_level + padding)
  * **cost floor** (spread + slippage + market noise)
  * **ATR floor** (minimum ATR multiples)
* Broker constraints (stop/freeze levels) are always respected.
* **USD-based TP**: TP is calculated in USD based on balance tiers, then converted to price.
* **Dynamic SL**: SL uses ATR multiples, adjusted for trend/range regime.

### Multi-Order Behavior

Orders are opened based on **confidence tiers**:

* **1 order**: `confidence >= 80%` (XAU) or `>= 75%` (BTC)
* **2 orders**: `confidence >= 85%`
* **3 orders**: `confidence >= 90%`

Additional safety:
* Requires minimum `net_norm` strength (80% of threshold)
* TP may receive a bonus for multi-orders
* **SL is never tightened** for multi-orders
* Lots are split when allowed

---

## Professional Configuration (config)

Main config files:

* `config_xau.py` (Gold)
* `config_btc.py` (Bitcoin)

### Key Risk Parameters

**Daily Targets & Limits**:
* `daily_target_pct=0.10` (XAU) / `0.10` (BTC) → 10% daily target (Phase B activation, profit-lock logic)
* `daily_loss_b_pct=0.02` (XAU) / `0.03` (BTC) → Loss threshold (Phase A → B)
* `daily_loss_c_pct=0.05` (XAU) / `0.06` (BTC) → Loss threshold (Phase → C, hard stop)
* `max_daily_loss_pct=0.05` (XAU) / `0.03` (BTC) → Maximum daily loss before hard stop
* `enforce_daily_limits=True` → enables A/B/C regime logic
* `ignore_daily_stop_for_trading=False` → enables trade-lock (engine continues, trading locked)

**Signal Quality**:
* `min_confidence_signal=0.80` (XAU) / `0.75` (BTC) → Minimum confidence for Phase A
* `ultra_confidence_min=0.90` (XAU) / `0.90` (BTC) → Minimum confidence for Phase B
* `net_norm_signal_threshold=0.15` (XAU) / `0.12` (BTC) → Minimum signal strength
* `confidence_gain=70.0` (XAU) / `120.0` (BTC) → Confidence calculation multiplier
* `confidence_bias=50.0` → Base confidence value

**Spread & Execution Limits**:
* `exec_max_spread_points=500.0` (XAU) / `5000.0` (BTC) → Maximum spread for execution
* `exec_max_p95_latency_ms=550.0` → Maximum latency (95th percentile)
* `exec_max_p95_slippage_points=20.0` → Maximum slippage (95th percentile)
* `exec_max_ewma_slippage_points=15.0` → Maximum EWMA slippage

**Multi-Order Logic**:
* `multi_order_confidence_tiers=(0.85, 0.90, 0.90)` → Confidence thresholds for 1/2/3 orders
* `multi_order_max_orders=3` → Maximum simultaneous orders
* Multi-order behavior:
  - 80-85% confidence → 1 order
  - 85-90% confidence → 2 orders
  - 90-100% confidence → 3 orders

**Other Important Settings**:
* `max_signals_per_day=0` → unlimited (set > 0 to limit)
* `ignore_external_positions=True` → manual trades do not affect regime/risk state
* `magic=777001` → magic number to identify bot positions
* `protect_drawdown_from_peak_pct=0.20` → Peak drawdown protection (20%)
* `poll_seconds_fast=0.25` (XAU) → Fast polling interval
* `decision_debounce_ms=350.0` (XAU) → Signal debounce time
* `analysis_cooldown_sec=0.80` (XAU) → Analysis cooldown

### Dynamic Confidence System

The system calculates confidence **dynamically** based on signal strength:

* **Not fixed at 96%** — confidence varies from 70-96% based on:
  - `net_norm` strength (signal quality)
  - Spread penalties
  - Tick quality penalties
  - Strength-based caps (granular tiers)
* **Granular caps** prevent weak signals from showing high confidence
* **Strong signals** can reach 96%, but only for extremely strong signals (rare)

---

## Signal Generation Improvements

### XAU Sell Signals
- **Improved logic** — more flexible conditions for Sell signals
- `trend_ok_sell` now uses:
  - `close_p < ema50_l * 1.002` (was 1.005) — more sensitive
  - `adx_l < adx_lo_ltf * 1.5` (was 1.2) — more flexible
  - With `require_stack`: `(close_p < ema_s)` or `(close_p < ema_m)` — allows Sell when price below EMAs
- **Result**: System now generates Sell signals correctly even in downtrends

### BTC Trading
- **Fixed blocking issues**:
  - `meta_barrier_R`: 0.40 → **0.30** (more signals)
  - `tc_bps`: 1.0 → **0.8** (more signals)
  - `daily_loss_c_pct`: 0.05 → **0.06** (better activity)
  - `daily_loss_b_pct`: 0.02 → **0.03** (better activity)
- **Result**: BTC now trades more actively while maintaining safety

---

## Recent Improvements & Fixes

### Phase A/B/C Regimes
- ✅ **Fully functional** — automatic transitions based on daily P&L
- ✅ **Fixed transition logic** — checks `loss_c` first, then `loss_b` (prevents A→B→C double transitions)
- ✅ **Phase C blocks analysis** — saves CPU by skipping signal generation
- ✅ **Automatic daily reset** — Phase C → Phase A at UTC midnight
- ✅ **Proper confidence thresholds** — Phase A (80%/75%), Phase B (90%/90%), Phase C (blocked)

### Dynamic Confidence
- ✅ **Variable confidence** (70-96%) based on signal strength
- ✅ **Granular caps** prevent weak signals from showing high confidence
- ✅ **Not fixed at 96%** — confidence reflects actual signal quality

### Risk Management
- ✅ **Balance-based lot/TP calculation** — deterministic, tiered system
- ✅ **Phase C blocking** — `plan_order()` and `calculate_position_size()` check Phase C
- ✅ **Improved state management** — dataclass structures for better maintainability
- ✅ **Graceful shutdown** — critical data flushed on exit via `atexit`

### Signal Quality
- ✅ **Improved Sell signals for XAU** — better logic for downtrend detection
- ✅ **Fixed BTC blocking** — adjusted meta gate and phase thresholds
- ✅ **Stricter filters** — additional quality checks (ADX, spread, net_norm)
- ✅ **Multi-order logic** — based on confidence tiers (80-85%: 1, 85-90%: 2, 90%+: 3)
- ✅ **Reduced logging spam** — Phase C state changes logged only when state changes

### Telegram Bot & Reporting
- ✅ **Professional message formatting** — compact, clean, business-like style
- ✅ **Full history command** — `/history` shows 1 year of data with open positions
- ✅ **Detailed reports** — daily/weekly/monthly with Win Rate, Profit Factor, account info
- ✅ **Open positions display** — shows ticket, symbol, volume, P&L for up to 10 positions
- ✅ **Improved error handling** — `get_full_report_all()` handles MT5 errors gracefully

### Logging
- ✅ **Standardized logger names** — all modules use descriptive names:
  - `functions` — ExnessAPI/functions.py
  - `history` — ExnessAPI/history.py
  - `risk_xau` — StrategiesXau/xau_risk_management.py
  - `risk_btc` — StrategiesBtc/btc_risk_management.py
  - `indicators_xau` — StrategiesXau/xau_indicators.py
  - `indicators_btc` — StrategiesBtc/btc_indicators.py
  - `signal_xau` — StrategiesXau/xau_signal_engine.py
  - `signal_btc` — StrategiesBtc/btc_signal_engine.py
  - `feed_xau` — DataFeed/xau_market_feed.py
  - `feed_btc` — DataFeed/btc_market_feed.py

### Market Hours
- ✅ **BTC 24/7** — trades continuously
- ✅ **XAU 24/5** — trades Mon-Fri, closed on weekends (correct behavior)

### API Functions
- ✅ **New function**: `has_open_positions()` — checks if any open positions exist (returns True/False)
- ✅ **Improved**: `get_full_report_all()` — now fetches 1 year of history (was 1970-now)
- ✅ **Enhanced**: All report functions include open positions details

---

## Limitations

* If MT5 does not deliver fresh bars → trading is blocked.
* If spread is too high → trading is blocked.
* If M5/M15 are unavailable → system operates on **M1**.
* **Phase C blocks all trading** until daily reset (UTC midnight).

---

## MT5 Notes

* If you see `IPC timeout`, it is an MT5-side issue (not Python logic).
* MT5 must be open, logged in, and AutoTrading enabled.
* If required, define `mt5_path` inside `config_xau.py` / `config_btc.py`.
* **Error handling**: `history_deals_get()` errors are now caught and handled gracefully.

---

## Commands: `/tek_prof` and `/stop_ls`

These commands modify TP/SL for all open positions using **USD distance** (not points).

General formula (per position):

* `profit_per_tick = trade_tick_value * volume`
* `ticks_needed = usd / profit_per_tick`
* `price_delta = ticks_needed * trade_tick_size`

For TP:

* BUY → `TP = open_price + price_delta`
* SELL → `TP = open_price - price_delta`

For SL:

* BUY → `SL = open_price - price_delta`
* SELL → `SL = open_price + price_delta`

If the broker returns `trade_tick_value` or `trade_tick_size` as 0, the calculation is impossible and the position is skipped.

---

## Project Structure

```text
Exness/
├── Bot/
│   ├── portfolio_engine.py
│   └── bot.py
├── DataFeed/
│   ├── xau_market_feed.py
│   └── btc_market_feed.py
├── ExnessAPI/
│   ├── order_execution.py
│   ├── functions.py
│   └── history.py
├── StrategiesXau/
│   ├── xau_indicators.py
│   ├── xau_risk_management.py
│   └── xau_signal_engine.py
├── StrategiesBtc/
│   ├── btc_indicators.py
│   ├── btc_risk_management.py
│   └── btc_signal_engine.py
├── Logs/
├── config_xau.py
├── config_btc.py
└── main.py
```

---

## API Functions Reference

### Account & Positions

- `get_account_info()` → Returns detailed account information (login, server, balance, equity, margin, etc.)
- `get_balance()` → Returns current account balance
- `has_open_positions()` → **NEW**: Returns `True` if any open positions exist, `False` otherwise
- `get_all_open_positions()` → Returns list of all open positions
- `get_order_by_index(index)` → Get position by index (for navigation)
- `get_positions_summary()` → Get summary of all positions

### Reports

- `get_full_report_day(force_refresh=True)` → Daily report with statistics
- `get_full_report_week(force_refresh=True)` → Weekly report with statistics
- `get_full_report_month(force_refresh=True)` → Monthly report with statistics
- `get_full_report_all(force_refresh=True)` → **Full history report (1 year)** with:
  - All closed trades (wins, losses, profit, loss, net P&L)
  - Open positions details (ticket, symbol, volume, P&L)
  - Date range (from 1 year ago to now)
  - Account balance and statistics

### Order Management

- `close_order(ticket)` → Close specific order by ticket
- `close_all_position()` → Close all open positions
- `set_takeprofit_all_positions_usd(usd_profit)` → Set TP for all positions (USD-based)
- `set_stoploss_all_positions_usd(usd_loss)` → Set SL for all positions (USD-based)

---


Python Developer | Django Back-end | XAU - BTC - USD |
Trade Analyst    | Exness MT5      | Global Markets  |

Developed with ❤️ by Gafurov Kabir 📅 2026 | Tajikistan 🇹🇯 
