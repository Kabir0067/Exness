# Portfolio Scalping System (XAUUSDm + BTCUSDm)

This project is a production-grade algorithmic trading system connected to **MetaTrader 5 (MT5)**.  
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
- Includes a **Telegram bot** for supervision and control.
- Writes **health + diagnostic logs** into `Logs/`.
- Opens **1–3 orders** depending on signal confidence tiers (lot splitting; **SL is never tightened**).
- **Dynamic confidence calculation** (70-96%) based on signal strength, not fixed values.
- Continues analyzing **even while positions are open** (except in Phase C, where analysis is skipped).
- BTC is **24/7**; XAU is **24/5** (controlled by `market_open_24_5`, closed on weekends).
- **Automatic daily reset** — Phase C resets to Phase A at UTC midnight.

---

## Telegram Notifications

The system notifies the admin:

- When an order is opened (with full details).
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
- `indicators.py` — feature/indicator engine
- `signal_engine.py` — signal logic
- `risk_management.py` — risk control + sizing + regimes

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
- Admin-only commands: `/tek_prof` and `/stop_ls`
- Notifications for orders, regimes, trade-lock, and new day reset

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
````

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

All logs are stored under `Logs/`.

Key files:

* `Logs/portfolio_engine_health.log`
* `Logs/telegram.log`
* `Logs/main.log`
* `Logs/mt5.log`
* `Logs/order_execution.log`

Common fields:

* `ok_xau` / `ok_btc` → market data fresh?
* `reason_xau` / `reason_btc` → block reason
* `q` → execution queue size

---

## Risk & Safety Control

Implemented in:

* `StrategiesXau/risk_management.py`
* `StrategiesBtc/risk_management.py`

### Phase A/B/C Regimes (Automatic Daily Reset)

The system operates in three risk regimes that automatically reset at the start of each UTC day:

#### **Phase A (Normal Trading)**
- **Activation**: Default state, or when daily return < target
- **Confidence Threshold**: 
  - XAU: 80% (`min_confidence_signal`)
  - BTC: 75% (`min_confidence_signal`)
- **Behavior**: Standard trading with normal risk parameters

#### **Phase B (Conservative/Protection Mode)**
- **Activation Conditions**:
  1. Daily profit ≥ `daily_target_pct` (default: 20%) → Profit protection
  2. Daily loss ≤ `-daily_loss_b_pct` (default: -2%) → Loss protection
- **Confidence Threshold**: 
  - XAU: 90% (`ultra_confidence_min`)
  - BTC: 92% (`ultra_confidence_min`)
- **Behavior**: Only high-confidence signals allowed, reduced risk

#### **Phase C (Hard Stop)**
- **Activation**: Daily loss ≤ `-daily_loss_c_pct` (default: -5%)
- **Behavior**: 
  - **Trading completely blocked** (no orders, no signal analysis)
  - `plan_order()` returns `None, None, None, None`
  - `can_trade()` and `can_emit_signal()` return `False`
  - Analysis skipped to save CPU resources
- **Reset**: Automatically resets to Phase A at start of new UTC day

### Daily Target Lock (Profit Protection)

If daily profit:
1. Reaches ≥ `daily_target_pct` (e.g., 20%) → Phase B activated
2. Exceeds target by ≥ 0.5% (e.g., 20.5%) → Peak tracked
3. Falls back to ≤ target (e.g., 20%) → **Hard stop triggered** (Phase C)

This protects profits by locking trading when gains are given back.

### Key Safety Features

Includes:

* **Phase-based confidence thresholds** (A/B/C)
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
* `daily_target_pct=0.20` → 20% daily target (Phase B activation, profit-lock logic)
* `daily_loss_b_pct=0.02` → 2% loss threshold (Phase A → B)
* `daily_loss_c_pct=0.05` → 5% loss threshold (Phase → C, hard stop)
* `max_daily_loss_pct=0.10` (XAU) / `0.03` (BTC) → Maximum daily loss before hard stop
* `enforce_daily_limits=True` → enables A/B/C regime logic
* `ignore_daily_stop_for_trading=False` → enables trade-lock (engine continues, trading locked)

**Signal Quality**:
* `min_confidence_signal=0.80` (XAU) / `0.75` (BTC) → Minimum confidence for Phase A
* `ultra_confidence_min=0.90` (XAU) / `0.92` (BTC) → Minimum confidence for Phase B
* `net_norm_signal_threshold=0.10` (XAU) / `0.12` (BTC) → Minimum signal strength
* `confidence_gain=70.0` (XAU) / `120.0` (BTC) → Confidence calculation multiplier
* `confidence_bias=50.0` → Base confidence value

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
* `protect_drawdown_from_peak_pct=0.30` → Peak drawdown protection (30%)

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

## Recent Improvements & Fixes

### Phase A/B/C Regimes
- ✅ **Fully functional** — automatic transitions based on daily P&L
- ✅ **Phase C blocks analysis** — saves CPU by skipping signal generation
- ✅ **Automatic daily reset** — Phase C → Phase A at UTC midnight
- ✅ **Proper confidence thresholds** — Phase A (80%/75%), Phase B (90%/92%), Phase C (blocked)

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
- ✅ **Stricter filters** — additional quality checks (ADX, spread, net_norm)
- ✅ **Multi-order logic** — based on confidence tiers (80-85%: 1, 85-90%: 2, 90%+: 3)
- ✅ **Reduced logging spam** — Phase C state changes logged only when state changes

### Market Hours
- ✅ **BTC 24/7** — trades continuously
- ✅ **XAU 24/5** — trades Mon-Fri, closed on weekends (correct behavior)

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
│   ├── indicators.py
│   ├── risk_management.py
│   └── signal_engine.py
├── StrategiesBtc/
│   ├── indicators.py
│   ├── risk_management.py
│   └── signal_engine.py
├── Logs/
├── config_xau.py
├── config_btc.py
└── main.py



Python Developer | Django Back-end | XAU - BTC - USD |
Trade Analyst    | Exness MT5      | Global Markets  |

Developed with ❤️ by Gafurov Kabir 📅 2026 | 🇹🇯 Tajikistan