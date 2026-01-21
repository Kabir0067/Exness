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
- Opens **1–3 orders** depending on signal confidence (lot splitting; **SL is never tightened**).
- Continues analyzing **even while positions are open**.
- BTC is **24/7**; XAU is **24/5** (controlled by `market_open_24_5`).

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

Includes:

* Regimes **A/B/C** (resets on new day)
* Signal throttling (hour/day)
* Latency + spread breakers
* Execution quality monitor
* Optional session filtering
* Cooldowns after fill or high latency

Important rules:

* **Hard-stop does NOT stop the engine** — it only **locks trading** (trade-lock).
* Full engine stop happens only via the **“Stop Trading”** control.

---

## Core Daily Profit-Lock Rule

If daily profit exceeds **10%**, and then **returns back to 10% or below**, the system triggers **trade-lock**.
Trade-lock remains active until the end of the **UTC day**, then resets automatically on the new day.

---

## SL/TP and Lot Sizing

Computed by `RiskManager`:

* Initial SL/TP comes from micro-zones or ATR logic.
* Then enforced with:

  * **min distance**
  * **cost floor**
  * **ATR floor**
* Broker constraints (stop/freeze levels) are always respected.
* Lot sizing is computed from **equity × max_risk_per_trade**, then translated via `mt5.order_calc_profit`.
* Multi-order behavior:

  * TP may receive a bonus
  * SL is never tightened
  * lots are split when allowed

---

## Professional Configuration (config)

Main config files:

* `config_xau.py` (Gold)
* `config_btc.py` (Bitcoin)

Key parameters:

* `daily_target_pct=0.10` → 10% daily target (profit-lock logic)
* `daily_loss_b_pct=0.02` and `daily_loss_c_pct=0.05` → regimes B/C
* `enforce_daily_limits=True` → enables A/B/C regime logic
* `ignore_daily_stop_for_trading=False` → enables trade-lock (engine continues, trading locked)
* `multi_order_confidence_tiers` + `multi_order_max_orders` → 1–3 orders based on signal confidence
* `max_signals_per_day=0` → unlimited
* `ignore_external_positions=True` → manual trades do not affect regime/risk state
* `magic=777001` → magic number to identify bot positions

---

## Limitations

* If MT5 does not deliver fresh bars → trading is blocked.
* If spread is too high → trading is blocked.
* If M5/M15 are unavailable → system operates on **M1**.

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

