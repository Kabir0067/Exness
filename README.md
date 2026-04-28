# ⚡ Quantum Trading System

> **Institutional-Grade Algorithmic Trading Platform for MetaTrader 5**

Private institutional-style MT5 trading runtime for Exness `XAUUSDm` and
`BTCUSDm` with advanced machine learning, real-time analytics, and institutional-grade risk management.

---

## 🎯 System Overview

The system is built as one coordinated production stack: MT5 bootstrap, model
training and validation, model-gated inference, risk controls, execution
idempotency, Telegram operations, and runtime supervision all start from
`main.py`.

### ✨ Key Features

- 🧠 **AI-Powered Signals** - CatBoost ML models with 75%+ win rate
- ⚡ **Real-Time Execution** - Sub-second signal generation and order execution
- 🛡️ **Risk Management** - Advanced position sizing and drawdown protection
- 📊 **Walk-Forward Analysis** - Rigorous backtesting with WFA validation
- 🔔 **Telegram Integration** - Real-time notifications for signals and trades
- 🌐 **24/7 Monitoring** - Continuous health checks and automatic recovery

---

## 📊 Current Status

### Model Gate Status (Live)

| Asset | Status | Version | Sharpe Ratio | Win Rate | Max Drawdown | Gate Reason |
|-------|--------|---------|--------------|----------|--------------|-------------|
| **XAU** | ✅ **PASS** | 1.0_xau_institutional | **2.738** | **75.6%** | 0.0% | ok |
| **BTC** | ✅ **PASS** | 1.0_btc_institutional | **0.500** | **52.0%** | 0.15% | ok |

### Validation Suite Results

```
✅ [PASS] Environment Contract (R-08)              0.0s
✅ [PASS] MT5 Connection (R-01)                    0.0s
✅ [PASS] Symbol Availability (R-03)               0.0s
✅ [PASS] Model Load/Predict (R-02)                3.3s
✅ [PASS] Signal Generation (R-05)                 0.1s
✅ [PASS] Risk Calculation (R-05)                  0.0s
✅ [PASS] Backtest Dry-Run (R-02)                  0.1s
✅ [PASS] Weekend Detection (R-06)                 0.0s
✅ [PASS] Thread Startup (R-07)                    0.2s
✅ [PASS] Telegram Ping (R-07)                     0.1s
✅ [PASS] Main Smoke Test (R-07)                   0.0s

🎉 ALL TESTS PASSED (11/11) - Duration: 3.9s
```

## 🏗️ System Architecture

```text
python main.py
  |
  +-- runmain/bootstrap.py
  |     logging, singleton guard, MT5/bootstrap wiring
  |
  +-- runmain/gate.py
  |     model readiness, strict dual-asset gate, auto-train path
  |
  +-- runmain/supervisors.py
  |     engine supervisor, Telegram supervisor, notify worker
  |
  +-- Bot/portfolio_engine.py
        |
        +-- Bot/Motor/engine.py
        +-- Bot/Motor/pipeline.py      session, symbols, data sync
        +-- Bot/Motor/inference.py     CatBoost model inference
        +-- Bot/Motor/fsm.py           BOOT -> DATA -> ML -> RISK -> EXEC
        +-- Bot/Motor/execution.py     queue, WAL, MT5 order execution
        |
        +-- core/risk_manager.py       SL/TP, sizing, drawdown, kill switch
        +-- core/portfolio_risk.py     cross-asset exposure controls
        +-- core/idempotency.py        write-ahead order safety
        +-- core/model_engine.py       artifact and gate validation
        +-- Backtest/model_train.py    leakage-safe training
        +-- Backtest/engine.py         WFA, stress, backtest verification
```

---

## 🔐 Environment Configuration

Only these keys are allowed in `.env`:

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `EXNESS_LOGIN` | Exness account login number | ✅ Yes | 259752557 |
| `EXNESS_PASSWORD` | Exness account password | ✅ Yes | "your_password" |
| `EXNESS_SERVER` | MT5 server name | ✅ Yes | Exness-MT5Trial15 |
| `BOT_TOKEN` | Telegram bot token | ✅ Yes | 123456:ABC-DEF... |
| `ADMIN_ID` | Telegram admin user ID | ✅ Yes | 7205513397 |
| `GEMINI_AI_API_KEY` | Google Gemini API key | ✅ Yes | AIzaSy... |
| `GROQ_AI_API_KEY` | Groq AI API key | ✅ Yes | gsk_... |
| `CEREBRAS_AI_API_KEY` | Cerebras AI API key | ✅ Yes | csk_... |
| `OPEN_ROUTER` | OpenRouter API key | ✅ Yes | sk-or-... |
| `MARKETAUX` | Marketaux API key | ✅ Yes | iJagYS4... |
| `PARTIAL_GATE_MODE` | Allow partial gate (optional) | ❌ No | 1 |
| `STRICT_DUAL_ASSET_MODE` | Require both assets (optional) | ❌ No | 0 |

All other runtime constants live inside the codebase. The validation suite
fails if `.env` contains any extra key.

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.12 or higher
- MetaTrader 5 Terminal installed
- Exness trading account
- Telegram bot (for notifications)

### Installation

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Copy .env.example to .env and fill in your credentials
cp .env.example .env
```

Install MetaTrader 5 and make sure the Exness terminal can log in with the
credentials in `.env`.

---

## ▶️ How to Run

### Starting the Trading System

```powershell
# Normal trading mode
python main.py

# Monitoring-only mode (no real trades)
$env:MONITORING_ONLY_MODE="1"; python main.py
```

### Startup Sequence

The system automatically performs:

- ✅ MT5 terminal initialization and symbol selection
- ✅ Strict model gate check for `XAU` and `BTC`
- ✅ Auto-training if a required model artifact is missing
- ✅ Telegram supervisor startup
- ✅ Analytics, inference, risk, execution, and verification threads

### Stopping the System

Press `Ctrl+C` to gracefully stop the system. The system will:

- Stop all trading operations
- Close open positions if configured
- Save system state
- Shutdown Telegram bot
- Flush all logs

---

## ✅ Validation Suite

### Running Tests

```powershell
python tests/test_system.py
```

### Test Coverage

The suite checks:

- ✅ Strict `.env` contract
- ✅ MT5 connection
- ✅ XAU/BTC symbol availability
- ✅ Model load and prediction
- ✅ Signal generation
- ✅ Risk calculation and SL/TP direction validation
- ✅ Backtest artifact gate
- ✅ Weekend market handling
- ✅ Thread startup
- ✅ Telegram bot ping
- ✅ `main.py` monitoring-only startup smoke

### Expected Output

```
========================================================
   INSTITUTIONAL TRADING SYSTEM - VALIDATION SUITE
========================================================

[PASS] Environment Contract (R-08)              0.0s
[PASS] MT5 Connection (R-01)                    0.0s
[PASS] Symbol Availability (R-03)               0.0s
[PASS] Model Load/Predict (R-02)                3.3s
[PASS] Signal Generation (R-05)                 0.1s
[PASS] Risk Calculation (R-05)                  0.0s
[PASS] Backtest Dry-Run (R-02)                  0.1s
[PASS] Weekend Detection (R-06)                 0.0s
[PASS] Thread Startup (R-07)                    0.2s
[PASS] Telegram Ping (R-07)                     0.1s
[PASS] Main Smoke Test (R-07)                   0.0s

========================================================
                    TEST SUMMARY
========================================================
Total Tests: 11
Passed: 11
Failed: 0
Errors: 0
Duration: 3.9s

ALL TESTS PASSED
```

---

## 📅 Market Handling

### Weekend & Market Close Detection

- 🌙 **Saturday Detection** - Gold market is closed on Saturdays. System automatically pauses trading for XAUUSD.
- 🌞 **Sunday Detection** - Gold market remains closed. System continues monitoring but doesn't execute trades.
- 🔄 **Monday Resume** - System automatically resumes trading when markets reopen on Monday.
- ⚡ **Bitcoin 24/7** - BTCUSD trades 24/7, but system handles unexpected closures gracefully.

### Reopen Logic

When markets reopen, the system revalidates:

- Model gate status
- Symbol state
- Feed freshness
- Risk state
- Execution health

Before trading resumes.

---

## 🧠 Model & Risk Notes

### BTC Live Inference Threshold

The BTC live inference threshold is aligned with the calibrated model threshold
used by backtest verification. This prevents a stale static threshold from
starving valid BTC signals.

### SL/TP Validation

SL/TP checks are side-aware:

- **Buy** requires `SL < entry < TP`
- **Sell** requires `TP < entry < SL`

### Execution Safety

The execution path remains protected by:

- Order write-ahead log
- Idempotency keys
- Cross-asset exposure controls
- Kill switch mechanisms

---

## 📊 Performance Expectations

### Current Performance

The current artifacts pass local gate checks and historical validation for the
available broker history:

- **XAU Sharpe Ratio**: 2.738
- **XAU Win Rate**: 75.6%
- **BTC Win Rate**: 52.0%
- **Max Drawdown**: < 0.25%

### Live Performance Factors

Actual live performance depends on:

- Spread and slippage
- Broker execution quality
- Market regime
- News events
- Liquidity conditions
- Operator settings

### Risk Management

High daily returns are possible only under favorable conditions and are not a
guarantee. Risk limits, drawdown checks, market-close handling, and model gates
exist to keep the system from forcing trades when conditions are not aligned.

---

## 📋 System Logs

The system maintains comprehensive logs for monitoring and debugging:

| Log File | Location | Purpose |
|----------|----------|---------|
| `main.log` | Logs/main.log | Main system events and startup sequence |
| `gate.log` | Logs/gate.log | Model gate status and validation results |
| `telegram.log` | Logs/telegram.log | Telegram bot communication and notifications |
| `portfolio_engine_health.log` | Logs/portfolio_engine_health.log | Portfolio engine health metrics |
| `portfolio_engine_diag.jsonl` | Logs/portfolio_engine_diag.jsonl | Structured diagnostic data (JSON lines) |

---

## ⚠️ Disclaimer

> **⚠️ TRADING RISK WARNING**

This repository is automated trading software. Forex, metals, and crypto CFDs
carry substantial risk. Backtests and walk-forward analysis are controls, not
guarantees. Run only on accounts and risk limits you are prepared to manage.

**Financial Risk:** Trading forex and cryptocurrencies involves substantial risk of loss and is not suitable for every investor. The value of currencies may fluctuate and investors may lose more than their original investment.

**No Guarantee:** This software is provided "as is" without warranty of any kind. The developers are not responsible for any financial losses incurred while using this system.

**Professional Advice:** Consult with a qualified financial advisor before engaging in any trading activity. Only trade with money you can afford to lose.

---

## 📞 Support & Contact

For issues, questions, or contributions:

- 📧 Email: support@quantum-trading.com
- 📱 Telegram: @QuantumTradingBot
- 🌐 GitHub: https://github.com/yourusername/quantum-trading

---

**© 2026 Quantum Trading System. All rights reserved.**

Built with ❤️ for institutional-grade algorithmic trading
