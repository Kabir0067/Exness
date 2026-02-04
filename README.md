# 🚀 QuantCore Pro: Institutional AI Trading System
## XAUUSDm • BTCUSDm • Exness MT5

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ██████╗ ██████╗ ███████╗ ║
║ ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗██╔════╝ ║
║ ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║     ██║   ██║██████╔╝█████╗   ║
║ ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║     ██║   ██║██╔══██╗██╔══╝   ║
║ ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╗╚██████╔╝██║  ██║███████╗ ║
║  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ║
║                                                                                 ║
║                    🏆 PROFESSIONAL ALGORITHMIC TRADING SYSTEM 🏆                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🌟 Overview

**QuantCore Pro** is a production-grade algorithmic trading system for Exness MetaTrader 5. It executes independent **multi-timeframe scalping strategies** for Gold (XAU) and Bitcoin (BTC) with institutional-level risk management and AI-powered analysis.

### ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🎯 **Dual-Asset Trading** | XAU and BTC operate independently in parallel |
| 🧠 **AI Neural Scoring** | 6-layer ensemble model with divergence detection |
| 📊 **6 Timeframe Analysis** | M1, M5, M15, H1, H4, D1 multi-timeframe fusion |
| 🛡️ **3-Phase Risk Regime** | Adaptive A/B/C system with automatic UTC reset |
| 🎯 **Sniper Filters** | Volume, momentum, spread, and MTF validation gates |
| ⚡ **Non-Blocking I/O** | Telegram decoupled from trading loop |
| 🤖 **Telegram Bot** | Full control panel with real-time notifications |
| 📈 **God Tier Detection** | Rare high-probability entry identification |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUANTCORE PRO ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   MT5 API    │───▶│  Data Feed   │───▶│ Rate Cache   │                   │
│  │   Exness     │    │  M1-D1 Bars  │    │  800 bars    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                    │                   │                          │
│         ▼                    ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    FEATURE ENGINE                            │            │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │            │
│  │  │  EMA    │ │  RSI    │ │  ADX    │ │  MACD   │            │            │
│  │  │ 9/21/50 │ │ Period  │ │ Trend   │ │ Diverg  │            │            │
│  │  │  /200   │ │   14    │ │   14    │ │ 12/26/9 │            │            │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │            │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │            │
│  │  │   FVG   │ │ Liqui-  │ │ Order   │ │ Round   │            │            │
│  │  │  Gaps   │ │ Sweep   │ │ Block   │ │ Numbers │            │            │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                     SIGNAL ENGINE                            │            │
│  │                                                               │            │
│  │  ┌─────────────────┐   ┌─────────────────┐                   │            │
│  │  │ Ensemble Score  │──▶│ MTF Alignment   │                   │            │
│  │  │  net: -1 to +1  │   │  Score: 0-6/6   │                   │            │
│  │  └─────────────────┘   └─────────────────┘                   │            │
│  │          │                     │                              │            │
│  │          ▼                     ▼                              │            │
│  │  ┌─────────────────────────────────────────┐                 │            │
│  │  │         CONFIDENCE CALCULATOR           │                 │            │
│  │  │  Divergence Boost • MTF Boost • Caps    │                 │            │
│  │  │       Output: 0-98% Confidence          │                 │            │
│  │  └─────────────────────────────────────────┘                 │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    SNIPER FILTERS                            │            │
│  │  ✓ Volume Check    ✓ Spread Check    ✓ MTF Gate            │            │
│  │  ✓ Tick Freshness  ✓ ADX Trend       ✓ Anomaly Block       │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    RISK MANAGER                              │            │
│  │  Phase A: Normal  │  Phase B: Protective  │  Phase C: STOP  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                   ORDER EXECUTOR                             │            │
│  │    MT5_LOCK • ATR-Based SL/TP • Retry Logic • Slippage      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Telegram   │◀───│ Notify Queue │◀───│   Engine     │                   │
│  │     Bot      │    │ Fire-Forget  │    │   Control    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## � AI Neural Scoring System

### Multi-Timeframe Analysis (MTF)

The system analyzes **6 timeframes** for each trade decision:

| # | Timeframe | Purpose | Weight |
|---|-----------|---------|--------|
| 1 | **M1** | Entry precision | +1 point |
| 2 | **M5** | Short-term trend | +1 point (+1 if strong ADX) |
| 3 | **M15** | Medium-term trend | +1 point (+1 if strong ADX) |
| 4 | **H1** | HTF trend gate | Block/Allow |
| 5 | **H4** | Macro trend | Analysis |
| 6 | **D1** | Global direction | Analysis |

### MTF Score Interpretation

```
mtf:6/6 = Perfect alignment   → +5% confidence boost
mtf:5/6 = Strong alignment    → +3% confidence boost  
mtf:4/6 = Good alignment      → +1% confidence boost
mtf:2/6 = Weak alignment      → -10% confidence penalty
```

### Ensemble Score Components

| Component | Description | Range |
|-----------|-------------|-------|
| **Net Score** | Weighted indicator fusion | -1.0 to +1.0 |
| **Divergence** | RSI/MACD price divergence | bullish/bearish/none |
| **Confluence** | Sweep + Divergence combo | Boost multiplier |
| **Extreme Guard** | Overbought/oversold filter | Block/Allow |

### Confidence Calculation

```python
# Base confidence from ensemble
net_norm, conf = _ensemble_score(indicators, book, tick_stats)

# Confluence boost (sweep + divergence)
if has_confluence and net_abs >= 0.15:
    conf = min(92, conf * 1.12)  # +12% boost

# MTF alignment adjustment  
if mtf_score >= 6:
    conf = min(98, conf * 1.05)  # Perfect: +5%
elif mtf_score <= 2:
    conf = max(0, conf * 0.90)   # Weak: -10%

# Strength caps
if net_abs < 0.08:  conf = min(80, conf)
if net_abs < 0.12:  conf = min(88, conf)
if net_abs < 0.18:  conf = min(95, conf)
```

### 🎯 God Tier Detection

Rare, high-probability setups identified when:

| Condition | Buy | Sell |
|-----------|-----|------|
| Order Block | bull_ob | bear_ob |
| RSI Zone | < 35 (oversold) | > 65 (overbought) |
| Divergence | Bullish | Bearish |
| H1 Trend | Not bearish | Not bullish |

---

## 🛡️ Risk Management: 3-Phase Regime

The system enforces adaptive risk limits that **reset daily at 00:00 UTC**.

### 🟢 Phase A: Normal Trading

| Parameter | XAU | BTC |
|-----------|-----|-----|
| Confidence Threshold | ≥55% | ≥55% |
| Max Lot | 0.05 | 0.01 |
| Multi-Order | Up to 3 | Up to 2 |
| Daily Loss Limit | 2% | 3% |

### 🟡 Phase B: Protective Mode

**Trigger**: Daily P&L hits ±target OR drawdown exceeds warning threshold

| Parameter | Change |
|-----------|--------|
| Lot Size | Reduced 50% |
| Confidence | ≥75% required |
| Multi-Order | Disabled (max 1) |

### 🔴 Phase C: Hard Stop

**Trigger**: Daily loss exceeds max threshold (5% XAU / 6% BTC)

| Behavior | Description |
|----------|-------------|
| Trading | **Completely blocked** |
| Analysis | Still runs (monitoring mode) |
| Signals | Sent to Telegram (no execution) |
| Reset | Automatic at 00:00 UTC |

---

## 🎯 Sniper Filter System

All signals pass through institutional-grade filters:

### 1. Volume Filter
```python
# Skip check for first 15 seconds of new bar
if bar_age_sec < 15.0:
    pass  # Volume still building
else:
    if current_vol < vol_ma * 0.8:
        return REJECT("low_volume", "sniper_reject")
```

### 2. MTF Gate
```python
# Buy requires M5 bullish AND M15 NOT bearish
trend_ok_buy = m5_bullish and (not m15_bearish)

# Sell requires M5 bearish AND M15 NOT bullish  
trend_ok_sell = m5_bearish and (not m15_bullish)
```

### 3. Spread Filter
```python
if spread_pct > max_spread_pct:
    return REJECT("spread_high", "risk_block")
```

### 4. Tick Freshness
```python
if tick_age_sec > 5.0:
    return REJECT("stale_data", "data_block")
```

### 5. Anomaly Detection
- Range spike detection
- Wick spike (manipulation) 
- Gap jump detection
- Stop-run rejection

---

## 📊 Signal Lifecycle

### Signal Duration
| Timeframe | Validity |
|-----------|----------|
| **M1 Signal** | 1-5 minutes |
| **M5 Confirmation** | 5-15 minutes |
| **M15 Trend** | 15-60 minutes |

### Order Duration
| Scenario | Expected Duration |
|----------|-------------------|
| Active Market (London/NY) | 1-5 minutes |
| Slow Market (Asia) | 5-15 minutes |
| Range Market | 15+ minutes or SL |

### Execution Speed
| Metric | Value |
|--------|-------|
| Signal Generation | 10-30ms |
| Order Placement | <100ms |
| Total Latency | <200ms |

---

## 💬 Telegram Bot Dashboard

### 📱 Control Panel

| Button | Function |
|--------|----------|
| ✅ **Оғоз** | Start trading |
| 🛑 **Қатъ** | Stop trading (monitoring mode) |
| 📊 **Статус** | Engine status |
| 💰 **Баланс** | Account balance |
| 📈 **Таърих** | Trading history |
| 🤖 **AI** | AI analysis menu |

### 📋 Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome + control panel |
| `/status` | Live engine status |
| `/balance` | Account balance |
| `/history` | Full trading history |
| `/ai` | AI market analysis |
| `/buttons` | Show control panel |

### 🔔 Notifications

| Event | Format |
|-------|--------|
| 🟢 **Buy Signal** | Asset, Price, SL/TP, Confidence% |
| � **Sell Signal** | Asset, Price, SL/TP, Confidence% |
| � **Trade Closed** | Profit/Loss, Duration |
| 🔄 **Phase Change** | A→B, B→C with reason |
| 🛑 **Hard Stop** | Automatic block alert |

---

## ⚙️ Technical Specifications

### Signal Engine
| Component | Specification |
|-----------|---------------|
| Timeframes | M1, M5, M15, H1, H4, D1 |
| Indicators | EMA, RSI, ADX, MACD, Bollinger |
| Patterns | FVG, Liquidity Sweep, Order Block |
| Confidence | 0-98% normalized output |

### Order Execution
| Parameter | Value |
|-----------|-------|
| SL/TP Calculation | ATR-based + USD target |
| Default Lot | 0.02 |
| Default TP | +5 USD |
| P95 Latency | <200ms |
| Max Slippage | 20 points |

### Data Pipeline
| Metric | Value |
|--------|-------|
| Loop Interval | ~2 seconds |
| Tick Age Threshold | 5 seconds |
| Bar Cache | 800 bars per asset |
| Dynamic Sleep | Skips when catching up |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- MetaTrader 5 (Exness Terminal)
- Windows OS (MT5 requirement)

### Installation
```bash
git clone <repo>
cd Exness
pip install -r requirements.txt
```

### Configuration (.env)
```ini
EXNESS_LOGIN=12345678
EXNESS_PASSWORD=your_password
EXNESS_SERVER=Exness-MT5Real
BOT_TOKEN=123456:ABC-DEF...
ADMIN_ID=987654321
```

### Run
```bash
# Full mode (with Telegram)
python main.py

# Headless mode (VPS)
python main.py --headless

# Engine only (no Telegram)
python main.py --engine-only
```

---

## 📊 Monitoring & Logs

### Log Files
| File | Content |
|------|---------|
| `portfolio_engine_health.log` | Pipeline stages, signals, orders |
| `portfolio_engine_error.log` | Errors and exceptions |
| `portfolio_engine_diag.jsonl` | Diagnostic JSON data |

### Log Patterns
```
PIPELINE_STAGE | step=market_data ok_xau=True age_xau=0.1s
PIPELINE_STAGE | step=signals asset=XAU signal=Buy confidence=87
ORDER_SELECTED | asset=XAU signal=Buy conf=87 lot=0.02
TRADE_CLOSED | asset=XAU profit=+$5.20 duration=3m
PHASE_CHANGE | asset=XAU old=A new=B reason=daily_target
```

### Understanding Signals
```
reasons=net:-0.380,mtf:1/6,phase:A
        │         │       │
        │         │       └── Risk phase (A=normal)
        │         └── MTF alignment (1 of 6)
        └── Net score (bearish -0.38)
```

---

## ✅ Production Readiness

| Feature | Status | Details |
|---------|--------|---------|
| Monday Wake-Up | ✅ | Auto-detects market open |
| 00:00 UTC Reset | ✅ | Daily stats and phases reset |
| Concurrency | ✅ | `MT5_LOCK` protects all API calls |
| Non-Blocking | ✅ | Telegram decoupled from loop |
| Stale Data Guard | ✅ | 5-second tick freshness |
| Dynamic Sleep | ✅ | Skips sleep when catching up |

---

## ⚠️ Risk Disclaimer

> **HIGH RISK INVESTMENT WARNING**
>
> This software is for educational and research purposes. Financial trading involves significant risk of loss.
>
> - **No Guarantee**: Past performance does not indicate future results
> - **Software Risk**: Bugs, network issues, or broker rejections can cause losses
> - **Market Risk**: Volatile markets can result in rapid capital loss
> - **Liability**: Authors assume no responsibility for financial damages
>
> **USE AT YOUR OWN RISK**

---

## 👨‍💻 Author

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   👤 Gafurov Kabir                                           ║
║   🐍 Python Developer | Django Back-end                      ║
║   📊 XAU • BTC • USD | Trade Analyst                         ║
║   🏢 Exness MT5 | Global Markets                             ║
║   🇹🇯 Tajikistan                                             ║
║   📅 2026                                                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

<div align="center">

### ⚡ Built with precision for institutional-grade execution ⚡

**QuantCore Pro** — *Where AI meets Trading*

</div>