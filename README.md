<div align="center">

```
        ╔════════════════════════════════════════════════════════════════════════════════════════════╗
        ║                                                                                            ║
        ║         ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ██████╗ ██████╗ ███████╗      ║
        ║        ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗██╔════╝      ║
        ║        ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║     ██║   ██║██████╔╝█████╗        ║
        ║        ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║     ██║   ██║██╔══██╗██╔══╝        ║
        ║        ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║     ██║   ██║██╔══██╗██╔══╝        ║
        ║        ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╗╚██████╔╝██║  ██║███████╗      ║
        ║         ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝      ║
        ║                                                                                            ║
        ║                                   ██████╗ ██████╗  ██████╗                                 ║
        ║                                   ██╔══██╗██╔══██╗██╔═══██╗                                ║
        ║                                   ██████╔╝██████╔╝██║   ██║                                ║
        ║                                   ██╔═══╝ ██╔══██╗██║   ██║                                ║
        ║                                   ██║     ██║  ██║╚██████╔╝                                ║
        ║                                   ╚═╝     ╚═╝  ╚═╝ ╚═════╝                                 ║
        ║                                                                                            ║
        ║                                INSTITUTIONAL AI TRADING SYSTEM                             ║
        ║                               Version 2.0 - Autonomous Sniper Edition                      ║
        ║                                                                                            ║
        ║                               AI-Powered Gold & Bitcoin Trading                            ║
        ║                                   Sub-200ms Execution Speed                                ║
        ║                               Institutional-Grade Risk Management                          ║
        ║                                                                                            ║
        ╚════════════════════════════════════════════════════════════════════════════════════════════╝
```

<br/>

<!-- ANIMATED BADGES -->
<a href="#"><img src="https://img.shields.io/badge/💰_GOLD-XAUUSDm-FFD700?style=for-the-badge&labelColor=000000" /></a>
<a href="#"><img src="https://img.shields.io/badge/₿_BITCOIN-BTCUSDm-F7931A?style=for-the-badge&labelColor=000000" /></a>
<a href="#"><img src="https://img.shields.io/badge/🏦_BROKER-Exness_MT5-00D4FF?style=for-the-badge&labelColor=000000" /></a>
<a href="#"><img src="https://img.shields.io/badge/🦁_MODE-SNIPER-FF0000?style=for-the-badge&labelColor=000000" /></a>

<br/><br/>

<!-- TECH STACK BADGES -->
<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/MetaTrader-5-0078D4?style=flat-square&logo=metatrader&logoColor=white" />
<img src="https://img.shields.io/badge/CatBoost-ML_Engine-FFCC00?style=flat-square&logo=catboost&logoColor=black" />
<img src="https://img.shields.io/badge/Telegram-Bot_API-26A5E4?style=flat-square&logo=telegram&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Production-00FF88?style=flat-square" />

<br/><br/>

<!-- STATS CARDS -->
<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/📈_Assets-Multi--Core-00D4FF?style=for-the-badge&labelColor=1a1a2e" /><br/>
<sub><b>XAU + BTC Parallel</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/🎯_Doctrine-Sniper-FF0000?style=for-the-badge&labelColor=1a1a2e" /><br/>
<sub><b>Conf ≥ 75%</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/⚖️_R:R->1:2-00FF88?style=for-the-badge&labelColor=1a1a2e" /><br/>
<sub><b>Hard Constraint</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/⚡_Latency-<200ms-7C3AED?style=for-the-badge&labelColor=1a1a2e" /><br/>
<sub><b>HFT Grade</b></sub>
</td>
<td align="center">
<img src="https://img.shields.io/badge/🧠_Model-CatBoost-FFCC00?style=for-the-badge&labelColor=1a1a2e" /><br/>
<sub><b>2000 Iterations</b></sub>
</td>
</tr>
</table>

</div>

---
<div align="center">

## THE SNIPER DOCTRINE (v2.1, REAL STATE)

**"Train -> Backtest -> Gate -> Trade. If gate fails, no live orders."**

</div>

This README now reflects the current implementation and current artifacts in this repository, not a marketing target state.

### 1. Entry Doctrine (Live)
- Signal engine uses a weighted 100-point model (`trend`, `momentum`, `volatility`, `structure`, `flow`, `mean_reversion`).
- Internal score gate: `signal_min_score=70`.
- Confidence gate: `min_confidence=80` (strict sniper mode).
- MTF penalties are active (M5/M15 conflict reduces confidence).
- D1 confluence is integrated into scoring with `d1_confluence_weight=5.0`.

### 2. Risk Doctrine (Live)
- Dynamic ATR-based planning for SL/TP and lot sizing.
- Tight spread guard: `spread_gate_multiplier=1.5` (not 2.0).
- Spread spike blocker: 3-sigma guard on spread history.
- Black Swan circuit breaker:
  - ATR regime expansion (`ATR / SMA(ATR,50) > 3.0`)
  - Gap detection (`gap > 2 x ATR`)
  - Auto hard-stop cooldown (`1800s` default)

### 3. Institutional Gate Doctrine (Live)
An asset is tradable only if all of these pass:
- State + metadata contract validity
- `status=VERIFIED` and `real_backtest=true`
- `unsafe=false`
- Sharpe/WinRate/Drawdown thresholds
- WFA passed and stress test passed
- Model + meta files physically exist

---

<div align="center">

## REAL SYSTEM STATUS SNAPSHOT (2026-02-25)

</div>

Snapshot from `Artifacts/models` and `Artifacts/backtest_institutional`:

| Item | Current Value |
|------|---------------|
| Gate (`XAU+BTC`) | `ok=false` |
| Gate reason | `BTC:state_missing:global_state_other_asset:XAU` |
| XAU model meta | `v1.0_xau_institutional.json` exists |
| XAU state | `status=UNSAFE`, `verified=false` |
| XAU sample quality | `insufficient_trades:10<20` |
| BTC model/state | missing (`v1.0_btc_institutional.*`, `model_state_BTC.pkl`) |
| Global state | points to XAU only |

Interpretation:
- The system can start, but strict gate for both assets is currently not satisfied.
- In partial mode (`PARTIAL_GATE_MODE=1`), engine can run with passing assets only.
- Right now XAU is also blocked by gate because state is marked `UNSAFE`.

---

<div align="center">

## COMPLETE SYSTEM PIPELINE (ACTUAL)

**Boot -> Preflight -> (Auto-Train if needed) -> Gate -> Engine Supervisor -> Telegram**

</div>

```
python main.py
  |
  +-- Preflight env + MT5 checks
  +-- _models_ready()
  |     |
  |     +-- if not ready: _auto_train_models_strict()
  |     |      +-- run_institutional_backtest(XAU/BTC as needed)
  |     |      +-- post-gate check
  |     |
  |     +-- if ready: skip training
  |
  +-- engine._check_model_health()
  |     +-- strict gate or partial gate mode
  |
  +-- run_engine_supervisor()
        +-- hourly model-age retrain check
        +-- gate-blocked retrain cooldown logic
        +-- engine start/restart with backoff
```

Important behavior:
- `run_institutional_backtest(asset)` always performs fresh model training for that asset when called.
- Program restart does **not** always retrain; retrain is conditional (missing/failed/expired gate path).

---

<div align="center">

## MODEL TRAINING AND VALIDATION

</div>

### Phase 1: Institutional Training
- Backend: CatBoost regression.
- Data source: MT5 historical bars (`M1` base training dataset).
- Split: train/val/test/holdout in `Backtest/model_train.py`.
- Output artifacts:
  - `Artifacts/models/v<version>.pkl`
  - `Artifacts/models/v<version>.json`

### Phase 2: Institutional Backtest
Implemented in `Backtest/engine.py`:
- No look-ahead simulation path.
- Transaction costs and execution effects.
- WFA, Monte Carlo (`10000` runs), stress scenarios.
- Sample quality gate (`MIN_GATE_TRADES`, side-balance checks).

### Phase 3: Gate Decision
Core thresholds (`core/config.py`):
- `MIN_GATE_SHARPE = 0.5`
- `MIN_GATE_WIN_RATE = 0.52`
- `MAX_GATE_DRAWDOWN = 0.25`

Additional sample-quality controls (`Backtest/engine.py`):
- `MIN_GATE_TRADES` default `20`
- `GATE_REQUIRE_BOTH_SIDES` default `1`
- `MIN_GATE_WINNING_TRADES` default `1`
- `MIN_GATE_LOSING_TRADES` default `1`

Why your XAU can look "good" but still fail:
- Even with high Sharpe/WinRate, asset is blocked when sample quality fails (example: only 10 trades).

---

<div align="center">

## SIGNAL ENGINE (CURRENT IMPLEMENTATION)

</div>

Live signal logic in `core/signal_engine.py` includes:
- M1/M15/H1 + D1 confluence.
- `_d1_confluence_score()` for daily trend bias.
- `_vector_alignment()` cosine-like MTF alignment score (`M1/M15/H1/D1`).
- `_momentum_ignition_score()` for early breakout + volume impulse detection.
- Final filters include D1 conflict penalty and MTF conflict penalties.

Execution routing in engine:
- If CatBoost payload exists for asset, prediction path is used first.
- ML fallback path exists, but gate-blocked assets are skipped in pipeline.

---

<div align="center">

## RISK ENGINE (CURRENT IMPLEMENTATION)

</div>

`core/risk_engine.py` (live state):
- Uses `threading.RLock` for mutable shared state.
- Concurrency-safe methods include `update_pnl`, `evaluate_account_state`, and `guard_decision`.
- Execution metrics tracked and flushed to per-asset CSV:
  - `Logs/exec_metrics_XAUUSDm.csv`
  - `Logs/exec_metrics_BTCUSDm.csv`
  (file appears only after execution metrics are produced for that asset)

Stability guards:
- Volatility circuit breaker + cooldown.
- Spread spike detection.
- Latency/slippage execution breaker.

---

<div align="center">

## ARCHITECTURE AND SCHEDULING

</div>

### Asset scheduling (`Bot/Motor/scheduler.py`)
- Weekdays: `XAU + BTC`
- Weekends: `BTC only`

### Core runtime (`main.py` + `Bot/Motor/engine.py`)
- Multi-asset engine with partial-gate support.
- Engine supervisor with backoff and controlled retraining.
- Telegram supervisor isolated from trading loop.

### Data-structure complexity (runtime safety)
- `_seen_index`: dictionary key lookup (`O(1)` average).
- `_seen`: deque TTL queue with bounded cleanup budget.
- `_catboost_pred_history`: per-asset `deque(maxlen=200)` (bounded memory, `O(1)` append).

---

<div align="center">

## ARTIFACTS (EXPECTED VS CURRENT)

</div>

### Current files present now
```
Artifacts/
|-- models/
|   |-- model_state.pkl
|   |-- model_state_XAU.pkl
|   |-- v1.0_xau_institutional.pkl
|   `-- v1.0_xau_institutional.json
`-- backtest_institutional/
    |-- xau_1_0_xau_institutional_institutional_metrics.json
    `-- xau_1_0_xau_institutional_institutional_report.txt
```

### Files expected when BTC also passes pipeline
```
Artifacts/models/model_state_BTC.pkl
Artifacts/models/v1.0_btc_institutional.pkl
Artifacts/models/v1.0_btc_institutional.json
Artifacts/backtest_institutional/btc_*_institutional_metrics.json
Artifacts/backtest_institutional/btc_*_institutional_report.txt
```

---

<div align="center">

## LOGGING (CURRENT)

</div>

System-wide aggregated `system.log` is disabled.

Active log model:
- `Logs/main.log`
- `Logs/telegram.log`
- `Logs/portfolio_engine_health.log`
- `Logs/portfolio_engine_error.log`
- `Logs/portfolio_engine_diag.jsonl`
- Per-asset execution metrics CSV in `Logs/`

---

<div align="center">

## SOURCE CODE MAP

</div>

```
Exness/
|
|-- main.py
|-- log_config.py
|
|-- core/
|   |-- config.py
|   |-- feature_engine.py
|   |-- signal_engine.py
|   |-- risk_engine.py
|   |-- model_gate.py
|   |-- model_manager.py
|   |-- model_retrainer.py
|   `-- portfolio_risk.py
|
|-- Backtest/
|   |-- engine.py
|   |-- model_train.py
|   `-- metrics.py
|
|-- Bot/
|   |-- bot.py
|   |-- bot_utils.py
|   |-- portfolio_engine.py
|   `-- Motor/
|       |-- engine.py
|       |-- pipeline.py
|       |-- scheduler.py
|       |-- execution.py
|       |-- logging_setup.py
|       `-- models.py
|
|-- DataFeed/
|-- ExnessAPI/
|-- strategies/
`-- tests/
```

---

<div align="center">

## ENVIRONMENT VARIABLES (KEY ONES)

</div>

| Variable | Default | Meaning |
|----------|---------|---------|
| `DRY_RUN` | `0` | simulation mode |
| `PARTIAL_GATE_MODE` | `1` | allow start with passing assets only |
| `AUTO_TRAIN_ASSETS` | required assets | startup auto-train asset list |
| `RETRAIN_ASSETS` | `XAU,BTC` | runtime retrain asset list |
| `GATE_RETRAIN_COOLDOWN_SEC` | `300` | delay between gate-triggered retrains |
| `MIN_GATE_SHARPE` | `0.5` | minimum Sharpe gate |
| `MIN_GATE_WIN_RATE` | `0.52` | minimum WinRate gate |
| `MAX_GATE_DRAWDOWN` | `0.25` | maximum drawdown gate |
| `MIN_GATE_TRADES` | `20` | minimum trade count for sample quality |
| `GATE_REQUIRE_BOTH_SIDES` | `1` | require both winning and losing trades |

---

## INSTALLATION AND RUN

1. Install dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```

2. Create `.env` with Exness and Telegram credentials.

3. Run live:
   ```bash
   py .\main.py
   ```

4. Run dry-run:
   ```bash
   $env:DRY_RUN="1"; py .\main.py
   ```

---

## FAST VERIFICATION COMMANDS

Gate snapshot:
```bash
py -c "from core.model_gate import gate_details; import json; print(json.dumps(gate_details(required_assets=('XAU','BTC')), indent=2))"
```

Show current model artifacts:
```bash
Get-ChildItem Artifacts\models
Get-ChildItem Artifacts\backtest_institutional
```

Run concurrency + D1 math tests:
```bash
py -m pytest tests/test_risk_thread_safety.py -v
py -m pytest tests/test_signal_d1_vector.py -v
```

---

<div align="center">

## NOTE ABOUT METRICS LIKE `profit_factor`

</div>

If you see extreme values (e.g., old `Infinity`), check current metrics code:
- Profit factor is now capped and tracked with flags (`profit_factor_capped`, `metric_notes`).
- `expectancy_ratio` uses safe fallback denominator when average loss is ~0.
- Small trade samples can still make metrics unstable, which is exactly why sample-quality gate exists.

---

<div align="center">

## AUTHOR

**Gafurov Kabir**

</div>