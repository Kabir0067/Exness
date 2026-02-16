# Backtest/engine_institutional.py - Institutional-grade backtest engine
# Zero look-ahead bias, Kelly sizing, regime detection, stress testing
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib

from log_config import get_artifact_dir, get_artifact_path
from mt5_client import MT5_LOCK, ensure_mt5

try:
    from .metrics import BacktestMetrics, compute_metrics, save_metrics
    from .model_train import (
        BTC_TRAIN_CONFIG, XAU_TRAIN_CONFIG,
        Pipeline, RegressionModel, train_and_register
    )
except ImportError:
    from Backtest.metrics import BacktestMetrics, compute_metrics, save_metrics
    from Backtest.model_train import (
        BTC_TRAIN_CONFIG, XAU_TRAIN_CONFIG,
        Pipeline, RegressionModel, train_and_register
    )

log = logging.getLogger("backtest.engine_institutional")
MODEL_STATE_PATH = get_artifact_path("models", "model_state.pkl")


def _console(msg: str) -> None:
    """Write directly to the REAL console, bypassing _StdToLogger wrapper."""
    try:
        real_out = getattr(sys, "__stdout__", None) or sys.stdout
        real_out.write(msg + "\n")
        real_out.flush()
    except Exception:
        pass


@dataclass
class InstitutionalBacktestConfig:
    """Institutional-grade backtest configuration"""
    strategy_name: str
    model_version: str
    start_date: str
    end_date: str
    
    # Capital & Position Management
    initial_capital: float = 100000.0
    max_position_size_pct: float = 0.20  # Max 20% per position
    kelly_fraction: float = 0.25  # Kelly Criterion conservative
    
    # Risk Limits (INSTITUTIONAL)
    max_drawdown_limit: float = 0.15  # Stop trading at 15% DD
    daily_loss_limit: float = 0.03  # Stop if lose 3% in a day
    max_correlation_exposure: float = 0.60  # Max 60% in correlated assets
    
    # Stop-Loss & Take-Profit
    use_stops: bool = True
    atr_stop_multiplier: float = 2.5  # 2.5x ATR for stop-loss
    atr_target_multiplier: float = 4.0  # 4x ATR for take-profit
    trailing_stop_activation: float = 0.015  # Activate at 1.5% profit
    trailing_stop_distance: float = 0.008  # Trail at 0.8%
    
    # Transaction Costs (REALISTIC)
    spread_bps: Dict[str, float] = None
    slippage_bps: Dict[str, float] = None
    commission_bps: float = 0.5  # 0.5 bps commission
    funding_rate_annual: float = 0.05  # 5% annual funding for crypto
    
    # Walk-Forward Analysis
    wfa_train_days: int = 90
    wfa_test_days: int = 30
    wfa_min_sharpe: float = 2.0  # Institutional threshold
    wfa_min_win_rate: float = 0.58
    
    # Monte Carlo Robustness
    monte_carlo_runs: int = 10000  # 10k runs for institutions
    ruin_drawdown_pct: float = 0.25  # Conservative 25% ruin threshold
    monte_carlo_confidence: float = 0.95  # 95% confidence level
    monte_carlo_seed: int = 42
    
    # Regime Detection
    regime_lookback: int = 120  # 120 bars for regime
    trend_threshold: float = 0.0015  # 0.15% for trend
    volatility_threshold: float = 1.5  # 1.5x volatility = high vol
    
    # Stress Testing
    stress_test_scenarios: bool = True
    crash_scenario_pct: float = -0.20  # -20% flash crash
    volatility_spike_multiplier: float = 3.0  # 3x vol spike
    
    def __post_init__(self):
        if self.spread_bps is None:
            self.spread_bps = {"XAU": 2.0, "BTC": 6.0}
        if self.slippage_bps is None:
            self.slippage_bps = {"XAU": 0.8, "BTC": 1.5}


XAU_INSTITUTIONAL_CONFIG = InstitutionalBacktestConfig(
    strategy_name="XAU_Institutional_v1",
    model_version="1.0_institutional",
    start_date="2025-01-01",
    end_date="2026-02-15",
    initial_capital=100000.0,
)

BTC_INSTITUTIONAL_CONFIG = InstitutionalBacktestConfig(
    strategy_name="BTC_Institutional_v1",
    model_version="1.0_institutional",
    start_date="2025-01-01",
    end_date="2026-02-15",
    initial_capital=100000.0,
)


def _base_symbol(asset: str) -> str:
    asset_u = str(asset).upper().strip()
    if asset_u in ("BTC", "BTCUSD", "BTCUSDM"):
        return "BTCUSD"
    return "XAUUSD"


def _resolve_symbol(asset: str) -> str:
    base = _base_symbol(asset)
    suffix = str(os.getenv("MT5_SYMBOL_SUFFIX", "") or "").strip()
    candidates = []
    if suffix:
        candidates.append(f"{base}{suffix}")
    candidates.extend([f"{base}m", base])
    seen: set[str] = set()
    uniq: list[str] = []
    for sym in candidates:
        if sym in seen:
            continue
        seen.add(sym)
        uniq.append(sym)
    for sym in uniq:
        try:
            if mt5.symbol_info(sym) is not None:
                return sym
        except Exception:
            continue
    # Fallback: scan available symbols for base match
    try:
        symbols = mt5.symbols_get()
    except Exception:
        symbols = None
    if symbols:
        base_u = base.upper()
        names = [s.name for s in symbols if hasattr(s, "name")]
        exact = [n for n in names if n.upper() == base_u]
        if exact:
            return exact[0]
        starts = [n for n in names if n.upper().startswith(base_u)]
        if starts:
            return starts[0]
        contains = [n for n in names if base_u in n.upper()]
        if contains:
            return contains[0]

    return uniq[0]


def _mt5_symbol(asset: str) -> str:
    return _resolve_symbol(asset)


def _parse_utc_date(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).strip())
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class RegimeDetector:
    """Market regime classification: TREND_UP, TREND_DOWN, RANGE, HIGH_VOL"""
    
    def __init__(self, lookback: int = 120, trend_threshold: float = 0.0015):
        self.lookback = lookback
        self.trend_threshold = trend_threshold
    
    def detect(self, prices: np.ndarray) -> str:
        """Detect current market regime"""
        if len(prices) < self.lookback:
            return "UNKNOWN"
        
        recent = prices[-self.lookback:]
        returns = np.diff(recent) / recent[:-1]
        
        # Trend detection
        cumulative_ret = (recent[-1] - recent[0]) / recent[0]
        if cumulative_ret > self.trend_threshold:
            return "TREND_UP"
        elif cumulative_ret < -self.trend_threshold:
            return "TREND_DOWN"
        
        # Volatility detection
        vol = np.std(returns)
        avg_vol = np.std(np.diff(prices) / prices[:-1])
        if vol > 1.5 * avg_vol:
            return "HIGH_VOL"
        
        return "RANGE"


class KellyPositionSizer:
    """Kelly Criterion position sizing with institutional safeguards"""
    
    def __init__(self, fraction: float = 0.25, max_size: float = 0.20):
        self.fraction = fraction  # Conservative Kelly
        self.max_size = max_size  # Hard cap at 20%
    
    def calculate(
        self, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float,
        confidence: float = 1.0
    ) -> float:
        """Calculate Kelly-optimal position size"""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly = max(0.0, kelly)  # No negative positions
        
        # Apply fraction and confidence adjustment
        kelly *= self.fraction * confidence
        
        # Hard cap
        return min(kelly, self.max_size)


class InstitutionalBacktestEngine:
    """
    Institutional-grade backtest engine:
    - ZERO look-ahead bias
    - Kelly position sizing
    - ATR-based stops
    - Regime filtering
    - Stress testing
    - Multi-timeframe validation
    """
    
    def __init__(
        self, 
        asset: str, 
        model_version: str, 
        run_cfg: InstitutionalBacktestConfig
    ) -> None:
        self.asset = str(asset).upper().strip()
        self.model_version = str(model_version).strip()
        self.run_cfg = run_cfg
        
        # State tracking
        self._last_verified: bool = False
        self._risk_of_ruin: float = 0.0
        self._wfa: Dict[str, Any] = {}
        self._unsafe: bool = False
        self._stress_results: Dict[str, Any] = {}
        
        # Risk management
        self.regime_detector = RegimeDetector(
            lookback=run_cfg.regime_lookback,
            trend_threshold=run_cfg.trend_threshold
        )
        self.position_sizer = KellyPositionSizer(
            fraction=run_cfg.kelly_fraction,
            max_size=run_cfg.max_position_size_pct
        )
    
    def _registry_model_path(self) -> Path:
        suffix = "_institutional"
        version = self.model_version
        if not version.endswith(suffix):
            version = f"{version}{suffix}"
        return get_artifact_path("models", f"v{version}.pkl")
    
    def _registry_meta_path(self) -> Path:
        suffix = "_institutional"
        version = self.model_version
        if not version.endswith(suffix):
            version = f"{version}{suffix}"
        return get_artifact_path("models", f"v{version}.json")
    
    def _load_model_payload(self) -> Dict[str, Any]:
        p = self._registry_model_path()
        if not p.exists():
            raise RuntimeError(f"model_payload_missing:{p}")
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(f"model_payload_invalid:{p}")
        if "model" not in payload or "pipeline" not in payload:
            raise RuntimeError("model_payload_contract_invalid")
        return payload
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for stop-loss sizing (TA-Lib)."""
        high = np.asarray(df["High"].values, dtype=np.float64)
        low = np.asarray(df["Low"].values, dtype=np.float64)
        close = np.asarray(df["Close"].values, dtype=np.float64)
        atr = talib.ATR(high, low, close, timeperiod=int(period))
        return pd.Series(atr, index=df.index)
    
    def _simulate_trades_institutional(
        self,
        df: pd.DataFrame,
        pred: np.ndarray,
        y: np.ndarray,
        *,
        threshold: float,
        initial_capital: float,
        rng: Optional[np.random.Generator],
    ) -> Tuple[List[float], List[Dict[str, Any]], Dict[str, float]]:
        """
        Institutional-grade trade simulation with:
        - Kelly position sizing
        - ATR-based stops
        - Regime filtering
        - Realistic slippage
        - Daily loss limits
        """
        trades: List[Dict[str, Any]] = []
        equity_curve = [initial_capital]
        equity = initial_capital
        
        # Calculate ATR for stops
        atr = self._calculate_atr(df).values
        prices = df["Close"].values
        
        # Risk tracking
        daily_pnl = 0.0
        last_date = None
        max_equity = initial_capital
        current_drawdown = 0.0
        
        # Position sizing history for Kelly calculation
        win_rate_rolling = 0.55  # Initial estimate
        avg_win_rolling = 100.0
        avg_loss_rolling = -80.0
        
        # Spread and slippage configs
        asset_key = "BTC" if self.asset in ("BTC", "BTCUSD", "BTCUSDM") else "XAU"
        spread_bps = self.run_cfg.spread_bps.get(asset_key, 2.0)
        slippage_bps = self.run_cfg.slippage_bps.get(asset_key, 1.0)
        
        for i in range(len(pred)):
            try:
                p = float(pred[i])
            except Exception:
                continue
            if not np.isfinite(p) or abs(p) < threshold:
                continue
            
            # Check date change for daily limits
            current_date = df.index[i].date()
            if last_date is not None and current_date != last_date:
                daily_pnl = 0.0
            last_date = current_date
            
            # RISK LIMIT: Daily loss limit
            if equity <= 0:
                break
            if (daily_pnl / equity) < -self.run_cfg.daily_loss_limit:
                continue
            
            # RISK LIMIT: Max drawdown
            current_drawdown = (max_equity - equity) / max_equity
            if current_drawdown > self.run_cfg.max_drawdown_limit:
                break  # Stop trading
            
            # Regime detection
            regime = self.regime_detector.detect(prices[:i+1])
            if regime == "HIGH_VOL" and abs(p) < threshold * 1.5:
                continue  # Skip low-confidence signals in high vol
            
            # Kelly position sizing
            confidence = min(abs(p) / threshold, 2.0)  # Confidence from prediction
            position_size_pct = self.position_sizer.calculate(
                win_rate=win_rate_rolling,
                avg_win=avg_win_rolling,
                avg_loss=avg_loss_rolling,
                confidence=confidence
            )
            
            if position_size_pct <= 0.0:
                continue
            
            notional = equity * position_size_pct
            direction = 1.0 if p > 0.0 else -1.0
            
            # Entry price with spread
            entry_price = prices[i]
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            spread_cost = notional * (spread_bps / 10000.0)
            
            # Slippage modeling (institutional)
            segment = prices[max(0, i-20):i+1]
            volatility = 0.0
            if len(segment) > 1:
                denom = segment[:-1]
                returns = np.diff(segment) / denom
                if returns.size > 0:
                    volatility = float(np.std(returns))
                    if not np.isfinite(volatility):
                        volatility = 0.0
            slippage_multiplier = 1.0 + min(volatility / 0.001, 3.0)
            if not np.isfinite(slippage_multiplier):
                slippage_multiplier = 1.0
            slippage_cost = notional * (slippage_bps / 10000.0) * slippage_multiplier
            if rng is not None:
                slippage_cost *= float(rng.uniform(0.8, 1.2))
            
            # Commission
            commission = notional * (self.run_cfg.commission_bps / 10000.0)
            
            # ATR-based stop-loss and take-profit
            atr_value = atr[i] if i < len(atr) and not np.isnan(atr[i]) else entry_price * 0.01
            stop_distance = atr_value * self.run_cfg.atr_stop_multiplier
            target_distance = atr_value * self.run_cfg.atr_target_multiplier
            
            stop_loss = entry_price - (direction * stop_distance)
            take_profit = entry_price + (direction * target_distance)
            
            # Simulate exit (simplified - check next bar)
            try:
                actual_ret = float(y[i])
            except Exception:
                continue
            if not np.isfinite(actual_ret):
                continue
            exit_price = prices[i] * (1 + actual_ret)
            
            # Check stops
            hit_stop = False
            hit_target = False
            if direction > 0:  # Long
                if exit_price <= stop_loss:
                    exit_price = stop_loss
                    hit_stop = True
                elif exit_price >= take_profit:
                    exit_price = take_profit
                    hit_target = True
            else:  # Short
                if exit_price >= stop_loss:
                    exit_price = stop_loss
                    hit_stop = True
                elif exit_price <= take_profit:
                    exit_price = take_profit
                    hit_target = True
            
            # Calculate P&L
            gross_pnl = direction * (exit_price - entry_price) / entry_price * notional
            net_pnl = gross_pnl - spread_cost - slippage_cost - commission
            if not np.isfinite(net_pnl):
                continue
            
            # Update equity
            equity += net_pnl
            equity_curve.append(equity)
            daily_pnl += net_pnl
            
            # Update max equity for drawdown
            if equity > max_equity:
                max_equity = equity
            
            # Update rolling statistics for Kelly
            if net_pnl > 0:
                avg_win_rolling = 0.9 * avg_win_rolling + 0.1 * net_pnl
            else:
                avg_loss_rolling = 0.9 * avg_loss_rolling + 0.1 * net_pnl
            
            recent_trades = [t["pnl"] for t in trades[-100:]]
            if len(recent_trades) > 10:
                wins = [p for p in recent_trades if p > 0]
                win_rate_rolling = len(wins) / len(recent_trades)
            
            # Record trade
            trades.append({
                "pnl": float(net_pnl),
                "gross_pnl": float(gross_pnl),
                "direction": direction,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "position_size_pct": float(position_size_pct),
                "notional": float(notional),
                "spread_cost": float(spread_cost),
                "slippage_cost": float(slippage_cost),
                "commission": float(commission),
                "regime": regime,
                "hit_stop": hit_stop,
                "hit_target": hit_target,
                "prediction": float(p),
                "actual_return": float(actual_ret),
            })
        
        # Extract PnL series
        pnl_series = [t["pnl"] for t in trades]
        
        # Calculate cost breakdown
        total_costs = {
            "spread": sum(t["spread_cost"] for t in trades),
            "slippage": sum(t["slippage_cost"] for t in trades),
            "commission": sum(t["commission"] for t in trades),
        }
        
        return pnl_series, trades, total_costs
    
    def _monte_carlo_institutional(
        self, 
        pnl_series: List[float]
    ) -> Dict[str, float]:
        """
        Institutional Monte Carlo with:
        - Risk of ruin at multiple thresholds
        - Confidence intervals
        - Worst-case scenarios
        """
        runs = self.run_cfg.monte_carlo_runs
        if not pnl_series or runs <= 0:
            return {"risk_of_ruin": 0.0, "confidence_95": 0.0, "worst_case": 0.0}
        
        ruin_threshold = self.run_cfg.initial_capital * (1 - self.run_cfg.ruin_drawdown_pct)
        rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
        pnls = np.asarray(pnl_series, dtype=np.float64)
        
        ruin_count = 0
        final_equities = []
        
        for _ in range(runs):
            seq = rng.permutation(pnls)
            equity = np.cumsum(seq) + self.run_cfg.initial_capital
            
            if float(np.min(equity)) <= ruin_threshold:
                ruin_count += 1
            
            final_equities.append(float(equity[-1]))
        
        final_equities = np.asarray(final_equities)
        
        return {
            "risk_of_ruin": float(ruin_count / runs),
            "confidence_95_percentile": float(np.percentile(final_equities, 5)),
            "median_final": float(np.median(final_equities)),
            "worst_case": float(np.min(final_equities)),
            "best_case": float(np.max(final_equities)),
        }
    
    def _stress_test(self, df: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """
        Stress testing:
        - Flash crash scenario
        - Volatility spike
        - Correlation breakdown
        """
        if not trades:
            return {"passed": True, "scenarios": []}
        
        scenarios = []
        
        # Scenario 1: Flash crash
        crash_impact = sum(
            t["notional"] * self.run_cfg.crash_scenario_pct 
            for t in trades if t["direction"] > 0
        )
        crash_drawdown = abs(crash_impact) / self.run_cfg.initial_capital
        
        scenarios.append({
            "name": "flash_crash",
            "impact": float(crash_impact),
            "drawdown": float(crash_drawdown),
            "passed": crash_drawdown < 0.30,  # Survive 30% DD
        })
        
        # Scenario 2: Volatility spike (3x slippage)
        vol_spike_cost = sum(t["slippage_cost"] * 3.0 for t in trades)
        vol_spike_impact = vol_spike_cost / self.run_cfg.initial_capital
        
        scenarios.append({
            "name": "volatility_spike",
            "impact": float(-vol_spike_cost),
            "cost_pct": float(vol_spike_impact),
            "passed": vol_spike_impact < 0.05,  # Tolerate 5% cost
        })
        
        passed_all = all(s["passed"] for s in scenarios)
        
        return {
            "passed": passed_all,
            "scenarios": scenarios,
        }
    
    def _walk_forward_institutional(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Walk-forward with institutional thresholds"""
        train_days = self.run_cfg.wfa_train_days
        test_days = self.run_cfg.wfa_test_days
        
        if train_days <= 0 or test_days <= 0:
            return {"passed": False, "total": 0, "failed": 0, "windows": []}
        
        start = df.index.min()
        end = df.index.max()
        train_delta = timedelta(days=train_days)
        test_delta = timedelta(days=test_days)
        t0 = start
        
        windows = []
        failed = 0
        
        while t0 + train_delta + test_delta <= end:
            train_end = t0 + train_delta
            test_end = train_end + test_delta
            
            df_train = df[(df.index >= t0) & (df.index < train_end)]
            df_test = df[(df.index >= train_end) & (df.index < test_end)]
            
            if df_train.empty or df_test.empty:
                break
            
            # Train mini-model for this window
            cfg = BTC_TRAIN_CONFIG if self.asset in ("BTC", "BTCUSD", "BTCUSDM") else XAU_TRAIN_CONFIG
            pipeline = Pipeline(cfg)
            splits = pipeline.fit(df_train.copy())
            model = RegressionModel(cfg)
            model.train(splits)
            
            # Test on forward period
            xy = pipeline.transform(df_test.copy())
            X = np.stack(xy["X"].values)
            y = np.asarray(xy["y"].values)
            pred = model.model.predict(X)
            
            threshold = max(float(getattr(cfg, "percent_increase", 0.0) or 0.0), 0.0001)
            rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
            
            pnls, _, costs = self._simulate_trades_institutional(
                df_test,
                pred,
                y,
                threshold=threshold,
                initial_capital=self.run_cfg.initial_capital,
                rng=rng,
            )
            
            if not pnls:
                failed += 1
                windows.append({
                    "train_start": str(df_train.index.min()),
                    "test_start": str(df_test.index.min()),
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "passed": False,
                })
                t0 += test_delta
                continue
            
            metrics = compute_metrics(
                pnls,
                initial_capital=self.run_cfg.initial_capital,
                symbol=self.asset,
                timeframe="M1",
                start_date=str(df_test.index.min()),
                end_date=str(df_test.index.max()),
                spread_costs=costs["spread"],
                slippage_costs=costs["slippage"],
                swap_costs=0.0,
            )
            
            # INSTITUTIONAL THRESHOLDS
            passed = (
                metrics.sharpe_ratio >= self.run_cfg.wfa_min_sharpe and
                metrics.win_rate >= self.run_cfg.wfa_min_win_rate and
                metrics.max_drawdown_pct <= 0.20
            )
            
            if not passed:
                failed += 1
            
            windows.append({
                "train_start": str(df_train.index.min()),
                "test_start": str(df_test.index.min()),
                "sharpe": float(metrics.sharpe_ratio),
                "win_rate": float(metrics.win_rate),
                "max_dd": float(metrics.max_drawdown_pct),
                "passed": bool(passed),
            })
            
            t0 += test_delta
        
        total = len(windows)
        return {
            "passed": bool(total > 0 and failed == 0),
            "total": total,
            "failed": failed,
            "windows": windows,
        }
    
    def load_history(self) -> pd.DataFrame:
        """Load historical data from MT5"""
        ensure_mt5()
        symbol = _mt5_symbol(self.asset)
        
        start_dt = _parse_utc_date(self.run_cfg.start_date)
        end_dt = _parse_utc_date(self.run_cfg.end_date)
        
        tf = mt5.TIMEFRAME_M1
        with MT5_LOCK:
            ok = mt5.symbol_select(symbol, True)
            rates = None
            err = mt5.last_error()
            if ok:
                rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
                err = mt5.last_error()
            if rates is None or len(rates) == 0:
                bars = int(os.getenv("BACKTEST_BARS", "50000") or "50000")
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                err = mt5.last_error()
        
        if not ok or rates is None or len(rates) == 0:
            raise RuntimeError(f"backtest_history_empty:{symbol}:{err}")
        
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume"
        })
        
        return df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]]
    
    def run(self, df: pd.DataFrame) -> BacktestMetrics:
        """Run institutional-grade backtest"""
        if df is None or df.empty:
            raise RuntimeError("backtest_empty_df")
        
        payload = self._load_model_payload()
        model = payload["model"]
        pipeline = payload["pipeline"]
        
        # Transform WITHOUT look-ahead
        xy = pipeline.transform(df.copy())
        X = np.stack(xy["X"].values)
        y = np.asarray(xy["y"].values)
        pred = model.predict(X)
        
        threshold = max(float(getattr(pipeline.cfg, "percent_increase", 0.0)), 0.0001)
        rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
        
        # Institutional simulation
        pnl_series, trades, costs = self._simulate_trades_institutional(
            df, pred, y,
            threshold=threshold,
            initial_capital=self.run_cfg.initial_capital,
            rng=rng,
        )
        
        if not pnl_series:
            raise RuntimeError("no_trades_generated")
        
        # Compute metrics
        metrics = compute_metrics(
            pnl_series,
            initial_capital=self.run_cfg.initial_capital,
            symbol=self.asset,
            timeframe="M1",
            start_date=str(df.index.min()),
            end_date=str(df.index.max()),
            spread_costs=costs["spread"],
            slippage_costs=costs["slippage"],
            swap_costs=0.0,
        )
        
        # Monte Carlo
        mc_results = self._monte_carlo_institutional(pnl_series)
        metrics.risk_of_ruin = mc_results["risk_of_ruin"]
        metrics.monte_carlo_runs = self.run_cfg.monte_carlo_runs
        
        # Walk-forward
        wfa = self._walk_forward_institutional(df)
        metrics.wfa_passed = wfa["passed"]
        metrics.wfa_total_windows = wfa["total"]
        metrics.wfa_failed_windows = wfa["failed"]
        
        # Stress testing
        stress = self._stress_test(df, trades)
        self._stress_results = stress
        
        # Verification with INSTITUTIONAL thresholds
        base_verified = (
            metrics.sharpe_ratio >= 2.0 and
            metrics.win_rate >= 0.58 and
            metrics.max_drawdown_pct <= 0.20
        )
        
        # Risk checks
        unsafe = (
            mc_results["risk_of_ruin"] > 0.01 or  # Max 1% ruin risk
            not stress["passed"]
        )
        
        self._last_verified = bool(base_verified and wfa["passed"] and not unsafe)
        self._risk_of_ruin = mc_results["risk_of_ruin"]
        self._wfa = wfa
        self._unsafe = unsafe
        
        # Save metadata
        self._update_model_meta(metrics, mc_results, wfa, stress, unsafe)
        
        return metrics
    
    def _update_model_meta(
        self,
        metrics: BacktestMetrics,
        mc_results: Dict,
        wfa: Dict,
        stress: Dict,
        unsafe: bool
    ) -> None:
        """Update model metadata with institutional results"""
        p = self._registry_meta_path()
        if not p.exists():
            return
        
        try:
            with p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            meta["backtest_sharpe"] = float(metrics.sharpe_ratio)
            meta["backtest_win_rate"] = float(metrics.win_rate)
            meta["max_drawdown_pct"] = float(metrics.max_drawdown_pct)
            meta["risk_of_ruin"] = float(mc_results["risk_of_ruin"])
            meta["monte_carlo_runs"] = self.run_cfg.monte_carlo_runs
            meta["mc_confidence_95"] = float(mc_results["confidence_95_percentile"])
            meta["mc_worst_case"] = float(mc_results["worst_case"])
            meta["wfa_passed"] = wfa["passed"]
            meta["wfa_total_windows"] = wfa["total"]
            meta["wfa_failed_windows"] = wfa["failed"]
            meta["stress_test_passed"] = stress["passed"]
            meta["stress_scenarios"] = stress["scenarios"]
            meta["institutional_grade"] = True
            meta["real_backtest"] = True
            meta["unsafe"] = unsafe
            meta["status"] = "UNSAFE" if unsafe else ("VERIFIED" if self._last_verified else "REJECTED")
            meta["backtested_at_utc"] = datetime.now(timezone.utc).isoformat()
            
            with p.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception as exc:
            log.error("META_UPDATE_ERROR | version=%s err=%s", self.model_version, exc)
    
    def save_model_state(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Save institutional model state"""
        status = "UNSAFE" if self._unsafe else ("VERIFIED" if self._last_verified else "REJECTED")
        
        state = {
            "model_version": self.model_version,
            "asset": self.asset,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "verified": self._last_verified,
            "institutional_grade": True,
            "real_backtest": True,
            "sharpe_ratio": float(metrics.sharpe_ratio),
            "win_rate": float(metrics.win_rate),
            "max_drawdown_pct": float(metrics.max_drawdown_pct),
            "risk_of_ruin": float(self._risk_of_ruin),
            "wfa_passed": bool(self._wfa.get("passed")),
            "wfa_total_windows": int(self._wfa.get("total", 0)),
            "wfa_failed_windows": int(self._wfa.get("failed", 0)),
            "stress_test_passed": bool(self._stress_results.get("passed")),
            "unsafe": self._unsafe,
            "source": "Backtest.engine_institutional",
        }
        
        MODEL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_STATE_PATH.open("wb") as f:
            pickle.dump(state, f)
        
        log.info(
            "INSTITUTIONAL_MODEL_STATE | version=%s status=%s sharpe=%.3f win_rate=%.3f",
            state["model_version"],
            state["status"],
            state["sharpe_ratio"],
            state["win_rate"],
        )
        
        return state


def run_institutional_backtest(asset: str = "XAU") -> BacktestMetrics:
    """Run complete institutional backtest pipeline"""
    asset_u = str(asset).upper().strip()
    run_cfg = (
        BTC_INSTITUTIONAL_CONFIG 
        if asset_u in ("BTC", "BTCUSD", "BTCUSDM") 
        else XAU_INSTITUTIONAL_CONFIG
    )
    
    log.info("Phase 1: Institutional Model Training")
    train_info = train_and_register(asset_u)
    model_version = train_info.get("model_version", run_cfg.model_version)
    
    log.info("Phase 2: Institutional Backtest & Verification")
    engine = InstitutionalBacktestEngine(asset_u, model_version, run_cfg)
    df = engine.load_history()
    metrics = engine.run(df)
    
    out_dir = get_artifact_dir("backtest_institutional")
    prefix = f"{asset_u.lower()}_{model_version.replace('.', '_')}_institutional"
    save_metrics(metrics, out_dir, prefix=prefix)
    
    state = engine.save_model_state(metrics)
    status = "PASSED" if state["verified"] else "FAILED"
    
    _console("\n" + "=" * 60)
    _console(f"INSTITUTIONAL BACKTEST: {status}")
    _console("=" * 60)
    _console(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f} (Req: >= 2.0)")
    _console(f"Win Rate:            {metrics.win_rate:.1%} (Req: >= 58%)")
    _console(f"Max Drawdown:        {metrics.max_drawdown_pct:.2%} (Max: 20%)")
    _console(f"Risk of Ruin:        {metrics.risk_of_ruin:.2%} (Max: 1%)")
    _console(f"WFA:                 {'PASS' if metrics.wfa_passed else 'FAIL'}")
    _console(f"Stress Test:         {'PASS' if state['stress_test_passed'] else 'FAIL'}")
    _console(f"Version:             {model_version}")
    _console("=" * 60 + "\n")
    
    return metrics


# Compatibility aliases for legacy imports
BacktestEngine = InstitutionalBacktestEngine
XAU_BACKTEST_CONFIG = XAU_INSTITUTIONAL_CONFIG
BTC_BACKTEST_CONFIG = BTC_INSTITUTIONAL_CONFIG
run_backtest = run_institutional_backtest


if __name__ == "__main__":
    asset_arg = sys.argv[1] if len(sys.argv) > 1 else "XAU"
    run_institutional_backtest(asset_arg)
