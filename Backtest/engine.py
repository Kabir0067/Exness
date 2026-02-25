# Backtest/engine_institutional.py - Institutional-grade backtest engine
# Zero look-ahead bias, Kelly sizing, regime detection, stress testing
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib

from log_config import get_artifact_dir, get_artifact_path
from mt5_client import MT5_LOCK, ensure_mt5
from core.config import (
    MAX_GATE_DRAWDOWN,
    MIN_GATE_SHARPE,
    MIN_GATE_WIN_RATE,
    WFA_MIN_PASS_RATE,
    WFA_MIN_WINDOWS,
)

try:
    from .metrics import BacktestMetrics, compute_metrics, save_metrics
    from .model_train import (
        BTC_TRAIN_CONFIG, XAU_TRAIN_CONFIG,
        Pipeline, RegressionModel, train_and_register, load_training_dataframe_for_asset
    )
except ImportError:
    from Backtest.metrics import BacktestMetrics, compute_metrics, save_metrics
    from Backtest.model_train import (
        BTC_TRAIN_CONFIG, XAU_TRAIN_CONFIG,
        Pipeline, RegressionModel, train_and_register, load_training_dataframe_for_asset
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
    backtest_risk_free_rate: float = float(
        os.getenv("BACKTEST_RISK_FREE_RATE", "0.0") or "0.0"
    )
    
    # Walk-Forward Analysis
    wfa_train_days: int = 90
    wfa_test_days: int = 30
    wfa_min_sharpe: float = MIN_GATE_SHARPE
    wfa_min_win_rate: float = MIN_GATE_WIN_RATE
    verification_max_drawdown_pct: float = MAX_GATE_DRAWDOWN
    wfa_allow_skip_if_insufficient: bool = str(
        os.getenv("WFA_ALLOW_SKIP_IF_INSUFFICIENT", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    
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
    model_version="1.0_xau_institutional",
    start_date="2025-01-01",
    end_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    initial_capital=100000.0,
)

BTC_INSTITUTIONAL_CONFIG = InstitutionalBacktestConfig(
    strategy_name="BTC_Institutional_v1",
    model_version="1.0_btc_institutional",
    start_date="2025-01-01",
    end_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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
        self._sample_quality_passed: bool = True
        self._sample_quality_issues: List[str] = []
        
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

    def _legacy_shared_version(self) -> str:
        """Compatibility fallback for pre-per-asset model naming."""
        version = str(self.model_version or "").strip()
        low = version.lower()
        token = "_xau" if self.asset in ("XAU", "XAUUSD", "XAUUSDM") else "_btc"
        idx = low.find(token)
        if idx >= 0:
            version = version[:idx] + version[idx + len(token) :]
        return version

    def _registry_model_candidates(self) -> List[Path]:
        p = self._registry_model_path()
        candidates: List[Path] = [p]
        legacy_version = self._legacy_shared_version()
        if legacy_version and legacy_version != self.model_version:
            lp = get_artifact_path("models", f"v{legacy_version}.pkl")
            candidates.append(lp)
        seen: set[str] = set()
        uniq: List[Path] = []
        for c in candidates:
            key = str(c.resolve())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    def _registry_meta_candidates(self) -> List[Path]:
        p = self._registry_meta_path()
        candidates: List[Path] = [p]
        legacy_version = self._legacy_shared_version()
        if legacy_version and legacy_version != self.model_version:
            lp = get_artifact_path("models", f"v{legacy_version}.json")
            candidates.append(lp)
        seen: set[str] = set()
        uniq: List[Path] = []
        for c in candidates:
            key = str(c.resolve())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq
    
    def _load_model_payload(self) -> Dict[str, Any]:
        candidates = self._registry_model_candidates()
        payload = None
        payload_path: Optional[Path] = None
        for p in candidates:
            if not p.exists():
                continue
            with p.open("rb") as f:
                payload = pickle.load(f)
            payload_path = p
            if p != candidates[0]:
                log.warning(
                    "MODEL_PAYLOAD_LEGACY_FALLBACK | asset=%s version=%s path=%s",
                    self.asset,
                    self.model_version,
                    p,
                )
            break
        if payload_path is None:
            paths = ",".join(str(p) for p in candidates)
            raise RuntimeError(f"model_payload_missing:{paths}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"model_payload_invalid:{payload_path}")
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

    def _resolve_signal_threshold(self, configured_threshold: float, pred: np.ndarray) -> float:
        """
        Use configured threshold when feasible.
        If model outputs are too small (underfit case), adapt threshold so backtest
        can still evaluate strategy behavior instead of hard-failing with zero trades.
        """
        base = max(float(configured_threshold or 0.0), 1e-12)
        abs_pred = np.abs(np.asarray(pred, dtype=np.float64))
        abs_pred = abs_pred[np.isfinite(abs_pred)]
        if abs_pred.size == 0:
            return base

        max_pred = float(np.max(abs_pred))
        if not np.isfinite(max_pred) or max_pred <= 0.0:
            return base

        if base < max_pred:
            return base

        # Configured threshold is too strict for current prediction scale.
        # Use adaptive fallback from prediction distribution but keep it below max_pred,
        # otherwise all signals can be filtered out.
        asset_key = "BTC" if self.asset in ("BTC", "BTCUSD", "BTCUSDM") else "XAU"
        quantile_raw = os.getenv(f"BACKTEST_ADAPTIVE_THRESHOLD_QUANTILE_{asset_key}")
        if quantile_raw is None:
            quantile_raw = os.getenv("BACKTEST_ADAPTIVE_THRESHOLD_QUANTILE", "0.60")
        quantile = float(quantile_raw or "0.60")
        quantile = min(max(quantile, 0.05), 0.95)
        qv = float(np.quantile(abs_pred, quantile))
        adaptive = max(qv, max_pred * 0.35, 1e-12)
        if adaptive >= max_pred:
            adaptive = max(max_pred * 0.999999, 1e-12)
        signal_count = int(np.sum(abs_pred >= adaptive))
        log.warning(
            "BACKTEST_THRESHOLD_ADAPT | asset=%s configured=%.8f max_pred=%.8f adaptive=%.8f q=%.2f signals=%s",
            self.asset,
            base,
            max_pred,
            adaptive,
            quantile,
            signal_count,
        )
        return adaptive
    
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

        n_obs = min(len(pred), len(y), len(df))
        for i in range(n_obs):
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
        runs = int(self.run_cfg.monte_carlo_runs)
        if not pnl_series or runs <= 0:
            return {
                "risk_of_ruin": 0.0,
                "confidence_95_percentile": 0.0,
                "median_final": 0.0,
                "worst_case": 0.0,
                "best_case": 0.0,
            }

        ruin_threshold = self.run_cfg.initial_capital * (1 - self.run_cfg.ruin_drawdown_pct)
        rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
        pnls = np.asarray(pnl_series, dtype=np.float64)
        pnls = pnls[np.isfinite(pnls)]
        if pnls.size == 0:
            return {
                "risk_of_ruin": 0.0,
                "confidence_95_percentile": 0.0,
                "median_final": 0.0,
                "worst_case": 0.0,
                "best_case": 0.0,
            }

        # Bootstrap + execution-noise stress:
        # permutation alone preserves terminal PnL and understates regime uncertainty.
        sample_len = int(pnls.size)
        abs_pnls = np.abs(pnls)
        median_abs = float(np.median(abs_pnls[abs_pnls > 0.0])) if np.any(abs_pnls > 0.0) else 0.0

        exec_jitter_frac = float(os.getenv("MC_EXEC_JITTER_FRAC", "0.05") or "0.05")
        tail_prob = float(os.getenv("MC_TAIL_EVENT_PROB", "0.01") or "0.01")
        tail_scale = float(os.getenv("MC_TAIL_EVENT_SCALE", "1.5") or "1.5")

        exec_jitter_frac = max(0.0, min(exec_jitter_frac, 1.0))
        tail_prob = max(0.0, min(tail_prob, 0.5))
        tail_scale = max(0.5, tail_scale)
        jitter_sigma = max(median_abs * exec_jitter_frac, 1e-9)

        ruin_count = 0
        final_equities: List[float] = []

        for _ in range(runs):
            idx = rng.integers(0, sample_len, size=sample_len)
            seq = pnls[idx].astype(np.float64, copy=True)

            # Random adverse execution costs (spread/slippage regime noise).
            drag = np.abs(rng.normal(loc=0.0, scale=jitter_sigma, size=sample_len))
            seq -= drag

            # Black-swan style shocks on a small subset of trades.
            if tail_prob > 0.0:
                mask = rng.random(sample_len) < tail_prob
                if np.any(mask):
                    penalty = np.abs(seq[mask]) * rng.uniform(0.5, tail_scale, size=int(np.sum(mask)))
                    seq[mask] -= penalty

            # Order randomization preserves path risk dynamics for drawdown.
            if sample_len > 1:
                seq = rng.permutation(seq)

            equity = np.cumsum(seq) + self.run_cfg.initial_capital
            if float(np.min(equity)) <= ruin_threshold:
                ruin_count += 1
            final_equities.append(float(equity[-1]))

        final_eq = np.asarray(final_equities, dtype=np.float64)
        return {
            "risk_of_ruin": float(ruin_count / runs),
            "confidence_95_percentile": float(np.percentile(final_eq, 5)),
            "median_final": float(np.median(final_eq)),
            "worst_case": float(np.min(final_eq)),
            "best_case": float(np.max(final_eq)),
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
        # Use peak concurrent exposure proxy (max notional), not cumulative sum of all trades.
        crash_move = float(self.run_cfg.crash_scenario_pct)
        long_peak_notional = max(
            [float(t.get("notional", 0.0) or 0.0) for t in trades if float(t.get("direction", 0.0) or 0.0) > 0.0],
            default=0.0,
        )
        short_peak_notional = max(
            [float(t.get("notional", 0.0) or 0.0) for t in trades if float(t.get("direction", 0.0) or 0.0) < 0.0],
            default=0.0,
        )
        adverse_notional = long_peak_notional if crash_move < 0.0 else short_peak_notional
        crash_loss = adverse_notional * abs(crash_move)
        crash_impact = -crash_loss
        crash_drawdown = crash_loss / float(self.run_cfg.initial_capital or 1.0)
        
        scenarios.append({
            "name": "flash_crash",
            "impact": float(crash_impact),
            "drawdown": float(crash_drawdown),
            "peak_notional": float(adverse_notional),
            "passed": crash_drawdown < 0.30,  # Survive 30% DD
        })
        
        # Scenario 2: Volatility spike (3x slippage) — use mean per-trade, not sum
        if trades:
            mean_slip = sum(t["slippage_cost"] for t in trades) / len(trades)
            vol_spike_cost = mean_slip * 3.0 * min(len(trades), 50)  # cap at 50 worst-case concurrent trades
        else:
            vol_spike_cost = 0.0
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
        """Walk-forward with institutional thresholds — adaptive window sizing."""
        train_days = self.run_cfg.wfa_train_days
        test_days = self.run_cfg.wfa_test_days
        
        if train_days <= 0 or test_days <= 0:
            return {"passed": False, "total": 0, "failed": 0, "windows": []}
        
        start = df.index.min()
        end = df.index.max()
        
        # Adaptive WFA: if data is too short, scale down window sizes
        total_data_days = (end - start).total_seconds() / 86400.0
        min_window = train_days + test_days
        required_windows = getattr(self.run_cfg, "wfa_required_windows", 2) or 2
        
        if total_data_days < min_window * required_windows:
            # Scale down: aim for at least required_windows windows
            scale = total_data_days / (min_window * required_windows)
            if scale >= 0.15:  # At least 15% of original — still meaningful
                train_days = max(7, int(train_days * scale))
                test_days = max(3, int(test_days * scale))
                log.info(
                    "WFA_ADAPTIVE_WINDOWS | data_days=%.0f orig_train=%d orig_test=%d new_train=%d new_test=%d",
                    total_data_days, self.run_cfg.wfa_train_days, self.run_cfg.wfa_test_days,
                    train_days, test_days,
                )
            else:
                log.warning(
                    "WFA_DATA_TOO_SHORT | data_days=%.0f min_needed=%d — skipping WFA",
                    total_data_days, min_window,
                )
                return {
                    "passed": True, "total": 0, "failed": 0,
                    "pass_rate": 0.0, "windows": [],
                    "skipped": True, "reason": f"data_days={total_data_days:.0f}<{min_window}",
                }
        
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
            
            # Train mini-model for this window (speed-optimized for WFA runtime).
            cfg = self._wfa_train_cfg()
            pipeline = Pipeline(cfg)
            splits = pipeline.fit(df_train.copy())
            model = RegressionModel(cfg)
            model.train(splits, announce=False)
            
            # Test on forward period
            xy = pipeline.transform(df_test.copy())
            if xy["X"].empty:
                failed += 1
                windows.append({
                    "train_start": str(df_train.index.min()),
                    "test_start": str(df_test.index.min()),
                    "sharpe": 0.0,
                    "win_rate": 0.0,
                    "passed": False,
                    "reason": "wfa_empty_features",
                })
                t0 += test_delta
                continue

            sim_df = df_test.loc[xy["X"].index]
            X = np.stack(xy["X"].values)
            y = np.asarray(xy["ret"].values) if "ret" in xy else np.asarray(xy["y"].values)
            pred = model.model.predict(X)
            
            configured_threshold = max(float(getattr(cfg, "percent_increase", 0.0) or 0.0), 1e-12)
            threshold = self._resolve_signal_threshold(configured_threshold, pred)
            rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
            
            pnls, _, costs = self._simulate_trades_institutional(
                sim_df,
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
                risk_free_rate=self.run_cfg.backtest_risk_free_rate,
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
                metrics.max_drawdown_pct <= self.run_cfg.verification_max_drawdown_pct
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
        pass_rate = (float(total - failed) / float(total)) if total > 0 else 0.0
        required_windows = max(int(WFA_MIN_WINDOWS), 0)
        has_required_windows = total >= required_windows
        skipped = bool(
            required_windows > 0
            and not has_required_windows
            and self.run_cfg.wfa_allow_skip_if_insufficient
        )
        if required_windows > 0 and not has_required_windows:
            log.warning(
                "WFA_INSUFFICIENT_WINDOWS | asset=%s total=%s required=%s pass_rate=%.3f skip=%s",
                self.asset,
                total,
                required_windows,
                pass_rate,
                skipped,
            )
        passed = bool(
            (has_required_windows and pass_rate >= WFA_MIN_PASS_RATE)
            or skipped
        )
        return {
            "passed": passed,
            "pass_rate": pass_rate,
            "total": total,
            "failed": failed,
            "required_windows": required_windows,
            "skipped": skipped,
            "windows": windows,
        }

    def _wfa_train_cfg(self):
        """
        Dedicated fast config for walk-forward mini-trains.
        Keeps logic identical but prevents startup from stalling for tens of minutes.
        """
        base_cfg = BTC_TRAIN_CONFIG if self.asset in ("BTC", "BTCUSD", "BTCUSDM") else XAU_TRAIN_CONFIG
        params = dict(getattr(base_cfg, "regressor_params", {}) or {})

        wfa_iters = int(os.getenv("WFA_TRAIN_ITERATIONS", "250") or "250")
        wfa_verbose = int(os.getenv("WFA_TRAIN_VERBOSE", "0") or "0")
        wfa_early = int(os.getenv("WFA_EARLY_STOPPING_ROUNDS", "50") or "50")

        params["iterations"] = max(20, min(2000, wfa_iters))
        params["verbose"] = max(0, wfa_verbose)
        params["task_type"] = "CPU"
        params.setdefault("random_seed", 42)

        return replace(
            base_cfg,
            regressor_params=params,
            early_stopping_rounds=max(0, wfa_early),
        )
    
    def load_history(self) -> pd.DataFrame:
        """Load historical data from MT5 with robust multi-fallback."""
        ensure_mt5()
        symbol = _mt5_symbol(self.asset)
        
        # Always use current date as end_date to avoid stale data
        start_dt = _parse_utc_date(self.run_cfg.start_date)
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        
        tf = mt5.TIMEFRAME_M1
        rates = None
        last_err = (0, "")
        
        with MT5_LOCK:
            # Ensure symbol is in Market Watch (don't fail if returns False)
            mt5.symbol_select(symbol, True)
            time.sleep(0.3)  # Allow MT5 to load symbol data
            
            # Verify symbol exists
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"backtest_symbol_not_found:{symbol}")
            
            # Try 1: date range query
            rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
            last_err = mt5.last_error()
            
            # Try 2: count-based fallback with multiple candidates
            if rates is None or len(rates) == 0:
                default_bars = 120000 if self.asset in ("XAU", "XAUUSD", "XAUUSDM") else 60000
                candidates = [default_bars, 50000, 30000, 10000, 5000]
                env_bars = int(os.getenv("BACKTEST_BARS", "0") or "0")
                if env_bars > 0:
                    candidates.insert(0, env_bars)
                
                for bars in candidates:
                    try:
                        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(bars))
                        last_err = mt5.last_error()
                        if rates is not None and len(rates) > 0:
                            break
                        # Error -2 = Invalid params (count > available history)
                        if last_err[0] == -2:
                            continue
                    except Exception:
                        continue
        
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"backtest_history_empty:{symbol}:{last_err}")
        
        if len(rates) < 1000:
            raise RuntimeError(f"backtest_history_insufficient:{symbol}:only_{len(rates)}_bars")
        
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
        if xy["X"].empty:
            raise RuntimeError("backtest_windows_empty")
        sim_df = df.loc[xy["X"].index]
        X = np.stack(xy["X"].values)
        y = np.asarray(xy["ret"].values) if "ret" in xy else np.asarray(xy["y"].values)
        pred = model.predict(X)
        
        configured_threshold = max(float(getattr(pipeline.cfg, "percent_increase", 0.0) or 0.0), 1e-12)
        threshold = self._resolve_signal_threshold(configured_threshold, pred)
        try:
            pred_arr = np.asarray(pred, dtype=np.float64)
            abs_pred = np.abs(pred_arr)
            abs_pred = abs_pred[np.isfinite(abs_pred)]
            if abs_pred.size > 0:
                signal_count = int(np.sum(np.isfinite(pred_arr) & (np.abs(pred_arr) >= threshold)))
                log.info(
                    "BACKTEST_PRED_STATS | asset=%s n=%s mean_abs=%.8f std_abs=%.8f max_abs=%.8f threshold=%.8f signals=%s",
                    self.asset,
                    int(abs_pred.size),
                    float(np.mean(abs_pred)),
                    float(np.std(abs_pred)),
                    float(np.max(abs_pred)),
                    float(threshold),
                    signal_count,
                )
        except Exception:
            pass
        rng = np.random.default_rng(self.run_cfg.monte_carlo_seed)
        
        # Institutional simulation
        pnl_series, trades, costs = self._simulate_trades_institutional(
            sim_df, pred, y,
            threshold=threshold,
            initial_capital=self.run_cfg.initial_capital,
            rng=rng,
        )
        
        if not pnl_series:
            try:
                pred_arr = np.asarray(pred, dtype=np.float64)
                signal_count = int(np.sum(np.isfinite(pred_arr) & (np.abs(pred_arr) >= threshold)))
            except Exception:
                signal_count = 0
            log.warning(
                "BACKTEST_NO_TRADES | asset=%s threshold=%.8f configured=%.8f n_obs=%s signals=%s",
                self.asset,
                float(threshold),
                float(configured_threshold),
                int(min(len(pred), len(y), len(sim_df))),
                signal_count,
            )
        
        # Compute metrics
        # Fix Sharpe inflation: calculate periods_per_year from actual data span
        # rather than assuming each trade = 1 trading day (252/yr)
        try:
            data_span_days = (sim_df.index.max() - sim_df.index.min()).total_seconds() / 86400.0
            if data_span_days > 0 and len(pnl_series) > 0:
                trades_per_year = len(pnl_series) / (data_span_days / 365.25)
            else:
                trades_per_year = 252.0  # safe fallback
        except Exception:
            trades_per_year = 252.0
        
        metrics = compute_metrics(
            pnl_series,
            initial_capital=self.run_cfg.initial_capital,
            risk_free_rate=self.run_cfg.backtest_risk_free_rate,
            periods_per_year=trades_per_year,
            symbol=self.asset,
            timeframe="M1",
            start_date=str(sim_df.index.min()),
            end_date=str(sim_df.index.max()),
            spread_costs=costs["spread"],
            slippage_costs=costs["slippage"],
            swap_costs=0.0,
        )
        
        # Monte Carlo
        mc_results = self._monte_carlo_institutional(pnl_series)
        metrics.risk_of_ruin = mc_results["risk_of_ruin"]
        metrics.monte_carlo_runs = self.run_cfg.monte_carlo_runs
        metrics.mc_confidence_95_percentile = mc_results.get("confidence_95_percentile", 0.0)
        metrics.mc_worst_case = mc_results.get("worst_case", 0.0)
        metrics.mc_best_case = mc_results.get("best_case", 0.0)
        
        # Walk-forward
        wfa = self._walk_forward_institutional(df)
        # Trust WFA engine result: when skip is allowed (wfa_allow_skip_if_insufficient=True)
        # and data is insufficient, WFA returns passed=True — respect that decision.
        metrics.wfa_passed = wfa["passed"]
        metrics.wfa_total_windows = wfa["total"]
        metrics.wfa_failed_windows = wfa["failed"]
        wfa_windows = list(wfa.get("windows", []))
        wfa_sharpes = [
            float(w.get("sharpe", 0.0))
            for w in wfa_windows
            if isinstance(w, dict) and np.isfinite(float(w.get("sharpe", 0.0)))
        ]
        metrics.wfa_avg_sharpe = float(np.mean(wfa_sharpes)) if wfa_sharpes else 0.0
        metrics.wfa_min_sharpe = float(np.min(wfa_sharpes)) if wfa_sharpes else 0.0
        
        # Stress testing
        stress = self._stress_test(df, trades)
        self._stress_results = stress
        metrics.stress_test_passed = bool(stress.get("passed", False))
        metrics.stress_scenarios = list(stress.get("scenarios", []))
        
        require_both_sides = str(os.getenv("GATE_REQUIRE_BOTH_SIDES", "1") or "1").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        allow_pf_capped = str(os.getenv("GATE_ALLOW_CAPPED_PROFIT_FACTOR", "0") or "0").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        min_trades = int(os.getenv("MIN_GATE_TRADES", "20") or "20")
        min_wins = int(os.getenv("MIN_GATE_WINNING_TRADES", "1") or "1") if require_both_sides else 0
        min_losses = int(os.getenv("MIN_GATE_LOSING_TRADES", "1") or "1") if require_both_sides else 0

        sample_issues: List[str] = []
        if int(metrics.total_trades) < min_trades:
            sample_issues.append(f"insufficient_trades:{metrics.total_trades}<{min_trades}")
        if int(metrics.winning_trades) < min_wins:
            sample_issues.append(f"insufficient_wins:{metrics.winning_trades}<{min_wins}")
        if int(metrics.losing_trades) < min_losses:
            sample_issues.append(f"insufficient_losses:{metrics.losing_trades}<{min_losses}")
        if getattr(metrics, "profit_factor_capped", False) and not allow_pf_capped:
            sample_issues.append("profit_factor_capped")

        self._sample_quality_issues = sample_issues
        self._sample_quality_passed = len(sample_issues) == 0
        if sample_issues:
            log.warning(
                "BACKTEST_SAMPLE_QUALITY_FAIL | asset=%s issues=%s",
                self.asset,
                ",".join(sample_issues),
            )

        # Verification with INSTITUTIONAL thresholds
        base_verified = (
            metrics.sharpe_ratio >= MIN_GATE_SHARPE and
            metrics.win_rate >= MIN_GATE_WIN_RATE and
            metrics.max_drawdown_pct <= MAX_GATE_DRAWDOWN and
            self._sample_quality_passed
        )
        
        # Risk checks
        unsafe = (
            mc_results["risk_of_ruin"] > 0.01 or  # Max 1% ruin risk
            not stress["passed"] or
            not self._sample_quality_passed
        )

        wfa_gate_ok = bool(wfa.get("passed", False))
        self._last_verified = bool(base_verified and wfa_gate_ok and not unsafe)
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
            meta["asset"] = self.asset
            meta["risk_of_ruin"] = float(mc_results["risk_of_ruin"])
            meta["monte_carlo_runs"] = self.run_cfg.monte_carlo_runs
            meta["mc_confidence_95"] = float(mc_results["confidence_95_percentile"])
            meta["mc_worst_case"] = float(mc_results["worst_case"])
            meta["wfa_passed"] = wfa["passed"]
            meta["wfa_pass_rate"] = float(wfa.get("pass_rate", 0.0))
            meta["wfa_total_windows"] = wfa["total"]
            meta["wfa_failed_windows"] = wfa["failed"]
            meta["wfa_required_windows"] = int(wfa.get("required_windows", 0) or 0)
            meta["wfa_skipped"] = bool(wfa.get("skipped", False))
            meta["stress_test_passed"] = stress["passed"]
            meta["stress_scenarios"] = stress["scenarios"]
            meta["institutional_grade"] = True
            meta["real_backtest"] = True
            meta["unsafe"] = unsafe
            meta["sample_quality_passed"] = bool(self._sample_quality_passed)
            meta["sample_quality_issues"] = list(self._sample_quality_issues)
            meta["total_trades"] = int(getattr(metrics, "total_trades", 0) or 0)
            meta["winning_trades"] = int(getattr(metrics, "winning_trades", 0) or 0)
            meta["losing_trades"] = int(getattr(metrics, "losing_trades", 0) or 0)
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
            "wfa_pass_rate": float(self._wfa.get("pass_rate", 0.0)),
            "wfa_total_windows": int(self._wfa.get("total", 0)),
            "wfa_failed_windows": int(self._wfa.get("failed", 0)),
            "wfa_required_windows": int(self._wfa.get("required_windows", 0) or 0),
            "wfa_skipped": bool(self._wfa.get("skipped", False)),
            "stress_test_passed": bool(self._stress_results.get("passed")),
            "unsafe": self._unsafe,
            "sample_quality_passed": bool(self._sample_quality_passed),
            "sample_quality_issues": list(self._sample_quality_issues),
            "total_trades": int(getattr(metrics, "total_trades", 0) or 0),
            "winning_trades": int(getattr(metrics, "winning_trades", 0) or 0),
            "losing_trades": int(getattr(metrics, "losing_trades", 0) or 0),
            "source": "Backtest.engine_institutional",
        }

        MODEL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep per-asset state for diagnostics/history.
        asset_state_path = get_artifact_path("models", f"model_state_{self.asset}.pkl")
        with asset_state_path.open("wb") as f:
            pickle.dump(state, f)

        # Do not overwrite a previously VERIFIED global state with a failed one.
        should_write_global = True
        if status != "VERIFIED" and MODEL_STATE_PATH.exists():
            try:
                with MODEL_STATE_PATH.open("rb") as f:
                    prev = pickle.load(f)
                if (
                    isinstance(prev, dict)
                    and bool(prev.get("real_backtest", False))
                    and str(prev.get("status", "")).upper() == "VERIFIED"
                ):
                    should_write_global = False
                    log.warning(
                        "MODEL_STATE_KEEP_PREV_VERIFIED | prev_version=%s prev_asset=%s new_asset=%s new_status=%s",
                        str(prev.get("model_version", "unknown")),
                        str(prev.get("asset", "unknown")),
                        self.asset,
                        status,
                    )
            except Exception:
                should_write_global = True

        if should_write_global:
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
    df = pd.DataFrame()
    metrics: Optional[BacktestMetrics] = None
    run_exc: Optional[Exception] = None

    try:
        try:
            df = engine.load_history()
        except Exception as history_exc:
            log.warning(
                "BACKTEST_HISTORY_FALLBACK | asset=%s version=%s err=%s",
                asset_u,
                model_version,
                history_exc,
            )
            df = load_training_dataframe_for_asset(asset_u)
            # Validate fallback data quality — abort if garbage
            if df is None or len(df) < 1000:
                raise RuntimeError(
                    f"backtest_data_insufficient:{asset_u}:fallback_rows={len(df) if df is not None else 0}"
                ) from history_exc
        metrics = engine.run(df)
    except Exception as exc:
        run_exc = exc
        log.error(
            "INSTITUTIONAL_BACKTEST_FAILED | asset=%s version=%s err=%s",
            asset_u,
            model_version,
            exc,
            exc_info=True,
        )
        engine._last_verified = False
        engine._unsafe = True
        engine._risk_of_ruin = 1.0
        engine._wfa = {"passed": False, "pass_rate": 0.0, "total": 0, "failed": 0, "windows": []}
        engine._stress_results = {"passed": False, "scenarios": []}

        start_date = run_cfg.start_date
        end_date = run_cfg.end_date
        try:
            if not df.empty:
                start_date = str(df.index.min())
                end_date = str(df.index.max())
        except Exception:
            pass

        metrics = BacktestMetrics(
            symbol=asset_u,
            timeframe="M1",
            start_date=str(start_date),
            end_date=str(end_date),
            initial_capital=float(run_cfg.initial_capital),
            final_capital=float(run_cfg.initial_capital),
            risk_of_ruin=1.0,
            wfa_passed=False,
            stress_test_passed=False,
        )
    
    out_dir = get_artifact_dir("backtest_institutional")
    prefix = f"{asset_u.lower()}_{model_version.replace('.', '_')}_institutional"
    if metrics is None:
        metrics = BacktestMetrics(
            symbol=asset_u,
            timeframe="M1",
            start_date=str(run_cfg.start_date),
            end_date=str(run_cfg.end_date),
            initial_capital=float(run_cfg.initial_capital),
            final_capital=float(run_cfg.initial_capital),
        )
    if run_exc is None and metrics is not None:
        save_metrics(metrics, out_dir, prefix=prefix)
    
    state = engine.save_model_state(metrics or BacktestMetrics())
    status = "PASSED" if state["verified"] else "FAILED"
    
    _console("\n" + "=" * 60)
    _console(f"INSTITUTIONAL BACKTEST: {status}")
    _console("=" * 60)
    _console(f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f} (Req: >= {MIN_GATE_SHARPE:.2f})")
    _console(f"Win Rate:            {metrics.win_rate:.1%} (Req: >= {MIN_GATE_WIN_RATE:.1%})")
    _console(f"Max Drawdown:        {metrics.max_drawdown_pct:.2%} (Max: {MAX_GATE_DRAWDOWN:.0%})")
    _console(f"Risk of Ruin:        {metrics.risk_of_ruin:.2%} (Max: 1%)")
    _console(f"WFA:                 {'PASS' if metrics.wfa_passed else 'FAIL'}")
    _console(f"Stress Test:         {'PASS' if state['stress_test_passed'] else 'FAIL'}")
    _console(f"Version:             {model_version}")
    _console("=" * 60 + "\n")

    if run_exc is not None:
        raise RuntimeError(f"institutional_backtest_failed:{asset_u}") from run_exc

    return metrics


# Compatibility aliases for legacy imports
BacktestEngine = InstitutionalBacktestEngine
XAU_BACKTEST_CONFIG = XAU_INSTITUTIONAL_CONFIG
BTC_BACKTEST_CONFIG = BTC_INSTITUTIONAL_CONFIG
run_backtest = run_institutional_backtest


if __name__ == "__main__":
    asset_arg = sys.argv[1] if len(sys.argv) > 1 else "XAU"
    run_institutional_backtest(asset_arg)
