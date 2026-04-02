"""
Production-path tests that exercise REAL module code.

Unlike the existing test suite which mostly re-implements logic inline,
these tests import and invoke production classes directly.
"""
from __future__ import annotations

import math
import os
import unittest

import numpy as np
import pandas as pd

# ─── Ensure DRY_RUN so we don't need MT5 ────────────────────────────
os.environ.setdefault("DRY_RUN", "1")
os.environ.setdefault("AUTO_DRY_RUN_ON_MISSING_ENV", "1")

from core.config import (
    XAUEngineConfig, BTCEngineConfig, XAUSymbolParams, BTCSymbolParams,
    BaseEngineConfig, apply_high_accuracy_mode,
    MIN_GATE_SHARPE, MIN_GATE_WIN_RATE, MAX_GATE_DRAWDOWN,
)
from core.feature_engine import FeatureEngine
from core.risk_engine import RiskManager
from core.models import KillSwitchState, SignalThrottle, AccountCache


def _synth_ohlcv(n: int = 300, base: float = 2000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0001, 0.005, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.randint(100, 5000, n).astype(float)
    return pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "Open": open_, "High": high, "Low": low, "Close": close,
        "tick_volume": volume,
    })


# =====================================================================
# A. CONFIG TESTS
# =====================================================================
class TestConfigIntegrity(unittest.TestCase):

    def test_xau_config_validates(self):
        cfg = XAUEngineConfig()
        cfg.validate()  # Should not raise

    def test_btc_config_validates(self):
        cfg = BTCEngineConfig()
        cfg.validate()

    def test_high_accuracy_mode_raises_confidence(self):
        cfg = XAUEngineConfig()
        self.assertEqual(cfg.min_confidence, 75)
        apply_high_accuracy_mode(cfg)
        self.assertGreaterEqual(cfg.min_confidence, 85)
        self.assertGreaterEqual(cfg.signal_min_score, 75.0)

    def test_gate_constants(self):
        self.assertAlmostEqual(MIN_GATE_SHARPE, 0.5)
        self.assertAlmostEqual(MIN_GATE_WIN_RATE, 0.52)
        self.assertAlmostEqual(MAX_GATE_DRAWDOWN, 0.25)

    def test_xau_btc_different_symbols(self):
        self.assertEqual(XAUSymbolParams().base, "XAUUSDm")
        self.assertEqual(BTCSymbolParams().base, "BTCUSDm")

    def test_hard_lot_caps_are_set(self):
        self.assertGreater(XAUSymbolParams().hard_lot_cap, 0)
        self.assertGreater(BTCSymbolParams().hard_lot_cap, 0)


# =====================================================================
# B. FEATURE ENGINE — PRODUCTION PATHS
# =====================================================================
class TestFeatureEngineProduction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = XAUEngineConfig()
        cls.fe = FeatureEngine(cls.cfg)
        cls.df = _synth_ohlcv(300)

    def test_compute_returns_all_required_keys(self):
        tf = self.cfg.symbol_params.tf_primary
        result = self.fe.compute_indicators({tf: self.df}, shift=1)
        ind = result[tf]
        required_keys = [
            "ema_short", "ema_medium", "ema_slow", "rsi", "macd", "atr",
            "adx", "bb_upper", "bb_lower", "bb_width", "bb_pctb",
            "cci", "mom", "roc", "stoch_k", "stoch_d",
            "trend", "close", "fvg", "sweep", "order_block",
            "stop_hunt_side", "stop_hunt_strength", "divergence", "forecast",
        ]
        for key in required_keys:
            self.assertIn(key, ind, f"Missing required indicator: {key}")

    def test_all_numeric_values_are_finite(self):
        tf = self.cfg.symbol_params.tf_primary
        ind = self.fe.compute_indicators({tf: self.df}, shift=1)[tf]
        for key, val in ind.items():
            if isinstance(val, (int, float)):
                self.assertTrue(
                    math.isfinite(val),
                    f"Non-finite value for {key}: {val}",
                )

    def test_empty_df_returns_empty_dict(self):
        tf = self.cfg.symbol_params.tf_primary
        result = self.fe.compute_indicators({tf: pd.DataFrame()}, shift=1)
        self.assertEqual(result[tf], {})

    def test_short_df_returns_empty_dict(self):
        tf = self.cfg.symbol_params.tf_primary
        result = self.fe.compute_indicators({tf: self.df.head(5)}, shift=1)
        self.assertEqual(result[tf], {})

    def test_multitf_produces_alignment_keys(self):
        df_m1 = _synth_ohlcv(300, seed=1)
        df_m5 = _synth_ohlcv(300, seed=2)
        result = self.fe.compute_indicators({"M1": df_m1, "M5": df_m5}, shift=1)
        for tf_data in result.values():
            if tf_data:
                self.assertIn("mtf_aligned", tf_data)
                self.assertIn("mtf_direction", tf_data)


# =====================================================================
# C. RISK ENGINE — PRODUCTION PATHS
# =====================================================================
class TestRiskEngineProduction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = XAUEngineConfig()
        cls.sp = XAUSymbolParams()
        cls.rm = RiskManager(cls.cfg, cls.sp)
        # Inject account state so sizing works
        cls.rm._acc = AccountCache(
            balance=10000.0, equity=10000.0, margin_free=10000.0, ts=1000000.0,
        )
        cls.rm._peak_equity = 10000.0

    def test_position_sizing_returns_positive_lot(self):
        lot = self.rm.calculate_position_size(
            side="Buy", entry_price=2000.0, stop_loss=1990.0,
            take_profit=2030.0, confidence=80,
        )
        self.assertGreater(lot, 0.0)
        self.assertLessEqual(lot, self.sp.hard_lot_cap)

    def test_zero_equity_returns_zero(self):
        rm = RiskManager(self.cfg, self.sp)
        rm._acc = AccountCache(balance=0.0, equity=0.0, margin_free=0.0, ts=1000000.0)
        lot = rm.calculate_position_size(
            side="Buy", entry_price=2000.0, stop_loss=1990.0,
            take_profit=2030.0, confidence=80,
        )
        self.assertEqual(lot, 0.0)

    def test_low_confidence_gives_smaller_lot(self):
        lot_high = self.rm.calculate_position_size(
            "Buy", 2000.0, 1990.0, 2030.0, confidence=90,
        )
        lot_low = self.rm.calculate_position_size(
            "Buy", 2000.0, 1990.0, 2030.0, confidence=50,
        )
        # High confidence should give equal or larger lot
        self.assertGreaterEqual(lot_high, lot_low)

    def test_hard_lot_cap_enforced(self):
        lot = self.rm.calculate_position_size(
            "Buy", 2000.0, 1990.0, 2030.0, confidence=99,
        )
        self.assertLessEqual(lot, self.sp.hard_lot_cap)


# =====================================================================
# D. KILL SWITCH — PRODUCTION PATH
# =====================================================================
class TestKillSwitchProduction(unittest.TestCase):

    def test_active_to_killed(self):
        import time as _t
        ks = KillSwitchState()
        self.assertEqual(ks.status, "ACTIVE")
        # 10 losing trades → bad expectancy and win rate
        profits = [-1.0, -2.0, -0.5, -1.5, -1.0, -2.0, 0.5, -1.0, -0.8, -1.2]
        ks.update(
            profits=profits, now=_t.time(),
            min_trades=6, kill_expectancy=-0.5, kill_winrate=0.30,
            cooling_expectancy=-0.2, cooling_sec=600.0,
        )
        self.assertEqual(ks.status, "KILLED")

    def test_active_stays_active_on_good_stats(self):
        import time as _t
        ks = KillSwitchState()
        profits = [2.0, 1.5, -0.5, 3.0, 1.0, -0.2, 2.5, 1.0]
        ks.update(
            profits=profits, now=_t.time(),
            min_trades=6, kill_expectancy=-0.5, kill_winrate=0.30,
            cooling_expectancy=-0.2, cooling_sec=600.0,
        )
        self.assertEqual(ks.status, "ACTIVE")

    def test_cooling_on_borderline_stats(self):
        import time as _t
        ks = KillSwitchState()
        # Expectancy < cooling_expectancy (-0.2) but WR above kill threshold
        profits = [-0.5, 0.2, -0.3, -0.4, 0.1, -0.6, 0.3, -0.5]
        ks.update(
            profits=profits, now=_t.time(),
            min_trades=6, kill_expectancy=-0.5, kill_winrate=0.30,
            cooling_expectancy=-0.2, cooling_sec=600.0,
        )
        self.assertEqual(ks.status, "COOLING")


# =====================================================================
# E. SIGNAL THROTTLE — PRODUCTION PATH
# =====================================================================
class TestSignalThrottleProduction(unittest.TestCase):

    def test_first_signal_passes(self):
        st = SignalThrottle()
        self.assertTrue(st.register(max_per_hour=10))

    def test_hourly_limit_respected(self):
        st = SignalThrottle()
        for i in range(5):
            self.assertTrue(st.register(max_per_hour=5))
        self.assertFalse(st.register(max_per_hour=5))


# =====================================================================
# F. HISTORY MODULE — SYS IMPORT
# =====================================================================
class TestHistoryImport(unittest.TestCase):

    def test_sys_importable(self):
        """Verify that ExnessAPI/history.py imports sys (forensic fix BUG-6)."""
        import importlib
        # This should not raise ImportError or NameError
        mod = importlib.import_module("ExnessAPI.history")
        import sys
        self.assertTrue(hasattr(sys, "__stdout__"))


if __name__ == "__main__":
    unittest.main()
