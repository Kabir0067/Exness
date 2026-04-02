"""
Regression tests proving the FeatureEngine has NO lookahead bias when shift=1.

Core invariant:
  When shift=1, compute_indicators(df, shift=1) must produce IDENTICAL results
  whether or not the last row of the DataFrame changes (because it should be
  invisible to the engine).

Test strategy:
  1. Create a synthetic OHLCV DataFrame with known values.
  2. Compute features with shift=1.
  3. Mutate ONLY the last row (the "future" bar).
  4. Recompute features with shift=1.
  5. Assert all indicator values are IDENTICAL.
     Any difference = lookahead bias.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from core.config import XAUEngineConfig
from core.feature_engine import FeatureEngine, _shifted_last


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling XAU price action."""
    rng = np.random.RandomState(seed)
    # Random walk for close prices starting at 2000
    returns = rng.normal(0.0001, 0.005, n)
    close = 2000.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.randint(100, 5000, n).astype(float)
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="1min"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "tick_volume": volume,
    })
    return df


class TestNoLookaheadBias(unittest.TestCase):
    """
    The definitive lookahead test:
    mutate the last bar → results with shift=1 must not change.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = XAUEngineConfig()
        cls.fe = FeatureEngine(cls.cfg)
        cls.df_base = _make_ohlcv(300, seed=42)

    def _compute(self, df: pd.DataFrame) -> dict:
        tf = self.cfg.symbol_params.tf_primary  # "M1"
        result = self.fe.compute_indicators({tf: df}, shift=1)
        return result.get(tf, {})

    def test_mutating_last_bar_does_not_change_indicators(self):
        """
        CORE TEST: Change last bar's OHLCV drastically.
        With shift=1, all indicators must remain identical.
        """
        df1 = self.df_base.copy()
        ind1 = self._compute(df1)
        self.assertTrue(len(ind1) > 0, "Indicators must not be empty")

        # Mutate last bar to extreme values
        df2 = self.df_base.copy()
        df2.iloc[-1, df2.columns.get_loc("Close")] = 9999.0
        df2.iloc[-1, df2.columns.get_loc("High")] = 10500.0
        df2.iloc[-1, df2.columns.get_loc("Low")] = 9500.0
        df2.iloc[-1, df2.columns.get_loc("Open")] = 9800.0
        df2.iloc[-1, df2.columns.get_loc("tick_volume")] = 999999.0
        ind2 = self._compute(df2)

        # Compare every numeric indicator
        mismatches = []
        for key in ind1:
            v1, v2 = ind1[key], ind2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if math.isfinite(v1) and math.isfinite(v2):
                    if abs(v1 - v2) > 1e-10:
                        mismatches.append(f"{key}: {v1} != {v2}")
                elif math.isfinite(v1) != math.isfinite(v2):
                    mismatches.append(f"{key}: {v1} != {v2}")
            elif isinstance(v1, str) and isinstance(v2, str):
                if v1 != v2:
                    mismatches.append(f"{key}: '{v1}' != '{v2}'")

        self.assertEqual(
            mismatches, [],
            f"Lookahead bias detected! These indicators changed when "
            f"only the last bar was mutated:\n" + "\n".join(mismatches),
        )

    def test_atr_uses_shifted_value(self):
        """ATR must use arr[-2] when shift=1, not arr[-1]."""
        df = self.df_base.copy()
        ind = self._compute(df)
        # Manually compute what shifted ATR should be
        import talib
        c = df["Close"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        l = df["Low"].values.astype(np.float64)
        atr_arr = talib.ATR(h, l, c, timeperiod=14)
        expected_atr = float(atr_arr[-2])  # shift=1 means previous bar
        self.assertAlmostEqual(
            ind["atr"], expected_atr, places=6,
            msg="ATR should use arr[-2] when shift=1",
        )

    def test_adx_uses_shifted_value(self):
        """ADX must use arr[-2] when shift=1, not arr[-1]."""
        df = self.df_base.copy()
        ind = self._compute(df)
        import talib
        h = df["High"].values.astype(np.float64)
        l = df["Low"].values.astype(np.float64)
        c = df["Close"].values.astype(np.float64)
        adx_arr = talib.ADX(h, l, c, timeperiod=14)
        adx_arr = np.where(np.isfinite(adx_arr), adx_arr, 0.0)
        expected_adx = float(adx_arr[-2])
        self.assertAlmostEqual(
            ind["adx"], expected_adx, places=6,
            msg="ADX should use arr[-2] when shift=1",
        )

    def test_bb_uses_shifted_values(self):
        """Bollinger Bands must use shifted values."""
        df = self.df_base.copy()
        ind = self._compute(df)
        import talib
        c = df["Close"].values.astype(np.float64)
        bb_u, bb_m, bb_l = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        bb_u = np.where(np.isfinite(bb_u), bb_u, c[0])
        bb_l = np.where(np.isfinite(bb_l), bb_l, c[0])
        bb_m = np.where(np.isfinite(bb_m), bb_m, c[0])
        self.assertAlmostEqual(ind["bb_upper"], float(bb_u[-2]), places=4)
        self.assertAlmostEqual(ind["bb_lower"], float(bb_l[-2]), places=4)
        self.assertAlmostEqual(ind["bb_middle"], float(bb_m[-2]), places=4)

    def test_close_uses_shifted_value(self):
        """Close must be c[-2] when shift=1."""
        df = self.df_base.copy()
        ind = self._compute(df)
        expected = float(df["Close"].values[-2])
        self.assertAlmostEqual(ind["close"], expected, places=6)

    def test_smart_money_patterns_ignore_last_bar(self):
        """
        FVG, sweep, OB, stop-hunt, divergence, forecast should not
        change when only the last bar is mutated.
        """
        df1 = self.df_base.copy()
        ind1 = self._compute(df1)

        df2 = self.df_base.copy()
        df2.iloc[-1, df2.columns.get_loc("Close")] = 1.0  # extreme
        df2.iloc[-1, df2.columns.get_loc("High")] = 1.0
        df2.iloc[-1, df2.columns.get_loc("Low")] = 0.5
        df2.iloc[-1, df2.columns.get_loc("Open")] = 0.8
        ind2 = self._compute(df2)

        pattern_keys = [
            "fvg", "sweep", "order_block", "stop_hunt_side",
            "stop_hunt_strength", "ob_touch_proximity",
            "ob_pretouch_bias", "divergence", "forecast",
        ]
        for key in pattern_keys:
            v1 = ind1.get(key)
            v2 = ind2.get(key)
            if isinstance(v1, float) and isinstance(v2, float):
                self.assertAlmostEqual(
                    v1, v2, places=8,
                    msg=f"Lookahead in pattern '{key}': {v1} != {v2}",
                )
            else:
                self.assertEqual(
                    v1, v2,
                    msg=f"Lookahead in pattern '{key}': {v1} != {v2}",
                )

    def test_shift0_uses_current_bar(self):
        """Sanity: shift=0 SHOULD use the current bar (no shift)."""
        tf = self.cfg.symbol_params.tf_primary
        df = self.df_base.copy()
        ind = self.fe.compute_indicators({tf: df}, shift=0).get(tf, {})
        import talib
        c = df["Close"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        l = df["Low"].values.astype(np.float64)
        atr_arr = talib.ATR(h, l, c, timeperiod=14)
        # shift=0 should use arr[-1]
        self.assertAlmostEqual(
            ind["atr"], float(atr_arr[-1]), places=6,
            msg="shift=0 should use arr[-1] (current bar)",
        )


class TestShiftedLastHelper(unittest.TestCase):
    """Unit tests for the _shifted_last helper function."""

    def test_shift0_returns_last(self):
        arr = np.array([10.0, 20.0, 30.0])
        self.assertEqual(_shifted_last(arr, 0), 30.0)

    def test_shift1_returns_second_last(self):
        arr = np.array([10.0, 20.0, 30.0])
        self.assertEqual(_shifted_last(arr, 1), 20.0)

    def test_shift_too_large_returns_default(self):
        arr = np.array([10.0, 20.0, 30.0])
        # shift=5: 1+5=6 > len=3, no valid position → return default
        self.assertEqual(_shifted_last(arr, 5), 0.0)
        self.assertEqual(_shifted_last(arr, 5, default=99.0), 99.0)

    def test_empty_returns_default(self):
        arr = np.array([])
        self.assertEqual(_shifted_last(arr, 0, default=99.0), 99.0)

    def test_nan_returns_default(self):
        arr = np.array([1.0, float("nan")])
        self.assertEqual(_shifted_last(arr, 0, default=42.0), 42.0)


if __name__ == "__main__":
    unittest.main()
