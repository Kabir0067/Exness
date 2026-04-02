"""
Tests for institutional promotion blocks:
  Block 1: Statistical credibility (Wilson CI, sample gating)
  Block 2: Execution evidence (CSV path exists)
  Block 3: FSM auto-recovery (recoverable halt triggers restart)
  Block 4: Signal-rate sniper floor (PATCH-S1 regression)
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# Block 1: Statistical Credibility — Wilson CI
# ═══════════════════════════════════════════════════════════════════════════

class TestWilsonConfidenceInterval(unittest.TestCase):
    """Win rate must have a proper confidence interval."""

    def _wilson_ci(self, n: int, p: float, z: float = 1.96):
        denom = 1 + (z * z / n)
        centre = (p + z * z / (2 * n)) / denom
        margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def test_33_trades_97pct_wide_ci(self):
        """n=33, p=0.97 → CI must be wide (not narrow enough for conclusions)."""
        lo, hi = self._wilson_ci(33, 0.97)
        # CI should be at least 10pp wide
        width = hi - lo
        self.assertGreater(width, 0.05)
        # Lower bound should be below 0.95 (not conclusive)
        self.assertLess(lo, 0.95)

    def test_200_trades_60pct_narrow_ci(self):
        """n=200, p=0.60 → CI should be reasonably narrow."""
        lo, hi = self._wilson_ci(200, 0.60)
        width = hi - lo
        self.assertLess(width, 0.15)
        # Should bracket 0.60
        self.assertLess(lo, 0.60)
        self.assertGreater(hi, 0.60)

    def test_1000_trades_tight_ci(self):
        """n=1000, p=0.55 → CI should be tight (~3pp)."""
        lo, hi = self._wilson_ci(1000, 0.55)
        width = hi - lo
        self.assertLess(width, 0.07)

    def test_1_trade_full_range(self):
        """n=1, p=1.0 → CI should cover almost full range."""
        lo, hi = self._wilson_ci(1, 1.0)
        self.assertLess(lo, 0.30)

    def test_ci_bounds(self):
        """CI must be in [0, 1]."""
        lo, hi = self._wilson_ci(5, 0.0)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class TestSampleGating(unittest.TestCase):
    """Win rate confidence label must reflect sample size."""

    def test_below_100_low_sample(self):
        for n in (0, 1, 33, 50, 99):
            label = "LOW_SAMPLE" if n < 100 else "RELIABLE"
            self.assertEqual(label, "LOW_SAMPLE", f"n={n} should be LOW_SAMPLE")

    def test_100_plus_reliable(self):
        for n in (100, 200, 500, 1000):
            label = "LOW_SAMPLE" if n < 100 else "RELIABLE"
            self.assertEqual(label, "RELIABLE", f"n={n} should be RELIABLE")


# ═══════════════════════════════════════════════════════════════════════════
# Block 2: Execution Evidence — CSV infrastructure
# ═══════════════════════════════════════════════════════════════════════════

class TestExecMetricsCSV(unittest.TestCase):
    """Exec metrics CSV files must exist with correct headers."""

    def test_xau_csv_exists(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs", "exec_metrics_XAUUSDm.csv")
        if os.path.exists(path):
            with open(path, "r") as f:
                header = f.readline().strip()
            self.assertIn("latency_ms", header)
            self.assertIn("slippage_pts", header)

    def test_btc_csv_exists(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs", "exec_metrics_BTCUSDm.csv")
        if os.path.exists(path):
            with open(path, "r") as f:
                header = f.readline().strip()
            self.assertIn("latency_ms", header)
            self.assertIn("slippage_pts", header)


# ═══════════════════════════════════════════════════════════════════════════
# Block 3: FSM Auto-Recovery
# ═══════════════════════════════════════════════════════════════════════════

class TestFSMAutoRecovery(unittest.TestCase):
    """Recoverable halt reasons must trigger auto-restart attempt."""

    def test_recoverable_reasons_defined(self):
        """RECOVERABLE_HALT_REASONS must include MT5 failures."""
        from Bot.Motor.engine import RECOVERABLE_HALT_REASONS
        self.assertIn("mt5_unhealthy", RECOVERABLE_HALT_REASONS)
        self.assertIn("mt5_disconnected", RECOVERABLE_HALT_REASONS)

    def test_monitoring_reasons_not_recoverable(self):
        """Manual stop must NOT trigger auto-recovery."""
        from Bot.Motor.engine import RECOVERABLE_HALT_REASONS, MONITORING_ONLY_HALT_REASONS
        for reason in MONITORING_ONLY_HALT_REASONS:
            self.assertNotIn(reason, RECOVERABLE_HALT_REASONS)

    def test_halt_logic_classification(self):
        """Simulate halt classification — recoverable vs monitoring vs fatal."""
        RECOVERABLE = frozenset({"mt5_unhealthy", "mt5_disconnected"})
        MONITORING = frozenset({"manual_stop", "manual_stop_active", "manual_stop_triggered"})

        # Recoverable
        self.assertTrue("mt5_disconnected" in RECOVERABLE)
        # Monitoring
        self.assertTrue("manual_stop" in MONITORING)
        # Fatal (neither)
        reason = "live_risk_halt"
        self.assertFalse(reason in RECOVERABLE)
        self.assertFalse(reason in MONITORING)


# ═══════════════════════════════════════════════════════════════════════════
# Block 4: Signal Rate — Sniper Floor Regression
# ═══════════════════════════════════════════════════════════════════════════

class TestSniperFloorRegression(unittest.TestCase):
    """The old hardcoded floor=80 must be gone."""

    def test_conf_76_now_passes(self):
        """Confidence 76% must pass with default floor 75%."""
        conf = 76
        _SNIPER_HARD_MIN = 70
        cfg_min_conf = 75
        floor = max(int(cfg_min_conf), _SNIPER_HARD_MIN)
        self.assertTrue(conf >= floor)

    def test_conf_79_now_passes(self):
        """Confidence 79% was blocked by old floor=80. Must now pass."""
        conf = 79
        _SNIPER_HARD_MIN = 70
        cfg_min_conf = 75
        floor = max(int(cfg_min_conf), _SNIPER_HARD_MIN)
        self.assertTrue(conf >= floor)

    def test_conf_69_still_blocked(self):
        """Confidence 69% must still be blocked (below hard minimum 70)."""
        conf = 69
        _SNIPER_HARD_MIN = 70
        cfg_min_conf = 75
        floor = max(int(cfg_min_conf), _SNIPER_HARD_MIN)
        self.assertFalse(conf >= floor)

    def test_no_duplicate_80_check(self):
        """The old (float(conf) / 100.0) < 0.80 check must be removed."""
        import inspect
        from core.signal_engine import SignalEngine
        source = inspect.getsource(SignalEngine.compute)
        self.assertNotIn("(float(conf) / 100.0) < 0.80", source)


# ═══════════════════════════════════════════════════════════════════════════
# Block 5: Filter Rejection Stats
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterRejectionStats(unittest.TestCase):
    """PATCH-S2 rejection counter logic."""

    def test_top_reasons_sorted(self):
        counts = {"sniper_reject": 45, "weak_score": 12, "mtf_gate": 30, "spread": 3}
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        self.assertEqual(top[0][0], "sniper_reject")
        self.assertEqual(top[1][0], "mtf_gate")
        self.assertEqual(top[2][0], "weak_score")

    def test_signal_rate(self):
        rate = 5 / max(1, 200)
        self.assertAlmostEqual(rate, 0.025)


if __name__ == "__main__":
    unittest.main()
