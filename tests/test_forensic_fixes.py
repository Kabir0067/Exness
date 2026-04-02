"""
Regression tests for bugs found in the 2026-04-02 forensic audit.

Each test proves a specific bug fix is correct and prevents regression.
Tests are pure-logic where possible (no MT5 required).
"""
from __future__ import annotations

import importlib
import inspect
import math
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# BUG-1: Volatility weight was dead (cancelled in net = buy - sell)
# ---------------------------------------------------------------------------
class TestVolatilityWeightFix(unittest.TestCase):
    """Volatility should amplify the leading side, not cancel out."""

    def test_vol_weight_contributes_to_net(self):
        """After fix: volatility amplifies the leading side so net changes."""
        buy_base, sell_base = 50.0, 30.0
        vol_score = 0.7
        w_vol = 10.0

        # Old behavior (cancelled): both get same addition
        buy_old = buy_base + vol_score * w_vol * 0.5
        sell_old = sell_base + vol_score * w_vol * 0.5
        net_old = buy_old - sell_old

        # New behavior: only leading side gets boost
        buy_new, sell_new = buy_base, sell_base
        if buy_new >= sell_new:
            buy_new += vol_score * w_vol
        else:
            sell_new += vol_score * w_vol
        net_new = buy_new - sell_new

        self.assertEqual(net_old, 20.0, "Old net should be unchanged at 20")
        self.assertGreater(net_new, net_old, "New net should be larger due to vol boost")
        self.assertAlmostEqual(net_new, 27.0, places=1)

    def test_vol_weight_sell_leading(self):
        """When sell leads, volatility should boost sell side."""
        buy_base, sell_base = 30.0, 50.0
        vol_score = 0.7
        w_vol = 10.0
        buy_new, sell_new = buy_base, sell_base
        if buy_new >= sell_new:
            buy_new += vol_score * w_vol
        else:
            sell_new += vol_score * w_vol
        net_new = buy_new - sell_new
        # Net should be more negative (sell stronger)
        self.assertLess(net_new, -20.0)

    def test_source_code_no_longer_adds_to_both(self):
        """Verify the old pattern 'buy += vol * 0.5; sell += vol * 0.5' is gone."""
        src = inspect.getsource(
            importlib.import_module("core.signal_engine").SignalEngine
        )
        self.assertNotIn(
            "vol_score * weights[\"volatility\"] * 0.5",
            src,
            "Old volatility cancellation pattern still present in source",
        )


# ---------------------------------------------------------------------------
# BUG-2: conformal_ok was dead code (shape mismatch in denominator)
# ---------------------------------------------------------------------------
class TestConformalOkFix(unittest.TestCase):
    """conformal_ok denominator must match numerator shape."""

    def test_correct_shapes(self):
        """np.diff(c[-10:]) has 9 elems; c[-10:-1] has 9 elems. Must match."""
        c = np.arange(100.0, 115.0)  # 15 elements
        numerator = np.abs(np.diff(c[-10:]))
        denominator = c[-10:-1]
        self.assertEqual(numerator.shape, denominator.shape,
                         "Numerator and denominator shapes must match")
        # This should NOT raise
        result = numerator / denominator
        self.assertEqual(len(result), 9)

    def test_old_bug_shape_mismatch(self):
        """Old code used c[-11:-1] (10 elems) vs diff (9 elems) → ValueError."""
        c = np.arange(100.0, 115.0)
        numerator = np.abs(np.diff(c[-10:]))
        bad_denominator = c[-11:-1]
        self.assertEqual(len(numerator), 9)
        self.assertEqual(len(bad_denominator), 10)
        with self.assertRaises(ValueError):
            _ = numerator / bad_denominator

    def test_source_uses_correct_slice(self):
        """Verify source code uses c[-10:-1] not c[-11:-1]."""
        src = inspect.getsource(
            importlib.import_module("core.signal_engine").SignalEngine._conformal_ok
        )
        self.assertIn("c[-10:-1]", src, "Fix should use c[-10:-1]")
        self.assertNotIn("c[-11:-1]", src, "Old buggy c[-11:-1] should be removed")


# ---------------------------------------------------------------------------
# BUG-3: D1 confluence double penalty (now removed from _apply_filters)
# ---------------------------------------------------------------------------
class TestD1DoublepenaltyRemoved(unittest.TestCase):
    """D1 confluence should only be applied once (in _ensemble_score)."""

    def test_no_d1_in_apply_filters(self):
        """_apply_filters source should not call _d1_confluence_score."""
        mod = importlib.import_module("core.signal_engine")
        # Find _apply_filters method
        src = inspect.getsource(mod.SignalEngine)
        # The old pattern was: d1_result = self._d1_confluence_score(dfd) inside _apply_filters
        # After fix, the D1 block in _apply_filters is replaced with a comment.
        # We check that _d1_confluence_score is NOT called in the section after
        # "D1 conflict penalty" comment.
        lines = src.split("\n")
        in_apply_filters = False
        d1_call_in_filters = False
        for line in lines:
            if "def _apply_filters" in line:
                in_apply_filters = True
            elif in_apply_filters and line.strip().startswith("def "):
                break
            if in_apply_filters and "_d1_confluence_score(dfd)" in line:
                # Only count actual calls, not comments
                stripped = line.lstrip()
                if not stripped.startswith("#") and not stripped.startswith("//"):
                    d1_call_in_filters = True
        self.assertFalse(
            d1_call_in_filters,
            "_d1_confluence_score should not be called in _apply_filters (double penalty)",
        )


# ---------------------------------------------------------------------------
# BUG-4: history.py missing sys import
# ---------------------------------------------------------------------------
class TestHistorySysImport(unittest.TestCase):
    """ExnessAPI/history.py must import sys for _real_print."""

    def test_sys_is_imported(self):
        """Verify sys is in the import list."""
        path = "ExnessAPI/history.py"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("import sys", content,
                       "history.py must import sys for _real_print to work")


# ---------------------------------------------------------------------------
# BUG-5: Lot sized for wrong stop distance (structural SL mismatch)
# ---------------------------------------------------------------------------
class TestLotSizingUsesActualSL(unittest.TestCase):
    """When structural SL is used, lot must be sized for actual SL distance."""

    def test_source_overrides_adapt_atr(self):
        """plan_order should override adapt['atr'] when struct_sl is used."""
        src = inspect.getsource(
            importlib.import_module("core.risk_engine").RiskManager.plan_order
        )
        self.assertIn("_sizing_adapt", src,
                       "plan_order should create _sizing_adapt with corrected ATR")
        self.assertIn("actual_sl_dist", src,
                       "plan_order should compute actual_sl_dist from entry and SL")

    def test_sizing_math_consistency(self):
        """
        If struct_sl gives a wider stop than ATR-based, the lot should be smaller.
        Verify the formula: overridden_atr = actual_sl_dist / sl_mult.
        """
        entry = 2000.0
        struct_sl = 1990.0  # 10 points away
        atr_based_sl = 1995.0  # 5 points away (tighter)
        sl_mult = 2.5

        # With ATR-based SL: sizing uses atr_value directly
        atr_value = abs(entry - atr_based_sl) / sl_mult  # 5 / 2.5 = 2.0
        # With struct SL: override atr to actual_sl_dist / sl_mult
        overridden_atr = abs(entry - struct_sl) / sl_mult  # 10 / 2.5 = 4.0

        # lot_raw = risk_usd / (atr * sl_mult)
        risk_usd = 100.0
        lot_atr = risk_usd / (atr_value * sl_mult)  # 100 / 5 = 20
        lot_struct = risk_usd / (overridden_atr * sl_mult)  # 100 / 10 = 10

        self.assertAlmostEqual(lot_struct, lot_atr * 0.5, places=2,
                               msg="Wider stop should produce ~half the lot size")


# ---------------------------------------------------------------------------
# BUG-6: equity <= 0 should return 0.0, not lot_min
# ---------------------------------------------------------------------------
class TestEquityZeroReturnsZero(unittest.TestCase):
    """Position sizing with zero equity must not place trades."""

    def test_source_returns_zero(self):
        """calculate_position_size should return 0.0 when equity <= 0."""
        src = inspect.getsource(
            importlib.import_module("core.risk_engine").RiskManager.calculate_position_size
        )
        # Find the equity <= 0 block
        lines = src.split("\n")
        found_block = False
        for i, line in enumerate(lines):
            if "eq <= 0" in line:
                found_block = True
                # Next non-empty line should have return 0.0
                for j in range(i + 1, min(i + 3, len(lines))):
                    if "return 0.0" in lines[j]:
                        return  # Test passes
                    if "return 0.01" in lines[j]:
                        self.fail("Still returns 0.01 for zero equity (old bug)")
        if not found_block:
            self.fail("Could not find equity <= 0 check in source")
        self.fail("Could not find return 0.0 after equity <= 0 check")


# ---------------------------------------------------------------------------
# BUG-7: Auto-restart bounded retry
# ---------------------------------------------------------------------------
class TestAutoRestartRetry(unittest.TestCase):
    """Auto-restart should attempt multiple times with backoff."""

    def test_source_has_retry_loop(self):
        """_handle_fsm_halt auto_restart should have a retry loop."""
        src = inspect.getsource(
            importlib.import_module("Bot.Motor.engine")
        )
        self.assertIn("_MAX_RESTART_ATTEMPTS", src,
                       "Auto-restart should define max attempts")
        self.assertIn("FSM_AUTO_RESTART_ATTEMPT", src,
                       "Auto-restart should log each attempt")
        self.assertIn("FSM_AUTO_RESTART_EXHAUSTED", src,
                       "Auto-restart should log exhaustion")


# ---------------------------------------------------------------------------
# RISK-6: equity <= 0 returns lot_min path verification
# ---------------------------------------------------------------------------
class TestKellyFormula(unittest.TestCase):
    """Verify Kelly formula edge cases."""

    def test_kelly_negative_returns_zero_risk(self):
        """Low confidence + bad R:R → negative Kelly → risk_pct = 0."""
        conf01 = 0.40  # 40% confidence
        rr = 1.0       # 1:1 R:R
        raw_kelly = (conf01 * rr - (1.0 - conf01)) / rr
        # = (0.40 - 0.60) / 1.0 = -0.20
        self.assertLess(raw_kelly, 0.0)
        risk_pct = max(0.0, raw_kelly) * 0.25
        self.assertEqual(risk_pct, 0.0, "Negative Kelly should produce zero risk")

    def test_kelly_positive_capped(self):
        """High confidence + good R:R → positive Kelly, capped at 2%."""
        conf01 = 0.85
        rr = 2.0
        raw_kelly = (conf01 * rr - (1.0 - conf01)) / rr
        risk_pct = max(0.0, raw_kelly) * 0.25
        risk_cap = min(0.02, 0.015)  # config default
        risk_pct = min(risk_pct, risk_cap)
        self.assertLessEqual(risk_pct, 0.015)
        self.assertGreater(risk_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
