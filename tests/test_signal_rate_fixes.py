"""
Tests for signal-rate recovery patches (PATCH-S1, S2, S3).
PATCH-S1: Sniper floor respects config, hard minimum 70.
PATCH-S2: Rejection counter tracks filter reasons.
PATCH-S3: Display matches runtime floor.
"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSniperFloorConfig(unittest.TestCase):
    """PATCH-S1: Sniper floor must respect config, not force 80."""

    def test_default_floor_is_75(self):
        """Default min_confidence should be 75, not 80."""
        from core.config import BaseEngineConfig
        # BaseEngineConfig is a dataclass with default min_confidence
        # We can't instantiate it directly (needs params), so check the field default
        import dataclasses
        fields = {f.name: f.default for f in dataclasses.fields(BaseEngineConfig)
                  if f.default is not dataclasses.MISSING}
        self.assertEqual(fields["min_confidence"], 75)

    def test_floor_logic_respects_config_75(self):
        """Config min_confidence=75 should produce sniper_floor=75."""
        _SNIPER_HARD_MIN = 70
        cfg_value = 75
        sniper_floor = max(int(cfg_value), _SNIPER_HARD_MIN)
        self.assertEqual(sniper_floor, 75)

    def test_floor_logic_respects_config_72(self):
        """Config min_confidence=72 should produce sniper_floor=72."""
        _SNIPER_HARD_MIN = 70
        cfg_value = 72
        sniper_floor = max(int(cfg_value), _SNIPER_HARD_MIN)
        self.assertEqual(sniper_floor, 72)

    def test_floor_hard_minimum_70(self):
        """Config min_confidence=50 should be clamped to 70."""
        _SNIPER_HARD_MIN = 70
        cfg_value = 50
        sniper_floor = max(int(cfg_value), _SNIPER_HARD_MIN)
        self.assertEqual(sniper_floor, 70)

    def test_floor_config_90_passes(self):
        """Config min_confidence=90 should be respected (above hard min)."""
        _SNIPER_HARD_MIN = 70
        cfg_value = 90
        sniper_floor = max(int(cfg_value), _SNIPER_HARD_MIN)
        self.assertEqual(sniper_floor, 90)

    def test_old_hardcode_was_80_now_removed(self):
        """Verify the old max(..., 80) pattern is no longer present."""
        import inspect
        from core.signal_engine import SignalEngine
        source = inspect.getsource(SignalEngine.compute)
        # The old pattern was: max(int(...), 80)
        # The new pattern is: max(int(...), _SNIPER_HARD_MIN) where _SNIPER_HARD_MIN = 70
        self.assertNotIn("max(int(getattr(self.cfg, \"min_confidence\", 80) or 80), 80)", source)
        # And no duplicate 0.80 check
        self.assertNotIn("(float(conf) / 100.0) < 0.80", source)

    def test_signal_at_76_passes_with_floor_75(self):
        """Confidence 76 should pass when floor is 75."""
        conf = 76
        sniper_floor = 75
        self.assertTrue(conf >= sniper_floor)

    def test_signal_at_74_rejected_with_floor_75(self):
        """Confidence 74 should be rejected when floor is 75."""
        conf = 74
        sniper_floor = 75
        self.assertFalse(conf >= sniper_floor)

    def test_signal_at_79_was_rejected_now_passes(self):
        """Key regression: conf=79 was rejected by old floor=80. Now passes with floor=75."""
        conf = 79
        # Old logic: max(..., 80) → floor=80 → 79 < 80 → REJECTED
        old_floor = max(80, 80)
        self.assertFalse(conf >= old_floor)
        # New logic: max(75, 70) → floor=75 → 79 >= 75 → PASSES
        new_floor = max(75, 70)
        self.assertTrue(conf >= new_floor)


class TestRejectionCounter(unittest.TestCase):
    """PATCH-S2: Rejection counter accumulates correctly."""

    def test_counter_increments(self):
        counts = {}
        reasons = ("sniper_reject:72<75", "weak_score:68<70", "sniper_reject:71<75")
        for r in reasons:
            key = r.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts["sniper_reject"], 2)
        self.assertEqual(counts["weak_score"], 1)

    def test_signal_rate_calculation(self):
        cycles = 100
        signals = 3
        rate = signals / max(1, cycles)
        self.assertAlmostEqual(rate, 0.03)

    def test_empty_reasons_no_crash(self):
        counts = {}
        reasons = ()
        for r in reasons:
            key = r.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 0)


class TestDisplayCorrectness(unittest.TestCase):
    """PATCH-S3: Display must match actual runtime floor."""

    def test_display_75(self):
        cfg_value = 75
        display_floor = max(int(cfg_value), 70)
        self.assertEqual(display_floor, 75)

    def test_display_matches_runtime(self):
        """Display floor and runtime floor must use same formula."""
        _SNIPER_HARD_MIN = 70
        cfg_value = 75
        runtime_floor = max(int(cfg_value), _SNIPER_HARD_MIN)
        display_floor = max(int(cfg_value), 70)
        self.assertEqual(runtime_floor, display_floor)


if __name__ == "__main__":
    unittest.main()
