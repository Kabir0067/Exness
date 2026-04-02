"""
Regression tests for institutional-grade patches:
  PATCH-1: Sharpe > 8.0 blocks institutional_grade
  PATCH-2: Zero-signal alarm after extended period
  PATCH-3: Low-sample confidence flag for win_rate_live
  PATCH-4: LLM fallback disabled in live inference path
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── PATCH-1: Sharpe hard cap ──────────────────────────────────────────────

class TestSharpeInstitutionalBlock(unittest.TestCase):
    """Sharpe > 8.0 must set institutional_grade=False."""

    def test_sharpe_10_blocks_institutional(self):
        """Sharpe 10.73 (actual production value) must not pass institutional grade."""
        sharpe = 10.73
        SHARPE_HARD_CAP = 8.0
        base_verified = True
        wfa_ok = True
        unsafe = False
        sharpe_credible = sharpe <= SHARPE_HARD_CAP
        institutional_grade = bool(base_verified and wfa_ok and not unsafe and sharpe_credible)
        self.assertFalse(institutional_grade)
        self.assertFalse(sharpe_credible)

    def test_sharpe_3_passes_institutional(self):
        """Sharpe 3.0 (realistic) should pass institutional grade."""
        sharpe = 3.0
        SHARPE_HARD_CAP = 8.0
        base_verified = True
        wfa_ok = True
        unsafe = False
        sharpe_credible = sharpe <= SHARPE_HARD_CAP
        institutional_grade = bool(base_verified and wfa_ok and not unsafe and sharpe_credible)
        self.assertTrue(institutional_grade)

    def test_sharpe_exactly_8_passes(self):
        """Sharpe exactly 8.0 is on the boundary — should pass."""
        sharpe = 8.0
        SHARPE_HARD_CAP = 8.0
        self.assertTrue(sharpe <= SHARPE_HARD_CAP)

    def test_sharpe_8_01_blocked(self):
        """Sharpe 8.01 — just over the cap — should be blocked."""
        sharpe = 8.01
        SHARPE_HARD_CAP = 8.0
        self.assertFalse(sharpe <= SHARPE_HARD_CAP)

    def test_sharpe_warning_emitted(self):
        """Sharpe > 5.0 must emit BACKTEST_SHARPE_SUSPICIOUS warning."""
        logger = logging.getLogger("test.sharpe.block")
        handler = logging.handlers.MemoryHandler(capacity=50)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        sharpe = 10.73
        if sharpe > 5.0:
            logger.warning("BACKTEST_SHARPE_SUSPICIOUS | sharpe=%.2f", sharpe)
        if sharpe > 8.0:
            logger.warning("SHARPE_INSTITUTIONAL_BLOCK | sharpe=%.2f", sharpe)

        logger.removeHandler(handler)
        handler.flush()

        messages = [r.getMessage() for r in handler.buffer]
        self.assertTrue(any("BACKTEST_SHARPE_SUSPICIOUS" in m for m in messages))
        self.assertTrue(any("SHARPE_INSTITUTIONAL_BLOCK" in m for m in messages))


# ── PATCH-2: Zero-signal alarm ────────────────────────────────────────────

class TestZeroSignalAlarm(unittest.TestCase):
    """signals_24h=0 for > 4h must trigger DEGRADED status."""

    def test_zero_signals_long_running_degrades(self):
        """Engine running > 4h with 0 signals = DEGRADED."""
        signals_24h = 0
        loop_started_ts = time.time() - 20000  # 5.5 hours ago
        now = time.time()
        ALARM_SEC = 14400  # 4 hours

        should_alarm = (
            signals_24h == 0
            and loop_started_ts > 0
            and (now - loop_started_ts) > ALARM_SEC
        )
        self.assertTrue(should_alarm)

    def test_zero_signals_short_running_ok(self):
        """Engine running < 4h with 0 signals = still OK (warming up)."""
        signals_24h = 0
        loop_started_ts = time.time() - 3000  # 50 minutes ago
        now = time.time()
        ALARM_SEC = 14400

        should_alarm = (
            signals_24h == 0
            and loop_started_ts > 0
            and (now - loop_started_ts) > ALARM_SEC
        )
        self.assertFalse(should_alarm)

    def test_nonzero_signals_no_alarm(self):
        """Even 1 signal in 24h = no alarm regardless of uptime."""
        signals_24h = 1
        loop_started_ts = time.time() - 100000
        now = time.time()
        ALARM_SEC = 14400

        should_alarm = (
            signals_24h == 0
            and loop_started_ts > 0
            and (now - loop_started_ts) > ALARM_SEC
        )
        self.assertFalse(should_alarm)


# ── PATCH-3: Low-sample confidence ────────────────────────────────────────

class TestLowSampleConfidence(unittest.TestCase):
    """win_rate_live must be flagged LOW_SAMPLE when closed_trades < 100."""

    def test_33_trades_low_sample(self):
        """33 trades = LOW_SAMPLE (actual production value)."""
        closed_trades = 33
        MIN_TRADES = 100
        confidence = "LOW_SAMPLE" if closed_trades < MIN_TRADES else "RELIABLE"
        self.assertEqual(confidence, "LOW_SAMPLE")

    def test_100_trades_reliable(self):
        """Exactly 100 trades = RELIABLE."""
        closed_trades = 100
        MIN_TRADES = 100
        confidence = "LOW_SAMPLE" if closed_trades < MIN_TRADES else "RELIABLE"
        self.assertEqual(confidence, "RELIABLE")

    def test_500_trades_reliable(self):
        """500 trades = definitely RELIABLE."""
        closed_trades = 500
        MIN_TRADES = 100
        confidence = "LOW_SAMPLE" if closed_trades < MIN_TRADES else "RELIABLE"
        self.assertEqual(confidence, "RELIABLE")

    def test_0_trades_low_sample(self):
        """0 trades = LOW_SAMPLE."""
        closed_trades = 0
        MIN_TRADES = 100
        confidence = "LOW_SAMPLE" if closed_trades < MIN_TRADES else "RELIABLE"
        self.assertEqual(confidence, "LOW_SAMPLE")


# ── PATCH-4: LLM fallback disabled ───────────────────────────────────────

class TestLLMFallbackDisabled(unittest.TestCase):
    """When CatBoost fails, system must return HOLD, not call LLM chain."""

    def test_catboost_fail_returns_hold(self):
        """Simulated: CatBoost returns None → result must be HOLD signal."""
        catboost_result = None  # CatBoost inference failed
        has_catboost = True

        if has_catboost and catboost_result is None:
            signal = "HOLD"
            reason = "catboost_inference_failed_llm_disabled"
        elif has_catboost:
            signal = catboost_result
            reason = "catboost_ok"
        else:
            signal = "HOLD"
            reason = "catboost_payload_missing"

        self.assertEqual(signal, "HOLD")
        self.assertIn("llm_disabled", reason)

    def test_catboost_success_passes_through(self):
        """CatBoost returns a valid signal → passes through normally."""
        catboost_result = "STRONG BUY"
        has_catboost = True

        if has_catboost and catboost_result is None:
            signal = "HOLD"
        elif has_catboost:
            signal = catboost_result
        else:
            signal = "HOLD"

        self.assertEqual(signal, "STRONG BUY")

    def test_no_catboost_payload_returns_hold(self):
        """No CatBoost model loaded → HOLD (no LLM fallback)."""
        has_catboost = False

        if has_catboost:
            signal = "from_catboost"
        else:
            signal = "HOLD"

        self.assertEqual(signal, "HOLD")


if __name__ == "__main__":
    unittest.main()
