# core/risk_engine.py — Unified RiskManager for all assets.
# Merges StrategiesBtc/_btc_risk/risk_manager.py (2400 lines)
#    and StrategiesXau/_xau_risk/risk_manager.py (2191 lines)
# into a single parameterized class (~2000 lines, zero duplication).
from __future__ import annotations

import atexit
import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .config import BaseEngineConfig, BaseSymbolParams
from .models import AccountCache, KillSwitchState, SignalThrottle
from .utils import (
    _atr_fallback,
    _atomic_write_text,
    _is_finite,
    _side_norm,
    _utcnow,
    atr_take_profit,
    breakeven_price,
    clamp01,
    fractal_volatility_stop,
    kaufman_efficiency_ratio,
    percentile_rank,
    relative_volatility_index,
    volatility_trailing_stop,
)

import logging

log = logging.getLogger("core.risk_engine")


class RiskManager:
    """
    Unified, asset-agnostic RiskManager.

    Handles:
      - Phase escalation (A → B → C) with monotonic progression
      - Kill switch (ACTIVE/COOLING/KILLED) state machine
      - Dynamic position sizing (equity × risk% × confidence × phase × DD factor)
      - ATR-based SL/TP with structural and microzone overrides
      - Trailing stop (breakeven + ATR-trail)
      - Execution quality metrics (latency, slippage, spread)
      - Daily state reset, hard/soft stops
      - Maintenance blackout windows (parameterized per asset)
      - Execution metric tracking and CSV persistence

    All asset-specific behavior is driven by config:
      - BTC: 24/7, wider spreads, crypto-specific round numbers
      - XAU: 24/5 with rollover blackout, tighter spreads

    Usage:
        from core.config import BTCEngineConfig, BTCSymbolParams
        cfg = BTCEngineConfig(...)
        rm = RiskManager(cfg, cfg.symbol_params)
    """

    _REQUIRED_CFG_FIELDS = (
        "daily_target_pct",
        "max_daily_loss_pct",
        "protect_drawdown_from_peak_pct",
        "max_drawdown",
        "max_risk_per_trade",
        "analysis_cooldown_sec",
    )

    def __init__(self, cfg: BaseEngineConfig, sp: BaseSymbolParams) -> None:
        self.cfg = cfg
        self.sp = sp
        self._validate_cfg()

        # ── internal state ──
        self.phase: str = "A"
        self._phase_reason: str = "init"
        self._soft_stop: bool = False
        self._hard_stop: bool = False
        self._hard_stop_reason: str = ""
        self._analysis_blocked_until: float = 0.0
        self._position_open: bool = False
        self._last_date: str = ""

        # ── kill switch ──
        self._ks = KillSwitchState()
        self._throttle = SignalThrottle()

        # ── account cache ──
        self._acc = AccountCache()
        self._bot_acc = AccountCache()
        self._bot_balance_base: float = 0.0
        self._peak_equity: float = 0.0
        self._daily_peak: float = 0.0

        # ── execution metrics ──
        self._exec_latencies: List[float] = []
        self._exec_slippages: List[float] = []
        self._exec_spreads: List[float] = []
        self._exec_hist: List[dict] = []
        self._exec_breaker_active: bool = False
        self._exec_breaker_until: float = 0.0
        self._exec_csv_path = Path(f"logs/exec_metrics_{sp.base}.csv")
        self._exec_csv_flush_interval = 300.0
        self._exec_csv_last_flush: float = 0.0

        # ── latency cooldown ──
        self._latency_cd_until: float = 0.0

        # ── trade tracking ──
        self._trades_today: List[float] = []
        self._signals_today: int = 0
        self._daily_pnl: float = 0.0
        self._daily_realized_pnl: float = 0.0
        self._open_lots: float = 0.0

        # ── bars / ATR cache ──
        self._cached_atr: float = 0.0
        self._cached_atr_ts: float = 0.0

        # ── shutdown hook ──
        def _shutdown_wrapper():
            try:
                self._flush_exec_csv(force=True)
            except Exception:
                pass
        atexit.register(_shutdown_wrapper)

        # init daily
        self._reset_daily_state()

        # load persisted state (optional)
        try:
            self.load_state()
        except Exception:
            pass

        log.info(
            "RiskManager(%s) initialized | phase=%s max_risk=%.3f",
            sp.base, self.phase, cfg.max_risk_per_trade,
        )

    # ─── Validation ──────────────────────────────────────────────────

    def _validate_cfg(self) -> None:
        for f in self._REQUIRED_CFG_FIELDS:
            if not hasattr(self.cfg, f):
                raise ValueError(f"Config missing: {f}")
            v = getattr(self.cfg, f)
            if not isinstance(v, (int, float)):
                raise TypeError(f"Config field '{f}' must be numeric, got {type(v)}")

    # ─── Time ────────────────────────────────────────────────────────

    @staticmethod
    def _utc_date() -> str:
        return _utcnow().strftime("%Y-%m-%d")

    @staticmethod
    def _utc_now() -> datetime:
        return _utcnow()

    def _seconds_until_next_utc_day(self) -> float:
        now = _utcnow()
        eod = now.replace(hour=23, minute=59, second=59)
        return max(0.0, (eod - now).total_seconds())

    def _is_xau_overlap(self) -> bool:
        """London/NY overlap window (UTC hours) for XAU trailing-stop tightening."""
        if "XAU" not in str(self.sp.base).upper():
            return False
        try:
            start = int(getattr(self.cfg, "xau_overlap_start_hour_utc", 12) or 12)
            end = int(getattr(self.cfg, "xau_overlap_end_hour_utc", 16) or 16)
            h = _utcnow().hour
            if start <= end:
                return start <= h < end
            return h >= start or h < end
        except Exception:
            return False

    # ─── Phase management ────────────────────────────────────────────

    @staticmethod
    def _phase_rank(p: str) -> int:
        return {"A": 0, "B": 1, "C": 2}.get(str(p).upper(), 0)

    def _set_phase(self, phase: str, reason: str) -> None:
        """Monotonic risk escalation within the day: A → B → C only."""
        if self._phase_rank(phase) <= self._phase_rank(self.phase):
            return
        old = self.phase
        self.phase = phase
        self._phase_reason = reason
        log.info("PHASE %s → %s | reason=%s | asset=%s", old, phase, reason, self.sp.base)

    def _enter_soft_stop(self, reason: str) -> None:
        """Trade-block until next UTC day (engine keeps running)."""
        if self._soft_stop:
            return
        self._soft_stop = True
        self._set_phase("C", reason)
        log.warning("SOFT_STOP | %s | %s", self.sp.base, reason)

    def _enter_hard_stop(self, reason: str) -> None:
        """Hard stop until next UTC day (engine-level)."""
        if self._hard_stop:
            return
        self._hard_stop = True
        self._hard_stop_reason = reason
        self._set_phase("C", reason)
        log.critical("HARD_STOP | %s | %s", self.sp.base, reason)

    @property
    def requires_hard_stop(self) -> bool:
        if self._exec_breaker_active and time.time() < self._exec_breaker_until:
            return True
        today = self._utc_date()
        if today != self._last_date:
            self._reset_daily_state()
            return False
        return self._hard_stop

    @property
    def hard_stop_reason(self) -> str:
        return self._hard_stop_reason

    @property
    def phase_reason(self) -> str:
        return self._phase_reason

    def block_analysis(self, seconds: float) -> None:
        self._analysis_blocked_until = time.time() + seconds

    def can_analyze(self) -> bool:
        return time.time() >= self._analysis_blocked_until

    def on_position_opened(self) -> None:
        self._position_open = True

    def on_all_flat(self) -> None:
        self._position_open = False
        # Refresh account on flat
        try:
            self._account_snapshot()
        except Exception:
            pass

    # ─── Daily reset ─────────────────────────────────────────────────

    def _reset_daily_state(self) -> None:
        today = self._utc_date()
        if today == self._last_date:
            return
        self._last_date = today
        self.phase = "A"
        self._phase_reason = "daily_reset"
        self._soft_stop = False
        self._hard_stop = False
        self._hard_stop_reason = ""
        self._ks.status = "ACTIVE"
        self._ks.cooling_until_ts = 0.0
        self._throttle.daily_count = 0
        self._throttle.hour_window_count = 0
        self._throttle.hour_window_start_ts = time.time()
        self._trades_today.clear()
        self._signals_today = 0
        self._daily_pnl = 0.0
        self._daily_realized_pnl = 0.0
        self._exec_latencies.clear()
        self._exec_slippages.clear()
        self._exec_spreads.clear()
        self._exec_breaker_active = False

        try:
            self._account_snapshot()
            self._peak_equity = self._acc.equity
            self._daily_peak = self._acc.equity
        except Exception:
            pass

        self._reset_bot_balance_base()
        log.info(
            "DAILY_RESET | %s | equity=%.2f peak=%.2f",
            self.sp.base, self._acc.equity, self._peak_equity,
        )

    def _reset_bot_balance_base(self) -> None:
        try:
            if mt5 is not None:
                info = mt5.account_info()
                if info:
                    self._bot_balance_base = float(info.balance or 0.0)
        except Exception:
            pass

    # ─── Account snapshots ───────────────────────────────────────────

    def _ensure_ready(self) -> None:
        today = self._utc_date()
        if today != self._last_date:
            self._reset_daily_state()
        now = time.time()
        if now - self._acc.ts > 30.0:
            self._account_snapshot()

    def _minutes_utc(self) -> int:
        now = _utcnow()
        return now.hour * 60 + now.minute

    def _in_maintenance_blackout(self) -> bool:
        """
        Parameterized maintenance blackout.
        XAU: rollover window (e.g. 23:55-00:05 UTC)
        BTC: no blackout (24/7)
        """
        if self.sp.is_24_7:
            return False
        m = self._minutes_utc()
        start = self.sp.rollover_blackout_start
        end = self.sp.rollover_blackout_end
        if start > end:
            return m >= start or m <= end
        return start <= m <= end

    def _trade_mode_state(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "soft_stop": self._soft_stop,
            "hard_stop": self._hard_stop,
            "ks_status": self._ks.status,
            "signals_today": self._signals_today,
            "daily_pnl": round(self._daily_pnl, 2),
        }

    # ─── Market hours ────────────────────────────────────────────────

    def market_open_24_5(self) -> bool:
        """Check if forex market is open (Mon 00:00 to Fri 23:59 UTC)."""
        now = _utcnow()
        wd = now.weekday()
        if wd >= 5:
            return False
        m = self._minutes_utc()
        return self.sp.market_start_minutes <= m <= self.sp.market_end_minutes

    def market_open(self) -> bool:
        """Check if market is open for this asset."""
        if self.sp.is_24_7:
            return True
        return self.market_open_24_5()

    def rollover_blackout(self) -> bool:
        if self.sp.is_24_7:
            return False
        return self._in_maintenance_blackout()

    # ─── Quote / data hooks ──────────────────────────────────────────

    def on_quote(self, bid: float, ask: float) -> None:
        spread = ask - bid
        if _is_finite(spread) and spread > 0:
            self._exec_spreads.append(spread)
            if len(self._exec_spreads) > 500:
                self._exec_spreads = self._exec_spreads[-200:]

    def on_execution_result(self, result: Any) -> None:
        if hasattr(result, "ok") and result.ok:
            self.on_position_opened()

    def on_reconcile_positions(self, positions: Any) -> None:
        if positions is None:
            self._position_open = False
            return
        try:
            cnt = len(positions) if hasattr(positions, "__len__") else 0
            self._position_open = cnt > 0
        except Exception:
            pass

    def register_signal_emitted(self) -> None:
        self._signals_today += 1
        allowed = self._throttle.register(max_per_hour=20)
        if not allowed:
            log.warning("SIGNAL_THROTTLED | %s | hour_count=%d", self.sp.base, self._throttle.hour_window_count)

    # ─── Account helpers ─────────────────────────────────────────────

    def _account_snapshot(self) -> None:
        if mt5 is None:
            return
        try:
            info = mt5.account_info()
            if info is None:
                return
            now = time.time()
            b = float(info.balance or 0.0)
            e = float(info.equity or 0.0)
            mf = float(info.margin_free or 0.0)
            if b > 0:
                self._acc.ts = now
                self._acc.balance = b
                self._acc.equity = e
                self._acc.margin_free = mf
                if e > self._peak_equity:
                    self._peak_equity = e
                if e > self._daily_peak:
                    self._daily_peak = e
        except Exception as exc:
            log.error("_account_snapshot: %s", exc)

    def _bot_account_snapshot(self) -> None:
        if mt5 is None:
            return
        try:
            info = mt5.account_info()
            if info is None:
                return
            now = time.time()
            b = float(info.balance or 0.0)
            e = float(info.equity or 0.0)
            mf = float(info.margin_free or 0.0)
            if b > 0:
                self._bot_acc.ts = now
                self._bot_acc.balance = b
                self._bot_acc.equity = e
                self._bot_acc.margin_free = mf
        except Exception as exc:
            log.error("_bot_account_snapshot: %s", exc)

    def _get_balance(self) -> float:
        self._ensure_ready()
        return self._acc.balance

    def _get_equity(self) -> float:
        self._ensure_ready()
        return self._acc.equity

    def _get_margin_free(self) -> float:
        self._ensure_ready()
        return self._acc.margin_free

    # ─── Trade history ───────────────────────────────────────────────

    def _recent_closed_trade_profits(self, limit: int) -> List[float]:
        if mt5 is None:
            return list(self._trades_today)
        try:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=1)
            deals = mt5.history_deals_get(start, now)
            if deals is None:
                return list(self._trades_today)
            profits = [
                float(d.profit)
                for d in deals
                if d.entry == 1 and d.symbol == self.sp.symbol
            ]
            return profits[-limit:] if len(profits) > limit else profits
        except Exception:
            return list(self._trades_today)

    # ─── Kill switch ─────────────────────────────────────────────────

    def _refresh_kill_switch(self) -> None:
        profits = self._recent_closed_trade_profits(20)
        self._ks.update(
            profits,
            time.time(),
            self.cfg.kill_min_trades,
            self.cfg.kill_expectancy,
            self.cfg.kill_winrate,
            self.cfg.cooling_expectancy,
            self.cfg.cooling_sec,
        )
        if self._ks.status == "KILLED":
            # Suppress hard-stop escalation while the market is closed.
            # This avoids noisy weekend hard-stop logs for non-24/7 assets (e.g. XAU).
            if not self.market_open():
                return
            self._enter_hard_stop(
                f"kill_switch: exp={self._ks.last_expectancy:.3f} wr={self._ks.last_winrate:.2f}"
            )

    def strategy_status(self) -> Dict[str, Any]:
        return {"kill_switch": self._ks.status, "phase": self.phase}

    # ─── Phase evaluation ────────────────────────────────────────────

    def evaluate_account_state(self) -> None:
        """
        Deterministic A/B/C logic + real C blocking + peak drawdown protect.
        """
        self._ensure_ready()
        eq = self._acc.equity
        bal = self._acc.balance

        if eq <= 0 or bal <= 0:
            return

        # Daily PnL
        if self._bot_balance_base > 0:
            self._daily_pnl = eq - self._bot_balance_base
            daily_pnl_pct = self._daily_pnl / self._bot_balance_base
        else:
            daily_pnl_pct = 0.0

        # Update peak
        if eq > self._peak_equity:
            self._peak_equity = eq
        if eq > self._daily_peak:
            self._daily_peak = eq

        # Phase escalation
        if daily_pnl_pct >= self.cfg.phase_a_target:
            self._set_phase("B", f"daily_pnl={daily_pnl_pct:+.3f}")
        if daily_pnl_pct >= self.cfg.phase_b_target:
            self._set_phase("C", f"daily_pnl={daily_pnl_pct:+.3f}")

        # Drawdown checks
        if self._peak_equity > 0:
            dd_from_peak = (self._peak_equity - eq) / self._peak_equity
            if dd_from_peak >= self.cfg.max_drawdown:
                self._enter_hard_stop(f"MAX_DD={dd_from_peak:.3f}")
                return
            if dd_from_peak >= self.cfg.protect_drawdown_from_peak_pct:
                self._enter_soft_stop(f"DD_PROTECT={dd_from_peak:.3f}")

        # Daily loss limit
        if daily_pnl_pct <= -self.cfg.max_daily_loss_pct:
            self._enter_hard_stop(f"MAX_DAILY_LOSS={daily_pnl_pct:+.3f}")
            return

        # Kill switch refresh
        self._refresh_kill_switch()

    def update_phase(self) -> None:
        self.evaluate_account_state()

    # ─── Latency management ──────────────────────────────────────────

    def latency_cooldown(self) -> bool:
        now = time.time()
        if now < self._latency_cd_until:
            return True
        if self._exec_breaker_active and now < self._exec_breaker_until:
            return True
        return False

    def register_latency_violation(self) -> None:
        self._latency_cd_until = time.time() + 5.0

    # ─── ATR percentile ──────────────────────────────────────────────

    def atr_percentile(self, dfp: pd.DataFrame, lookback: int = 100) -> float:
        try:
            if dfp is None or len(dfp) < 20:
                return 50.0
            h = dfp["high"].values if "high" in dfp else dfp["High"].values
            l = dfp["low"].values if "low" in dfp else dfp["Low"].values
            c = dfp["close"].values if "close" in dfp else dfp["Close"].values
            atr_arr = _atr_fallback(h, l, c, 14)
            n = min(lookback, len(atr_arr) - 1)
            if n < 5:
                return 50.0
            return percentile_rank(atr_arr[-n:], float(atr_arr[-1]))
        except Exception:
            return 50.0

    # ─── Guard decision ──────────────────────────────────────────────

    def guard_decision(
        self,
        *,
        spread_pct: float,
        tick_ok: bool,
        tick_reason: str,
        ingest_ms: float,
        last_bar_age: float,
        in_session: bool,
        drawdown_exceeded: bool,
        latency_cooldown: bool,
        tz: Any = None,
    ) -> Tuple[bool, List[str]]:
        """
        Pre-signal gate checks. Returns (allowed, reasons).
        """
        reasons: List[str] = []

        if self.requires_hard_stop:
            reasons.append(f"HARD_STOP:{self._hard_stop_reason}")
            return False, reasons

        if self._soft_stop:
            reasons.append("SOFT_STOP")
            return False, reasons

        if self._ks.status != "ACTIVE":
            reasons.append(f"KILL_SWITCH:{self._ks.status}")
            return False, reasons

        if not self.market_open():
            reasons.append("MARKET_CLOSED")
            return False, reasons

        if self._in_maintenance_blackout():
            reasons.append("MAINTENANCE_BLACKOUT")
            return False, reasons

        if not tick_ok:
            reasons.append(f"TICK_REJECT:{tick_reason}")
            return False, reasons

        if spread_pct > self.sp.spread_limit_pct * 2:
            reasons.append(f"SPREAD_WIDE:{spread_pct:.5f}")
            return False, reasons

        if drawdown_exceeded:
            reasons.append("DD_EXCEEDED")
            return False, reasons

        if latency_cooldown:
            reasons.append("LATENCY_CD")
            return False, reasons

        if self._exec_breaker_active and time.time() < self._exec_breaker_until:
            reasons.append("EXEC_BREAKER")
            return False, reasons

        if not in_session and not (self.cfg.ignore_sessions or self.sp.is_24_7):
            reasons.append("OUT_OF_SESSION")
            return False, reasons

        return True, reasons

    # ─── Can-trade checks ────────────────────────────────────────────

    def can_trade(self, confidence: float, signal_type: str) -> Tuple[bool, str]:
        self._ensure_ready()

        if self.requires_hard_stop:
            return False, f"hard_stop:{self._hard_stop_reason}"

        if self._soft_stop:
            return False, "soft_stop"

        if self._ks.status != "ACTIVE":
            return False, f"kill_switch:{self._ks.status}"

        if self.phase == "C":
            return False, "phase_C_block"

        if not self.market_open():
            return False, "market_closed"

        if self._in_maintenance_blackout():
            return False, "maintenance"

        # Refresh account state
        self.evaluate_account_state()

        if self.requires_hard_stop:
            return False, f"hard_stop_after_eval:{self._hard_stop_reason}"

        return True, ""

    def can_emit_signal(self, confidence: int, tz: Any = None) -> Tuple[bool, str]:
        if confidence < self.cfg.min_confidence:
            return False, f"low_conf:{confidence}"

        if self.requires_hard_stop:
            return False, f"hard_stop:{self._hard_stop_reason}"

        if self._ks.status == "KILLED":
            return False, "killed"

        if self._ks.status == "COOLING":
            return False, f"cooling:{self._ks.cooling_until_ts - time.time():.0f}s"

        if not self.market_open():
            return False, "market_closed"

        if self._in_maintenance_blackout():
            return False, "maintenance"

        return True, ""

    # ─── SL/TP calculation ───────────────────────────────────────────

    def _apply_broker_constraints(
        self, side: str, entry: float, sl: float, tp: float,
    ) -> Tuple[float, float, float]:
        """Ensure SL/TP respect broker minimum distance constraints."""
        if mt5 is None:
            return entry, sl, tp
        try:
            info = mt5.symbol_info(self.sp.symbol)
            if info is None:
                return entry, sl, tp
            stops_level = float(info.trade_stops_level or 0) * float(info.point or 0.00001)
            if _side_norm(side) == "Buy":
                sl = min(sl, entry - stops_level)
                tp = max(tp, entry + stops_level)
            else:
                sl = max(sl, entry + stops_level)
                tp = min(tp, entry - stops_level)
        except Exception:
            pass
        return entry, sl, tp

    @staticmethod
    def _rr(entry: float, sl: float, tp: float) -> float:
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return 0.0
        return abs(tp - entry) / sl_dist

    # ─── Position sizing ─────────────────────────────────────────────

    def calculate_position_size(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        *,
        adapt: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Smart lot sizing (ATR-driven):
          Lot = (Equity * Risk%) / (ATR * StopLoss_Multiplier)

        ATR is normalized by contract size and broker volume constraints.
        """
        self._ensure_ready()
        eq = float(self._acc.equity or 0.0)
        if eq <= 0:
            return 0.01

        # Fractional Kelly risk percent (SAFE institutional sizing)
        conf_raw = float(confidence or 0.0)
        if conf_raw > 1.0:
            conf_raw = conf_raw / 100.0
        conf01 = clamp01(conf_raw)
        rr = self._rr(entry_price, stop_loss, take_profit)
        raw_kelly = 0.0
        if rr > 0:
            raw_kelly = (conf01 * rr - (1.0 - conf01)) / rr

        risk_pct = max(0.0, raw_kelly) * 0.25
        risk_cap = 0.02
        cfg_cap = float(self.cfg.max_risk_per_trade or 0.0)
        if cfg_cap > 0:
            risk_cap = min(risk_cap, cfg_cap)
        risk_pct = min(risk_pct, risk_cap)
        if risk_pct <= 0.0:
            return 0.0

        sl_mult = max(float(self.cfg.atr_sl_multiplier or 0.0), 0.01)
        atr = 0.0
        try:
            if adapt:
                atr = float(adapt.get("atr", 0.0) or 0.0)
                if atr <= 0:
                    atr_pct = float(adapt.get("atr_pct", 0.0) or 0.0)
                    if atr_pct > 0 and entry_price > 0:
                        atr = atr_pct * float(entry_price)
        except Exception:
            atr = 0.0

        if atr <= 0 and entry_price > 0 and stop_loss > 0:
            # Recover ATR from the computed stop distance when direct ATR is missing.
            atr = abs(float(entry_price) - float(stop_loss)) / sl_mult

        if atr <= 0:
            return 0.01

        # Preserve daily phase and drawdown controls while keeping ATR-driven sizing.
        phase_mult = {"A": 1.0, "B": 0.75, "C": 0.5}.get(str(self.phase).upper(), 1.0)
        dd_mult = 1.0
        if self._peak_equity > 0:
            dd_from_peak = (self._peak_equity - eq) / self._peak_equity
            if dd_from_peak >= float(self.cfg.protect_drawdown_from_peak_pct or 0.0):
                dd_mult = 0.5

        risk_usd = eq * risk_pct * phase_mult * dd_mult

        if risk_usd <= 0:
            return 0.01

        # Core formula requested by user:
        # lot_raw = (Equity * Risk%) / (ATR * SL_mult)
        lot_raw = risk_usd / (atr * sl_mult)

        contract_size = float(getattr(self.sp, "contract_size", 1.0) or 1.0)
        lot_step = float(getattr(self.sp, "lot_step", 0.01) or 0.01)
        lot_min = float(getattr(self.sp, "lot_min", 0.01) or 0.01)
        lot_max = float(getattr(self.sp, "lot_max", 100.0) or 100.0)

        try:
            if mt5 is not None:
                info = mt5.symbol_info(self.sp.symbol)
                if info:
                    contract_size = float(info.trade_contract_size or contract_size)
                    lot_step = float(info.volume_step or lot_step)
                    lot_min = float(info.volume_min or lot_min)
                    lot_max = float(info.volume_max or lot_max)
        except Exception as exc:
            log.error("position_size error: %s", exc)

        # Convert cash-units to broker lot-units.
        if contract_size > 0:
            lot_raw = lot_raw / contract_size

        if lot_step > 0:
            lot_raw = math.floor(lot_raw / lot_step) * lot_step

        if not _is_finite(lot_raw):
            return lot_min
        return max(lot_min, min(lot_max, round(lot_raw, 8)))

    # ─── SL/TP calculation methods ───────────────────────────────────

    def _fallback_atr_sl_tp(
        self, side: str, entry: float, adapt: Dict[str, Any],
    ) -> Tuple[float, float]:
        """ATR-based SL/TP using config multipliers."""
        atr = float(adapt.get("atr", 0.0))
        if atr <= 0:
            atr = float(adapt.get("atr_pct", 0.001)) * entry
        if atr <= 0:
            atr = max(entry * 0.001, 0.00001)

        atr_pct = float(adapt.get("atr_pct", 0.0) or 0.0)
        if atr_pct <= 0 and entry > 0:
            atr_pct = atr / entry

        ker = float(adapt.get("ker", 0.0) or 0.0)
        rvi = float(adapt.get("rvi", 0.0) or 0.0)
        sl = fractal_volatility_stop(
            entry,
            atr,
            side,
            base_mult=max(float(self.cfg.atr_sl_multiplier or 0.0), 0.01),
            rvi=rvi,
            ker=ker,
            rvi_weight=float(getattr(self.cfg, "rvi_weight", 1.0) or 1.0),
            chaos_mult=float(getattr(self.cfg, "ker_chaos_mult", 1.0) or 1.0),
        )
        confidence = clamp01(float(adapt.get("confidence", 0.5)))
        regime = str(adapt.get("regime", "normal")).lower()

        tp_min = float(self.cfg.atr_tp_min_multiplier or 1.0)
        tp_max = float(self.cfg.atr_tp_max_multiplier or tp_min)
        hi_vol_ref = float(getattr(self.sp, "atr_rel_hi", 0.008) or 0.008)
        lo_vol_ref = max(0.0005, hi_vol_ref * 0.35)

        # Sniper TP adaptation:
        # - low volatility: lock profit earlier
        # - high volatility: let winners run
        if atr_pct <= lo_vol_ref or regime in ("compressed", "quiet"):
            tp_min *= 0.85
            tp_max *= 0.88
        elif atr_pct >= hi_vol_ref or regime in ("explosive", "volatile"):
            tp_min *= 1.10
            tp_max *= 1.30

        tp_min = max(0.8, tp_min)
        tp_max = max(tp_min, tp_max)
        tp = atr_take_profit(
            entry, atr, side, confidence,
            min_mult=tp_min,
            max_mult=tp_max,
        )
        return sl, tp

    def _calculate_structure_sl(
        self, side: str, entry: float, df: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """
        Finds the recent Swing Low (Buy) or Swing High (Sell)
        to use as a structural Stop Loss.
        """
        if df is None or len(df) < 5:
            return None
        try:
            lookback = min(20, len(df) - 1)
            recent = df.iloc[-lookback:]
            if _side_norm(side) == "Buy":
                swing = float(recent["low"].min() if "low" in recent else recent["Low"].min())
                if swing < entry:
                    return swing
            else:
                swing = float(recent["high"].max() if "high" in recent else recent["High"].max())
                if swing > entry:
                    return swing
        except Exception:
            pass
        return None

    # ─── Trailing stop ───────────────────────────────────────────────

    def check_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        side: str,
        atr: float,
    ) -> Optional[float]:
        """
        Dynamic trailing:
        1) Fast breakeven lock once trade reaches early progress.
        2) ATR trail that tightens as progress improves.

        Returns new SL if it should be moved, else None.
        """
        if not _is_finite(current_price, entry_price, sl_price, tp_price, atr):
            return None
        if atr <= 0:
            return None

        # 1. Breakeven check
        be = breakeven_price(
            entry_price, tp_price, current_price, side,
            be_trigger_pct=self.cfg.breakeven_trigger_pct,
        )

        # 2. Volatility trailing stop
        trail_mult = float(self.cfg.trailing_stop_atr_mult)
        if self._is_xau_overlap():
            overlap_mult = float(getattr(self.cfg, "xau_overlap_trail_mult", 0.80) or 0.80)
            trail_mult = max(0.10, trail_mult * overlap_mult)
        trail = volatility_trailing_stop(
            entry_price, current_price, atr, side,
            trail_atr_mult=trail_mult,
        )

        # Pick the best (most protective) SL
        if _side_norm(side) == "Buy":
            candidates = [sl_price]
            if be is not None:
                candidates.append(be)
            candidates.append(trail)
            new_sl = max(candidates)
            return new_sl if new_sl > sl_price else None
        else:
            candidates = [sl_price]
            if be is not None:
                candidates.append(be)
            candidates.append(trail)
            new_sl = min(candidates)
            return new_sl if new_sl < sl_price else None

    # ─── Plan order ──────────────────────────────────────────────────

    def plan_order(
        self,
        side: str,
        confidence: float,
        ind: Dict[str, Any],
        adapt: Dict[str, Any],
        *,
        entry: Optional[float] = None,
        ticks: Any = None,
        zones: Any = None,
        tick_volatility: float = 0.0,
        open_positions: int = 0,
        max_positions: int = 0,
        unrealized_pl: float = 0.0,
        allow_when_blocked: bool = False,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Full order planning: entry → SL → TP → lot → validation.

        Returns dict with:
          blocked, reason, side, entry, sl, tp, lot, rr
        """
        result: Dict[str, Any] = {
            "blocked": True, "reason": "init", "side": _side_norm(side),
            "entry": 0.0, "sl": 0.0, "tp": 0.0, "lot": 0.0, "rr": 0.0,
        }

        adapt = dict(adapt or {})
        conf01 = clamp01(confidence)
        adapt["confidence"] = conf01

        if conf01 <= 0.0:
            result["reason"] = "low_confidence:0.00"
            return result

        # ML router strict floor: risk must obey model confidence.
        regime = str(adapt.get("regime", "") or "").lower()
        if regime == "ml_router" and conf01 < 0.75:
            result["reason"] = f"low_confidence_ml:{conf01:.3f}<0.75"
            return result

        # ── Can trade? ──
        if not allow_when_blocked:
            can, reason = self.can_trade(confidence, "signal")
            if not can:
                result["reason"] = reason
                return result

        # ── Max positions ──
        if max_positions > 0 and open_positions >= max_positions:
            result["reason"] = f"max_positions:{open_positions}/{max_positions}"
            return result

        # ── Entry price ──
        if entry is None or entry <= 0:
            try:
                if mt5 is not None:
                    tick = mt5.symbol_info_tick(self.sp.symbol)
                    if tick:
                        entry = float(tick.ask) if _side_norm(side) == "Buy" else float(tick.bid)
            except Exception:
                pass
        if entry is None or entry <= 0:
            result["reason"] = "no_entry_price"
            return result

        # Enrich adaptive params with KER/RVI from primary bars if missing
        if df is not None:
            try:
                cols = {c.lower(): c for c in df.columns}
                c = df[cols.get("close", "Close")].values
                if "ker" not in adapt or not _is_finite(float(adapt.get("ker", 0.0))):
                    adapt["ker"] = kaufman_efficiency_ratio(c, period=int(getattr(self.cfg, "ker_period", 10) or 10))
                if "rvi" not in adapt or not _is_finite(float(adapt.get("rvi", 0.0))):
                    adapt["rvi"] = relative_volatility_index(c, period=int(getattr(self.cfg, "rvi_period", 14) or 14))
            except Exception:
                pass

        # ── SL/TP ──
        # Try structural SL first, fallback to ATR
        struct_sl = self._calculate_structure_sl(side, entry, df)
        atr_sl, atr_tp = self._fallback_atr_sl_tp(side, entry, adapt)

        sl = struct_sl if struct_sl is not None else atr_sl
        tp = atr_tp

        # Apply broker constraints
        entry, sl, tp = self._apply_broker_constraints(side, entry, sl, tp)

        # ── RR check ──
        rr = self._rr(entry, sl, tp)
        if rr < 1.0:
            result["reason"] = f"low_rr:{rr:.2f}"
            result["entry"] = entry
            result["sl"] = sl
            result["tp"] = tp
            result["rr"] = rr
            return result

        # ── Position size ──
        lot = self.calculate_position_size(side, entry, sl, tp, confidence, adapt=adapt)
        if lot <= 0:
            result["reason"] = "kelly_zero"
            result["entry"] = entry
            result["sl"] = sl
            result["tp"] = tp
            result["rr"] = rr
            return result

        result["blocked"] = False
        result["reason"] = "ok"
        result["entry"] = entry
        result["sl"] = sl
        result["tp"] = tp
        result["lot"] = lot
        result["rr"] = rr

        return result

    # ─── Trade recording ─────────────────────────────────────────────

    def record_trade(self, *args, **kwargs) -> None:
        profit = kwargs.get("profit", 0.0)
        if args and len(args) >= 1 and isinstance(args[0], (int, float)):
            profit = float(args[0])
        self._trades_today.append(profit)
        self._daily_realized_pnl += profit

    def update_market_conditions(self, spread: float, slippage: float) -> None:
        if _is_finite(spread):
            self._exec_spreads.append(spread)
        if _is_finite(slippage):
            self._exec_slippages.append(slippage)

    # ─── Execution metrics ───────────────────────────────────────────

    def record_execution_metrics(
        self,
        order_id: str,
        side: str,
        enqueue_time: float,
        send_time: float,
        fill_time: float,
        expected_price: float,
        filled_price: float,
        slippage: float = 0.0,
    ) -> None:
        """Record execution quality metrics for monitoring."""
        # Handle backward-compatible positional args
        if isinstance(side, (int, float)):
            # Old callers pass without 'side'
            log.debug("record_execution_metrics: legacy call, remapping args")
            filled_price = float(fill_time)
            fill_time = float(send_time)
            send_time = float(enqueue_time)
            enqueue_time = float(side)
            side = "Buy"

        latency_ms = (fill_time - send_time) * 1000 if fill_time > send_time else 0.0

        try:
            if mt5 is not None:
                info = mt5.symbol_info(self.sp.symbol)
                point = float(info.point) if info else 0.00001
            else:
                point = 0.00001
        except Exception:
            point = 0.00001

        slip_pts = abs(filled_price - expected_price) / point

        metrics = {
            "ts": time.time(),
            "order_id": str(order_id),
            "side": _side_norm(side),
            "latency_ms": round(latency_ms, 1),
            "slippage_pts": round(slip_pts, 2),
            "expected": expected_price,
            "filled": filled_price,
            "asset": self.sp.base,
        }

        self._exec_hist.append(metrics)
        self._exec_latencies.append(latency_ms)
        self._exec_slippages.append(slip_pts)

        # Trim
        if len(self._exec_latencies) > 200:
            self._exec_latencies = self._exec_latencies[-100:]
        if len(self._exec_slippages) > 200:
            self._exec_slippages = self._exec_slippages[-100:]

        # Breaker check
        if len(self._exec_latencies) >= 5:
            p95_lat = float(np.percentile(self._exec_latencies[-20:], 95))
            p95_slip = float(np.percentile(self._exec_slippages[-20:], 95))

            if p95_lat > self.cfg.exec_max_p95_latency_ms:
                self._exec_breaker_active = True
                self._exec_breaker_until = time.time() + self.cfg.exec_breaker_sec
                log.warning(
                    "EXEC_BREAKER_LATENCY | p95=%.1fms | %s",
                    p95_lat, self.sp.base,
                )

            if p95_slip > self.cfg.exec_max_p95_slippage_points:
                self._exec_breaker_active = True
                self._exec_breaker_until = time.time() + self.cfg.exec_breaker_sec
                log.warning(
                    "EXEC_BREAKER_SLIPPAGE | p95=%.1f pts | %s",
                    p95_slip, self.sp.base,
                )

        # Periodic CSV flush
        self._update_execution_analysis(metrics)

    def record_execution_failure(
        self, order_id: str, enqueue_time: float, send_time: float, reason: str,
    ) -> None:
        metrics = {
            "ts": time.time(),
            "order_id": str(order_id),
            "event": "FAILURE",
            "reason": reason,
            "latency_ms": round((send_time - enqueue_time) * 1000, 1),
            "asset": self.sp.base,
        }
        self._exec_hist.append(metrics)
        log.error("EXEC_FAILURE | %s | %s", self.sp.base, reason)

    def _update_execution_analysis(self, metrics: dict) -> None:
        """Fast: update RAM hist; flush periodically to CSV."""
        now = time.time()
        if now - self._exec_csv_last_flush > self._exec_csv_flush_interval:
            self._flush_exec_csv()

    def _flush_exec_csv(self, force: bool = False) -> None:
        if not self._exec_hist and not force:
            return
        try:
            self._exec_csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._exec_csv_path.exists()
            with open(self._exec_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ts", "order_id", "side", "latency_ms",
                    "slippage_pts", "expected", "filled", "event",
                    "reason", "asset",
                ])
                if write_header:
                    writer.writeheader()
                for m in self._exec_hist:
                    writer.writerow(m)
            self._exec_hist.clear()
            self._exec_csv_last_flush = time.time()
        except Exception as exc:
            log.error("_flush_exec_csv: %s", exc)

    # ─── Execution quality getters ───────────────────────────────────

    def exec_p95_latency(self) -> float:
        if len(self._exec_latencies) < 3:
            return 0.0
        return float(np.percentile(self._exec_latencies[-50:], 95))

    def exec_p95_slippage(self) -> float:
        if len(self._exec_slippages) < 3:
            return 0.0
        return float(np.percentile(self._exec_slippages[-50:], 95))

    def exec_avg_spread(self) -> float:
        if not self._exec_spreads:
            return 0.0
        return float(np.mean(self._exec_spreads[-50:]))

    # ─── State persistence ───────────────────────────────────────────

    def save_state(self) -> None:
        try:
            state = {
                "phase": self.phase,
                "daily_pnl": self._daily_pnl,
                "signals_today": self._signals_today,
                "ks_status": self._ks.status,
                "peak_equity": self._peak_equity,
                "last_date": self._last_date,
                "ts": time.time(),
            }
            path = Path(f"state/risk_{self.sp.base}.json")
            _atomic_write_text(path, json.dumps(state, indent=2))
        except Exception as exc:
            log.error("save_state: %s", exc)

    def load_state(self) -> None:
        try:
            path = Path(f"state/risk_{self.sp.base}.json")
            if not path.exists():
                return
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("last_date") == self._utc_date():
                self.phase = state.get("phase", "A")
                self._daily_pnl = state.get("daily_pnl", 0.0)
                self._signals_today = state.get("signals_today", 0)
                self._peak_equity = state.get("peak_equity", 0.0)
                log.info("Loaded state for %s: phase=%s", self.sp.base, self.phase)
        except Exception:
            pass

    # ─── Summary for health logging ──────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        return {
            "asset": self.sp.base,
            "phase": self.phase,
            "soft_stop": self._soft_stop,
            "hard_stop": self._hard_stop,
            "ks_status": self._ks.status,
            "equity": round(self._acc.equity, 2),
            "balance": round(self._acc.balance, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "peak_equity": round(self._peak_equity, 2),
            "signals_today": self._signals_today,
            "trades_today": len(self._trades_today),
            "exec_p95_lat": round(self.exec_p95_latency(), 1),
            "exec_p95_slip": round(self.exec_p95_slippage(), 2),
        }
