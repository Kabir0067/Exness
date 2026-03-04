# core/risk_engine.py — Unified RiskManager for all assets.
# Merges StrategiesBtc/_btc_risk/risk_manager.py (2400 lines)
#    and StrategiesXau/_xau_risk/risk_manager.py (2191 lines)
# into a single parameterized class (~2000 lines, zero duplication).
from __future__ import annotations

import atexit
import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
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

_DUMMY_LOCK = threading.RLock()


def _mt5_lock():
    # Lazy import avoids hard dependency and circular imports at module load.
    try:
        from mt5_client import MT5_LOCK as lock  # type: ignore
        return lock
    except Exception:
        return _DUMMY_LOCK


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

        # ── thread safety lock (protects ALL mutable state) ──
        self._lock = threading.RLock()

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
        self._spread_pct_hist: List[float] = []
        self._exec_hist: List[dict] = []
        self._exec_breaker_active: bool = False
        self._exec_breaker_until: float = 0.0
        self._exec_csv_path = Path(f"logs/exec_metrics_{sp.base}.csv")
        self._exec_csv_flush_interval: float = 120.0
        self._exec_csv_last_flush: float = 0.0
        self._ensure_exec_csv_exists()

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

        # ── volatility circuit breaker (Black Swan) ──
        self._vol_breaker_active: bool = False
        self._vol_breaker_until: float = 0.0
        self._vol_breaker_reason: str = ""
        self._last_bar_close: float = 0.0   # track previous bar close for gap detection

        # Daily target lock: once target is reached, trading stays blocked until UTC reset.
        self._daily_target_locked: bool = False
        self._daily_target_lock_reason: str = ""
        self._daily_target_lock_ts: float = 0.0

        # MT5 connectivity circuit breaker for order-path safety.
        self._mt5_disconnect_streak: int = 0
        self._mt5_breaker_active: bool = False
        self._mt5_breaker_until: float = 0.0
        self._mt5_breaker_reason: str = ""
        self._last_mt5_breaker_log_ts: float = 0.0

        # Gate account-refresh throttle (cheap, deterministic).
        self._gate_refresh_ttl_sec: float = max(
            0.25,
            float(getattr(self.cfg, "gate_account_refresh_ttl_sec", 1.0) or 1.0),
        )
        self._last_gate_refresh_ts: float = 0.0

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

    def _daily_target_pct(self) -> float:
        raw = float(getattr(self.cfg, "daily_target_pct", 0.10) or 0.10)
        # Business rule hard-floor: daily target must be at least 10%.
        return max(0.10, min(1.0, raw))

    def _lock_daily_target(self, daily_pnl_pct: float) -> None:
        """Lock strategy to monitoring-only until UTC reset after target hit."""
        with self._lock:
            if self._daily_target_locked:
                return
            self._daily_target_locked = True
            self._daily_target_lock_reason = f"DAILY_TARGET_LOCK={daily_pnl_pct:+.3f}"
            self._daily_target_lock_ts = time.time()
        self._enter_soft_stop(self._daily_target_lock_reason)
        log.warning(
            "DAILY_TARGET_LOCKED | %s | pnl_pct=%.3f target=%.3f",
            self.sp.base,
            float(daily_pnl_pct),
            self._daily_target_pct(),
        )

    @property
    def daily_target_locked(self) -> bool:
        with self._lock:
            return bool(self._daily_target_locked)

    @property
    def daily_target_lock_reason(self) -> str:
        with self._lock:
            return str(self._daily_target_lock_reason or "")

    def _mt5_order_path_ok(self) -> bool:
        if mt5 is None:
            return False
        try:
            with _mt5_lock():
                term = mt5.terminal_info()
                acc = mt5.account_info()
            return bool(
                term
                and acc
                and getattr(term, "connected", False)
                and getattr(term, "trade_allowed", False)
                and getattr(acc, "trade_allowed", False)
            )
        except Exception:
            return False

    def _check_mt5_circuit_breaker(self) -> Tuple[bool, str]:
        """
        MT5 connectivity breaker:
        - opens after N consecutive unhealthy checks,
        - cools down for fixed seconds,
        - prevents order dispatch while open.
        """
        now = time.time()
        threshold = max(1, int(getattr(self.cfg, "mt5_breaker_fail_threshold", 3) or 3))
        cooldown = max(1.0, float(getattr(self.cfg, "mt5_breaker_cooldown_sec", 30.0) or 30.0))

        with self._lock:
            if self._mt5_breaker_active and now < self._mt5_breaker_until:
                remain = max(0.0, self._mt5_breaker_until - now)
                return False, f"mt5_breaker_open:{remain:.1f}s"
            if self._mt5_breaker_active and now >= self._mt5_breaker_until:
                self._mt5_breaker_active = False
                self._mt5_breaker_reason = ""

        if self._mt5_order_path_ok():
            with self._lock:
                self._mt5_disconnect_streak = 0
            return True, "ok"

        with self._lock:
            self._mt5_disconnect_streak += 1
            streak = int(self._mt5_disconnect_streak)
            if streak >= threshold:
                self._mt5_breaker_active = True
                self._mt5_breaker_until = now + cooldown
                self._mt5_breaker_reason = "mt5_unhealthy"
                if (now - self._last_mt5_breaker_log_ts) >= 5.0:
                    self._last_mt5_breaker_log_ts = now
                    log.error(
                        "MT5_CIRCUIT_OPEN | %s | streak=%d cooldown=%.1fs",
                        self.sp.base,
                        streak,
                        cooldown,
                    )
                return False, f"mt5_breaker_open:{cooldown:.1f}s"
        return False, "mt5_unhealthy"

    # ─── Phase management ────────────────────────────────────────────

    @staticmethod
    def _phase_rank(p: str) -> int:
        return {"A": 0, "B": 1, "C": 2}.get(str(p).upper(), 0)

    def _set_phase(self, phase: str, reason: str) -> None:
        """Monotonic risk escalation within the day: A → B → C only."""
        with self._lock:
            if self._phase_rank(phase) <= self._phase_rank(self.phase):
                return
            old = self.phase
            self.phase = phase
            self._phase_reason = reason
            log.info("PHASE %s → %s | reason=%s | asset=%s", old, phase, reason, self.sp.base)

    def _enter_soft_stop(self, reason: str) -> None:
        """Trade-block until next UTC day (engine keeps running)."""
        with self._lock:
            if self._soft_stop:
                return
            self._soft_stop = True
            self._set_phase("C", reason)
            log.warning("SOFT_STOP | %s | %s", self.sp.base, reason)

    def _enter_hard_stop(self, reason: str) -> None:
        """Hard stop until next UTC day (engine-level)."""
        with self._lock:
            if self._hard_stop:
                return
            reason = str(reason or "").strip() or "unspecified"
            self._hard_stop = True
            self._hard_stop_reason = reason
            self._set_phase("C", reason)
            log.critical("HARD_STOP | %s | %s", self.sp.base, reason)

    @property
    def requires_hard_stop(self) -> bool:
        today = self._utc_date()
        with self._lock:
            if self._exec_breaker_active and time.time() < self._exec_breaker_until:
                return True
            need_reset = today != self._last_date
            hard_stop = self._hard_stop
        if need_reset:
            self._reset_daily_state()
            return False
        return hard_stop

    @property
    def hard_stop_reason(self) -> str:
        with self._lock:
            return str(self._hard_stop_reason or "unspecified")

    @property
    def phase_reason(self) -> str:
        with self._lock:
            return self._phase_reason

    def block_analysis(self, seconds: float) -> None:
        with self._lock:
            self._analysis_blocked_until = time.time() + seconds

    def can_analyze(self) -> bool:
        with self._lock:
            return time.time() >= self._analysis_blocked_until

    def on_position_opened(self) -> None:
        with self._lock:
            self._position_open = True

    def on_all_flat(self) -> None:
        with self._lock:
            self._position_open = False
        # Refresh account on flat
        try:
            self._account_snapshot()
        except Exception:
            pass

    # ─── Daily reset ─────────────────────────────────────────────────

    def _reset_daily_state(self) -> None:
        today = self._utc_date()
        with self._lock:
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
            self._spread_pct_hist.clear()
            self._exec_breaker_active = False
            # Reset volatility circuit breaker on new day
            self._vol_breaker_active = False
            self._vol_breaker_until = 0.0
            self._vol_breaker_reason = ""
            self._last_bar_close = 0.0
            self._daily_target_locked = False
            self._daily_target_lock_reason = ""
            self._daily_target_lock_ts = 0.0
            self._mt5_disconnect_streak = 0
            self._mt5_breaker_active = False
            self._mt5_breaker_until = 0.0
            self._mt5_breaker_reason = ""

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
                with _mt5_lock():
                    info = mt5.account_info()
                if info:
                    with self._lock:
                        self._bot_balance_base = float(info.balance or 0.0)
        except Exception:
            pass

    # ─── Account snapshots ───────────────────────────────────────────

    def _ensure_ready(self) -> None:
        today = self._utc_date()
        with self._lock:
            need_reset = today != self._last_date
        if need_reset:
            self._reset_daily_state()
        now = time.time()
        with self._lock:
            stale = now - self._acc.ts > 30.0
        if stale:
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
        with self._lock:
            return {
                "phase": self.phase,
                "soft_stop": self._soft_stop,
                "hard_stop": self._hard_stop,
                "ks_status": self._ks.status,
                "signals_today": self._signals_today,
                "daily_pnl": round(self._daily_pnl, 2),
                "daily_target_locked": self._daily_target_locked,
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
            with self._lock:
                self._exec_spreads.append(spread)
                if len(self._exec_spreads) > 500:
                    self._exec_spreads = self._exec_spreads[-200:]

    def on_execution_result(self, result: Any) -> None:
        try:
            ok = bool(getattr(result, "ok", False))
            order_id = str(getattr(result, "order_id", "") or "")
            sent_ts = float(getattr(result, "sent_ts", 0.0) or 0.0)
            fill_ts = float(getattr(result, "fill_ts", 0.0) or 0.0)
            slippage = float(getattr(result, "slippage", 0.0) or 0.0)
        except Exception:
            ok = False
            order_id = ""
            sent_ts = 0.0
            fill_ts = 0.0
            slippage = 0.0

        with self._lock:
            if ok:
                self._position_open = True

            has_telemetry = False
            if order_id:
                has_telemetry = any(
                    str(m.get("order_id", "")) == order_id
                    for m in self._exec_hist[-100:]
                )

            if (not has_telemetry) and fill_ts > sent_ts > 0:
                lat_ms = (fill_ts - sent_ts) * 1000.0
                if _is_finite(lat_ms):
                    self._exec_latencies.append(lat_ms)
                    if len(self._exec_latencies) > 200:
                        self._exec_latencies = self._exec_latencies[-100:]

            if (not has_telemetry) and _is_finite(slippage):
                self._exec_slippages.append(abs(float(slippage)))
                if len(self._exec_slippages) > 200:
                    self._exec_slippages = self._exec_slippages[-100:]

    def on_reconcile_positions(self, positions: Any) -> None:
        if positions is None:
            with self._lock:
                self._position_open = False
            return
        try:
            cnt = len(positions) if hasattr(positions, "__len__") else 0
            with self._lock:
                self._position_open = cnt > 0
        except Exception:
            pass

    def register_signal_emitted(self) -> None:
        with self._lock:
            self._signals_today += 1
            allowed = self._throttle.register(max_per_hour=20)
            if not allowed:
                log.warning("SIGNAL_THROTTLED | %s | hour_count=%d", self.sp.base, self._throttle.hour_window_count)

    # ─── Account helpers ─────────────────────────────────────────────

    def _account_snapshot(self) -> None:
        if mt5 is None:
            return
        try:
            with _mt5_lock():
                info = mt5.account_info()
            if info is None:
                return
            now = time.time()
            b = float(info.balance or 0.0)
            e = float(info.equity or 0.0)
            mf = float(info.margin_free or 0.0)
            if b > 0:
                with self._lock:
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
            with _mt5_lock():
                info = mt5.account_info()
            if info is None:
                return
            now = time.time()
            b = float(info.balance or 0.0)
            e = float(info.equity or 0.0)
            mf = float(info.margin_free or 0.0)
            if b > 0:
                with self._lock:
                    self._bot_acc.ts = now
                    self._bot_acc.balance = b
                    self._bot_acc.equity = e
                    self._bot_acc.margin_free = mf
        except Exception as exc:
            log.error("_bot_account_snapshot: %s", exc)

    def _get_balance(self) -> float:
        self._ensure_ready()
        with self._lock:
            return self._acc.balance

    def _get_equity(self) -> float:
        self._ensure_ready()
        with self._lock:
            return self._acc.equity

    def _get_margin_free(self) -> float:
        self._ensure_ready()
        with self._lock:
            return self._acc.margin_free

    # ─── Trade history ───────────────────────────────────────────────

    def _recent_closed_trade_profits(self, limit: int) -> List[float]:
        if mt5 is None:
            with self._lock:
                return list(self._trades_today)
        try:
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=1)
            with _mt5_lock():
                deals = mt5.history_deals_get(start, now)
            if deals is None:
                with self._lock:
                    return list(self._trades_today)
            profits = [
                float(d.profit)
                for d in deals
                if d.entry == 1 and d.symbol == self.sp.symbol
            ]
            return profits[-limit:] if len(profits) > limit else profits
        except Exception:
            with self._lock:
                return list(self._trades_today)

    # ─── Kill switch ─────────────────────────────────────────────────

    def _refresh_kill_switch(self) -> None:
        profits = self._recent_closed_trade_profits(20)
        with self._lock:
            self._ks.update(
                profits,
                time.time(),
                self.cfg.kill_min_trades,
                self.cfg.kill_expectancy,
                self.cfg.kill_winrate,
                self.cfg.cooling_expectancy,
                self.cfg.cooling_sec,
            )
            ks_status = self._ks.status
            ks_exp = self._ks.last_expectancy
            ks_wr = self._ks.last_winrate
        if ks_status == "KILLED":
            # Suppress hard-stop escalation while the market is closed.
            # This avoids noisy weekend hard-stop logs for non-24/7 assets (e.g. XAU).
            if not self.market_open():
                return
            self._enter_hard_stop(
                f"kill_switch: exp={ks_exp:.3f} wr={ks_wr:.2f}"
            )

    def strategy_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "kill_switch": self._ks.status,
                "phase": self.phase,
                "daily_target_locked": self._daily_target_locked,
            }

    # ─── Phase evaluation ────────────────────────────────────────────

    def evaluate_account_state(self) -> None:
        """
        Deterministic A/B/C logic + real C blocking + peak drawdown protect.
        """
        self._ensure_ready()
        with self._lock:
            eq = self._acc.equity
            bal = self._acc.balance

        if eq <= 0 or bal <= 0:
            return

        # Daily PnL
        with self._lock:
            bot_balance_base = self._bot_balance_base
        if bot_balance_base > 0:
            daily_pnl = eq - bot_balance_base
            daily_pnl_pct = daily_pnl / bot_balance_base
            with self._lock:
                self._daily_pnl = daily_pnl
        else:
            daily_pnl_pct = 0.0

        target_pct = self._daily_target_pct()
        if target_pct > 0.0 and daily_pnl_pct >= target_pct:
            self._lock_daily_target(daily_pnl_pct)

        # Update peak
        with self._lock:
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
        with self._lock:
            peak_eq = self._peak_equity
        if peak_eq > 0:
            dd_from_peak = (peak_eq - eq) / peak_eq
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

    def update_pnl(self, pnl_delta: float) -> None:
        """
        Atomic PnL/equity update for concurrent callers.
        Uses Decimal to reduce floating precision drift under high-frequency updates.
        """
        if not _is_finite(pnl_delta):
            return
        with self._lock:
            delta_dec = Decimal(str(float(pnl_delta)))
            eq_dec = Decimal(str(float(self._acc.equity or 0.0)))
            base_dec = Decimal(str(float(self._bot_balance_base or 0.0)))

            new_eq = eq_dec + delta_dec
            self._acc.equity = float(new_eq)
            self._daily_realized_pnl = float(Decimal(str(self._daily_realized_pnl)) + delta_dec)

            if base_dec > 0:
                self._daily_pnl = float(new_eq - base_dec)
            else:
                self._daily_pnl = 0.0

            if self._acc.equity > self._peak_equity:
                self._peak_equity = self._acc.equity
            if self._acc.equity > self._daily_peak:
                self._daily_peak = self._acc.equity

    # ─── Volatility Circuit Breaker (Black Swan Protection) ─────────

    def check_volatility_circuit_breaker(self, dfp: Optional[pd.DataFrame] = None) -> bool:
        """
        Black Swan protection: detect extreme volatility conditions.

        Checks:
          1. ATR expansion ratio: ATR / SMA(ATR, 50) > threshold
          2. Price gap: |close[-1] - close[-2]| > gap_mult × ATR

        Returns True if circuit breaker was triggered.
        """
        with self._lock:
            if self._vol_breaker_active and time.time() < self._vol_breaker_until:
                return True  # already tripped
        if dfp is None or len(dfp) < 60:
            return False
        try:
            cols = {c.lower(): c for c in dfp.columns}
            h = dfp[cols.get("high", "High")].values.astype(np.float64)
            l = dfp[cols.get("low", "Low")].values.astype(np.float64)
            c = dfp[cols.get("close", "Close")].values.astype(np.float64)
            atr_arr = _atr_fallback(h, l, c, 14)
            if atr_arr is None or len(atr_arr) < 55:
                return False

            atr_last = float(atr_arr[-1])
            if not _is_finite(atr_last) or atr_last <= 0:
                return False

            # 1. ATR expansion check
            atr_sma50 = float(np.mean(atr_arr[-50:]))
            if atr_sma50 > 0:
                atr_ratio = atr_last / atr_sma50
                threshold = float(getattr(self.cfg, 'circuit_breaker_atr_ratio', 3.0) or 3.0)
                if atr_ratio > threshold:
                    reason = f"ATR_EXPANSION:{atr_ratio:.2f}x>{threshold:.1f}x"
                    self._trigger_vol_breaker(reason)
                    return True

            # 2. Price gap check
            if len(c) >= 2:
                gap = abs(float(c[-1]) - float(c[-2]))
                gap_mult = float(getattr(self.cfg, 'circuit_breaker_gap_atr_mult', 2.0) or 2.0)
                if gap > gap_mult * atr_last:
                    reason = f"PRICE_GAP:{gap:.2f}>{gap_mult:.1f}xATR({atr_last:.2f})"
                    self._trigger_vol_breaker(reason)
                    return True

        except Exception as exc:
            log.error("check_volatility_circuit_breaker: %s", exc)
        return False

    def _trigger_vol_breaker(self, reason: str) -> None:
        """Activate the volatility circuit breaker."""
        with self._lock:
            cooldown = float(getattr(self.cfg, 'circuit_breaker_cooldown_sec', 1800.0) or 1800.0)
            self._vol_breaker_active = True
            self._vol_breaker_until = time.time() + cooldown
            self._vol_breaker_reason = reason
            log.critical(
                "VOL_CIRCUIT_BREAKER_TRIGGERED | %s | %s | cooldown=%.0fs",
                self.sp.base, reason, cooldown,
            )
            self._enter_hard_stop(f"vol_circuit_breaker:{reason}")

    def update_phase(self) -> None:
        self.evaluate_account_state()

    # ─── Latency management ──────────────────────────────────────────

    def latency_cooldown(self) -> bool:
        now = time.time()
        with self._lock:
            if now < self._latency_cd_until:
                return True
            if self._exec_breaker_active and now < self._exec_breaker_until:
                return True
            return False

    def register_latency_violation(self) -> None:
        with self._lock:
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
            reasons.append(f"HARD_STOP:{self.hard_stop_reason}")
            return False, reasons

        with self._lock:
            soft_stop = self._soft_stop
            ks_status = self._ks.status
        if soft_stop:
            reasons.append("SOFT_STOP")
            return False, reasons

        if ks_status != "ACTIVE":
            reasons.append(f"KILL_SWITCH:{ks_status}")
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

        min_age = float(getattr(self.cfg, "market_min_bar_age_sec", 30.0) or 30.0)
        max_age_mult = float(getattr(self.cfg, "market_max_bar_age_mult", 2.0) or 2.0)
        max_age = max(min_age * max_age_mult, min_age + 10.0)
        if _is_finite(last_bar_age) and last_bar_age > max_age:
            reasons.append(f"STALE_FEED:{last_bar_age:.1f}s>{max_age:.1f}s")
            return False, reasons

        base_limit = float(getattr(self.sp, "spread_limit_pct", 0.0) or 0.0)
        cfg_limit = float(getattr(self.cfg, "spread_cb_pct", 0.0) or 0.0)
        if bool(getattr(self.sp, "is_24_7", False)):
            eff_limit = max(base_limit, cfg_limit) if cfg_limit > 0 else base_limit
        else:
            eff_limit = base_limit

        spread_mult = float(getattr(self.cfg, "spread_gate_multiplier", 1.5) or 1.5)
        spread_limit = eff_limit * spread_mult
        if spread_limit > 0 and spread_pct > spread_limit:
            reasons.append(f"SPREAD_WIDE:{spread_pct:.5f}>{spread_limit:.5f}")
            return False, reasons

        # Spread spike detection: block if spread > Nσ above recent history
        with self._lock:
            if _is_finite(spread_pct) and spread_pct > 0:
                recent_pct = self._spread_pct_hist[-50:]
            else:
                recent_pct = []
        if len(recent_pct) >= 10:
            try:
                mu = float(np.mean(recent_pct))
                sigma = float(np.std(recent_pct))
                if sigma > 0:
                    sigma_thresh = float(
                        getattr(
                            self.cfg,
                            "spread_spike_sigma_threshold",
                            getattr(self.cfg, "spread_spike_sigma", 3.0),
                        )
                        or 3.0
                    )
                    threshold = mu + sigma_thresh * sigma
                    if spread_pct > threshold:
                        reasons.append(f"SPREAD_SPIKE:{spread_pct:.5f}>{threshold:.5f}")
                        return False, reasons
            except Exception:
                pass
        with self._lock:
            if _is_finite(spread_pct) and spread_pct > 0:
                self._spread_pct_hist.append(spread_pct)
                if len(self._spread_pct_hist) > 500:
                    self._spread_pct_hist = self._spread_pct_hist[-200:]

        # Volatility circuit breaker (Black Swan)
        with self._lock:
            vol_active = self._vol_breaker_active and time.time() < self._vol_breaker_until
            vol_reason = self._vol_breaker_reason
        if vol_active:
            reasons.append(f"VOL_CIRCUIT_BREAKER:{vol_reason}")
            return False, reasons

        if drawdown_exceeded:
            reasons.append("DD_EXCEEDED")
            return False, reasons

        if latency_cooldown:
            reasons.append("LATENCY_CD")
            return False, reasons

        with self._lock:
            exec_breaker_active = self._exec_breaker_active and time.time() < self._exec_breaker_until
        if exec_breaker_active:
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
            return False, f"hard_stop:{self.hard_stop_reason}"

        with self._lock:
            soft_stop = self._soft_stop
            ks_status = self._ks.status
            phase = self.phase
            daily_locked = self._daily_target_locked
            daily_lock_reason = self._daily_target_lock_reason
        if soft_stop:
            if daily_locked:
                return False, f"daily_target_locked:{daily_lock_reason or 'target_reached'}"
            return False, "soft_stop"

        if ks_status != "ACTIVE":
            return False, f"kill_switch:{ks_status}"

        if phase == "C":
            return False, "phase_C_block"

        if not self.market_open():
            return False, "market_closed"

        if self._in_maintenance_blackout():
            return False, "maintenance"

        # Refresh account state
        self.evaluate_account_state()

        if self.requires_hard_stop:
            return False, f"hard_stop_after_eval:{self.hard_stop_reason}"

        return True, ""

    def can_emit_signal(self, confidence: int, tz: Any = None) -> Tuple[bool, str]:
        if confidence < self.cfg.min_confidence:
            return False, f"low_conf:{confidence}"

        if self.requires_hard_stop:
            return False, f"hard_stop:{self.hard_stop_reason}"

        with self._lock:
            ks_status = self._ks.status
            ks_cooling_until = self._ks.cooling_until_ts
        if ks_status == "KILLED":
            return False, "killed"

        if ks_status == "COOLING":
            return False, f"cooling:{ks_cooling_until - time.time():.0f}s"

        if not self.market_open():
            return False, "market_closed"

        if self._in_maintenance_blackout():
            return False, "maintenance"

        return True, ""

    def pre_order_gate(
        self,
        *,
        side: str,
        confidence: float,
        lot: float,
        entry_price: float,
        sl: float,
        tp: float,
        signal_id: str = "",
        stage: str = "enqueue",
        base_lot: Optional[float] = None,
        phase_snapshot: str = "",
    ) -> Tuple[bool, float, str]:
        """
        Final fail-safe order gate.
        Runs before enqueue/dispatch and blocks risk-violating orders.
        """
        side_n = _side_norm(side)
        lot_val = float(lot or 0.0)
        entry = float(entry_price or 0.0)
        sl_val = float(sl or 0.0)
        tp_val = float(tp or 0.0)
        base_lot_val = float(base_lot) if base_lot is not None else lot_val
        if base_lot_val <= 0.0:
            base_lot_val = lot_val

        if side_n not in ("Buy", "Sell"):
            return False, 0.0, "gate_bad_side"
        if not _is_finite(lot_val, entry, sl_val, tp_val) or lot_val <= 0.0:
            return False, 0.0, "gate_bad_values"
        if side_n == "Buy" and not (sl_val < entry < tp_val):
            return False, 0.0, "gate_bad_sl_tp_buy"
        if side_n == "Sell" and not (tp_val < entry < sl_val):
            return False, 0.0, "gate_bad_sl_tp_sell"

        # Cheap refresh for deterministic gating under high-frequency flow.
        now = time.time()
        refresh = False
        with self._lock:
            if (now - self._last_gate_refresh_ts) >= self._gate_refresh_ttl_sec:
                self._last_gate_refresh_ts = now
                refresh = True
        if refresh:
            self._account_snapshot()

        self.evaluate_account_state()

        if mt5 is not None and not bool(getattr(self.cfg, "dry_run", False)):
            mt5_ok, mt5_reason = self._check_mt5_circuit_breaker()
            if not mt5_ok:
                return False, 0.0, mt5_reason

        can, reason = self.can_trade(float(confidence), "order_gate")
        if not can:
            return False, 0.0, reason

        with self._lock:
            phase_now = str(self.phase).upper()
            daily_locked = bool(self._daily_target_locked)
            daily_reason = str(self._daily_target_lock_reason or "")

        if daily_locked:
            return False, 0.0, f"daily_target_locked:{daily_reason or 'target_reached'}"

        if phase_now == "C":
            return False, 0.0, "phase_C_block"

        # Strict A/B/C exposure gate on dispatch path.
        if str(stage).lower() == "dispatch":
            mode_mult = {"A": 1.0, "B": 0.5, "C": 0.0}.get(phase_now, 1.0)
            allowed_lot = max(0.0, base_lot_val * mode_mult)
            tol = max(1e-8, allowed_lot * 1e-6)
            if lot_val > (allowed_lot + tol):
                return False, 0.0, f"mode_{phase_now}_lot_violation:{lot_val:.8f}>{allowed_lot:.8f}"

        return True, lot_val, "ok"

    # ─── SL/TP calculation ───────────────────────────────────────────

    def _apply_broker_constraints(
        self, side: str, entry: float, sl: float, tp: float,
    ) -> Tuple[float, float, float]:
        """Ensure SL/TP respect broker minimum distance constraints."""
        if mt5 is None:
            return entry, sl, tp
        try:
            with _mt5_lock():
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
        lot_floor = float(getattr(self.sp, "lot_min", 0.01) or 0.01)
        if risk_pct <= 0.0:
            if conf01 >= 0.80 and rr > 1.5:
                return lot_floor
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
        with self._lock:
            phase_snapshot = str(self.phase).upper()
            peak_eq = float(self._peak_equity or 0.0)
        phase_mult = {"A": 1.0, "B": 0.75, "C": 0.5}.get(phase_snapshot, 1.0)
        dd_mult = 1.0
        if peak_eq > 0:
            dd_from_peak = (peak_eq - eq) / peak_eq
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
                with _mt5_lock():
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
                    with _mt5_lock():
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
        # ML router signals already enforce high confidence (>=0.75), allow lower R:R
        _regime = str(adapt.get("regime", "") or "") if isinstance(adapt, dict) else ""
        min_rr = 0.5 if _regime == "ml_router" else 1.0
        if rr < min_rr:
            result["reason"] = f"low_rr:{rr:.2f}<{min_rr:.1f}"
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
        with self._lock:
            self._trades_today.append(profit)
            self._daily_realized_pnl += profit

    def update_market_conditions(self, spread: float, slippage: float) -> None:
        with self._lock:
            if _is_finite(spread):
                self._exec_spreads.append(spread)
                if len(self._exec_spreads) > 500:
                    self._exec_spreads = self._exec_spreads[-200:]
            if _is_finite(slippage):
                self._exec_slippages.append(slippage)
                if len(self._exec_slippages) > 200:
                    self._exec_slippages = self._exec_slippages[-100:]

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
                with _mt5_lock():
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

        with self._lock:
            self._exec_hist.append(metrics)
            self._exec_latencies.append(latency_ms)
            self._exec_slippages.append(slip_pts)

            # Trim (prevent memory leaks)
            if len(self._exec_latencies) > 200:
                self._exec_latencies = self._exec_latencies[-100:]
            if len(self._exec_slippages) > 200:
                self._exec_slippages = self._exec_slippages[-100:]
            if len(self._exec_hist) > 500:
                self._exec_hist = self._exec_hist[-250:]

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
        with self._lock:
            self._exec_hist.append(metrics)
            if len(self._exec_hist) > 500:
                self._exec_hist = self._exec_hist[-250:]
        log.error("EXEC_FAILURE | %s | %s", self.sp.base, reason)

    def _update_execution_analysis(self, metrics: dict) -> None:
        """Fast: update RAM hist; flush periodically to CSV."""
        now = time.time()
        with self._lock:
            should_flush = now - self._exec_csv_last_flush > self._exec_csv_flush_interval
        if should_flush:
            self._flush_exec_csv()

    def _flush_exec_csv(self, force: bool = False) -> None:
        with self._lock:
            if not self._exec_hist and not force:
                return
        try:
            with self._lock:
                rows = list(self._exec_hist)
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
                for m in rows:
                    writer.writerow(m)
            with self._lock:
                if len(rows) >= len(self._exec_hist):
                    self._exec_hist.clear()
                else:
                    self._exec_hist = self._exec_hist[len(rows):]
                self._exec_csv_last_flush = time.time()
        except Exception as exc:
            log.error("_flush_exec_csv: %s", exc)

    def _ensure_exec_csv_exists(self) -> None:
        """Create per-asset execution metrics CSV with header upfront."""
        try:
            self._exec_csv_path.parent.mkdir(parents=True, exist_ok=True)
            if self._exec_csv_path.exists():
                return
            with open(self._exec_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ts", "order_id", "side", "latency_ms",
                        "slippage_pts", "expected", "filled", "event",
                        "reason", "asset",
                    ],
                )
                writer.writeheader()
        except Exception as exc:
            log.error("_ensure_exec_csv_exists: %s", exc)

    # ─── Execution quality getters ───────────────────────────────────

    def exec_p95_latency(self) -> float:
        with self._lock:
            if len(self._exec_latencies) < 3:
                return 0.0
            return float(np.percentile(self._exec_latencies[-50:], 95))

    def exec_p95_slippage(self) -> float:
        with self._lock:
            if len(self._exec_slippages) < 3:
                return 0.0
            return float(np.percentile(self._exec_slippages[-50:], 95))

    def exec_avg_spread(self) -> float:
        with self._lock:
            if not self._exec_spreads:
                return 0.0
            return float(np.mean(self._exec_spreads[-50:]))

    # ─── State persistence ───────────────────────────────────────────

    def save_state(self) -> None:
        try:
            with self._lock:
                state = {
                    "phase": self.phase,
                    "daily_pnl": self._daily_pnl,
                    "signals_today": self._signals_today,
                    "ks_status": self._ks.status,
                    "peak_equity": self._peak_equity,
                    "daily_target_locked": self._daily_target_locked,
                    "daily_target_lock_reason": self._daily_target_lock_reason,
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
                with self._lock:
                    self.phase = state.get("phase", "A")
                    self._daily_pnl = state.get("daily_pnl", 0.0)
                    self._signals_today = state.get("signals_today", 0)
                    self._peak_equity = state.get("peak_equity", 0.0)
                    self._daily_target_locked = bool(state.get("daily_target_locked", False))
                    self._daily_target_lock_reason = str(state.get("daily_target_lock_reason", "") or "")
                log.info("Loaded state for %s: phase=%s", self.sp.base, self.phase)
        except Exception:
            pass

    # ─── Summary for health logging ──────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        with self._lock:
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
                "vol_breaker": self._vol_breaker_active,
                "daily_target_locked": self._daily_target_locked,
                "daily_target_lock_reason": self._daily_target_lock_reason,
            }
