# _btc_risk/risk_manager.py — BTC risk management (uses BrokerAdapter + submodules)
from __future__ import annotations

import atexit
import csv
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:
    talib = None  # type: ignore

from config_btc import EngineConfig, SymbolParams
from DataFeed.btc_market_feed import MicroZones, TickStats
from mt5_client import MT5_LOCK, ensure_mt5


from .broker_adapter import BrokerAdapter
from .execution_quality import ExecSample, ExecutionQualityMonitor
from .logging_ import _FILE_LOCK, LOG_DIR, log_risk
from .models import RiskDecision, SignalSurvivalState
from .utils import (
    _atr_fallback,
    _atomic_write_text,
    _is_finite,
    _side_norm,
    _utcnow,
    percentile_rank,
    adaptive_risk_money,
    percentile_rank,
    adaptive_risk_money,
    atr_take_profit,
)


@dataclass
class KillSwitchState:
    status: str = "ACTIVE"
    cooling_until_ts: float = 0.0
    last_expectancy: float = 0.0
    last_winrate: float = 0.0
    last_trades: int = 0

    def update(self, profits: List[float], now: float, min_trades: int, kill_expectancy: float, kill_winrate: float, cooling_expectancy: float, cooling_sec: float) -> None:
        self.last_trades = len(profits)
        if self.last_trades < min_trades:
            return  # Not enough data, keep status

        wins = sum(1 for p in profits if p > 0)
        # Avoid numpy dependency here if possible, or use it if imported
        avg_win = float(np.mean([p for p in profits if p > 0])) if wins > 0 else 0.0
        losses = [abs(p) for p in profits if p <= 0]
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        self.last_winrate = wins / self.last_trades if self.last_trades > 0 else 0.0
        self.last_expectancy = (self.last_winrate * avg_win) - ((1.0 - self.last_winrate) * avg_loss)

        # Logic
        if self.last_expectancy < kill_expectancy:
            self.status = "KILLED"
        elif self.last_expectancy < cooling_expectancy:
             if self.status != "COOLING":
                 self.status = "COOLING"
                 self.cooling_until_ts = now + cooling_sec
        else:
            if self.status == "COOLING" and now < self.cooling_until_ts:
                pass 
            else:
                self.status = "ACTIVE"


@dataclass
class AccountCache:
    ts: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0


@dataclass
class SignalThrottle:
    daily_count: int = 0
    hour_window_start_ts: float = field(default_factory=lambda: time.time())
    hour_window_count: int = 0


class RiskManager:
    _REQUIRED_CFG_FIELDS = (
        "daily_target_pct",
        "max_daily_loss_pct",
        "protect_drawdown_from_peak_pct",
        "max_drawdown",
        "max_risk_per_trade",
        "analysis_cooldown_sec",
        "cooldown_seconds",
        "cooldown_after_latency_s",
        "min_confidence_signal",
        "ultra_confidence_min",
        "min_bar_age_sec",
        "max_signals_per_day",
        "sl_atr_mult_trend",
        "tp_atr_mult_trend",
        "sl_atr_mult_range",
        "tp_atr_mult_range",
        "multi_order_tp_bonus_pct",
        "multi_order_sl_tighten_pct",
        "fixed_volume",
    )

    def __init__(self, cfg: EngineConfig, sp: SymbolParams) -> None:
        self.cfg = cfg
        self.sp = sp
        self.symbol = sp.resolved or sp.base

        self._validate_cfg()

        # Daily state (UTC)
        self.daily_date = self._utc_date()
        self.daily_start_balance = 0.0
        self.daily_peak_equity = 0.0
        self._target_breached = False
        self._target_breached_return = 0.0
        self._phase_reason_last: Optional[str] = None
        self.current_phase = "A"
        self._trading_disabled_until: float = 0.0   # HARD stop (engine-level)
        self._hard_stop_reason: Optional[str] = None

        # SOFT stop (trade-block only, engine keeps running)
        self._soft_stop_until: float = 0.0
        self._soft_stop_reason: Optional[str] = None

        # Runtime
        self._current_drawdown = 0.0
        self._analysis_blocked_until: float = 0.0
        self._last_fill_ts: float = 0.0
        self._latency_bad_since: Optional[float] = None

        # Account snapshot cache (unified structure)
        self._acc_cache = AccountCache()
        self._bot_balance_base: float = 0.0

        # Signal throttling (unified structure)
        self._signal_throttle = SignalThrottle()

        # MT5 readiness (via broker)
        self._ready_ts: float = 0.0
        self._ready_ok: bool = False

        # Broker adapter (symbol meta, price/vol, margin/profit, spread)
        self.broker = BrokerAdapter(self.symbol)

        # Execution quality breaker
        self.execmon = ExecutionQualityMonitor(window=int(getattr(self.cfg, "exec_window", 300) or 300))
        self._exec_breaker_until: float = 0.0

        # Reconcile tracking
        self._last_open_positions: int = 0

        # Signal survival (with memory limit)
        self._survival: Dict[str, SignalSurvivalState] = {}
        self._survival_last_flush_ts: float = 0.0
        self._survival_last_update_emit: Dict[str, float] = {}
        self._survival_max_size: int = int(getattr(self.cfg, "signal_survival_max_entries", 1000) or 1000)
        self._survival_max_size: int = int(getattr(self.cfg, "signal_survival_max_entries", 1000) or 1000)
        self._init_signal_survival()

        # --- FIX: Kill Switch Initialization (Critical) ---
        self._kill_last_check_ts: float = 0.0
        self._kill_check_interval_sec: float = 60.0
        self._kill_state = KillSwitchState()
        
        # Configurable thresholds (or defaults)
        self._kill_history_days = float(getattr(self.cfg, "kill_switch_lookback_days", 1) or 1)
        self._kill_min_trades = int(getattr(self.cfg, "kill_switch_min_trades", 5) or 5)
        self._kill_expectancy_threshold = float(getattr(self.cfg, "kill_switch_expectancy", -50.0) or -50.0) # Very loose default
        self._kill_winrate_threshold = float(getattr(self.cfg, "kill_switch_min_winrate", 0.05) or 0.05)
        self._cooling_expectancy_threshold = float(getattr(self.cfg, "kill_switch_cooling_expectancy", -20.0) or -20.0)
        self._cooling_duration_sec = float(getattr(self.cfg, "kill_switch_cooling_sec", 1800.0) or 1800.0)
        # --------------------------------------------------

        # Execution hist (fast RAM + periodic flush, with memory limits)
        self._slip_bins = [-float("inf"), -50, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, float("inf")]
        self._slip_counts = [0] * (len(self._slip_bins) - 1)
        self._delay_bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, float("inf")]
        self._delay_counts = [0] * (len(self._delay_bins) - 1)
        self._hist_last_flush_ts: float = 0.0
        self._reject_counts: Dict[str, int] = {}
        self._reject_last_flush_ts: float = 0.0
        self._reject_counts_max_size: int = int(getattr(self.cfg, "reject_counts_max_size", 500) or 500)
        
        # Register shutdown handler for data safety (use lambda to ensure method is bound)
        def _shutdown_wrapper() -> None:
            try:
                if hasattr(self, '_shutdown_flush_all'):
                    self._shutdown_flush_all()
            except Exception:
                pass  # Ignore errors during shutdown
        atexit.register(_shutdown_wrapper)

        # init daily
        self._reset_daily_state()

        # load persisted state (optional)
        try:
            self.load_state()
        except Exception:
            pass

        # load active survival cache (optional)
        try:
            self._load_survival_active()
        except Exception:
            pass

    # ------------------- config validation -------------------
    def _validate_cfg(self) -> None:
        missing = [k for k in self._REQUIRED_CFG_FIELDS if not hasattr(self.cfg, k)]
        if missing:
            raise RuntimeError(f"EngineConfig missing required fields: {missing}")

        if float(self.cfg.max_drawdown) <= 0 or float(self.cfg.max_drawdown) >= 1:
            raise RuntimeError("EngineConfig.max_drawdown must be in (0, 1)")
        if float(self.cfg.max_risk_per_trade) <= 0 or float(self.cfg.max_risk_per_trade) >= 1:
            raise RuntimeError("EngineConfig.max_risk_per_trade must be in (0, 1)")
        if float(self.cfg.max_daily_loss_pct) <= 0 or float(self.cfg.max_daily_loss_pct) >= 1:
            raise RuntimeError("EngineConfig.max_daily_loss_pct must be in (0, 1)")
        if float(self.cfg.daily_target_pct) <= 0 or float(self.cfg.daily_target_pct) >= 1:
            raise RuntimeError("EngineConfig.daily_target_pct must be in (0, 1)")
        if float(self.cfg.ultra_confidence_min) <= 0 or float(self.cfg.ultra_confidence_min) > 1:
            raise RuntimeError("EngineConfig.ultra_confidence_min must be in (0, 1]")
        if float(self.cfg.min_confidence_signal) <= 0 or float(self.cfg.min_confidence_signal) > 1:
            raise RuntimeError("EngineConfig.min_confidence_signal must be in (0, 1]")

    # ------------------- time / daily -------------------
    @staticmethod
    def _utc_date():
        return _utcnow().date()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _seconds_until_next_utc_day(self) -> float:
        now = _utcnow()
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(0.0, (nxt - now).total_seconds())

    @staticmethod
    def _phase_rank(p: str) -> int:
        pp = str(p or "A").upper()
        if pp == "A":
            return 0
        if pp == "B":
            return 1
        return 2

    def _set_phase(self, phase: str, reason: str) -> None:
        """
        Monotonic risk escalation within the day: A -> B -> C only.
        """
        ph = str(phase or "").upper()
        if ph not in ("A", "B", "C"):
            return

        if self._phase_rank(ph) >= self._phase_rank(self.current_phase):
            self.current_phase = ph
            self._phase_reason_last = str(reason)

    def _enter_soft_stop(self, reason: str) -> None:
        """
        Trade-block until next UTC day (engine keeps running).
        """
        lock = self._seconds_until_next_utc_day()
        until = time.time() + lock
        self._soft_stop_until = max(self._soft_stop_until, until)
        self._soft_stop_reason = str(reason)
        self._set_phase("C", str(reason))
        # Also block analysis to avoid noisy CPU loops
        self._analysis_blocked_until = max(self._analysis_blocked_until, until)

    def _enter_hard_stop(self, reason: str) -> None:
        """
        Hard stop until next UTC day (engine-level).
        """
        self._set_phase("C", str(reason))
        self._hard_stop_reason = str(reason)
        self._target_breached = True
        lock = self._seconds_until_next_utc_day()
        self._trading_disabled_until = time.time() + lock
        self._analysis_blocked_until = max(self._analysis_blocked_until, time.time() + lock)

    def requires_hard_stop(self) -> bool:
        if not bool(getattr(self.cfg, "enforce_daily_limits", False)):
            return False
        if bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
            return False
        if self._trading_disabled_until <= 0:
            return False
        if time.time() >= self._trading_disabled_until:
            self._trading_disabled_until = 0.0
            self._hard_stop_reason = None
            if self.current_phase == "C":
                self.current_phase = "A"
            return False
        return True

    def hard_stop_reason(self) -> Optional[str]:
        return self._hard_stop_reason

    def phase_reason(self) -> Optional[str]:
        return self._phase_reason_last

    def block_analysis(self, seconds: float) -> None:
        self._analysis_blocked_until = max(self._analysis_blocked_until, time.time() + float(seconds))

    def can_analyze(self) -> bool:
        return time.time() >= self._analysis_blocked_until

    def on_position_opened(self) -> None:
        if bool(getattr(self.cfg, "pause_analysis_on_position_open", False)):
            self.block_analysis(float(getattr(self.cfg, "cooldown_seconds", 2.0) or 2.0))

    def on_all_flat(self) -> None:
        dd = float(self._current_drawdown)
        cd = float(self.cfg.analysis_cooldown_sec)
        if dd > float(self.cfg.max_drawdown) * 0.5:
            cd *= 1.5
        self.block_analysis(cd)


    def _reset_daily_state(self) -> None:
        self.daily_date = self._utc_date()

        # base snapshot
        bal, eq = self._account_snapshot()
        current_balance = float(bal if bal > 0 else eq)
        start_balance = float(current_balance if current_balance > 0 else 0.0)
        peak_equity = float(max(float(eq or 0.0), start_balance))

        try:
            from ExnessAPI.daily_balance import initialize_daily_balance, get_peak_balance

            # balance provider (kept exactly like your style with fallback import)
            try:
                from ExnessAPI.functions import get_balance as _get_balance  # type: ignore
            except Exception:  # pragma: no cover
                from ExnessApi.functions import get_balance as _get_balance  # type: ignore

            saved_balance, _saved_mode = initialize_daily_balance(
                float(current_balance),
                asset="BTC",
                balance_provider=_get_balance,
            )

            if float(saved_balance) > 0:
                start_balance = float(saved_balance)

            peak_file = float(get_peak_balance(asset="BTC") or 0.0)
            peak_equity = float(max(peak_equity, start_balance, peak_file))
        except Exception as e:
            log_risk.error("_reset_daily_state daily_balance error: %s | tb=%s", e, traceback.format_exc())

        self.daily_start_balance = float(start_balance)
        self.daily_peak_equity = float(peak_equity)

        # IMPORTANT: new UTC day always starts from Phase A
        self.current_phase = "A"
        self._phase_reason_last = None

        self._target_breached = False
        self._target_breached_return = 0.0

        # Stops reset daily
        self._trading_disabled_until = 0.0
        self._hard_stop_reason = None
        self._soft_stop_until = 0.0
        self._soft_stop_reason = None

        self._signal_throttle.daily_count = 0
        self._signal_throttle.hour_window_start_ts = time.time()
        self._signal_throttle.hour_window_count = 0
        self._reset_bot_balance_base()



    def _reset_bot_balance_base(self) -> None:
        if not bool(getattr(self.cfg, "ignore_external_positions", False)):
            return
        try:
            if not self._ensure_ready():
                return
            with MT5_LOCK:
                acc = mt5.account_info()
            if acc:
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                if bal > 0:
                    self._bot_balance_base = bal
        except Exception:
            pass

    # ------------------- MT5 readiness (via broker) -------------------
    def _ensure_ready(self) -> bool:
        try:
            now = time.time()
            if (now - self._ready_ts) < 1.0:
                return bool(self._ready_ok)
            self._ready_ok = self.broker.ensure_ready()
            self._ready_ts = now
            return bool(self._ready_ok)
        except Exception as exc:
            self._ready_ok = False
            self._ready_ts = time.time()
            log_risk.error("ensure_ready error: %s | tb=%s", exc, traceback.format_exc())
            return False

    # ------------------- BTC session: trade_mode + blackout -------------------
    def _minutes_utc(self) -> Tuple[int, int]:
        now = self._utc_now()
        return int(now.weekday()), int(now.hour * 60 + now.minute)

    def _in_maintenance_blackout(self) -> bool:
        try:
            windows = getattr(self.cfg, "crypto_blackout_windows_utc", None)
            if not windows:
                return False

            wd, m = self._minutes_utc()
            for w in windows:
                if int(w.get("weekday", -1)) != wd:
                    continue
                start = int(w.get("start_min", -1))
                end = int(w.get("end_min", -1))
                if start < 0 or end < 0:
                    continue

                if start <= end:
                    if start <= m <= end:
                        return True
                else:
                    if (m >= start) or (m <= end):
                        return True
            return False
        except Exception:
            return False

    def _trade_mode_state(self) -> Tuple[bool, str]:
        if not self._ensure_ready():
            return False, "mt5_not_ready"

        try:
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return False, "symbol_info_missing"

            tm = int(getattr(info, "trade_mode", mt5.SYMBOL_TRADE_MODE_FULL) or mt5.SYMBOL_TRADE_MODE_FULL)

            if tm == mt5.SYMBOL_TRADE_MODE_DISABLED:
                return False, "trade_mode_disabled"
            if tm == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
                return False, "trade_mode_close_only"

            return True, ""
        except Exception as exc:
            log_risk.error("_trade_mode_state error: %s | tb=%s", exc, traceback.format_exc())
            return False, "trade_mode_error"


    def market_open_24_5(self) -> bool:
        if self._in_maintenance_blackout():
            return False
        ok, _ = self._trade_mode_state()
        return bool(ok)


    def rollover_blackout(self) -> bool:
        sec = float(getattr(self.cfg, "rollover_blackout_sec", 0.0) or 0.0)
        if sec <= 0:
            return False
        now = _utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = abs((now - midnight).total_seconds())
        return delta <= sec

    # ------------------- quote/execution hooks -------------------
    def on_quote(self, bid: float, ask: float) -> None:
        try:
            if not self.broker.ensure_ready():
                return
            self.execmon.on_quote(float(bid), float(ask), float(self.broker.point_value()))
        except Exception as exc:
            log_risk.error("on_quote error: %s | tb=%s", exc, traceback.format_exc())

    def on_execution_result(self, result: Any) -> None:
        try:
            ok = bool(getattr(result, "ok", False))
            if ok:
                self._last_fill_ts = time.time()
        except Exception:
            pass

    def on_reconcile_positions(self, positions: Any) -> None:
        try:
            n = len(positions) if positions else 0
            if n > 0 and self._last_open_positions == 0:
                self.on_position_opened()
            if n == 0 and self._last_open_positions > 0:
                self.on_all_flat()
            self._last_open_positions = n
        except Exception:
            pass

    def register_signal_emitted(self) -> None:
        now = time.time()

        if self._utc_date() != self.daily_date:
            self._reset_daily_state()

        if (now - self._signal_throttle.hour_window_start_ts) >= 3600.0:
            self._signal_throttle.hour_window_start_ts = now
            self._signal_throttle.hour_window_count = 0

        self._signal_throttle.hour_window_count += 1
        self._signal_throttle.daily_count += 1

    # ------------------- account snapshot (cached) -------------------
    def _account_snapshot(self) -> Tuple[float, float]:
        if bool(getattr(self.cfg, "ignore_external_positions", False)):
            return self._bot_account_snapshot()

        cache_sec = float(getattr(self.cfg, "account_snapshot_cache_sec", 0.5) or 0.5)
        now = time.time()

        if (now - float(self._acc_cache.ts)) < cache_sec:
            return float(self._acc_cache.balance), float(self._acc_cache.equity)

        if not self._ensure_ready():
            return float(self._acc_cache.balance), float(self._acc_cache.equity)

        try:
            with MT5_LOCK:
                acc = mt5.account_info()

            if acc:
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                eq = float(getattr(acc, "equity", 0.0) or 0.0)
                mf = float(getattr(acc, "margin_free", 0.0) or 0.0)

                if bal > 0:
                    self._acc_cache.balance = bal
                if eq > 0:
                    self._acc_cache.equity = eq
                if mf > 0:
                    self._acc_cache.margin_free = mf

                self._acc_cache.ts = now

        except Exception as exc:
            log_risk.error("_account_snapshot error: %s | tb=%s", exc, traceback.format_exc())

        return float(self._acc_cache.balance), float(self._acc_cache.equity)

    def _bot_account_snapshot(self) -> Tuple[float, float]:
        cache_sec = float(getattr(self.cfg, "account_snapshot_cache_sec", 0.5) or 0.5)
        now = time.time()

        if (now - float(self._acc_cache.ts)) < cache_sec:
            return float(self._acc_cache.balance), float(self._acc_cache.equity)

        if not self._ensure_ready():
            return float(self._acc_cache.balance), float(self._acc_cache.equity)

        try:
            magic = int(getattr(self.cfg, "magic", 777001) or 777001)
        except Exception:
            magic = 777001

        try:
            with MT5_LOCK:
                acc = mt5.account_info()
                positions = mt5.positions_get()
                day_start = self._utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
                deals = mt5.history_deals_get(day_start, self._utc_now())

            base = float(self._bot_balance_base)
            if base <= 0 and acc:
                base = float(getattr(acc, "balance", 0.0) or 0.0)
                if base > 0:
                    self._bot_balance_base = base

            realized = 0.0
            if deals:
                realized = float(
                    sum(float(getattr(d, "profit", 0.0) or 0.0) for d in deals if int(getattr(d, "magic", 0) or 0) == magic)
                )

            open_pnl = 0.0
            if positions:
                open_pnl = float(
                    sum(float(getattr(p, "profit", 0.0) or 0.0) for p in positions if int(getattr(p, "magic", 0) or 0) == magic)
                )

            bal = max(0.0, base + realized)
            eq = max(0.0, bal + open_pnl)

            if bal > 0:
                self._acc_cache.balance = bal
            if eq > 0:
                self._acc_cache.equity = eq
            if acc:
                mf = float(getattr(acc, "margin_free", 0.0) or 0.0)
                if mf > 0:
                    self._acc_cache.margin_free = mf

            self._acc_cache.ts = now

        except Exception as exc:
            log_risk.error("_bot_account_snapshot error: %s | tb=%s", exc, traceback.format_exc())

        return float(self._acc_cache.balance), float(self._acc_cache.equity)

    def _get_balance(self) -> float:
        bal, _ = self._account_snapshot()
        return float(bal)

    def _get_equity(self) -> float:
        _, eq = self._account_snapshot()
        return float(eq)

    def _get_margin_free(self) -> float:
        _ = self._account_snapshot()
        return float(self._acc_cache.margin_free)

    # ------------------- kill switch (strategy governance) -------------------
    def _recent_closed_trade_profits(self, limit: int) -> List[float]:
        try:
            if not self._ensure_ready():
                return []
            days = int(self._kill_history_days)
            end = self._utc_now()
            start = end - timedelta(days=days)
            with MT5_LOCK:
                deals = mt5.history_deals_get(start, end) or ()
            entry_out = int(getattr(mt5, "DEAL_ENTRY_OUT", 1))
            magic = int(getattr(self.cfg, "magic", 777001) or 777001)
            out: List[Tuple[int, float]] = []
            for d in deals:
                if str(getattr(d, "symbol", "")) != self.symbol:
                    continue
                if int(getattr(d, "magic", 0) or 0) != magic:
                    continue
                if int(getattr(d, "entry", -1) or -1) != entry_out:
                    continue
                t_msc = int(getattr(d, "time_msc", 0) or 0)
                profit = float(getattr(d, "profit", 0.0) or 0.0)
                out.append((t_msc, profit))
            out.sort(key=lambda x: x[0])
            profits = [p for _, p in out]
            return profits[-max(0, int(limit)):]
        except Exception as exc:
            log_risk.error("kill_switch history error: %s | tb=%s", exc, traceback.format_exc())
            return []

    def _refresh_kill_switch(self) -> None:
        now = time.time()
        if (now - float(self._kill_last_check_ts)) < float(self._kill_check_interval_sec):
            return
        self._kill_last_check_ts = now

        profits = self._recent_closed_trade_profits(int(self._kill_min_trades))
        prev = self._kill_state.status
        self._kill_state.update(
            profits,
            now,
            min_trades=int(self._kill_min_trades),
            kill_expectancy=float(self._kill_expectancy_threshold),
            kill_winrate=float(self._kill_winrate_threshold),
            cooling_expectancy=float(self._cooling_expectancy_threshold),
            cooling_sec=float(self._cooling_duration_sec),
        )

        if self._kill_state.status != prev:
            log_risk.warning(
                "KILL_SWITCH | status=%s->%s | exp=%.3f winrate=%.1f%% trades=%s",
                prev,
                self._kill_state.status,
                float(self._kill_state.last_expectancy),
                float(self._kill_state.last_winrate) * 100.0,
                int(self._kill_state.last_trades),
            )

    def strategy_status(self) -> str:
        self._refresh_kill_switch()
        return str(self._kill_state.status)


    # ------------------- state evaluation -------------------
    def evaluate_account_state(self) -> None:
        """
        FIXED: deterministic A/B/C logic + real C blocking + peak drawdown protect.
        """
        try:
            if self._utc_date() != self.daily_date:
                self._reset_daily_state()

            bal, eq = self._account_snapshot()
            if bal > 0 and eq > 0:
                self._current_drawdown = max(0.0, float((bal - eq) / bal))

            # Peak equity (daily)
            if eq > 0:
                self.daily_peak_equity = max(self.daily_peak_equity, float(eq))

            # Ensure start balance (BALANCE-based so day starts at 0 return)
            if self.daily_start_balance <= 0 and bal > 0:
                self.daily_start_balance = float(bal)
                self.daily_peak_equity = max(self.daily_peak_equity, self.daily_start_balance)

            if self.daily_start_balance <= 0 or bal <= 0 or eq <= 0:
                return

            # Daily returns:
            # - realized uses BALANCE (stable)
            # - equity uses EQUITY (includes floating) for emergency stops
            daily_realized = float((bal - self.daily_start_balance) / self.daily_start_balance)
            daily_equity = float((eq - self.daily_start_balance) / self.daily_start_balance)

            peak_return = float((self.daily_peak_equity - self.daily_start_balance) / self.daily_start_balance) if self.daily_start_balance > 0 else 0.0
            dd_from_peak = float((self.daily_peak_equity - eq) / max(1e-9, self.daily_peak_equity)) if self.daily_peak_equity > 0 else 0.0

            # thresholds
            target = float(self.cfg.daily_target_pct)
            max_loss = float(self.cfg.max_daily_loss_pct)

            loss_c = float(getattr(self.cfg, "daily_loss_c_pct", max_loss) or max_loss)
            loss_c = max(0.0, min(loss_c, max_loss))
            loss_b = float(getattr(self.cfg, "daily_loss_b_pct", loss_c * 0.5) or (loss_c * 0.5))
            loss_b = max(0.0, min(loss_b, loss_c))

            protect_peak_dd = float(getattr(self.cfg, "protect_drawdown_from_peak_pct", 0.0) or 0.0)

            # If already soft-stopped today, keep it until time passes
            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                self._set_phase("C", self._soft_stop_reason or "soft_stop_active")
                return
            if self._soft_stop_until > 0 and time.time() >= self._soft_stop_until:
                self._soft_stop_until = 0.0
                self._soft_stop_reason = None
                # Do not auto-downgrade phase: reset occurs next day

            # PROFIT TARGET -> Phase B (realized)
            if daily_realized >= target:
                prev_b = str(self.current_phase)
                if not self._target_breached:
                    self._target_breached = True
                    self._target_breached_return = float(daily_realized)
                self._target_breached_return = max(float(self._target_breached_return), float(daily_realized))
                self._set_phase("B", "daily_target_reached")
                if prev_b != self.current_phase:
                    log_risk.error(
                        "PHASE_CHANGE | %s->%s | reason=daily_target_reached daily_return=%.4f target=%.4f",
                        prev_b, self.current_phase, daily_realized, target,
                    )

            # =================================================================
            # TIERED DAILY DRAWDOWN LOGIC (Aggressive "Money Maker" Mode)
            # Old: -3% -> B, -5% -> C (Too conservative)
            # New: -15% -> B, -40% -> C (Aggressive)
            # =================================================================
            PHASE_B_DRAWDOWN_TRIGGER = -0.15  # -15% triggers defensive mode
            HARD_STOP_DRAWDOWN_TRIGGER = -0.40  # -40% triggers hard stop

            # Trigger 1: -15% Loss -> Phase B (Defensive Mode)
            if daily_equity <= PHASE_B_DRAWDOWN_TRIGGER and self.current_phase == "A":
                prev = str(self.current_phase)
                self._set_phase("B", "drawdown_defensive_15pct")
                log_risk.warning(
                    "PHASE_B_ACTIVATED | daily_loss=%.2f%% EXCEEDS 15%% | phase=%s->B | DEFENSIVE MODE",
                    abs(daily_equity) * 100,
                    prev,
                )

            # Trigger 2: -40% Loss -> HARD STOP (Kill Switch)
            if daily_equity <= HARD_STOP_DRAWDOWN_TRIGGER and self.current_phase != "C":
                prev = str(self.current_phase)
                self._enter_hard_stop("emergency_40pct_hard_stop")
                log_risk.error(
                    "HARD_STOP_ACTIVATED | daily_loss=%.2f%% EXCEEDS 40%% LIMIT | phase=%s->C",
                    abs(daily_equity) * 100,
                    prev,
                )

            # PROFIT PROTECT: after hitting target/peak, if equity falls from peak too much -> stop trading
            if protect_peak_dd > 0:
                if self._target_breached or peak_return >= target:
                    if dd_from_peak >= protect_peak_dd:
                        self._enter_soft_stop("profit_protect_peak_dd")

        except Exception as exc:
            log_risk.error("evaluate_account_state error: %s | tb=%s", exc, traceback.format_exc())

    def update_phase(self) -> None:
        self.evaluate_account_state()

    # ------------------- latency -------------------
    def latency_cooldown(self) -> bool:
        try:
            if not self._latency_bad_since:
                return True
            return (time.time() - self._latency_bad_since) >= float(self.cfg.cooldown_after_latency_s)
        except Exception as exc:
            log_risk.error("latency_cooldown error: %s | tb=%s", exc, traceback.format_exc())
            return True

    def register_latency_violation(self) -> None:
        self._latency_bad_since = time.time()

    # ------------------- ATR percentile (analytics) -------------------
    def atr_percentile(self, dfp: pd.DataFrame, lookback: int) -> float:
        try:
            h = dfp["high"].values.astype(np.float64)
            l = dfp["low"].values.astype(np.float64)
            c = dfp["close"].values.astype(np.float64)

            period = int(getattr(self.cfg, "atr_period_for_percentile", 14) or 14)
            if talib is not None:
                atr_series = talib.ATR(h, l, c, period)  # type: ignore[attr-defined]
            else:
                atr_series = _atr_fallback(h, l, c, period)

            atr_rel_series = atr_series / np.maximum(1e-9, c)

            look = max(30, min(len(atr_rel_series) - 2, int(lookback)))
            return percentile_rank(atr_rel_series[-look - 2 : -2], float(atr_rel_series[-2]))
        except Exception as exc:
            log_risk.error("atr_percentile error: %s | tb=%s", exc, traceback.format_exc())
            return 50.0

    # ------------------- guards -------------------
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
        tz: Any,
    ) -> RiskDecision:
        try:
            _ = (ingest_ms, last_bar_age, tz)

            self.evaluate_account_state()
            reasons: List[str] = []

            if self.requires_hard_stop():
                reasons.append(f"risk:{self.hard_stop_reason() or 'hard_stop'}")

            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                reasons.append(f"risk:{self._soft_stop_reason or 'soft_stop'}")

            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append(f"risk:{self._phase_reason_last or 'phase_c'}")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)) and drawdown_exceeded:
                reasons.append("risk:max_drawdown")

            if spread_pct > float(self.sp.spread_limit_pct):
                reasons.append("risk:spread_high")

            ignore_micro = bool(getattr(self.cfg, "ignore_microstructure", False))
            if not ignore_micro:
                if not tick_ok and tick_reason not in ("tps_low", "no_rates"):
                    reasons.append(f"risk:micro_{tick_reason}")

            if not bool(getattr(self.cfg, "ignore_sessions", False)):
                if not in_session:
                    ok, r = self._trade_mode_state()
                    reasons.append(f"risk:{r or ('maintenance_blackout' if not ok else 'session_blocked')}")

            if self.rollover_blackout():
                reasons.append("risk:rollover_blackout")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                    if spread_pct > float(self.sp.spread_limit_pct) * 1.2:
                        reasons.append("risk:dd_strict_spread")

            max_per_day = int(getattr(self.cfg, "max_signals_per_day", 0) or 0)
            if max_per_day > 0:
                per_hour = max(1, int(float(max_per_day) / 24.0))
                now = time.time()
                if (now - self._signal_throttle.hour_window_start_ts) >= 3600.0:
                    self._signal_throttle.hour_window_start_ts = now
                    self._signal_throttle.hour_window_count = 0
                if self._signal_throttle.hour_window_count >= per_hour:
                    reasons.append("risk:hourly_signal_limit")

            if spread_pct > float(self.sp.spread_limit_pct) * 1.5:
                reasons.append("risk:excessive_spread_pct")

            if not latency_cooldown:
                reasons.append("risk:latency_cooldown")

            return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)
        except Exception as exc:
            log_risk.error("guard_decision error: %s | tb=%s", exc, traceback.format_exc())
            return RiskDecision(allowed=False, reasons=["risk:guard_exception"])

    def can_trade(self, confidence: float, signal_type: str) -> RiskDecision:
        try:
            _ = signal_type

            reasons: List[str] = []
            self.evaluate_account_state()

            self._refresh_kill_switch()
            if self._kill_state.status == "KILLED":
                reasons.append("kill_switch_killed")
                return RiskDecision(False, reasons)
            if self._kill_state.status == "COOLING" and time.time() < float(self._kill_state.cooling_until_ts):
                reasons.append("kill_switch_cooling")
                return RiskDecision(False, reasons)

            if self.requires_hard_stop():
                reasons.append(f"risk:{self.hard_stop_reason() or 'hard_stop'}")
                return RiskDecision(False, reasons)

            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                reasons.append(f"risk:{self._soft_stop_reason or 'soft_stop'}")
                return RiskDecision(False, reasons)

            # Phase C MUST block trading (unless explicitly ignored)
            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append(f"risk:{self._phase_reason_last or 'phase_c'}")
                return RiskDecision(False, reasons)

            if not self.market_open_24_5():
                ok, r = self._trade_mode_state()
                reasons.append(f"risk:{r or 'maintenance_blackout'}")
                return RiskDecision(False, reasons)

            if self.rollover_blackout():
                reasons.append("risk:rollover_blackout")

            if float(confidence) < float(self.cfg.min_confidence_signal):
                reasons.append(f"risk:low_confidence:{confidence:.2f}<{self.cfg.min_confidence_signal}")

            if self.current_phase == "B" and float(confidence) < float(self.cfg.ultra_confidence_min):
                reasons.append(f"risk:not_ultra_conf_phase_b:{confidence:.2f}<{self.cfg.ultra_confidence_min}")

            if not self.can_analyze():
                reasons.append("risk:analysis_cooldown")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown):
                    reasons.append(f"risk:high_drawdown:{self._current_drawdown:.2%}")

            if self._last_fill_ts and (time.time() - self._last_fill_ts) < float(self.cfg.cooldown_seconds):
                reasons.append("risk:cooldown_period")

            if not self.latency_cooldown():
                reasons.append("risk:latency")

            max_spread_points = getattr(self.cfg, "max_spread_points", None)
            if max_spread_points is not None:
                sp_pts = self.broker.current_spread_points()
                if sp_pts > float(max_spread_points):
                    reasons.append("spread_points_cap")

            # SENIOR FIX: Enforce max_positions limit to prevent spamming orders
            max_pos = int(getattr(self.cfg, "max_positions", 3) or 3)
            # _last_open_positions is updated via on_reconcile_positions from engine
            if int(self._last_open_positions) >= max_pos:
                reasons.append(f"max_positions_reached:{self._last_open_positions}>={max_pos}")

            if time.time() < self._exec_breaker_until:
                reasons.append("exec_breaker_active")
                return RiskDecision(False, reasons)

            anoms = self.execmon.anomaly_reasons(cfg=self.cfg)
            if anoms:
                sec = float(getattr(self.cfg, "exec_breaker_sec", 120.0) or 120.0)
                self._exec_breaker_until = time.time() + sec
                reasons.extend(anoms)
                reasons.append("exec_breaker_triggered")
                return RiskDecision(False, reasons)

            return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)

        except Exception as exc:
            log_risk.error("can_trade error: %s | tb=%s", exc, traceback.format_exc())
            return RiskDecision(allowed=False, reasons=["can_trade_error"])

    def can_emit_signal(self, confidence: int, tz: Any) -> Tuple[bool, List[str]]:
        try:
            _ = tz
            self.evaluate_account_state()

            self._refresh_kill_switch()
            if self._kill_state.status == "KILLED":
                reasons.append("kill_switch_killed")
                return False, reasons
            if self._kill_state.status == "COOLING" and time.time() < float(self._kill_state.cooling_until_ts):
                reasons.append("kill_switch_cooling")
                return False, reasons

            reasons: List[str] = []
            conf_f = float(confidence)
            if conf_f > 1.5:
                conf_f = conf_f / 100.0

            if self.requires_hard_stop():
                reasons.append(self.hard_stop_reason() or "hard_stop")
                return False, reasons

            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                reasons.append(self._soft_stop_reason or "soft_stop")
                return False, reasons

            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append(self._phase_reason_last or "phase_c")
                return False, reasons

            if conf_f < float(self.cfg.min_confidence_signal):
                reasons.append("low_confidence")

            if not self.can_analyze():
                reasons.append("analysis_cooldown")

            if self.current_phase == "B" and conf_f < float(self.cfg.ultra_confidence_min):
                reasons.append("not_ultra_confidence")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                    if conf_f < float(self.cfg.min_confidence_signal) + 0.1:
                        reasons.append("drawdown_strict_confidence")

            if time.time() < self._exec_breaker_until:
                reasons.append("exec_breaker_active")

            if not self.market_open_24_5():
                ok, r = self._trade_mode_state()
                reasons.append(r or "maintenance_blackout")

            allowed = (len(reasons) == 0)
            return allowed, reasons
        except Exception as exc:
            log_risk.error("can_emit_signal error: %s | tb=%s", exc, traceback.format_exc())
            return False, ["signal_error"]

    # ------------------- broker-true sizing (critical) -------------------
    def _apply_broker_constraints(self, side: str, entry: float, sl: float, tp: float) -> Tuple[float, float]:
        if not self.broker.ensure_ready():
            return sl, tp

        side_n = _side_norm(side)
        min_dist = self.broker.min_stop_distance()

        if min_dist > 0:
            if side_n == "Buy":
                sl = min(sl, entry - min_dist)
                tp = max(tp, entry + min_dist)
            else:
                sl = max(sl, entry + min_dist)
                tp = min(tp, entry - min_dist)

        return self.broker.normalize_price(sl), self.broker.normalize_price(tp)

    @staticmethod
    def _rr(entry: float, sl: float, tp: float) -> float:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        return float(reward / max(1e-9, risk))

    def calculate_position_size(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
    ) -> float:
        """
        Adaptive lot sizing:
          Lot = (Equity * BaseRisk% * Confidence * PhaseFactor * DrawdownFactor) / Risk_1lot
        """
        try:
            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return 0.0

            if not _is_finite(entry_price, stop_loss, take_profit, confidence):
                return 0.0

            if not self.broker.ensure_ready():
                return 0.0

            min_rr = getattr(self.cfg, "min_rr", None)
            if min_rr is not None:
                rr_ratio = self._rr(entry_price, stop_loss, take_profit)
                rr_min = max(float(min_rr), 1.5)
                if rr_ratio < rr_min:
                    return 0.0

            self.evaluate_account_state()
            _, eq = self._account_snapshot()
            if eq <= 0:
                return 0.0

            conf = float(confidence)
            if conf > 1.5:
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))

            # Base risk budget scaled by confidence, phase, and drawdown protection
            risk_money = adaptive_risk_money(
                float(eq),
                float(self.cfg.max_risk_per_trade),
                conf,
                str(self.current_phase or "A"),
                float(self._current_drawdown),
                phase_factors={"A": 1.2, "B": 0.8, "C": 0.5},
                dd_cut=0.10,
                dd_mult=0.5,
            )

            if risk_money <= 0:
                return 0.0

            risk_1lot = abs(self.broker.calc_profit_money(side_n, 1.0, entry_price, stop_loss))
            if risk_1lot <= 0:
                return 0.0

            raw_lot = float(risk_money) / float(risk_1lot)

            if hasattr(self.cfg, "max_position_percentage"):
                cap_pct = float(getattr(self.cfg, "max_position_percentage")) / 100.0
                if cap_pct > 0:
                    margin_cap = eq * cap_pct
                    margin_1lot = self.broker.calc_margin(side_n, 1.0, entry_price)
                    if margin_1lot > 0:
                        raw_lot = min(raw_lot, margin_cap / margin_1lot)

            margin_free = self._get_margin_free()
            mf_mult = float(getattr(self.cfg, "margin_free_safety_mult", 1.25) or 1.25)
            if margin_free > 0 and mf_mult > 1.0:
                margin_1lot = self.broker.calc_margin(side_n, 1.0, entry_price)
                if margin_1lot > 0:
                    cap_by_mf = (margin_free / mf_mult) / margin_1lot
                    raw_lot = min(raw_lot, cap_by_mf)

            lot = self.broker.normalize_volume_floor(raw_lot)
            max_lot = float(getattr(self.cfg, "max_lot", 0.20) or 0.20)
            if max_lot > 0:
                lot = min(float(lot), float(max_lot))
            return float(lot) if lot > 0 else 0.0
        except Exception as exc:
            log_risk.error("calculate_position_size error: %s | tb=%s", exc, traceback.format_exc())
            return 0.0

    # ------------------- SL/TP planners
    # ------------------- SL/TP planners -------------------
    def _micro_zone_sl_tp(
        self,
        side: str,
        entry: float,
        zones: Optional[MicroZones],
        tick_volatility: float,
    ) -> Tuple[float, float]:
        try:
            if not _is_finite(entry) or entry <= 0:
                return entry, entry

            base_pct = float(getattr(self.cfg, "micro_buffer_pct", 0.0008) or 0.0008)
            base_buffer = entry * base_pct

            tv = float(max(0.0, tick_volatility))
            if self.broker.ensure_ready() and tv > entry * 0.02:
                tv = tv * float(self.broker.point_value())

            vol_mult = float(getattr(self.cfg, "micro_vol_mult", 2.0) or 2.0)
            vol_buffer = tv * vol_mult

            micro_buffer = float(base_buffer + vol_buffer)

            side_n = _side_norm(side)

            rr_micro = float(getattr(self.cfg, "micro_rr", 2.2) or 2.2)
            rr_floor = float(getattr(self.cfg, "min_rr", 1.0) or 1.0)

            if side_n == "Buy":
                sl = entry - micro_buffer
                tp_target = entry + micro_buffer * rr_micro
                tp = tp_target

                if zones:
                    levels = [
                        float(getattr(zones, "strong_resistance", 0.0) or 0.0),
                        float(getattr(zones, "value_area_high", 0.0) or 0.0),
                        float(getattr(zones, "poc", 0.0) or 0.0),
                    ]
                    levels = [x for x in levels if x > entry]
                    if levels:
                        nearest = min(levels)
                        if self._rr(entry, sl, nearest) >= rr_floor:
                            tp = nearest

            else:
                sl = entry + micro_buffer
                tp_target = entry - micro_buffer * rr_micro
                tp = tp_target

                if zones:
                    levels = [
                        float(getattr(zones, "strong_support", 0.0) or 0.0),
                        float(getattr(zones, "value_area_low", 0.0) or 0.0),
                        float(getattr(zones, "poc", 0.0) or 0.0),
                    ]
                    levels = [x for x in levels if 0.0 < x < entry]
                    if levels:
                        nearest = max(levels)
                        if self._rr(entry, sl, nearest) >= rr_floor:
                            tp = nearest

            return float(sl), float(tp)

        except Exception as exc:
            log_risk.error("_micro_zone_sl_tp error: %s | tb=%s", exc, traceback.format_exc())
            return entry, entry

    def _fallback_atr_sl_tp(self, side: str, entry: float, adapt: Dict[str, Any]) -> Tuple[float, float]:
        try:
            atr = float(adapt.get("atr", 0.0) or 0.0)
            
            # SAFE FALLBACK: If ATR is missing/0, use 0.5% default distance
            if atr <= 1e-9:
                if hasattr(self.cfg, "fallback_atr_abs") and float(getattr(self.cfg, "fallback_atr_abs") or 0.0) > 0:
                     atr = float(getattr(self.cfg, "fallback_atr_abs"))
                else:
                    atr = entry * 0.005  # 0.5% fallback

            if atr <= 1e-9:
                # Should not happen unless entry is 0
                return entry, entry

            regime = str(adapt.get("regime", "trend") or "trend")
            if regime == "range":
                sl_mult = float(self.cfg.sl_atr_mult_range)
                tp_mult = float(self.cfg.tp_atr_mult_range)
            else:
                sl_mult = float(self.cfg.sl_atr_mult_trend)
                tp_mult = float(self.cfg.tp_atr_mult_trend)

            side_n = _side_norm(side)
            if side_n == "Buy":
                return float(entry - atr * sl_mult), float(entry + atr * tp_mult)
            return float(entry + atr * sl_mult), float(entry - atr * tp_mult)
        except Exception as exc:
            log_risk.error("_fallback_atr_sl_tp error: %s | tb=%s", exc, traceback.format_exc())
            return entry, entry

    def _calculate_structure_sl(self, side: str, entry: float, df: Optional[pd.DataFrame]) -> Tuple[float, str]:
        """
        Finds the recent Swing Low (Buy) or Swing High (Sell) to use as a structural Stop Loss.
        Logic: Look back 20 candles.
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return entry, "no_structure"

            # Lookback period for swing
            lookback = 20
            # Use High/Low from the last N candles
            recent_data = df.tail(lookback)

            if side == "Buy":
                # For Buy, SL is below the lowest low
                swing_low = float(recent_data["low"].min())
                if swing_low > 0 and swing_low < entry:
                    return swing_low, "swing_low"
                else:
                    return entry, "invalid_swing_low"

            else: # Sell
                # For Sell, SL is above the highest high
                swing_high = float(recent_data["high"].max())
                if swing_high > 0 and swing_high > entry:
                    return swing_high, "swing_high"
                else:
                    return entry, "invalid_swing_high"
        
        except Exception as exc:
            log_risk.error("_calculate_structure_sl error: %s", exc)
            return entry, "structure_error"

    # ------------------- trailing stop / breakeven -------------------
    def check_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        side: str,
        atr: float
    ) -> Optional[float]:
        """
        Logic:
        1. Breakeven: If Profit >= 40% of MaxProfit -> Move SL to Entry.
        2. Trailing: After BE, trail by 1.5 * ATR.
        """
        try:
            side_n = _side_norm(side)
            dist_total = abs(tp_price - entry_price)
            if dist_total <= 1e-9:
                return None
                
            current_dist = 0.0
            if side_n == "Buy":
                current_dist = current_price - entry_price
            else:
                current_dist = entry_price - current_price
                
            # 1. Breakeven Check (40% to TP)
            progress = current_dist / dist_total
            new_sl = None
            
            # Breakeven Level
            if progress >= 0.40:
                # Target SL = Entry (plus small buffer maybe? strict Entry for now)
                be_price = entry_price
                
                # Check if we can improve current SL
                if side_n == "Buy":
                    if sl_price < be_price:
                        new_sl = be_price
                else:
                    if sl_price > be_price:
                        new_sl = be_price
                        
            # 2. Trailing Stop (1.5x ATR) if we are in profit
            # Only trail if we are profitable (e.g. past BE or significantly up)
            if progress > 0.40: 
                trail_dist = atr * 1.5
                if side_n == "Buy":
                    trail_target = current_price - trail_dist
                    # Only move UP
                    current_sl_to_use = new_sl if new_sl is not None else sl_price
                    if trail_target > current_sl_to_use:
                        new_sl = trail_target
                else:
                    trail_target = current_price + trail_dist
                    # Only move DOWN
                    current_sl_to_use = new_sl if new_sl is not None else sl_price
                    if trail_target < current_sl_to_use:
                        new_sl = trail_target
                        
            return new_sl
            
        except Exception as exc:
            log_risk.error("check_trailing_stop error: %s", exc)
            return None

    def plan_order(
        self,
        side: str,
        confidence: float,
        ind: Dict[str, Any],
        adapt: Dict[str, Any],
        *,
        entry: Optional[float] = None,
        ticks: Optional[TickStats] = None,
        zones: Optional[MicroZones] = None,
        tick_volatility: float = 0.0,
        open_positions: int = 0,
        max_positions: int = 0,
        unrealized_pl: float = 0.0,
        allow_when_blocked: bool = False,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
        try:
            _ = (ind, ticks)

            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return None, None, None, None, f"invalid_side_{side}"
            
            if not self._ensure_ready():
                return None, None, None, None, "mt5_not_ready"

            # Real phase C block
            self.evaluate_account_state()
            if (not allow_when_blocked) and (
                (self._soft_stop_until > 0 and time.time() < self._soft_stop_until)
                or (self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)))
            ):
                return None, None, None, None, f"blocked_{self.current_phase}_{self._phase_reason_last}"

            ok, reason = self._trade_mode_state()
            if not ok:
                return None, None, None, None, f"trade_mode_{reason}"
            
            if self._in_maintenance_blackout():
                return None, None, None, None, "maintenance_blackout"

            if entry is None:
                with MT5_LOCK:
                    t = mt5.symbol_info_tick(self.symbol)
                if not t:
                    return None, None, None, None, "no_tick"
                entry = float(t.ask) if side_n == "Buy" else float(t.bid)

            entry_price = float(entry)
            if not _is_finite(entry_price) or entry_price <= 0:
                log_risk.warning(f"plan_order blocked: invalid entry_price={entry_price}")
                return None, None, None, None, f"invalid_entry_{entry_price}"

            # =================================================================
            # NEW "SENIOR" SL LOGIC: Volatility-Adjusted + Structure (BTC)
            # =================================================================

            # 1. Get ATR
            atr = 0.0
            try:
                atr = float((adapt or {}).get("atr", 0.0) or ind.get("atr", 0.0) or 0.0)
            except Exception:
                atr = 0.0
            
            if atr <= 1e-9:
                atr = max(abs(entry_price - float(sl or 0)), entry_price * 0.005) # 0.5% fallback for BTC

            # =================================================================
            # SNIPER MODE: KILL ZONE FILTER
            # =================================================================
            # Reject if market is dead (low volatility)
            # BTC needs larger threshold than Gold due to scale
            min_vol_threshold = entry_price * 0.00025  # 0.025% move required for BTC
            if tick_volatility < min_vol_threshold and atr < (entry_price * 0.001):
                log_risk.info(f"SNIPER_REJECT | {side_n} | Low Volatility: TickVol={tick_volatility:.2f} ATR={atr:.2f}")
                return None, None, None, None, "kill_zone_low_volatility"

            # Reject if we are in "Noise" (price too close to recent structure loops)
            # (Simplified: We rely on Spread/ATR ratio)
            spread = self.broker.current_spread_points() * float(self.broker.point_value())
            if spread > 0 and (atr / spread) < 2.0:
                 log_risk.info(f"SNIPER_REJECT | {side_n} | High Spread/Noise Ratio: ATR/Spread = {atr/spread:.2f}")
                 return None, None, None, None, "kill_zone_high_noise"
            # =================================================================

            # 2. Calculate ATR-Based SL (Base 2.5x for Volatility Survival)
            # CRITICAL OPTIMIZATION: Wide Base SL to survive noise
            sl_mult_atr = 2.5
            atr_sl_dist = atr * sl_mult_atr
            
            if side_n == "Buy":
                atr_sl_price = entry_price - atr_sl_dist
            else:
                atr_sl_price = entry_price + atr_sl_dist

            # 3. Structure SL
            struct_sl_price, struct_reason = self._calculate_structure_sl(side_n, entry_price, df)
            
            # 4. Final SL (Structure Priority + Buffer)
            # The user wants SL behind Swings + Buffer.
            sl_buffer = atr * 0.5  # 0.5 ATR buffer behind the swing
            final_sl = atr_sl_price
            sl_source = "ATR_2.5x"

            if side_n == "Buy":
                if struct_reason == "swing_low" and struct_sl_price < entry_price:
                    # Use structure low -> subtract buffer
                    final_sl = struct_sl_price - sl_buffer
                    sl_source = "Structure_Low_Buffered"
                else:
                    # No structure? Use Wide ATR (Sniper patience: maybe reject? For now use wide ATR)
                    final_sl = entry_price - (atr * 2.5)
                    sl_source = "ATR_Fallback_Wide"
            else:
                if struct_reason == "swing_high" and struct_sl_price > entry_price:
                    # Use structure high -> add buffer
                    final_sl = struct_sl_price + sl_buffer
                    sl_source = "Structure_High_Buffered"
                else:
                    final_sl = entry_price + (atr * 2.5)
                    sl_source = "ATR_Fallback_Wide"

            # 5. Safety Check (Min Distance)
            min_dist_pct = 0.002
            min_safety_dist = entry_price * min_dist_pct
            real_dist = abs(entry_price - final_sl)
            
            if real_dist < min_safety_dist:
                log_risk.info(f"SNIPER_ADJUST | SL too close ({real_dist:.2f}). Widening to {min_safety_dist:.2f}")
                if side_n == "Buy":
                    final_sl = entry_price - min_safety_dist
                else:
                    final_sl = entry_price + min_safety_dist
                sl_source += "_ForceWiden"

            sl = self.broker.normalize_price(final_sl)

            log_risk.info(
                f"SNIPER_SL | Symbol: {self.symbol} | Side: {side_n} | Source: {sl_source} | "
                f"Entry: {entry_price:.2f} | SL: {sl:.2f} | Dist: {abs(entry_price - sl):.2f}"
            )

            # --- Restore Broker Min Limits (deleted in previous step) ---
            min_dist = self.broker.min_stop_distance()
            min_sl_pct = float(getattr(self.cfg, "min_sl_pct", 0.0008) or 0.0008)
            min_tp_pct = float(getattr(self.cfg, "min_tp_pct", 0.0012) or 0.0012)

            min_sl = max(min_dist, entry_price * min_sl_pct)
            min_tp = max(min_dist, entry_price * min_tp_pct)

            # Enforce minimum SL distance
            if side_n == "Buy":
                if abs(entry_price - float(sl)) < min_sl:
                    sl = self.broker.normalize_price(entry_price - min_sl)
            else:
                if abs(entry_price - float(sl)) < min_sl:
                    sl = self.broker.normalize_price(entry_price + min_sl)

            # CRITICAL OPTIMIZATION: Dynamic TP (Scalp vs Swing)
            # Conf < 85 -> Scalp Mode (1.5x Risk)
            # Conf >= 85 -> Swing Mode (2.5x Risk)
            dist_sl_p = abs(entry_price - atr_sl_price)
            if float(confidence) >= 85:
                 # Swing Mode
                 tp_dist = dist_sl_p * 2.5
            else:
                 # Scalp Mode
                 tp_dist = dist_sl_p * 1.5
             
            if side_n == "Buy":
                 tp = entry_price + tp_dist
            else:
                 tp = entry_price - tp_dist

            # Enforce minimum TP distance
            if side_n == "Buy":
                if abs(float(tp) - entry_price) < min_tp:
                    tp = entry_price + min_tp
            else:
                if abs(float(tp) - entry_price) < min_tp:
                    tp = entry_price - min_tp

            if not _is_finite(sl, tp) or sl == entry_price or tp == entry_price:
                log_risk.info("plan_order blocked: invalid_sl_tp sl=%s tp=%s entry=%s", sl, tp, entry_price)
                return None, None, None, None, f"invalid_sl_tp_calc:sl={sl},tp={tp}"

            sl, tp = self._apply_broker_constraints(side_n, entry_price, float(sl), float(tp))

            if side_n == "Buy" and not (sl < entry_price < tp):
                log_risk.info("plan_order blocked: bad_buy_limits sl=%s entry=%s tp=%s", sl, entry_price, tp)
                return None, None, None, None, f"bad_buy_limits:sl={sl},en={entry_price},tp={tp}"
            if side_n == "Sell" and not (tp < entry_price < sl):
                log_risk.info("plan_order blocked: bad_sell_limits tp=%s entry=%s sl=%s", tp, entry_price, sl)
                return None, None, None, None, f"bad_sell_limits:tp={tp},en={entry_price},sl={sl}"

            # Enforce SNIPER RR >= 2.0
            rr_cfg = float(getattr(self.cfg, "min_rr", 2.0) or 2.0)
            rr_min = max(rr_cfg, 2.0) # FORCE 2.0
            dist_sl = abs(entry_price - float(sl))
            if dist_sl > 0:
                dist_tp = abs(float(tp) - entry_price)
                if dist_tp < dist_sl * rr_min:
                    tp = self.broker.normalize_price(
                        entry_price + dist_sl * rr_min if side_n == "Buy" else entry_price - dist_sl * rr_min
                    )

            # RR cap (do not drop below rr_min)
            rr_cap = float(getattr(self.cfg, "tp_rr_cap", 0.0) or 0.0)
            if rr_cap > 0:
                rr_cap = max(rr_cap, rr_min)
                dist_tp = abs(float(tp) - entry_price)
                if dist_sl > 0 and dist_tp > dist_sl * rr_cap:
                    tp = self.broker.normalize_price(
                        entry_price + dist_sl * rr_cap if side_n == "Buy" else entry_price - dist_sl * rr_cap
                    )
                    if abs(float(tp) - entry_price) < min_tp:
                        tp = self.broker.normalize_price(
                            entry_price + min_tp if side_n == "Buy" else entry_price - min_tp
                        )

            sl, tp = self._apply_broker_constraints(side_n, entry_price, float(sl), float(tp))

            if side_n == "Buy" and not (sl < entry_price < tp):
                return None, None, None, None, f"bad_buy_limits:sl={sl},en={entry_price},tp={tp}"
            if side_n == "Sell" and not (tp < entry_price < sl):
                return None, None, None, None, f"bad_sell_limits:tp={tp},en={entry_price},sl={sl}"

            lot = self.calculate_position_size(side_n, entry_price, sl, tp, float(confidence))
            if lot <= 0 or not _is_finite(lot):
                log_risk.info("plan_order blocked: zero_lot lot=%s conf=%.2f", lot, confidence)
                return None, None, None, None, f"zero_lot_calc:conf={confidence}"

            # --- STRICT MAX LOT ENFORCEMENT ---
            HARD_MAX_LOT = 0.20
            cfg_max_lot = float(getattr(self.cfg, "max_lot", 0.20) or 0.20)
            effective_max_lot = min(HARD_MAX_LOT, cfg_max_lot)
            
            if lot > effective_max_lot:
                lot = effective_max_lot
            
            lot = self.broker.normalize_volume_floor(float(lot))
            # ----------------------------------

            if (
                int(open_positions) > 0
                and int(open_positions) < int(max_positions)
                and float(unrealized_pl) >= 0.0
                and float(confidence) >= float(self.cfg.ultra_confidence_min)
            ):
                scale_factor = max(0.3, 1.0 - (int(open_positions) / max(1, int(max_positions))))
                lot = self.broker.normalize_volume_floor(lot * scale_factor)

                tp_bonus = 1.0 + float(getattr(self.cfg, "multi_order_tp_bonus_pct", 0.12) or 0.12)

                if side_n == "Buy":
                    tp = self.broker.normalize_price(entry_price + abs(tp - entry_price) * tp_bonus)
                else:
                    tp = self.broker.normalize_price(entry_price - abs(tp - entry_price) * tp_bonus)

                sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

            return float(entry_price), float(sl), float(tp), float(lot), None

        except Exception as exc:
            log_risk.error("plan_order error: %s | tb=%s", exc, traceback.format_exc())
            return None, None, None, None, f"plan_exception_{exc}"

    # ------------------- trade recording (engine hooks) -------------------
    # ------------------- trade recording (engine hooks) -------------------
    def record_trade(self, *args, **kwargs) -> None:
        try:
            positions_open: int
            if self._ensure_ready():
                with MT5_LOCK:
                    pos = mt5.positions_get(symbol=self.symbol) or ()
                positions_open = int(len(pos))
            else:
                positions_open = 0
                if "positions_open" in kwargs:
                    positions_open = int(kwargs.get("positions_open", 0) or 0)
                elif len(args) >= 5:
                    positions_open = int(args[4] or 0)

            if positions_open > 0:
                self.on_position_opened()
            else:
                self.on_all_flat()

            self._last_fill_ts = time.time()
            self.evaluate_account_state()
        except Exception as exc:
            log_risk.error("record_trade error: %s | tb=%s", exc, traceback.format_exc())

    def update_market_conditions(self, spread: float, slippage: float) -> None:
        try:
            _ = float(spread)
            _ = float(slippage)
        except Exception as exc:
            log_risk.error("update_market_conditions error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- execution metrics (analytics + breaker input) -------------------

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
        """
        Backward-compatible:
        - Old engine might call positional without 'side' (shifted args).
        - We detect numeric 'side' and remap safely.

        Always produces:
          slippage_points = abs(filled - expected) / point
        """
        try:
            # --------- remap old positional call ----------
            side_n: str
            if isinstance(side, (int, float)) and isinstance(enqueue_time, (int, float)):
                # old call style (shifted):
                # record_execution_metrics(order_id, enqueue, send, fill, expected, filled, slippage)
                old_enqueue = float(side)
                old_send = float(enqueue_time)
                old_fill = float(send_time)
                old_expected = float(fill_time)
                old_filled = float(expected_price)
                old_slip = float(filled_price)  # last positional was slippage

                enqueue_time = old_enqueue
                send_time = old_send
                fill_time = old_fill
                expected_price = old_expected
                filled_price = old_filled
                slippage = old_slip
                side_n = "Buy"  # side unknown in old call; keep deterministic default
            else:
                side_n = _side_norm(side)

            # --------- compute robust slippage points ----------
            pt = float(self.broker.point_value() or 0.0)
            if pt > 0 and _is_finite(float(expected_price), float(filled_price)):
                slip_abs = abs(float(filled_price) - float(expected_price))
                slip_points = slip_abs / pt
            else:
                # fallback
                slip_abs = abs(float(slippage))
                slip_points = float(slip_abs)

            metrics = {
                "order_id": str(order_id),
                "timestamp": self._utc_now().isoformat(),
                "side": side_n,
                "enqueue_time": float(enqueue_time),
                "send_time": float(send_time),
                "fill_time": float(fill_time),
                "expected_price": float(expected_price),
                "filled_price": float(filled_price),
                "slippage": float(slip_abs),
                "slippage_points": float(slip_points),
                "latency_enqueue_to_send_ms": (float(send_time) - float(enqueue_time)) * 1000.0,
                "latency_send_to_fill_ms": (float(fill_time) - float(send_time)) * 1000.0,
                "total_latency_ms": (float(fill_time) - float(enqueue_time)) * 1000.0,
            }

            # Optional: disable heavy logs
            if bool(getattr(self.cfg, "enable_execution_metrics_logs", True)):
                with _FILE_LOCK:
                    csv_file = LOG_DIR / "execution_quality_btc.csv"
                    write_header = not csv_file.exists()
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(metrics)

                    jsonl_file = LOG_DIR / "execution_metrics_btc.jsonl"
                    with open(jsonl_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            # Feed execmon (breaker)
            try:
                if self.broker.ensure_ready():
                    spread_points = float(self.broker.current_spread_points())
                    self.execmon.on_execution(
                        ExecSample(
                            ts=time.time(),
                            side=side_n,
                            expected_price=float(expected_price),
                            filled_price=float(filled_price),
                            point=float(self.broker.point_value()),
                            spread_points=float(spread_points),
                            latency_total_ms=float(metrics["total_latency_ms"]),
                        )
                    )
            except Exception as exc2:
                log_risk.error("execmon feed error: %s | tb=%s", exc2, traceback.format_exc())

            self._update_execution_analysis(metrics)

        except Exception as exc:
            log_risk.error("record_execution_metrics error: %s | tb=%s", exc, traceback.format_exc())


    def record_execution_failure(self, order_id: str, enqueue_time: float, send_time: float, reason: str) -> None:
        try:
            metrics = {
                "order_id": str(order_id),
                "timestamp": self._utc_now().isoformat(),
                "enqueue_time": float(enqueue_time),
                "send_time": float(send_time),
                "reason": str(reason),
                "latency_enqueue_to_send_ms": (float(send_time) - float(enqueue_time)) * 1000.0,
            }

            if bool(getattr(self.cfg, "enable_execution_metrics_logs", True)):
                with _FILE_LOCK:
                    csv_file = LOG_DIR / "execution_failures_btc.csv"
                    write_header = not csv_file.exists()
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(metrics)

            self._update_rejection_analysis(str(reason))
        except Exception as exc:
            log_risk.error("record_execution_failure error: %s | tb=%s", exc, traceback.format_exc())

    def _update_execution_analysis(self, metrics: dict) -> None:
        """
        FAST: update RAM hist; flush periodically to CSV (no per-trade CSV read/write).
        CRITICAL: Always flush on shutdown to prevent data loss.
        """
        try:
            s = float(metrics.get("slippage_points", 0.0))
            d = float(metrics.get("latency_send_to_fill_ms", 0.0))

            # Update slip bins
            for i in range(len(self._slip_bins) - 1):
                if self._slip_bins[i] <= s < self._slip_bins[i + 1]:
                    self._slip_counts[i] += 1
                    break

            # Update delay bins
            for i in range(len(self._delay_bins) - 1):
                if self._delay_bins[i] <= d < self._delay_bins[i + 1]:
                    self._delay_counts[i] += 1
                    break

            # Force flush if metrics dict is empty (shutdown signal)
            force_flush = len(metrics) == 0
            
            flush_sec = float(getattr(self.cfg, "exec_hist_flush_sec", 5.0) or 5.0)
            now = time.time()
            if (not force_flush) and (now - self._hist_last_flush_ts) < flush_sec:
                return
            self._hist_last_flush_ts = now

            if not bool(getattr(self.cfg, "enable_execution_metrics_logs", True)):
                return

            with _FILE_LOCK:
                slippage_file = LOG_DIR / "slippage_histogram_btc.csv"
                slip_rows = []
                for i in range(len(self._slip_bins) - 1):
                    slip_rows.append(
                        {"bin": f"{self._slip_bins[i]} to {self._slip_bins[i + 1]}", "count": int(self._slip_counts[i])}
                    )
                pd.DataFrame(slip_rows).to_csv(slippage_file, index=False)

                delay_file = LOG_DIR / "fill_delay_histogram_btc.csv"
                delay_rows = []
                for i in range(len(self._delay_bins) - 1):
                    delay_rows.append(
                        {"bin": f"{self._delay_bins[i]} to {self._delay_bins[i + 1]}", "count": int(self._delay_counts[i])}
                    )
                pd.DataFrame(delay_rows).to_csv(delay_file, index=False)

        except Exception as exc:
            log_risk.error("_update_execution_analysis error: %s | tb=%s", exc, traceback.format_exc())

    def _update_rejection_analysis(self, reason: str) -> None:
        """
        FAST: count in RAM; flush periodically.
        """
        try:
            r = str(reason or "unknown")
            self._reject_counts[r] = int(self._reject_counts.get(r, 0)) + 1
            
            # Memory limit: keep only top N reasons
            if len(self._reject_counts) > self._reject_counts_max_size:
                sorted_items = sorted(self._reject_counts.items(), key=lambda kv: -kv[1])
                self._reject_counts = dict(sorted_items[:self._reject_counts_max_size])

            # Force flush if reason is empty (shutdown signal)
            force_flush = not r or r == ""
            
            flush_sec = float(getattr(self.cfg, "reject_flush_sec", 7.0) or 7.0)
            now = time.time()
            if (not force_flush) and (now - self._reject_last_flush_ts) < flush_sec:
                return
            self._reject_last_flush_ts = now

            if not bool(getattr(self.cfg, "enable_execution_metrics_logs", True)):
                return

            rows = []
            total = int(sum(self._reject_counts.values()))
            for k, c in sorted(self._reject_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                rows.append({"reason": k, "count": int(c), "total_attempts": total, "reject_rate": float(c) / max(1, total)})

            with _FILE_LOCK:
                reject_file = LOG_DIR / "reject_rate_analysis_btc.csv"
                pd.DataFrame(rows).to_csv(reject_file, index=False)

        except Exception as exc:
            log_risk.error("_update_rejection_analysis error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- signal survival (append-only, atomic active) -------------------
    def _survival_paths(self) -> Tuple[Path, Path, Path]:
        final_csv = Path(getattr(self.cfg, "signal_survival_log_file", str(LOG_DIR / "signal_survival_final_btc.csv")))
        active_json = Path(getattr(self.cfg, "signal_survival_active_file", str(LOG_DIR / "signal_survival_active_btc.json")))
        updates_jsonl = Path(getattr(self.cfg, "signal_survival_updates_jsonl_file", str(LOG_DIR / "signal_survival_updates_btc.jsonl")))
        return final_csv, active_json, updates_jsonl

    def _init_signal_survival(self) -> None:
        try:
            final_csv, active_json, updates_jsonl = self._survival_paths()
            final_csv.parent.mkdir(parents=True, exist_ok=True)
            active_json.parent.mkdir(parents=True, exist_ok=True)
            updates_jsonl.parent.mkdir(parents=True, exist_ok=True)

            with _FILE_LOCK:
                if not final_csv.exists():
                    with open(final_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "order_id",
                                "direction",
                                "entry_price",
                                "sl_price",
                                "tp_price",
                                "entry_time",
                                "confidence",
                                "exit_price",
                                "exit_time",
                                "mfe",
                                "mae",
                                "bars_held",
                                "hit_sl",
                                "hit_tp",
                            ]
                        )

                if not active_json.exists():
                    _atomic_write_text(active_json, json.dumps({"ts": time.time(), "active": {}}, ensure_ascii=False))

                if not updates_jsonl.exists():
                    updates_jsonl.write_text("", encoding="utf-8")

        except Exception as exc:
            log_risk.error("_init_signal_survival error: %s | tb=%s", exc, traceback.format_exc())


    def _flush_survival_active(self, *, force: bool = False) -> None:
        try:
            flush_sec = float(getattr(self.cfg, "signal_survival_flush_sec", 2.0) or 2.0)
            min_force = float(getattr(self.cfg, "signal_survival_force_min_interval_sec", 0.5) or 0.5)

            now = time.time()
            if force:
                if (now - self._survival_last_flush_ts) < min_force:
                    return
            else:
                if (now - self._survival_last_flush_ts) < flush_sec:
                    return

            active: Dict[str, Any] = {}
            for oid, st in self._survival.items():
                active[str(oid)] = {
                    "order_id": st.order_id,
                    "direction": st.direction,
                    "entry_price": st.entry_price,
                    "sl_price": st.sl_price,
                    "tp_price": st.tp_price,
                    "entry_time": st.entry_time,
                    "confidence": st.confidence,
                    "mfe": st.mfe,
                    "mae": st.mae,
                    "bars_held": st.bars_held,
                    "hit_sl": st.hit_sl,
                    "hit_tp": st.hit_tp,
                    "last_price": st.last_price,
                    "last_update_ts": st.last_update_ts,
                }

            _, active_json, _ = self._survival_paths()
            payload = {"ts": now, "active": active}

            with _FILE_LOCK:
                _atomic_write_text(active_json, json.dumps(payload, ensure_ascii=False))

            self._survival_last_flush_ts = now
        except Exception as exc:
            log_risk.error("_flush_survival_active error: %s | tb=%s", exc, traceback.format_exc())



    def _emit_survival_update(self, order_id: str, st: SignalSurvivalState, kind: str) -> None:
        try:
            emit_sec = float(getattr(self.cfg, "signal_survival_update_emit_sec", 5.0) or 5.0)
            now = time.time()
            last = float(self._survival_last_update_emit.get(order_id, 0.0) or 0.0)
            if (now - last) < emit_sec and kind == "update":
                return

            _, _, updates_jsonl = self._survival_paths()
            ev = {
                "ts": now,
                "kind": str(kind),
                "order_id": st.order_id,
                "direction": st.direction,
                "entry_price": st.entry_price,
                "sl_price": st.sl_price,
                "tp_price": st.tp_price,
                "entry_time": st.entry_time,
                "confidence": st.confidence,
                "mfe": st.mfe,
                "mae": st.mae,
                "bars_held": st.bars_held,
                "hit_sl": st.hit_sl,
                "hit_tp": st.hit_tp,
                "last_price": st.last_price,
            }

            with _FILE_LOCK:
                with open(updates_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")

            self._survival_last_update_emit[order_id] = now
        except Exception as exc:
            log_risk.error("_emit_survival_update error: %s | tb=%s", exc, traceback.format_exc())

    def _load_survival_active(self) -> None:
        try:
            _, active_json, _ = self._survival_paths()
            if not active_json.exists():
                return
            with _FILE_LOCK:
                raw = active_json.read_text(encoding="utf-8")

            if not raw.strip():
                data = {"ts": time.time(), "active": {}}
            else:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    log_risk.error("_load_survival_active invalid JSON -> resetting active cache")
                    data = {"ts": time.time(), "active": {}}

            if "active" not in data or not isinstance(data["active"], dict):
                log_risk.error("_load_survival_active missing active map -> resetting")
                data["active"] = {}

            active = data.get("active") or {}
            restored: Dict[str, SignalSurvivalState] = {}

            for oid, x in active.items():
                restored[str(oid)] = SignalSurvivalState(
                    order_id=str(x.get("order_id", oid)),
                    direction=str(x.get("direction", "Buy")),
                    entry_price=float(x.get("entry_price", 0.0) or 0.0),
                    sl_price=float(x.get("sl_price", 0.0) or 0.0),
                    tp_price=float(x.get("tp_price", 0.0) or 0.0),
                    entry_time=float(x.get("entry_time", 0.0) or 0.0),
                    confidence=float(x.get("confidence", 0.0) or 0.0),
                    mfe=float(x.get("mfe", 0.0) or 0.0),
                    mae=float(x.get("mae", 0.0) or 0.0),
                    bars_held=int(x.get("bars_held", 0) or 0),
                    hit_sl=bool(x.get("hit_sl", False)),
                    hit_tp=bool(x.get("hit_tp", False)),
                    last_price=float(x.get("last_price", 0.0) or 0.0),
                    last_update_ts=float(x.get("last_update_ts", 0.0) or 0.0),
                )

            self._survival = restored
        except Exception as exc:
            log_risk.error("_load_survival_active error: %s | tb=%s", exc, traceback.format_exc())

    def track_signal_survival(
        self,
        order_id: str,
        signal: str,
        entry_price: float,
        sl: float,
        tp: float,
        entry_time: float,
        confidence: float,
    ) -> None:
        try:
            oid = str(order_id)

            conf = float(confidence)
            if conf > 1.5:
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))

            st = SignalSurvivalState(
                order_id=oid,
                direction=_side_norm(signal),
                entry_price=float(entry_price),
                sl_price=float(sl),
                tp_price=float(tp),
                entry_time=float(entry_time),
                confidence=float(conf),
                mfe=0.0,
                mae=0.0,
                bars_held=0,
                hit_sl=False,
                hit_tp=False,
                last_price=float(entry_price),
                last_update_ts=time.time(),
            )

            self._survival[oid] = st
            self._emit_survival_update(oid, st, kind="entry")
            self._flush_survival_active(force=False)
        except Exception as exc:
            log_risk.error("track_signal_survival error: %s | tb=%s", exc, traceback.format_exc())


    def update_signal_survival(self, order_id: str, current_price: float, current_time: float, tick_count: int) -> None:
        try:
            _ = (current_time, tick_count)
            oid = str(order_id)
            st = self._survival.get(oid)
            if st is None:
                return
    
            st.update(current_price=float(current_price))
            st.last_update_ts = time.time()
    
            px = float(current_price)
            if _side_norm(st.direction) == "Buy":
                if st.sl_price > 0 and px <= float(st.sl_price):
                    st.hit_sl = True
                if st.tp_price > 0 and px >= float(st.tp_price):
                    st.hit_tp = True
            else:
                if st.sl_price > 0 and px >= float(st.sl_price):
                    st.hit_sl = True
                if st.tp_price > 0 and px <= float(st.tp_price):
                    st.hit_tp = True
    
            self._emit_survival_update(oid, st, kind="update")
    
            # Force flush only on critical events (SL/TP hit), otherwise periodic flush
            self._flush_survival_active(force=bool(st.hit_sl or st.hit_tp))
        except Exception as exc:
            log_risk.error("update_signal_survival error: %s | tb=%s", exc, traceback.format_exc())



    def finalize_signal_survival(
        self,
        order_id: str,
        *,
        exit_price: float,
        exit_time: float,
    ) -> None:
        try:
            oid = str(order_id)
            st = self._survival.get(oid)
            if st is None:
                return

            final_csv, _, _ = self._survival_paths()
            row = {
                "order_id": st.order_id,
                "direction": st.direction,
                "entry_price": float(st.entry_price),
                "sl_price": float(st.sl_price),
                "tp_price": float(st.tp_price),
                "entry_time": float(st.entry_time),
                "confidence": float(st.confidence),
                "exit_price": float(exit_price),
                "exit_time": float(exit_time),
                "mfe": float(st.mfe),
                "mae": float(st.mae),
                "bars_held": int(st.bars_held),
                "hit_sl": bool(st.hit_sl),
                "hit_tp": bool(st.hit_tp),
            }

            with _FILE_LOCK:
                with open(final_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writerow(row)

            self._emit_survival_update(oid, st, kind="exit")

            self._survival.pop(oid, None)
            self._survival_last_update_emit.pop(oid, None)

            self._flush_survival_active(force=True)
        except Exception as exc:
            log_risk.error("finalize_signal_survival error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- slippage model (analytics) -------------------
    def compute_slippage_model(
        self,
        spread: float,
        volatility: float,
        latency_ms: float,
        is_backtest: bool = False,
        is_dry_run: bool = False,
        confidence: float = 0.75,
    ) -> dict:
        try:
            spread = float(spread)
            volatility = float(volatility)
            latency_ms = float(latency_ms)
            confidence = float(confidence)

            spread_slippage = spread * 0.5
            vol_factor = min(2.5, max(0.4, volatility / 25.0))
            latency_factor = min(1.8, max(0.1, latency_ms / 120.0))

            total_slippage = spread_slippage + spread_slippage * vol_factor + spread_slippage * latency_factor
            partial_fill_prob = min(0.35, max(0.05, (volatility / 60.0) + (latency_ms / 250.0)))

            if is_backtest:
                total_slippage *= 1.25
                partial_fill_prob *= 1.10
            elif is_dry_run:
                total_slippage *= 0.85
                partial_fill_prob *= 0.95

            confidence_factor = max(0.7, min(1.3, 1.0 - (confidence - 0.5) * 0.2))
            total_slippage *= confidence_factor

            data = {
                "timestamp": self._utc_now().isoformat(),
                "spread": spread,
                "volatility": volatility,
                "latency_ms": latency_ms,
                "total_slippage": total_slippage,
                "partial_fill_probability": partial_fill_prob,
                "mode": "backtest" if is_backtest else ("dry_run" if is_dry_run else "live"),
                "confidence_factor": confidence_factor,
            }

            if bool(getattr(self.cfg, "enable_execution_metrics_logs", True)):
                with _FILE_LOCK:
                    csv_file = LOG_DIR / "slippage_model_btc.csv"
                    write_header = not csv_file.exists()
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(data)

            return data
        except Exception as exc:
            log_risk.error("compute_slippage_model error: %s | tb=%s", exc, traceback.format_exc())
            return {}

    # ------------------- state persistence -------------------
    def _state_path(self) -> Path:
        return Path(getattr(self.cfg, "risk_state_file", str(LOG_DIR / "risk_state_btc.json")))

    def save_state(self) -> None:
        p = self._state_path()
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "ts": time.time(),
            "daily_date": str(self.daily_date),
            "daily_start_balance": float(self.daily_start_balance),
            "daily_peak_equity": float(self.daily_peak_equity),
            "_target_breached": bool(self._target_breached),
            "_target_breached_return": float(self._target_breached_return),
            "current_phase": str(self.current_phase),
            "_trading_disabled_until": float(self._trading_disabled_until),
            "_hard_stop_reason": self._hard_stop_reason,
            "_soft_stop_until": float(self._soft_stop_until),
            "_soft_stop_reason": self._soft_stop_reason,
            "_current_drawdown": float(self._current_drawdown),
            "_analysis_blocked_until": float(self._analysis_blocked_until),
            "_last_fill_ts": float(self._last_fill_ts),
            "_latency_bad_since": float(self._latency_bad_since or 0.0),
            "_daily_signal_count": int(self._signal_throttle.daily_count),
            "_hour_window_start_ts": float(self._signal_throttle.hour_window_start_ts),
            "_hour_window_count": int(self._signal_throttle.hour_window_count),
            "_exec_breaker_until": float(self._exec_breaker_until),
            "execmon": {
                "ewma_slip": float(self.execmon.ewma_slip or 0.0),
                "ewma_lat": float(self.execmon.ewma_lat or 0.0),
                "ewma_spread": float(self.execmon.ewma_spread or 0.0),
            },
            "kill_switch": {
                "status": str(self._kill_state.status),
                "cooling_until_ts": float(self._kill_state.cooling_until_ts),
                "last_expectancy": float(self._kill_state.last_expectancy),
                "last_winrate": float(self._kill_state.last_winrate),
                "last_trades": int(self._kill_state.last_trades),
            },
        }

        with _FILE_LOCK:
            _atomic_write_text(p, json.dumps(state, ensure_ascii=False))
    
    def _shutdown_flush_all(self) -> None:
        """Flush all critical data on shutdown to prevent data loss."""
        try:
            # Flush survival data immediately
            self._flush_survival_active(force=True)
            
            # Flush execution analysis immediately
            if self._hist_last_flush_ts > 0:
                self._update_execution_analysis({})  # Trigger flush
            
            # Flush rejection analysis immediately
            if self._reject_last_flush_ts > 0:
                self._update_rejection_analysis("")  # Trigger flush
            
            # Save state immediately
            self.save_state()
        except Exception as exc:
            log_risk.error("_shutdown_flush_all error: %s | tb=%s", exc, traceback.format_exc())

    def load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return

        with _FILE_LOCK:
            raw = p.read_text(encoding="utf-8")
        st = json.loads(raw)

        ks = st.get("kill_switch") or {}
        self._kill_state.status = str(ks.get("status", self._kill_state.status) or self._kill_state.status)
        self._kill_state.cooling_until_ts = float(ks.get("cooling_until_ts", self._kill_state.cooling_until_ts) or 0.0)
        self._kill_state.last_expectancy = float(ks.get("last_expectancy", self._kill_state.last_expectancy) or 0.0)
        self._kill_state.last_winrate = float(ks.get("last_winrate", self._kill_state.last_winrate) or 0.0)
        self._kill_state.last_trades = int(ks.get("last_trades", self._kill_state.last_trades) or 0)
        self._kill_last_status = self._kill_state.status

        saved_date = str(st.get("daily_date", "")).strip()
        if saved_date and saved_date != str(self._utc_date()):
            # carry only breaker stats across days
            self._exec_breaker_until = float(st.get("_exec_breaker_until", 0.0) or 0.0)
            em = st.get("execmon") or {}
            self.execmon.ewma_slip = float(em.get("ewma_slip", 0.0) or 0.0)
            self.execmon.ewma_lat = float(em.get("ewma_lat", 0.0) or 0.0)
            self.execmon.ewma_spread = float(em.get("ewma_spread", 0.0) or 0.0)

            # Clean day
            self.daily_date = self._utc_date()
            self._trading_disabled_until = 0.0
            self._hard_stop_reason = None
            self._soft_stop_until = 0.0
            self._soft_stop_reason = None
            self._phase_reason_last = None
            self.current_phase = "A"
            self._target_breached = False
            self._target_breached_return = 0.0
            return

        self.daily_date = self._utc_date()
        self.daily_start_balance = float(st.get("daily_start_balance", self.daily_start_balance) or 0.0)
        self.daily_peak_equity = float(st.get("daily_peak_equity", self.daily_peak_equity) or 0.0)
        self._target_breached = bool(st.get("_target_breached", self._target_breached))
        self._target_breached_return = float(st.get("_target_breached_return", self._target_breached_return) or 0.0)
        self.current_phase = str(st.get("current_phase", self.current_phase) or self.current_phase)
        self._trading_disabled_until = float(st.get("_trading_disabled_until", self._trading_disabled_until) or 0.0)
        self._hard_stop_reason = st.get("_hard_stop_reason", self._hard_stop_reason)
        self._soft_stop_until = float(st.get("_soft_stop_until", self._soft_stop_until) or 0.0)
        self._soft_stop_reason = st.get("_soft_stop_reason", self._soft_stop_reason)
        self._current_drawdown = float(st.get("_current_drawdown", self._current_drawdown) or 0.0)
        self._analysis_blocked_until = float(st.get("_analysis_blocked_until", self._analysis_blocked_until) or 0.0)
        self._last_fill_ts = float(st.get("_last_fill_ts", self._last_fill_ts) or 0.0)
        lbs = float(st.get("_latency_bad_since", 0.0) or 0.0)
        self._latency_bad_since = lbs if lbs > 0 else None
        self._signal_throttle.daily_count = int(st.get("_daily_signal_count", self._signal_throttle.daily_count) or 0)
        self._signal_throttle.hour_window_start_ts = float(st.get("_hour_window_start_ts", self._signal_throttle.hour_window_start_ts) or time.time())
        self._signal_throttle.hour_window_count = int(st.get("_hour_window_count", self._signal_throttle.hour_window_count) or 0)
        self._exec_breaker_until = float(st.get("_exec_breaker_until", self._exec_breaker_until) or 0.0)

        em = st.get("execmon") or {}
        self.execmon.ewma_slip = float(em.get("ewma_slip", 0.0) or 0.0)
        self.execmon.ewma_lat = float(em.get("ewma_lat", 0.0) or 0.0)
        self.execmon.ewma_spread = float(em.get("ewma_spread", 0.0) or 0.0)


__all__ = [
    "RiskManager",
    "RiskDecision",
    "ExecutionQualityMonitor",
    "ExecSample",
    "percentile_rank",
    "SignalSurvivalState",
]
