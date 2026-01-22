


# StrategiesBtc/risk_management.py  (BTC)  — PRODUCTION / FAST / PHASE A-B-C FIXED

from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

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
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# ============================================================
# ERROR-only logging (isolated) + rotation
# ============================================================
LOG_DIR = LOG_ROOT

log_risk = logging.getLogger("risk_manager_btc")
log_risk.setLevel(logging.ERROR)
log_risk.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log_risk.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("risk_manager_btc.log")),
        maxBytes=5242880,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log_risk.addHandler(fh)

_FILE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def percentile_rank(series: np.ndarray, value: float) -> float:
    try:
        if series is None or len(series) == 0:
            return 50.0
        x = series[np.isfinite(series)]
        if len(x) == 0:
            return 50.0
        return float(np.sum(x <= value) / len(x) * 100.0)
    except Exception as exc:
        log_risk.error("percentile_rank error: %s | tb=%s", exc, traceback.format_exc())
        return 50.0


def _is_finite(*xs: float) -> bool:
    try:
        return all(bool(np.isfinite(x)) for x in xs)
    except Exception:
        return False


def _side_norm(side: str) -> str:
    s = str(side or "").strip().lower()
    if s in ("buy", "long", "b", "1", "order_type_buy", "op_buy"):
        return "Buy"
    if s in ("sell", "short", "s", "-1", "order_type_sell", "op_sell"):
        return "Sell"
    return "Buy" if "buy" in s else "Sell" if "sell" in s else str(side)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atr_fallback(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    alpha = 1.0 / max(1, int(period))
    out = np.empty_like(tr, dtype=np.float64)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


# ============================================================
# Execution quality monitor (breaker input)
# ============================================================
@dataclass(slots=True)
class ExecSample:
    ts: float
    side: str
    expected_price: float
    filled_price: float
    point: float
    spread_points: float
    latency_total_ms: float

    @property
    def slippage_points(self) -> float:
        """
        ABS slippage in points (always >= 0).
        This is critical for robust breakers: do NOT allow sign cancellation.
        """
        try:
            if self.point <= 0:
                return 0.0
            return float(abs(self.filled_price - self.expected_price) / self.point)
        except Exception:
            return 0.0


class ExecutionQualityMonitor:
    def __init__(self, *, window: int = 300) -> None:
        self.window = int(window)

        self.samples: Deque[ExecSample] = deque(maxlen=self.window)
        self.spreads: Deque[float] = deque(maxlen=self.window)  # points
        self.mid_moves: Deque[float] = deque(maxlen=self.window)  # points per tick

        self._last_mid: Optional[float] = None

        # EWMA
        self.ewma_slip: Optional[float] = None
        self.ewma_lat: Optional[float] = None
        self.ewma_spread: Optional[float] = None

    @staticmethod
    def _ewma(prev: Optional[float], x: float, alpha: float) -> float:
        return x if prev is None else (alpha * x + (1.0 - alpha) * prev)

    def on_quote(self, bid: float, ask: float, point: float) -> None:
        if point <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            return

        mid = (bid + ask) / 2.0
        spread_points = (ask - bid) / point

        self.spreads.append(float(spread_points))
        self.ewma_spread = self._ewma(self.ewma_spread, float(spread_points), alpha=0.08)

        if self._last_mid is not None:
            move_points = abs(mid - self._last_mid) / point
            self.mid_moves.append(float(move_points))
        self._last_mid = mid

    def on_execution(self, sample: ExecSample) -> None:
        self.samples.append(sample)
        self.ewma_slip = self._ewma(self.ewma_slip, float(sample.slippage_points), alpha=0.10)
        self.ewma_lat = self._ewma(self.ewma_lat, float(sample.latency_total_ms), alpha=0.10)

    def snapshot(self) -> Dict[str, float]:
        slip = [s.slippage_points for s in self.samples]
        lat = [s.latency_total_ms for s in self.samples]
        spr = list(self.spreads)

        def p95(x: List[float]) -> float:
            if not x:
                return 0.0
            return float(np.percentile(np.array(x, dtype=np.float64), 95))

        return {
            "n": float(len(self.samples)),
            "ewma_slippage_points": float(self.ewma_slip or 0.0),
            "ewma_latency_ms": float(self.ewma_lat or 0.0),
            "ewma_spread_points": float(self.ewma_spread or 0.0),
            "p95_slippage_points": p95(slip),
            "p95_latency_ms": p95(lat),
            "p95_spread_points": p95(spr) if spr else 0.0,
            "max_spread_points": float(max(spr)) if spr else 0.0,
            "p95_mid_move_points": float(np.percentile(np.array(self.mid_moves, dtype=np.float64), 95))
            if self.mid_moves
            else 0.0,
        }

    def anomaly_reasons(self, *, cfg: Any) -> List[str]:
        s = self.snapshot()

        max_p95_lat = float(getattr(cfg, "exec_max_p95_latency_ms", 650.0))
        max_p95_slip = float(getattr(cfg, "exec_max_p95_slippage_points", 30.0))
        max_spread = float(getattr(cfg, "exec_max_spread_points", 120.0))
        max_ewma_slip = float(getattr(cfg, "exec_max_ewma_slippage_points", 18.0))

        reasons: List[str] = []
        if s["p95_latency_ms"] > max_p95_lat:
            reasons.append("exec:latency_p95")
        if s["p95_slippage_points"] > max_p95_slip:
            reasons.append("exec:slippage_p95")
        if s["ewma_slippage_points"] > max_ewma_slip:
            reasons.append("exec:slippage_ewma")
        if s["max_spread_points"] > max_spread:
            reasons.append("mkt:spread_shock")
        return reasons


_BASE_LOT = 0.02
_BASE_TP_USD = 2.0
_STEP_BALANCE = 100.0
_STEP_LOT = 0.01
_STEP_TP_USD = 1.0
_STEP_START_BALANCE = 200.0


def btc_lot_and_tp_usd(balance: float) -> Tuple[float, float]:
    try:
        bal = float(balance or 0.0)
    except Exception:
        return 0.0, 0.0

    if bal < 1.0:
        return 0.0, 0.0

    if bal < _STEP_START_BALANCE:
        step = 0
    else:
        step = int((bal - _STEP_START_BALANCE) // _STEP_BALANCE) + 1

    lot = _BASE_LOT + (step * _STEP_LOT)
    tp = _BASE_TP_USD + (step * _STEP_TP_USD)
    return round(float(lot), 2), round(float(tp), 2)


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class AccountCache:
    """Unified account snapshot cache structure."""
    ts: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0


@dataclass
class SignalThrottle:
    """Unified signal throttling structure."""
    daily_count: int = 0
    hour_window_start_ts: float = field(default_factory=lambda: time.time())
    hour_window_count: int = 0


@dataclass
class MT5Throttle:
    """Unified MT5 readiness throttling structure."""
    ts: float = 0.0
    ok: bool = False


@dataclass
class SymbolMeta:
    """Unified symbol metadata cache structure."""
    ts: float = 0.0
    digits: int = 2
    point: float = 0.01
    vol_min: float = 0.01
    vol_max: float = 100.0
    vol_step: float = 0.01
    stops_level_points: int = 0
    freeze_level_points: int = 0


@dataclass(slots=True)
class SignalSurvivalState:
    order_id: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    entry_time: float
    confidence: float
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0
    hit_sl: bool = False
    hit_tp: bool = False
    last_price: float = 0.0
    last_update_ts: float = 0.0

    def update(self, *, current_price: float) -> None:
        d = _side_norm(self.direction)
        entry = float(self.entry_price)
        px = float(current_price)
        self.last_price = px

        if d == "Buy":
            self.mfe = float(max(self.mfe, px - entry))
            self.mae = float(min(self.mae, px - entry))
        else:
            self.mfe = float(max(self.mfe, entry - px))
            self.mae = float(min(self.mae, entry - px))

        self.bars_held = int(self.bars_held) + 1


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

        # MT5 readiness throttling (unified structure)
        self._mt5_throttle = MT5Throttle()

        # Symbol meta cache (unified structure)
        self._symbol_meta = SymbolMeta()

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
        self._init_signal_survival()

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

        bal, eq = self._account_snapshot()
        base = bal if bal > 0 else eq

        self.daily_start_balance = float(base if base > 0 else 0.0)
        self.daily_peak_equity = float(max(eq, self.daily_start_balance))

        self._target_breached = False
        self._target_breached_return = 0.0
        self._phase_reason_last = None
        self.current_phase = "A"

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

    # ------------------- MT5 readiness / symbol meta -------------------
    def _ensure_ready(self) -> bool:
        try:
            now = time.time()
            if (now - self._mt5_throttle.ts) < 1.0:
                return bool(self._mt5_throttle.ok)

            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                if info is None:
                    self._ready_ok = False
                else:
                    if not info.visible:
                        self._mt5_throttle.ok = bool(mt5.symbol_select(self.symbol, True))
                    else:
                        self._mt5_throttle.ok = True

            self._mt5_throttle.ts = now
            return bool(self._mt5_throttle.ok)
        except Exception as exc:
            self._mt5_throttle.ok = False
            self._mt5_throttle.ts = time.time()
            log_risk.error("ensure_ready error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _ensure_symbol_meta(self) -> bool:
        """Ensure symbol meta is loaded and fresh (returns True if valid)."""
        meta = self._symbol_meta  # Access attribute directly
        if not isinstance(meta, SymbolMeta):
            return False
        if time.time() - meta.ts < 10.0:
            return True
        if not self._ensure_ready():
            return False
        try:
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return False

            meta.digits = int(getattr(info, "digits", 2) or 2)
            meta.point = float(getattr(info, "point", 0.01) or 0.01)
            meta.vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            meta.vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
            meta.vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
            meta.stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
            meta.freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)

            meta.ts = time.time()
            return True
        except Exception as exc:
            log_risk.error("symbol_meta error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _min_stop_distance(self) -> float:
        if not self._ensure_symbol_meta():
            return 0.0
        return float(self._symbol_meta.stops_level_points) * float(self._symbol_meta.point)

    def _normalize_price(self, price: float) -> float:
        if not self._ensure_symbol_meta():
            return float(price)
        return float(round(float(price), int(self._symbol_meta.digits)))

    def _normalize_volume_floor(self, vol: float) -> float:
        if not self._ensure_symbol_meta():
            return float(vol)

        step = max(1e-9, float(self._symbol_meta.vol_step))
        v = max(float(self._symbol_meta.vol_min), min(float(vol), float(self._symbol_meta.vol_max)))
        v = math.floor(v / step) * step
        return float(round(v, 8))

    def _tp_usd_to_price(self, entry: float, side: str, volume: float, usd_profit: float) -> Optional[float]:
        try:
            if float(entry) <= 0 or float(volume) <= 0 or float(usd_profit) <= 0:
                return None
            if not self._ensure_ready():
                return None
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return None

            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            digits = int(getattr(info, "digits", 5) or 5)

            if tick_value <= 0.0 or tick_size <= 0.0:
                return None

            ticks_needed = float(usd_profit) / (tick_value * float(volume))
            if not np.isfinite(ticks_needed) or ticks_needed <= 0:
                return None

            price_delta = ticks_needed * tick_size
            side_n = _side_norm(side)
            tp = float(entry) + price_delta if side_n == "Buy" else float(entry) - price_delta
            tp = round(float(tp), digits)
            if tp <= 0:
                return None

            if side_n == "Buy" and not (tp > float(entry)):
                return None
            if side_n == "Sell" and not (tp < float(entry)):
                return None

            return tp
        except Exception:
            return None

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
            return True, ""

        try:
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return True, ""

            tm = int(getattr(info, "trade_mode", mt5.SYMBOL_TRADE_MODE_FULL) or mt5.SYMBOL_TRADE_MODE_FULL)

            if tm == mt5.SYMBOL_TRADE_MODE_DISABLED:
                return False, "trade_mode_disabled"
            if tm == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
                return False, "trade_mode_close_only"

            return True, ""
        except Exception:
            return True, ""

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
            if not self._ensure_symbol_meta():
                return
            self.execmon.on_quote(float(bid), float(ask), float(self._symbol_meta.point))
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

            # Ensure start balance
            if self.daily_start_balance <= 0 and (bal > 0 or eq > 0):
                self.daily_start_balance = float(max(bal, eq))
                self.daily_peak_equity = max(self.daily_peak_equity, self.daily_start_balance)

            if self.daily_start_balance <= 0 or eq <= 0:
                return

            daily_return = float((eq - self.daily_start_balance) / self.daily_start_balance)

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

            # PROFIT TARGET -> Phase B
            if daily_return >= target:
                if not self._target_breached:
                    self._target_breached = True
                    self._target_breached_return = float(daily_return)
                self._target_breached_return = max(float(self._target_breached_return), float(daily_return))
                self._set_phase("B", "daily_target_reached")

            # LOSS -> Phase B/C
            if loss_b > 0 and daily_return <= -loss_b:
                self._set_phase("B", "daily_loss_b")
            if loss_c > 0 and daily_return <= -loss_c:
                # C is REAL now (trade-block). Hard stop only if extreme below max_loss and limits enabled.
                self._set_phase("C", "daily_loss_c")
                self._enter_soft_stop("daily_loss_c")

            # HARD STOP (absolute)
            if bool(getattr(self.cfg, "enforce_daily_limits", False)):
                if daily_return <= -max_loss:
                    self._enter_hard_stop("max_daily_loss")

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

    # ------------------- helpers: spread points -------------------
    def _current_spread_points(self) -> float:
        try:
            if not self._ensure_symbol_meta():
                return 0.0
            with MT5_LOCK:
                t = mt5.symbol_info_tick(self.symbol)
            if not t:
                return 0.0
            return float((float(t.ask) - float(t.bid)) / max(1e-9, float(self._point)))
        except Exception:
            return 0.0

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
                reasons.append(self.hard_stop_reason() or "hard_stop")

            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                reasons.append(self._soft_stop_reason or "soft_stop")

            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append(self._phase_reason_last or "phase_c")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)) and drawdown_exceeded:
                reasons.append("max_drawdown")

            if spread_pct > float(self.sp.spread_limit_pct):
                reasons.append("spread_pct")

            ignore_micro = bool(getattr(self.cfg, "ignore_microstructure", False))
            if not ignore_micro:
                if not tick_ok and tick_reason not in ("tps_low", "no_rates"):
                    reasons.append(f"micro:{tick_reason}")

            if not bool(getattr(self.cfg, "ignore_sessions", False)):
                if not in_session:
                    ok, r = self._trade_mode_state()
                    reasons.append(r or ("maintenance_blackout" if not ok else "session_blocked"))

            if self.rollover_blackout():
                reasons.append("rollover_blackout")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                    if spread_pct > float(self.sp.spread_limit_pct) * 1.2:
                        reasons.append("dd_strict_spread")

            max_per_day = int(getattr(self.cfg, "max_signals_per_day", 0) or 0)
            if max_per_day > 0:
                per_hour = max(1, int(float(max_per_day) / 24.0))
                now = time.time()
                if (now - self._signal_throttle.hour_window_start_ts) >= 3600.0:
                    self._signal_throttle.hour_window_start_ts = now
                    self._signal_throttle.hour_window_count = 0
                if self._signal_throttle.hour_window_count >= per_hour:
                    reasons.append("hourly_signal_limit")

            if spread_pct > float(self.sp.spread_limit_pct) * 1.5:
                reasons.append("excessive_spread_pct")

            if not latency_cooldown:
                reasons.append("latency_cooldown")

            return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)
        except Exception as exc:
            log_risk.error("guard_decision error: %s | tb=%s", exc, traceback.format_exc())
            return RiskDecision(allowed=False, reasons=["guard_error"])

    def can_trade(self, confidence: float, signal_type: str) -> RiskDecision:
        try:
            _ = signal_type

            reasons: List[str] = []
            self.evaluate_account_state()

            if self.requires_hard_stop():
                reasons.append(self.hard_stop_reason() or "hard_stop")
                return RiskDecision(False, reasons)

            if self._soft_stop_until > 0 and time.time() < self._soft_stop_until:
                reasons.append(self._soft_stop_reason or "soft_stop")
                return RiskDecision(False, reasons)

            # Phase C MUST block trading (unless explicitly ignored)
            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append(self._phase_reason_last or "phase_c")
                return RiskDecision(False, reasons)

            if not self.market_open_24_5():
                ok, r = self._trade_mode_state()
                reasons.append(r or "maintenance_blackout")
                return RiskDecision(False, reasons)

            if self.rollover_blackout():
                reasons.append("rollover_blackout")

            if float(confidence) < float(self.cfg.min_confidence_signal):
                reasons.append("low_confidence")

            if self.current_phase == "B" and float(confidence) < float(self.cfg.ultra_confidence_min):
                reasons.append("not_ultra_confidence")

            if not self.can_analyze():
                reasons.append("analysis_cooldown")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown):
                    reasons.append("high_drawdown")

            if self._last_fill_ts and (time.time() - self._last_fill_ts) < float(self.cfg.cooldown_seconds):
                reasons.append("cooldown_period")

            if not self.latency_cooldown():
                reasons.append("latency")

            max_spread_points = getattr(self.cfg, "max_spread_points", None)
            if max_spread_points is not None:
                sp_pts = self._current_spread_points()
                if sp_pts > float(max_spread_points):
                    reasons.append("spread_points_cap")

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
    def _order_calc_profit_money(self, side: str, volume: float, entry: float, price2: float) -> float:
        try:
            if not self._ensure_ready():
                return 0.0
            order_type = mt5.ORDER_TYPE_BUY if _side_norm(side) == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                val = mt5.order_calc_profit(order_type, self.symbol, float(volume), float(entry), float(price2))
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    def _order_calc_margin(self, side: str, volume: float, entry: float) -> float:
        try:
            if not self._ensure_ready():
                return 0.0
            order_type = mt5.ORDER_TYPE_BUY if _side_norm(side) == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                m = mt5.order_calc_margin(order_type, self.symbol, float(volume), float(entry))
            return float(m) if m is not None else 0.0
        except Exception:
            return 0.0

    def _apply_broker_constraints(self, side: str, entry: float, sl: float, tp: float) -> Tuple[float, float]:
        if not self._ensure_symbol_meta():
            return sl, tp

        side_n = _side_norm(side)
        min_dist = self._min_stop_distance()

        if min_dist > 0:
            if side_n == "Buy":
                sl = min(sl, entry - min_dist)
                tp = max(tp, entry + min_dist)
            else:
                sl = max(sl, entry + min_dist)
                tp = min(tp, entry - min_dist)

        return self._normalize_price(sl), self._normalize_price(tp)

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
        try:
            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return 0.0

            if not _is_finite(entry_price, stop_loss, take_profit, confidence):
                return float(self.cfg.fixed_volume)

            if not self._ensure_symbol_meta():
                return float(self.cfg.fixed_volume)

            min_rr = getattr(self.cfg, "min_rr", None)
            if min_rr is not None:
                rr_ratio = self._rr(entry_price, stop_loss, take_profit)
                if rr_ratio < float(min_rr):
                    return 0.0

            _, eq = self._account_snapshot()
            if eq <= 0:
                return float(self.cfg.fixed_volume)

            risk_money = float(eq) * float(self.cfg.max_risk_per_trade)

            if self.current_phase == "B":
                risk_money *= float(getattr(self.cfg, "phase_b_risk_mult", 1.0) or 1.0)

            if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                risk_money *= float(getattr(self.cfg, "dd_risk_mult", 1.0) or 1.0)

            conf = float(confidence)
            if conf > 1.5:
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))

            min_mult = float(getattr(self.cfg, "confidence_risk_min_mult", 0.75) or 0.75)
            max_mult = float(getattr(self.cfg, "confidence_risk_max_mult", 1.15) or 1.15)
            if max_mult < min_mult:
                max_mult = min_mult
            risk_money *= (min_mult + (max_mult - min_mult) * conf)

            risk_1lot = abs(self._order_calc_profit_money(side_n, 1.0, entry_price, stop_loss))
            if risk_1lot <= 0:
                return float(self.cfg.fixed_volume)

            raw_lot = risk_money / risk_1lot

            if hasattr(self.cfg, "max_position_percentage"):
                cap_pct = float(getattr(self.cfg, "max_position_percentage")) / 100.0
                if cap_pct > 0:
                    margin_cap = eq * cap_pct
                    margin_1lot = self._order_calc_margin(side_n, 1.0, entry_price)
                    if margin_1lot > 0:
                        raw_lot = min(raw_lot, margin_cap / margin_1lot)

            margin_free = self._get_margin_free()
            mf_mult = float(getattr(self.cfg, "margin_free_safety_mult", 1.25) or 1.25)
            if margin_free > 0 and mf_mult > 1.0:
                margin_1lot = self._order_calc_margin(side_n, 1.0, entry_price)
                if margin_1lot > 0:
                    cap_by_mf = (margin_free / mf_mult) / margin_1lot
                    raw_lot = min(raw_lot, cap_by_mf)

            lot = self._normalize_volume_floor(raw_lot)
            return float(lot) if lot > 0 else 0.0
        except Exception as exc:
            log_risk.error("calculate_position_size error: %s | tb=%s", exc, traceback.format_exc())
            return float(self.cfg.fixed_volume)

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
            if self._ensure_symbol_meta() and tv > entry * 0.02:
                tv = tv * float(self._symbol_meta.point)

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
            if atr <= 0 and hasattr(self.cfg, "fallback_atr_abs"):
                atr = float(getattr(self.cfg, "fallback_atr_abs"))

            if atr <= 0:
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

    def plan_order(
        self,
        side: str,
        confidence: float,
        ind: Dict[str, Any],
        adapt: Dict[str, Any],
        entry: Optional[float] = None,
        ticks: Optional[TickStats] = None,
        zones: Optional[MicroZones] = None,
        tick_volatility: float = 0.0,
        open_positions: int = 0,
        max_positions: int = 3,
        unrealized_pl: float = 0.0,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        try:
            _ = (ind, ticks)

            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return None, None, None, None
            if not self._ensure_ready():
                return None, None, None, None

            # Real phase C block
            self.evaluate_account_state()
            if (self._soft_stop_until > 0 and time.time() < self._soft_stop_until) or (
                self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False))
            ):
                return None, None, None, None

            ok, reason = self._trade_mode_state()
            if not ok:
                log_risk.error("plan_order blocked: %s", reason)
                return None, None, None, None
            if self._in_maintenance_blackout():
                return None, None, None, None

            if entry is None:
                with MT5_LOCK:
                    t = mt5.symbol_info_tick(self.symbol)
                if not t:
                    return None, None, None, None
                entry = float(t.ask) if side_n == "Buy" else float(t.bid)

            entry_price = float(entry)
            if not _is_finite(entry_price) or entry_price <= 0:
                return None, None, None, None

            sl, tp = self._micro_zone_sl_tp(side_n, entry_price, zones, float(tick_volatility))

            min_dist = self._min_stop_distance()
            min_sl_pct = float(getattr(self.cfg, "min_sl_pct", 0.0008) or 0.0008)
            min_tp_pct = float(getattr(self.cfg, "min_tp_pct", 0.0012) or 0.0012)

            min_sl = max(min_dist, entry_price * min_sl_pct)
            min_tp = max(min_dist, entry_price * min_tp_pct)

            if (not _is_finite(sl, tp)) or abs(entry_price - sl) < min_sl or abs(tp - entry_price) < min_tp:
                sl, tp = self._fallback_atr_sl_tp(side_n, entry_price, adapt)

            if not _is_finite(sl, tp) or sl == entry_price or tp == entry_price:
                return None, None, None, None

            sl, tp = self._apply_broker_constraints(side_n, entry_price, float(sl), float(tp))

            if side_n == "Buy" and not (sl < entry_price < tp):
                return None, None, None, None
            if side_n == "Sell" and not (tp < entry_price < sl):
                return None, None, None, None

            use_usd_tp = False
            balance = float(self._get_balance())
            base_lot, tp_usd = btc_lot_and_tp_usd(balance) if balance > 0 else (0.0, 0.0)
            base_lot = self._normalize_volume_floor(float(base_lot or 0.0))

            if base_lot > 0 and float(tp_usd or 0.0) > 0.0:
                tp_price = self._tp_usd_to_price(entry_price, side_n, base_lot, float(tp_usd))
                if tp_price is not None:
                    tp = float(tp_price)
                    sl, tp = self._apply_broker_constraints(side_n, entry_price, float(sl), float(tp))
                    if (side_n == "Buy" and tp > entry_price) or (side_n == "Sell" and tp < entry_price):
                        use_usd_tp = True

            if use_usd_tp:
                lot = float(base_lot)
            else:
                min_rr = getattr(self.cfg, "min_rr", None)
                if min_rr is not None:
                    rr = self._rr(entry_price, sl, tp)
                    rr_min = float(min_rr)
                    if rr < rr_min:
                        dist = abs(entry_price - sl)
                        if side_n == "Buy":
                            tp = self._normalize_price(entry_price + dist * rr_min)
                        else:
                            tp = self._normalize_price(entry_price - dist * rr_min)
                        sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

                rr_cap = float(getattr(self.cfg, "tp_rr_cap", 0.0) or 0.0)
                if rr_cap > 0:
                    dist_sl = abs(entry_price - sl)
                    dist_tp = abs(tp - entry_price)
                    if dist_sl > 0 and dist_tp > dist_sl * rr_cap:
                        if side_n == "Buy":
                            tp = self._normalize_price(entry_price + dist_sl * rr_cap)
                        else:
                            tp = self._normalize_price(entry_price - dist_sl * rr_cap)
                        if abs(tp - entry_price) < min_tp:
                            if side_n == "Buy":
                                tp = self._normalize_price(entry_price + min_tp)
                            else:
                                tp = self._normalize_price(entry_price - min_tp)
                        sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

                lot = self.calculate_position_size(side_n, entry_price, sl, tp, float(confidence))
                if lot <= 0 or not _is_finite(lot):
                    return None, None, None, None

            if (
                int(open_positions) > 0
                and int(open_positions) < int(max_positions)
                and float(unrealized_pl) >= 0.0
                and float(confidence) >= float(self.cfg.ultra_confidence_min)
            ):
                if not use_usd_tp:
                    scale_factor = max(0.3, 1.0 - (int(open_positions) / max(1, int(max_positions))))
                    lot = self._normalize_volume_floor(lot * scale_factor)

                    tp_bonus = 1.0 + float(getattr(self.cfg, "multi_order_tp_bonus_pct", 0.12) or 0.12)

                    if side_n == "Buy":
                        tp = self._normalize_price(entry_price + abs(tp - entry_price) * tp_bonus)
                    else:
                        tp = self._normalize_price(entry_price - abs(tp - entry_price) * tp_bonus)

                    sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

            return float(entry_price), float(sl), float(tp), float(lot)

        except Exception as exc:
            log_risk.error("plan_order error: %s | tb=%s", exc, traceback.format_exc())
            return None, None, None, None

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
        slippage: float,
    ) -> None:
        """
        Backward-compatible: if called by older engine without 'side',
        the 'side' parameter arrives as a number (enqueue_time). We remap safely.
        """
        try:
            # Back-compat remap (old call style: order_id, enqueue, sent, fill, expected, filled, slippage)
            if isinstance(side, (int, float)) and isinstance(enqueue_time, (int, float)):
                enqueue_time, send_time, fill_time, expected_price, filled_price, slippage = (
                    float(side),
                    float(enqueue_time),
                    float(send_time),
                    float(fill_time),
                    float(expected_price),
                    float(filled_price),
                )
                side_n = "Buy"
            else:
                side_n = _side_norm(side)

            if self._ensure_symbol_meta() and float(self._symbol_meta.point) > 0:
                slip_points = float(slippage) / max(1e-9, float(self._symbol_meta.point))
            else:
                slip_points = float(slippage)

            metrics = {
                "order_id": str(order_id),
                "timestamp": self._utc_now().isoformat(),
                "side": side_n,
                "enqueue_time": float(enqueue_time),
                "send_time": float(send_time),
                "fill_time": float(fill_time),
                "expected_price": float(expected_price),
                "filled_price": float(filled_price),
                "slippage": float(slippage),
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
                if self._ensure_symbol_meta():
                    spread_points = 0.0
                    with MT5_LOCK:
                        t = mt5.symbol_info_tick(self.symbol)
                    if t:
                        spread_points = (float(t.ask) - float(t.bid)) / max(1e-9, float(self._symbol_meta.point))

                    self.execmon.on_execution(
                        ExecSample(
                            ts=time.time(),
                            side=side_n,
                            expected_price=float(expected_price),
                            filled_price=float(filled_price),
                            point=float(self._point),
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
            now = time.time()
            if (not force) and (now - self._survival_last_flush_ts) < flush_sec:
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
            self._flush_survival_active(force=True)  # Immediate flush for critical updates
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
