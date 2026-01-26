# StrategiesXau/risk_management.py  (PRODUCTION-GRADE / FAST / SAFE / PEP8)
from __future__ import annotations

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

from config_xau import EngineConfig, SymbolParams, GOLD_MARKET_END_MINUTES, GOLD_MARKET_START_MINUTES
from DataFeed.xau_market_feed import MicroZones, TickStats
from mt5_client import MT5_LOCK, ensure_mt5
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# ============================================================
# ERROR-only logging (isolated) + rotation
# ============================================================
LOG_DIR = LOG_ROOT

log_risk = logging.getLogger("risk_xau")
log_risk.setLevel(logging.ERROR)
log_risk.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log_risk.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("risk_manager_xau.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log_risk.addHandler(fh)

_FILE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    """Naive UTC datetime (tzinfo=None) for safe JSON/csv writes and date compares."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def percentile_rank(series: np.ndarray, value: float) -> float:
    """Percentile rank for finite values only. Returns 50.0 on any failure."""
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


_SIDE_MAP = {
    "buy": "Buy",
    "long": "Buy",
    "b": "Buy",
    "1": "Buy",
    "order_type_buy": "Buy",
    "op_buy": "Buy",
    "sell": "Sell",
    "short": "Sell",
    "s": "Sell",
    "-1": "Sell",
    "order_type_sell": "Sell",
    "op_sell": "Sell",
}


def _side_norm(side: str) -> str:
    s = str(side or "").strip().lower()
    if not s:
        return "Buy"
    if s in _SIDE_MAP:
        return _SIDE_MAP[s]
    if "buy" in s:
        return "Buy"
    if "sell" in s:
        return "Sell"
    return str(side)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -------------------- deterministic ladder (keep your logic) --------------------
_BASE_LOT = 0.02
_BASE_TP_USD = 2.0
_STEP_BALANCE = 100.0
_STEP_LOT = 0.01
_STEP_TP_USD = 1.0
_STEP_START_BALANCE = 200.0


def gold_lot_and_takeprofit(balance: float) -> Tuple[float, float]:
    """
    Balance-based ladder:
      1..199   -> lot=0.02, tp=2
      200..299 -> lot=0.03, tp=3
      300..399 -> lot=0.04, tp=4
      ... (each +100 balance adds +0.01 lot and +1 USD TP)
    """
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


def _atr_fallback(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR fallback (no TA-Lib)."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    p = max(1, int(period))
    alpha = 1.0 / float(p)
    out = np.empty_like(tr, dtype=np.float64)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


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
        Positive = worse execution:
          - Buy: filled выше expected => хуже => +
          - Sell: filled ниже expected => хуже => +
        """
        if self.point <= 0:
            return 0.0
        raw = (self.filled_price - self.expected_price) / self.point
        return float(raw) if _side_norm(self.side) == "Buy" else float(-raw)


class ExecutionQualityMonitor:
    """
    Rolling monitor for execution + quote anomalies.
    Делает только защиту: детект -> пауза/ужесточение.
    """

    def __init__(self, *, window: int = 300) -> None:
        self.window = int(window)

        self.samples: Deque[ExecSample] = deque(maxlen=self.window)
        self.spreads: Deque[float] = deque(maxlen=self.window)  # points
        self.mid_moves: Deque[float] = deque(maxlen=self.window)  # points per tick

        self._last_mid: Optional[float] = None

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
        """
        Пороги берём из cfg, если есть. Если отсутствуют — используем безопасные дефолты мониторинга.
        """
        s = self.snapshot()

        max_p95_lat = float(getattr(cfg, "exec_max_p95_latency_ms", 400.0))  # Ислоҳ: Аз 450 ба 400
        max_p95_slip = float(getattr(cfg, "exec_max_p95_slippage_points", 4.0))  # Ислоҳ: Аз 6 ба 4 барои slippage тангтар
        max_spread = float(getattr(cfg, "exec_max_spread_points", 20.0))  # Ислоҳ: Аз 25 ба 20
        max_ewma_slip = float(getattr(cfg, "exec_max_ewma_slippage_points", 3.0))  # Ислоҳ: Аз 4 ба 3

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


@dataclass(slots=True)
@dataclass
class RiskDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class PlanOrderResult:
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None
    ok: bool = False
    reason: str = ""


# ============================================================
# Signal survival: no CSV rewrites
# ============================================================
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
    """
    Risk management for XAUUSDm scalping on Exness MT5.

    Production guarantees:
      - No sleeps under MT5_LOCK
      - MT5 snapshot cache prevents temporary 0.0 glitches from breaking risk decisions
      - Atomic state writes
      - Signal survival logs are append-only / throttled
      - Heavy pandas usage removed from histograms (csv-based counters instead)
    """

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
        self._trading_disabled_until: float = 0.0
        self._hard_stop_reason: Optional[str] = None

        # Runtime
        self._current_drawdown = 0.0
        self._analysis_blocked_until: float = 0.0
        self._last_fill_ts: float = 0.0
        self._latency_bad_since: Optional[float] = None

        # Account snapshot cache
        self._acc_cache_ts: float = 0.0
        self._acc_cache_balance: float = 0.0
        self._acc_cache_equity: float = 0.0
        self._bot_balance_base: float = 0.0

        # Signal throttling
        self._daily_signal_count: int = 0
        self._hour_window_start_ts: float = time.time()
        self._hour_window_count: int = 0

        # MT5 readiness throttling
        self._ready_ts: float = 0.0
        self._ready_ok: bool = False

        # Symbol meta cache
        self._meta_ts: float = 0.0
        self._digits: int = 2
        self._point: float = 0.01
        self._vol_min: float = 0.01
        self._vol_max: float = 100.0
        self._vol_step: float = 0.01
        self._stops_level_points: int = 0
        self._freeze_level_points: int = 0

        # Execution quality breaker
        self.execmon = ExecutionQualityMonitor(window=int(getattr(self.cfg, "exec_window", 300)))
        self._exec_breaker_until: float = 0.0

        # Reconcile tracking
        self._last_open_positions: int = 0

        # Signal survival
        self._survival: Dict[str, SignalSurvivalState] = {}
        self._survival_last_flush_ts: float = 0.0
        self._survival_last_update_emit: Dict[str, float] = {}
        self._init_signal_survival()

        # lightweight analytics counters (csv rewrite is tiny and atomic)
        self._hist_slip_counts: Optional[Dict[str, int]] = None
        self._hist_delay_counts: Optional[Dict[str, int]] = None
        self._reject_counts: Optional[Dict[str, int]] = None

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

        def _in01(x: float, name: str) -> None:
            if x <= 0 or x >= 1:
                raise RuntimeError(f"EngineConfig.{name} must be in (0, 1)")

        _in01(float(self.cfg.max_drawdown), "max_drawdown")
        _in01(float(self.cfg.max_risk_per_trade), "max_risk_per_trade")
        _in01(float(self.cfg.max_daily_loss_pct), "max_daily_loss_pct")
        _in01(float(self.cfg.daily_target_pct), "daily_target_pct")

        ucm = float(self.cfg.ultra_confidence_min)
        mcs = float(self.cfg.min_confidence_signal)
        if ucm <= 0 or ucm > 1:
            raise RuntimeError("EngineConfig.ultra_confidence_min must be in (0, 1]")
        if mcs <= 0 or mcs > 1:
            raise RuntimeError("EngineConfig.min_confidence_signal must be in (0, 1]")

        fv = getattr(self.cfg, "fixed_volume", None)
        if fv is not None:
            try:
                fv_f = float(fv)
            except Exception as exc:
                raise RuntimeError("EngineConfig.fixed_volume must be float when provided") from exc
            if fv_f <= 0:
                raise RuntimeError("EngineConfig.fixed_volume must be > 0 when provided")

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

    def _enter_hard_stop(self, reason: str) -> None:
        self.current_phase = "C"
        self._hard_stop_reason = reason
        self._phase_reason_last = str(reason)
        self._target_breached = True
        lock = self._seconds_until_next_utc_day()
        now = time.time()
        self._trading_disabled_until = now + lock
        self._analysis_blocked_until = max(self._analysis_blocked_until, now + lock)

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
        self._trading_disabled_until = 0.0
        self._hard_stop_reason = None

        self._daily_signal_count = 0
        self._hour_window_start_ts = time.time()
        self._hour_window_count = 0
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
            if (now - self._ready_ts) < 1.0:
                return bool(self._ready_ok)

            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                if info is None:
                    self._ready_ok = False
                else:
                    if not info.visible:
                        self._ready_ok = bool(mt5.symbol_select(self.symbol, True))
                    else:
                        self._ready_ok = True

            self._ready_ts = now
            return bool(self._ready_ok)
        except Exception as exc:
            self._ready_ok = False
            self._ready_ts = time.time()
            log_risk.error("ensure_ready error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _symbol_meta(self) -> bool:
        if time.time() - self._meta_ts < 10.0:
            return True
        if not self._ensure_ready():
            return False
        try:
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return False

            self._digits = int(getattr(info, "digits", 2) or 2)
            self._point = float(getattr(info, "point", 0.01) or 0.01)
            self._vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            self._vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
            self._vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
            self._stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
            self._freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)

            self._meta_ts = time.time()
            return True
        except Exception as exc:
            log_risk.error("symbol_meta error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _min_stop_distance(self) -> float:
        if not self._symbol_meta():
            return 0.0
        pts = int(max(0, self._stops_level_points)) + int(max(0, self._freeze_level_points)) + 2
        return float(pts) * float(self._point)

    def _normalize_price(self, price: float) -> float:
        if not self._symbol_meta():
            return float(price)
        return float(round(float(price), int(self._digits)))

    def _normalize_volume_floor(self, vol: float) -> float:
        """Round DOWN to step so risk is never exceeded."""
        if not self._symbol_meta():
            return float(vol)

        step = max(1e-9, float(self._vol_step))
        v = max(float(self._vol_min), min(float(vol), float(self._vol_max)))
        v = math.floor(v / step) * step
        return float(round(v, 8))

    def _tp_usd_to_price(self, entry: float, side: str, volume: float, usd_profit: float) -> Optional[float]:
        """
        Convert desired TP in USD to a price using broker tick specs.
        Robust to symbols where trade_tick_value is 0 (uses profit/loss tick values).
        """
        try:
            if float(entry) <= 0 or float(volume) <= 0 or float(usd_profit) <= 0:
                return None
            if not self._ensure_ready():
                return None
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return None

            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = float(getattr(info, "point", 0.0) or 0.0)

            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tvp = float(getattr(info, "trade_tick_value_profit", 0.0) or 0.0)
            tvl = float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0)
            tick_value = max(tick_value, tvp, tvl)

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

            return float(tp)
        except Exception:
            return None

    def _fallback_volume(self) -> float:
        """
        Safe fallback volume when dynamic sizing cannot be computed.

        Rules:
          - If cfg.fixed_volume is set -> return normalized broker-safe volume.
          - If cfg.fixed_volume is None -> return 0.0 (skip trade).
        """
        fv = getattr(self.cfg, "fixed_volume", None)
        if fv is None:
            return 0.0
        try:
            return self._normalize_volume_floor(float(fv))
        except Exception:
            return 0.0

    # ------------------- quote/execution hooks -------------------
    def on_quote(self, bid: float, ask: float) -> None:
        try:
            if not self._symbol_meta():
                return
            self.execmon.on_quote(float(bid), float(ask), float(self._point))
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

        if (now - self._hour_window_start_ts) >= 3600.0:
            self._hour_window_start_ts = now
            self._hour_window_count = 0

        self._hour_window_count += 1
        self._daily_signal_count += 1

    # ------------------- account snapshot (cached) -------------------
    def _account_snapshot(self) -> Tuple[float, float]:
        """
        Single MT5 account_info() snapshot with short cache.
        Prevents temporary 0.0 reads from MT5 from triggering wrong risk decisions.
        """
        if bool(getattr(self.cfg, "ignore_external_positions", False)):
            return self._bot_account_snapshot()

        cache_sec = float(getattr(self.cfg, "account_snapshot_cache_sec", 0.5) or 0.5)
        now = time.time()

        if (now - float(self._acc_cache_ts)) < cache_sec:
            return float(self._acc_cache_balance), float(self._acc_cache_equity)

        if not self._ensure_ready():
            return float(self._acc_cache_balance), float(self._acc_cache_equity)

        try:
            with MT5_LOCK:
                acc = mt5.account_info()

            if acc:
                bal = float(getattr(acc, "balance", 0.0) or 0.0)
                eq = float(getattr(acc, "equity", 0.0) or 0.0)

                if bal > 0:
                    self._acc_cache_balance = bal
                if eq > 0:
                    self._acc_cache_equity = eq

                self._acc_cache_ts = now

        except Exception as exc:
            log_risk.error("_account_snapshot error: %s | tb=%s", exc, traceback.format_exc())

        return float(self._acc_cache_balance), float(self._acc_cache_equity)

    def _bot_account_snapshot(self) -> Tuple[float, float]:
        """
        Bot-only balance/equity snapshot (ignores manual/external positions).
        Uses magic filtering and a fixed base balance captured at daily reset.
        """
        cache_sec = float(getattr(self.cfg, "account_snapshot_cache_sec", 0.5) or 0.5)
        now = time.time()

        if (now - float(self._acc_cache_ts)) < cache_sec:
            return float(self._acc_cache_balance), float(self._acc_cache_equity)

        if not self._ensure_ready():
            return float(self._acc_cache_balance), float(self._acc_cache_equity)

        try:
            magic = int(getattr(self.cfg, "magic", 777001) or 777001)
        except Exception:
            magic = 777001

        try:
            with MT5_LOCK:
                acc = mt5.account_info()
                positions = mt5.positions_get()
                day_start = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                deals = mt5.history_deals_get(day_start, _utcnow())

            base = float(self._bot_balance_base)
            if base <= 0 and acc:
                base = float(getattr(acc, "balance", 0.0) or 0.0)
                if base > 0:
                    self._bot_balance_base = base

            realized = 0.0
            if deals:
                realized = float(
                    sum(
                        float(getattr(d, "profit", 0.0) or 0.0)
                        for d in deals
                        if int(getattr(d, "magic", 0) or 0) == magic
                    )
                )

            open_pnl = 0.0
            if positions:
                open_pnl = float(
                    sum(
                        float(getattr(p, "profit", 0.0) or 0.0)
                        for p in positions
                        if int(getattr(p, "magic", 0) or 0) == magic
                    )
                )

            bal = max(0.0, base + realized)
            eq = max(0.0, bal + open_pnl)

            if bal > 0:
                self._acc_cache_balance = bal
            if eq > 0:
                self._acc_cache_equity = eq

            self._acc_cache_ts = now

        except Exception as exc:
            log_risk.error("_bot_account_snapshot error: %s | tb=%s", exc, traceback.format_exc())

        return float(self._acc_cache_balance), float(self._acc_cache_equity)

    def _get_balance(self) -> float:
        bal, _ = self._account_snapshot()
        return float(bal)

    def _get_equity(self) -> float:
        _, eq = self._account_snapshot()
        return float(eq)

    # ------------------- state evaluation (Phase A/B/C) -------------------
    def evaluate_account_state(self) -> None:
        try:
            # 1. Check for Daily Reset (UTC)
            today = self._utc_date()
            if today != self.daily_date:
                self._reset_daily_state()

            # 2. Update snapshots
            bal, eq = self._account_snapshot()

            # 3. Initialize Anchor if missing (e.g. first run of the day)
            if self.daily_start_balance <= 0:
                base = bal if bal > 0 else eq
                if base > 0:
                    self.daily_start_balance = float(base)
                    log_risk.info("DAILY_ANCHOR_INIT | start_balance=%.2f", self.daily_start_balance)

            # 4. Calculate Stats
            if bal > 0:
                self._current_drawdown = max(0.0, float((bal - eq) / bal))

            self.daily_peak_equity = max(self.daily_peak_equity, eq)

            daily_return = 0.0
            if self.daily_start_balance > 0:
                daily_return = float((eq - self.daily_start_balance) / self.daily_start_balance)

            # Log periodically (every ~1 min) or on significant change could be nice, but keep it clean for now.
            # 5. Phase Logic
            # A -> B (Target Reached)
            if (not self._target_breached) and daily_return >= float(self.cfg.daily_target_pct):
                if self.current_phase != "B":
                    self.current_phase = "B"
                    self._target_breached = True
                    self._target_breached_return = float(daily_return)
                    self._phase_reason_last = "daily_target_reached"
                    log_risk.error(
                        "PHASE_CHANGE | A->B | reason=daily_target_reached daily_return=%.4f target=%.4f",
                        daily_return,
                        float(self.cfg.daily_target_pct),
                    )

            # A/B -> C by loss thresholds (optional)
            if bool(getattr(self.cfg, "enforce_daily_limits", False)):
                loss_b = float(getattr(self.cfg, "daily_loss_b_pct", 0.0) or 0.0)
                loss_c = float(getattr(self.cfg, "daily_loss_c_pct", 0.0) or 0.0)

                # Сначала проверяем loss_c (более критичный) - если превышен, сразу в C
                if loss_c > 0 and daily_return <= -loss_c and self.current_phase != "C":
                    self.current_phase = "C"
                    self._phase_reason_last = "daily_loss_c"
                    log_risk.error(
                        "PHASE_CHANGE | ->C | reason=daily_loss_c daily_return=%.4f loss_c=%.4f",
                        daily_return,
                        loss_c,
                    )
                # Если loss_c не превышен, но превышен loss_b - переходим в B
                elif loss_b > 0 and daily_return <= -loss_b and self.current_phase == "A":
                    self.current_phase = "B"
                    self._phase_reason_last = "daily_loss_b"
                    log_risk.error(
                        "PHASE_CHANGE | A->B | reason=daily_loss_b daily_return=%.4f loss_b=%.4f",
                        daily_return,
                        loss_b,
                    )

                # target lock: after overshoot, if returns back to target -> hard stop (Profit Protection)
                if self._target_breached and self.current_phase == "B":
                    self._target_breached_return = max(float(self._target_breached_return), float(daily_return))
                    target_pct = float(self.cfg.daily_target_pct)
                    
                    # === FIX: Add buffer before protection kicks in ===
                    # Only apply protection if we went MEANINGFULLY above target (>= target + 2%)
                    profit_buffer = 0.02  # 2% buffer above target before protection kicks in
                    
                    # 1. Standard Lock: if we went ABOVE target+buffer, and now dropped BELOW target -> STOP
                    if self._target_breached_return >= (target_pct + profit_buffer) and daily_return < target_pct:
                         self._enter_hard_stop("profit_protection_target_lost")

                    # 2. Trailing Profit Stop (User Request: "if 11, 12, 13 ... and drops -> STOP")
                    # Only apply if we've exceeded the target by the buffer amount
                    prot_pct = float(getattr(self.cfg, "protect_drawdown_from_peak_pct", 0.05) or 0.05) 
                    
                    # Calc drop from peak return
                    drop_from_peak = self._target_breached_return - daily_return
                    
                    # Only apply if we are well into profit (above target + buffer)
                    if self._target_breached_return >= (target_pct + profit_buffer) and drop_from_peak >= prot_pct:
                        self._enter_hard_stop(f"profit_protection_drop:{drop_from_peak:.1%}")

        except Exception as exc:
            log_risk.error("evaluate_account_state error: %s | tb=%s", exc, traceback.format_exc())

    def update_phase(self) -> None:
        self.evaluate_account_state()

    # ------------------- market hours -------------------
    def market_open_24_5(self) -> bool:
        try:
            start_min = int(getattr(self.cfg, "gold_market_start_minutes", GOLD_MARKET_START_MINUTES))
            end_min = int(getattr(self.cfg, "gold_market_end_minutes", GOLD_MARKET_END_MINUTES))

            now = _utcnow()
            wd = now.weekday()
            minutes = now.hour * 60 + now.minute

            if wd == 0:
                return minutes >= start_min
            if wd in (1, 2, 3):
                return True
            if wd == 4:
                return minutes <= end_min
            return False
        except Exception as exc:
            log_risk.error("market_open_24_5 error: %s | tb=%s", exc, traceback.format_exc())
            return True

    def rollover_blackout(self) -> bool:
        sec = float(getattr(self.cfg, "rollover_blackout_sec", 0.0) or 0.0)
        if sec <= 0:
            return False
        now = _utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = abs((now - midnight).total_seconds())
        return delta <= sec

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

    # ------------------- ATR percentile -------------------
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
            if not self._symbol_meta():
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
        tz,
    ) -> RiskDecision:
        try:
            _ = (ingest_ms, last_bar_age, tz)

            self.evaluate_account_state()
            reasons: List[str] = []

            if self.requires_hard_stop():
                reasons.append(self.hard_stop_reason() or "hard_stop")

            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                reasons.append("phase_c_protect")
            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)) and drawdown_exceeded:
                reasons.append("max_drawdown")

            if spread_pct > float(self.sp.spread_limit_pct):
                reasons.append("spread")

            if not tick_ok and tick_reason not in ("tps_low", "no_rates"):
                reasons.append(f"micro:{tick_reason}")

            if not bool(getattr(self.cfg, "ignore_sessions", False)) and not in_session:
                reasons.append("session")

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
                if (now - self._hour_window_start_ts) >= 3600.0:
                    self._hour_window_start_ts = now
                    self._hour_window_count = 0
                if self._hour_window_count >= per_hour:
                    reasons.append("hourly_signal_limit")

            if spread_pct > float(self.sp.spread_limit_pct) * 1.5:
                reasons.append("excessive_spread")

            if not latency_cooldown:
                reasons.append("latency")

            return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)
        except Exception as exc:
            log_risk.error("guard_decision error: %s | tb=%s", exc, traceback.format_exc())
            return RiskDecision(allowed=False, reasons=["guard_error"])

    def can_trade(self, confidence: float, signal_type: str) -> RiskDecision:
        try:
            _ = signal_type
            reasons: List[str] = []

            self.evaluate_account_state()
            phase = str(self.current_phase or "A")

            if self.requires_hard_stop():
                reasons.append(self.hard_stop_reason() or "hard_stop")
                return RiskDecision(False, reasons)

            if phase == "C":
                reasons.append("phase_c_protect")
                return RiskDecision(False, reasons)

            if not self.market_open_24_5():
                reasons.append("session_closed")

            if self.rollover_blackout():
                reasons.append("rollover_blackout")

            conf = float(confidence)
            if phase == "A":
                if conf < float(self.cfg.min_confidence_signal):
                    reasons.append("low_confidence")
            elif phase == "B":
                ultra_min = float(self.cfg.ultra_confidence_min)
                if conf < ultra_min:
                    reasons.append(f"not_ultra_confidence (phase_b requires {ultra_min:.2f}, got {conf:.2f})")

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
                sec = float(getattr(self.cfg, "exec_breaker_sec", 90.0))
                self._exec_breaker_until = time.time() + sec
                reasons.extend(anoms)
                reasons.append("exec_breaker_triggered")
                return RiskDecision(False, reasons)

            return RiskDecision(allowed=(len(reasons) == 0), reasons=reasons)

        except Exception as exc:
            log_risk.error("can_trade error: %s | tb=%s", exc, traceback.format_exc())
            return RiskDecision(allowed=False, reasons=["can_trade_error"])

    def can_emit_signal(self, confidence: int, tz) -> Tuple[bool, List[str]]:
        try:
            _ = tz
            self.evaluate_account_state()

            reasons: List[str] = []

            conf = float(confidence)
            conf_f = conf / 100.0 if conf > 1.5 else conf
            conf_f = max(0.0, min(1.0, conf_f))

            phase = str(self.current_phase or "A")

            if phase == "C":
                reasons.append("phase_c_protect")
                return False, reasons

            if not self.can_analyze():
                reasons.append("analysis_cooldown")

            if phase == "A":
                if conf_f < float(self.cfg.min_confidence_signal):
                    reasons.append("low_confidence")
            elif phase == "B":
                ultra_min = float(self.cfg.ultra_confidence_min)
                if conf_f < ultra_min:
                    reasons.append(f"not_ultra_confidence (phase_b requires {ultra_min:.2f}, got {conf_f:.2f})")

            if bool(getattr(self.cfg, "enforce_drawdown_limits", False)):
                if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                    if conf_f < float(self.cfg.min_confidence_signal) + 0.1:
                        reasons.append("drawdown_strict_confidence")

            if time.time() < self._exec_breaker_until:
                reasons.append("exec_breaker_active")

            allowed = (len(reasons) == 0)
            return allowed, reasons
        except Exception as exc:
            log_risk.error("can_emit_signal error: %s | tb=%s", exc, traceback.format_exc())
            return False, ["signal_error"]

    # ------------------- broker-true sizing -------------------
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

    def _apply_broker_constraints(
        self, side: str, entry: float, sl: float, tp: float, ref_price: Optional[float] = None
    ) -> Tuple[float, float]:
        if not self._symbol_meta():
            return sl, tp

        side_n = _side_norm(side)
        min_dist = self._min_stop_distance()
        px = float(ref_price) if ref_price and ref_price > 0 else float(entry)

        if min_dist > 0:
            if side_n == "Buy":
                sl = min(sl, px - min_dist)
                tp = max(tp, px + min_dist)
            else:
                sl = max(sl, px + min_dist)
                tp = min(tp, px - min_dist)

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
        """
        Broker-true risk sizing.

        Safety contract:
          - Any invalid inputs or calc failures -> return 0.0 (skip trade) or fixed_volume fallback if configured.
        """
        try:
            self.evaluate_account_state()
            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                return 0.0

            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return 0.0

            ep = float(entry_price)
            sl = float(stop_loss)
            tp = float(take_profit)
            conf = float(confidence)

            if (not _is_finite(ep, sl, tp, conf)) or ep <= 0 or sl <= 0 or tp <= 0:
                return 0.0

            if not self._symbol_meta():
                return self._fallback_volume()

            min_rr = getattr(self.cfg, "min_rr", None)
            if min_rr is not None:
                rr_ratio = self._rr(ep, sl, tp)
                if rr_ratio < float(min_rr):
                    return 0.0

            _, eq = self._account_snapshot()
            if float(eq) <= 0:
                return self._fallback_volume()

            risk_money = float(eq) * float(self.cfg.max_risk_per_trade)

            if self.current_phase == "B":
                risk_money *= float(getattr(self.cfg, "phase_b_risk_mult", 1.0) or 1.0)
            if self._current_drawdown > float(self.cfg.max_drawdown) * 0.5:
                risk_money *= float(getattr(self.cfg, "dd_risk_mult", 1.0) or 1.0)

            risk_1lot = abs(self._order_calc_profit_money(side_n, 1.0, ep, sl))
            if risk_1lot <= 0:
                return self._fallback_volume()

            raw_lot = float(risk_money) / float(risk_1lot)

            if hasattr(self.cfg, "max_position_percentage"):
                cap_pct = float(getattr(self.cfg, "max_position_percentage")) / 100.0
                if cap_pct > 0:
                    margin_cap = float(eq) * cap_pct
                    margin_1lot = self._order_calc_margin(side_n, 1.0, ep)
                    if margin_1lot > 0:
                        raw_lot = min(raw_lot, float(margin_cap) / float(margin_1lot))

            lot = self._normalize_volume_floor(raw_lot)
            return float(lot) if lot > 0 and _is_finite(lot) else 0.0

        except Exception as exc:
            log_risk.error("calculate_position_size error: %s | tb=%s", exc, traceback.format_exc())
            return self._fallback_volume()

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

            base_pct = float(getattr(self.cfg, "micro_buffer_pct", 0.0005) or 0.0005)
            base_buffer = entry * base_pct

            vol_buffer = float(max(0.0, tick_volatility)) * 2.0
            micro_buffer = base_buffer + vol_buffer

            z = zones
            bid_floor = float(getattr(z, "bid_floor", 0.0) or 0.0) if z else 0.0
            ask_ceil = float(getattr(z, "ask_ceiling", 0.0) or 0.0) if z else 0.0

            side_n = _side_norm(side)

            if side_n == "Buy":
                sl = min(bid_floor, entry - micro_buffer) if bid_floor > 0 else (entry - micro_buffer)
                tp = max(ask_ceil, entry + micro_buffer * 2.0) if ask_ceil > 0 else (entry + micro_buffer * 2.5)
            else:
                sl = max(ask_ceil, entry + micro_buffer) if ask_ceil > 0 else (entry + micro_buffer)
                tp = min(bid_floor, entry - micro_buffer * 2.0) if bid_floor > 0 else (entry - micro_buffer * 2.5)

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
    ) -> PlanOrderResult:
        """
        Plan order (entry/sl/tp/lot).
        Returns PlanOrderResult(ok=False, reason="...") on failure.
        """
        try:
            _ = (ind, ticks)

            side_n = _side_norm(side)
            if side_n not in ("Buy", "Sell"):
                return PlanOrderResult(ok=False, reason="invalid_side")

            if not self._ensure_ready():
                return PlanOrderResult(ok=False, reason="risk_not_ready")

            self.evaluate_account_state()
            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                log_risk.error("plan_order BLOCKED | phase=C | reason=%s", self._phase_reason_last or "phase_c")
                return PlanOrderResult(ok=False, reason=f"phase_c:{self._phase_reason_last}")

            if entry is None:
                with MT5_LOCK:
                    t = mt5.symbol_info_tick(self.symbol)
                if not t:
                    return PlanOrderResult(ok=False, reason="no_tick_mt5")
                entry = float(t.ask) if side_n == "Buy" else float(t.bid)

            entry_price = float(entry)
            if not _is_finite(entry_price) or entry_price <= 0:
                return PlanOrderResult(ok=False, reason="invalid_price")

            # 1) ATR injection
            atr = 0.0
            try:
                atr = float((adapt or {}).get("atr", 0.0) or ind.get("atr", 0.0) or 0.0)
            except Exception:
                atr = 0.0

            adapt2 = dict(adapt or {})
            adapt2["atr"] = float(atr)

            # 2) Base ATR plan
            sl, tp = self._fallback_atr_sl_tp(side_n, entry_price, adapt2)

            # 3) If ATR missing, fallback to micro-zone (last resort)
            if (not _is_finite(sl, tp)) or sl == entry_price or tp == entry_price:
                sl, tp = self._micro_zone_sl_tp(side_n, entry_price, zones, float(tick_volatility))

            # 4) Cost-floor from broker constraints + exec monitor
            min_dist = self._min_stop_distance()
            min_sl_pct = float(getattr(self.cfg, "min_sl_pct", 0.00045) or 0.00045)
            min_tp_pct = float(getattr(self.cfg, "min_tp_pct", 0.00055) or 0.00055)

            cost_floor = float(min_dist)
            try:
                snap = self.execmon.snapshot()
                sp_pts = max(
                    float(snap.get("p95_spread_points", 0.0) or 0.0),
                    float(snap.get("ewma_spread_points", 0.0) or 0.0),
                )
                slp_pts = max(
                    float(snap.get("p95_slippage_points", 0.0) or 0.0),
                    float(snap.get("ewma_slippage_points", 0.0) or 0.0),
                )
                mv_pts = float(snap.get("p95_mid_move_points", 0.0) or 0.0)

                if self._symbol_meta() and float(self._point) > 0:
                    sp_px = sp_pts * float(self._point)
                    slp_px = slp_pts * float(self._point)
                    mv_px = mv_pts * float(self._point)
                else:
                    sp_px = slp_px = mv_px = 0.0

                k_sp = float(getattr(self.cfg, "sltp_cost_spread_mult", 1.8) or 1.8)
                k_slp = float(getattr(self.cfg, "sltp_cost_slip_mult", 1.0) or 1.0)
                k_mv = float(getattr(self.cfg, "sltp_cost_move_mult", 0.6) or 0.6)

                cost_floor = max(float(min_dist), sp_px * k_sp + slp_px * k_slp + mv_px * k_mv)
            except Exception:
                cost_floor = float(min_dist)

            sl_floor_mult = float(getattr(self.cfg, "sltp_sl_floor_mult", 1.15) or 1.15)
            tp_floor_mult = float(getattr(self.cfg, "sltp_tp_floor_mult", 1.75) or 1.75)

            min_sl_atr_floor = float(getattr(self.cfg, "min_sl_atr_floor", 0.55) or 0.55) * float(atr)
            min_tp_atr_floor = float(getattr(self.cfg, "min_tp_atr_floor", 0.65) or 0.65) * float(atr)

            min_sl = max(float(min_dist), entry_price * min_sl_pct, cost_floor * sl_floor_mult, min_sl_atr_floor)
            min_tp = max(float(min_dist), entry_price * min_tp_pct, cost_floor * tp_floor_mult, min_tp_atr_floor)

            if not _is_finite(sl, tp):
                return PlanOrderResult(ok=False, reason="sl_tp_calc_failed_nonfinite")

            # 5) Enforce floors
            if side_n == "Buy":
                if abs(entry_price - float(sl)) < min_sl:
                    sl = self._normalize_price(entry_price - min_sl)
                if abs(float(tp) - entry_price) < min_tp:
                    tp = self._normalize_price(entry_price + min_tp)
            else:
                if abs(entry_price - float(sl)) < min_sl:
                    sl = self._normalize_price(entry_price + min_sl)
                if abs(float(tp) - entry_price) < min_tp:
                    tp = self._normalize_price(entry_price - min_tp)

            # 6) Micro-zones only EXTEND
            if zones is not None:
                slz, tpz = self._micro_zone_sl_tp(side_n, entry_price, zones, float(tick_volatility))
                if _is_finite(slz, tpz) and slz != entry_price and tpz != entry_price:
                    if side_n == "Buy":
                        sl = min(float(sl), float(slz))
                        tp = max(float(tp), float(tpz))
                    else:
                        sl = max(float(sl), float(slz))
                        tp = min(float(tp), float(tpz))

            if not _is_finite(sl, tp) or sl == entry_price or tp == entry_price:
                return PlanOrderResult(ok=False, reason="sl_tp_calc_failed_no_dist")

            sl, tp = self._apply_broker_constraints(side_n, entry_price, float(sl), float(tp))

            if side_n == "Buy" and not (sl < entry_price < tp):
                return PlanOrderResult(ok=False, reason="invalid_sl_tp_buy")
            if side_n == "Sell" and not (tp < entry_price < sl):
                return PlanOrderResult(ok=False, reason="invalid_sl_tp_sell")

            # ---- lot/TP (your deterministic ladder + USD-TP conversion) ----
            self.evaluate_account_state()
            if self.current_phase == "C" and not bool(getattr(self.cfg, "ignore_daily_stop_for_trading", False)):
                log_risk.error(
                    "plan_order BLOCKED (lot/TP calc) | phase=C | reason=%s",
                    self._phase_reason_last or "phase_c",
                )
                return PlanOrderResult(ok=False, reason=f"phase_c:{self._phase_reason_last}")

            use_usd_tp = False
            balance = float(self._get_balance())
            base_lot, tp_usd = gold_lot_and_takeprofit(balance) if balance > 0 else (0.0, 0.0)
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
                        tp = (
                            self._normalize_price(entry_price + dist * rr_min)
                            if side_n == "Buy"
                            else self._normalize_price(entry_price - dist * rr_min)
                        )
                        sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

                rr_cap = float(getattr(self.cfg, "tp_rr_cap", 0.0) or 0.0)
                if rr_cap > 0:
                    dist_sl = abs(entry_price - sl)
                    dist_tp = abs(tp - entry_price)
                    if dist_sl > 0 and dist_tp > dist_sl * rr_cap:
                        tp = (
                            self._normalize_price(entry_price + dist_sl * rr_cap)
                            if side_n == "Buy"
                            else self._normalize_price(entry_price - dist_sl * rr_cap)
                        )
                        if abs(tp - entry_price) < min_tp:
                            tp = (
                                self._normalize_price(entry_price + min_tp)
                                if side_n == "Buy"
                                else self._normalize_price(entry_price - min_tp)
                            )
                        sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

                lot = self.calculate_position_size(side_n, entry_price, sl, tp, float(confidence))
                if lot <= 0 or not _is_finite(lot):
                    return PlanOrderResult(ok=False, reason="invalid_lot_size")

            # multi-order logic (keep semantics; no SL tightening)
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
                    tp = (
                        self._normalize_price(entry_price + abs(tp - entry_price) * tp_bonus)
                        if side_n == "Buy"
                        else self._normalize_price(entry_price - abs(tp - entry_price) * tp_bonus)
                    )
                    sl, tp = self._apply_broker_constraints(side_n, entry_price, sl, tp)

            # Final check
            if not _is_finite(sl, tp, lot):
                return PlanOrderResult(ok=False, reason="final_calc_nonfinite")

            return PlanOrderResult(
                ok=True,
                entry=entry_price,
                sl=float(sl),
                tp=float(tp),
                lot=float(lot),
                reason="ok"
            )

        except Exception as exc:
            log_risk.error("plan_order exception: %s", exc, exc_info=True)
            return PlanOrderResult(ok=False, reason=f"exception:{type(exc).__name__}")

    # ------------------- trade recording -------------------
    def record_trade(self, *args, **kwargs) -> None:
        try:
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
            _ = spread
            _ = float(slippage)
        except Exception as exc:
            log_risk.error("update_market_conditions error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- execution metrics -------------------
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
        try:
            side_n = _side_norm(side)
            point = float(self._point) if self._symbol_meta() else 0.0
            slip_points = float(slippage) / max(1e-9, point) if point > 0 else float(slippage)

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

            with _FILE_LOCK:
                csv_file = LOG_DIR / "execution_quality_xau.csv"
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

                jsonl_file = LOG_DIR / "execution_metrics_xau.jsonl"
                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            # feed exec monitor
            try:
                spread_points = 0.0
                if self._symbol_meta() and float(self._point) > 0:
                    with MT5_LOCK:
                        t = mt5.symbol_info_tick(self.symbol)
                    if t:
                        spread_points = (float(t.ask) - float(t.bid)) / max(1e-9, float(self._point))

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

            with _FILE_LOCK:
                csv_file = LOG_DIR / "execution_failures_xau.csv"
                write_header = not csv_file.exists()
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

            self._update_rejection_analysis(str(reason))
        except Exception as exc:
            log_risk.error("record_execution_failure error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- lightweight histograms (NO pandas) -------------------
    @staticmethod
    def _slip_bins() -> List[float]:
        return [-float("inf"), -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, float("inf")]

    @staticmethod
    def _delay_bins() -> List[float]:
        return [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000, float("inf")]

    @staticmethod
    def _bin_label(a: float, b: float) -> str:
        return f"{a} to {b}"

    def _load_histogram_counts(self, path: Path, bins: List[float]) -> Dict[str, int]:
        counts: Dict[str, int] = {self._bin_label(bins[i], bins[i + 1]): 0 for i in range(len(bins) - 1)}
        if not path.exists():
            return counts
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    k = str(row.get("bin", "")).strip()
                    if k in counts:
                        try:
                            counts[k] = int(float(row.get("count", 0) or 0))
                        except Exception:
                            counts[k] = 0
        except Exception:
            return counts
        return counts

    def _write_histogram_counts_atomic(self, path: Path, counts: Dict[str, int]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bin", "count"])
            w.writeheader()
            for k, v in counts.items():
                w.writerow({"bin": k, "count": int(v)})
        tmp.replace(path)

    def _update_execution_analysis(self, metrics: dict) -> None:
        try:
            s = float(metrics.get("slippage_points", 0.0) or 0.0)
            d = float(metrics.get("latency_send_to_fill_ms", 0.0) or 0.0)

            with _FILE_LOCK:
                # slippage histogram
                slippage_file = LOG_DIR / "slippage_histogram_xau.csv"
                bins = self._slip_bins()
                if self._hist_slip_counts is None:
                    self._hist_slip_counts = self._load_histogram_counts(slippage_file, bins)

                for i in range(len(bins) - 1):
                    if bins[i] <= s < bins[i + 1]:
                        label = self._bin_label(bins[i], bins[i + 1])
                        self._hist_slip_counts[label] = int(self._hist_slip_counts.get(label, 0)) + 1
                        break
                self._write_histogram_counts_atomic(slippage_file, self._hist_slip_counts)

                # fill delay histogram
                delay_file = LOG_DIR / "fill_delay_histogram_xau.csv"
                dbins = self._delay_bins()
                if self._hist_delay_counts is None:
                    self._hist_delay_counts = self._load_histogram_counts(delay_file, dbins)

                for i in range(len(dbins) - 1):
                    if dbins[i] <= d < dbins[i + 1]:
                        label = self._bin_label(dbins[i], dbins[i + 1])
                        self._hist_delay_counts[label] = int(self._hist_delay_counts.get(label, 0)) + 1
                        break
                self._write_histogram_counts_atomic(delay_file, self._hist_delay_counts)

        except Exception as exc:
            log_risk.error("_update_execution_analysis error: %s | tb=%s", exc, traceback.format_exc())

    def _update_rejection_analysis(self, reason: str) -> None:
        try:
            reject_file = LOG_DIR / "reject_rate_analysis_xau.csv"
            with _FILE_LOCK:
                if self._reject_counts is None:
                    self._reject_counts = {}
                    if reject_file.exists():
                        try:
                            with open(reject_file, "r", encoding="utf-8", newline="") as f:
                                r = csv.DictReader(f)
                                for row in r:
                                    rs = str(row.get("reason", "")).strip()
                                    if not rs:
                                        continue
                                    try:
                                        self._reject_counts[rs] = int(float(row.get("count", 0) or 0))
                                    except Exception:
                                        self._reject_counts[rs] = 0
                        except Exception:
                            self._reject_counts = {}

                key = str(reason or "unknown").strip() or "unknown"
                self._reject_counts[key] = int(self._reject_counts.get(key, 0)) + 1

                total = int(sum(self._reject_counts.values()))
                tmp = reject_file.with_suffix(reject_file.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["reason", "count", "total_attempts", "reject_rate"])
                    w.writeheader()
                    for rs, cnt in sorted(self._reject_counts.items(), key=lambda x: (-x[1], x[0])):
                        rate = float(cnt) / max(1, total)
                        w.writerow(
                            {
                                "reason": rs,
                                "count": int(cnt),
                                "total_attempts": int(total),
                                "reject_rate": float(rate),
                            }
                        )
                tmp.replace(reject_file)

        except Exception as exc:
            log_risk.error("_update_rejection_analysis error: %s | tb=%s", exc, traceback.format_exc())

    # ------------------- signal survival -------------------
    def _survival_paths(self) -> Tuple[Path, Path, Path]:
        final_csv = Path(getattr(self.cfg, "signal_survival_log_file", str(LOG_DIR / "signal_survival_final_xau.csv")))
        active_json = Path(getattr(self.cfg, "signal_survival_active_file", str(LOG_DIR / "signal_survival_active_xau.json")))
        updates_jsonl = Path(
            getattr(self.cfg, "signal_survival_updates_jsonl_file", str(LOG_DIR / "signal_survival_updates_xau.jsonl"))
        )
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
            self._flush_survival_active(force=False)
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
            vol_factor = min(2.0, max(0.5, volatility / 20.0))
            latency_factor = min(1.5, max(0.1, latency_ms / 100.0))

            total_slippage = spread_slippage + spread_slippage * vol_factor + spread_slippage * latency_factor
            partial_fill_prob = min(0.3, max(0.05, (volatility / 50.0) + (latency_ms / 200.0)))

            if is_backtest:
                total_slippage *= 1.2
                partial_fill_prob *= 1.1
            elif is_dry_run:
                total_slippage *= 0.8
                partial_fill_prob *= 0.9

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

            with _FILE_LOCK:
                csv_file = LOG_DIR / "slippage_model_xau.csv"
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
        return Path(getattr(self.cfg, "risk_state_file", str(LOG_DIR / "risk_state_xau.json")))

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
            "_current_drawdown": float(self._current_drawdown),
            "_analysis_blocked_until": float(self._analysis_blocked_until),
            "_last_fill_ts": float(self._last_fill_ts),
            "_latency_bad_since": float(self._latency_bad_since or 0.0),
            "_daily_signal_count": int(self._daily_signal_count),
            "_hour_window_start_ts": float(self._hour_window_start_ts),
            "_hour_window_count": int(self._hour_window_count),
            "_exec_breaker_until": float(self._exec_breaker_until),
            "execmon": {
                "ewma_slip": float(self.execmon.ewma_slip or 0.0),
                "ewma_lat": float(self.execmon.ewma_lat or 0.0),
                "ewma_spread": float(self.execmon.ewma_spread or 0.0),
            },
        }

        with _FILE_LOCK:
            _atomic_write_text(p, json.dumps(state, ensure_ascii=False))

    def load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return

        with _FILE_LOCK:
            raw = p.read_text(encoding="utf-8")
        st = json.loads(raw)

        saved_date = str(st.get("daily_date", "")).strip()
        if saved_date and saved_date != str(self._utc_date()):
            self._exec_breaker_until = float(st.get("_exec_breaker_until", 0.0) or 0.0)
            em = st.get("execmon") or {}
            self.execmon.ewma_slip = float(em.get("ewma_slip", 0.0) or 0.0)
            self.execmon.ewma_lat = float(em.get("ewma_lat", 0.0) or 0.0)
            self.execmon.ewma_spread = float(em.get("ewma_spread", 0.0) or 0.0)

            self.daily_date = self._utc_date()
            self._trading_disabled_until = 0.0
            self._hard_stop_reason = None
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
        self._current_drawdown = float(st.get("_current_drawdown", self._current_drawdown) or 0.0)
        self._analysis_blocked_until = float(st.get("_analysis_blocked_until", self._analysis_blocked_until) or 0.0)
        self._last_fill_ts = float(st.get("_last_fill_ts", self._last_fill_ts) or 0.0)
        lbs = float(st.get("_latency_bad_since", 0.0) or 0.0)
        self._latency_bad_since = lbs if lbs > 0 else None
        self._daily_signal_count = int(st.get("_daily_signal_count", self._daily_signal_count) or 0)
        self._hour_window_start_ts = float(st.get("_hour_window_start_ts", self._hour_window_start_ts) or time.time())
        self._hour_window_count = int(st.get("_hour_window_count", self._hour_window_count) or 0)
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

