# core/data_integrity.py - data integrity and market validation.

from __future__ import annotations

# ---- merged from core/data_integrity.py ----
import logging
import math
import threading
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import feature_engine as _feature_engine

_log_sync = logging.getLogger("core.clock_sync")

DATA_VALID = "data valid"
DATA_STALE = "data stale"
DATA_INCOMPLETE = "data incomplete"
ABNORMAL_SPREAD = "abnormal spread"
MARKET_UNUSABLE = "market unusable"

for _feature_export in _feature_engine.__all__:
    globals()[_feature_export] = getattr(_feature_engine, _feature_export)


# =============================================================================
# ServerClockSync — Institutional-grade MT5↔Local clock drift compensation
# =============================================================================


class ServerClockSync:
    """
    Measures and compensates for clock drift between the local machine and
    the MT5 broker server.  Institutional trading desks call this
    "clock arbitrage prevention" — a mismatch of even 1-2 seconds can cause
    the data-integrity layer to flag perfectly valid bars as `future_leakage`.

    Design:
      - Probes MT5 server time via ``symbol_info_tick().time``.
      - Computes ``drift = server_epoch - local_epoch`` with EWMA smoothing
        (α = 0.3) to dampen one-off outliers.
      - Caches the drift for ``ttl_sec`` (default 10 s) to avoid flooding MT5.
      - Falls back to drift = 0 if MT5 is unavailable.

    Usage:
        sync = ServerClockSync()
        server_now = sync.now()          # datetime in UTC, server-aligned
        drift_sec  = sync.drift_seconds  # current estimated drift
    """

    _EWMA_ALPHA: float = 0.30

    def __init__(
        self,
        *,
        probe_symbol: str = "XAUUSDm",
        ttl_sec: float = 10.0,
        max_drift_sec: float = 120.0,
    ) -> None:
        self._probe_symbol = str(probe_symbol or "XAUUSDm")
        self._ttl_sec = max(1.0, float(ttl_sec))
        self._max_drift_sec = max(5.0, float(max_drift_sec))
        self._drift: float = 0.0
        self._last_probe_ts: float = 0.0
        self._lock = threading.Lock()
        self._mt5 = None  # lazy
        self._mt5_lock = None  # lazy
        self._initialized = False
        self._probe_failures: int = 0

    # ── Public API ──────────────────────────────────────────────

    @property
    def drift_seconds(self) -> float:
        """Current estimated drift (server − local) in seconds."""
        self._refresh_if_stale()
        return self._drift

    def now(self) -> datetime:
        """Return ``datetime.now(UTC) + drift`` — i.e. the server's 'now'."""
        self._refresh_if_stale()
        return datetime.now(timezone.utc) + timedelta(seconds=self._drift)

    def now_epoch(self) -> float:
        """Server-aligned epoch seconds."""
        self._refresh_if_stale()
        return _time.time() + self._drift

    def is_drift_exceeded(self, threshold_sec: float = 3.0) -> Tuple[bool, float]:
        """
        Hard-gate check for trading loops.

        Returns ``(exceeded, abs_drift_seconds)`` where ``exceeded`` is True
        when the absolute drift is greater than ``threshold_sec``. A positive
        result means the local clock and the broker are out of sync enough
        that any order we send will execute against stale prices — the
        calling layer MUST pause trading until drift returns to within
        tolerance.
        """
        self._refresh_if_stale()
        abs_drift = abs(float(self._drift))
        try:
            threshold = max(0.5, float(threshold_sec))
        except Exception:
            threshold = 3.0
        return (abs_drift > threshold), abs_drift

    # ── Internal ────────────────────────────────────────────────

    def _ensure_mt5(self) -> bool:
        if self._mt5 is not None:
            return True
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5
        except ImportError:
            return False
        try:
            from mt5_client import MT5_LOCK

            self._mt5_lock = MT5_LOCK
        except ImportError:
            self._mt5_lock = threading.RLock()
        self._initialized = True
        return True

    def _refresh_if_stale(self) -> None:
        now_local = _time.time()
        if (now_local - self._last_probe_ts) < self._ttl_sec:
            return
        with self._lock:
            if (now_local - self._last_probe_ts) < self._ttl_sec:
                return  # another thread beat us
            self._probe(now_local)

    def _probe(self, now_local: float) -> None:
        if not self._ensure_mt5():
            self._last_probe_ts = now_local
            return

        server_epoch: Optional[float] = None
        try:
            lock = self._mt5_lock or threading.RLock()
            with lock:
                tick = self._mt5.symbol_info_tick(self._probe_symbol)
            if tick is not None:
                raw = getattr(tick, "time", None)
                if raw is not None and float(raw) > 1_000_000_000:
                    server_epoch = float(raw)
        except Exception:
            pass

        # Fallback: try a second common symbol
        if server_epoch is None:
            for alt in ("BTCUSDm", "EURUSDm", "EURUSD"):
                try:
                    lock = self._mt5_lock or threading.RLock()
                    with lock:
                        tick = self._mt5.symbol_info_tick(alt)
                    if tick is not None:
                        raw = getattr(tick, "time", None)
                        if raw is not None and float(raw) > 1_000_000_000:
                            server_epoch = float(raw)
                            break
                except Exception:
                    continue

        self._last_probe_ts = _time.time()  # update after probe (includes latency)

        if server_epoch is None:
            self._probe_failures += 1
            if self._probe_failures <= 3:
                _log_sync.warning(
                    "CLOCK_SYNC_PROBE_FAIL | symbol=%s failures=%d drift_kept=%.3f",
                    self._probe_symbol,
                    self._probe_failures,
                    self._drift,
                )
            return

        self._probe_failures = 0
        local_at_probe = _time.time()
        raw_drift = server_epoch - local_at_probe

        # Sanity: reject insane drifts (> max_drift_sec)
        if abs(raw_drift) > self._max_drift_sec:
            _log_sync.warning(
                "CLOCK_SYNC_DRIFT_EXTREME | raw=%.3fs max=%.1fs — clamping",
                raw_drift,
                self._max_drift_sec,
            )
            raw_drift = max(-self._max_drift_sec, min(self._max_drift_sec, raw_drift))

        # EWMA smoothing
        if abs(self._drift) < 0.001 and not self._initialized:
            self._drift = raw_drift  # seed
        else:
            alpha = self._EWMA_ALPHA
            self._drift = alpha * raw_drift + (1.0 - alpha) * self._drift
        self._initialized = True

        _log_sync.info(
            "CLOCK_SYNC_OK | drift=%.3fs raw=%.3fs symbol=%s",
            self._drift,
            raw_drift,
            self._probe_symbol,
        )


_STATE_PRIORITY = {
    DATA_VALID: 0,
    DATA_INCOMPLETE: 1,
    DATA_STALE: 2,
    ABNORMAL_SPREAD: 3,
    MARKET_UNUSABLE: 4,
}

_TF_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}


def tf_seconds(tf: str) -> int:
    return int(_TF_SECONDS.get(str(tf or "").strip().upper(), 60))


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    message: str
    state: str


@dataclass(slots=True)
class AuditResult:
    state: str = DATA_VALID
    tradable: bool = True
    issues: List[AuditIssue] = field(default_factory=list)
    normalized_df: Optional[pd.DataFrame] = None
    missing_bars: int = 0
    duplicate_bars: int = 0
    largest_gap_sec: float = 0.0
    stale_tick_age_sec: float = 0.0
    spread_points: float = 0.0
    spread_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        code: str,
        message: str,
        state: str,
        **details: Any,
    ) -> None:
        issue = AuditIssue(code=code, message=message, state=state)
        if issue not in self.issues:
            self.issues.append(issue)
        if _STATE_PRIORITY[state] > _STATE_PRIORITY[self.state]:
            self.state = state
        self.tradable = (
            _STATE_PRIORITY.get(self.state, 0) < _STATE_PRIORITY[ABNORMAL_SPREAD]
        )
        if details:
            self.metadata[code] = details

    def merge(self, other: "AuditResult") -> "AuditResult":
        if other.normalized_df is not None and self.normalized_df is None:
            self.normalized_df = other.normalized_df
        self.missing_bars = max(int(self.missing_bars), int(other.missing_bars))
        self.duplicate_bars = max(
            int(self.duplicate_bars),
            int(other.duplicate_bars),
        )
        self.largest_gap_sec = max(
            float(self.largest_gap_sec),
            float(other.largest_gap_sec),
        )
        self.stale_tick_age_sec = max(
            float(self.stale_tick_age_sec),
            float(other.stale_tick_age_sec),
        )
        self.spread_points = max(
            float(self.spread_points),
            float(other.spread_points),
        )
        self.spread_pct = max(float(self.spread_pct), float(other.spread_pct))
        self.metadata.update(other.metadata)
        for issue in other.issues:
            self.add_issue(issue.code, issue.message, issue.state)
        return self

    def reason_codes(self) -> List[str]:
        reasons = [f"data_state:{self.state}"]
        for issue in self.issues:
            reasons.append(issue.code)
        return reasons


@dataclass(frozen=True, slots=True)
class SymbolSanityProfile:
    symbol: str
    expected_digits: int
    expected_point: float
    expected_contract_size: float
    spread_limit_pct: float
    max_spread_points: float
    tick_stale_threshold_sec: float
    allowed_clock_skew_sec: float
    gap_price_atr_mult: float
    spike_atr_mult: float
    spike_return_mult: float
    is_24_7: bool
    market_start_minutes: int
    market_end_minutes: int
    rollover_blackout_start: int
    rollover_blackout_end: int


class Validator:
    """
    Unified Phase-1 market-data validator.

    The validator is designed for every OHLCV dataframe that enters the signal
    path. It normalizes timestamps to UTC, enforces strict OHLC integrity, and
    exposes dedicated detectors for missing bars, stale ticks, gap anomalies,
    impossible spikes, spread anomalies, broker feed inconsistencies, and
    symbol-spec mismatches.
    """

    def __init__(
        self, cfg: Any, sp: Any, *, clock_sync: Optional[ServerClockSync] = None
    ) -> None:
        self.cfg = cfg
        self.sp = sp
        symbol = getattr(sp, "symbol", "") or getattr(sp, "resolved", "")
        self.symbol = str(symbol or getattr(sp, "base", "") or "").strip()
        self._profiles: Dict[str, SymbolSanityProfile] = {}
        self._spread_histories: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=256)
        )
        self._clock_sync = clock_sync

    def profile_for(
        self,
        symbol: Optional[str] = None,
        symbol_info: Any = None,
    ) -> SymbolSanityProfile:
        sym = str(symbol or self.symbol or getattr(self.sp, "base", "")).strip()
        cached = self._profiles.get(sym)
        if cached is not None:
            return cached

        expected_digits = int(getattr(self.sp, "digits", 0) or 0)
        observed_digits = int(getattr(symbol_info, "digits", 0) or 0)
        digits = expected_digits or observed_digits or 5

        expected_point = 10.0 ** (-digits)

        profile = SymbolSanityProfile(
            symbol=sym,
            expected_digits=digits,
            expected_point=float(expected_point),
            expected_contract_size=float(getattr(self.sp, "contract_size", 1.0) or 1.0),
            spread_limit_pct=float(getattr(self.sp, "spread_limit_pct", 0.0) or 0.0),
            max_spread_points=float(
                getattr(self.cfg, "max_spread_points", 0.0)
                or (1800.0 if "BTC" in sym.upper() else 350.0)
            ),
            tick_stale_threshold_sec=float(
                getattr(
                    self.cfg,
                    "tick_stale_threshold_sec",
                    max(10.0, float(tf_seconds(getattr(self.sp, "tf_primary", "M1")))),
                )
                or 60.0
            ),
            allowed_clock_skew_sec=float(
                getattr(self.cfg, "allowed_clock_skew_sec", 10.0) or 10.0
            ),
            gap_price_atr_mult=float(
                getattr(self.cfg, "integrity_gap_price_atr_mult", 6.0) or 6.0
            ),
            spike_atr_mult=float(
                getattr(self.cfg, "integrity_spike_atr_mult", 12.0) or 12.0
            ),
            spike_return_mult=float(
                getattr(self.cfg, "integrity_spike_return_mult", 15.0) or 15.0
            ),
            is_24_7=bool(getattr(self.sp, "is_24_7", False)),
            market_start_minutes=int(getattr(self.sp, "market_start_minutes", 0) or 0),
            market_end_minutes=int(
                getattr(self.sp, "market_end_minutes", 1440) or 1440
            ),
            rollover_blackout_start=int(
                getattr(self.sp, "rollover_blackout_start", 0) or 0
            ),
            rollover_blackout_end=int(
                getattr(self.sp, "rollover_blackout_end", 0) or 0
            ),
        )
        self._profiles[sym] = profile
        return profile

    def validate_ohlcv(
        self,
        df: Optional[pd.DataFrame],
        timeframe: str,
        *,
        symbol: Optional[str] = None,
        now: Optional[Any] = None,
    ) -> AuditResult:
        result = AuditResult(
            metadata={
                "symbol": str(symbol or self.symbol),
                "timeframe": str(timeframe),
            }
        )
        profile = self.profile_for(symbol=symbol)

        if not isinstance(df, pd.DataFrame) or df.empty:
            result.add_issue(
                "empty_dataframe",
                "OHLCV dataframe is empty",
                DATA_INCOMPLETE,
            )
            return result

        normalized, tz_issue = self._normalize_ohlcv(df)
        if normalized is None or normalized.empty:
            result.add_issue(
                "bad_ohlcv_schema",
                "Unable to normalize OHLCV dataframe",
                DATA_INCOMPLETE,
            )
            return result

        result.normalized_df = normalized

        if tz_issue:
            result.add_issue(
                "wrong_timezone",
                "Incoming timestamps are not strict UTC",
                DATA_INCOMPLETE,
            )

        if len(normalized) < 3:
            result.add_issue(
                "insufficient_bars",
                "Not enough bars for deterministic validation",
                DATA_INCOMPLETE,
            )
            return result

        duplicate_mask = normalized["time"].duplicated(keep=False)
        duplicate_count = int(duplicate_mask.sum())
        result.duplicate_bars = duplicate_count
        if duplicate_count > 0:
            result.add_issue(
                "duplicate_bars",
                f"Detected {duplicate_count} duplicate bars",
                DATA_INCOMPLETE,
                duplicate_bars=duplicate_count,
            )

        diffs = normalized["time"].diff().dt.total_seconds().dropna()
        if not diffs.empty and bool((diffs < 0.0).any()):
            result.add_issue(
                "bad_candle_order",
                "Bar timestamps must be strictly increasing",
                MARKET_UNUSABLE,
            )

        now_dt = self._coerce_now(now)
        # Institutional clock-sync: use adaptive server-aligned time
        # to prevent false future_leakage when MT5 server clock leads local
        if self._clock_sync is not None:
            sync_now = self._clock_sync.now()
            # Use whichever is later: synced server time or provided 'now'
            if sync_now > now_dt:
                now_dt = sync_now
        future_threshold = now_dt + timedelta(
            seconds=float(profile.allowed_clock_skew_sec)
        )
        if bool((normalized["time"] > future_threshold).any()):
            result.add_issue(
                "future_leakage",
                "Detected timestamps beyond the audit clock",
                DATA_STALE,
            )

        numeric_cols = ("open", "high", "low", "close", "tick_volume")
        numeric_df = normalized.loc[:, numeric_cols]
        if bool(numeric_df.isna().any().any()):
            result.add_issue(
                "nan_ohlcv",
                "OHLCV contains NaN values",
                DATA_INCOMPLETE,
            )
            return result

        self._assert_ohlc_integrity(normalized, result)

        missing_bars, largest_gap_sec = self._detect_missing_bars(
            normalized["time"],
            timeframe,
            profile,
        )
        result.missing_bars = missing_bars
        result.largest_gap_sec = largest_gap_sec
        if missing_bars > 0:
            result.add_issue(
                "missing_bars",
                f"Detected {missing_bars} missing bars",
                DATA_INCOMPLETE,
                missing_bars=missing_bars,
                largest_gap_sec=largest_gap_sec,
            )

        gap_anomalies = self.detect_gap_anomalies(
            normalized,
            timeframe,
            symbol=symbol,
        )
        if gap_anomalies:
            result.add_issue(
                "gap_anomaly",
                "Detected unjustifiable inter-bar price gaps",
                MARKET_UNUSABLE,
                gap_anomalies=gap_anomalies[:5],
            )

        spike_anomalies = self.detect_impossible_spikes(
            normalized,
            timeframe,
            symbol=symbol,
        )
        if spike_anomalies:
            result.add_issue(
                "impossible_spike",
                "Detected mathematically unjustifiable price spike",
                MARKET_UNUSABLE,
                impossible_spikes=spike_anomalies[:5],
            )

        return result

    def validate_market_context(
        self,
        df: pd.DataFrame,
        timeframe: str,
        *,
        symbol: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        tick_timestamp: Optional[Any] = None,
        symbol_info: Any = None,
        peer_bid: Optional[float] = None,
        peer_ask: Optional[float] = None,
        now: Optional[Any] = None,
    ) -> AuditResult:
        result = AuditResult(
            metadata={
                "symbol": str(symbol or self.symbol),
                "timeframe": str(timeframe),
            }
        )
        result.normalized_df = df

        spec_result = self._validate_symbol_spec(
            symbol_info=symbol_info,
            symbol=symbol,
        )
        result.merge(spec_result)

        stale_result = self.detect_stale_ticks(
            tick_timestamp=tick_timestamp,
            symbol=symbol,
            now=now,
        )
        result.merge(stale_result)

        spread_result = self._detect_spread_anomaly(
            bid=bid,
            ask=ask,
            symbol=symbol,
            now=now,
        )
        result.merge(spread_result)

        broker_result = self._detect_broker_feed_inconsistency(
            df=df,
            timeframe=timeframe,
            symbol=symbol,
            bid=bid,
            ask=ask,
            peer_bid=peer_bid,
            peer_ask=peer_ask,
        )
        result.merge(broker_result)

        return result

    def detect_stale_ticks(
        self,
        *,
        tick_timestamp: Optional[Any],
        symbol: Optional[str] = None,
        now: Optional[Any] = None,
    ) -> AuditResult:
        result = AuditResult(metadata={"symbol": str(symbol or self.symbol)})
        profile = self.profile_for(symbol=symbol)
        if tick_timestamp is None:
            return result

        tick_ts = self._coerce_timestamp(tick_timestamp)
        if tick_ts is None:
            result.add_issue(
                "bad_tick_timestamp",
                "Tick timestamp is invalid",
                DATA_STALE,
            )
            result.stale_tick_age_sec = math.inf
            return result

        age_sec = max(
            0.0,
            (self._coerce_now(now) - tick_ts).total_seconds(),
        )
        result.stale_tick_age_sec = float(age_sec)
        if age_sec > float(profile.tick_stale_threshold_sec):
            result.add_issue(
                "stale_ticks",
                (
                    "Latest tick is stale "
                    f"({age_sec:.3f}s > {profile.tick_stale_threshold_sec:.3f}s)"
                ),
                DATA_STALE,
                tick_age_sec=age_sec,
            )
        return result

    def detect_gap_anomalies(
        self,
        df: pd.DataFrame,
        timeframe: str,
        *,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if df is None or len(df) < 20:
            return []

        profile = self.profile_for(symbol=symbol)
        tf_sec = float(max(1, tf_seconds(timeframe)))

        o = df["open"].to_numpy(dtype=np.float64, copy=False)
        h = df["high"].to_numpy(dtype=np.float64, copy=False)
        low_arr = df["low"].to_numpy(dtype=np.float64, copy=False)
        c = df["close"].to_numpy(dtype=np.float64, copy=False)

        prev_close = c[:-1]
        curr_open = o[1:]
        gap_abs = np.abs(curr_open - prev_close)

        prev2 = c[:-2]
        high_1 = h[1:-1]
        low_1 = low_arr[1:-1]
        tr = np.maximum(
            high_1 - low_1,
            np.maximum(
                np.abs(high_1 - prev2),
                np.abs(low_1 - prev2),
            ),
        )
        baseline_tr = float(np.nanmedian(tr)) if tr.size > 0 else 0.0
        threshold = max(
            float(profile.expected_point) * 25.0,
            baseline_tr * float(profile.gap_price_atr_mult),
        )

        anomalies: List[Dict[str, Any]] = []
        for idx in range(1, len(df)):
            diff_sec = (df["time"].iat[idx] - df["time"].iat[idx - 1]).total_seconds()
            if gap_abs[idx - 1] <= threshold:
                continue
            if diff_sec > tf_sec and self._gap_is_expected(
                df["time"].iat[idx - 1],
                df["time"].iat[idx],
                tf_sec,
                profile,
            ):
                continue
            anomalies.append(
                {
                    "bar_index": int(idx),
                    "gap_abs": float(gap_abs[idx - 1]),
                    "threshold": float(threshold),
                    "time": df["time"].iat[idx].isoformat(),
                }
            )
        return anomalies

    def detect_impossible_spikes(
        self,
        df: pd.DataFrame,
        timeframe: str,
        *,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if df is None or len(df) < 30:
            return []

        profile = self.profile_for(symbol=symbol)
        tf_sec = float(max(1, tf_seconds(timeframe)))

        h = df["high"].to_numpy(dtype=np.float64, copy=False)
        low_arr = df["low"].to_numpy(dtype=np.float64, copy=False)
        c = df["close"].to_numpy(dtype=np.float64, copy=False)

        prev_close = c[:-1]
        tr = np.maximum(
            h[1:] - low_arr[1:],
            np.maximum(
                np.abs(h[1:] - prev_close),
                np.abs(low_arr[1:] - prev_close),
            ),
        )
        returns = np.abs(np.diff(c) / np.maximum(np.abs(prev_close), 1e-12))

        tr_ref = float(np.nanmedian(tr[:-1])) if tr.size > 1 else 0.0
        ret_ref = float(np.nanmedian(returns[:-1])) if returns.size > 1 else 0.0
        tr_threshold = max(
            float(profile.expected_point) * 50.0,
            tr_ref * float(profile.spike_atr_mult),
        )
        ret_threshold = max(
            float(profile.expected_point) * 100.0 / max(float(c[-1]), 1.0),
            ret_ref * float(profile.spike_return_mult),
            0.01,
        )

        anomalies: List[Dict[str, Any]] = []
        for idx in range(1, len(df)):
            diff_sec = (df["time"].iat[idx] - df["time"].iat[idx - 1]).total_seconds()
            tr_val = float(tr[idx - 1])
            ret_val = float(returns[idx - 1])
            if tr_val <= tr_threshold or ret_val <= ret_threshold:
                continue
            if diff_sec > tf_sec and self._gap_is_expected(
                df["time"].iat[idx - 1],
                df["time"].iat[idx],
                tf_sec,
                profile,
            ):
                continue
            anomalies.append(
                {
                    "bar_index": int(idx),
                    "true_range": tr_val,
                    "true_range_threshold": float(tr_threshold),
                    "return_abs": ret_val,
                    "return_threshold": float(ret_threshold),
                    "time": df["time"].iat[idx].isoformat(),
                }
            )
        return anomalies

    def _normalize_ohlcv(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        frame = df.copy()
        rename_map: Dict[str, str] = {}
        for col in frame.columns:
            lower = str(col).strip().lower()
            if lower in {"open", "high", "low", "close", "time"}:
                rename_map[col] = lower
            elif lower in {"volume", "tick_volume", "real_volume"}:
                rename_map[col] = "tick_volume"
        if rename_map:
            frame = frame.rename(columns=rename_map)

        if frame.columns.duplicated().any():
            frame = self._collapse_duplicate_columns(frame)

        if "time" not in frame.columns:
            index_series = pd.Series(frame.index, index=frame.index, name="time")
            frame = frame.copy()
            frame["time"] = index_series

        required = {"time", "open", "high", "low", "close", "tick_volume"}
        if not required.issubset(set(frame.columns)):
            return None, False

        tz_issue = self._timezone_issue(frame["time"])
        time_series = self._coerce_time_series(frame["time"])
        if time_series is None or bool(time_series.isna().any()):
            return None, tz_issue

        frame = frame.loc[
            :,
            ["time", "open", "high", "low", "close", "tick_volume"],
        ].copy()
        frame["time"] = time_series
        for col in ("open", "high", "low", "close", "tick_volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return frame.reset_index(drop=True), tz_issue

    @staticmethod
    def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        collapsed = pd.DataFrame(index=df.index)
        seen: List[Any] = []
        for name in df.columns:
            if name in seen:
                continue
            seen.append(name)
            cols = df.loc[:, df.columns == name]
            if isinstance(cols, pd.DataFrame) and cols.shape[1] > 1:
                collapsed[name] = cols.bfill(axis=1).iloc[:, 0]
            elif isinstance(cols, pd.DataFrame):
                collapsed[name] = cols.iloc[:, 0]
            else:
                collapsed[name] = cols
        return collapsed

    @staticmethod
    def _assert_ohlc_integrity(df: pd.DataFrame, result: AuditResult) -> None:
        try:
            o = df["open"].to_numpy(dtype=np.float64, copy=False)
            h = df["high"].to_numpy(dtype=np.float64, copy=False)
            low_arr = df["low"].to_numpy(dtype=np.float64, copy=False)
            c = df["close"].to_numpy(dtype=np.float64, copy=False)
            v = df["tick_volume"].to_numpy(dtype=np.float64, copy=False)

            assert np.isfinite(o).all()
            assert np.isfinite(h).all()
            assert np.isfinite(low_arr).all()
            assert np.isfinite(c).all()
            assert np.isfinite(v).all()
            assert (o > 0.0).all()
            assert (h > 0.0).all()
            assert (low_arr > 0.0).all()
            assert (c > 0.0).all()
            assert (v >= 0.0).all()
            assert (h >= low_arr).all()
            assert (h >= o).all()
            assert (h >= c).all()
            assert (low_arr <= o).all()
            assert (low_arr <= c).all()
        except AssertionError:
            result.add_issue(
                "ohlc_integrity_failed",
                "OHLC integrity assertion failed",
                MARKET_UNUSABLE,
            )

    def _detect_missing_bars(
        self,
        times: pd.Series,
        timeframe: str,
        profile: SymbolSanityProfile,
    ) -> Tuple[int, float]:
        if times is None or len(times) < 2:
            return 0, 0.0

        expected_sec = float(max(1, tf_seconds(timeframe)))
        missing_bars = 0
        largest_gap_sec = 0.0

        for idx in range(1, len(times)):
            prev_t = times.iat[idx - 1]
            curr_t = times.iat[idx]
            gap_sec = float((curr_t - prev_t).total_seconds())
            if gap_sec <= expected_sec + 1e-9:
                continue
            if self._gap_is_expected(prev_t, curr_t, expected_sec, profile):
                continue
            largest_gap_sec = max(largest_gap_sec, gap_sec)
            gap_missing = max(1, int(round(gap_sec / expected_sec)) - 1)
            missing_bars += gap_missing

        return missing_bars, largest_gap_sec

    def _gap_is_expected(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe_sec: float,
        profile: SymbolSanityProfile,
    ) -> bool:
        if profile.is_24_7:
            return False

        total_gap_sec = float((end - start).total_seconds())
        if total_gap_sec <= timeframe_sec + 1e-9:
            return False

        probe = start + timedelta(seconds=timeframe_sec)
        max_steps = int(min(20_000, math.ceil(total_gap_sec / max(timeframe_sec, 1.0))))
        for _ in range(max_steps):
            if probe >= end:
                break
            if self._is_session_open(probe.to_pydatetime(), profile):
                return False
            probe += timedelta(seconds=timeframe_sec)
        return True

    def _weekly_market_closed(
        self,
        ts_utc: datetime,
        profile: SymbolSanityProfile,
    ) -> bool:
        if profile.is_24_7:
            return False

        friday_close_min = int(
            getattr(
                self.sp,
                "weekly_close_friday_minutes",
                getattr(self.cfg, "weekly_close_friday_minutes", 21 * 60),
            )
            or (21 * 60)
        )
        sunday_open_min = int(
            getattr(
                self.sp,
                "weekly_open_sunday_minutes",
                getattr(self.cfg, "weekly_open_sunday_minutes", 22 * 60),
            )
            or (22 * 60)
        )

        weekday = int(ts_utc.weekday())
        minute_of_day = (ts_utc.hour * 60) + ts_utc.minute
        if weekday == 5:
            return True
        if weekday == 4 and minute_of_day >= friday_close_min:
            return True
        if weekday == 6 and minute_of_day < sunday_open_min:
            return True
        return False

    @staticmethod
    def _minute_in_window(minute_of_day: int, start: int, end: int) -> bool:
        start_i = int(start)
        end_i = int(end)
        if start_i == end_i:
            return False
        if start_i < end_i:
            return start_i <= minute_of_day < end_i
        return minute_of_day >= start_i or minute_of_day < end_i

    def _is_session_open(
        self,
        ts: datetime,
        profile: SymbolSanityProfile,
    ) -> bool:
        if profile.is_24_7:
            return True

        ts_utc = ts.astimezone(timezone.utc)
        if self._weekly_market_closed(ts_utc, profile):
            return False

        minute_of_day = (ts_utc.hour * 60) + ts_utc.minute
        if not self._minute_in_window(
            minute_of_day,
            profile.market_start_minutes,
            profile.market_end_minutes,
        ):
            return False

        if self._minute_in_window(
            minute_of_day,
            profile.rollover_blackout_start,
            profile.rollover_blackout_end,
        ):
            return False

        return True

    def _validate_symbol_spec(
        self,
        *,
        symbol_info: Any,
        symbol: Optional[str] = None,
    ) -> AuditResult:
        result = AuditResult(metadata={"symbol": str(symbol or self.symbol)})
        if symbol_info is None:
            return result

        profile = self.profile_for(symbol=symbol, symbol_info=symbol_info)
        observed_digits = int(getattr(symbol_info, "digits", 0) or 0)
        observed_point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        observed_contract_size = float(
            getattr(symbol_info, "trade_contract_size", 0.0)
            or getattr(symbol_info, "contract_size", 0.0)
            or 0.0
        )

        if observed_digits > 0 and observed_digits != profile.expected_digits:
            result.add_issue(
                "symbol_digits_mismatch",
                (
                    "Observed symbol digits "
                    f"{observed_digits} != expected {profile.expected_digits}"
                ),
                MARKET_UNUSABLE,
            )

        if observed_point <= 0.0:
            result.add_issue(
                "invalid_point",
                "Observed symbol point is invalid",
                MARKET_UNUSABLE,
            )
        elif not math.isclose(
            observed_point,
            profile.expected_point,
            rel_tol=0.0,
            abs_tol=max(profile.expected_point * 0.1, 1e-12),
        ):
            result.add_issue(
                "symbol_point_mismatch",
                (
                    "Observed symbol point "
                    f"{observed_point} != expected {profile.expected_point}"
                ),
                MARKET_UNUSABLE,
            )

        if observed_contract_size > 0.0 and not math.isclose(
            observed_contract_size,
            profile.expected_contract_size,
            rel_tol=1e-9,
            abs_tol=max(profile.expected_contract_size * 1e-6, 1e-12),
        ):
            result.add_issue(
                "contract_size_mismatch",
                (
                    "Observed contract size "
                    f"{observed_contract_size} != expected "
                    f"{profile.expected_contract_size}"
                ),
                MARKET_UNUSABLE,
            )

        return result

    def _detect_spread_anomaly(
        self,
        *,
        bid: Optional[float],
        ask: Optional[float],
        symbol: Optional[str] = None,
        now: Optional[Any] = None,
    ) -> AuditResult:
        result = AuditResult(metadata={"symbol": str(symbol or self.symbol)})
        if bid is None or ask is None:
            return result

        bid_f = float(bid)
        ask_f = float(ask)
        if bid_f <= 0.0 or ask_f <= 0.0 or ask_f < bid_f:
            result.add_issue(
                "invalid_quote",
                "Bid/ask snapshot is invalid",
                MARKET_UNUSABLE,
            )
            return result

        profile = self.profile_for(symbol=symbol)
        mid = 0.5 * (bid_f + ask_f)
        spread = ask_f - bid_f
        spread_pct = float(spread / mid) if mid > 0.0 else math.inf
        spread_points = (
            float(spread / profile.expected_point)
            if profile.expected_point > 0.0
            else math.inf
        )

        history = self._spread_histories[profile.symbol]
        baseline = np.asarray(history, dtype=np.float64)
        z_score = 0.0
        if baseline.size >= 20:
            mean = float(np.mean(baseline))
            std = float(np.std(baseline, ddof=0))
            if std > 1e-12:
                z_score = (spread_pct - mean) / std

        result.spread_pct = spread_pct
        result.spread_points = spread_points
        hard_pct_cap = max(profile.spread_limit_pct * 5.0, 0.0)
        hard_points_cap = max(
            float(profile.max_spread_points),
            float(getattr(self.cfg, "exec_max_spread_points", 0.0) or 0.0),
        )

        if (
            (hard_pct_cap > 0.0 and spread_pct > hard_pct_cap)
            or (hard_points_cap > 0.0 and spread_points > hard_points_cap)
            or (baseline.size >= 20 and z_score >= 6.0)
        ):
            result.add_issue(
                "spread_anomaly",
                (
                    "Abnormal spread widening detected "
                    f"(pct={spread_pct:.6f}, points={spread_points:.2f})"
                ),
                ABNORMAL_SPREAD,
                spread_pct=spread_pct,
                spread_points=spread_points,
                spread_z=z_score,
                audited_at=self._coerce_now(now).isoformat(),
            )

        history.append(spread_pct)
        return result

    def _detect_broker_feed_inconsistency(
        self,
        *,
        df: pd.DataFrame,
        timeframe: str,
        symbol: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        peer_bid: Optional[float] = None,
        peer_ask: Optional[float] = None,
    ) -> AuditResult:
        del timeframe
        result = AuditResult(metadata={"symbol": str(symbol or self.symbol)})
        if df is None or df.empty or bid is None or ask is None:
            return result

        bid_f = float(bid)
        ask_f = float(ask)
        if bid_f <= 0.0 or ask_f <= 0.0 or ask_f < bid_f:
            return result

        profile = self.profile_for(symbol=symbol)
        last_row = df.iloc[-1]
        last_low = float(last_row["low"])
        last_high = float(last_row["high"])
        last_close = float(last_row["close"])
        mid = 0.5 * (bid_f + ask_f)
        tolerance = max(
            float(profile.expected_point) * 10.0,
            (last_high - last_low) * 0.50,
            abs(last_close) * 0.0010,
        )

        if mid < (last_low - tolerance) or mid > (last_high + tolerance):
            result.add_issue(
                "broker_feed_inconsistency",
                (
                    "Live quote is inconsistent with the most recent candle "
                    f"(mid={mid:.8f}, range=[{last_low:.8f}, {last_high:.8f}])"
                ),
                MARKET_UNUSABLE,
            )

        if not math.isfinite(last_close) or last_close <= 0.0:
            result.add_issue(
                "bad_last_close",
                "Latest close is invalid",
                MARKET_UNUSABLE,
            )

        if peer_bid is not None and peer_ask is not None:
            peer_mid = 0.5 * (float(peer_bid) + float(peer_ask))
            ref_tol = max(float(profile.expected_point) * 10.0, abs(mid) * 0.0025)
            if abs(mid - peer_mid) > ref_tol:
                result.add_issue(
                    "broker_feed_divergence",
                    "Primary and peer broker quotes diverge beyond tolerance",
                    MARKET_UNUSABLE,
                )

        return result

    @staticmethod
    def _timezone_issue(series: pd.Series) -> bool:
        dtype = getattr(series, "dtype", None)
        if isinstance(dtype, pd.DatetimeTZDtype):
            return str(dtype.tz) != "UTC"
        if pd.api.types.is_datetime64_ns_dtype(dtype):
            return True
        return False

    @staticmethod
    def _coerce_time_series(series: pd.Series) -> Optional[pd.Series]:
        dtype = getattr(series, "dtype", None)
        if isinstance(dtype, pd.DatetimeTZDtype):
            return pd.to_datetime(series, utc=True, errors="coerce")
        if pd.api.types.is_datetime64_ns_dtype(dtype):
            return pd.to_datetime(series, utc=True, errors="coerce")

        numeric = pd.to_numeric(series, errors="coerce")
        if bool(numeric.notna().all()):
            unit = "ms" if float(numeric.abs().max()) > 10**12 else "s"
            return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

        return pd.to_datetime(series, utc=True, errors="coerce")

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return value.tz_localize("UTC").to_pydatetime()
            return value.tz_convert("UTC").to_pydatetime()
        if isinstance(value, (np.integer, int, np.floating, float)):
            epoch = float(value)
            if not math.isfinite(epoch):
                return None
            if epoch > 10**12:
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        return None

    def _coerce_now(self, now: Optional[Any]) -> datetime:
        if now is None:
            # Institutional: prefer server-synced clock over local
            if self._clock_sync is not None:
                return self._clock_sync.now()
            return datetime.now(timezone.utc)
        coerced = Validator._coerce_timestamp(now)
        if coerced is None:
            if self._clock_sync is not None:
                return self._clock_sync.now()
            return datetime.now(timezone.utc)
        return coerced


MarketDataValidator = Validator


__all__ = [
    "ABNORMAL_SPREAD",
    "AuditIssue",
    "AuditResult",
    "DATA_INCOMPLETE",
    "DATA_STALE",
    "DATA_VALID",
    "MARKET_UNUSABLE",
    "MarketDataValidator",
    "ServerClockSync",
    "SymbolSanityProfile",
    "tf_seconds",
    "Validator",
]

__all__ += list(_feature_engine.__all__)
__all__ = tuple(__all__)
