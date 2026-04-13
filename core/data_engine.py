# core/data_engine.py - data integrity plus feature engineering.

from __future__ import annotations



# ---- merged from core/data_integrity.py ----

import logging
import math
import time as _time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_log_sync = logging.getLogger("core.clock_sync")

DATA_VALID = "data valid"
DATA_STALE = "data stale"
DATA_INCOMPLETE = "data incomplete"
ABNORMAL_SPREAD = "abnormal spread"
MARKET_UNUSABLE = "market unusable"


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
                    self._probe_symbol, self._probe_failures, self._drift,
                )
            return

        self._probe_failures = 0
        local_at_probe = _time.time()
        raw_drift = server_epoch - local_at_probe

        # Sanity: reject insane drifts (> max_drift_sec)
        if abs(raw_drift) > self._max_drift_sec:
            _log_sync.warning(
                "CLOCK_SYNC_DRIFT_EXTREME | raw=%.3fs max=%.1fs — clamping",
                raw_drift, self._max_drift_sec,
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
            self._drift, raw_drift, self._probe_symbol,
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
        self.tradable = _STATE_PRIORITY.get(self.state, 0) < _STATE_PRIORITY[ABNORMAL_SPREAD]
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

    def __init__(self, cfg: Any, sp: Any, *, clock_sync: Optional[ServerClockSync] = None) -> None:
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
            expected_contract_size=float(
                getattr(self.sp, "contract_size", 1.0) or 1.0
            ),
            spread_limit_pct=float(
                getattr(self.sp, "spread_limit_pct", 0.0) or 0.0
            ),
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
            market_start_minutes=int(
                getattr(self.sp, "market_start_minutes", 0) or 0
            ),
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
        l = df["low"].to_numpy(dtype=np.float64, copy=False)
        c = df["close"].to_numpy(dtype=np.float64, copy=False)

        prev_close = c[:-1]
        curr_open = o[1:]
        gap_abs = np.abs(curr_open - prev_close)

        prev2 = c[:-2]
        high_1 = h[1:-1]
        low_1 = l[1:-1]
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
            diff_sec = (
                df["time"].iat[idx] - df["time"].iat[idx - 1]
            ).total_seconds()
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
        l = df["low"].to_numpy(dtype=np.float64, copy=False)
        c = df["close"].to_numpy(dtype=np.float64, copy=False)

        prev_close = c[:-1]
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(
                np.abs(h[1:] - prev_close),
                np.abs(l[1:] - prev_close),
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
            diff_sec = (
                df["time"].iat[idx] - df["time"].iat[idx - 1]
            ).total_seconds()
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
            l = df["low"].to_numpy(dtype=np.float64, copy=False)
            c = df["close"].to_numpy(dtype=np.float64, copy=False)
            v = df["tick_volume"].to_numpy(dtype=np.float64, copy=False)

            assert np.isfinite(o).all()
            assert np.isfinite(h).all()
            assert np.isfinite(l).all()
            assert np.isfinite(c).all()
            assert np.isfinite(v).all()
            assert (o > 0.0).all()
            assert (h > 0.0).all()
            assert (l > 0.0).all()
            assert (c > 0.0).all()
            assert (v >= 0.0).all()
            assert (h >= l).all()
            assert (h >= o).all()
            assert (h >= c).all()
            assert (l <= o).all()
            assert (l <= c).all()
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
        max_steps = int(
            min(20_000, math.ceil(total_gap_sec / max(timeframe_sec, 1.0)))
        )
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
        hard_points_cap = float(profile.max_spread_points)

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

__all__ = (
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
    "Validator",
)

# ---- merged from core/feature_engine.py ----

# core/feature_engine.py — Unified FeatureEngine for all assets.
# Merges _btc_indicators/feature_engine.py (1106 lines)
#    and _xau_indicators/feature_engine.py (927 lines)
# into a single config-driven class.

import math
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib

from .core_config import BaseEngineConfig
from .utils import (
    kaufman_efficiency_ratio,
    relative_volatility_index,
    tf_seconds,
)

log = logging.getLogger("core.data_engine")

FEATURES_VALID = "features valid"
NAN_DETECTED = "nan detected"
INSUFFICIENT_WARMUP = "insufficient warmup"
LOOKAHEAD_BIAS_DETECTED = "lookahead bias detected"
MTF_ALIGNMENT_ERROR = "mtf alignment error"
SCALING_ERROR = "scaling error"

_FEATURE_STATE_PRIORITY = {
    FEATURES_VALID: 0,
    INSUFFICIENT_WARMUP: 1,
    NAN_DETECTED: 2,
    SCALING_ERROR: 3,
    LOOKAHEAD_BIAS_DETECTED: 4,
    MTF_ALIGNMENT_ERROR: 5,
}


def safe_last(arr: np.ndarray, default: float = 0.0) -> float:
    """Extract last finite value from an array."""
    if arr is None or len(arr) == 0:
        return default
    v = float(arr[-1])
    return v if math.isfinite(v) else default


def _shifted_last(arr: np.ndarray, shift: int, default: float = 0.0) -> float:
    """
    Extract shift-aware last value.
    shift=0 → arr[-1]  (current bar)
    shift=1 → arr[-2]  (previous bar, prevents lookahead)
    """
    if arr is None or len(arr) == 0:
        return default
    pos = 1 + shift  # 1-based distance from end
    if pos > len(arr):
        return default
    v = float(arr[-pos])
    return v if math.isfinite(v) else default


@dataclass
class AnomalyResult:
    score: float = 0.0
    reasons: List[str] = None
    blocked: bool = False
    level: str = "normal"

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass(frozen=True)
class FeatureAuditIssue:
    code: str
    message: str
    state: str


@dataclass
class FeatureAuditResult:
    state: str = FEATURES_VALID
    valid: bool = True
    issues: List[FeatureAuditIssue] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}

    def add_issue(
        self,
        code: str,
        message: str,
        state: str,
        **details: Any,
    ) -> None:
        issue = FeatureAuditIssue(code=code, message=message, state=state)
        if issue not in self.issues:
            self.issues.append(issue)
        if _FEATURE_STATE_PRIORITY[state] > _FEATURE_STATE_PRIORITY[self.state]:
            self.state = state
        self.valid = self.state == FEATURES_VALID
        if details:
            self.metadata[code] = details

    def merge(self, other: "FeatureAuditResult") -> "FeatureAuditResult":
        if other is None:
            return self
        self.metadata.update(other.metadata)
        for issue in other.issues:
            self.add_issue(issue.code, issue.message, issue.state)
        return self

    def reason_codes(self) -> List[str]:
        reasons = [f"feature_state:{self.state}"]
        for issue in self.issues:
            reasons.append(issue.code)
        return reasons


class FeatureIntegrityError(RuntimeError):
    def __init__(self, audit: FeatureAuditResult) -> None:
        self.audit = audit
        super().__init__(audit.state)


class FeatureEngine:
    """
    Unified feature engine for indicator computation, pattern detection,
    and market microstructure analysis.

    Config-driven: all thresholds, lookback periods, and round-number
    levels are parameterized via BaseEngineConfig.

    Computes:
      - EMAs (8, 21, 50, 200), RSI, MACD, ADX, Bollinger Bands, ATR
      - CCI, MOM, ROC, STOCH (K/D), OBV
      - VWAP, Volume Z-Score
      - Smart-money: FVG, Liquidity Sweep, Order Blocks, Round Numbers
      - Trend/Regime analysis: linreg slope, volatility regime
      - Market anomalies: flash crash, volume spike, spread anomalies
      - Divergence detection: RSI divergence (bullish/bearish)
      - Forecast continuation: breakout + volume confirmation

    Usage:
        fe = FeatureEngine(cfg)
        indicators = fe.compute_indicators({"M1": df_m1, "M5": df_m5, "M15": df_m15})
    """

    def __init__(self, cfg: BaseEngineConfig) -> None:
        self.cfg = cfg
        ic = cfg.indicator

        # Indicator periods from config
        self._ema_short = ic.get("ema_short", 8)
        self._ema_medium = ic.get("ema_medium", 21)
        self._ema_slow = ic.get("ema_slow", 50)
        self._ema_anchor = ic.get("ema_anchor", 200)
        self._rsi_period = ic.get("rsi_period", 14)
        self._atr_period = ic.get("atr_period", 14)
        self._macd_fast = ic.get("macd_fast", 12)
        self._macd_slow = ic.get("macd_slow", 26)
        self._macd_signal = ic.get("macd_signal", 9)
        self._adx_period = ic.get("adx_period", 14)
        self._bb_period = ic.get("bb_period", 20)
        self._bb_std = cfg.bb_std
        self._cci_period = ic.get("cci_period", 20)
        self._mom_period = ic.get("mom_period", 10)
        self._roc_period = ic.get("roc_period", 10)
        self._stoch_k_period = ic.get("stoch_k_period", 14)
        self._stoch_slowk_period = ic.get("stoch_slowk_period", 3)
        self._stoch_slowd_period = ic.get("stoch_slowd_period", 3)
        self._stoch_matype = ic.get("stoch_matype", 0)
        self._ker_period = int(getattr(cfg, "ker_period", 10) or 10)
        self._rvi_period = int(getattr(cfg, "rvi_period", 14) or 14)

        # Round number levels — asset-specific
        symbol = cfg.symbol_params.base.upper()
        if "BTC" in symbol:
            self._round_levels = [500, 1000, 2500, 5000, 10000]
            self._round_atr_mult = 1.5
        else:
            self._round_levels = [5, 10, 25, 50, 100]
            self._round_atr_mult = 1.0

        # Anomaly thresholds
        self._anomaly_z_vol_thresh = 4.0
        self._anomaly_atr_spike_mult = 3.0
        self._anomaly_spread_z_thresh = 5.0

        self._last_feature_audit: FeatureAuditResult = FeatureAuditResult()

        self._validate_cfg()
        log.info("FeatureEngine initialized for %s", symbol)

    def _validate_cfg(self) -> None:
        for attr in ["_ema_short", "_ema_medium", "_rsi_period", "_atr_period", "_cci_period", "_mom_period", "_roc_period"]:
            v = getattr(self, attr)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"Invalid indicator config: {attr}={v}")

    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════

    def compute_indicators(
        self, df_dict: Dict[str, pd.DataFrame], shift: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute indicators for all timeframes.

        Args:
            df_dict: {timeframe_str: DataFrame} with OHLCV data
            shift: 1 = use prev bar (prevent look-ahead), 0 = use current

        Returns:
            {timeframe_str: {indicator_name: value}} for each TF
        """
        shift_i = int(shift)
        audit = FeatureAuditResult(metadata={"shift": shift_i})
        self._last_feature_audit = audit
        if shift_i < 1:
            audit.add_issue(
                "shift_not_closed_bar",
                "Live feature computation must use closed bars only",
                LOOKAHEAD_BIAS_DETECTED,
                shift=shift_i,
            )
            raise FeatureIntegrityError(audit)

        result: Dict[str, Dict[str, Any]] = {}
        frame_meta: Dict[str, Dict[str, Any]] = {}

        for tf, df in df_dict.items():
            input_audit = self._audit_input_frame(tf=tf, df=df, shift=shift_i)
            audit.merge(input_audit)
            if not input_audit.valid:
                result[tf] = {}
                frame_meta[tf] = self._frame_meta(tf=tf, df=df, shift=shift_i)
                continue
            try:
                indicators = self._compute_tf(tf=tf, df=df, shift=shift_i)
            except Exception as exc:
                log.error(
                    "compute_indicators(tf=%s) failed: %s\n%s",
                    tf, exc, traceback.format_exc(),
                )
                raise RuntimeError(
                    f"feature computation failed for tf={tf}: {exc}"
                ) from exc
            meta = self._frame_meta(tf=tf, df=df, shift=shift_i)
            frame_meta[tf] = meta
            indicators["feature_lag_bars"] = shift_i
            indicators["feature_valid"] = True
            if meta.get("used_time") is not None:
                indicators["feature_source_time"] = meta["used_time"].isoformat()
            result[tf] = indicators
            audit.merge(
                self._audit_feature_output(
                    tf=tf,
                    df=df,
                    out=indicators,
                    shift=shift_i,
                    meta=meta,
                )
            )

        # MTF alignment
        if len(result) >= 2:
            self._check_mtf_alignment(result, frame_meta, audit)

        self._last_feature_audit = audit
        if not audit.valid:
            raise FeatureIntegrityError(audit)

        return result

    def last_feature_audit(self) -> FeatureAuditResult:
        return self._last_feature_audit

    def classify_market_regime(
        self,
        indicators: Dict[str, Dict[str, Any]],
        *,
        asset: str = "",
        now: Optional[Any] = None,
        primary_tf: Optional[str] = None,
    ) -> str:
        """Classify the live market state from closed-bar features only."""
        if not indicators:
            return "range"

        tf = str(primary_tf or getattr(self.cfg.symbol_params, "tf_primary", "M1") or "M1")
        primary = indicators.get(tf) or indicators.get("M1") or next(iter(indicators.values()), {})
        if not isinstance(primary, dict) or not primary:
            return "range"

        asset_u = str(asset or getattr(self.cfg.symbol_params, "base", "") or "").upper()
        ts = self._coerce_timestamp(now)
        if ts is None:
            ts = pd.Timestamp.utcnow()
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        trend = str(primary.get("trend", "flat") or "flat").lower()
        forecast = str(primary.get("forecast", "none") or "none").lower()
        volatility = str(primary.get("volatility_regime", primary.get("regime", "normal")) or "normal").lower()
        adx = float(primary.get("adx", 0.0) or 0.0)
        atr_ratio = float(primary.get("atr_ratio", 1.0) or 1.0)
        linreg_slope = float(primary.get("linreg_slope", 0.0) or 0.0)
        confluence = float(primary.get("confluence", 0.0) or 0.0)
        stop_hunt_strength = abs(float(primary.get("stop_hunt_strength", 0.0) or 0.0))
        divergence = str(primary.get("divergence", "") or "").lower()
        anomaly = primary.get("anomaly")
        anomaly_level = str(getattr(anomaly, "level", "normal") or "normal").lower()

        tags: List[str] = []
        minute_utc = int(ts.hour * 60 + ts.minute)
        weekday = int(ts.weekday())

        if "BTC" in asset_u and weekday >= 5:
            tags.append("weekend BTC regime")

        if minute_utc in range(420, 481) or minute_utc in range(750, 811):
            tags.append("session open")
        elif minute_utc < 360 or minute_utc >= 1320:
            tags.append("session dead hours")

        if minute_utc in range(745, 851) or minute_utc in range(360, 391):
            tags.append("news hours")

        breakout = forecast in {"bull_continuation", "bear_continuation"}
        fake_breakout = bool(
            breakout
            and (
                stop_hunt_strength >= 0.60
                or divergence in {"bullish", "bearish"}
                or anomaly_level in {"warning", "critical"}
            )
        )

        if fake_breakout:
            tags.append("fake breakout")
        elif breakout:
            tags.append("breakout")

        if volatility == "explosive" or atr_ratio >= 1.50:
            tags.append("high volatility")
        elif volatility == "dead" or atr_ratio <= 0.70:
            tags.append("low volatility")

        strong_trend = bool(
            trend in {"bull", "bear"}
            and (
                adx >= 25.0
                or abs(linreg_slope) >= max(
                    1e-4,
                    abs(float(primary.get("close", 0.0) or 0.0)) * 1e-6,
                )
                or confluence >= 0.70
            )
        )

        if strong_trend:
            tags.append("trend strong")
        elif trend in {"bull", "bear"}:
            tags.append("trend weak")
        elif not breakout and volatility != "explosive":
            tags.append("range")

        if not tags:
            tags.append("range")

        unique_tags: List[str] = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        return "|".join(unique_tags[:4])

    def _required_warmup_bars(self, timeframe: str, shift: int) -> int:
        del timeframe
        return int(
            max(
                self._ema_anchor + shift + 5,
                self._macd_slow + self._macd_signal + shift + 5,
                self._bb_period + shift + 5,
                self._atr_period + shift + 5,
                self._stoch_k_period + self._stoch_slowd_period + shift + 5,
                50,
            )
        )

    def _audit_input_frame(
        self,
        *,
        tf: str,
        df: Optional[pd.DataFrame],
        shift: int,
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        if df is None or len(df) == 0:
            audit.add_issue(
                f"{tf.lower()}_frame_missing",
                f"{tf} frame is missing",
                INSUFFICIENT_WARMUP,
            )
            return audit

        warmup_bars = self._required_warmup_bars(tf, shift)
        if len(df) < warmup_bars:
            audit.add_issue(
                f"{tf.lower()}_insufficient_warmup",
                f"{tf} requires {warmup_bars} bars, got {len(df)}",
                INSUFFICIENT_WARMUP,
                required=warmup_bars,
                actual=len(df),
            )

        cols = {c.lower(): c for c in df.columns}
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(cols)):
            audit.add_issue(
                f"{tf.lower()}_schema_error",
                f"{tf} frame is missing required OHLC columns",
                SCALING_ERROR,
            )
        return audit

    def _frame_meta(
        self,
        *,
        tf: str,
        df: Optional[pd.DataFrame],
        shift: int,
    ) -> Dict[str, Any]:
        tf_sec = int(tf_seconds(tf))
        meta: Dict[str, Any] = {
            "tf": tf,
            "tf_seconds": tf_sec,
            "shift": int(shift),
            "used_time": None,
            "latest_time": None,
            "used_close_time": None,
        }
        if df is None or len(df) <= shift:
            return meta

        time_col = None
        cols = {c.lower(): c for c in df.columns}
        if "time" in cols:
            time_col = cols["time"]

        used_idx = len(df) - 1 - shift
        latest_idx = len(df) - 1
        if time_col is not None:
            used_time = self._coerce_timestamp(df.iloc[used_idx][time_col])
            latest_time = self._coerce_timestamp(df.iloc[latest_idx][time_col])
        else:
            used_time = self._coerce_timestamp(df.index[used_idx])
            latest_time = self._coerce_timestamp(df.index[latest_idx])

        meta["used_time"] = used_time
        meta["latest_time"] = latest_time
        if used_time is not None:
            meta["used_close_time"] = used_time + pd.Timedelta(seconds=tf_sec)
        return meta

    def _audit_feature_output(
        self,
        *,
        tf: str,
        df: pd.DataFrame,
        out: Dict[str, Any],
        shift: int,
        meta: Dict[str, Any],
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        if not isinstance(out, dict) or not out:
            audit.add_issue(
                f"{tf.lower()}_empty_features",
                f"{tf} features are empty after computation",
                INSUFFICIENT_WARMUP,
            )
            return audit

        for key, value in out.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not math.isfinite(float(value)):
                    audit.add_issue(
                        f"{tf.lower()}_{key}_nan",
                        f"{tf}:{key} is non-finite",
                        NAN_DETECTED,
                    )
            elif isinstance(value, AnomalyResult):
                if not math.isfinite(float(value.score)):
                    audit.add_issue(
                        f"{tf.lower()}_anomaly_score_nan",
                        f"{tf}: anomaly score is non-finite",
                        NAN_DETECTED,
                    )

        if out.get("feature_lag_bars") != shift:
            audit.add_issue(
                f"{tf.lower()}_feature_lag_mismatch",
                f"{tf} feature lag does not match requested shift",
                LOOKAHEAD_BIAS_DETECTED,
            )

        df = self._ensure_volume(df)
        cols = {c.lower(): c for c in df.columns}
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        l = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        used_len = len(c) - shift
        if used_len <= 0:
            audit.add_issue(
                f"{tf.lower()}_lookahead_underflow",
                f"{tf} used length is invalid for shift={shift}",
                LOOKAHEAD_BIAS_DETECTED,
            )
            return audit

        c_seen = c[:used_len]
        h_seen = h[:used_len]
        l_seen = l[:used_len]
        price_min = float(np.min(c_seen))
        price_max = float(np.max(c_seen))

        self._check_bounded_feature(
            audit, tf=tf, key="rsi", value=out.get("rsi"), lower=0.0, upper=100.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="adx", value=out.get("adx"), lower=0.0, upper=100.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="stoch_k", value=out.get("stoch_k"), lower=0.0, upper=100.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="stoch_d", value=out.get("stoch_d"), lower=0.0, upper=100.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="ker", value=out.get("ker"), lower=0.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="rvi", value=out.get("rvi"), lower=0.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="confluence", value=out.get("confluence"), lower=0.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="ob_touch_proximity", value=out.get("ob_touch_proximity"), lower=0.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="linreg_slope", value=out.get("linreg_slope"), lower=-1.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="ob_pretouch_bias", value=out.get("ob_pretouch_bias"), lower=-1.0, upper=1.0,
        )
        self._check_bounded_feature(
            audit, tf=tf, key="stop_hunt_strength", value=out.get("stop_hunt_strength"), lower=-1.0, upper=1.0,
        )

        for ema_key in ("ema_short", "ema_medium", "ema_slow", "ema_anchor"):
            ema_val = float(out.get(ema_key, 0.0) or 0.0)
            tol = self._feature_tol(max(abs(price_max), abs(price_min), abs(ema_val), 1.0))
            if ema_val < (price_min - tol) or ema_val > (price_max + tol):
                audit.add_issue(
                    f"{tf.lower()}_{ema_key}_range_error",
                    f"{tf}:{ema_key} fell outside the observed price envelope",
                    SCALING_ERROR,
                )

        atr_val = float(out.get("atr", 0.0) or 0.0)
        if atr_val < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_negative",
                f"{tf}: ATR must be non-negative",
                SCALING_ERROR,
            )
        prev_close = c_seen[:-1]
        if len(prev_close) > 0:
            tr_seen = np.maximum(
                h_seen[1:] - l_seen[1:],
                np.maximum(
                    np.abs(h_seen[1:] - prev_close),
                    np.abs(l_seen[1:] - prev_close),
                ),
            )
            if tr_seen.size > 0:
                atr_tol = self._feature_tol(max(float(np.max(tr_seen)), atr_val, 1.0))
                if atr_val > float(np.max(tr_seen)) + atr_tol:
                    audit.add_issue(
                        f"{tf.lower()}_atr_reference_error",
                        f"{tf}: ATR exceeded the observed true-range envelope",
                        SCALING_ERROR,
                    )

        macd_val = float(out.get("macd", 0.0) or 0.0)
        macd_signal_val = float(out.get("macd_signal", 0.0) or 0.0)
        macd_hist_val = float(out.get("macd_hist", 0.0) or 0.0)
        macd_tol = self._feature_tol(
            max(abs(macd_val), abs(macd_signal_val), abs(macd_hist_val), 1.0)
        )
        if abs((macd_val - macd_signal_val) - macd_hist_val) > macd_tol:
            audit.add_issue(
                f"{tf.lower()}_macd_identity_error",
                f"{tf}: MACD histogram violated line-signal identity",
                SCALING_ERROR,
            )

        if float(out.get("bb_width", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_bb_width_negative",
                f"{tf}: Bollinger width must be non-negative",
                SCALING_ERROR,
            )
        if float(out.get("atr_pct", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_pct_negative",
                f"{tf}: ATR percent must be non-negative",
                SCALING_ERROR,
            )
        if float(out.get("atr_ratio", 0.0) or 0.0) < 0.0:
            audit.add_issue(
                f"{tf.lower()}_atr_ratio_negative",
                f"{tf}: ATR ratio must be non-negative",
                SCALING_ERROR,
            )

        audit.merge(
            self._verify_reference_alignment(
                tf=tf,
                df=df,
                out=out,
                shift=shift,
                meta=meta,
            )
        )
        return audit

    def _verify_reference_alignment(
        self,
        *,
        tf: str,
        df: pd.DataFrame,
        out: Dict[str, Any],
        shift: int,
        meta: Dict[str, Any],
    ) -> FeatureAuditResult:
        audit = FeatureAuditResult(metadata={"tf": tf})
        del meta
        cols = {c.lower(): c for c in df.columns}
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        l = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        used_len = len(c) - shift
        if used_len <= 1:
            audit.add_issue(
                f"{tf.lower()}_reference_underflow",
                f"{tf}: reference validation length is insufficient",
                INSUFFICIENT_WARMUP,
            )
            return audit

        c_cut = c[:used_len]
        h_cut = h[:used_len]
        l_cut = l[:used_len]
        c0 = float(c_cut[0]) if len(c_cut) > 0 else 0.0

        ema_ref = safe_last(self._nan_to_num(talib.EMA(c_cut, timeperiod=self._ema_medium), c0))
        rsi_ref = safe_last(self._nan_to_num(talib.RSI(c_cut, timeperiod=self._rsi_period), 50.0))
        macd_line_ref, macd_signal_ref, _ = talib.MACD(
            c_cut,
            fastperiod=self._macd_fast,
            slowperiod=self._macd_slow,
            signalperiod=self._macd_signal,
        )
        macd_line_ref = safe_last(self._nan_to_num(macd_line_ref, 0.0))
        macd_signal_ref = safe_last(self._nan_to_num(macd_signal_ref, 0.0))
        atr_ref = safe_last(
            self._nan_to_num(
                talib.ATR(h_cut, l_cut, c_cut, timeperiod=self._atr_period),
                0.0,
            )
        )

        checks = (
            ("ema_medium", float(out.get("ema_medium", 0.0) or 0.0), float(ema_ref)),
            ("rsi", float(out.get("rsi", 0.0) or 0.0), float(rsi_ref)),
            ("macd", float(out.get("macd", 0.0) or 0.0), float(macd_line_ref)),
            ("macd_signal", float(out.get("macd_signal", 0.0) or 0.0), float(macd_signal_ref)),
            ("atr", float(out.get("atr", 0.0) or 0.0), float(atr_ref)),
        )
        for key, live_val, ref_val in checks:
            tol = self._feature_tol(max(abs(live_val), abs(ref_val), 1.0))
            if abs(live_val - ref_val) > tol:
                audit.add_issue(
                    f"{tf.lower()}_{key}_lookahead",
                    f"{tf}:{key} diverged from closed-bar reference",
                    LOOKAHEAD_BIAS_DETECTED,
                    live=live_val,
                    reference=ref_val,
                )
        return audit

    def _check_bounded_feature(
        self,
        audit: FeatureAuditResult,
        *,
        tf: str,
        key: str,
        value: Any,
        lower: float,
        upper: float,
    ) -> None:
        if value is None:
            return
        val = float(value)
        if not math.isfinite(val):
            audit.add_issue(
                f"{tf.lower()}_{key}_nan",
                f"{tf}:{key} is non-finite",
                NAN_DETECTED,
            )
            return
        tol = self._feature_tol(max(abs(lower), abs(upper), abs(val), 1.0))
        if val < (lower - tol) or val > (upper + tol):
            audit.add_issue(
                f"{tf.lower()}_{key}_scale_error",
                f"{tf}:{key} breached expected bounds [{lower}, {upper}]",
                SCALING_ERROR,
            )

    @staticmethod
    def _feature_tol(scale: float) -> float:
        return max(1e-8, float(scale) * 1e-8)

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return value.tz_localize("UTC")
            return value.tz_convert("UTC")
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")
        except Exception:
            return None

    # ═════════════════════════════════════════════════════════════════
    # TIMEFRAME COMPUTATION
    # ═════════════════════════════════════════════════════════════════

    def _compute_tf(
        self, *, tf: str, df: pd.DataFrame, shift: int,
    ) -> Dict[str, Any]:
        """Compute all indicators for a single timeframe."""
        df = self._ensure_volume(df)
        min_bars = self._min_bars(tf)
        if len(df) < min_bars:
            return {}

        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        h = self._as_1d_float(df[cols.get("high", "High")].to_numpy())
        l = self._as_1d_float(df[cols.get("low", "Low")].to_numpy())
        c = self._as_1d_float(df[cols.get("close", "Close")].to_numpy())
        o = self._as_1d_float(df[cols.get("open", "Open")].to_numpy())
        v = (
            self._as_1d_float(df[cols.get("tick_volume", cols.get("volume", "tick_volume"))].to_numpy())
            if any(k in cols for k in ("tick_volume", "volume"))
            else np.ones(len(c), dtype=np.float64)
        )

        # Defensive alignment in case duplicated columns produced odd shapes.
        n = int(min(len(h), len(l), len(c), len(o), len(v)))
        if n < min_bars:
            return {}
        if len(h) != n:
            h = h[-n:]
        if len(l) != n:
            l = l[-n:]
        if len(c) != n:
            c = c[-n:]
        if len(o) != n:
            o = o[-n:]
        if len(v) != n:
            v = v[-n:]

        # Fill NaN/inf
        c = self._ffill_finite(c)
        h = self._ffill_finite(h, default=c[0] if len(c) > 0 else 0.0)
        l = self._ffill_finite(l, default=c[0] if len(c) > 0 else 0.0)
        o = self._ffill_finite(o, default=c[0] if len(c) > 0 else 0.0)

        c0 = float(c[0]) if len(c) > 0 else 0.0

        # ── EMAs (TA-Lib) ──
        ema_s = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_short), c0)
        ema_m = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_medium), c0)
        ema_l = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_slow), c0)
        if len(c) >= self._ema_anchor:
            ema_a = self._nan_to_num(talib.EMA(c, timeperiod=self._ema_anchor), c0)
        else:
            ema_a = ema_l

        # ── RSI (TA-Lib) ──
        rsi = self._nan_to_num(talib.RSI(c, timeperiod=self._rsi_period), 50.0)

        # ── MACD (TA-Lib) ──
        macd_line, macd_signal, macd_hist = talib.MACD(
            c,
            fastperiod=self._macd_fast,
            slowperiod=self._macd_slow,
            signalperiod=self._macd_signal,
        )
        macd_line = self._nan_to_num(macd_line, 0.0)
        macd_signal = self._nan_to_num(macd_signal, 0.0)
        macd_hist = self._nan_to_num(macd_hist, 0.0)

        # ── ATR (TA-Lib) ──
        atr = self._nan_to_num(talib.ATR(h, l, c, timeperiod=self._atr_period), 0.0)
        atr_last = _shifted_last(atr, shift)
        _cl_pos = 1 + shift  # 1-based offset from end
        close_last = float(c[-_cl_pos]) if _cl_pos <= len(c) else float(c[-1])
        atr_pct = atr_last / close_last if close_last > 0 else 0.0

        # ── Fractal efficiency & volatility (shift-aware) ──
        _c_for_ker = c[:len(c)-shift] if shift > 0 and shift < len(c) else c
        ker = kaufman_efficiency_ratio(_c_for_ker, period=self._ker_period)
        rvi = relative_volatility_index(_c_for_ker, period=self._rvi_period)

        # ── ADX (TA-Lib) ──
        adx = self._nan_to_num(talib.ADX(h, l, c, timeperiod=self._adx_period), 0.0)

        # ── Bollinger Bands (TA-Lib) ──
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            c,
            timeperiod=self._bb_period,
            nbdevup=self._bb_std,
            nbdevdn=self._bb_std,
            matype=0,
        )
        bb_upper = self._nan_to_num(bb_upper, c0)
        bb_middle = self._nan_to_num(bb_middle, c0)
        bb_lower = self._nan_to_num(bb_lower, c0)
        _bb_u = _shifted_last(bb_upper, shift)
        _bb_l = _shifted_last(bb_lower, shift)
        _bb_m = _shifted_last(bb_middle, shift)
        bb_width = (_bb_u - _bb_l) / _bb_m if _bb_m > 0 else 0.0
        bb_pctb = (close_last - _bb_l) / (_bb_u - _bb_l) \
            if (_bb_u - _bb_l) > 0 else 0.5

        # ── Momentum & Volatility (TA-Lib) ──
        cci = self._nan_to_num(talib.CCI(h, l, c, timeperiod=self._cci_period), 0.0)
        mom = self._nan_to_num(talib.MOM(c, timeperiod=self._mom_period), 0.0)
        roc = self._nan_to_num(talib.ROC(c, timeperiod=self._roc_period), 0.0)
        stoch_k, stoch_d = talib.STOCH(
            h, l, c,
            fastk_period=self._stoch_k_period,
            slowk_period=self._stoch_slowk_period,
            slowk_matype=self._stoch_matype,
            slowd_period=self._stoch_slowd_period,
            slowd_matype=self._stoch_matype,
        )
        stoch_k = self._nan_to_num(stoch_k, 50.0)
        stoch_d = self._nan_to_num(stoch_d, 50.0)
        obv = self._nan_to_num(talib.OBV(c, v), 0.0)

        # ── Shift-aware arrays and DataFrame for pattern/helper functions ──
        # When shift>0, trim arrays/df so [-1] refers to the shifted bar,
        # preventing lookahead bias in backtest mode.  (forensic fix 2026-04-02)
        if shift > 0 and shift < len(c):
            _c_s = c[:-shift]
            _h_s = h[:-shift]
            _l_s = l[:-shift]
            _o_s = o[:-shift]
            _v_s = v[:-shift]
            _atr_s = atr[:-shift]
            _rsi_s = rsi[:-shift]
            _df_s = df.iloc[:-shift]
        else:
            _c_s, _h_s, _l_s, _o_s, _v_s = c, h, l, o, v
            _atr_s, _rsi_s = atr, rsi
            _df_s = df

        # Recompute atr_last and z_vol on shifted arrays for pattern functions
        _atr_last_s = safe_last(_atr_s)

        # ── VWAP (shift-aware) ──
        vwap = self._vwap(_df_s, window=20)

        # ── Volume Z-Score (shift-aware) ──
        z_vol = self._z_score(_v_s, period=50)
        z_vol_series = self._z_score_series(v, period=50)
        if shift > 0 and shift < len(z_vol_series):
            _z_vol_s = z_vol_series[:-shift]
        else:
            _z_vol_s = z_vol_series
        _z_vol_s_scalar = z_vol

        # ── Trend (shift-aware) ──
        trend = self._determine_trend(
            c, ema_m, ema_l, ema_a, adx, shift=shift,
        )

        # ── Linreg slope (shift-aware) ──
        linreg_slope = self.linreg_trend_slope(_c_s, period=20)

        # ── Volatility regime (shift-aware) ──
        vol_regime, atr_ratio = self.volatility_regime(_atr_s, period=50)

        # ── Divergence (shift-aware) ──
        div = self._detect_divergence_swings(ind=_rsi_s, price=_c_s)

        # ── Forecast (shift-aware) ──
        forecast = self.forecast_continuation(_df_s, _v_s)

        # ── Smart money patterns (shift-aware) ──
        fvg = self._detect_fvg(_df_s, _atr_s)
        sweep = self._liquidity_sweep(
            _df_s, atr_last=_atr_last_s, z_volume=_z_vol_s_scalar, adx=adx,
        )
        ob = self._order_block(
            _df_s, atr_last=_atr_last_s, v=_v_s, z_vol_series=_z_vol_s,
        )
        stop_hunt_side, stop_hunt_strength = self._stop_hunt_profile(
            df=_df_s,
            atr_last=_atr_last_s,
            z_volume=_z_vol_s_scalar,
            sweep=sweep,
        )
        ob_touch_prox, ob_pretouch_bias = self._order_block_pretouch(
            df=_df_s, atr_last=_atr_last_s,
        )
        near_rn = self._near_round_number(price=close_last, atr=atr_last)

        # ── Anomalies (shift-aware) ──
        anomaly = self._detect_market_anomalies(
            dfp=_df_s, atr_series=_atr_s, z_vol_series=_z_vol_s,
            adx=_shifted_last(adx, shift),
        )

        # ── Confluence ──
        confl = self._calculate_confluence_increment(
            has_fvg=bool(fvg),
            has_sweep=bool(sweep),
            has_ob=bool(ob),
            has_near_rn=near_rn,
            has_div=bool(div),
            z_volume=z_vol,
            trend=trend,
            bb_width=bb_width,
            vwap_dist_atr=abs(close_last - vwap) / atr_last if atr_last > 0 else 0.0,
        )

        # Build output (shift-aware): shift=0 → idx=-1 (current), shift=1 → idx=-2 (prev)
        idx = -(1 + shift) if (1 + shift) <= len(c) else -1
        out = {
            # EMAs
            "ema_short": float(ema_s[idx]),
            "ema_medium": float(ema_m[idx]),
            "ema_slow": float(ema_l[idx]),
            "ema_anchor": float(ema_a[idx]) if len(ema_a) > abs(idx) else 0.0,
            # RSI
            "rsi": float(rsi[idx]),
            # MACD
            "macd": float(macd_line[idx]),
            "macd_signal": float(macd_signal[idx]),
            "macd_hist": float(macd_hist[idx]),
            # ATR
            "atr": atr_last,
            "atr_pct": atr_pct,
            # Fractal / volatility
            "ker": float(ker),
            "rvi": float(rvi),
            # ADX
            "adx": _shifted_last(adx, shift),
            # Bollinger
            "bb_upper": _bb_u,
            "bb_lower": _bb_l,
            "bb_middle": _bb_m,
            "bb_width": bb_width,
            "bb_pctb": bb_pctb,
            # Momentum / Volatility
            "cci": float(cci[idx]),
            "mom": float(mom[idx]),
            "roc": float(roc[idx]),
            "stoch_k": float(stoch_k[idx]),
            "stoch_d": float(stoch_d[idx]),
            "obv": float(obv[idx]),
            # VWAP
            "vwap": vwap,
            # Volume
            "z_volume": z_vol,
            # Trend / regime
            "trend": trend,
            "linreg_slope": linreg_slope,
            "regime": vol_regime,
            "atr_ratio": atr_ratio,
            "volatility_regime": vol_regime,
            # Patterns
            "fvg": fvg,
            "sweep": sweep,
            "order_block": ob,
            "stop_hunt_side": stop_hunt_side,
            "stop_hunt_strength": float(stop_hunt_strength),
            "ob_touch_proximity": float(ob_touch_prox),
            "ob_pretouch_bias": float(ob_pretouch_bias),
            "near_round": near_rn,
            "divergence": div,
            "forecast": forecast,
            # Anomaly
            "anomaly": anomaly,
            # Confluence
            "confluence": confl,
            # Close
            "close": close_last,
        }

        return out

    # ═════════════════════════════════════════════════════════════════
    # INDICATOR CALCULATIONS (TA-Lib)
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _ffill_finite(a: np.ndarray, default: float = 0.0) -> np.ndarray:
        """Vectorized forward-fill for non-finite values."""
        out = np.asarray(a, dtype=np.float64).copy()
        if out.size == 0:
            return out
        finite = np.isfinite(out)
        if finite.all():
            return out
        if not finite.any():
            out.fill(float(default))
            return out

        idx = np.where(finite, np.arange(out.size, dtype=np.int64), -1)
        np.maximum.accumulate(idx, out=idx)
        leading = idx < 0
        if np.any(leading):
            out[leading] = float(default)
        missing = ~finite & ~leading
        if np.any(missing):
            out[missing] = out[idx[missing]]
        return out

    @staticmethod
    def _as_1d_float(values: Any) -> np.ndarray:
        """
        Normalize unknown input shape to 1D float array.
        Handles duplicated DataFrame columns where pandas may return 2D values.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=np.float64)
        if arr.ndim == 1:
            return arr
        if arr.shape[0] == 0:
            return np.array([], dtype=np.float64)
        arr2 = arr.reshape(arr.shape[0], -1)
        return np.asarray(arr2[:, -1], dtype=np.float64).reshape(-1)

    @staticmethod
    def _nan_to_num(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)

    def _vwap(self, df: pd.DataFrame, window: int = 20) -> float:
        """Session VWAP approximation."""
        try:
            cols = {c.lower(): c for c in df.columns}
            c = df[cols.get("close", "Close")].values
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values

            vol_col = None
            for k in ("tick_volume", "volume"):
                if k in cols:
                    vol_col = cols[k]
                    break
            if vol_col is None:
                return float(c[-1]) if len(c) > 0 else 0.0
            v = df[vol_col].values.astype(np.float64)

            w = min(window, len(c))
            typical = (c[-w:] + h[-w:] + l[-w:]) / 3.0
            vol = v[-w:]
            vol_sum = np.sum(vol)
            if vol_sum <= 0:
                return float(np.mean(typical))
            return float(np.sum(typical * vol) / vol_sum)
        except Exception:
            return 0.0

    @staticmethod
    def _z_score(values: np.ndarray, period: int = 50) -> float:
        """Z-score of last value relative to recent window."""
        vals = FeatureEngine._as_1d_float(values)
        if len(vals) < 5:
            return 0.0
        p = max(2, int(period))
        w = vals[-p:]
        m = float(np.mean(w))
        s = float(np.std(w))
        if not np.isfinite(s) or s <= 1e-12:
            return 0.0
        last = float(vals[-1])
        if not np.isfinite(last) or not np.isfinite(m):
            return 0.0
        return float((last - m) / s)

    @staticmethod
    def _z_score_series(values: np.ndarray, period: int = 50) -> np.ndarray:
        """Rolling Z-score series."""
        vals = FeatureEngine._as_1d_float(values)
        out = np.zeros_like(vals, dtype=np.float64)
        p = max(2, int(period))
        for i in range(p, len(vals)):
            w = vals[i - p:i]
            m = float(np.mean(w))
            s = float(np.std(w))
            if np.isfinite(s) and s > 1e-12 and np.isfinite(m):
                out[i] = (vals[i] - m) / s
        return out

    # ═════════════════════════════════════════════════════════════════
    # PATTERN DETECTION
    # ═════════════════════════════════════════════════════════════════

    def _determine_trend(
        self, c: np.ndarray, ema21: np.ndarray, ema50: np.ndarray,
        ema200: np.ndarray, adx: np.ndarray, *, shift: int = 0,
    ) -> str:
        """Determine trend direction from EMA stack (shift-aware)."""
        s = _shifted_last(ema21, shift)
        m = _shifted_last(ema50, shift)
        l_val = _shifted_last(ema200, shift)
        if s > m > l_val > 0:
            return "bull"
        if s < m < l_val and l_val > 0:
            return "bear"
        return "flat"

    def _min_bars(self, timeframe: str) -> int:
        return max(self._ema_anchor + 10, 50)

    def _ensure_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        has_vol = any(k in cols for k in ("tick_volume", "volume"))
        if not has_vol:
            df = df.copy()
            df["tick_volume"] = 1.0
        return df

    def _detect_fvg(self, df: pd.DataFrame, atr: np.ndarray) -> str:
        """Detect Fair Value Gaps."""
        try:
            cols = {c.lower(): c for c in df.columns}
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values
            if len(h) < 3:
                return ""
            atr_last = safe_last(atr)
            if atr_last <= 0:
                return ""

            # Bullish FVG: gap between bar[-3] high and bar[-1] low
            gap = l[-1] - h[-3]
            if gap > atr_last * 0.5:
                return "bull_fvg"

            # Bearish FVG
            gap = l[-3] - h[-1]
            if gap > atr_last * 0.5:
                return "bear_fvg"
        except Exception:
            pass
        return ""

    def _liquidity_sweep(
        self, df: pd.DataFrame, *, atr_last: float, z_volume: float, adx: np.ndarray,
    ) -> str:
        """Detect liquidity sweeps (stop hunts)."""
        try:
            cols = {c.lower(): c for c in df.columns}
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values
            c = df[cols.get("close", "Close")].values
            if len(h) < 10 or atr_last <= 0:
                return ""

            # Lookback for recent high/low
            look = min(20, len(h) - 1)
            recent_high = float(np.max(h[-look:-1]))
            recent_low = float(np.min(l[-look:-1]))

            # Bullish sweep: price dips below recent low then closes above
            if l[-1] < recent_low and c[-1] > recent_low:
                if z_volume > 1.0:
                    return "bull_sweep"

            # Bearish sweep
            if h[-1] > recent_high and c[-1] < recent_high:
                if z_volume > 1.0:
                    return "bear_sweep"
        except Exception:
            pass
        return ""

    def _order_block(
        self, df: pd.DataFrame, *, atr_last: float, v: np.ndarray, z_vol_series: np.ndarray,
    ) -> str:
        """Detect order blocks (institutional candle patterns)."""
        try:
            cols = {c.lower(): c for c in df.columns}
            o = df[cols.get("open", "Open")].values
            c = df[cols.get("close", "Close")].values
            if len(o) < 5 or atr_last <= 0:
                return ""

            # Bullish OB: last down candle before a strong up move
            body_prev = c[-2] - o[-2]
            body_curr = c[-1] - o[-1]
            if body_prev < -atr_last * 0.5 and body_curr > atr_last:
                return "bull_ob"

            # Bearish OB
            if body_prev > atr_last * 0.5 and body_curr < -atr_last:
                return "bear_ob"
        except Exception:
            pass
        return ""

    def _stop_hunt_profile(
        self,
        *,
        df: pd.DataFrame,
        atr_last: float,
        z_volume: float,
        sweep: str,
    ) -> Tuple[str, float]:
        """
        Stop-hunt interpretation:
          - bull_sweep -> bear_trap (bullish response likely)
          - bear_sweep -> bull_trap (bearish response likely)
        Returns (side, signed_strength in [-1, +1]).
        """
        try:
            side = ""
            signed = 0.0
            if sweep == "bull_sweep":
                side = "bear_trap"
                signed = 1.0
            elif sweep == "bear_sweep":
                side = "bull_trap"
                signed = -1.0
            else:
                return "", 0.0

            cols = {c.lower(): c for c in df.columns}
            h = pd.to_numeric(df[cols.get("high", "High")], errors="coerce")
            l = pd.to_numeric(df[cols.get("low", "Low")], errors="coerce")
            o = pd.to_numeric(df[cols.get("open", "Open")], errors="coerce")
            c = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
            if len(h) < 3:
                return side, float(signed * 0.30)
            rng = float((h.iloc[-1] - l.iloc[-1]) or 0.0)
            if rng <= 0.0:
                return side, float(signed * 0.30)
            lower_wick = float(max(0.0, min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]))
            upper_wick = float(max(0.0, h.iloc[-1] - max(o.iloc[-1], c.iloc[-1])))
            wick_dom = (lower_wick - upper_wick) / max(rng, 1e-12)
            wick_mag = abs(wick_dom)
            atr_norm = min(1.0, rng / max(float(atr_last), 1e-12))
            vol_mag = min(1.0, max(0.0, float(z_volume)) / 3.0)
            strength = min(1.0, 0.40 * wick_mag + 0.35 * atr_norm + 0.25 * vol_mag)
            return side, float(np.clip(signed * strength, -1.0, 1.0))
        except Exception:
            return "", 0.0

    def _order_block_pretouch(
        self,
        *,
        df: pd.DataFrame,
        atr_last: float,
    ) -> Tuple[float, float]:
        """
        Order-block pre-touch anticipation:
          - proximity in [0, 1]: closeness to nearest recent OB anchor
          - bias in [-1, 1]: +bull / -bear pressure into that touch
        """
        try:
            cols = {c.lower(): c for c in df.columns}
            o = pd.to_numeric(df[cols.get("open", "Open")], errors="coerce")
            c = pd.to_numeric(df[cols.get("close", "Close")], errors="coerce")
            if len(o) < 30 or atr_last <= 0.0:
                return 0.0, 0.0
            body = c - o
            body_abs = body.abs()
            thr = float(max(1e-12, atr_last * 0.55))

            prev_o = o.shift(1)
            prev_body = body.shift(1)
            bull_ob = pd.Series(
                np.where((prev_body < 0.0) & (body > thr), prev_o, np.nan),
                index=df.index,
            ).ffill()
            bear_ob = pd.Series(
                np.where((prev_body > 0.0) & (body < -thr), prev_o, np.nan),
                index=df.index,
            ).ffill()

            cur = float(c.iloc[-1])
            b_anchor = float(bull_ob.iloc[-1]) if np.isfinite(bull_ob.iloc[-1]) else np.nan
            s_anchor = float(bear_ob.iloc[-1]) if np.isfinite(bear_ob.iloc[-1]) else np.nan
            dist_b = abs(cur - b_anchor) / atr_last if np.isfinite(b_anchor) else np.inf
            dist_s = abs(cur - s_anchor) / atr_last if np.isfinite(s_anchor) else np.inf
            nearest = float(min(dist_b, dist_s))
            prox = float(np.clip(1.0 - min(nearest, 3.0) / 3.0, 0.0, 1.0))

            if np.isfinite(dist_b) and np.isfinite(dist_s):
                bias = float(np.clip((dist_s - dist_b) / 3.0, -1.0, 1.0))
            elif np.isfinite(dist_b):
                bias = 0.5
            elif np.isfinite(dist_s):
                bias = -0.5
            else:
                bias = 0.0

            # Emphasize bias only when we are actually near a block.
            return prox, float(np.clip(bias * prox, -1.0, 1.0))
        except Exception:
            return 0.0, 0.0

    def _near_round_number(self, *, price: float, atr: float) -> bool:
        """Check if price is near a psychologically significant round number."""
        if price <= 0 or atr <= 0:
            return False
        threshold = atr * self._round_atr_mult
        for level in self._round_levels:
            remainder = price % level
            dist = min(remainder, level - remainder)
            if dist <= threshold:
                return True
        return False

    def _detect_divergence_swings(
        self, *, ind: np.ndarray, price: np.ndarray,
    ) -> str:
        """Detect bullish/bearish RSI divergence."""
        try:
            if len(ind) < 20 or len(price) < 20:
                return ""

            # Simple 2-point divergence on last 20 bars
            p = price[-20:]
            r = ind[-20:]

            # Find local lows
            p_low1 = float(np.min(p[:10]))
            p_low2 = float(np.min(p[10:]))
            r_low1 = float(np.min(r[:10]))
            r_low2 = float(np.min(r[10:]))

            # Bullish: price makes lower low, RSI makes higher low
            if p_low2 < p_low1 and r_low2 > r_low1:
                return "bull_div"

            # Bearish: price makes higher high, RSI makes lower high
            p_hi1 = float(np.max(p[:10]))
            p_hi2 = float(np.max(p[10:]))
            r_hi1 = float(np.max(r[:10]))
            r_hi2 = float(np.max(r[10:]))
            if p_hi2 > p_hi1 and r_hi2 < r_hi1:
                return "bear_div"
        except Exception:
            pass
        return ""

    def _detect_market_anomalies(
        self, *, dfp: pd.DataFrame, atr_series: np.ndarray,
        z_vol_series: np.ndarray, adx: float,
    ) -> AnomalyResult:
        """Detect market anomalies that should block or flag trading."""
        result = AnomalyResult()
        reasons = []

        try:
            cols = {c.lower(): c for c in dfp.columns}
            c = dfp[cols.get("close", "Close")].values

            # Flash crash detection: >3x ATR single-bar move
            if len(c) >= 2 and len(atr_series) >= 1:
                last_move = abs(c[-1] - c[-2])
                atr_last = safe_last(atr_series)
                if atr_last > 0 and last_move > atr_last * self._anomaly_atr_spike_mult:
                    result.score += 0.5
                    reasons.append(f"flash_move:{last_move/atr_last:.1f}xATR")

            # Volume spike
            z_vol_last = safe_last(z_vol_series)
            if z_vol_last > self._anomaly_z_vol_thresh:
                result.score += 0.3
                reasons.append(f"vol_spike:z={z_vol_last:.1f}")

            result.reasons = reasons
            if result.score > 0.7:
                result.blocked = True
                result.level = "critical"
            elif result.score > 0.3:
                result.level = "warning"
        except Exception:
            pass

        return result

    # ═════════════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ═════════════════════════════════════════════════════════════════

    def linreg_trend_slope(
        self, closes: np.ndarray, period: Optional[int] = None,
    ) -> float:
        """
        Linear regression slope normalized to [-1, +1].
        Used for H1/M15 trend direction detection.
        """
        if period is None:
            period = 20
        if len(closes) < period:
            return 0.0
        try:
            y = closes[-period:]
            x = np.arange(period, dtype=np.float64)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                return 0.0
            slope = np.sum((x - x_mean) * (y - y_mean)) / denom

            # Normalize: slope relative to price makes it dimensionless
            norm = slope / y_mean if y_mean != 0 else 0.0
            # Scale to [-1, 1]
            return float(max(-1.0, min(1.0, norm * 1000)))
        except Exception:
            return 0.0

    def volatility_regime(
        self, atr_series: np.ndarray, period: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Detect volatility regime based on ATR ratio.
        Returns (regime, atr_ratio).
        Note: caller passes shift-trimmed atr_series so [-1] is the correct bar.
        """
        if period is None:
            period = 50
        if len(atr_series) < period + 1:
            return "normal", 1.0
        try:
            current = float(atr_series[-1])
            sma = float(np.mean(atr_series[-period:]))
            if sma <= 0:
                return "normal", 1.0
            ratio = current / sma
            if ratio > 1.5:
                return "explosive", ratio
            if ratio < 0.5:
                return "dead", ratio
            return "normal", ratio
        except Exception:
            return "normal", 1.0

    def forecast_continuation(
        self, df: pd.DataFrame, vol_series: np.ndarray,
    ) -> str:
        """Predict next candle continuation: breakout + volume confirmation."""
        try:
            cols = {c.lower(): c for c in df.columns}
            c = df[cols.get("close", "Close")].values
            h = df[cols.get("high", "High")].values
            l = df[cols.get("low", "Low")].values

            if len(c) < 3:
                return "none"

            avg_vol = float(np.mean(vol_series[-20:])) if len(vol_series) >= 20 else 1.0
            last_vol = float(vol_series[-1]) if len(vol_series) > 0 else 0.0
            vol_mult = last_vol / avg_vol if avg_vol > 0 else 0.0

            if c[-1] > h[-2] and vol_mult > 1.5:
                return "bull_continuation"
            if c[-1] < l[-2] and vol_mult > 1.5:
                return "bear_continuation"
        except Exception:
            pass
        return "none"

    # ═════════════════════════════════════════════════════════════════
    # CONFLUENCE & SIGNAL STRENGTH
    # ═════════════════════════════════════════════════════════════════

    def _calculate_confluence_increment(
        self,
        *,
        has_fvg: bool,
        has_sweep: bool,
        has_ob: bool,
        has_near_rn: bool,
        has_div: bool,
        z_volume: float,
        trend: str,
        bb_width: float,
        vwap_dist_atr: float,
    ) -> float:
        """Calculate confluence bonus score (0.0 to 1.0)."""
        score = 0.0
        if has_fvg:
            score += 0.15
        if has_sweep:
            score += 0.20
        if has_ob:
            score += 0.15
        if has_near_rn:
            score += 0.10
        if has_div:
            score += 0.15
        if z_volume > 1.5:
            score += 0.10
        if trend in ("bull", "bear"):
            score += 0.05
        if bb_width > 0.02:
            score += 0.05
        if vwap_dist_atr < 1.0:
            score += 0.05
        return min(1.0, score)

    def _determine_signal_strength(
        self, score: float, m1_data: Dict[str, Any],
    ) -> str:
        if score >= 70:
            return "strong"
        elif score >= 50:
            return "moderate"
        elif score >= 30:
            return "weak"
        return "none"

    def _check_mtf_alignment(
        self,
        out: Dict[str, Any],
        frame_meta: Dict[str, Dict[str, Any]],
        audit: Optional[FeatureAuditResult] = None,
    ) -> None:
        """Check multi-timeframe trend alignment and closed-candle safety."""
        primary_tf = str(
            getattr(getattr(self.cfg, "symbol_params", None), "tf_primary", "") or ""
        )
        if primary_tf not in frame_meta and frame_meta:
            primary_tf = min(frame_meta, key=lambda x: int(tf_seconds(x)))
        primary_meta = frame_meta.get(primary_tf, {})
        primary_sec = int(primary_meta.get("tf_seconds", 0) or 0)
        primary_close_time = primary_meta.get("used_close_time")

        if primary_close_time is not None:
            for tf, meta in frame_meta.items():
                tf_sec = int(meta.get("tf_seconds", 0) or 0)
                used_close_time = meta.get("used_close_time")
                if tf_sec <= primary_sec or used_close_time is None:
                    continue
                if used_close_time > primary_close_time:
                    if audit is not None:
                        audit.add_issue(
                            f"{tf.lower()}_mtf_open_candle",
                            f"{tf} features were derived from a candle not yet closed",
                            MTF_ALIGNMENT_ERROR,
                            used_close_time=str(used_close_time),
                            primary_close_time=str(primary_close_time),
                        )
                    if isinstance(out.get(tf), dict):
                        out[tf]["mtf_closed"] = False
                elif isinstance(out.get(tf), dict):
                    out[tf]["mtf_closed"] = True

        trends = [v.get("trend", "flat") for v in out.values() if isinstance(v, dict)]
        if all(t == "bull" for t in trends):
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = True
                    v["mtf_direction"] = "bull"
        elif all(t == "bear" for t in trends):
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = True
                    v["mtf_direction"] = "bear"
        else:
            for v in out.values():
                if isinstance(v, dict):
                    v["mtf_aligned"] = False
                    v["mtf_direction"] = "mixed"

__all__ = (
    'DATA_VALID',
    'DATA_STALE',
    'DATA_INCOMPLETE',
    'ABNORMAL_SPREAD',
    'MARKET_UNUSABLE',
    'tf_seconds',
    'AuditIssue',
    'AuditResult',
    'SymbolSanityProfile',
    'Validator',
    'MarketDataValidator',
    'FEATURES_VALID',
    'NAN_DETECTED',
    'INSUFFICIENT_WARMUP',
    'LOOKAHEAD_BIAS_DETECTED',
    'MTF_ALIGNMENT_ERROR',
    'SCALING_ERROR',
    'safe_last',
    'AnomalyResult',
    'FeatureAuditIssue',
    'FeatureAuditResult',
    'FeatureIntegrityError',
    'FeatureEngine',
)
