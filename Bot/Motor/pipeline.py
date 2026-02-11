from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import MetaTrader5 as mt5

from ExnessAPI.functions import market_is_open
from mt5_client import MT5_LOCK

from .logging_setup import log_err, log_health
from .models import AssetCandidate
from .utils import parse_bar_key, tf_seconds

# =============================================================================
# CRITICAL: Stale Data Guard - Reject signals when data is too old
# =============================================================================
STALE_DATA_THRESHOLD_SEC = 2.0  # STRICT: Reject signals if data is older than 2 seconds


def _to_epoch_seconds(x: Any) -> Optional[float]:
    """Robust conversion for pandas/mt5 datetime representations to epoch seconds."""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            # Guard against RangeIndex / zero timestamps (non-epoch)
            return v if v >= 1_000_000_000.0 else None
        if isinstance(x, datetime):
            dt = x
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = float(dt.timestamp())
            return ts if ts >= 1_000_000_000.0 else None
        # pandas.Timestamp has .timestamp()
        ts_fn = getattr(x, "timestamp", None)
        if callable(ts_fn):
            ts = float(ts_fn())
            return ts if ts >= 1_000_000_000.0 else None
    except Exception:
        pass
    try:
        # numpy datetime64 / pandas can be cast via float? often fails, but keep safe
        v = float(x)  # type: ignore[arg-type]
        return v if v >= 1_000_000_000.0 else None
    except Exception:
        return None


class _AssetPipeline:
    def __init__(self, asset: str, cfg: Any, feed: Any, features: Any, risk: Any, signal: Any) -> None:
        self.asset = str(asset)
        self.cfg = cfg
        self.feed = feed
        self.features = features
        self.risk = risk
        self.signal = signal

        # Signal
        self.last_signal = "Neutral"
        self.last_signal_ts = 0.0
        self.last_latency_ms = 0.0

        # Market snapshot
        self.last_market_ok = True
        self.last_market_reason = "init"
        self.last_bar_age_sec = 0.0
        self.last_market_rows = 0
        self.last_market_close = 0.0
        self.last_market_volume = 0.0
        self.last_market_ts = "-"
        self.last_market_tf = (
            str(getattr(self.cfg.symbol_params, "tf_primary", "-"))
            if getattr(self.cfg, "symbol_params", None)
            else "-"
        )

        self._signal_log_every = 5.0
        self._last_signal_log_ts = 0.0

        # Market freshness thresholds
        self._max_bar_age_mult = float(getattr(cfg, "market_max_bar_age_mult", 2.0) or 2.0)
        # default tightened for scalping; can be overridden from cfg
        self._min_bar_age_sec = float(getattr(cfg, "market_min_bar_age_sec", 30.0) or 30.0)

        # Tick-based freshness (real-time data age)
        self.last_tick_age_sec = 0.0  # Age from most recent tick, not bar
        self._last_tick_ts = 0.0  # Timestamp of last tick update

        self._last_market_ok_ts = 0.0
        self._market_validate_every = float(getattr(cfg, "market_validate_interval_sec", 2.0) or 2.0)
        self._last_reconcile_ts = 0.0

        # Baseline sync (gap-safe)
        self._baseline_sync_bars = int(getattr(self.cfg, "baseline_sync_bars", 1) or 1)
        self._baseline_gap_mult = float(getattr(self.cfg, "baseline_gap_mult", 3.0) or 3.0)
        self._baseline_ready_bars = 0
        self._baseline_last_key: Optional[str] = None
        self._baseline_last_dt: Optional[datetime] = None

        self._reconcile_interval = float(getattr(cfg, "reconcile_interval_sec", 15.0) or 15.0)

        # Stale data guard metrics
        self._stale_data_rejections = 0
        self._last_stale_log_ts = 0.0
        self._stale_log_interval_sec = 5.0  # Log every 5 seconds max (throttled)

        # Signal caching to reduce CPU load
        self._last_computed_close = 0.0
        self._last_computed_ts = 0.0
        self._cache_ttl_sec = 0.3  # Only recompute if price changed or 300ms elapsed
        self._cached_candidate: Optional[AssetCandidate] = None

    def _baseline_gate(self, bar_key: str) -> Tuple[bool, str]:
        """Return (ok, reason). If ok=False => block trade for baseline sync."""
        dt = parse_bar_key(bar_key)
        if dt is None:
            return True, "baseline_ok(parse_skip)"

        sp = getattr(self.cfg, "symbol_params", None)
        tf_sec = tf_seconds(getattr(sp, "tf_primary", "M1") if sp else "M1") or 60.0
        gap_limit = max(60.0, self._baseline_gap_mult * float(tf_sec))

        if self._baseline_last_dt is None or self._baseline_last_key is None:
            self._baseline_last_dt = dt
            self._baseline_last_key = bar_key
            self._baseline_ready_bars = 0
            return (self._baseline_sync_bars <= 0), "baseline_warmup"

        gap = (dt - self._baseline_last_dt).total_seconds()

        if gap < -0.5 * float(tf_sec) or gap > gap_limit:
            self._baseline_last_dt = dt
            self._baseline_last_key = bar_key
            self._baseline_ready_bars = 0
            return (self._baseline_sync_bars <= 0), "baseline_reset_gap"

        if bar_key != self._baseline_last_key:
            self._baseline_ready_bars += 1
            self._baseline_last_key = bar_key
            self._baseline_last_dt = dt

        if self._baseline_ready_bars < self._baseline_sync_bars:
            return False, f"baseline_sync({self._baseline_ready_bars}/{self._baseline_sync_bars})"

        return True, "baseline_ok"

    @property
    def symbol(self) -> str:
        sp = getattr(self.cfg, "symbol_params", None)
        if not sp:
            return ""
        return str(getattr(sp, "resolved", "") or getattr(sp, "base", ""))

    def ensure_symbol_selected(self) -> None:
        symbol = self.symbol
        if not symbol:
            raise RuntimeError(f"{self.asset}: empty symbol")
        with MT5_LOCK:
            info = mt5.symbol_info(symbol)
            if info is None:
                if not mt5.symbol_select(symbol, True):
                    raise RuntimeError(f"{self.asset}: symbol_select failed: {symbol}")
            else:
                if hasattr(info, "visible") and (not bool(info.visible)):
                    if not mt5.symbol_select(symbol, True):
                        raise RuntimeError(f"{self.asset}: symbol_select failed (invisible): {symbol}")

    def _fetch_df_for_validation(self):
        tf = getattr(self.cfg.symbol_params, "tf_primary", None)
        if tf is None:
            self.last_market_reason = "tf_none"
            log_err.error("%s: tf_primary is None in config", self.asset)
            return None

        symbol = self.symbol
        if not symbol:
            self.last_market_reason = "symbol_empty"
            return None

        try:
            # Feed.get_rates(symbol, timeframe) — no tf= or count= kwargs
            df = self.feed.get_rates(symbol, tf)
            return df
        except Exception:
            self.last_market_reason = "feed_exception"
            return None

    def validate_market_data(self) -> bool:
        now = time.time()
        if (now - self._last_market_ok_ts) < self._market_validate_every:
            return bool(self.last_market_ok)

        self._last_market_ok_ts = now

        try:
            df = self._fetch_df_for_validation()
            if df is None:
                self.last_market_ok = False
                return False

            if len(df) <= 0:
                self.last_market_ok = False
                self.last_market_reason = "empty"
                return False

            last_ts_epoch: Optional[float] = None
            try:
                if hasattr(df, "columns") and "time" in df.columns:
                    last_ts_epoch = _to_epoch_seconds(df["time"].iloc[-1])
                if last_ts_epoch is None and hasattr(df, "index") and len(getattr(df, "index", [])) > 0:
                    idx = df.index
                    idx_val = idx[-1]
                    idx_type = str(getattr(idx, "inferred_type", "") or "").lower()
                    idx_dtype = str(getattr(idx, "dtype", "") or "").lower()
                    if "datetime" in idx_type or "datetime" in idx_dtype or isinstance(idx_val, datetime):
                        last_ts_epoch = _to_epoch_seconds(idx_val)
            except Exception:
                last_ts_epoch = None

            if last_ts_epoch is None:
                self.last_market_ok = False
                self.last_market_reason = "bad_time"
                return False

            if last_ts_epoch is not None:
                age = max(0.0, now - float(last_ts_epoch))
                self.last_bar_age_sec = float(age)
                sp = getattr(self.cfg, "symbol_params", None)
                tf_sec = tf_seconds(getattr(sp, "tf_primary", "M1") if sp else "M1") or 60.0
                max_age = max(float(self._min_bar_age_sec), self._max_bar_age_mult * float(tf_sec))
                if age > max_age:
                    self.last_market_ok = False
                    self.last_market_reason = "stale"
                    return False

            # Basic sanity on close
            if hasattr(df, "columns") and "close" in df.columns:
                try:
                    s = df["close"]
                    if getattr(s, "isna", lambda: False)().any():
                        self.last_market_ok = False
                        self.last_market_reason = "nan_close"
                        return False
                    if (s <= 0).any():
                        self.last_market_ok = False
                        self.last_market_reason = "bad_close"
                        return False
                    self.last_market_close = float(s.iloc[-1])
                except Exception:
                    pass

            self.last_market_rows = int(len(df))
            if hasattr(df, "columns") and "tick_volume" in df.columns:
                try:
                    self.last_market_volume = float(df["tick_volume"].iloc[-1])
                except Exception:
                    self.last_market_volume = 0.0
            else:
                self.last_market_volume = 0.0

            if last_ts_epoch is not None:
                try:
                    self.last_market_ts = datetime.utcfromtimestamp(float(last_ts_epoch)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    self.last_market_ts = "-"
            else:
                self.last_market_ts = "-"

            self.last_market_tf = str(getattr(self.cfg.symbol_params, "tf_primary", None) or "-")

            # ==============================================================
            # TICK FRESHNESS CHECK (Real-time data age, not bar age)
            # M1 bars only update at minute boundaries, but ticks are live
            # ==============================================================
            try:
                symbol = getattr(self.cfg.symbol_params, "resolved", None) or getattr(self.cfg.symbol_params, "base", "")
                with MT5_LOCK:
                    tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    tick_time = getattr(tick, "time", 0)
                    if tick_time > 0:
                        self._last_tick_ts = float(tick_time)
                        self.last_tick_age_sec = max(0.0, now - float(tick_time))
            except Exception:
                pass  # Keep existing tick age if fetch fails

            self.last_market_ok = True
            self.last_market_reason = "ok"
            return True

        except Exception as exc:
            self.last_market_ok = False
            self.last_market_reason = "exception"
            log_err.error("%s market_data_validate error: %s | tb=%s", self.asset, exc, traceback.format_exc())
            return False

    def reconcile_positions(self) -> None:
        now = time.time()
        if (now - self._last_reconcile_ts) < self._reconcile_interval:
            return
        self._last_reconcile_ts = now

        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                try:
                    magic = int(getattr(self.cfg, "magic", 777001) or 777001)
                except Exception:
                    magic = 777001
                pos = [p for p in pos if int(getattr(p, "magic", 0) or 0) == magic]
            if self.risk and hasattr(self.risk, "on_reconcile_positions"):
                try:
                    self.risk.on_reconcile_positions(pos)
                except Exception:
                    pass
        except Exception as exc:
            log_err.error("%s reconcile error: %s | tb=%s", self.asset, exc, traceback.format_exc())

    def open_positions(self) -> int:
        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                try:
                    magic = int(getattr(self.cfg, "magic", 777001) or 777001)
                except Exception:
                    magic = 777001
                pos = [p for p in pos if int(getattr(p, "magic", 0) or 0) == magic]
            return int(len(pos))
        except Exception:
            return 0

    def compute_candidate(self) -> Optional[AssetCandidate]:
        try:
            now = time.time()

            # =================================================================
            # CRITICAL FIX: STALE DATA GUARD (Uses TICK freshness, not bar age)
            # M1 bars only update at minute boundaries (always appear 0-60s old)
            # Ticks are real-time, so tick age is the true data freshness
            # =================================================================
            # Use tick age for freshness check (reads from config, default 60s)
            TICK_STALE_THRESHOLD_SEC = float(getattr(self.cfg, "tick_stale_threshold_sec", 60.0) or 60.0)
            data_age = self.last_tick_age_sec if self._last_tick_ts > 0 else self.last_bar_age_sec
            
            if data_age > TICK_STALE_THRESHOLD_SEC:
                self._stale_data_rejections += 1
                
                # Log periodically to avoid log spam (Time-based now)
                now_log = time.time()
                if (now_log - self._last_stale_log_ts) > self._stale_log_interval_sec:
                    self._last_stale_log_ts = now_log
                    log_health.warning(
                        "STALE_DATA_REJECT | asset=%s tick_age=%.1fs bar_age=%.1fs threshold=%.1fs total_rejections=%d (spam_throttled)",
                        self.asset,
                        self.last_tick_age_sec,
                        self.last_bar_age_sec,
                        TICK_STALE_THRESHOLD_SEC,
                        self._stale_data_rejections,
                    )
                return AssetCandidate(
                    asset=self.asset,
                    symbol=self.symbol,
                    signal="Neutral",
                    confidence=0.0,
                    lot=0.0,
                    sl=0.0,
                    tp=0.0,
                    latency_ms=0.0,
                    blocked=True,
                    reasons=("stale_data",),
                    signal_id=f"{self.asset}_STALE_{int(now)}",
                    raw_result=None,
                )

            # =================================================================
            # OPTIMIZATION: Signal Caching
            # Don't recompute if price hasn't changed and cache is fresh
            # =================================================================
            price_unchanged = abs(self.last_market_close - self._last_computed_close) < 0.01
            cache_fresh = (now - self._last_computed_ts) < self._cache_ttl_sec

            if price_unchanged and cache_fresh and self._cached_candidate is not None:
                # Return cached candidate but update timestamp
                return self._cached_candidate

            # execute=True SAFE (no side effects) -> calculates lot/sl/tp
            res = self.signal.compute(execute=True)

            signal = str(getattr(res, "signal", "Neutral") or "Neutral")
            self.last_signal = signal
            self.last_signal_ts = time.time()
            self.last_latency_ms = float(getattr(res, "latency_ms", 0.0) or 0.0)

            conf_raw = float(getattr(res, "confidence", 0.0) or 0.0)
            conf = conf_raw / 100.0 if conf_raw > 1 else conf_raw
            conf = max(0.0, min(1.0, float(conf)))

            lot = float(getattr(res, "lot", 0.0) or 0.0)
            sl = float(getattr(res, "sl", 0.0) or 0.0)
            tp = float(getattr(res, "tp", 0.0) or 0.0)

            blocked = bool(getattr(res, "trade_blocked", False))
            reasons = tuple(str(r) for r in (getattr(res, "reasons", []) or [])[:10])
            signal_id = str(getattr(res, "signal_id", "") or f"{self.asset}_SIG_{int(time.time() * 1000)}")

            now_ts = time.time()
            if (now_ts - self._last_signal_log_ts) >= self._signal_log_every:
                self._last_signal_log_ts = now_ts
                log_health.info(
                    "PIPELINE_STAGE | step=signals asset=%s signal=%s confidence=%.4f latency_ms=%.1f "
                    "lot=%.4f sl=%.4f tp=%.4f blocked=%s reasons=%s",
                    self.asset,
                    signal,
                    conf,
                    self.last_latency_ms,
                    lot,
                    sl,
                    tp,
                    blocked,
                    ",".join(reasons) if reasons else "-",
                )

            # Explicit block if market closed
            if self.asset == "XAU" and not market_is_open("XAU"):
                return AssetCandidate(
                    asset=self.asset,
                    symbol=self.symbol,
                    signal="Neutral",
                    confidence=0.0,
                    lot=0.0,
                    sl=0.0,
                    tp=0.0,
                    latency_ms=0.0,
                    blocked=True,
                    reasons=("market_closed_weekend",),
                    signal_id=f"{self.asset}_CLOSED_{int(time.time())}",
                    raw_result=None,
                )

            baseline_ok, baseline_reason = self._baseline_gate(getattr(res, "bar_key", "") or "")
            if not baseline_ok:
                blocked = True
                reasons = reasons + (baseline_reason,)

            candidate = AssetCandidate(
                asset=self.asset,
                symbol=self.symbol,
                signal=signal,
                confidence=conf,
                lot=lot,
                sl=sl,
                tp=tp,
                latency_ms=self.last_latency_ms,
                blocked=blocked,
                reasons=reasons,
                signal_id=signal_id,
                raw_result=res,
            )

            # Store in cache for optimization
            self._last_computed_close = self.last_market_close
            self._last_computed_ts = now
            self._cached_candidate = candidate

            return candidate
        except Exception as exc:
            log_err.error("%s compute_candidate error: %s | tb=%s", self.asset, exc, traceback.format_exc())
            return None
