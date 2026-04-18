"""
Bot/Motor/pipeline.py - Portfolio engine pipeline utilities and scheduling.

Ин модул мантиқи вақтбандии бозор, назорати зичии сигналҳо,
ва Pipeline-и дохилии як активро (AssetPipeline) дар бар мегирад.
"""

from __future__ import annotations

import os
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque, List, Optional, Tuple

import MetaTrader5 as mt5

# =============================================================================
# Imports
# =============================================================================
from core.config import get_config_from_env as _get_core_config
from core.feature_engine import FeatureEngine
from core.risk_manager import RiskManager
from core.signal_engine import SignalEngine
from DataFeed.btc_market_feed import MarketFeed as BtcMarketFeed
from DataFeed.xau_market_feed import MarketFeed as XauMarketFeed
from ExnessAPI.functions import market_is_open
from mt5_client import MT5_LOCK, ensure_mt5, mt5_async_call, mt5_status

from .models import AssetCandidate, log_err, log_health, parse_bar_key, tf_seconds

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine

# =============================================================================
# Global Constants / States
# =============================================================================
# CRITICAL: Stale Data Guard - Reject signals when data is too old
STALE_DATA_THRESHOLD_SEC = 2.0  # STRICT: Reject signals if data is older than 2 seconds


# =============================================================================
# Classes
# =============================================================================
class UTCScheduler:
    """Source of Truth for Day/Time logic.

    Weekdays (Mon-Fri): XAU + BTC
    Weekends (Sat-Sun): BTC only
    """

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def is_weekend() -> bool:
        # 5=Sat, 6=Sun
        return UTCScheduler.now_utc().weekday() >= 5

    @staticmethod
    def get_active_assets() -> Tuple[str, ...]:
        if UTCScheduler.is_weekend():
            return ("BTC",)
        return ("XAU", "BTC")

    @staticmethod
    def market_status(asset: str) -> bool:
        """Internal market open check.

        BTC: always open (24/7).
        XAU: weekdays only (hours logic guarded elsewhere by MT5/market feed).
        """
        if asset == "BTC":
            return True
        if asset == "XAU":
            return not UTCScheduler.is_weekend()
        return False


class EngineScheduleManager:
    """Time/phase/idempotency coordinator extracted from the engine."""

    def __init__(self, engine: MultiAssetTradingEngine) -> None:
        self._e = engine

    def refresh_signal_cooldowns(self) -> None:
        e = self._e
        try:
            if (
                e._signal_cooldown_override_sec is not None
                and e._signal_cooldown_override_sec > 0
            ):
                cd = float(e._signal_cooldown_override_sec)
                e._signal_cooldown_sec_by_asset["XAU"] = cd
                e._signal_cooldown_sec_by_asset["BTC"] = cd
                return

            def _pipe_tf_sec(pipe: Any) -> float:
                try:
                    if pipe is None:
                        return 60.0
                    tf = getattr(
                        getattr(pipe.cfg, "symbol_params", None), "tf_primary", None
                    )
                    sec = tf_seconds(tf)
                    return float(sec) if sec else 60.0
                except Exception:
                    return 60.0

            e._signal_cooldown_sec_by_asset["XAU"] = max(_pipe_tf_sec(e._xau), 60.0)
            e._signal_cooldown_sec_by_asset["BTC"] = max(_pipe_tf_sec(e._btc), 60.0)
        except Exception:
            e._signal_cooldown_sec_by_asset["XAU"] = 60.0
            e._signal_cooldown_sec_by_asset["BTC"] = 60.0

    @staticmethod
    def get_phase(risk: Any) -> str:
        if risk is None:
            return "A"
        for attr in ("current_phase", "phase", "mode", "risk_phase"):
            v = getattr(risk, attr, None)
            if v:
                return str(v)
        return "A"

    @staticmethod
    def get_daily_date(risk: Any) -> str:
        if risk is None:
            return ""
        for attr in ("daily_date", "today_date", "day_key", "session_date"):
            v = getattr(risk, attr, None)
            if v:
                return str(v)
        return ""

    def phase_reason(self, risk: Any, new_phase: str) -> str:
        if risk is None:
            return ""
        fn = getattr(risk, "phase_reason", None)
        if callable(fn):
            try:
                return str(fn() or "")
            except Exception:
                return ""
        if new_phase == "C":
            fn = getattr(risk, "hard_stop_reason", None)
            if callable(fn):
                try:
                    return str(fn() or "")
                except Exception:
                    return ""
        return ""

    def check_phase_change(self, asset: str, risk: Any) -> None:
        e = self._e
        try:
            if risk is None:
                return
            asset_u = str(asset).upper()
            if not UTCScheduler.market_status(asset_u):
                return
            fn = getattr(risk, "update_phase", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
            current = self.get_phase(risk) or "A"
            prev = str(e._last_phase_by_asset.get(asset_u, "A") or "A")
            if current != prev:
                e._last_phase_by_asset[asset_u] = current
                if e._phase_notifier:
                    reason = self.phase_reason(risk, current)
                    e._phase_notifier(asset_u, prev, current, reason)
        except Exception:
            return

    def check_daily_start(self, asset: str, risk: Any) -> None:
        e = self._e
        try:
            if risk is None:
                return
            asset_u = str(asset).upper()
            if not UTCScheduler.market_status(asset_u):
                return
            current_date = self.get_daily_date(risk)
            if not current_date:
                return
            prev_date = str(e._last_daily_date_by_asset.get(asset_u, "") or "")
            if not prev_date:
                e._last_daily_date_by_asset[asset_u] = current_date
                return
            if current_date != prev_date:
                e._last_daily_date_by_asset[asset_u] = current_date
                if e._daily_start_notifier:
                    e._daily_start_notifier(asset_u, current_date)
        except Exception:
            return

    def cooldown_for_asset(self, asset: str) -> float:
        return float(self._e._signal_cooldown_sec_by_asset.get(asset, 60.0))

    def is_duplicate(
        self,
        asset: str,
        signal_id: str,
        now: float,
        max_orders: int,
        *,
        order_index: int = 0,
    ) -> bool:
        e = self._e
        last = e._seen_index.get((asset, signal_id))
        if not last:
            return False
        last_ts, count = last

        max_orders = max(1, int(max_orders))
        if int(count) >= int(max_orders):
            return True

        cd = self.cooldown_for_asset(asset)
        if (now - float(last_ts)) < float(cd):
            if int(order_index) > 0 and int(count) < int(max_orders):
                return False
            return True

        min_scale_delay = 30.0
        if (
            int(count) > 0
            and int(order_index) <= 0
            and (now - float(last_ts)) < min_scale_delay
        ):
            return True

        return False

    def mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        e = self._e
        key = (asset, signal_id)
        last = e._seen_index.get(key)
        count = 0
        if last:
            _, last_count = last
            count = int(last_count)

        count += 1
        e._seen_index[key] = (float(now), int(count))
        e._seen.append((asset, signal_id, now))

        max_cd = max(e._signal_cooldown_sec_by_asset.values(), default=60.0)
        ttl = max(2.0 * float(max_cd), 120.0)
        cleaned = 0
        while (
            e._seen and (now - e._seen[0][2]) > ttl and cleaned < e._seen_cleanup_budget
        ):
            a, sid, ts = e._seen.popleft()
            rec = e._seen_index.get((a, sid))
            if rec and float(rec[0]) == float(ts):
                e._seen_index.pop((a, sid), None)
            cleaned += 1

    @staticmethod
    def prune_signal_window(
        window: Deque[float], now_ts: float, lookback_sec: float = 86_400.0
    ) -> None:
        while window and (now_ts - float(window[0])) > float(lookback_sec):
            window.popleft()

    @staticmethod
    def p95(values: Deque[float] | List[float]) -> float:
        seq = [float(v) for v in values if float(v) >= 0.0]
        if len(seq) < 3:
            return 0.0
        seq.sort()
        idx = int(round((len(seq) - 1) * 0.95))
        idx = max(0, min(len(seq) - 1, idx))
        return float(seq[idx])

    @staticmethod
    def utc_day_progress() -> float:
        now = datetime.utcnow()
        sec = (now.hour * 3600) + (now.minute * 60) + now.second
        return max(0.0, min(1.0, float(sec) / 86400.0))

    def signal_density_controls(
        self, asset: str, now_ts: float
    ) -> Tuple[float, float, float, int]:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        window = e._signal_emit_ts_by_asset.setdefault(asset_u, deque(maxlen=512))
        global_window = e._signal_emit_ts_global
        self.prune_signal_window(window, now_ts)
        self.prune_signal_window(global_window, now_ts)

        count_24h = int(len(window))
        global_count_24h = int(len(global_window))
        try:
            active_assets = max(1, len(tuple(UTCScheduler.get_active_assets())))
        except Exception:
            active_assets = 2

        min_target = float(
            max(1.0, float(e._target_signals_per_24h_min) / float(active_assets))
        )
        max_target = float(
            max(min_target, float(e._target_signals_per_24h_max) / float(active_assets))
        )

        threshold_mult = 1.0
        min_zscore = float(e._catboost_min_zscore)
        min_flow = float(e._catboost_min_flow_confirm)

        if count_24h < min_target:
            deficit = (min_target - float(count_24h)) / max(min_target, 1.0)
            relax = min(0.40, 0.40 * deficit)
            threshold_mult = max(0.60, 1.0 - relax)
            min_zscore = max(0.20, min_zscore * (1.0 - (0.45 * deficit)))
            min_flow = max(0.01, min_flow * (1.0 - (0.45 * deficit)))
        elif count_24h > max_target:
            excess = (float(count_24h) - max_target) / max(max_target, 1.0)
            tighten = min(0.25, 0.25 * excess)
            threshold_mult = min(1.25, 1.0 + tighten)
            min_zscore = min(1.75, min_zscore * (1.0 + (0.20 * excess)))
            min_flow = min(0.20, min_flow * (1.0 + (0.20 * excess)))

        day_progress = self.utc_day_progress()
        min_target_total = float(e._target_signals_per_24h_min)
        max_target_total = float(e._target_signals_per_24h_max)
        expected_min_now = min_target_total * day_progress

        if global_count_24h < expected_min_now:
            pace_deficit = (expected_min_now - float(global_count_24h)) / max(
                min_target_total, 1.0
            )
            urgency = min(1.0, max(0.0, pace_deficit))
            threshold_mult = max(0.52, float(threshold_mult) * (1.0 - (0.28 * urgency)))
            min_zscore = max(0.12, float(min_zscore) * (1.0 - (0.35 * urgency)))
            min_flow = max(0.005, float(min_flow) * (1.0 - (0.35 * urgency)))
        elif global_count_24h > max_target_total:
            overflow = (float(global_count_24h) - max_target_total) / max(
                max_target_total, 1.0
            )
            pressure = min(1.0, max(0.0, overflow))
            threshold_mult = min(
                1.35, float(threshold_mult) * (1.0 + (0.20 * pressure))
            )
            min_zscore = min(1.95, float(min_zscore) * (1.0 + (0.25 * pressure)))
            min_flow = min(0.25, float(min_flow) * (1.0 + (0.25 * pressure)))

        return threshold_mult, min_zscore, min_flow, count_24h

    def record_signal_emit(self, asset: str, now_ts: float) -> None:
        e = self._e
        asset_u = str(asset or "").upper().strip()
        window = e._signal_emit_ts_by_asset.setdefault(asset_u, deque(maxlen=512))
        self.prune_signal_window(window, now_ts)
        self.prune_signal_window(e._signal_emit_ts_global, now_ts)
        window.append(float(now_ts))
        e._signal_emit_ts_global.append(float(now_ts))

    @staticmethod
    def select_active_asset(open_xau: int, open_btc: int) -> str:
        active_assets = UTCScheduler.get_active_assets()

        if "XAU" not in active_assets:
            if open_xau > 0 and open_btc == 0:
                return "XAU"
            return "BTC"

        if open_xau > 0 and open_btc > 0:
            return "BOTH"
        if open_xau > 0:
            return "XAU"
        if open_btc > 0:
            return "BTC"
        return "XAU+BTC"


class _AssetPipeline:
    def __init__(
        self, asset: str, cfg: Any, feed: Any, features: Any, risk: Any, signal: Any
    ) -> None:
        self.asset = str(asset)
        self.cfg = cfg
        self.feed = feed
        self.features = features
        self.risk = risk
        self.signal = signal

        self.last_signal = "Neutral"
        self.last_signal_ts = 0.0
        self.last_latency_ms = 0.0

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

        self._rejection_counts: dict = {}
        self._rejection_log_every = 300.0
        self._rejection_last_log_ts = 0.0
        self._total_cycles = 0
        self._total_signals = 0

        self._max_bar_age_mult = float(
            getattr(cfg, "market_max_bar_age_mult", 2.0) or 2.0
        )
        self._min_bar_age_sec = float(
            getattr(cfg, "market_min_bar_age_sec", 30.0) or 30.0
        )

        self.last_tick_age_sec = 0.0
        self._last_tick_ts = 0.0

        self._last_market_ok_ts = 0.0
        self._market_validate_every = float(
            getattr(cfg, "market_validate_interval_sec", 2.0) or 2.0
        )
        self._last_reconcile_ts = 0.0

        self._baseline_sync_bars = int(getattr(self.cfg, "baseline_sync_bars", 1) or 1)
        self._baseline_gap_mult = float(
            getattr(self.cfg, "baseline_gap_mult", 3.0) or 3.0
        )
        self._baseline_ready_bars = 0
        self._baseline_last_key: Optional[str] = None
        self._baseline_last_dt: Optional[datetime] = None

        self._reconcile_interval = float(
            getattr(cfg, "reconcile_interval_sec", 15.0) or 15.0
        )

        self._stale_data_rejections = 0
        self._last_stale_log_ts = 0.0
        self._stale_log_interval_sec = 5.0

        self._last_computed_close = 0.0
        self._last_computed_ts = 0.0
        self._cache_ttl_sec = 0.3
        self._cached_candidate: Optional[AssetCandidate] = None

    def _baseline_gate(self, bar_key: str) -> Tuple[bool, str]:
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
            return (
                False,
                f"baseline_sync({self._baseline_ready_bars}/{self._baseline_sync_bars})",
            )

        return True, "baseline_ok"

    @property
    def symbol(self) -> str:
        sp = getattr(self.cfg, "symbol_params", None)
        if not sp:
            return ""
        return str(getattr(sp, "resolved", "") or getattr(sp, "base", ""))

    def _apply_positions_filter(self, positions: List[Any]) -> List[Any]:
        if bool(getattr(self.cfg, "ignore_external_positions", False)):
            try:
                magic = int(getattr(self.cfg, "magic", 777001) or 777001)
            except Exception:
                magic = 777001
            return [p for p in positions if int(getattr(p, "magic", 0) or 0) == magic]
        return positions

    def ensure_symbol_selected(self) -> None:
        symbol = self.symbol
        if not symbol:
            raise RuntimeError(f"{self.asset}: empty symbol")

        if mt5.terminal_info() is None:
            log_health.warning(
                "%s: terminal_info=None before ensure_symbol_selected, reconnecting...",
                self.asset,
            )
            try:
                ensure_mt5()
            except Exception as exc:
                raise RuntimeError(f"{self.asset}: ensure_mt5 failed: {exc}") from exc

        last_err: Optional[str] = None
        for attempt in range(1, 4):
            with MT5_LOCK:
                info = mt5.symbol_info(symbol)
                needs_select = info is None or (
                    hasattr(info, "visible") and not bool(info.visible)
                )
                if not needs_select:
                    return
                if mt5.symbol_select(symbol, True):
                    return
                last_err = f"symbol_select failed: {symbol}"

            if attempt < 3:
                log_health.warning(
                    "%s: ensure_symbol_selected attempt %d failed (%s), retrying in 2s...",
                    self.asset,
                    attempt,
                    last_err,
                )
                time.sleep(2.0)

        raise RuntimeError(
            f"{self.asset}: {last_err or f'symbol_select failed: {symbol}'}"
        )

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
            df = self.feed.get_rates(symbol, tf)
            if df is None:
                self.last_market_reason = (
                    "no_rates_dry_run"
                    if bool(getattr(self.cfg, "dry_run", False))
                    else "no_rates"
                )
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
                if (
                    last_ts_epoch is None
                    and hasattr(df, "index")
                    and len(getattr(df, "index", [])) > 0
                ):
                    idx = df.index
                    idx_val = idx[-1]
                    if isinstance(idx_val, datetime):
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
                tf_sec = (
                    tf_seconds(getattr(sp, "tf_primary", "M1") if sp else "M1") or 60.0
                )
                max_age = max(
                    float(self._min_bar_age_sec), self._max_bar_age_mult * float(tf_sec)
                )
                if age > max_age:
                    self.last_market_ok = False
                    self.last_market_reason = "stale"
                    return False

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
                    self.last_market_ts = datetime.utcfromtimestamp(
                        float(last_ts_epoch)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    self.last_market_ts = "-"
            else:
                self.last_market_ts = "-"

            self.last_market_tf = str(
                getattr(self.cfg.symbol_params, "tf_primary", None) or "-"
            )

            try:
                symbol = getattr(self.cfg.symbol_params, "resolved", None) or getattr(
                    self.cfg.symbol_params, "base", ""
                )
                tick = mt5_async_call(
                    "symbol_info_tick", symbol, timeout=0.35, default=None
                )
                if tick is not None:
                    tick_time = getattr(tick, "time", 0)
                    if tick_time > 0:
                        self._last_tick_ts = float(tick_time)
                        self.last_tick_age_sec = max(0.0, now - float(tick_time))
            except Exception:
                pass

            self.last_market_ok = True
            self.last_market_reason = "ok"
            return True

        except Exception as exc:
            self.last_market_ok = False
            self.last_market_reason = "exception"
            log_err.error(
                "%s market_data_validate error: %s | tb=%s",
                self.asset,
                exc,
                traceback.format_exc(),
            )
            return False

    def reconcile_positions(self) -> None:
        now = time.time()
        if (now - self._last_reconcile_ts) < self._reconcile_interval:
            return
        self._last_reconcile_ts = now

        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            pos = self._apply_positions_filter(pos)
            if self.risk and hasattr(self.risk, "on_reconcile_positions"):
                try:
                    self.risk.on_reconcile_positions(pos)
                except Exception:
                    pass
        except Exception as exc:
            log_err.error(
                "%s reconcile error: %s | tb=%s",
                self.asset,
                exc,
                traceback.format_exc(),
            )

    def open_positions(self) -> int:
        try:
            with MT5_LOCK:
                pos = mt5.positions_get(symbol=self.symbol) or []
            pos = self._apply_positions_filter(pos)
            return int(len(pos))
        except Exception:
            return 0

    def compute_candidate(self) -> Optional[AssetCandidate]:
        try:
            now = time.time()

            TICK_STALE_THRESHOLD_SEC = float(
                getattr(self.cfg, "tick_stale_threshold_sec", 60.0) or 60.0
            )
            data_age = (
                self.last_tick_age_sec
                if self._last_tick_ts > 0
                else self.last_bar_age_sec
            )

            if data_age > TICK_STALE_THRESHOLD_SEC:
                self._stale_data_rejections += 1
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

            price_unchanged = (
                abs(self.last_market_close - self._last_computed_close) < 0.01
            )
            cache_fresh = (now - self._last_computed_ts) < self._cache_ttl_sec

            if price_unchanged and cache_fresh and self._cached_candidate is not None:
                return self._cached_candidate

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
            signal_id = str(
                getattr(res, "signal_id", "")
                or f"{self.asset}_SIG_{int(time.time() * 1000)}"
            )

            self._total_cycles += 1
            if signal in ("Buy", "Sell", "Strong Buy", "Strong Sell"):
                self._total_signals += 1
            for r in reasons:
                key = r.split(":")[0] if ":" in r else r
                self._rejection_counts[key] = self._rejection_counts.get(key, 0) + 1

            now_ts = time.time()
            if (now_ts - self._rejection_last_log_ts) >= self._rejection_log_every:
                self._rejection_last_log_ts = now_ts
                top_reasons = sorted(
                    self._rejection_counts.items(), key=lambda x: -x[1]
                )[:8]
                top_str = (
                    " ".join(f"{k}={v}" for k, v in top_reasons)
                    if top_reasons
                    else "none"
                )
                log_health.info(
                    "FILTER_REJECTION_STATS | asset=%s cycles=%d signals=%d signal_rate=%.3f top_rejections=[%s]",
                    self.asset,
                    self._total_cycles,
                    self._total_signals,
                    (self._total_signals / max(1, self._total_cycles)),
                    top_str,
                )

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

            if "same_bar_dedup" not in reasons:
                baseline_ok, baseline_reason = self._baseline_gate(
                    getattr(res, "bar_key", "") or ""
                )
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

            self._last_computed_close = self.last_market_close
            self._last_computed_ts = now
            self._cached_candidate = candidate

            return candidate
        except Exception as exc:
            log_err.error(
                "%s compute_candidate error: %s | tb=%s",
                self.asset,
                exc,
                traceback.format_exc(),
            )
            return None


class EnginePipelineMixin:
    def _expected_mt5_identity(self) -> Tuple[int, str]:
        login = int(getattr(self, "_expected_mt5_login", 0) or 0)
        server = str(getattr(self, "_expected_mt5_server", "") or "").strip()
        if login > 0 or server:
            return login, server

        for cfg in (getattr(self, "_xau_cfg", None), getattr(self, "_btc_cfg", None)):
            if cfg is None:
                continue
            if login <= 0:
                try:
                    login = int(getattr(cfg, "login", 0) or 0)
                except Exception:
                    login = 0
            if not server:
                server = str(getattr(cfg, "server", "") or "").strip()
        return login, server

    def _mt5_identity_ok(self, acc: Any) -> Tuple[bool, str]:
        if not acc:
            return False, "account_missing"

        expected_login, expected_server = self._expected_mt5_identity()
        actual_login = int(getattr(acc, "login", 0) or 0)
        actual_server = str(getattr(acc, "server", "") or "").strip()

        if expected_login > 0 and actual_login != expected_login:
            return False, f"wrong_account:{actual_login}!={expected_login}"
        if expected_server and actual_server.lower() != expected_server.lower():
            return False, f"wrong_server:{actual_server or '-'}!={expected_server}"
        return True, "ok"

    @staticmethod
    def _mt5_identity_mismatch(reason: str) -> bool:
        reason_s = str(reason or "").strip().lower()
        return reason_s.startswith("wrong_account:") or reason_s.startswith(
            "wrong_server:"
        )

    def _mt5_session_status(self) -> Tuple[bool, str]:
        with MT5_LOCK:
            term = mt5.terminal_info()
            acc = mt5.account_info()
        if not term:
            return False, "terminal_missing"
        if not getattr(term, "connected", False):
            return False, "terminal_not_connected"
        if not getattr(term, "trade_allowed", False):
            return False, "terminal_trade_disabled"
        if not acc:
            return False, "account_missing"
        if not getattr(acc, "trade_allowed", True):
            return False, "account_trade_disabled"

        identity_ok, identity_reason = self._mt5_identity_ok(acc)
        if not identity_ok:
            return False, identity_reason
        return True, "ok"

    def _init_mt5(self) -> bool:
        try:
            ensure_mt5()
            with MT5_LOCK:
                acc = mt5.account_info()
                term = mt5.terminal_info()
            if not acc or not term or not getattr(term, "connected", False):
                log_err.error(
                    "MT5 init failed | acc=%s term=%s connected=%s",
                    bool(acc),
                    bool(term),
                    getattr(term, "connected", False) if term else None,
                )
                return False

            identity_ok, identity_reason = self._mt5_identity_ok(acc)
            if not identity_ok:
                self._mt5_ready = False
                with self._lock:
                    self._manual_stop = True
                log_err.critical(
                    "MT5_INIT_IDENTITY_MISMATCH | reason=%s login=%s server=%s",
                    identity_reason,
                    getattr(acc, "login", "-"),
                    getattr(acc, "server", "-"),
                )
                return False

            self._mt5_ready = True
            log_health.info(
                "MT5_INIT_OK | login=%s server=%s term_connected=%s term_trade=%s acc_trade=%s",
                getattr(acc, "login", "-"),
                getattr(acc, "server", "-"),
                getattr(term, "connected", False),
                getattr(term, "trade_allowed", False),
                getattr(acc, "trade_allowed", False),
            )
            return True
        except Exception as exc:
            self._mt5_ready = False
            if self._is_non_retriable_mt5_error(exc):
                with self._lock:
                    if not self._manual_stop:
                        self._manual_stop = True
                        log_health.warning(
                            "MANUAL_STOP_REQUESTED | reason=mt5_config_block detail=%s",
                            exc,
                        )
                log_health.warning("MT5_INIT_BLOCKED | %s", exc)
                return False
            log_err.error("MT5 init error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _mt5_session_ok(self) -> bool:
        ok, _reason = self._mt5_session_status()
        return ok

    def _throttled_mt5_health_log(
        self, level: str, message: str, *, ttl_sec: Optional[float] = None
    ) -> None:
        ttl = max(
            2.0,
            float(
                ttl_sec
                if ttl_sec is not None
                else (os.getenv("MT5_HEALTH_LOG_TTL_SEC", "8.0") or 8.0)
            ),
        )
        now = time.time()
        last_sig = str(getattr(self, "_mt5_health_log_sig", "") or "")
        last_ts = float(getattr(self, "_mt5_health_log_ts", 0.0) or 0.0)

        if message == last_sig and (now - last_ts) < ttl:
            return

        setattr(self, "_mt5_health_log_sig", message)
        setattr(self, "_mt5_health_log_ts", now)

        logger_fn = (
            log_err.error if str(level).lower() == "error" else log_health.warning
        )
        logger_fn("%s", message)

    def _check_mt5_health(self) -> bool:
        if getattr(self, "dry_run", False):
            return True
        try:
            self._touch_runtime_progress()
            session_ok, session_reason = self._mt5_session_status()
            if session_ok:
                self._mt5_ready = True
                setattr(self, "_mt5_health_fail_count", 0)
                return True

            self._mt5_ready = False
            fail_count = int(getattr(self, "_mt5_health_fail_count", 0) or 0) + 1
            setattr(self, "_mt5_health_fail_count", fail_count)
            if self._mt5_identity_mismatch(session_reason):
                with self._lock:
                    self._manual_stop = True
                status_ok, status_reason = mt5_status()
                log_err.critical(
                    "MT5_IDENTITY_MISMATCH | reason=%s status_ok=%s status_reason=%s | action=manual_stop",
                    session_reason,
                    status_ok,
                    status_reason,
                )
                return False

            status_ok, status_reason = mt5_status()
            self._throttled_mt5_health_log(
                "warning",
                (
                    "MT5_HEALTH_DEGRADED | status_ok=%s reason=%s "
                    "session_reason=%s fail_count=%d | action=ensure_mt5"
                )
                % (
                    status_ok,
                    status_reason,
                    session_reason,
                    fail_count,
                ),
            )

            ensure_mt5()
            self._touch_runtime_progress()
            recovered, recovered_reason = self._mt5_session_status()
            self._mt5_ready = bool(recovered)
            if not recovered:
                status_ok, status_reason = mt5_status()
                if self._mt5_identity_mismatch(recovered_reason):
                    with self._lock:
                        self._manual_stop = True
                    log_err.critical(
                        "MT5_IDENTITY_MISMATCH | reason=%s status_ok=%s status_reason=%s | action=manual_stop",
                        recovered_reason,
                        status_ok,
                        status_reason,
                    )
                else:
                    self._throttled_mt5_health_log(
                        "warning",
                        (
                            "MT5_HEALTH_RECOVERY_INCOMPLETE | status_ok=%s "
                            "reason=%s session_reason=%s fail_count=%d"
                        )
                        % (
                            status_ok,
                            status_reason,
                            recovered_reason,
                            fail_count,
                        ),
                    )
            else:
                setattr(self, "_mt5_health_fail_count", 0)
            return recovered
        except Exception as exc:
            status_ok, status_reason = mt5_status()
            fail_count = int(getattr(self, "_mt5_health_fail_count", 0) or 0) + 1
            setattr(self, "_mt5_health_fail_count", fail_count)
            self._throttled_mt5_health_log(
                "error",
                (
                    "MT5_HEALTH_EXCEPTION | err=%s status_ok=%s reason=%s "
                    "fail_count=%d | tb=%s"
                )
                % (
                    exc,
                    status_ok,
                    status_reason,
                    fail_count,
                    traceback.format_exc(),
                ),
                ttl_sec=12.0,
            )
            return False

    def _sync_symbol_params_from_mt5(self, cfg: Any, asset: str) -> None:
        if getattr(self, "dry_run", False):
            return

        sp = getattr(cfg, "symbol_params", None)
        if sp is None:
            return

        symbol = str(
            getattr(sp, "resolved", None) or getattr(sp, "base", "") or ""
        ).strip()
        if not symbol:
            return

        try:
            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(symbol)
                if info is None or (
                    hasattr(info, "visible") and not bool(info.visible)
                ):
                    try:
                        mt5.symbol_select(symbol, True)
                    except Exception:
                        pass
                    info = mt5.symbol_info(symbol)

            if info is None:
                log_health.warning(
                    "SYMBOL_SPEC_SYNC_SKIP | asset=%s symbol=%s reason=symbol_info_none",
                    asset,
                    symbol,
                )
                return

            resolved = str(getattr(info, "name", symbol) or symbol).strip() or symbol
            digits = int(getattr(info, "digits", 0) or 0)
            point = float(getattr(info, "point", 0.0) or 0.0)
            contract_size = float(
                getattr(info, "trade_contract_size", 0.0)
                or getattr(info, "contract_size", 0.0)
                or 0.0
            )
            lot_step = float(getattr(info, "volume_step", 0.0) or 0.0)
            lot_min = float(getattr(info, "volume_min", 0.0) or 0.0)
            lot_max = float(getattr(info, "volume_max", 0.0) or 0.0)

            setattr(sp, "resolved", resolved)
            if digits > 0:
                setattr(sp, "digits", digits)
            if contract_size > 0.0:
                setattr(sp, "contract_size", contract_size)
            if lot_step > 0.0:
                setattr(sp, "lot_step", lot_step)
            if lot_min > 0.0:
                setattr(sp, "lot_min", lot_min)
            if lot_max > 0.0:
                setattr(sp, "lot_max", lot_max)

            point_value = 0.0
            if point > 0.0:
                ref_contract = float(
                    getattr(sp, "contract_size", 0.0) or contract_size or 0.0
                )
                if ref_contract > 0.0:
                    point_value = float(point) * ref_contract
            if point_value > 0.0:
                setattr(sp, "point_value", point_value)

            log_health.info(
                "SYMBOL_SPEC_SYNC | asset=%s symbol=%s digits=%d point=%.6f contract_size=%.4f lot_step=%.4f lot_min=%.4f lot_max=%.4f",
                asset,
                resolved,
                int(getattr(sp, "digits", 0) or 0),
                float(point),
                float(getattr(sp, "contract_size", 0.0) or 0.0),
                float(getattr(sp, "lot_step", 0.0) or 0.0),
                float(getattr(sp, "lot_min", 0.0) or 0.0),
                float(getattr(sp, "lot_max", 0.0) or 0.0),
            )
        except Exception as exc:
            log_health.warning(
                "SYMBOL_SPEC_SYNC_FAILED | asset=%s symbol=%s err=%s",
                asset,
                symbol,
                exc,
            )

    def _build_pipelines(self) -> bool:
        if not self._build_pipelines_lock.acquire(blocking=False):
            log_health.warning("BUILD_PIPELINES_SKIPPED | reason=already_building")
            return False
        try:
            if not getattr(self, "dry_run", False):
                self._sync_symbol_params_from_mt5(self._xau_cfg, "XAU")
            xau_feed = XauMarketFeed(self._xau_cfg, self._xau_cfg.symbol_params)
            xau_features = FeatureEngine(self._xau_cfg)
            xau_risk = RiskManager(self._xau_cfg, self._xau_cfg.symbol_params)
            xau_signal = SignalEngine(
                self._xau_cfg,
                self._xau_cfg.symbol_params,
                xau_feed,
                xau_features,
                xau_risk,
            )
            self._xau = _AssetPipeline(
                "XAU", self._xau_cfg, xau_feed, xau_features, xau_risk, xau_signal
            )
            if not getattr(self, "dry_run", False):
                self._xau.ensure_symbol_selected()

            if not getattr(self, "dry_run", False):
                self._sync_symbol_params_from_mt5(self._btc_cfg, "BTC")
            btc_feed = BtcMarketFeed(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_features = FeatureEngine(self._btc_cfg)
            btc_risk = RiskManager(self._btc_cfg, self._btc_cfg.symbol_params)
            btc_signal = SignalEngine(
                self._btc_cfg,
                self._btc_cfg.symbol_params,
                btc_feed,
                btc_features,
                btc_risk,
            )
            self._btc = _AssetPipeline(
                "BTC", self._btc_cfg, btc_feed, btc_features, btc_risk, btc_signal
            )
            if not getattr(self, "dry_run", False):
                self._btc.ensure_symbol_selected()

            self._refresh_signal_cooldowns()

            log_health.info(
                "PIPELINES_BUILT | xau=%s btc=%s", self._xau.symbol, self._btc.symbol
            )
            return True
        except Exception as exc:
            log_err.error(
                "build pipelines error: %s | tb=%s", exc, traceback.format_exc()
            )
            return False
        finally:
            self._build_pipelines_lock.release()

    def _recover_all(self) -> bool:
        now = time.time()
        if (now - self._last_recover_ts) < 8.0:
            return False
        self._last_recover_ts = now
        try:
            log_health.info("RECOVER_START")

            self._stop_exec_workers(timeout=4.0)

            self._drain_queue(self._order_q)
            self._drain_queue(self._result_q)
            with self._order_state_lock:
                self._order_rm_by_id.clear()
                self._pending_order_meta.clear()

            try:
                with MT5_LOCK:
                    mt5.shutdown()
            except Exception:
                pass
            time.sleep(0.3)

            if not self._init_mt5():
                return False
            if not self._build_pipelines():
                return False
            unsafe_account_reason = self._preflight_live_account_state()
            if unsafe_account_reason:
                with self._lock:
                    self._manual_stop = True
                log_health.warning(
                    "RECOVER_MONITORING_UNSAFE_ACCOUNT_STATE | trading_disabled=True analytics_alive=True reason=%s",
                    unsafe_account_reason,
                )
                self._restart_exec_worker()
                return True

            self._restart_exec_worker()

            log_health.info("RECOVER_OK")
            return True
        except Exception as exc:
            log_err.error("recover error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _refresh_signal_cooldowns(self) -> None:
        self._schedule_manager.refresh_signal_cooldowns()

    @staticmethod
    def _get_phase(risk: Any) -> str:
        return EngineScheduleManager.get_phase(risk)

    @staticmethod
    def _get_daily_date(risk: Any) -> str:
        return EngineScheduleManager.get_daily_date(risk)

    def _phase_reason(self, risk: Any, new_phase: str) -> str:
        return self._schedule_manager.phase_reason(risk, new_phase)

    def _check_phase_change(self, asset: str, risk: Any) -> None:
        self._schedule_manager.check_phase_change(asset, risk)

    def _check_daily_start(self, asset: str, risk: Any) -> None:
        self._schedule_manager.check_daily_start(asset, risk)

    def _cooldown_for_asset(self, asset: str) -> float:
        return self._schedule_manager.cooldown_for_asset(asset)

    def _is_duplicate(
        self,
        asset: str,
        signal_id: str,
        now: float,
        max_orders: int,
        *,
        order_index: int = 0,
    ) -> bool:
        return self._schedule_manager.is_duplicate(
            asset, signal_id, now, max_orders, order_index=order_index
        )

    def _mark_seen(self, asset: str, signal_id: str, now: float) -> None:
        self._schedule_manager.mark_seen(asset, signal_id, now)

    @staticmethod
    def _prune_signal_window(
        window: Deque[float], now_ts: float, lookback_sec: float = 86_400.0
    ) -> None:
        EngineScheduleManager.prune_signal_window(
            window, now_ts, lookback_sec=lookback_sec
        )

    @staticmethod
    def _p95(values: Deque[float] | List[float]) -> float:
        return EngineScheduleManager.p95(values)

    @staticmethod
    def _utc_day_progress() -> float:
        return EngineScheduleManager.utc_day_progress()

    def _signal_density_controls(
        self, asset: str, now_ts: float
    ) -> Tuple[float, float, float, int]:
        return self._schedule_manager.signal_density_controls(asset, now_ts)

    def _record_signal_emit(self, asset: str, now_ts: float) -> None:
        self._schedule_manager.record_signal_emit(asset, now_ts)

    def _select_active_asset(self, open_xau: int, open_btc: int) -> str:
        return EngineScheduleManager.select_active_asset(open_xau, open_btc)


# =============================================================================
# Global Functions / Backward Compatibility
# =============================================================================
def _to_epoch_seconds(x: Any) -> Optional[float]:
    """Robust conversion for pandas/mt5 datetime representations to epoch seconds."""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            return v if v >= 1_000_000_000.0 else None
        if isinstance(x, datetime):
            dt = x
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = float(dt.timestamp())
            return ts if ts >= 1_000_000_000.0 else None
        ts_fn = getattr(x, "timestamp", None)
        if callable(ts_fn):
            ts = float(ts_fn())
            return ts if ts >= 1_000_000_000.0 else None
    except Exception:
        pass
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if v >= 1_000_000_000.0 else None
    except Exception:
        return None


def _apply_high_accuracy_mode(cfg, enable=True):
    pass


def get_xau_config():
    return _get_core_config("XAU")


def get_btc_config():
    return _get_core_config("BTC")


# Aliases
xau_apply_high_accuracy_mode = _apply_high_accuracy_mode
btc_apply_high_accuracy_mode = _apply_high_accuracy_mode
XauFeatureEngine = FeatureEngine
BtcFeatureEngine = FeatureEngine
XauRiskManager = RiskManager
BtcRiskManager = RiskManager
XauSignalEngine = SignalEngine
BtcSignalEngine = SignalEngine

# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "BtcFeatureEngine",
    "BtcRiskManager",
    "BtcSignalEngine",
    "EnginePipelineMixin",
    "EngineScheduleManager",
    "STALE_DATA_THRESHOLD_SEC",
    "UTCScheduler",
    "XauFeatureEngine",
    "XauRiskManager",
    "XauSignalEngine",
    "_AssetPipeline",
    "_apply_high_accuracy_mode",
    "_to_epoch_seconds",
    "btc_apply_high_accuracy_mode",
    "get_btc_config",
    "get_xau_config",
    "xau_apply_high_accuracy_mode",
]
