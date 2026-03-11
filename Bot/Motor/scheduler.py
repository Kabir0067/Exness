from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque, List, Tuple

from .logging_setup import log_health
from .utils import tf_seconds

if TYPE_CHECKING:
    from .engine import MultiAssetTradingEngine


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

    def __init__(self, engine: "MultiAssetTradingEngine") -> None:
        self._e = engine

    def refresh_signal_cooldowns(self) -> None:
        e = self._e
        try:
            if e._signal_cooldown_override_sec is not None and e._signal_cooldown_override_sec > 0:
                cd = float(e._signal_cooldown_override_sec)
                e._signal_cooldown_sec_by_asset["XAU"] = cd
                e._signal_cooldown_sec_by_asset["BTC"] = cd
                return

            def _pipe_tf_sec(pipe: Any) -> float:
                try:
                    if pipe is None:
                        return 60.0
                    tf = getattr(getattr(pipe.cfg, "symbol_params", None), "tf_primary", None)
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
        if int(count) > 0 and int(order_index) <= 0 and (now - float(last_ts)) < min_scale_delay:
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
        while e._seen and (now - e._seen[0][2]) > ttl and cleaned < e._seen_cleanup_budget:
            a, sid, ts = e._seen.popleft()
            rec = e._seen_index.get((a, sid))
            if rec and float(rec[0]) == float(ts):
                e._seen_index.pop((a, sid), None)
            cleaned += 1

    @staticmethod
    def prune_signal_window(window: Deque[float], now_ts: float, lookback_sec: float = 86_400.0) -> None:
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

    def signal_density_controls(self, asset: str, now_ts: float) -> Tuple[float, float, float, int]:
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
        min_target = float(max(1.0, float(e._target_signals_per_24h_min) / float(active_assets)))
        max_target = float(
            max(
                min_target,
                float(e._target_signals_per_24h_max) / float(active_assets),
            )
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
            pace_deficit = (expected_min_now - float(global_count_24h)) / max(min_target_total, 1.0)
            urgency = min(1.0, max(0.0, pace_deficit))
            threshold_mult = max(0.52, float(threshold_mult) * (1.0 - (0.28 * urgency)))
            min_zscore = max(0.12, float(min_zscore) * (1.0 - (0.35 * urgency)))
            min_flow = max(0.005, float(min_flow) * (1.0 - (0.35 * urgency)))
        elif global_count_24h > max_target_total:
            overflow = (float(global_count_24h) - max_target_total) / max(max_target_total, 1.0)
            pressure = min(1.0, max(0.0, overflow))
            threshold_mult = min(1.35, float(threshold_mult) * (1.0 + (0.20 * pressure)))
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
