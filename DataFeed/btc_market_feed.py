# btc_feed.py  (BTCUSDm ONLY, 24/7, PRODUCTION-GRADE)
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config_btc import EngineConfig, SymbolParams, TF_MAP
from mt5_client import MT5_LOCK, ensure_mt5
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# =============================================================================
# Logging (ERROR-only) + safe Logs dir
# =============================================================================
LOG_DIR = LOG_ROOT

log_feed = logging.getLogger("btc_feed")
log_feed.setLevel(logging.ERROR)
log_feed.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in log_feed.handlers):
    fh = logging.FileHandler(str(get_log_path("btc_feed.log")), encoding="utf-8", delay=True)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log_feed.addHandler(fh)


# =============================================================================
# Data models
# =============================================================================
@dataclass(slots=True)
class TickStats:
    ok: bool
    reason: str

    # last quote snapshot (for RiskManager.execmon.on_quote)
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    spread_pct: float = 1.0
    spread_ticks: float = 0.0

    # microstructure
    tps: float = 0.0
    flips: int = 0
    volatility: float = 0.0
    imbalance: float = 0.0

    # orderflow-ish
    tick_delta: float = 0.0
    cumulative_delta: float = 0.0
    aggr_delta: float = 0.0

    # micro trend / regime
    micro_trend: float = 0.0
    regime: str = "unknown"


@dataclass(slots=True)
class LatencyStats:
    py_mt5_ms: float = 0.0
    mt5_ping_ms: float = 0.0
    order_queue: int = 0


@dataclass(slots=True)
class MicroZones:
    bid_floor: Optional[float] = None
    ask_ceiling: Optional[float] = None
    mid: Optional[float] = None
    strong_support: Optional[float] = None
    strong_resistance: Optional[float] = None
    poc: Optional[float] = None
    value_area_high: Optional[float] = None
    value_area_low: Optional[float] = None


# =============================================================================
# Market feed (BTC scalping)
# =============================================================================
class MarketFeed:
    """
    BTCUSDm scalping feed:
      - MT5-safe calls via MT5_LOCK
      - ultra-short TTL caching for M1/M5/M15
      - tick microstructure metrics (TPS, flips, imbalance, spread gates)
      - order-book support (if broker/terminal provides market_book)
      - UTC-aware bars + stale-bar protection
    """

    def __init__(self, cfg: EngineConfig, sp: SymbolParams) -> None:
        self.cfg = cfg
        self.sp = sp

        # HARD enforce BTC only at runtime too
        self.symbol = (sp.resolved or sp.base).strip()
        if self.symbol != "BTCUSDm":
            raise RuntimeError(f"btc_feed: symbol must be BTCUSDm only. Got={self.symbol!r}")

        # Internal data lock (never hold MT5_LOCK and _data_lock simultaneously)
        self._data_lock = threading.RLock()

        # Rates cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._ttl: Dict[str, float] = {}
        self._cache_ttl_sec: Dict[str, float] = {
            "M1": 0.05,
            "M5": 0.10,
            "M15": 0.20,
        }
        self._cache_max_keys = int(getattr(self.cfg, "rates_cache_max_keys", 128) or 128)

        # Tick buffer (store last ~2000 ticks)
        self._tick_cache: Dict[str, np.ndarray] = {}
        self._last_tick_sync: float = 0.0
        self._tick_deque: deque = deque(maxlen=2000)

        # Book cache
        self._last_book: Optional[Dict[str, Any]] = None
        self._last_book_ts: float = 0.0
        self._book_subscribed: bool = False

        # CVD-like accumulator
        self._cumulative_delta: float = 0.0
        self._last_tick_msc: int = 0  # dedup ticks + safe CVD accumulation

        # Latency
        self._last_latency: LatencyStats = LatencyStats()

        # Symbol readiness debounce
        self._last_ready_ts: float = 0.0

        # housekeeping
        self._last_cache_gc_ts: float = 0.0

        # timezone cache for now_local
        tz_name = str(getattr(self.cfg, "tz_local", "UTC") or "UTC")
        try:
            self._tz: ZoneInfo = ZoneInfo(tz_name)
        except Exception:
            self._tz = ZoneInfo("UTC")

    # ---------------------------------------------------------------------
    # MT5 guards
    # ---------------------------------------------------------------------
    def _ensure_symbol_ready(self) -> bool:
        """
        Ensures MT5 is initialized/logged-in and symbol is selected/visible.
        Debounced to avoid heavy calls every tick.
        """
        try:
            now = time.time()
            if (now - self._last_ready_ts) < 1.0:
                return True

            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                if info is None:
                    log_feed.error("Symbol not found: %s", self.symbol)
                    return False
                if hasattr(info, "visible") and (not bool(info.visible)):
                    if not mt5.symbol_select(self.symbol, True):
                        log_feed.error("symbol_select failed: %s | last_error=%s", self.symbol, mt5.last_error())
                        return False

            self._last_ready_ts = now
            return True
        except Exception as exc:
            log_feed.error("MT5/symbol prepare error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def _ensure_book_subscribed(self) -> None:
        if self._book_subscribed:
            return
        try:
            with MT5_LOCK:
                ok = mt5.market_book_add(self.symbol)
            self._book_subscribed = bool(ok)
        except Exception:
            self._book_subscribed = False

    # ---------------------------------------------------------------------
    # Cache housekeeping
    # ---------------------------------------------------------------------
    def _gc_caches(self, now: float) -> None:
        if (now - self._last_cache_gc_ts) < 10.0:
            return
        self._last_cache_gc_ts = now

        with self._data_lock:
            expired = [k for k, t_exp in self._ttl.items() if now >= float(t_exp)]
            for k in expired:
                self._ttl.pop(k, None)
                self._cache.pop(k, None)

            if len(self._cache) > self._cache_max_keys:
                keys_sorted = sorted(self._ttl.items(), key=lambda kv: float(kv[1]))
                drop_n = max(0, len(self._cache) - self._cache_max_keys)
                for k, _ in keys_sorted[:drop_n]:
                    self._ttl.pop(k, None)
                    self._cache.pop(k, None)

    # ---------------------------------------------------------------------
    # Rates
    # ---------------------------------------------------------------------
    def _bars_for_tf(self, timeframe: str) -> int:
        need = max(
            300,
            int(getattr(self.cfg, "vol_lookback", 80)),
            int(getattr(self.cfg, "extreme_lookback", 120)),
            int(getattr(self.cfg, "conformal_window", 300)),
            int(getattr(self.cfg, "brier_window", 800)),
            int(getattr(self.cfg, "meta_h_bars", 6)) * 50,
        )
        cap = 2000
        if timeframe == "M1":
            return min(max(need, 700), cap)
        if timeframe == "M5":
            return min(max(int(need * 0.60), 350), cap)
        if timeframe == "M15":
            return min(max(int(need * 0.40), 250), cap)
        return min(max(need, 300), cap)

    def fetch_rates(self, timeframe: str, bars: int = 250) -> Optional[pd.DataFrame]:
        df = self.get_rates(self.symbol, timeframe)
        if df is None:
            return None
        if bars and len(df) > bars:
            return df.iloc[-bars:].copy()
        return df.copy()

    def get_rates(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            if not symbol or timeframe not in TF_MAP:
                return None

            now = time.time()
            self._gc_caches(now)
            key = f"{symbol}_{timeframe}"

            # fast cache
            with self._data_lock:
                ttl = float(self._ttl.get(key, 0.0))
                if now < ttl and key in self._cache:
                    df_cached = self._cache[key]
                    if df_cached is not None and not df_cached.empty:
                        # protect against absurdly stale returns
                        if self.last_bar_age(df_cached) <= max(
                            2.0, float(getattr(self.cfg, "min_bar_age_sec", 1)) + 2.0
                        ):
                            return df_cached

            # refresh path
            if not self._ensure_symbol_ready():
                with self._data_lock:
                    return self._cache.get(key)

            tf_id = TF_MAP[timeframe]
            bars = self._bars_for_tf(timeframe)

            rates = None
            for _ in range(3):
                with MT5_LOCK:
                    rates = mt5.copy_rates_from_pos(symbol, tf_id, 0, bars)
                if rates is not None and len(rates) >= 50:
                    break
                time.sleep(0.05)

            if rates is None or len(rates) == 0:
                with self._data_lock:
                    return self._cache.get(key)

            df = pd.DataFrame(rates)
            required = {"time", "open", "high", "low", "close", "tick_volume"}
            if not required.issubset(set(df.columns)):
                with self._data_lock:
                    return self._cache.get(key)

            # UTC-aware timestamps
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if not df["time"].is_monotonic_increasing:
                df = df.sort_values("time").reset_index(drop=True)

            # sanity: if last bar is very old, prefer cache
            if self.last_bar_age(df) > 120.0:
                with self._data_lock:
                    cached_df = self._cache.get(key)
                    if cached_df is not None:
                        return cached_df
                return df

            ttl_sec = float(
                self._cache_ttl_sec.get(timeframe, max(0.05, float(getattr(self.cfg, "poll_seconds_fast", 0.12))))
            )
            with self._data_lock:
                self._cache[key] = df
                self._ttl[key] = now + ttl_sec

            return df

        except Exception as exc:
            log_feed.error("get_rates error: %s | tb=%s", exc, traceback.format_exc())
            with self._data_lock:
                return self._cache.get(f"{symbol}_{timeframe}")

    def last_bar_age(self, df: pd.DataFrame) -> float:
        try:
            if df is None or df.empty or "time" not in df.columns:
                return 9999.0
            last_t = df.iloc[-1]["time"]
            if isinstance(last_t, pd.Timestamp):
                last_ts = float(last_t.timestamp())
            elif isinstance(last_t, (np.integer, int, float)):
                last_ts = float(last_t)
            else:
                last_ts = float(pd.Timestamp(last_t).timestamp())
            return float(max(0.0, time.time() - last_ts))
        except Exception:
            return 9999.0

    # ---------------------------------------------------------------------
    # Quotes / Spread (BTC-aware)
    # ---------------------------------------------------------------------
    def quote(self) -> Tuple[float, float, float, float, float]:
        """
        Returns: bid, ask, mid, spread, spread_pct
        """
        if not self._ensure_symbol_ready():
            return 0.0, 0.0, 0.0, 0.0, 1.0
        with MT5_LOCK:
            t = mt5.symbol_info_tick(self.symbol)
        if not t:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        bid = float(getattr(t, "bid", 0.0) or 0.0)
        ask = float(getattr(t, "ask", 0.0) or 0.0)
        mid = 0.5 * (bid + ask)
        spr = float(abs(ask - bid))
        spr_pct = float(spr / mid) if mid > 0 else 1.0
        return bid, ask, mid, spr, spr_pct

    def spread_pct(self) -> float:
        """Current spread as percentage of mid-price."""
        try:
            _, _, mid, spread, spread_pct = self.quote()
            if spread_pct > 0.0:
                return float(spread_pct)
            if mid > 0.0:
                return float(spread / mid)
            return 1.0
        except Exception:
            return 1.0

    def spread_ticks(self) -> float:
        """
        Spread in ticks using trade_tick_size if available; else uses point/digits fallback.
        """
        try:
            if not self._ensure_symbol_ready():
                return 0.0
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                tick = mt5.symbol_info_tick(self.symbol)
            if not info or not tick:
                return 0.0
            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = float(getattr(info, "point", 0.0) or 0.0)
            if tick_size <= 0:
                return 0.0
            return float(abs(ask - bid) / tick_size)
        except Exception:
            return 0.0

    # ---------------------------------------------------------------------
    # Order book
    # ---------------------------------------------------------------------
    def fetch_book(self, levels: int = 10) -> Optional[Dict[str, Any]]:
        try:
            if not self._ensure_symbol_ready():
                return None

            self._ensure_book_subscribed()

            with MT5_LOCK:
                book = mt5.market_book_get(self.symbol)

            if book is None:
                with self._data_lock:
                    if self._last_book and (time.time() - self._last_book_ts) < 0.5:
                        return self._last_book
                return None

            bids: List[Tuple[float, float]] = []
            asks: List[Tuple[float, float]] = []
            for lv in book[: levels * 4]:
                if lv.type == mt5.BOOK_TYPE_BUY:
                    bids.append((float(lv.price), float(lv.volume)))
                elif lv.type == mt5.BOOK_TYPE_SELL:
                    asks.append((float(lv.price), float(lv.volume)))

            result = {
                "bids": sorted(bids, key=lambda x: -x[0])[:levels],
                "asks": sorted(asks, key=lambda x: x[0])[:levels],
            }

            with self._data_lock:
                self._last_book = result
                self._last_book_ts = time.time()

            return result

        except Exception as exc:
            log_feed.error("fetch_book error: %s | tb=%s", exc, traceback.format_exc())
            with self._data_lock:
                if self._last_book and (time.time() - self._last_book_ts) < 0.5:
                    return self._last_book
            return None

    # ---------------------------------------------------------------------
    # Ticks
    # ---------------------------------------------------------------------
    def _sync_ticks(self, symbol: str, n_ticks: int = 1600) -> Optional[np.ndarray]:
        """
        Bounded tick sync; debounced by poll_seconds_fast.
        Keeps a rolling deque for micro-window computations.
        """
        now = time.time()
        if (now - self._last_tick_sync) < float(getattr(self.cfg, "poll_seconds_fast", 0.12) or 0.12):
            with self._data_lock:
                return self._tick_cache.get(symbol)

        if not self._ensure_symbol_ready():
            return None

        try:
            # MT5 copies ticks from a datetime; naive is treated as terminal time in many builds.
            since = datetime.utcnow() - timedelta(seconds=int(max(2, self.sp.micro_window_sec) * 3))

            with MT5_LOCK:
                ticks = mt5.copy_ticks_from(symbol, since, int(n_ticks), mt5.COPY_TICKS_ALL)

            if ticks is None or len(ticks) < 2:
                return None

            # time_msc int64 is safe in float64 up to 2^53; we store float64 array for fast vector ops.
            arr = np.column_stack(
                (
                    ticks["time_msc"].astype(np.int64, copy=False),
                    ticks["bid"].astype(np.float64, copy=False),
                    ticks["ask"].astype(np.float64, copy=False),
                    ticks["volume"].astype(np.float64, copy=False),
                    ticks["last"].astype(np.float64, copy=False),
                    ticks["flags"].astype(np.uint32, copy=False),
                )
            ).astype(np.float64, copy=False)

            with self._data_lock:
                self._tick_cache[symbol] = arr
                self._last_tick_sync = time.time()

                # Append ONLY new ticks (dedup overlaps) to keep TPS/volatility real
                last_msc = int(self._last_tick_msc or 0)
                if last_msc > 0:
                    new_arr = arr[arr[:, 0] > float(last_msc)]
                else:
                    new_arr = arr

                if new_arr is not None and len(new_arr) > 0:
                    tail = new_arr[-min(len(new_arr), 700):]
                    for row in tail:
                        self._tick_deque.append(row)

                    # Update cumulative delta from NEW ticks only (prevents double counting)
                    try:
                        BUY_FLAG = int(getattr(mt5, "TICK_FLAG_BUY", 128))
                        SELL_FLAG = int(getattr(mt5, "TICK_FLAG_SELL", 64))
                        flags = new_arr[:, 5].astype(np.uint32, copy=False)
                        vols = new_arr[:, 3].astype(np.float64, copy=False)
                        buy_vol = float(np.sum(vols[(flags & BUY_FLAG) != 0])) if BUY_FLAG else 0.0
                        sell_vol = float(np.sum(vols[(flags & SELL_FLAG) != 0])) if SELL_FLAG else 0.0
                        self._cumulative_delta += (buy_vol - sell_vol)
                    except Exception:
                        pass

                    # advance watermark
                    try:
                        self._last_tick_msc = max(int(self._last_tick_msc or 0), int(new_arr[-1, 0]))
                    except Exception:
                        self._last_tick_msc = int(self._last_tick_msc or 0)

            return arr
        except Exception as exc:
            log_feed.error("_sync_ticks error: %s | tb=%s", exc, traceback.format_exc())
            return None

    def tick_stats(self, df_m1: Optional[pd.DataFrame] = None) -> TickStats:
        """
        BTC scalping gate:
          - Liquidity: TPS within [micro_min_tps .. micro_max_tps]
          - Noise: quote flips <= quote_flips_max
          - Spread: soft cap spread_limit_pct, hard breaker spread_cb_pct
          - Optional regime estimation using ATR ratio + BB width
        """
        try:
            ignore_micro = bool(getattr(self.cfg, "ignore_microstructure", False))
            self._sync_ticks(self.symbol, n_ticks=2000)
            book = self.fetch_book(levels=20)

            def _fallback_tick(reason: str) -> TickStats:
                bid = ask = 0.0
                try:
                    with MT5_LOCK:
                        t = mt5.symbol_info_tick(self.symbol)
                    if t:
                        bid = float(getattr(t, "bid", 0.0) or 0.0)
                        ask = float(getattr(t, "ask", 0.0) or 0.0)
                except Exception:
                    pass
                mid = 0.5 * (bid + ask) if bid and ask else 0.0
                spread = float(abs(ask - bid)) if bid and ask else 0.0
                spread_pct = float(spread / mid) if mid > 0 else 1.0
                return TickStats(
                    ok=True,
                    reason=reason,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread=spread,
                    spread_pct=spread_pct,
                    spread_ticks=self.spread_ticks(),
                    tps=0.0,
                    flips=0,
                    volatility=0.0,
                    imbalance=0.0,
                    tick_delta=0.0,
                    cumulative_delta=0.0,
                    aggr_delta=0.0,
                    micro_trend=0.0,
                    regime="unknown",
                )

            with self._data_lock:
                min_ticks = int(max(12, float(self.sp.micro_min_tps) * float(self.sp.micro_window_sec) * 2.0))
                if len(self._tick_deque) < min_ticks:
                    if ignore_micro:
                        return _fallback_tick(f"low_ticks_buf_ignored:{len(self._tick_deque)}/{min_ticks}")
                    return TickStats(ok=False, reason=f"low_ticks_buf:{len(self._tick_deque)}/{min_ticks}")
                rows = list(self._tick_deque)

            arr = np.asarray(rows, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 6:
                return TickStats(ok=False, reason="bad_tick_shape")

            now_ms = float(time.time() * 1000.0)
            win_ms = float(max(1, int(self.sp.micro_window_sec)) * 1000.0)

            w = arr[arr[:, 0] >= (now_ms - win_ms)]
            if w.shape[0] < min_ticks:
                if ignore_micro:
                    return _fallback_tick(f"low_ticks_ignored:{int(w.shape[0])}/{min_ticks}")
                return TickStats(ok=False, reason=f"low_ticks:{int(w.shape[0])}/{min_ticks}")

            times = w[:, 0] / 1000.0
            bids = w[:, 1]
            asks = w[:, 2]
            vols = w[:, 3]
            flags = w[:, 5].astype(np.uint32, copy=False)

            last_bid = float(bids[-1]) if bids.size else 0.0
            last_ask = float(asks[-1]) if asks.size else 0.0
            mid = 0.5 * (last_bid + last_ask)
            spread = float(abs(last_ask - last_bid))
            spread_pct = float(spread / mid) if mid > 0 else 1.0

            # TPS
            horizon = max(1e-6, float(times[-1] - times[0]))
            tps = float(w.shape[0] / horizon)

            # Volatility on mids
            mids = (bids + asks) / 2.0
            mid_diff = np.diff(mids)
            volatility = float(np.std(mid_diff)) if mid_diff.size > 1 else 0.0

            # Quote flips (direction changes of mid diff)
            if mid_diff.size > 2:
                s = np.sign(mid_diff)
                s = s[s != 0]
                flips = int(np.sum(s[:-1] != s[1:])) if s.size > 2 else 0
            else:
                flips = 0

            # Delta (best-effort via MT5 tick flags if present)
            BUY_FLAG = int(getattr(mt5, "TICK_FLAG_BUY", 128))
            SELL_FLAG = int(getattr(mt5, "TICK_FLAG_SELL", 64))

            buy_mask = (flags & BUY_FLAG) != 0
            sell_mask = (flags & SELL_FLAG) != 0

            buy_vol = float(np.sum(vols[buy_mask])) if bool(buy_mask.any()) else 0.0
            sell_vol = float(np.sum(vols[sell_mask])) if bool(sell_mask.any()) else 0.0
            total_vol = max(1e-6, buy_vol + sell_vol)

            tick_delta = float((buy_vol - sell_vol) / total_vol)
            aggr_delta = tick_delta

            with self._data_lock:
                cumulative_delta = float(self._cumulative_delta)

            # Book imbalance
            imb = 0.0
            if book and book.get("bids") and book.get("asks"):
                bid_vols = np.asarray([v for _, v in book["bids"]], dtype=np.float64)
                ask_vols = np.asarray([v for _, v in book["asks"]], dtype=np.float64)
                total_book = max(1e-6, float(np.sum(bid_vols) + np.sum(ask_vols)))
                imb = float((np.sum(bid_vols) - np.sum(ask_vols)) / total_book)

            # Micro-trend as t-stat-like ratio
            std = float(np.std(mid_diff)) if mid_diff.size > 1 else 0.0
            micro_trend = float(np.mean(mid_diff) / (std / np.sqrt(max(1, mid_diff.size)))) if std > 0 else 0.0

            # Regime (BTC): range / breakout / trend
            regime = "unknown"
            if df_m1 is not None and len(df_m1) >= 80 and {"close", "high", "low"}.issubset(df_m1.columns):
                c = df_m1["close"].to_numpy(dtype=np.float64, copy=False)
                h = df_m1["high"].to_numpy(dtype=np.float64, copy=False)
                l = df_m1["low"].to_numpy(dtype=np.float64, copy=False)

                c50 = c[-50:]
                h50 = h[-50:]
                l50 = l[-50:]

                atr_period = int(getattr(self.cfg, "atr_period", 14) or 14)
                atr_rel_hi = float(getattr(self.cfg, "atr_rel_hi", 0.0100) or 0.0100)
                bb_width_range_max = float(getattr(self.cfg, "bb_width_range_max", 0.0080) or 0.0080)

                prev_c = np.roll(c50, 1)
                tr = np.maximum(h50 - l50, np.maximum(np.abs(h50 - prev_c), np.abs(l50 - prev_c)))[1:]
                alpha = 2.0 / (float(atr_period) + 1.0)
                atr = np.zeros_like(tr)
                atr[0] = tr[0]
                for i in range(1, tr.size):
                    atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
                atr_val = float(atr[-1])
                atr_ratio = atr_val / float(c50[-1]) if float(c50[-1]) > 0 else 0.0

                bb_mid = float(np.mean(c50[-20:]))
                bb_std = float(np.std(c50[-20:]))
                bb_width = float((4.0 * bb_std / bb_mid) if bb_mid > 0 else 0.0)

                if bb_width < bb_width_range_max:
                    regime = "range"
                elif atr_ratio > atr_rel_hi:
                    regime = "breakout"
                elif abs(micro_trend) >= float(self.sp.micro_tstat_thresh):
                    regime = "trend_up" if micro_trend > 0 else "trend_down"
                else:
                    regime = "range"
            else:
                # fallback: use micro_trend only
                if abs(micro_trend) >= float(self.sp.micro_tstat_thresh):
                    regime = "trend_up" if micro_trend > 0 else "trend_down"
                else:
                    regime = "range"

            # -----------------------------------------------------------------
            # Guards (BTC scalping rules)
            # -----------------------------------------------------------------
            ok = True
            reason = "ok"

            if tps < float(self.sp.micro_min_tps):
                ok = False
                reason = "low_liquidity"

            if tps > float(self.sp.micro_max_tps):
                ok = False
                reason = "tps_spike"

            # Spread gates
            spread_limit_pct = float(getattr(self.sp, "spread_limit_pct", 0.0050) or 0.0050)
            spread_cb_pct = float(getattr(self.cfg, "spread_cb_pct", 0.0075) or 0.0075)

            if spread_pct > spread_cb_pct:
                ok = False
                reason = "spread_cb"
            elif spread_pct > spread_limit_pct:
                ok = False
                reason = "spread_limit"

            if flips > int(self.sp.quote_flips_max):
                ok = False
                reason = "noise_flips"

            return TickStats(
                ok=ok,
                reason=reason,
                bid=last_bid,
                ask=last_ask,
                mid=mid,
                spread=spread,
                spread_pct=spread_pct,
                spread_ticks=self.spread_ticks(),
                tps=tps,
                flips=flips,
                volatility=volatility,
                imbalance=imb,
                tick_delta=tick_delta,
                cumulative_delta=cumulative_delta,
                aggr_delta=aggr_delta,
                micro_trend=micro_trend,
                regime=regime,
            )
        except Exception as exc:
            log_feed.error("tick_stats error: %s | tb=%s", exc, traceback.format_exc())
            return TickStats(ok=False, reason="exception")

    # ---------------------------------------------------------------------
    # Latency
    # ---------------------------------------------------------------------
    def latency_stats(self) -> LatencyStats:
        try:
            if not self._ensure_symbol_ready():
                return LatencyStats()

            t0 = time.time()
            with MT5_LOCK:
                term = mt5.terminal_info()
            py_mt5_ms = (time.time() - t0) * 1000.0

            # ping_last sometimes is in microseconds or milliseconds depending on build;
            # safest: treat as microseconds if it's large, else ms. We normalize to ms.
            ping_last = float(getattr(term, "ping_last", 0) or 0.0) if term else 0.0
            mt5_ping_ms = (ping_last / 1000.0) if ping_last > 10_000 else ping_last

            with MT5_LOCK:
                orders = mt5.orders_get()
            queue = int(len(orders) if orders else 0)

            self._last_latency = LatencyStats(
                py_mt5_ms=float(py_mt5_ms),
                mt5_ping_ms=float(mt5_ping_ms),
                order_queue=queue,
            )
            return self._last_latency
        except Exception as exc:
            log_feed.error("latency_stats error: %s | tb=%s", exc, traceback.format_exc())
            return LatencyStats()

    # ---------------------------------------------------------------------
    # Micro price zones (from book)
    # ---------------------------------------------------------------------
    def micro_price_zones(self, book: Optional[Dict[str, Any]] = None) -> MicroZones:
        try:
            if book is None:
                book = self.fetch_book(levels=50)
            if not book or not book.get("bids") or not book.get("asks"):
                return MicroZones()

            bid_prices = np.asarray([p for p, _ in book["bids"]], dtype=np.float64)
            bid_vols = np.asarray([v for _, v in book["bids"]], dtype=np.float64)
            ask_prices = np.asarray([p for p, _ in book["asks"]], dtype=np.float64)
            ask_vols = np.asarray([v for _, v in book["asks"]], dtype=np.float64)

            strong_support = float(np.average(bid_prices, weights=bid_vols)) if float(bid_vols.sum()) > 0 else None
            strong_resistance = float(np.average(ask_prices, weights=ask_vols)) if float(ask_vols.sum()) > 0 else None
            mid = float(0.5 * (bid_prices[0] + ask_prices[0]))

            all_prices = np.concatenate((bid_prices, ask_prices))
            all_vols = np.concatenate((bid_vols, ask_vols))

            idx = np.argsort(all_prices)
            p_sorted = all_prices[idx]
            v_sorted = all_vols[idx]

            uniq_p, start_idx = np.unique(p_sorted, return_index=True)
            vp = np.add.reduceat(v_sorted, start_idx)

            poc_i = int(np.argmax(vp))
            poc = float(uniq_p[poc_i])

            total = float(np.sum(vp))
            target = 0.70 * total

            left = poc_i
            right = poc_i
            cum = float(vp[poc_i])

            while cum < target and (left > 0 or right < len(vp) - 1):
                next_left = float(vp[left - 1]) if left > 0 else -1.0
                next_right = float(vp[right + 1]) if right < len(vp) - 1 else -1.0

                if next_right >= next_left:
                    if right < len(vp) - 1:
                        right += 1
                        cum += float(vp[right])
                    else:
                        left -= 1
                        cum += float(vp[left])
                else:
                    if left > 0:
                        left -= 1
                        cum += float(vp[left])
                    else:
                        right += 1
                        cum += float(vp[right])

            return MicroZones(
                bid_floor=float(bid_prices[-1]) if bid_prices.size else None,
                ask_ceiling=float(ask_prices[-1]) if ask_prices.size else None,
                mid=mid,
                strong_support=strong_support,
                strong_resistance=strong_resistance,
                poc=poc,
                value_area_high=float(uniq_p[right]),
                value_area_low=float(uniq_p[left]),
            )
        except Exception as exc:
            log_feed.error("micro_price_zones error: %s | tb=%s", exc, traceback.format_exc())
            return MicroZones()


    def now_local(self) -> datetime:
        """Timezone-aware local timestamp for signal/risk components."""
        try:
            return datetime.now(tz=self._tz)
        except Exception:
            return datetime.now()


    @property
    def tz(self) -> ZoneInfo:
        """Public accessor for configured timezone."""
        return self._tz


__all__ = ["MarketFeed", "TickStats", "LatencyStats", "MicroZones"]
