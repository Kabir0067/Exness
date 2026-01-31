# _btc_risk/broker_adapter.py — MT5 broker interactions for BTC (symbol meta, price/vol, margin/profit, spread)
from __future__ import annotations

import math
import time
import traceback
from typing import Optional

import MetaTrader5 as mt5
import numpy as np

from mt5_client import MT5_LOCK, ensure_mt5
from .logging_ import log_risk
from .utils import _side_norm


class BrokerAdapter:
    """
    Encapsulates MT5 broker interactions for BTC:
    - Symbol meta info (digits, point, stops level)
    - Price/Volume normalization
    - Margin & Profit calculations
    - Spread checks
    - TP from USD profit (robust tick_value inference via order_calc_profit)
    """

    _META_TTL_SEC = 10.0

    def __init__(self, symbol: str):
        self.symbol = symbol

        self._meta_ts: float = 0.0
        self._digits: int = 2
        self._point: float = 0.01
        self._vol_min: float = 0.01
        self._vol_max: float = 100.0
        self._vol_step: float = 0.01
        self._stops_level_points: int = 0
        self._freeze_level_points: int = 0

    def ensure_ready(self) -> bool:
        """Lightweight check if symbol information is accessible."""
        return self._refresh_meta()

    def _refresh_meta(self) -> bool:
        now = time.time()
        if (now - self._meta_ts) < self._META_TTL_SEC:
            return True

        try:
            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                if not info:
                    log_risk.error("BrokerAdapter: symbol_info not found | symbol=%s", self.symbol)
                    return False

                if not bool(getattr(info, "visible", True)):
                    if not mt5.symbol_select(self.symbol, True):
                        log_risk.error("BrokerAdapter: symbol_select failed | symbol=%s", self.symbol)
                        return False
                    info = mt5.symbol_info(self.symbol)
                    if not info:
                        log_risk.error("BrokerAdapter: symbol_info missing after select | symbol=%s", self.symbol)
                        return False

                self._digits = int(getattr(info, "digits", 2) or 2)
                self._point = float(getattr(info, "point", 0.01) or 0.01)
                self._vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
                self._vol_max = float(getattr(info, "volume_max", 100.0) or 100.0)
                self._vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
                self._stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
                self._freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)

            self._meta_ts = now
            return True
        except Exception as exc:
            log_risk.error("BrokerAdapter meta error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def point_value(self) -> float:
        self._refresh_meta()
        return float(self._point)

    def min_stop_distance(self) -> float:
        if not self._refresh_meta():
            return 0.0
        pts = int(max(0, self._stops_level_points)) + int(max(0, self._freeze_level_points)) + 2
        return float(pts) * float(self._point)

    def normalize_price(self, price: float) -> float:
        if not self._refresh_meta():
            return float(price)
        try:
            return float(round(float(price), int(self._digits)))
        except Exception:
            return float(price)

    def normalize_volume_floor(self, vol: float) -> float:
        """Round DOWN to step so risk is never exceeded; aligns to (vol_min + n*step)."""
        if not self._refresh_meta():
            return float(vol)

        try:
            step = max(1e-9, float(self._vol_step))
            vmin = float(self._vol_min)
            vmax = float(self._vol_max)

            v = float(vol)
            if not np.isfinite(v):
                v = vmin
            v = max(vmin, min(v, vmax))

            # align to vmin grid then floor
            n = math.floor((v - vmin) / step)
            v_aligned = vmin + (n * step)

            # safety clamp
            v_aligned = max(vmin, min(v_aligned, vmax))
            return float(round(v_aligned, 8))
        except Exception:
            return float(self._vol_min)

    def current_spread_points(self) -> float:
        """Conservative: return +inf when tick missing so caps block trading."""
        try:
            if not self._refresh_meta():
                return float("inf")
            with MT5_LOCK:
                t = mt5.symbol_info_tick(self.symbol)
            if not t:
                return float("inf")
            bid = float(getattr(t, "bid", 0.0) or 0.0)
            ask = float(getattr(t, "ask", 0.0) or 0.0)
            if bid <= 0 or ask <= 0 or ask < bid:
                return float("inf")
            pt = max(1e-9, float(self._point))
            return float((ask - bid) / pt)
        except Exception:
            return float("inf")

    def calc_profit_money(self, side: str, volume: float, entry: float, price2: float) -> float:
        try:
            if not self._refresh_meta():
                return 0.0
            side_n = _side_norm(side)
            order_type = mt5.ORDER_TYPE_BUY if side_n == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                val = mt5.order_calc_profit(order_type, self.symbol, float(volume), float(entry), float(price2))
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    def calc_margin(self, side: str, volume: float, entry: float) -> float:
        try:
            if not self._refresh_meta():
                return 0.0
            side_n = _side_norm(side)
            order_type = mt5.ORDER_TYPE_BUY if side_n == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                m = mt5.order_calc_margin(order_type, self.symbol, float(volume), float(entry))
            return float(m) if m is not None else 0.0
        except Exception:
            return 0.0

    def _infer_tick_value_money(self, entry: float, tick_size: float) -> float:
        """
        Robust fallback: infer money-per-(tick_size) for 1.0 lot using order_calc_profit.
        This fixes BTCUSDm cases where trade_tick_value returns 0.
        """
        try:
            if tick_size <= 0 or entry <= 0:
                return 0.0
            with MT5_LOCK:
                # BUY move up by tick_size
                v_up = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, self.symbol, 1.0, float(entry), float(entry + tick_size))
                # SELL move down by tick_size
                v_dn = mt5.order_calc_profit(mt5.ORDER_TYPE_SELL, self.symbol, 1.0, float(entry), float(entry - tick_size))
            a = abs(float(v_up) if v_up is not None else 0.0)
            b = abs(float(v_dn) if v_dn is not None else 0.0)
            return float(max(a, b))
        except Exception:
            return 0.0

    def tp_usd_to_price(self, entry: float, side: str, volume: float, usd_profit: float) -> Optional[float]:
        """
        TP price from USD profit.
        Fix: DO NOT use tick_size as tick_value (units mismatch). Infer tick_value via order_calc_profit when needed.
        """
        try:
            entry = float(entry)
            volume = float(volume)
            usd_profit = float(usd_profit)

            if entry <= 0 or volume <= 0 or usd_profit <= 0:
                return None
            if not self._refresh_meta():
                return None

            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
            if not info:
                return None

            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = float(getattr(info, "point", 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = float(self._point)

            tv = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            tvp = float(getattr(info, "trade_tick_value_profit", 0.0) or 0.0)
            tvl = float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0)
            tick_value = float(max(tv, tvp, tvl))

            # critical fallback (BTCUSDm often returns 0)
            if tick_value <= 0.0:
                tick_value = self._infer_tick_value_money(entry, tick_size)

            if tick_value <= 0.0 or tick_size <= 0.0:
                return None

            ticks_needed = usd_profit / (tick_value * volume)
            if not np.isfinite(ticks_needed) or ticks_needed <= 0:
                return None

            price_delta = ticks_needed * tick_size

            side_n = _side_norm(side)
            tp = entry + price_delta if side_n == "Buy" else entry - price_delta

            # broker min stop distance
            min_dist = self.min_stop_distance()
            if min_dist > 0:
                if side_n == "Buy":
                    tp = max(tp, entry + min_dist)
                else:
                    tp = min(tp, entry - min_dist)

            tp = self.normalize_price(tp)

            if tp <= 0:
                return None
            if side_n == "Buy" and not (tp > entry):
                return None
            if side_n == "Sell" and not (tp < entry):
                return None

            return float(tp)
        except Exception:
            return None
