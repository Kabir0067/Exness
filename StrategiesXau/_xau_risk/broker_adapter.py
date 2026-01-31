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
    Encapsulates MT5 broker interactions:
    - Symbol meta info (digits, point, stops level)
    - Price/Volume normalization
    - Margin & Profit calculations
    - Spread checks
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Meta cache
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
        if time.time() - self._meta_ts < 10.0:
            return True
        
        try:
            ensure_mt5()
            with MT5_LOCK:
                info = mt5.symbol_info(self.symbol)
                if not info:
                    log_risk.error("BrokerAdapter: Symbol info not found for %s", self.symbol)
                    return False
                
                # If not visible, try select
                if not info.visible:
                    if not mt5.symbol_select(self.symbol, True):
                        return False
                    # Refresh info after selection
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
            log_risk.error("BrokerAdapter meta error: %s | tb=%s", exc, traceback.format_exc())
            return False

    def point_value(self) -> float:
        self._refresh_meta()
        return self._point

    def min_stop_distance(self) -> float:
        if not self._refresh_meta():
            return 0.0
        pts = int(max(0, self._stops_level_points)) + int(max(0, self._freeze_level_points)) + 2
        return float(pts) * float(self._point)

    def normalize_price(self, price: float) -> float:
        if not self._refresh_meta():
            return float(price)
        return float(round(float(price), int(self._digits)))

    def normalize_volume_floor(self, vol: float) -> float:
        """Round DOWN to step so risk is never exceeded."""
        if not self._refresh_meta():
            return float(vol)

        step = max(1e-9, float(self._vol_step))
        v = max(float(self._vol_min), min(float(vol), float(self._vol_max)))
        # Floor logic
        v = math.floor(v / step) * step
        return float(round(v, 8))

    def current_spread_points(self) -> float:
        try:
            if not self._refresh_meta():
                return 0.0
            with MT5_LOCK:
                t = mt5.symbol_info_tick(self.symbol)
            if not t:
                return 0.0
            return float((float(t.ask) - float(t.bid)) / max(1e-9, float(self._point)))
        except Exception:
            return 0.0

    def calc_profit_money(self, side: str, volume: float, entry: float, price2: float) -> float:
        try:
            if not self._refresh_meta():
                return 0.0
            order_type = mt5.ORDER_TYPE_BUY if _side_norm(side) == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                val = mt5.order_calc_profit(order_type, self.symbol, float(volume), float(entry), float(price2))
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    def calc_margin(self, side: str, volume: float, entry: float) -> float:
        try:
            if not self._refresh_meta():
                return 0.0
            order_type = mt5.ORDER_TYPE_BUY if _side_norm(side) == "Buy" else mt5.ORDER_TYPE_SELL
            with MT5_LOCK:
                m = mt5.order_calc_margin(order_type, self.symbol, float(volume), float(entry))
            return float(m) if m is not None else 0.0
        except Exception:
            return 0.0

    def tp_usd_to_price(self, entry: float, side: str, volume: float, usd_profit: float) -> Optional[float]:
        try:
            if float(entry) <= 0 or float(volume) <= 0 or float(usd_profit) <= 0:
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
