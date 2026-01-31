from __future__ import annotations

import sys
from pathlib import Path

# Решаи лоиҳа ба sys.path — барои иҷро аз DataFeed
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import MetaTrader5 as mt5
from mt5_client import MT5_LOCK, ensure_mt5


Timeframe = Literal["M1", "M5", "M15"]

TIMEFRAME_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
}

log = logging.getLogger("ai.market_feed")
log.setLevel(logging.ERROR)
log.propagate = False


@dataclass(frozen=True)
class IndicatorPack:
    symbol: str
    ts_bar: int
    tf: Timeframe
    rsi_14: float
    ema_20: float
    ema_50: float
    ema_200: float
    vwap: float
    atr_14: float
    last_close: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class GetDataRealTimeBaseForAi:
    """
    NOTE:
    - Public API (class name + methods) нигоҳ дошта шудааст.
    - Оптимизатсия: кеши symbol_select + кеши meta барои чанд сония,
      то MT5-ро беҳуда “назанад”.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._symbol_ready: bool = False
        self._symbol_ready_ts: float = 0.0

        self._meta_cache: Optional[Dict[str, Any]] = None
        self._meta_cache_ts: float = 0.0

        # TTL-ҳо (сахт, барои скалпинг)
        self._symbol_ttl_sec: float = 5.0
        self._meta_ttl_sec: float = 2.0

    # ---------- MT5 helpers ----------
    def _ensure_symbol(self) -> None:
        ensure_mt5()
        now = time.time()
        if self._symbol_ready and (now - self._symbol_ready_ts) <= self._symbol_ttl_sec:
            return

        with MT5_LOCK:
            ok = mt5.symbol_select(self.symbol, True)
            err = mt5.last_error()

        if not ok:
            raise RuntimeError(f"Symbol {self.symbol} unavailable. MT5: {err}")

        self._symbol_ready = True
        self._symbol_ready_ts = now

    def _symbol_meta(self) -> Dict[str, Any]:
        now = time.time()
        if self._meta_cache and (now - self._meta_cache_ts) <= self._meta_ttl_sec:
            return dict(self._meta_cache)

        self._ensure_symbol()
        with MT5_LOCK:
            info = mt5.symbol_info(self.symbol)
            tick = mt5.symbol_info_tick(self.symbol)

        # PATCH: deterministic fallback meta (never return empty dict)
        if info is None:
            meta_fallback = {
                "symbol": self.symbol,
                "digits": 2,
                "point": 0.01,
                "stops_level_points": 0,
                "spread_points": None,
            }
            self._meta_cache = dict(meta_fallback)
            self._meta_cache_ts = now
            return dict(meta_fallback)

        point = float(getattr(info, "point", 0.0) or 0.0)
        digits = int(getattr(info, "digits", 0) or 0)
        stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)

        spread_points: Optional[int] = None
        if tick is not None and point > 0:
            spread_points = int(round((float(tick.ask) - float(tick.bid)) / point))

        meta = {
            "symbol": self.symbol,
            "digits": digits,
            "point": point,
            "stops_level_points": stops_level_points,
            "spread_points": spread_points,
        }

        self._meta_cache = dict(meta)
        self._meta_cache_ts = now
        return meta

    # ---------- DATA SOURCE ----------
    def fetch_ohlcv(self, tf: Timeframe, limit: int) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Returns CLOSED candles only:
        List[(timestamp, open, high, low, close, volume)] in ascending time order.
        """
        self._ensure_symbol()

        mt5_tf = TIMEFRAME_MAP.get(tf)
        if mt5_tf is None:
            raise ValueError(f"Invalid timeframe: {tf}")

        with MT5_LOCK:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 1, int(limit))
            err = mt5.last_error()

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {self.symbol}: {err}")

        # Usually already chronological; sort is cheap for 260
        rates_sorted = sorted(rates, key=lambda r: int(r["time"]))

        out: List[Tuple[int, float, float, float, float, float]] = []
        for r in rates_sorted:
            out.append(
                (
                    int(r["time"]),
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                    float(r["tick_volume"]),
                )
            )
        return out

    # ---------- INDICATORS ----------
    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        """
        EMA классикӣ: старт = SMA(period), баъд smoothing.
        Ин дақиқтар ва устувортар аз start=values[0].
        """
        n = len(values)
        if n < period:
            raise ValueError(f"Need >= {period} values for EMA({period})")

        k = 2.0 / (period + 1.0)
        sma = sum(values[:period]) / float(period)
        ema = sma
        for v in values[period:]:
            ema = (v * k) + (ema * (1.0 - k))
        return float(ema)

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> float:
        if len(values) < period + 1:
            raise ValueError(f"Need >= {period+1} values for RSI({period})")

        gains = 0.0
        losses = 0.0
        for i in range(1, period + 1):
            diff = values[i] - values[i - 1]
            if diff >= 0:
                gains += diff
            else:
                losses += -diff

        avg_gain = gains / period
        avg_loss = losses / period

        for i in range(period + 1, len(values)):
            diff = values[i] - values[i - 1]
            gain = diff if diff > 0 else 0.0
            loss = -diff if diff < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _vwap(ohlcv: List[Tuple[int, float, float, float, float, float]]) -> float:
        pv_sum = 0.0
        v_sum = 0.0
        for _, _o, h, l, c, v in ohlcv:
            tp = (h + l + c) / 3.0
            pv_sum += tp * v
            v_sum += v
        if v_sum == 0:
            _t, _o, h, l, c, _v = ohlcv[-1]
            return float((h + l + c) / 3.0)
        return float(pv_sum / v_sum)

    @staticmethod
    def _atr(ohlcv: List[Tuple[int, float, float, float, float, float]], period: int = 14) -> float:
        if len(ohlcv) < period + 1:
            raise ValueError(f"Need >= {period+1} candles for ATR({period})")

        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]

        trs: List[float] = []
        for i in range(1, len(ohlcv)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            trs.append(max(hl, hc, lc))

        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return float(atr)

    # ---------- PACK ----------
    def _build_pack(self, tf: Timeframe) -> IndicatorPack:
        # Минимал: 260 нигоҳ медорем (EMA200 + буфери устувор)
        limit = 260
        candles = self.fetch_ohlcv(tf=tf, limit=limit)

        closes = [c[4] for c in candles]
        last_close = float(closes[-1])
        ts_bar = int(candles[-1][0])

        rsi_14 = self._rsi(closes, 14)

        # Слайсҳо калон мемонанд, аммо EMA дақиқтар шуд (SMA start)
        ema_20 = self._ema(closes[-220:], 20)
        ema_50 = self._ema(closes[-250:], 50)
        ema_200 = self._ema(closes[-260:], 200)

        vwap = self._vwap(candles[-120:])
        atr_14 = self._atr(candles[-120:], 14)

        return IndicatorPack(
            symbol=self.symbol,
            ts_bar=ts_bar,
            tf=tf,
            rsi_14=rsi_14,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            vwap=vwap,
            atr_14=atr_14,
            last_close=last_close,
        )

    def return_real_time_data_str(self) -> str:
        """
        AI-friendly, stable format. Uses ONE fetch per timeframe (no duplicate).
        """
        self._ensure_symbol()

        p15 = self._build_pack("M15")
        p5 = self._build_pack("M5")
        p1 = self._build_pack("M1")
        meta = self._symbol_meta()

        ts_now = int(time.time())
        age_sec = ts_now - int(p1.ts_bar)

        def fmt(p: IndicatorPack) -> str:
            return (
                f"{p.tf}: close={p.last_close:.2f} | "
                f"RSI14={p.rsi_14:.2f} | "
                f"EMA20={p.ema_20:.2f}, EMA50={p.ema_50:.2f}, EMA200={p.ema_200:.2f} | "
                f"VWAP={p.vwap:.2f} | ATR14={p.atr_14:.2f} | TS_BAR={p.ts_bar}"
            )

        return (
            f"SYMBOL={self.symbol} | TS_NOW={ts_now} | AGE_SEC={age_sec} | "
            f"SPREAD_PTS={meta.get('spread_points')} | STOPS_LVL_PTS={meta.get('stops_level_points')}\n"
            f"{fmt(p15)}\n{fmt(p5)}\n{fmt(p1)}"
        )


class GetXauDataRealTimeForAi(GetDataRealTimeBaseForAi):
    def __init__(self) -> None:
        super().__init__(symbol="XAUUSDm")


class GetBtcDataRealTimeForAi(GetDataRealTimeBaseForAi):
    def __init__(self) -> None:
        super().__init__(symbol="BTCUSDm")


def create_xau_feed() -> GetXauDataRealTimeForAi:
    return GetXauDataRealTimeForAi()


def create_btc_feed() -> GetBtcDataRealTimeForAi:
    return GetBtcDataRealTimeForAi()


def _payload_from_packs(
    symbol: str,
    p15: IndicatorPack,
    p5: IndicatorPack,
    p1: IndicatorPack,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    ts_now = int(time.time())
    age_sec = ts_now - int(p1.ts_bar)

    summary_text = (
        f"SYMBOL={symbol} | TS_NOW={ts_now} | AGE_SEC={age_sec} | "
        f"SPREAD_PTS={meta.get('spread_points')} | STOPS_LVL_PTS={meta.get('stops_level_points')}\n"
        f"M15: close={p15.last_close:.2f} | RSI14={p15.rsi_14:.2f} | "
        f"EMA20={p15.ema_20:.2f} EMA50={p15.ema_50:.2f} EMA200={p15.ema_200:.2f} | "
        f"VWAP={p15.vwap:.2f} | ATR14={p15.atr_14:.2f} | TS_BAR={p15.ts_bar}\n"
        f"M5:  close={p5.last_close:.2f} | RSI14={p5.rsi_14:.2f} | "
        f"EMA20={p5.ema_20:.2f} EMA50={p5.ema_50:.2f} EMA200={p5.ema_200:.2f} | "
        f"VWAP={p5.vwap:.2f} | ATR14={p5.atr_14:.2f} | TS_BAR={p5.ts_bar}\n"
        f"M1:  close={p1.last_close:.2f} | RSI14={p1.rsi_14:.2f} | "
        f"EMA20={p1.ema_20:.2f} EMA50={p1.ema_50:.2f} EMA200={p1.ema_200:.2f} | "
        f"VWAP={p1.vwap:.2f} | ATR14={p1.atr_14:.2f} | TS_BAR={p1.ts_bar}"
    )

    def tf_dict(p: IndicatorPack) -> Dict[str, Any]:
        return {
            "last_close": round(p.last_close, 6),
            "rsi_14": round(p.rsi_14, 2),
            "ema_20": round(p.ema_20, 6),
            "ema_50": round(p.ema_50, 6),
            "ema_200": round(p.ema_200, 6),
            "vwap": round(p.vwap, 6),
            "atr_14": round(p.atr_14, 6),
            "ts_bar": p.ts_bar,
            "bars": 260,
        }

    return {
        "symbol": symbol,
        "summary_text": summary_text,
        "meta": {
            "ts_now": ts_now,
            "age_sec": age_sec,
            **meta,
        },
        "M15": tf_dict(p15),
        "M5": tf_dict(p5),
        "M1": tf_dict(p1),
    }


def get_ai_payload_xau() -> Optional[Dict[str, Any]]:
    try:
        xau = GetXauDataRealTimeForAi()
        meta = xau._symbol_meta()
        p15 = xau._build_pack("M15")
        p5 = xau._build_pack("M5")
        p1 = xau._build_pack("M1")
        return _payload_from_packs("XAUUSDm", p15, p5, p1, meta)
    except Exception as e:
        log.error("get_ai_payload_xau failed: %s", e)
        return None


def get_ai_payload_btc() -> Optional[Dict[str, Any]]:
    try:
        btc = GetBtcDataRealTimeForAi()
        meta = btc._symbol_meta()
        p15 = btc._build_pack("M15")
        p5 = btc._build_pack("M5")
        p1 = btc._build_pack("M1")
        return _payload_from_packs("BTCUSDm", p15, p5, p1, meta)
    except Exception as e:
        log.error("get_ai_payload_btc failed: %s", e)
        return None
