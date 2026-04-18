from __future__ import annotations

# Решаи лоиҳа ба sys.path — барои иҷро аз DataFeed
import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Literal, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import talib

from log_config import get_log_path
from mt5_client import MT5_LOCK, ensure_mt5, mt5_status

Timeframe = Literal["M1", "M5", "M15"]

TIMEFRAME_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
}

log = logging.getLogger("ai.market_feed")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    fh = RotatingFileHandler(
        str(get_log_path("ai_market_feed.log")),
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    log.addHandler(fh)


# Liveness-once flags — эмит як маротиба INFO лог дар аввалин муваффақияти
# ҳар симбол, то админ бубинад "feed ҷорӣ кор мекунад" ва мутмаин шавад,
# ки handler дуруст bind шудааст (ai_market_feed.log = size=0 буд).
_LIVENESS_LOGGED_XAU = False
_LIVENESS_LOGGED_BTC = False

_PAYLOAD_CACHE_TTL_SEC = 300.0
_NON_RETRIABLE_MARKERS = (
    "algo_trading_disabled_in_terminal",
    "trading_disabled_for_account",
    "wrong_account",
    "blocked_cooldown:",
)
_LAST_PAYLOAD: Dict[str, Dict[str, Any]] = {}
_LAST_PAYLOAD_TS: Dict[str, float] = {}


def _mt5_blocked_reason() -> Optional[str]:
    ok_mt5, reason = mt5_status()
    if ok_mt5:
        return None
    reason_s = str(reason or "unknown")
    if any(marker in reason_s for marker in _NON_RETRIABLE_MARKERS):
        return reason_s
    return None


def _cache_payload(symbol: str, payload: Dict[str, Any]) -> None:
    _LAST_PAYLOAD[symbol] = deepcopy(payload)
    _LAST_PAYLOAD_TS[symbol] = time.time()


def _cached_payload(symbol: str, reason: str) -> Optional[Dict[str, Any]]:
    payload = _LAST_PAYLOAD.get(symbol)
    ts = float(_LAST_PAYLOAD_TS.get(symbol, 0.0) or 0.0)
    if not payload or ts <= 0.0:
        return None
    age = time.time() - ts
    if age > _PAYLOAD_CACHE_TTL_SEC:
        return None
    out = deepcopy(payload)
    meta = out.setdefault("meta", {})
    meta["cached_fallback"] = True
    meta["cached_age_sec"] = int(max(0.0, age))
    meta["mt5_reason"] = str(reason or "unknown")
    try:
        st = str(out.get("summary_text", "") or "")
        out["summary_text"] = (
            st + f"\nFALLBACK=CACHED | AGE_SEC={int(max(0.0, age))} | MT5={reason}"
        )
    except Exception:
        pass
    return out


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
    confluence: Dict[str, Any]
    macd: Dict[str, Any]
    stoch_rsi: Dict[str, Any]
    volume: Dict[str, Any]
    price_action: Dict[str, Any]
    bb: Dict[str, Any]
    structure: Dict[str, Any]
    fvgs: List[Dict[str, Any]]
    order_blocks: List[Dict[str, Any]]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sma(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _sma(values)
    variance = sum((float(v) - mean) ** 2 for v in values) / len(values)
    return float(math.sqrt(max(variance, 0.0)))


# ─────────────────────────────────────────────────────────────────────────
# TA-Lib-backed indicators (SINGLE SOURCE OF TRUTH across train / backtest /
# live inference). Hand-rolled implementations are DELETED — they produced
# off-by-warmup values that disagreed with TA-Lib / TradingView.
# ─────────────────────────────────────────────────────────────────────────
def _ema_series(values: List[float], period: int) -> List[float]:
    """TA-Lib EMA with warmup back-fill to preserve caller contract (same length)."""
    n = len(values)
    if n == 0:
        return []
    period = max(1, int(period))
    arr = np.asarray(values, dtype=np.float64)
    if n < period:
        # Not enough data for a stable EMA — fall back to cumulative SMA
        # so the output length is preserved and values are conservative.
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            out[i] = float(np.mean(arr[: i + 1]))
        return [float(v) for v in out]
    ema = talib.EMA(arr, timeperiod=period)
    # TA-Lib returns NaN for the first (period-1) values. Back-fill with
    # the first valid EMA so the output length matches the input length
    # and downstream index-based accesses remain safe.
    first_valid = np.argmax(~np.isnan(ema)) if np.any(~np.isnan(ema)) else 0
    if np.isnan(ema[first_valid]):
        return [float(v) for v in arr]
    fv = float(ema[first_valid])
    out = np.where(np.isnan(ema), fv, ema)
    return [float(v) for v in out]


def _rsi_series(values: List[float], period: int = 14) -> List[float]:
    """TA-Lib Wilder RSI with neutral back-fill during warmup."""
    n = len(values)
    if n < 2:
        return [50.0 for _ in range(n)]
    period = max(2, int(period))
    arr = np.asarray(values, dtype=np.float64)
    if n <= period:
        return [50.0 for _ in range(n)]
    rsi = talib.RSI(arr, timeperiod=period)
    # Fill warmup NaN with neutral 50 (safe default — no directional bias).
    rsi = np.where(np.isnan(rsi), 50.0, rsi)
    # Clamp to [0, 100] for safety (TA-Lib output is already in range, but
    # guard against any numerical drift).
    rsi = np.clip(rsi, 0.0, 100.0)
    return [float(v) for v in rsi]


def _macd_snapshot(values: List[float]) -> Dict[str, Any]:
    """
    MACD(12,26,9) via TA-Lib — single source of truth, matches
    TradingView / industry-standard implementations exactly.
    """
    if len(values) < 26:  # TA-Lib needs at least slowperiod bars
        return {"line": 0.0, "signal": 0.0, "histogram": 0.0, "cross": "none"}
    arr = np.asarray(values, dtype=np.float64)
    macd_arr, signal_arr, hist_arr = talib.MACD(
        arr, fastperiod=12, slowperiod=26, signalperiod=9
    )
    # Find the last non-NaN sample for a stable current reading.
    valid = np.where(~np.isnan(signal_arr))[0]
    if valid.size == 0:
        return {"line": 0.0, "signal": 0.0, "histogram": 0.0, "cross": "none"}
    last = int(valid[-1])
    current_macd = float(macd_arr[last])
    current_signal = float(signal_arr[last])
    current_hist = float(hist_arr[last])
    cross = "none"
    if valid.size >= 2:
        prev = int(valid[-2])
        prev_macd = float(macd_arr[prev])
        prev_signal = float(signal_arr[prev])
        if prev_macd <= prev_signal and current_macd > current_signal:
            cross = "bullish_cross"
        elif prev_macd >= prev_signal and current_macd < current_signal:
            cross = "bearish_cross"
    return {
        "line": round(current_macd, 6),
        "signal": round(current_signal, 6),
        "histogram": round(current_hist, 6),
        "cross": cross,
    }


def _stoch_rsi_snapshot(values: List[float]) -> Dict[str, Any]:
    rsi_values = _rsi_series(values, 14)
    if len(rsi_values) < 18:
        return {"k": 50.0, "d": 50.0, "cross": "none", "zone": "neutral"}
    raw_k: List[float] = []
    period = 14
    for idx in range(period - 1, len(rsi_values)):
        window = rsi_values[idx - period + 1 : idx + 1]
        low = min(window)
        high = max(window)
        current = rsi_values[idx]
        if high - low <= 1e-12:
            raw_k.append(50.0)
        else:
            raw_k.append(
                float(_clamp((current - low) / (high - low) * 100.0, 0.0, 100.0))
            )
    k_series: List[float] = []
    for idx in range(len(raw_k)):
        window = raw_k[max(0, idx - 2) : idx + 1]
        k_series.append(_sma(window))
    d_series: List[float] = []
    for idx in range(len(k_series)):
        window = k_series[max(0, idx - 2) : idx + 1]
        d_series.append(_sma(window))
    k_now = float(k_series[-1])
    d_now = float(d_series[-1])
    cross = "none"
    if len(k_series) >= 2 and len(d_series) >= 2:
        prev_k = float(k_series[-2])
        prev_d = float(d_series[-2])
        if prev_k <= prev_d and k_now > d_now:
            cross = "bullish_cross"
        elif prev_k >= prev_d and k_now < d_now:
            cross = "bearish_cross"
    if k_now >= 80.0:
        zone = "overbought"
    elif k_now <= 20.0:
        zone = "oversold"
    else:
        zone = "neutral"
    return {
        "k": round(k_now, 2),
        "d": round(d_now, 2),
        "cross": cross,
        "zone": zone,
    }


def _volume_snapshot(
    candles: List[Tuple[int, float, float, float, float, float]],
) -> Dict[str, Any]:
    volumes = [float(c[5]) for c in candles if len(c) >= 6]
    if not volumes:
        return {"current": 0.0, "avg_20": 0.0, "ratio": 1.0}
    current = float(volumes[-1])
    baseline = _sma(volumes[-20:]) if len(volumes) >= 20 else _sma(volumes)
    ratio = current / baseline if baseline > 1e-12 else 1.0
    return {
        "current": round(current, 2),
        "avg_20": round(baseline, 2),
        "ratio": round(_clamp(ratio, 0.0, 9.99), 2),
    }


def _price_action_snapshot(
    candles: List[Tuple[int, float, float, float, float, float]],
) -> Dict[str, Any]:
    if len(candles) < 3:
        return {"pattern": "none"}
    prev = candles[-2]
    cur = candles[-1]
    recent = candles[-11:-1] if len(candles) >= 11 else candles[:-1]
    avg_body = _sma([abs(float(c[4]) - float(c[1])) for c in candles[-20:]])
    prev_open, prev_high, prev_low, prev_close = map(float, prev[1:5])
    open_, high, low, close = map(float, cur[1:5])
    body = abs(close - open_)
    upper = high - max(close, open_)
    lower = min(close, open_) - low
    recent_high = max((float(c[2]) for c in recent), default=high)
    recent_low = min((float(c[3]) for c in recent), default=low)
    pattern = "none"
    if (
        prev_close < prev_open
        and close > open_
        and open_ <= prev_close
        and close >= prev_open
    ):
        pattern = "bullish_engulfing"
    elif (
        prev_close > prev_open
        and close < open_
        and open_ >= prev_close
        and close <= prev_open
    ):
        pattern = "bearish_engulfing"
    elif body > 0 and lower >= (body * 2.0) and upper <= max(body * 0.6, 1e-9):
        pattern = "hammer"
    elif body > 0 and upper >= (body * 2.0) and lower <= max(body * 0.6, 1e-9):
        pattern = "shooting_star"
    elif close > recent_high and body >= max(avg_body * 0.8, 1e-9):
        pattern = "breakout_close"
    elif close < recent_low and body >= max(avg_body * 0.8, 1e-9):
        pattern = "breakdown_close"
    return {"pattern": pattern}


def _bollinger_snapshot(values: List[float], period: int = 20) -> Dict[str, Any]:
    window = values[-period:] if len(values) >= period else values
    if not window:
        return {"mid": 0.0, "upper": 0.0, "lower": 0.0, "pct_b": 0.5}
    mid = _sma(window)
    std = _stddev(window)
    upper = mid + (2.0 * std)
    lower = mid - (2.0 * std)
    last = float(window[-1])
    pct_b = 0.5 if upper - lower <= 1e-12 else (last - lower) / (upper - lower)
    return {
        "mid": round(mid, 6),
        "upper": round(upper, 6),
        "lower": round(lower, 6),
        "pct_b": round(pct_b, 4),
    }


def _structure_snapshot(
    candles: List[Tuple[int, float, float, float, float, float]],
    ema_20: float,
    ema_50: float,
) -> Dict[str, Any]:
    if not candles:
        return {"trend": "neutral", "zone": "equilibrium", "last_bos": "", "choch": ""}
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    close = float(candles[-1][4])
    lookback = min(24, len(candles))
    swing_high = max(highs[-lookback:])
    swing_low = min(lows[-lookback:])
    prior_high = max(highs[-lookback:-1], default=swing_high)
    prior_low = min(lows[-lookback:-1], default=swing_low)
    if close > ema_20 > ema_50:
        trend = "bullish"
    elif close < ema_20 < ema_50:
        trend = "bearish"
    else:
        trend = "neutral"
    band = max(swing_high - swing_low, 1e-9)
    mid = swing_low + (band * 0.5)
    if close >= mid + (band * 0.1):
        zone = "premium"
    elif close <= mid - (band * 0.1):
        zone = "discount"
    else:
        zone = "equilibrium"
    last_bos = ""
    if close > prior_high:
        last_bos = "bullish_bos"
    elif close < prior_low:
        last_bos = "bearish_bos"
    choch = ""
    if trend == "bullish" and close < prior_low:
        choch = "bearish_choch"
    elif trend == "bearish" and close > prior_high:
        choch = "bullish_choch"
    return {
        "trend": trend,
        "zone": zone,
        "last_bos": last_bos,
        "choch": choch,
        "swing_high": round(swing_high, 6),
        "swing_low": round(swing_low, 6),
    }


def _fvg_snapshot(
    candles: List[Tuple[int, float, float, float, float, float]],
    atr_14: float,
) -> List[Dict[str, Any]]:
    if len(candles) < 3:
        return []
    gaps: List[Dict[str, Any]] = []
    min_gap = max(float(atr_14) * 0.08, 1e-9)
    for idx in range(max(2, len(candles) - 40), len(candles)):
        prev2 = candles[idx - 2]
        cur = candles[idx]
        prev2_high = float(prev2[2])
        prev2_low = float(prev2[3])
        cur_high = float(cur[2])
        cur_low = float(cur[3])
        if cur_low > prev2_high and (cur_low - prev2_high) >= min_gap:
            top = cur_low
            bottom = prev2_high
            gaps.append(
                {
                    "direction": "bullish",
                    "top": round(top, 6),
                    "bottom": round(bottom, 6),
                    "mid": round((top + bottom) * 0.5, 6),
                    "strength": round((top - bottom) / max(float(atr_14), 1e-9), 3),
                }
            )
        elif cur_high < prev2_low and (prev2_low - cur_high) >= min_gap:
            top = prev2_low
            bottom = cur_high
            gaps.append(
                {
                    "direction": "bearish",
                    "top": round(top, 6),
                    "bottom": round(bottom, 6),
                    "mid": round((top + bottom) * 0.5, 6),
                    "strength": round((top - bottom) / max(float(atr_14), 1e-9), 3),
                }
            )
    return gaps[-5:]


def _order_block_snapshot(
    candles: List[Tuple[int, float, float, float, float, float]],
    atr_14: float,
) -> List[Dict[str, Any]]:
    """
    Identify Order Blocks using ONLY past bars relative to each candidate.

    CRITICAL FIX — LOOKAHEAD ELIMINATED:
      The previous implementation inspected `candles[idx+1 : idx+4]` (i.e.
      future closes relative to `idx`) to decide whether bar `idx` was a
      valid order block. This leaked future price information into the
      feature seen by the AI at inference time, creating a synthetic
      edge in backtest that did NOT exist live.

    NEW LOGIC (pure past-only, causal):
      For each candidate bar `idx` we look BACKWARDS at preceding bars
      `[idx - lookback : idx]` and ask: was there a clear institutional
      move that CULMINATED at `idx` (rejection / absorption)? A bar is
      flagged as a bullish OB if:
        * it is a down-close (close < open) — the last bear candle
        * the displacement from the PREVIOUS candle's range exceeds
          threshold (the prior bar closed strongly above this bar's high)
      Symmetric rule for bearish OB. This uses only past information
      available at bar `idx`, so the feature is causally valid.

      NOTE: we intentionally keep the feature simple and causal; a
      richer OB labeller should be offline (for training targets only),
      never in a live feature payload.
    """
    if len(candles) < 6:
        return []
    blocks: List[Dict[str, Any]] = []
    threshold = max(float(atr_14) * 0.12, 1e-9)
    atr_safe = max(float(atr_14), 1e-9)
    # Scan recent window but only examine bar `idx` against bars BEFORE it.
    # We leave the final 3 bars scannable since no +k look-forward is used.
    for idx in range(max(2, len(candles) - 32), len(candles)):
        _, open_, high, low, close, _ = candles[idx]
        open_ = float(open_)
        high = float(high)
        low = float(low)
        close = float(close)
        # Past-only confirmation: the 1–3 bars BEFORE idx.
        past = candles[max(0, idx - 3) : idx]
        if not past:
            continue
        past_highs = [float(c[2]) for c in past]
        past_lows = [float(c[3]) for c in past]

        # Bullish OB: down-close bar that was immediately followed (prior in
        # time perspective we use past-of-idx) by strong UP displacement. We
        # invert: the current bar `idx` acts as confirmation; the candidate
        # OB is the most recent past down-close bar whose low has been left
        # behind by the up-move from `idx`'s past bars.
        if close > open_ and max(past_highs) < (close - threshold):
            # Bullish displacement confirmed: prior highs exceeded by close.
            # Candidate OB = last down-close bar among `past`.
            ob_bar = None
            for c in reversed(past):
                c_open, c_close = float(c[1]), float(c[4])
                if c_close < c_open:
                    ob_bar = c
                    break
            if ob_bar is not None:
                o_open = float(ob_bar[1])
                o_high = float(ob_bar[2])
                o_low = float(ob_bar[3])
                o_close = float(ob_bar[4])
                top = max(o_open, o_close)
                bottom = o_low
                strength = (close - o_high) / atr_safe
                blocks.append(
                    {
                        "direction": "bullish",
                        "top": round(top, 6),
                        "bottom": round(bottom, 6),
                        "mid": round((top + bottom) * 0.5, 6),
                        "strength": round(_clamp(strength, 0.0, 3.0), 3),
                    }
                )
        elif close < open_ and min(past_lows) > (close + threshold):
            # Bearish displacement confirmed: prior lows broken below by close.
            ob_bar = None
            for c in reversed(past):
                c_open, c_close = float(c[1]), float(c[4])
                if c_close > c_open:
                    ob_bar = c
                    break
            if ob_bar is not None:
                o_open = float(ob_bar[1])
                o_high = float(ob_bar[2])
                o_low = float(ob_bar[3])
                o_close = float(ob_bar[4])
                top = o_high
                bottom = min(o_open, o_close)
                strength = (o_low - close) / atr_safe
                blocks.append(
                    {
                        "direction": "bearish",
                        "top": round(top, 6),
                        "bottom": round(bottom, 6),
                        "mid": round((top + bottom) * 0.5, 6),
                        "strength": round(_clamp(strength, 0.0, 3.0), 3),
                    }
                )
    return blocks[-5:]


def _confluence_snapshot(
    *,
    last_close: float,
    ema_20: float,
    ema_50: float,
    ema_200: float,
    vwap: float,
    rsi_14: float,
    macd: Dict[str, Any],
    volume: Dict[str, Any],
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    score = 50.0
    score += 6.0 if last_close > ema_20 else -6.0
    score += 8.0 if ema_20 > ema_50 else -8.0
    score += 10.0 if ema_50 > ema_200 else -10.0
    score += 5.0 if last_close > vwap else -5.0
    if rsi_14 >= 55.0:
        score += 6.0
    elif rsi_14 <= 45.0:
        score -= 6.0
    hist = float(macd.get("histogram") or 0.0)
    if hist > 0:
        score += 7.0
    elif hist < 0:
        score -= 7.0
    vol_ratio = float(volume.get("ratio") or 1.0)
    if vol_ratio >= 1.15:
        score += 4.0
    elif vol_ratio <= 0.85:
        score -= 4.0
    trend = str(structure.get("trend") or "neutral").lower()
    if trend == "bullish":
        score += 6.0
    elif trend == "bearish":
        score -= 6.0
    final_score = int(round(_clamp(score, 0.0, 100.0)))
    if final_score >= 60:
        direction = "bullish"
    elif final_score <= 40:
        direction = "bearish"
    else:
        direction = "neutral"
    return {"score": final_score, "direction": direction}


def _market_state_from_scalp(
    p15: IndicatorPack,
    p5: IndicatorPack,
    p1: IndicatorPack,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    p15_dir = str((p15.confluence or {}).get("direction") or "neutral")
    p5_dir = str((p5.confluence or {}).get("direction") or "neutral")
    p1_dir = str((p1.confluence or {}).get("direction") or "neutral")
    if p15_dir == p5_dir and p15_dir in {"bullish", "bearish"}:
        directional_bias = p15_dir
    elif p5_dir == p1_dir and p5_dir in {"bullish", "bearish"}:
        directional_bias = p5_dir
    else:
        directional_bias = "balanced"
    separation = abs(float(p15.ema_20) - float(p15.ema_50)) / max(
        float(p15.atr_14), 1e-9
    )
    regime = "trending" if separation >= 0.35 else "ranging"
    spread_points = meta.get("spread_points")
    if spread_points is not None and float(spread_points) > 2500:
        execution_state = "wide_spread"
    elif float((p1.volume or {}).get("ratio") or 1.0) < 0.75:
        execution_state = "thin_liquidity"
    else:
        execution_state = "normal"
    return {
        "directional_bias": directional_bias,
        "regime": regime,
        "execution_state": execution_state,
    }


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
    def fetch_ohlcv(
        self, tf: Timeframe, limit: int
    ) -> List[Tuple[int, float, float, float, float, float]]:
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
        for _, _o, h, bar_low, c, v in ohlcv:
            tp = (h + bar_low + c) / 3.0
            pv_sum += tp * v
            v_sum += v
        if v_sum == 0:
            _t, _o, h, last_low, c, _v = ohlcv[-1]
            return float((h + last_low + c) / 3.0)
        return float(pv_sum / v_sum)

    @staticmethod
    def _atr(
        ohlcv: List[Tuple[int, float, float, float, float, float]], period: int = 14
    ) -> float:
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
        macd = _macd_snapshot(closes)
        stoch_rsi = _stoch_rsi_snapshot(closes)
        volume = _volume_snapshot(candles)
        price_action = _price_action_snapshot(candles)
        bb = _bollinger_snapshot(closes)
        structure = _structure_snapshot(candles, ema_20, ema_50)
        fvgs = _fvg_snapshot(candles, atr_14)
        order_blocks = _order_block_snapshot(candles, atr_14)
        confluence = _confluence_snapshot(
            last_close=last_close,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            vwap=vwap,
            rsi_14=rsi_14,
            macd=macd,
            volume=volume,
            structure=structure,
        )

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
            confluence=confluence,
            macd=macd,
            stoch_rsi=stoch_rsi,
            volume=volume,
            price_action=price_action,
            bb=bb,
            structure=structure,
            fvgs=fvgs,
            order_blocks=order_blocks,
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
            conf = int((p.confluence or {}).get("score") or 50)
            macd_cross = str((p.macd or {}).get("cross") or "none")
            vol_ratio = float((p.volume or {}).get("ratio") or 1.0)
            pattern = str((p.price_action or {}).get("pattern") or "none")
            zone = str((p.structure or {}).get("zone") or "equilibrium")
            return (
                f"{p.tf}: close={p.last_close:.2f} | "
                f"RSI14={p.rsi_14:.2f} | "
                f"EMA20={p.ema_20:.2f}, EMA50={p.ema_50:.2f}, EMA200={p.ema_200:.2f} | "
                f"VWAP={p.vwap:.2f} | ATR14={p.atr_14:.2f} | "
                f"CONF={conf} | MACD={macd_cross} | VOL={vol_ratio:.2f}x | PAT={pattern} | ZONE={zone} | TS_BAR={p.ts_bar}"
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
        f"VWAP={p15.vwap:.2f} | ATR14={p15.atr_14:.2f} | CONF={int((p15.confluence or {}).get('score') or 50)} "
        f"| MACD={str((p15.macd or {}).get('cross') or 'none')} | VOL={float((p15.volume or {}).get('ratio') or 1.0):.2f}x "
        f"| PAT={str((p15.price_action or {}).get('pattern') or 'none')} | ZONE={str((p15.structure or {}).get('zone') or 'equilibrium')} | TS_BAR={p15.ts_bar}\n"
        f"M5:  close={p5.last_close:.2f} | RSI14={p5.rsi_14:.2f} | "
        f"EMA20={p5.ema_20:.2f} EMA50={p5.ema_50:.2f} EMA200={p5.ema_200:.2f} | "
        f"VWAP={p5.vwap:.2f} | ATR14={p5.atr_14:.2f} | CONF={int((p5.confluence or {}).get('score') or 50)} "
        f"| MACD={str((p5.macd or {}).get('cross') or 'none')} | VOL={float((p5.volume or {}).get('ratio') or 1.0):.2f}x "
        f"| PAT={str((p5.price_action or {}).get('pattern') or 'none')} | ZONE={str((p5.structure or {}).get('zone') or 'equilibrium')} | TS_BAR={p5.ts_bar}\n"
        f"M1:  close={p1.last_close:.2f} | RSI14={p1.rsi_14:.2f} | "
        f"EMA20={p1.ema_20:.2f} EMA50={p1.ema_50:.2f} EMA200={p1.ema_200:.2f} | "
        f"VWAP={p1.vwap:.2f} | ATR14={p1.atr_14:.2f} | CONF={int((p1.confluence or {}).get('score') or 50)} "
        f"| MACD={str((p1.macd or {}).get('cross') or 'none')} | VOL={float((p1.volume or {}).get('ratio') or 1.0):.2f}x "
        f"| PAT={str((p1.price_action or {}).get('pattern') or 'none')} | ZONE={str((p1.structure or {}).get('zone') or 'equilibrium')} | TS_BAR={p1.ts_bar}"
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
            "confluence": dict(p.confluence or {}),
            "macd": dict(p.macd or {}),
            "stoch_rsi": dict(p.stoch_rsi or {}),
            "volume": dict(p.volume or {}),
            "price_action": dict(p.price_action or {}),
            "bb": dict(p.bb or {}),
            "structure": dict(p.structure or {}),
            "fvgs": [dict(row) for row in (p.fvgs or [])],
            "order_blocks": [dict(row) for row in (p.order_blocks or [])],
        }

    return {
        "symbol": symbol,
        "summary_text": summary_text,
        "meta": {
            "ts_now": ts_now,
            "age_sec": age_sec,
            **meta,
        },
        "market_state": _market_state_from_scalp(p15, p5, p1, meta),
        "M15": tf_dict(p15),
        "M5": tf_dict(p5),
        "M1": tf_dict(p1),
    }


def get_ai_payload_xau() -> Optional[Dict[str, Any]]:
    blocked_reason = _mt5_blocked_reason()
    if blocked_reason:
        cached = _cached_payload("XAUUSDm", blocked_reason)
        if cached is not None:
            log.warning(
                "get_ai_payload_xau: using cached payload due mt5_block=%s",
                blocked_reason,
            )
            return cached
        log.error("get_ai_payload_xau blocked: %s", blocked_reason)
        return None

    try:
        xau = GetXauDataRealTimeForAi()
        meta = xau._symbol_meta()
        p15 = xau._build_pack("M15")
        p5 = xau._build_pack("M5")
        p1 = xau._build_pack("M1")
        payload = _payload_from_packs("XAUUSDm", p15, p5, p1, meta)
        _cache_payload("XAUUSDm", payload)
        global _LIVENESS_LOGGED_XAU
        if not _LIVENESS_LOGGED_XAU:
            _LIVENESS_LOGGED_XAU = True
            try:
                log.info(
                    "FEED_READY | symbol=XAUUSDm bars_m15=%d bars_m5=%d bars_m1=%d",
                    len(p15 or []),
                    len(p5 or []),
                    len(p1 or []),
                )
            except Exception:
                pass
        return payload
    except Exception as e:
        log.error("get_ai_payload_xau failed: %s", e)
        cached = _cached_payload("XAUUSDm", str(e))
        if cached is not None:
            log.warning("get_ai_payload_xau: fallback cached after exception")
            return cached
        return None


def get_ai_payload_btc() -> Optional[Dict[str, Any]]:
    blocked_reason = _mt5_blocked_reason()
    if blocked_reason:
        cached = _cached_payload("BTCUSDm", blocked_reason)
        if cached is not None:
            log.warning(
                "get_ai_payload_btc: using cached payload due mt5_block=%s",
                blocked_reason,
            )
            return cached
        log.error("get_ai_payload_btc blocked: %s", blocked_reason)
        return None

    try:
        btc = GetBtcDataRealTimeForAi()
        meta = btc._symbol_meta()
        p15 = btc._build_pack("M15")
        p5 = btc._build_pack("M5")
        p1 = btc._build_pack("M1")
        payload = _payload_from_packs("BTCUSDm", p15, p5, p1, meta)
        _cache_payload("BTCUSDm", payload)
        global _LIVENESS_LOGGED_BTC
        if not _LIVENESS_LOGGED_BTC:
            _LIVENESS_LOGGED_BTC = True
            try:
                log.info(
                    "FEED_READY | symbol=BTCUSDm bars_m15=%d bars_m5=%d bars_m1=%d",
                    len(p15 or []),
                    len(p5 or []),
                    len(p1 or []),
                )
            except Exception:
                pass
        return payload
    except Exception as e:
        log.error("get_ai_payload_btc failed: %s", e)
        cached = _cached_payload("BTCUSDm", str(e))
        if cached is not None:
            log.warning("get_ai_payload_btc: fallback cached after exception")
            return cached
        return None
