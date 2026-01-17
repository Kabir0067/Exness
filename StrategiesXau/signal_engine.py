from __future__ import annotations

import hashlib
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None  # type: ignore

from config_xau import EngineConfig, SymbolParams
from DataFeed.market_feed import MarketFeed, TickStats
from mt5_client import MT5_LOCK
from StrategiesXau.indicators import Classic_FeatureEngine, safe_last
from StrategiesXau.risk_management import RiskManager
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# ============================================================
# ERROR-only logging (isolated file, no collision)
# ============================================================
LOG_DIR = LOG_ROOT

log = logging.getLogger("signal_engine")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fh = logging.FileHandler(str(get_log_path("signal_engine.log")), encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log.addHandler(fh)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _atr_np(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Minimal ATR fallback (Wilder smoothing) when TA-Lib is unavailable.
    Returns array of length len(c) with NaNs until period-1.
    """
    n = int(len(c))
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    if n == 1:
        out[0] = float(h[0] - l[0])
        return out

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    if n < period:
        out[-1] = float(np.mean(tr))
        return out

    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    if talib is not None:
        try:
            return talib.ATR(h, l, c, period)  # type: ignore[attr-defined]
        except Exception:
            return _atr_np(h, l, c, period)
    return _atr_np(h, l, c, period)


@dataclass(frozen=True, slots=True)
class SignalResult:
    symbol: str
    signal: str  # "Buy" | "Sell" | "Neutral"
    confidence: int  # 0..100
    regime: Optional[str]
    reasons: List[str]
    spread_bps: Optional[float]
    latency_ms: float
    timestamp: str

    # plan (NOT execution)
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lot: Optional[float] = None

    # routing / control
    signal_id: str = ""
    trade_blocked: bool = False


class SignalEngine:
    """
    Production-grade SignalEngine (planner-only):
      - execute=True => attaches (entry/sl/tp/lot) via RiskManager.plan_order, NEVER sends orders
      - ERROR-only logging for real exceptions only
      - debounce + stability gating (monotonic time)
      - deterministic signal_id (bar-time based)
      - robust compatibility with different FeatureEngine outputs

    IMPORTANT:
      Bot/engine.py MUST call compute(execute=True) for Buy/Sell, otherwise lot/sl/tp stay None => no trades.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        sp: SymbolParams,
        market_feed: MarketFeed,
        feature_engine: Classic_FeatureEngine,
        risk_manager: RiskManager,
    ) -> None:
        self.cfg = cfg
        self.sp = sp
        self.feed = market_feed
        self.features = feature_engine
        self.risk = risk_manager

        self._last_decision_ms = 0.0
        self._last_net_norm = 0.0
        self._last_signal = "Neutral"

        self._current_drawdown = 0.0

        # throttle defaults (safe)
        self._debounce_ms = float(getattr(self.cfg, "decision_debounce_ms", 150.0) or 150.0)
        self._stable_eps = float(getattr(self.cfg, "signal_stability_eps", 0.03) or 0.03)

        # ensemble weights defaults (safe)
        if not hasattr(self.cfg, "weights") or not isinstance(getattr(self.cfg, "weights", None), dict):
            setattr(
                self.cfg,
                "weights",
                {"trend": 0.55, "momentum": 0.27, "meanrev": 0.10, "structure": 0.05, "volume": 0.03},
            )

    # --------------------------- public
    def compute(self, execute: bool = False) -> SignalResult:
        """
        execute=False: signal only
        execute=True : signal + plan (entry/sl/tp/lot) via RiskManager.plan_order
        NOTE: execute NEVER means order_send.
        """
        t0 = time.perf_counter()
        sym = self._symbol()

        try:
            # 1) Fetch MTF rates (fast fallbacks)
            dfp = self.feed.get_rates(sym, self.sp.tf_primary)
            if dfp is None or dfp.empty:
                return self._neutral(sym, ["no_rates_primary"], t0)

            dfc = self.feed.get_rates(sym, self.sp.tf_confirm)
            if dfc is None or dfc.empty:
                dfc = dfp

            dfl = self.feed.get_rates(sym, self.sp.tf_long)
            if dfl is None or dfl.empty:
                dfl = dfp

            bar_key = self._bar_key(dfp)

            # 2) Quick context
            spread_pct = float(self.feed.spread_pct() or 0.0)
            last_age = float(self.feed.last_bar_age(dfp) or 0.0)
            atr_lb = int(getattr(self.cfg, "atr_percentile_lookback", 400) or 400)
            atr_pct = float(self.risk.atr_percentile(dfp, atr_lb) or 0.0)
            tick_stats: TickStats = self.feed.tick_stats(dfp)
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            # 3) Guards
            if not self.risk.market_open_24_5():
                return self._neutral(
                    sym,
                    ["market_closed"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            in_session = self._in_active_session()
            drawdown_exceeded = self._check_drawdown()
            latency_ok = bool(self.risk.latency_cooldown())

            decision = self.risk.guard_decision(
                spread_pct=spread_pct,
                tick_ok=bool(getattr(tick_stats, "ok", True)),
                tick_reason=str(getattr(tick_stats, "reason", "") or ""),
                ingest_ms=float(ingest_ms),
                last_bar_age=float(last_age),
                in_session=bool(in_session),
                drawdown_exceeded=bool(drawdown_exceeded),
                latency_cooldown=bool(latency_ok),
                tz=getattr(self.feed, "tz", None),
            )
            if not getattr(decision, "allowed", False):
                reasons = list(getattr(decision, "reasons", []) or ["guard_block"])
                return self._neutral(
                    sym,
                    reasons,
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            # 4) Indicators (MTF)
            tf_data = {self.sp.tf_primary: dfp, self.sp.tf_confirm: dfc, self.sp.tf_long: dfl}
            indicators = self.features.compute_indicators(tf_data)
            if not indicators:
                return self._neutral(sym, ["no_indicators"], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True)

            if bool(indicators.get("trade_blocked", False)):
                reasons = ["anomaly_block"] + list(indicators.get("anomaly_reasons", []) or [])
                return self._neutral(sym, reasons[:25], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True)

            indp = indicators.get(self.sp.tf_primary) or {}
            indc = indicators.get(self.sp.tf_confirm) or indp
            indl = indicators.get(self.sp.tf_long) or indp
            if not indp:
                return self._neutral(sym, ["no_primary_indicators"], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True)

            if bool(indp.get("trade_blocked", False)):
                reasons = ["anomaly_block_primary"] + list(indp.get("anomaly_reasons", []) or [])
                return self._neutral(sym, reasons[:25], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True)

            if bool(getattr(self.cfg, "use_squeeze_filter", False)) and bool(indp.get("squeeze_on", False)):
                return self._neutral(
                    sym,
                    ["squeeze_wait"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=str(indp.get("regime") or None),
                )

            # 5) Adaptive params
            adapt = self._get_adaptive_params(indp, indl, atr_pct)

            # 6) HTF alignment
            close_p = float(indp.get("close", 0.0) or 0.0)
            ema50_l = float(indl.get("ema50", close_p) or close_p)
            adx_l = float(indl.get("adx", 0.0) or 0.0)
            adx_lo = float(getattr(self.cfg, "adx_trend_lo", 18.0) or 18.0)

            trend_ok_buy = (adx_l >= adx_lo and close_p > ema50_l * 0.999)
            trend_ok_sell = (adx_l >= adx_lo and close_p < ema50_l * 1.001)

            # 7) Ensemble score
            book = self.feed.fetch_book() or {}
            net_norm, conf = self._ensemble_score(indp, indc, indl, book, adapt, spread_pct, tick_stats)

            # 8) Confluence layer
            ext = self._extreme_flag(dfp, indp)
            sweep = self._liquidity_sweep(dfp, indp)
            div = self._divergence(dfp, indp)
            near_rn = self._near_round(close_p, indp)

            ok, why = self._confirm_layer(ext, sweep, div, near_rn)
            if not ok:
                return self._neutral(
                    sym,
                    ["extreme_guard"] + why,
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=str(adapt.get("regime")),
                    trade_blocked=True,
                )

            if sweep or div != "none":
                conf = min(100, int(conf * 1.15))
            if near_rn and not (sweep or div != "none"):
                conf = max(0, int(conf * 0.75))

            # 9) Decide signal
            signal = "Neutral"
            blocked_by_htf = False
            conf_min = int(adapt.get("conf_min", 98) or 98)

            if conf >= conf_min:
                if net_norm > 0.05 and trend_ok_buy:
                    signal = "Buy"
                elif net_norm < -0.05 and trend_ok_sell:
                    signal = "Sell"
                else:
                    blocked_by_htf = True

            # 10) Advanced filters
            if signal != "Neutral" and not self._conformal_ok(dfp):
                return self._neutral(sym, ["conformal_abstain"], t0, spread_pct=spread_pct, bar_key=bar_key, regime=str(adapt.get("regime")), trade_blocked=True)

            if signal in ("Buy", "Sell") and not self._gate_meta(signal, indp, dfp):
                return self._neutral(sym, ["meta_gate_block"], t0, spread_pct=spread_pct, bar_key=bar_key, regime=str(adapt.get("regime")), trade_blocked=True)

            signal, conf = self._apply_filters(signal, conf, indp)

            # 10.5) Risk-level signal emission gate (align with BTC stack)
            if signal in ("Buy", "Sell") and hasattr(self.risk, "can_emit_signal"):
                try:
                    allowed, gate_reasons = self.risk.can_emit_signal(int(conf), getattr(self.feed, "tz", None))
                    if not bool(allowed):
                        return self._neutral(
                            sym,
                            ["emit_gate"] + list(gate_reasons or []),
                            t0,
                            spread_pct=spread_pct,
                            bar_key=bar_key,
                            regime=str(adapt.get("regime")),
                            trade_blocked=True,
                        )
                except Exception:
                    # Fail-safe: block trade if gate crashes
                    return self._neutral(
                        sym,
                        ["emit_gate_error"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=str(adapt.get("regime")),
                        trade_blocked=True,
                    )

            if signal == "Neutral" and blocked_by_htf:
                conf = min(conf, max(0, conf_min - 1))

            # 11) Debounce + stability
            now_ms = time.monotonic() * 1000.0
            if (now_ms - self._last_decision_ms) < self._debounce_ms:
                return self._neutral(sym, ["debounce"], t0, spread_pct=spread_pct, bar_key=bar_key, regime=str(adapt.get("regime")))

            if signal == self._last_signal and abs(net_norm - self._last_net_norm) < self._stable_eps:
                return self._neutral(sym, ["stable"], t0, spread_pct=spread_pct, bar_key=bar_key, regime=str(adapt.get("regime")))

            self._last_decision_ms = now_ms
            self._last_net_norm = float(net_norm)
            self._last_signal = str(signal)

            # 12) Finalize (plan only)
            reasons: List[str] = []
            if sweep:
                reasons.append(f"sweep:{sweep}")
            if div != "none":
                reasons.append(f"div:{div}")
            if near_rn:
                reasons.append("near_rn")
            reasons.append(f"net:{net_norm:.3f}")

            return self._finalize(
                sym=sym,
                signal=signal,
                conf=conf,
                indp=indp,
                adapt=adapt,
                spread_pct=spread_pct,
                t0=t0,
                execute=bool(execute),
                tick_stats=tick_stats,
                book=book,
                bar_key=bar_key,
                reasons=reasons,
            )

        except Exception as exc:
            log.error("compute crash: %s | tb=%s", exc, traceback.format_exc())
            return self._neutral(sym, ["compute_error"], t0, trade_blocked=True)

    # --------------------------- helpers
    def _symbol(self) -> str:
        return self.sp.resolved or self.sp.base

    def _bar_key(self, df: pd.DataFrame) -> str:
        try:
            if df is None or df.empty:
                return "no_bar"
            if "time" in df.columns:
                t = df["time"].iloc[-1]
            elif "datetime" in df.columns:
                t = df["datetime"].iloc[-1]
            elif "ts" in df.columns:
                t = df["ts"].iloc[-1]
            else:
                t = df.index[-1]

            if isinstance(t, (pd.Timestamp, np.datetime64)):
                t = pd.to_datetime(t)
                return t.isoformat()
            return str(t)
        except Exception:
            return "bar_err"

    def _signal_id(self, sym: str, tf: str, bar_key: str, signal: str) -> str:
        raw = f"{sym}|{tf}|{bar_key}|{signal}"
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"{sym}:{tf}:{signal}:{h}"

    def _neutral(
        self,
        sym: str,
        reasons: List[str],
        t0: float,
        *,
        spread_pct: Optional[float] = None,
        regime: Optional[str] = None,
        bar_key: str = "no_bar",
        trade_blocked: bool = False,
    ) -> SignalResult:
        comp_ms = (time.perf_counter() - t0) * 1000.0
        sid = self._signal_id(sym, str(self.sp.tf_primary), bar_key, "Neutral")
        return SignalResult(
            symbol=sym,
            signal="Neutral",
            confidence=0,
            regime=regime,
            reasons=reasons[:25],
            spread_bps=None if spread_pct is None else float(spread_pct) * 10000.0,
            latency_ms=float(comp_ms),
            timestamp=self.feed.now_local().isoformat(),
            signal_id=sid,
            trade_blocked=bool(trade_blocked),
        )

    def _in_active_session(self) -> bool:
        try:
            if bool(getattr(self.cfg, "ignore_sessions", False)):
                return True
            h = int(self.feed.now_local().hour)
            sessions = list(getattr(self.cfg, "active_sessions", []) or [])
            return any(int(s) <= h < int(e) for s, e in sessions)
        except Exception:
            return True

    def _check_drawdown(self) -> bool:
        try:
            with MT5_LOCK:
                acc = mt5.account_info()
            if not acc:
                return False
            bal = float(acc.balance or 0.0)
            eq = float(acc.equity or 0.0)
            if bal <= 0:
                self._current_drawdown = 0.0
            else:
                self._current_drawdown = max(0.0, (bal - eq) / bal)
            return self._current_drawdown >= float(getattr(self.cfg, "max_drawdown", 0.09) or 0.09)
        except Exception:
            return False

    def _get_adaptive_params(self, indp: Dict[str, Any], indl: Dict[str, Any], atr_pct: float) -> Dict[str, Any]:
        phase = str(getattr(self.risk, "current_phase", "A") or "A")
        if hasattr(self.features, "adaptive_params"):
            try:
                return self.features.adaptive_params(indp, indl, atr_pct, phase=phase)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Fallback if adaptive params missing
        conf_min = int(getattr(self.cfg, "conf_min", 94) or 94)
        if phase not in ("A", "B"):
            conf_min = min(100, conf_min + 2)

        return {
            "conf_min": int(conf_min),
            "sl_mult": float(getattr(self.cfg, "sl_atr_mult_trend", 1.35) or 1.35),
            "tp_mult": float(getattr(self.cfg, "tp_atr_mult_trend", 3.2) or 3.2),
            "trail_mult": float(getattr(self.cfg, "trail_atr_mult", 1.0) or 1.0),
            "w_mul": {"trend": 1.0, "momentum": 1.0, "meanrev": 1.0, "structure": 1.0, "volume": 1.0},
            "regime": "trend",
            "phase": phase,
        }

    def _extreme_flag(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> bool:
        try:
            if hasattr(self.features, "extreme_zone"):
                z = self.features.extreme_zone(dfp, indp)  # type: ignore[attr-defined]
                if isinstance(z, dict):
                    return bool(z.get("near_top", False) or z.get("near_bot", False) or z.get("exhausted", False))
                return bool(z)
            return False
        except Exception:
            return False

    def _liquidity_sweep(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> str:
        try:
            if hasattr(self.features, "liquidity_sweep"):
                s: Any = self.features.liquidity_sweep(dfp, indp)  # type: ignore[attr-defined]
            else:
                s = indp.get("liquidity_sweep", "")
            s = str(s or "").strip().lower()
            return s if s in ("bull", "bear") else ""
        except Exception:
            return ""

    def _divergence(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> str:
        try:
            if hasattr(self.features, "hidden_divergence"):
                d = self.features.hidden_divergence(dfp)  # type: ignore[attr-defined]
                d = str(d or "").strip().lower()
                if d in ("bull", "bullish"):
                    return "bullish"
                if d in ("bear", "bearish"):
                    return "bearish"

            d2 = str(indp.get("rsi_div", "none") or "none").strip().lower()
            if d2 in ("bull", "bullish"):
                return "bullish"
            if d2 in ("bear", "bearish"):
                return "bearish"
            return "none"
        except Exception:
            return "none"

    def _near_round(self, price: float, indp: Dict[str, Any]) -> bool:
        try:
            if hasattr(self.features, "round_number_confluence"):
                return bool(self.features.round_number_confluence(price))  # type: ignore[attr-defined]
            return bool(indp.get("near_round", False))
        except Exception:
            return False

    def _ensemble_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        book: Dict[str, Any],
        adapt: Dict[str, Any],
        spread_pct: float,
        tick_stats: TickStats,
    ) -> Tuple[float, int]:
        try:
            _ = (book, spread_pct, tick_stats)

            w_mul = adapt.get("w_mul") or {}
            w = getattr(self.cfg, "weights", {}) or {}
            weights = np.array(
                [
                    float(w.get("trend", 0.55)) * float(w_mul.get("trend", 1.0)),
                    float(w.get("momentum", 0.27)) * float(w_mul.get("momentum", 1.0)),
                    float(w.get("meanrev", 0.10)) * float(w_mul.get("meanrev", 1.0)),
                    float(w.get("structure", 0.05)) * float(w_mul.get("structure", 1.0)),
                    float(w.get("volume", 0.03)) * float(w_mul.get("volume", 1.0)),
                ],
                dtype=np.float64,
            )
            s = float(np.sum(weights))
            weights = weights / s if s > 0 else np.array([0.55, 0.27, 0.10, 0.05, 0.03], dtype=np.float64)

            regime = str(adapt.get("regime", "trend") or "trend")

            scores = np.zeros(5, dtype=np.float64)
            scores[0] = float(self._trend_score(indp, indc, indl, regime))
            scores[1] = float(self._momentum_score(indp, indc))
            scores[2] = float(self._meanrev_score(indp, regime))
            scores[3] = float(self._structure_score(indp))

            z_vol = float(indp.get("z_vol", indp.get("z_volume", 0.0)) or 0.0)
            scores[4] = float(np.tanh(z_vol / 3.0))

            net_norm = float(np.sum(scores * weights))
            base_conf = 50.0 + (net_norm * 30.0)
            conf = int(_clamp(base_conf, 10.0, 99.0))
            return net_norm, conf
        except Exception as exc:
            log.error("_ensemble_score crash: %s | tb=%s", exc, traceback.format_exc())
            return 0.0, 0

    def _ema_s_m(self, indp: Dict[str, Any], close_p: float) -> Tuple[float, float]:
        ema_s = float(indp.get("ema9", indp.get("ema8", indp.get("ema_short", close_p))) or close_p)
        ema_m = float(indp.get("ema21", indp.get("ema_mid", close_p)) or close_p)
        return ema_s, ema_m

    def _trend_score(self, indp: Dict[str, Any], indc: Dict[str, Any], indl: Dict[str, Any], regime: str) -> float:
        try:
            _ = indc
            close_p = float(indp.get("close", 0.0) or 0.0)
            ema50_l = float(indl.get("ema50", close_p) or close_p)
            adx_l = float(indl.get("adx", 0.0) or 0.0)
            adx_lo = float(getattr(self.cfg, "adx_trend_lo", 18.0) or 18.0)

            sc = 0.0
            if adx_l >= adx_lo:
                if close_p > ema50_l * 0.999:
                    sc += 1.0
                elif close_p < ema50_l * 1.001:
                    sc -= 1.0

            if regime == "trend":
                ema_s, ema_m = self._ema_s_m(indp, close_p)
                if close_p > ema_s > ema_m:
                    sc += 0.5
                elif close_p < ema_s < ema_m:
                    sc -= 0.5
            return float(sc)
        except Exception:
            return 0.0

    def _momentum_score(self, indp: Dict[str, Any], indc: Dict[str, Any]) -> float:
        try:
            _ = indc
            sc = 0.0
            macd = float(indp.get("macd", 0.0) or 0.0)
            macd_sig = float(indp.get("macd_sig", indp.get("macd_signal", 0.0)) or 0.0)
            macd_hist = float(indp.get("macd_hist", macd - macd_sig) or (macd - macd_sig))

            if macd_hist > 0 and macd > macd_sig:
                sc += 1.6
            elif macd_hist < 0 and macd < macd_sig:
                sc -= 1.6

            rsi = float(indp.get("rsi", 50.0) or 50.0)
            if rsi > 60:
                sc += 1.0
            elif rsi < 40:
                sc -= 1.0
            return float(sc)
        except Exception:
            return 0.0

    def _meanrev_score(self, indp: Dict[str, Any], regime: str) -> float:
        try:
            if regime != "range":
                return 0.0
            bb_u = float(indp.get("bb_upper", 0.0) or 0.0)
            bb_l = float(indp.get("bb_lower", 0.0) or 0.0)
            close_p = float(indp.get("close", 0.0) or 0.0)
            width = max(1e-9, bb_u - bb_l)
            bb_pos = (close_p - bb_l) / width
            if bb_pos <= 0.12:
                return 1.6
            if bb_pos >= 0.88:
                return -1.6
            return 0.0
        except Exception:
            return 0.0

    def _structure_score(self, indp: Dict[str, Any]) -> float:
        try:
            sc = 0.0
            if str(indp.get("liquidity_sweep", "") or ""):
                sc += 1.0
            if bool(indp.get("fvg_bull", False)) or bool(indp.get("fvg_bear", False)):
                sc += 0.5
            if str(indp.get("rsi_div", "none") or "none") != "none":
                sc += 0.5
            if bool(indp.get("near_round", False)):
                sc += 0.25
            return float(sc)
        except Exception:
            return 0.0

    def _confirm_layer(self, ext: bool, sweep: str, div: str, near_rn: bool) -> Tuple[bool, List[str]]:
        try:
            if not ext:
                return True, []
            has_bear = (sweep == "bear" or div == "bearish")
            has_bull = (sweep == "bull" or div == "bullish")
            if not has_bear and not has_bull and not near_rn:
                return False, ["extreme_needs_confirmation"]
            return True, []
        except Exception:
            return True, []

    def _conformal_ok(self, dfp: pd.DataFrame) -> bool:
        try:
            W = int(getattr(self.cfg, "conformal_window", 300) or 300)
            qv = float(getattr(self.cfg, "conformal_q", 0.88) or 0.88)

            if dfp is None or len(dfp) < (W + 25):
                return True

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)

            atr = _atr(h, l, c, 14)
            rr = np.abs(np.diff(c)) / np.maximum(1e-9, atr[1:])
            if rr.size < 50:
                return True

            base = rr[-(W + 50): -50] if rr.size > (W + 60) else rr[:-5]
            if base.size < 30:
                return True

            q = float(np.quantile(base, qv))
            return bool(rr[-1] <= q)
        except Exception:
            return True

    def _gate_meta(self, side: str, ind: Dict[str, Any], dfp: pd.DataFrame) -> bool:
        try:
            _ = ind
            h_bars = int(getattr(self.cfg, "meta_h_bars", 6) or 6)
            R = float(getattr(self.cfg, "meta_barrier_R", 0.65) or 0.65)
            tc_bps = float(getattr(self.cfg, "tc_bps", 2.0) or 2.0)

            if dfp is None or len(dfp) < (h_bars + 25):
                return True

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)

            atr = _atr(h, l, c, 14)
            atr_val = float(safe_last(atr, 1.0) or 1.0)
            if atr_val <= 0:
                return True

            sign = 1.0 if side == "Buy" else -1.0
            wins = 0
            total = 0

            start_i = max(1, len(c) - h_bars - 20)
            end_i = max(start_i + 1, len(c) - h_bars - 1)
            for i in range(start_i, end_i):
                move = (c[i + h_bars] - c[i]) * sign
                if move >= (R * atr_val):
                    wins += 1
                total += 1

            if total < 10:
                return True

            hitrate = wins / total
            edge_needed = (tc_bps / 10000.0) * 3.0
            return bool(hitrate >= (0.5 + edge_needed))
        except Exception:
            return True

    def _apply_filters(self, signal: str, conf: int, ind: Dict[str, Any]) -> Tuple[str, int]:
        try:
            if signal == "Neutral":
                return "Neutral", int(conf)

            atr = float(ind.get("atr", 0.0) or 0.0)
            body = float(ind.get("body", 0.0) or 0.0)
            if body <= 0.0:
                try:
                    body = abs(float(ind.get("close", 0.0) or 0.0) - float(ind.get("open", 0.0) or 0.0))
                except Exception:
                    body = 0.0

            min_body_k = float(getattr(self.cfg, "min_body_pct_of_atr", 0.14) or 0.14)

            if atr > 0 and body < (min_body_k * atr):
                conf = max(0, conf - 25)

            close_p = float(ind.get("close", 0.0) or 0.0)
            ema_s, ema_m = self._ema_s_m(ind, close_p)

            if signal == "Buy" and close_p > ema_s > ema_m:
                conf = min(100, conf + 10)
            if signal == "Sell" and close_p < ema_s < ema_m:
                conf = min(100, conf + 10)

            return signal, int(conf)
        except Exception:
            return signal, int(conf)
    def _finalize(
        self,
        *,
        sym: str,
        signal: str,
        conf: int,
        indp: Dict[str, Any],
        adapt: Dict[str, Any],
        spread_pct: float,
        t0: float,
        execute: bool,
        tick_stats: TickStats,
        book: Dict[str, Any],
        bar_key: str,
        reasons: List[str],
    ) -> SignalResult:
        comp_ms = (time.perf_counter() - t0) * 1000.0
        sid = self._signal_id(sym, str(self.sp.tf_primary), bar_key, signal)

        try:
            regime = str(adapt.get("regime") or None)

            entry_val = sl_val = tp_val = lot_val = None

            if signal in ("Buy", "Sell") and execute:
                zones = self.feed.micro_price_zones(book)
                tick_vol = float(getattr(tick_stats, "volatility", 0.0) or 0.0)
                open_pos, unreal_pl = self._position_context(sym)

                                # Use executable market price (bid/ask) as entry for SL/TP planning (scalping-safe)
                entry_price = float(indp.get("close", 0.0) or 0.0)
                try:
                    with MT5_LOCK:
                        tick = mt5.symbol_info_tick(sym)
                    if tick is not None:
                        px = float(tick.ask if signal == "Buy" else tick.bid)
                        if px > 0:
                            entry_price = px
                except Exception:
                    pass

                entry_val, sl_val, tp_val, lot_val = self.risk.plan_order(
                    signal,
                    float(conf) / 100.0,
                    indp,
                    adapt,
                    entry=float(entry_price),
                    ticks=tick_stats,
                    zones=zones,
                    tick_volatility=tick_vol,
                    open_positions=open_pos,
                    max_positions=int(getattr(self.cfg, "max_positions", 3) or 3),
                    unrealized_pl=float(unreal_pl),
                )

                if entry_val is None or sl_val is None or tp_val is None or lot_val is None:
                    return SignalResult(
                        symbol=sym,
                        signal="Neutral",
                        confidence=0,
                        regime=regime,
                        reasons=["plan_order_failed"],
                        spread_bps=float(spread_pct) * 10000.0,
                        latency_ms=float(comp_ms),
                        timestamp=self.feed.now_local().isoformat(),
                        signal_id=self._signal_id(sym, str(self.sp.tf_primary), bar_key, "Neutral"),
                        trade_blocked=True,
                    )

            return SignalResult(
                symbol=sym,
                signal=signal,
                confidence=int(conf),
                regime=regime,
                reasons=reasons[:25],
                spread_bps=float(spread_pct) * 10000.0,
                latency_ms=float(comp_ms),
                timestamp=self.feed.now_local().isoformat(),
                entry=None if entry_val is None else float(entry_val),
                sl=None if sl_val is None else float(sl_val),
                tp=None if tp_val is None else float(tp_val),
                lot=None if lot_val is None else float(lot_val),
                signal_id=sid,
                trade_blocked=False,
            )
        except Exception as exc:
            log.error("_finalize crash: %s | tb=%s", exc, traceback.format_exc())
            return SignalResult(
                symbol=sym,
                signal="Neutral",
                confidence=0,
                regime=str(adapt.get("regime") or None),
                reasons=["finalize_error"],
                spread_bps=float(spread_pct) * 10000.0,
                latency_ms=float(comp_ms),
                timestamp=self.feed.now_local().isoformat(),
                signal_id=self._signal_id(sym, str(self.sp.tf_primary), bar_key, "Neutral"),
                trade_blocked=True,
            )

    def _position_context(self, sym: str) -> Tuple[int, float]:
        try:
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=sym) or []
            open_pos = int(len(positions))
            unreal_pl = float(sum(float(p.profit or 0.0) for p in positions))
            return open_pos, unreal_pl
        except Exception:
            return 0, 0.0


__all__ = ["SignalEngine", "SignalResult"]
