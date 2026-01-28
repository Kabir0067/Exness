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
from DataFeed.xau_market_feed import MarketFeed, TickStats
from log_config import LOG_DIR as LOG_ROOT, get_log_path
from mt5_client import MT5_LOCK
from StrategiesXau.xau_indicators import Classic_FeatureEngine, safe_last
from StrategiesXau.xau_risk_management import RiskManager

# ============================================================
# ERROR-only logging (isolated file, no collision)
# ============================================================
LOG_DIR = LOG_ROOT

log = logging.getLogger("signal_xau")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fh = logging.FileHandler(str(get_log_path("signal_engine_xau.log")), encoding="utf-8", delay=True)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s")
    )
    log.addHandler(fh)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _as_regime(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "none":
        return None
    return s


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

    prev_c = np.empty(n, dtype=np.float64)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]

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

        self._debounce_ms = float(getattr(self.cfg, "decision_debounce_ms", 150.0) or 150.0)
        self._stable_eps = float(getattr(self.cfg, "signal_stability_eps", 0.03) or 0.03)

        if not hasattr(self.cfg, "weights") or not isinstance(getattr(self.cfg, "weights", None), dict):
            setattr(
                self.cfg,
                "weights",
                {"trend": 0.50, "momentum": 0.25, "meanrev": 0.10, "structure": 0.10, "volume": 0.05},  # Ислоҳ: Вазнҳо беҳтар карда шуданд (structure ва volume баландтар)
            )

        self._w_keys = ("trend", "momentum", "meanrev", "structure", "volume")
        self._w_sig: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._w_base = np.array([0.50, 0.25, 0.10, 0.10, 0.05], dtype=np.float64)  # Ислоҳ: Вазнҳои нав истифода мешаванд

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

            spread_pct = float(self.feed.spread_pct() or 0.0)
            last_age = float(self.feed.last_bar_age(dfp) or 0.0)

            atr_lb = int(getattr(self.cfg, "atr_percentile_lookback", 400) or 400)
            atr_pct = float(self.risk.atr_percentile(dfp, atr_lb) or 0.0)

            tick_stats: TickStats = self.feed.tick_stats(dfp)
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            # Phase C fast-stop
            self.risk.evaluate_account_state()
            if self.risk.current_phase == "C" and not bool(
                getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False)
            ):
                return self._neutral(
                    sym,
                    ["phase_c_protect"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

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

            # Indicators (MTF)
            tf_data = {self.sp.tf_primary: dfp, self.sp.tf_confirm: dfc, self.sp.tf_long: dfl}
            indicators = self.features.compute_indicators(tf_data)
            if not indicators:
                return self._neutral(
                    sym, ["no_indicators"], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True
                )

            if bool(indicators.get("trade_blocked", False)):
                reasons = ["anomaly_block"] + list(indicators.get("anomaly_reasons", []) or [])
                return self._neutral(
                    sym, reasons[:25], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True
                )

            indp = indicators.get(self.sp.tf_primary) or {}
            indc = indicators.get(self.sp.tf_confirm) or indp
            indl = indicators.get(self.sp.tf_long) or indp
            if not indp:
                return self._neutral(
                    sym,
                    ["no_primary_indicators"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            if bool(indp.get("trade_blocked", False)):
                reasons = ["anomaly_block_primary"] + list(indp.get("anomaly_reasons", []) or [])
                return self._neutral(
                    sym, reasons[:25], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True
                )

            if bool(getattr(self.cfg, "use_squeeze_filter", False)) and bool(indp.get("squeeze_on", False)):
                return self._neutral(
                    sym,
                    ["squeeze_wait"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=_as_regime(indp.get("regime")),
                )

            adapt = self._get_adaptive_params(indp, indl, atr_pct)

            # HTF alignment
            close_p = float(indp.get("close", 0.0) or 0.0)
            ema50_l = float(indl.get("ema50", close_p) or close_p)
            adx_l = float(indl.get("adx", 0.0) or 0.0)
            adx_lo_ltf = float(getattr(self.cfg, "adx_trend_lo", 18.0) or 18.0)

            ema_s, ema_m = self._ema_s_m(indp, close_p)
            require_stack = bool(getattr(self.cfg, "require_ema_stack", True))

            trend_l = str(indl.get("trend", "") or "").lower()
            bias_up = "up" in trend_l
            bias_dn = "down" in trend_l

            # ИСЛОҲ: Шартҳои содатар барои сигналҳои Buy ва Sell
            # Buy: Нарх аз EMA50 боло бошад ва ADX >= 16 ё тренд боло
            trend_ok_buy = (close_p > ema50_l * 0.997) and (adx_l >= adx_lo_ltf * 0.8 or bias_up)
            
            # Sell: Шартҳо хело содатар - агар яке аз инҳо бошад = Sell иҷозат дорад:
            # 1. Нарх аз EMA50 поён бошад
            # 2. Тренд вниз бошад
            # 3. ADX паст бошад (= боҳисобии range)
            # 4. Нарх аз EMA_short поён бошад
            trend_ok_sell = (
                (close_p < ema50_l * 1.003) or 
                bias_dn or 
                (adx_l < adx_lo_ltf * 2.0) or
                (close_p < ema_s)
            )

            if require_stack:
                trend_ok_buy = trend_ok_buy and (close_p > ema_s * 0.999)
                # Барои Sell: Агар нарх аз EMA поён ё тренд вниз бошад - иҷозат
                trend_ok_sell = trend_ok_sell or (close_p < ema_m) or (not (close_p > ema_s > ema_m))

            # Ensemble score
            book = self.feed.fetch_book() or {}
            net_norm, conf = self._ensemble_score(indp, indc, indl, book, adapt, spread_pct, tick_stats)
            net_abs = abs(net_norm)

            # Confluence layer
            ext = self._extreme_flag(dfp, indp)
            sweep = self._liquidity_sweep(dfp, indp)
            div = self._divergence(dfp, indp)
            
            # Dynamic Round Number Check (User Feedback)
            atr_val = float(indp.get("atr", 0.0) or 0.0)
            near_rn = self._near_round(close_p, indp, atr_val, spread_pct)

            ok, why = self._confirm_layer(ext, sweep, div, near_rn)
            if not ok:
                return self._neutral(
                    sym,
                    ["extreme_guard"] + why,
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=_as_regime(adapt.get("regime")),
                    trade_blocked=True,
                )

            # Controlled boosts / penalties (no fake 100%)
            if (sweep or div != "none") and net_abs >= 0.05:
                if net_abs >= 0.15:  # Ислоҳ: Аз 0.12 ба 0.15 барои boost дақиқтар
                    conf = min(92, int(conf * 1.12))  # Ислоҳ: Аз 1.10 ба 1.12 барои фоида беҳтар
                elif net_abs >= 0.08:
                    conf = min(88, int(conf * 1.06))
                else:
                    conf = min(85, int(conf * 1.03))

            if near_rn and not (sweep or div != "none"):
                if net_abs >= 0.15:
                    conf = max(0, int(conf * 0.95))
                elif net_abs >= 0.10:
                    conf = max(0, int(conf * 0.85))
                else:
                    conf = max(0, int(conf * 0.75))

            # Strict confidence caps by strength
            if net_abs < 0.08:
                conf = min(conf, 80)
            elif net_abs < 0.12:
                conf = min(conf, 88)
            elif net_abs < 0.18:
                conf = min(conf, 95)

            # Ultra-confidence must be earned: require strong strength OR confluence.
            # Prevents "fake 96%" spam in choppy conditions.
            has_confluence = bool(sweep) or (div != "none")
            if conf >= 90 and (not has_confluence) and net_abs < 0.20:
                conf = min(conf, 89)

            # Decide signal
            signal = "Neutral"
            blocked_by_htf = False

            conf_min = int(adapt.get("conf_min", 80) or 80)
            net_thr = float(getattr(self.cfg, "net_norm_signal_threshold", 0.10) or 0.10)
            min_net_norm_for_signal = float(net_thr)

            regime = str(adapt.get("regime", "trend") or "trend").lower()
            if regime.startswith("range"):
                net_thr += 0.03
                min_net_norm_for_signal = net_thr
                if not (near_rn or sweep or div != "none"):
                    return self._neutral(
                        sym,
                        ["range_no_confluence"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            if conf >= conf_min and net_abs >= min_net_norm_for_signal:
                if net_norm > net_thr and trend_ok_buy:
                    signal = "Buy"
                elif net_norm < -net_thr and trend_ok_sell:
                    signal = "Sell"
                else:
                    blocked_by_htf = True
            else:
                blocked_by_htf = True

            # 11) Advanced filters
            if signal != "Neutral":
                # NOISE FILTER (M5/M15 focus)
                # If market is choppy/noisy, block signal to avoid whipsaw.
                is_noisy, noise_reason = self._is_noisy(indp, indc)
                if is_noisy:
                     return self._neutral(
                        sym,
                        [noise_reason],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

                if not self._conformal_ok(dfp):
                    return self._neutral(
                        sym,
                        ["conformal_abstain"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            if signal in ("Buy", "Sell") and not self._gate_meta(signal, indp, dfp):
                return self._neutral(
                    sym,
                    ["meta_gate_block"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=_as_regime(adapt.get("regime")),
                    trade_blocked=True,
                )

            signal, conf = self._apply_filters(signal, conf, indp)

            # Risk-level emission gate (optional)
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
                            regime=_as_regime(adapt.get("regime")),
                            trade_blocked=True,
                        )
                except Exception:
                    return self._neutral(
                        sym,
                        ["emit_gate_error"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            if signal == "Neutral" and blocked_by_htf:
                conf = min(conf, max(0, conf_min - 1))

            # Debounce + stability (OPTIMIZED for Scalping)
            # Only debounce if we have a ACTIVE signal (Buy/Sell) to prevent spamming opens.
            # If last was Neutral, we allow instant entry.
            now_ms = time.monotonic() * 1000.0
            if self._last_signal != "Neutral":
                if (now_ms - self._last_decision_ms) < self._debounce_ms:
                    return self._neutral(
                        sym,
                        ["debounce"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                    )

            strong_conf_min = int(getattr(self.cfg, "strong_conf_min", 88) or 88)
            if signal == self._last_signal and abs(net_norm - self._last_net_norm) < self._stable_eps:
                if conf < strong_conf_min and net_abs < 0.12:
                    return self._neutral(
                        sym,
                        ["stable"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                    )

            self._last_decision_ms = now_ms
            self._last_net_norm = float(net_norm)
            self._last_signal = str(signal)

            # Final strict quality gate before planning
            if signal in ("Buy", "Sell"):
                min_net_threshold = float(getattr(self.cfg, "net_norm_signal_threshold", 0.10) or 0.10)
                if net_abs < min_net_threshold:
                    return self._neutral(
                        sym,
                        ["weak_net_norm"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

                if conf < conf_min:
                    return self._neutral(
                        sym,
                        ["low_confidence_final"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

                # ADX gate only for trend regime (range trades may have low ADX by design)
                if not regime.startswith("range"):
                    adx_p = float(indp.get("adx", 0.0) or 0.0)
                    adx_lo_p = float(getattr(self.cfg, "adx_trend_lo", 16.0) or 16.0)
                    if adx_p < adx_lo_p * 0.85:
                        return self._neutral(
                            sym,
                            ["weak_adx"],
                            t0,
                            spread_pct=spread_pct,
                            bar_key=bar_key,
                            regime=_as_regime(adapt.get("regime")),
                            trade_blocked=True,
                        )

                if spread_pct > float(self.sp.spread_limit_pct) * 1.3:
                    return self._neutral(
                        sym,
                        ["excessive_spread"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            reasons: List[str] = []
            phase = str(adapt.get("phase", "A") or "A")
            if sweep:
                reasons.append(f"sweep:{sweep}")
            if div != "none":
                reasons.append(f"div:{div}")
            if near_rn:
                reasons.append("near_rn")
            reasons.append(f"net:{net_norm:.3f}")
            reasons.append(f"phase:{phase}")

            # === DYNAMIC CONFIDENCE CAPS (Not static 96%) ===
            # Cap based on actual signal strength (net_abs)
            if net_abs < 0.10:
                conf = min(conf, 82)
            elif net_abs < 0.15:
                conf = min(conf, 88)
            elif net_abs < 0.20:
                conf = min(conf, 93)
            else:
                conf = min(conf, 96)  # Only strongest signals reach 96%

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
                ts = pd.to_datetime(t)
                return ts.isoformat()
            return str(t)
        except Exception:
            return "bar_err"

    def _signal_id(self, sym: str, tf: str, bar_key: str, signal: str) -> str:
        raw = f"{sym}|{tf}|{bar_key}|{signal}"
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"{sym}:{tf}:{signal}:{h}"

    def _is_noisy(self, indp: Dict[str, Any], indc: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Enhanced Noise Filter for M5/M15 focus (v2).
        
        Criteria (Combined):
        1. ADX < 16 on BOTH TFs (Extreme trendless)
        2. ATR Rel < threshold (Dead market)
        3. RSI indecision (45-55) + Low ADX on both TFs
        4. M5/M15 Trend Alignment conflict
        5. BB Squeeze (consolidation before breakout)
        """
        try:
            # === Primary TF checks ===
            adx_p = float(indp.get("adx", 25.0) or 25.0)
            adx_c = float(indc.get("adx", 25.0) or 25.0)
            
            atr_rel = float(indp.get("atr_rel", 0.001) or 0.001)
            min_vol_thresh = float(getattr(self.cfg, "atr_rel_lo", 0.00055) or 0.00055)
            
            rsi_p = float(indp.get("rsi", 50.0) or 50.0)
            rsi_c = float(indc.get("rsi", 50.0) or 50.0)
            
            # BB Width for squeeze detection
            bb_width = float(indp.get("bb_width", 0.0) or 0.0)
            bb_width_thresh = float(getattr(self.cfg, "bb_width_range_max", 0.003) or 0.003)
            
            # === 1. Extreme ADX Low on BOTH timeframes = No trend at all ===
            if adx_p < 16.0 and adx_c < 16.0:
                return True, "noise_adx_extreme_dual"

            # === 2. Dead Market: Very low volatility ===
            if atr_rel < min_vol_thresh * 0.7:
                return True, f"noise_dead_vol:{atr_rel:.5f}"

            # === 3. RSI Indecision + Low ADX (choppy market) ===
            # Both TFs showing RSI in middle zone AND low ADX = pure noise
            if adx_p < 20.0 and (44.0 < rsi_p < 56.0):
                if adx_c < 22.0 and (44.0 < rsi_c < 56.0):
                    return True, "noise_rsi_indecision"

            # === 4. BB Squeeze (consolidation) ===
            if bb_width > 0 and bb_width < bb_width_thresh * 0.5:
                if adx_p < 20.0:
                    return True, "noise_bb_squeeze"

            # === 5. M5/M15 Trend Alignment Conflict (NEW) ===
            trend_p = str(indp.get("trend", "")).lower()
            trend_c = str(indc.get("trend", "")).lower()
            
            if trend_p and trend_c:
                p_up = "up" in trend_p or "bull" in trend_p
                p_dn = "down" in trend_p or "bear" in trend_p
                c_up = "up" in trend_c or "bull" in trend_c
                c_dn = "down" in trend_c or "bear" in trend_c
                
                # If primary says up but confirm says down (or vice versa) = noise
                if (p_up and c_dn) or (p_dn and c_up):
                    return True, "noise_tf_conflict"

            return False, ""
        except Exception:
            return False, ""

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
            regime=_as_regime(regime),
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
            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                bal, eq = self.risk._account_snapshot()
            else:
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

        base_conf_min_raw = getattr(self.cfg, "min_confidence_signal", None)
        if base_conf_min_raw is None:
            base_conf_min_raw = getattr(self.cfg, "conf_min", 84)

        base_conf_min = float(base_conf_min_raw)
        base_conf_min = int(base_conf_min * 100) if base_conf_min <= 1.0 else int(base_conf_min)

        if phase == "A":
            conf_min = base_conf_min
        elif phase == "B":
            ultra_min = getattr(self.cfg, "ultra_confidence_min", None)
            if ultra_min is not None:
                ultra_val = float(ultra_min)
                ultra_val = int(ultra_val * 100) if ultra_val <= 1.0 else int(ultra_val)
                conf_min = max(ultra_val, base_conf_min + 8)
            else:
                conf_min = min(100, base_conf_min + 8)
        else:
            conf_min = min(100, base_conf_min + 15)

        try:
            atr_p = float(indp.get("atr", 0.0) or 0.0)
        except Exception:
            atr_p = 0.0

        return {
            "conf_min": int(conf_min),
            "sl_mult": float(getattr(self.cfg, "sl_atr_mult_trend", 1.35) or 1.35),
            "tp_mult": float(getattr(self.cfg, "tp_atr_mult_trend", 3.2) or 3.2),
            "trail_mult": float(getattr(self.cfg, "trail_atr_mult", 1.0) or 1.0),
            "w_mul": {"trend": 1.0, "momentum": 1.0, "meanrev": 1.0, "structure": 1.0, "volume": 1.0},
            "regime": "trend",
            "phase": phase,
            "atr": float(atr_p),
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

    def _near_round(self, price: float, indp: Dict[str, Any], atr: float = 0.0, spread_pct: float = 0.0) -> bool:
        try:
            # Dynamic tolerance (User Feedback: max(spread*1.5, ATR*0.15))
            tol = 0.0
            if atr > 0:
                tol = max(tol, atr * 0.15)
            if spread_pct > 0 and price > 0:
                spread_price = price * spread_pct
                tol = max(tol, spread_price * 1.5)
            
            # Fallback to minimal if calculation failed (to prevent zero tolerance)
            if tol <= 0:
                tol = price * 0.0005 # 5bps default

            # Round Numbers: 100s, 50s
            rem_100 = price % 100.0
            dist_100 = min(rem_100, 100.0 - rem_100)
            
            rem_50 = price % 50.0
            dist_50 = min(rem_50, 50.0 - rem_50)

            if dist_100 < tol or dist_50 < (tol * 0.8):
                # Only trust if feature engine also flags it (hybrid) or if feature engine absent
                if hasattr(self.features, "round_number_confluence"):
                    # Check feature logic too, but accept our dynamic override
                    return True
                return True
                
            return False
        except Exception:
            return False

    def _refresh_weights_if_needed(self) -> None:
        w = getattr(self.cfg, "weights", {}) or {}
        sig = (
            float(w.get("trend", 0.50)),  # Ислоҳ: Вазнҳои нав
            float(w.get("momentum", 0.25)),
            float(w.get("meanrev", 0.10)),
            float(w.get("structure", 0.10)),
            float(w.get("volume", 0.05)),
        )
        if sig == self._w_sig:
            return
        vec = np.array(sig, dtype=np.float64)
        s = float(np.sum(vec))
        self._w_base = (vec / s) if s > 0 else np.array([0.50, 0.25, 0.10, 0.10, 0.05], dtype=np.float64)  # Ислоҳ: Вазнҳои нав истифода мешаванд
        self._w_sig = sig

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
            self._refresh_weights_if_needed()

            w_mul = adapt.get("w_mul") or {}
            mul_vec = np.array(
                [
                    float(w_mul.get("trend", 1.0)),
                    float(w_mul.get("momentum", 1.0)),
                    float(w_mul.get("meanrev", 1.0)),
                    float(w_mul.get("structure", 1.0)),
                    float(w_mul.get("volume", 1.0)),
                ],
                dtype=np.float64,
            )
            weights = self._w_base * mul_vec
            s = float(np.sum(weights))
            weights = weights / s if s > 0 else self._w_base

            regime = str(adapt.get("regime", "trend") or "trend")

            scores = np.zeros(5, dtype=np.float64)
            scores[0] = float(self._trend_score(indp, indc, indl, regime))
            scores[1] = float(self._momentum_score(indp, indc))
            scores[2] = float(self._meanrev_score(indp, regime))
            scores[3] = float(self._structure_score(indp))

            vol_sc = float(indp.get("z_vol", indp.get("z_volume", 0.0)) or 0.0)  # Ислоҳ: Истифодаи z_vol барои волум беҳтар (тилло волатил аст)
            scores[4] = float(np.tanh(vol_sc / 3.0))

            net_norm = float(np.sum(scores * weights))

            conf_bias = float(getattr(self.cfg, "confidence_bias", 50.0) or 50.0)
            conf_gain = float(getattr(self.cfg, "confidence_gain", 70.0) or 70.0)

            net_abs = abs(net_norm)
            if net_abs < 0.15:
                max_base = 65.0
            elif net_abs < 0.25:
                max_base = 75.0
            elif net_abs < 0.40:
                max_base = 82.0
            elif net_abs < 0.60:
                max_base = 88.0
            elif net_abs < 0.75:
                max_base = 92.0
            elif net_abs < 0.88:
                max_base = 94.0
            else:
                max_base = 96.0

            conf_gain_adj = conf_gain * 0.75  # Slightly reduced gain to separate weak/strong better
            base_conf = conf_bias + (net_abs * conf_gain_adj)
            conf = int(_clamp(base_conf, 10.0, max_base))
            
            # Volatility filter: reduce confidence if too explosive (risk of slippage/whipsaw)
            if float(getattr(tick_stats, "volatility", 0.0)) > 60.0:
                conf = max(0, conf - 15)
            
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
                elif close_p < ema50_l * 0.995:
                    sc -= 1.2
                elif close_p < ema50_l * 1.001:
                    sc -= 0.8

            if regime == "trend":
                ema_s, ema_m = self._ema_s_m(indp, close_p)
                if close_p > ema_s > ema_m:
                    sc += 0.5
                elif close_p < ema_s < ema_m:
                    sc -= 0.7
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
            sweep = str(indp.get("liquidity_sweep", "") or "").strip().lower()
            if sweep == "bull":
                sc += 1.0
            elif sweep == "bear":
                sc -= 1.0

            if bool(indp.get("fvg_bull", False)):
                sc += 0.5
            if bool(indp.get("fvg_bear", False)):
                sc -= 0.5

            div = str(indp.get("rsi_div", "none") or "none").strip().lower()
            if div in ("bull", "bullish"):
                sc += 0.5
            elif div in ("bear", "bearish"):
                sc -= 0.5

            macd_div = str(indp.get("macd_div", "none") or "none").strip().lower()
            if macd_div in ("bull", "bullish"):
                sc += 0.4
            elif macd_div in ("bear", "bearish"):
                sc -= 0.4

            ob = str(indp.get("order_block", "") or "").strip().lower()
            if ob == "bull_ob":
                sc += 0.4
            elif ob == "bear_ob":
                sc -= 0.4
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

            base = rr[-(W + 50) : -50] if rr.size > (W + 60) else rr[:-5]
            if base.size < 30:
                return True

            q = float(np.quantile(base, qv))
            return bool(rr[-1] <= q)
        except Exception:
            return True

    def _gate_meta(self, side: str, ind: Dict[str, Any], dfp: pd.DataFrame) -> bool:
        """
        Meta gate OPTIMIZED for Scalping 1-15 min (v4).
        Relaxed thresholds to allow more signals while maintaining quality.
        """
        try:
            _ = ind
            h_bars = int(getattr(self.cfg, "meta_h_bars", 6) or 6)
            R = float(getattr(self.cfg, "meta_barrier_R", 0.50) or 0.50)  # Use config value (0.50)
            tc_bps = float(getattr(self.cfg, "tc_bps", 1.0) or 1.0)  # Use config value (1.0)

            if dfp is None or len(dfp) < (h_bars + 30):
                return True  # Fail-open for insufficient data

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)

            atr = _atr(h, l, c, 14)
            atr_val = float(safe_last(atr, 1.0) or 1.0)
            if atr_val <= 0:
                return True  # Fail-open

            sign = 1.0 if side == "Buy" else -1.0
            # Optimized lookback for scalping: wider window for better statistics
            start_i = max(1, len(c) - h_bars - 50)  # Increased from 40
            end_i = max(start_i + 1, len(c) - h_bars - 1)

            idx = np.arange(start_i, end_i, dtype=np.int64)
            if idx.size < 3:  # Reduced from 5 for scalping
                return True  # Fail-open for insufficient samples

            moves = (c[idx + h_bars] - c[idx]) * sign
            wins = int(np.sum(moves >= (R * atr_val)))
            total = int(idx.size)

            if total < 3:
                return True  # Fail-open

            hitrate = wins / total
            edge_needed = (tc_bps / 10000.0) * 1.5  # Reduced from 2.0 for scalping
            threshold = 0.40 + edge_needed  # Reduced from 0.45 for scalping
            
            # OPTIMIZED for scalping: multiple fail-open conditions
            # 1. If hitrate is above threshold - allow
            if hitrate >= threshold:
                return True
            
            # 2. If hitrate is close to threshold (within 20% margin) - allow
            if hitrate >= (threshold - 0.20):  # Increased margin from 0.15
                return True
            
            # 3. If hitrate is above 30% (was 35%) - allow for scalping
            if hitrate >= 0.30:
                return True
            
            # 4. If we have very few samples but hitrate > 25% - allow
            if total < 10 and hitrate >= 0.25:
                return True
                
            # 5. Default: block only if hitrate is very low
            return bool(hitrate >= 0.25)  # Minimum fail-open threshold
        except Exception:
            return True  # Fail-open on any error

    def _apply_filters(self, signal: str, conf: int, ind: Dict[str, Any]) -> Tuple[str, int]:
        """
        OPTIMIZED filters for Scalping 1-15 min.
        Reduced penalties to allow more signals while maintaining quality.
        """
        try:
            if signal == "Neutral":
                return "Neutral", int(_clamp(float(conf), 0.0, 97.0))  # Increased cap to 97

            atr = float(ind.get("atr", 0.0) or 0.0)
            body = float(ind.get("body", 0.0) or 0.0)
            if body <= 0.0:
                try:
                    body = abs(float(ind.get("close", 0.0) or 0.0) - float(ind.get("open", 0.0) or 0.0))
                except Exception:
                    body = 0.0

            # Reduced penalty for small body (scalping bars can be small)
            min_body_k = float(getattr(self.cfg, "min_body_pct_of_atr", 0.09) or 0.09)
            if atr > 0 and body < (min_body_k * atr):
                conf = max(0, conf - 15)  # Reduced penalty from 25 to 15

            close_p = float(ind.get("close", 0.0) or 0.0)
            ema_s, ema_m = self._ema_s_m(ind, close_p)

            # Increased boost for EMA alignment (scalping benefits from alignment)
            if signal == "Buy" and close_p > ema_s > ema_m:
                conf += 12  # Increased from 10
            elif signal == "Sell" and close_p < ema_s < ema_m:
                conf += 12  # Increased from 10

            conf = int(_clamp(float(conf), 0.0, 97.0))  # Increased cap to 97
            return signal, conf
        except Exception:
            conf = int(_clamp(float(conf), 0.0, 97.0))
            return signal, conf

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
            regime = _as_regime(adapt.get("regime"))

            entry_val = sl_val = tp_val = lot_val = None

            if signal in ("Buy", "Sell") and execute:
                zones = self.feed.micro_price_zones(book)
                tick_vol = float(getattr(tick_stats, "volatility", 0.0) or 0.0)
                open_pos, unreal_pl = self._position_context(sym)

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

                plan = self.risk.plan_order(
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

                if not plan.ok:
                    return SignalResult(
                        symbol=sym,
                        signal="Neutral",
                        confidence=0,
                        regime=regime,
                        reasons=[f"plan_reject:{plan.reason}"],
                        spread_bps=float(spread_pct) * 10000.0,
                        latency_ms=float(comp_ms),
                        timestamp=self.feed.now_local().isoformat(),
                        signal_id=self._signal_id(sym, str(self.sp.tf_primary), bar_key, "Neutral"),
                        trade_blocked=True,
                    )
                
                entry_val = plan.entry
                sl_val = plan.sl
                tp_val = plan.tp
                lot_val = plan.lot

            return SignalResult(
                symbol=sym,
                signal=signal,
                confidence=int(_clamp(float(conf), 0.0, 96.0)),
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
                regime=_as_regime(adapt.get("regime")),
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

            if bool(getattr(self.cfg, "ignore_external_positions", False)):
                try:
                    magic = int(getattr(self.cfg, "magic", 777001) or 777001)
                except Exception:
                    magic = 777001
                positions = [p for p in positions if int(getattr(p, "magic", 0) or 0) == magic]

            open_pos = int(len(positions))
            unreal_pl = float(sum(float(p.profit or 0.0) for p in positions))
            return open_pos, unreal_pl
        except Exception:
            return 0, 0.0


__all__ = ["SignalEngine", "SignalResult"]
