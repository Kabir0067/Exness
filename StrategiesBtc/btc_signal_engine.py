from __future__ import annotations

import hashlib
import logging
import time
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None  # type: ignore

from config_btc import EngineConfig, SymbolParams
from DataFeed.btc_market_feed import MarketFeed, TickStats
from mt5_client import MT5_LOCK, ensure_mt5
from StrategiesBtc.btc_indicators import Classic_FeatureEngine, safe_last
from StrategiesBtc.btc_risk_management import RiskManager
from log_config import LOG_DIR as LOG_ROOT, get_log_path

# ============================================================
# ERROR-only rotating logging (isolated)
# ============================================================
LOG_DIR = LOG_ROOT

log = logging.getLogger("signal_btc")
log.setLevel(logging.ERROR)
log.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
    fh = RotatingFileHandler(
        filename=str(get_log_path("signal_engine_btc.log")),
        maxBytes=int(5_242_880),  # 5MB
        backupCount=int(5),
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"))
    log.addHandler(fh)


# ============================================================
# helpers
# ============================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _atr_np(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder ATR fallback without TA-Lib.
    Returns len(c) array with NaNs until period-1 where possible.
    """
    n = int(len(c))
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    if n == 1:
        out[0] = float(h[0] - l[0])
        return out

    period = max(1, int(period))

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
            return talib.ATR(h, l, c, int(period))  # type: ignore[attr-defined]
        except Exception:
            return _atr_np(h, l, c, int(period))
    return _atr_np(h, l, c, int(period))


def _side_norm(side: str) -> str:
    s = str(side or "").strip().lower()
    if s in ("buy", "long", "b", "1"):
        return "Buy"
    if s in ("sell", "short", "s", "-1"):
        return "Sell"
    return "Buy" if "buy" in s else "Sell" if "sell" in s else str(side)


# ============================================================
# result
# ============================================================
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


# ============================================================
# engine
# ============================================================
class SignalEngine:
    """
    Planner-only SignalEngine:
      - execute=True -> fills entry/sl/tp/lot using RiskManager.plan_order()
      - never sends orders itself
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

        # DEBUG LOGGING removed - was causing log noise at startup
        # Configuration values are available via self.cfg if needed for debugging

        self._last_decision_ms = 0.0
        self._last_net_norm = 0.0
        self._last_signal = "Neutral"
        self._current_drawdown = 0.0
        self._last_meta_gate_log_ts = 0.0  # For throttled diagnostic logging

        self._debounce_ms = float(getattr(self.cfg, "decision_debounce_ms", 150.0) or 150.0)
        self._stable_eps = float(getattr(self.cfg, "signal_stability_eps", 0.03) or 0.03)

        # ensemble weights defaults (safe)
        if not hasattr(self.cfg, "weights") or not isinstance(getattr(self.cfg, "weights", None), dict):
            setattr(
                self.cfg,
                "weights",
                {"trend": 0.55, "momentum": 0.27, "meanrev": 0.10, "structure": 0.05, "volume": 0.03},
            )

        # cache base weights for speed
        self._w_sig: Tuple[Tuple[str, float], ...] = tuple()
        self._w_base = np.array([0.55, 0.27, 0.10, 0.05, 0.03], dtype=np.float64)

    # --------------------------- public
    def compute(self, execute: bool = False) -> SignalResult:
        """
        execute=False: returns signal only
        execute=True : returns signal + (entry/sl/tp/lot) plan via RiskManager.plan_order
        """
        t0 = time.perf_counter()
        sym = self._symbol()

        try:
            # MT5 quote -> feed risk exec monitor (breaker)
            bid, ask = self._tick_bid_ask(sym)
            if bid and ask:
                self.risk.on_quote(bid, ask)

            # 1) Rates (MTF)
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

            # 2) Context
            spread_pct = float(self.feed.spread_pct() or 0.0)
            last_age = float(self.feed.last_bar_age(dfp) or 0.0)
            atr_lb = int(getattr(self.cfg, "atr_percentile_lookback", 400) or 400)
            atr_pct = float(self.risk.atr_percentile(dfp, atr_lb) or 0.0)
            tick_stats: TickStats = self.feed.tick_stats(dfp)
            ingest_ms = (time.perf_counter() - t0) * 1000.0

            # 3) Early Phase C check (skip analysis if hard stop active - save CPU)
            self.risk.evaluate_account_state()  # Update phase
            if self.risk.current_phase == "C" and not bool(getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False)):
                return self._neutral(
                    sym,
                    ["phase_c_protect"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            # 4) Hard market gate (BTC is 24/7, but can be close-only/disabled/maintenance)
            if not self.risk.market_open_24_5():
                return self._neutral(
                    sym,
                    ["market_blocked"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            # 4) Guards (risk-level)
            in_session = self._in_active_session()  # optional policy gate (cfg.active_sessions)
            drawdown_exceeded = self._check_drawdown()
            latency_flag = bool(self.risk.latency_cooldown())

            decision = self.risk.guard_decision(
                spread_pct=spread_pct,
                tick_ok=bool(getattr(tick_stats, "ok", True)),
                tick_reason=str(getattr(tick_stats, "reason", "") or ""),
                ingest_ms=float(ingest_ms),
                last_bar_age=float(last_age),
                in_session=bool(in_session),
                drawdown_exceeded=bool(drawdown_exceeded),
                latency_cooldown=bool(latency_flag),
                tz=getattr(self.feed, "tz", None),
            )
            if not getattr(decision, "allowed", False):
                reasons = list(getattr(decision, "reasons", []) or ["guard_block"])
                return self._neutral(
                    sym,
                    reasons[:25],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            # 4.1) BTC Spread Gate (USD-based)
            # BTC has wider spreads, especially on weekends
            # Scalping 1-15min requires higher tolerance
            if bid and ask:
                spread_usd = float(ask - bid)
                
                # Dynamic spread limit based on price level
                # For BTC at ~$87000, $25 spread = 0.03% (acceptable for scalping)
                max_spread_usd = 25.0  # Increased from $5 for realistic BTC trading
                
                if spread_usd > max_spread_usd:
                     return self._neutral(
                        sym,
                        [f"spread_gate_usd:{spread_usd:.2f}"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        trade_blocked=True,
                    )

            # 5) Indicators (MTF)
            tf_data = {self.sp.tf_primary: dfp, self.sp.tf_confirm: dfc, self.sp.tf_long: dfl}
            indicators = self.features.compute_indicators(tf_data)
            if not indicators:
                return self._neutral(
                    sym,
                    ["no_indicators"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    trade_blocked=True,
                )

            indp = indicators.get(self.sp.tf_primary, {})
            indc = indicators.get(self.sp.tf_confirm, {})
            indl = indicators.get(self.sp.tf_long, {})

            # 6) Adaptive params
            adapt = self._get_adaptive_params(indp, indl, atr_pct)

            # 7) Orderbook (for micro zones)
            try:
                book = self.feed.get_orderbook(sym) if hasattr(self.feed, "get_orderbook") else {}
            except Exception:
                book = {}

            # 8) Ensemble score (net_norm + confidence)
            net_norm, conf = self._ensemble_score(
                indp, indc, indl, book, adapt, spread_pct, tick_stats
            )

            # 9) Pattern detection
            sweep = self._liquidity_sweep(dfp, indp)
            div = self._divergence(dfp, indp)
            near_rn = bool(indp.get("near_rn", False))

            # 10) Signal determination
            signal = "Neutral"
            conf_min = int(adapt.get("conf_min", getattr(self.cfg, "conf_min", 60) or 60))
            net_thr = float(getattr(self.cfg, "net_norm_signal_threshold", 0.08) or 0.08)
            min_net_norm_for_signal = float(net_thr)

            close_p = float(indp.get("close", 0.0) or 0.0)
            ema_s, ema_m = self._ema_s_m(indp, close_p)
            
            # === RELAXED TREND CHECKS FOR CRYPTO VOLATILITY ===
            # Add tolerance for crypto's fast-moving nature
            ema_tolerance = 0.0002  # 0.02% tolerance
            trend_ok_buy = close_p > ema_s * (1 - ema_tolerance) > ema_m * (1 - ema_tolerance)
            trend_ok_sell = close_p < ema_s * (1 + ema_tolerance) < ema_m * (1 + ema_tolerance)
            
            # Allow BREAKOUT trades when ADX is high (strong trend developing)
            adx_p = float(indp.get("adx", 0.0) or 0.0)
            if adx_p > 30.0:  # Strong trend
                if net_norm > 0.12:  # Bullish breakout
                    trend_ok_buy = True
                elif net_norm < -0.12:  # Bearish breakout
                    trend_ok_sell = True

            net_norm_abs = abs(net_norm)

            if conf >= conf_min and net_norm_abs >= min_net_norm_for_signal:
                if net_norm > net_thr:
                    if trend_ok_buy:
                        signal = "Buy"
                elif net_norm < -net_thr:
                    if trend_ok_sell:
                        signal = "Sell"

            # 11) Advanced filters
            if signal != "Neutral" and not self._conformal_ok(dfp):
                return self._neutral(
                    sym,
                    ["conformal_abstain"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=str(adapt.get("regime")),
                    trade_blocked=True,
                )

            if signal in ("Buy", "Sell") and not self._gate_meta(signal, indp, dfp):
                return self._neutral(
                    sym,
                    ["meta_gate_block"],
                    t0,
                    spread_pct=spread_pct,
                    bar_key=bar_key,
                    regime=str(adapt.get("regime")),
                    trade_blocked=True,
                )

            # 12) Apply filters
            signal, conf = self._apply_filters(signal, int(conf), indp)

            # 13) Debounce + stability (OPTIMIZED for Scalping)
            now_ms = time.monotonic() * 1000.0
            if self._last_signal != "Neutral":
                if (now_ms - self._last_decision_ms) < self._debounce_ms:
                    return self._neutral(
                        sym,
                        ["debounce"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=str(adapt.get("regime")),
                    )

            strong_conf_min = int(getattr(self.cfg, "strong_conf_min", 85) or 85)
            if signal == self._last_signal and abs(net_norm - self._last_net_norm) < self._stable_eps:
                if int(conf) < strong_conf_min and abs(net_norm) < 0.15:
                    return self._neutral(
                        sym,
                        ["stable"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=str(adapt.get("regime")),
                        confidence=int(conf),
                    )

            self._last_decision_ms = now_ms
            self._last_net_norm = float(net_norm)
            self._last_signal = str(signal)

            # === DYNAMIC CONFIDENCE CAPS (Same as XAU) ===
            # Cap based on actual signal strength (net_abs)
            if net_norm_abs < 0.10:
                conf = min(conf, 82)
            elif net_norm_abs < 0.15:
                conf = min(conf, 88)
            elif net_norm_abs < 0.20:
                conf = min(conf, 93)
            else:
                conf = min(conf, 96)  # Only strongest signals reach 96%

            # 14) Finalize (plan only)
            reasons: List[str] = []
            if sweep:
                reasons.append(f"sweep:{sweep}")
            if div != "none":
                reasons.append(f"div:{div}")
            if near_rn:
                reasons.append("near_rn")
            reasons.append(f"net:{net_norm:.3f}")
            reasons.append(f"conf:{int(conf)}")

            return self._finalize(
                sym=sym,
                signal=signal,
                conf=int(conf),
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

    # --------------------------- core helpers
    def _symbol(self) -> str:
        return self.sp.resolved or self.sp.base

    def _tick_bid_ask(self, sym: str) -> Tuple[float, float]:
        try:
            ensure_mt5()
            with MT5_LOCK:
                t = mt5.symbol_info_tick(sym)
            if not t:
                return 0.0, 0.0
            bid = float(getattr(t, "bid", 0.0) or 0.0)
            ask = float(getattr(t, "ask", 0.0) or 0.0)
            if bid <= 0 or ask <= 0 or ask < bid:
                return 0.0, 0.0
            return bid, ask
        except Exception:
            return 0.0, 0.0

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
                tt = pd.to_datetime(t)
                return tt.isoformat()
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
        confidence: int = 0,
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
            confidence=confidence,
            regime=regime,
            reasons=(reasons or ["neutral"])[:25],
            spread_bps=None if spread_pct is None else float(spread_pct) * 10000.0,
            latency_ms=float(comp_ms),
            timestamp=self.feed.now_local().isoformat(),
            signal_id=sid,
            trade_blocked=bool(trade_blocked),
        )

    def _in_active_session(self) -> bool:
        """
        Optional policy gate: cfg.active_sessions = [(start_hour, end_hour), ...] in LOCAL time.
        If absent -> always True (BTC is 24/7).
        """
        try:
            if bool(getattr(self.cfg, "ignore_sessions", False)):
                return True
            sessions = list(getattr(self.cfg, "active_sessions", []) or [])
            if not sessions:
                return True
            h = int(self.feed.now_local().hour)
            return any(int(s) <= h < int(e) for s, e in sessions)
        except Exception:
            return True

    def _check_drawdown(self) -> bool:
        """
        Prefer RiskManager cached drawdown; fallback to direct MT5 query if needed.
        """
        try:
            self.risk.evaluate_account_state()
            dd = float(getattr(self.risk, "_current_drawdown", 0.0) or 0.0)
            self._current_drawdown = dd
            return dd >= float(getattr(self.cfg, "max_drawdown", 0.09) or 0.09)
        except Exception:
            pass

        try:
            ensure_mt5()
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
            base_conf_min_raw = getattr(self.cfg, "conf_min", 74)
        base_conf_min = float(base_conf_min_raw)
        if base_conf_min <= 1.0:
            base_conf_min = int(base_conf_min * 100)
        else:
            base_conf_min = int(base_conf_min)

        if phase == "A":
            conf_min = base_conf_min
        elif phase == "B":
            ultra_min = getattr(self.cfg, "ultra_confidence_min", None)
            if ultra_min is not None:
                ultra_val = float(ultra_min)
                if ultra_val <= 1.0:
                    ultra_val = int(ultra_val * 100)
                else:
                    ultra_val = int(ultra_val)
                conf_min = max(ultra_val, base_conf_min + 10)
            else:
                conf_min = min(96, base_conf_min + 10)
        else:
            conf_min = min(96, base_conf_min + 15)

        # Check if fixed_volume is set - if so, prefer fixed volume over dynamic sizing
        fixed_vol = float(getattr(self.cfg, "fixed_volume", 0.0) or 0.0)
        use_dynamic = True  # Default to dynamic sizing
        if fixed_vol > 0:
            # If fixed_volume is set, we'll use it as fallback, but still try dynamic first
            # This allows dynamic sizing to work when it can, but falls back to fixed_volume
            use_dynamic = True  # Still try dynamic, but will fallback to fixed_vol if it fails
        
        return {
            "conf_min": int(conf_min),
            "sl_mult": float(getattr(self.cfg, "sl_atr_mult_trend", 1.35) or 1.35),
            "tp_mult": float(getattr(self.cfg, "tp_atr_mult_trend", 3.2) or 3.2),
            "trail_mult": float(getattr(self.cfg, "trail_atr_mult", 1.0) or 1.0),
            "w_mul": {"trend": 1.0, "momentum": 1.0, "meanrev": 1.0, "structure": 1.0, "volume": 1.0},
            "regime": "trend",
            "phase": phase,
            "use_dynamic_sizing": bool(use_dynamic),
            "fixed_volume": float(fixed_vol),
        }

    # --------------------------- confluence helpers
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

    def _confirm_layer(self, ext: bool, sweep: str, div: str, near_rn: bool) -> Tuple[bool, List[str]]:
        try:
            if not ext:
                return True, []
            has_bear = (sweep == "bear") or (div == "bearish")
            has_bull = (sweep == "bull") or (div == "bullish")
            if not has_bear and not has_bull and not near_rn:
                return False, ["extreme_needs_confirmation"]
            return True, []
        except Exception:
            return True, []

    # --------------------------- ensemble scoring (FIXED)
    def _get_base_weights(self) -> np.ndarray:
        """
        Cache normalized base weights from cfg.weights. Keeps structure, improves speed.
        """
        try:
            w = getattr(self.cfg, "weights", {}) or {}
            sig = (
                ("trend", float(w.get("trend", 0.55))),
                ("momentum", float(w.get("momentum", 0.27))),
                ("meanrev", float(w.get("meanrev", 0.10))),
                ("structure", float(w.get("structure", 0.05))),
                ("volume", float(w.get("volume", 0.03))),
            )
            if sig != self._w_sig:
                arr = np.array([sig[0][1], sig[1][1], sig[2][1], sig[3][1], sig[4][1]], dtype=np.float64)
                s = float(np.sum(arr))
                if s <= 0:
                    arr = np.array([0.55, 0.27, 0.10, 0.05, 0.03], dtype=np.float64)
                    s = float(np.sum(arr))
                self._w_base = arr / max(1e-12, s)
                self._w_sig = sig
            return self._w_base
        except Exception:
            return np.array([0.55, 0.27, 0.10, 0.05, 0.03], dtype=np.float64)

    def _conf_from_strength(self, net_abs: float, spread_pct: float, tick_stats: TickStats) -> float:
        conf_bias = float(getattr(self.cfg, "confidence_bias", 52.0) or 52.0)

        # AFTER (FIX): respect config; only safety clamp
        gain_cfg = float(getattr(self.cfg, "confidence_gain", 70.0) or 70.0)
        conf_gain = float(_clamp(gain_cfg, 30.0, 140.0))

        net_abs = float(_clamp(net_abs, 0.0, 1.0))
        base = conf_bias + (net_abs * conf_gain)

        # spread penalty (same as before)
        try:
            sp_lim = float(getattr(self.sp, "spread_limit_pct", 0.0) or 0.0)
            if sp_lim > 0.0 and spread_pct > 0.0:
                ratio = spread_pct / max(1e-12, sp_lim)
                if ratio > 0.75:
                    base -= min(18.0, (ratio - 0.75) * 24.0)
        except Exception:
            pass

        # tick penalty (same as before)
        try:
            ok = bool(getattr(tick_stats, "ok", True))
            if not ok:
                base -= 14.0
        except Exception:
            pass

        return float(_clamp(base, 0.0, 96.0))

    def _cap_conf_by_strength(self, conf: int, net_abs: float) -> int:
        """
        Single consistent cap: prevents weak signals from showing high %,
        without killing valid Buy opportunities.
        FIXED: More granular caps to allow dynamic confidence variation.
        """
        try:
            if net_abs < 0.06:
                cap = 72
            elif net_abs < 0.10:
                cap = 80
            elif net_abs < 0.14:
                cap = 86
            elif net_abs < 0.18:
                cap = 90  # Reduced from 92 to allow more variation
            elif net_abs < 0.22:
                cap = 93  # New tier for very strong signals
            elif net_abs < 0.30:
                cap = 95  # New tier for extremely strong signals
            else:
                cap = 96  # Only the strongest signals reach 96
            return int(_clamp(float(min(int(conf), int(cap))), 0.0, 96.0))
        except Exception:
            return int(_clamp(float(conf), 0.0, 96.0))

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
            _ = (book,)

            # weights (cached base * adaptive multipliers)
            w_base = self._get_base_weights()
            w_mul = adapt.get("w_mul") or {}
            mul = np.array(
                [
                    float(w_mul.get("trend", 1.0)),
                    float(w_mul.get("momentum", 1.0)),
                    float(w_mul.get("meanrev", 1.0)),
                    float(w_mul.get("structure", 1.0)),
                    float(w_mul.get("volume", 1.0)),
                ],
                dtype=np.float64,
            )
            weights = w_base * mul
            s = float(np.sum(weights))
            if s <= 0:
                weights = w_base
            else:
                weights = weights / s

            regime = str(adapt.get("regime", "trend") or "trend")

            # scores (keep structure)
            scores = np.zeros(5, dtype=np.float64)
            scores[0] = float(self._trend_score(indp, indc, indl, regime))
            scores[1] = float(self._momentum_score(indp, indc))
            scores[2] = float(self._meanrev_score(indp, regime))
            scores[3] = float(self._structure_score(indp))

            z_vol = float(indp.get("z_vol", indp.get("z_volume", 0.0)) or 0.0)
            scores[4] = float(np.tanh(z_vol / 3.0))

            net_norm = float(np.dot(scores, weights))
            net_abs = abs(net_norm)

            # FIXED confidence
            base_conf = self._conf_from_strength(net_abs, float(spread_pct), tick_stats)
            conf = int(_clamp(base_conf, 10.0, 96.0))

            # strength caps (consistent)
            conf = self._cap_conf_by_strength(conf, net_abs)

            # DEBUG LOGGING removed - was causing excessive log spam (every ~50ms)
            # If needed for debugging, add throttling (e.g., log only once per second or on value change)

            return float(net_norm), int(conf)
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

    # --------------------------- meta gates
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
        Meta gate: checks if recent price movements show edge for the given direction.
        Returns True if signal should be allowed, False if blocked.
        
        OPTIMIZED FOR MEDIUM SCALPING (1-15 min):
        - Less strict thresholds
        - Fail-open approach for BTC volatility
        """
        try:
            _ = ind
            h_bars = int(getattr(self.cfg, "meta_h_bars", 4) or 4)
            R = float(getattr(self.cfg, "meta_barrier_R", 0.30) or 0.30)
            tc_bps = float(getattr(self.cfg, "tc_bps", 0.8) or 0.8)

            if dfp is None or len(dfp) < (h_bars + 25):
                return True

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)

            atr = _atr(h, l, c, 14)
            atr_val = float(safe_last(atr, 1.0) or 1.0)
            if atr_val <= 0:
                return True

            sign = 1.0 if _side_norm(side) == "Buy" else -1.0
            wins = 0
            total = 0

            # Look back for samples
            lookback_window = max(40, h_bars * 10)
            start_i = max(1, len(c) - lookback_window)
            end_i = max(start_i + 1, len(c) - h_bars - 1)
            
            if end_i <= start_i or (end_i - start_i) < 5:
                return True  # Fail-open for scalping

            for i in range(start_i, end_i):
                if (i + h_bars) >= len(c):
                    continue
                move = (c[i + h_bars] - c[i]) * sign
                if move >= (R * atr_val):
                    wins += 1
                total += 1

            if total < 5:  # Reduced from 10 for BTC scalping
                return True

            hitrate = wins / total
            edge_needed = (tc_bps / 10000.0) * 2.0  # Reduced from 3.0
            threshold = 0.45 + edge_needed  # Reduced from 0.50
            
            # === OPTIMIZED FOR BTC SCALPING ===
            # Much wider margin (15%) for BTC volatility
            if hitrate >= (threshold - 0.15):
                return True
            
            # If hitrate is above 35%, allow signal (BTC can be choppy)
            if hitrate >= 0.35:
                return True
            
            result = bool(hitrate >= threshold)
            
            if not result:
                now = time.time()
                if (now - self._last_meta_gate_log_ts) >= 30.0:
                    log.warning(
                        "meta_gate_block | side=%s hitrate=%.3f threshold=%.3f h_bars=%d R=%.2f tc_bps=%.2f total=%d wins=%d",
                        side, hitrate, threshold, h_bars, R, tc_bps, total, wins
                    )
                    self._last_meta_gate_log_ts = now
            
            return result
        except Exception as exc:
            log.error("_gate_meta exception: %s | tb=%s", exc, traceback.format_exc())
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
                conf = conf + 10
            if signal == "Sell" and close_p < ema_s < ema_m:
                conf = conf + 10

            conf = int(_clamp(float(conf), 0.0, 96.0))
            return signal, conf
        except Exception:
            conf = int(_clamp(float(conf), 0.0, 96.0))
            return signal, conf

    # --------------------------- finalize / planning
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
                trade_dec = self.risk.can_trade(float(conf) / 100.0, signal)
                if not trade_dec.allowed:
                    return self._neutral(
                        sym,
                        ["risk_trade_block"] + (trade_dec.reasons[:20] if trade_dec.reasons else []),
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=regime,
                        trade_blocked=True,
                    )

                zones = self.feed.micro_price_zones(book)
                tick_vol = float(getattr(tick_stats, "volatility", 0.0) or 0.0)
                open_pos, unreal_pl = self._position_context(sym)

                entry_val, sl_val, tp_val, lot_val = self.risk.plan_order(
                    signal,
                    float(conf) / 100.0,
                    indp,
                    adapt,
                    entry=None,
                    ticks=tick_stats,
                    zones=zones,
                    tick_volatility=tick_vol,
                    open_positions=open_pos,
                    max_positions=int(getattr(self.cfg, "max_positions", 3) or 3),
                    unrealized_pl=float(unreal_pl),
                )

                if entry_val is None or sl_val is None or tp_val is None or lot_val is None:
                    return self._neutral(
                        sym,
                        ["plan_order_failed"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=regime,
                        trade_blocked=True,
                    )

                # SIDE-EFFECT REMOVED: Moved to PortfolioEngine._enqueue_order
                # self.risk.register_signal_emitted()

            return SignalResult(
                symbol=sym,
                signal=signal,
                confidence=int(_clamp(float(conf), 0.0, 96.0)),
                regime=regime,
                reasons=(reasons or ["signal"])[:25],
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
            return self._neutral(sym, ["finalize_error"], t0, spread_pct=spread_pct, bar_key=bar_key, trade_blocked=True)

    def _position_context(self, sym: str) -> Tuple[int, float]:
        try:
            ensure_mt5()
            with MT5_LOCK:
                positions = mt5.positions_get(symbol=sym) or []
            open_pos = int(len(positions))
            unreal_pl = float(sum(float(p.profit or 0.0) for p in positions))
            return open_pos, unreal_pl
        except Exception:
            return 0, 0.0


__all__ = ["SignalEngine", "SignalResult"]
