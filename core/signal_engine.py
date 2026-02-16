# core/signal_engine.py — Unified SignalEngine for all assets.
# Merges ~1600 lines each from BTC and XAU signal engines.
from __future__ import annotations

import hashlib
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .config import BaseEngineConfig, BaseSymbolParams
from .models import SignalResult
from .risk_engine import RiskManager
from .utils import _is_finite, _side_norm, clamp01

log = logging.getLogger("core.signal_engine")


class SignalEngine:
    """
    Unified planner-only SignalEngine:
      - execute=True  → fills entry/sl/tp/lot using RiskManager.plan_order()
      - execute=False → returns signal only (no order)
      - never sends orders itself (engine.py dispatches to ExecutionWorker)

    All asset-specific behavior is driven by config:
      - BTC: wider ATR thresholds, crypto-specific round numbers, _sanitize_rates
      - XAU: forex session filters, _is_noisy filter, gold round levels

    This class uses the same 100-point scoring system for both assets, with
    weights and thresholds parameterized via EngineConfig.

    Usage:
        engine = SignalEngine(cfg, sp, market_feed, feature_engine, risk_manager)
        result = engine.compute(execute=True)
    """

    def __init__(
        self,
        cfg: BaseEngineConfig,
        sp: BaseSymbolParams,
        market_feed: Any,
        feature_engine: Any,
        risk_manager: RiskManager,
    ) -> None:
        self.cfg = cfg
        self.sp = sp
        self._feed = market_feed
        self._fe = feature_engine
        self._rm = risk_manager

        # Caches
        self._last_bar_key: str = ""
        self._last_signal_id: str = ""
        self._weights_cache: Optional[Dict[str, float]] = None

        log.info(
            "SignalEngine(%s) initialized | tf=%s/%s/%s",
            sp.base, sp.tf_primary, sp.tf_confirm, sp.tf_long,
        )

    # ─── Data retrieval ──────────────────────────────────────────────

    def _unwrap_rates_df(self, ret: Any) -> Optional[pd.DataFrame]:
        """
        feed.get_rates() may return:
          - df
          - (df, meta)
          - (df, meta, extra)
        This function handles all variants.
        """
        if ret is None:
            return None
        if isinstance(ret, pd.DataFrame):
            return ret
        if isinstance(ret, (tuple, list)):
            if len(ret) >= 1 and isinstance(ret[0], pd.DataFrame):
                return ret[0]
        return None

    def _sanitize_rates(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Ensure DataFrame is usable:
          - columns: open/high/low/close
          - sorted by time (if time column exists)
          - numeric close
        """
        if df is None or len(df) == 0:
            return None

        # Normalize column names
        renames = {}
        for col in df.columns:
            cl = str(col).lower()
            if cl in ("open",):
                renames[col] = "open"
            elif cl in ("high",):
                renames[col] = "high"
            elif cl in ("low",):
                renames[col] = "low"
            elif cl in ("close",):
                renames[col] = "close"
            elif cl in ("tick_volume", "real_volume", "volume"):
                renames[col] = "tick_volume"
            elif cl in ("time",):
                renames[col] = "time"
        if renames:
            df = df.rename(columns=renames)

        if "close" not in df.columns:
            return None

        # Sort by time if available
        if "time" in df.columns:
            df = df.sort_values("time")

        # Ensure numeric
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df.dropna(subset=["close"], inplace=True)

        if len(df) < 10:
            return None
        return df

    def _get_rates_df(
        self, sym: str, tf: str, *, bars: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Safe wrapper around feed.get_rates()."""
        try:
            if bars is not None:
                ret = self._feed.get_rates(sym, tf, bars=bars)
            else:
                ret = self._feed.get_rates(sym, tf)
            df = self._unwrap_rates_df(ret)
            return self._sanitize_rates(df)
        except Exception as exc:
            log.error("_get_rates_df(%s, %s): %s", sym, tf, exc)
            return None

    # ─── Main compute ────────────────────────────────────────────────

    def compute(self, execute: bool = False) -> SignalResult:
        """
        Main signal computation.
          execute=False: returns signal only
          execute=True : returns signal + (entry/sl/tp/lot) plan

        Steps:
          1. Fetch market data (primary + confirm + long timeframes)
          2. Compute indicators via FeatureEngine
          3. Run guard_decision pre-checks (risk, session, spread, etc.)
          4. Calculate ensemble score (100-point system)
          5. Apply filters and confirmation layers
          6. If execute: call RiskManager.plan_order for SL/TP/lot
        """
        t0 = time.time()
        sym = self._symbol()
        reasons: List[str] = []

        try:
            # ── 1. Fetch rates ──
            dfp = self._get_rates_df(sym, self.sp.tf_primary)
            if dfp is None or len(dfp) < 30:
                return self._neutral(sym, ["no_primary_data"], t0)

            dfc = self._get_rates_df(sym, self.sp.tf_confirm)
            dfl = self._get_rates_df(sym, self.sp.tf_long)

            # ── 2. Compute indicators ──
            df_dict = {
                self.sp.tf_primary: dfp,
                self.sp.tf_confirm: dfc if dfc is not None else dfp,
                self.sp.tf_long: dfl if dfl is not None else dfp,
            }

            try:
                indicators = self._fe.compute_indicators(df_dict, shift=1)
            except Exception as exc:
                return self._neutral(sym, [f"indicator_error:{exc}"], t0)

            if not indicators:
                return self._neutral(sym, ["no_indicators"], t0)

            indp = indicators.get(self.sp.tf_primary, {})
            indc = indicators.get(self.sp.tf_confirm, {})
            indl = indicators.get(self.sp.tf_long, {})

            if not indp:
                return self._neutral(sym, ["no_primary_indicators"], t0)

            # ── 3. Market state ──
            bid, ask = self._tick_bid_ask(sym)
            if bid <= 0 or ask <= 0:
                return self._neutral(sym, ["no_tick"], t0)

            spread_pct = (ask - bid) / bid if bid > 0 else 0.0
            bar_key = self._bar_key(dfp)

            # BTC flash-crash guard
            flash_ok, flash_reason = self._flash_crash_guard(dfp)
            if not flash_ok:
                return self._neutral(
                    sym, reasons + [flash_reason], t0,
                    spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                )

            # ATR for regime detection
            atr_pct = float(indp.get("atr_pct", 0.0))
            atr_val = float(indp.get("atr", 0.0))

            # Tick stats from feed
            tick_stats = getattr(self._feed, "tick_stats", None)
            tick_ok = True
            tick_reason = ""
            if tick_stats is not None:
                tick_ok = getattr(tick_stats, "ok", True)
                tick_reason = getattr(tick_stats, "reason", "")

            # Get adaptive params
            adapt = self._get_adaptive_params(indp, indl, atr_pct)

            # ── 4. Risk guard ──
            last_bar_age = self._last_bar_age(dfp)
            in_session = self._in_active_session()
            dd_exceeded = self._check_drawdown()

            guard_ok, guard_reasons = self._rm.guard_decision(
                spread_pct=spread_pct,
                tick_ok=tick_ok,
                tick_reason=tick_reason,
                ingest_ms=(time.time() - t0) * 1000,
                last_bar_age=last_bar_age,
                in_session=in_session,
                drawdown_exceeded=dd_exceeded,
                latency_cooldown=self._rm.latency_cooldown(),
            )

            if not guard_ok:
                return self._neutral(
                    sym, guard_reasons, t0,
                    spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                )

            # ── 5. Ensemble scoring ──
            book = getattr(self._feed, "order_book", {})
            if not isinstance(book, dict):
                book = {}

            score_result = self._ensemble_score(
                indp, indc, indl, book, adapt, spread_pct, tick_stats,
            )

            net_score = score_result.get("net_score", 0.0)
            net_abs = abs(net_score)
            signal_dir = score_result.get("direction", "Neutral")
            sub_reasons = score_result.get("reasons", [])
            reasons.extend(sub_reasons)

            # ── 6. Signal decision ──
            if net_abs < self.cfg.signal_min_score:
                return self._neutral(
                    sym, reasons + [f"weak_score:{net_abs:.1f}"], t0,
                    spread_pct=spread_pct, bar_key=bar_key,
                )

            # Confidence
            conf = self._conf_from_strength(net_abs, spread_pct, tick_stats)

            # Confirmation layers
            ext = self._extreme_flag(dfp, indp)
            sweep = self._liquidity_sweep(dfp, indp)
            div = self._divergence(dfp, indp)
            near_rn = self._near_round(bid, indp)
            cl_bonus = self._confirm_layer(ext, sweep, div, near_rn)
            conf = min(100, conf + cl_bonus)

            # Volume gate
            vol_ok = self._sniper_volume_gate(dfp, last_age=last_bar_age)
            if not vol_ok:
                conf = max(0, conf - 15)
                reasons.append("low_volume")

            # Meta gate
            meta_ok = self._gate_meta(signal_dir, indp, dfp)
            if not meta_ok:
                conf = max(0, conf - 10)
                reasons.append("meta_gate_fail")

            # Conformal check
            if not self._conformal_ok(dfp):
                conf = max(0, conf - 5)
                reasons.append("conformal_warn")

            # Apply filters
            signal_dir, conf = self._apply_filters(signal_dir, conf, indp, indc, indl, reasons)

            # Cap confidence by strength
            conf = self._cap_conf_by_strength(conf, net_abs)

            # ── QUANTUM SNIPER FILTER ──
            # Strict 80% confidence floor as demanded by USER
            sniper_floor = max(int(getattr(self.cfg, "min_confidence", 80) or 80), 80)
            if conf < sniper_floor or (float(conf) / 100.0) < 0.80:
                return self._neutral(
                    sym, reasons + [f"sniper_reject:{conf}<{sniper_floor}"], t0,
                    confidence=conf, spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                )

            # Can emit?
            can_emit, emit_reason = self._rm.can_emit_signal(conf)
            if not can_emit:
                return self._neutral(
                    sym, reasons + [f"emit_blocked:{emit_reason}"], t0,
                    confidence=conf, spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                )

            adapt["confidence"] = clamp01(conf / 100.0)

            # ── 7. Finalize ──
            return self._finalize(
                sym=sym,
                signal=signal_dir,
                conf=conf,
                indp=indp,
                adapt=adapt,
                spread_pct=spread_pct,
                t0=t0,
                execute=execute,
                tick_stats=tick_stats,
                book=book,
                bar_key=bar_key,
                reasons=reasons,
                dfp=dfp,
            )

        except Exception as exc:
            log.error(
                "SignalEngine.compute ERROR | %s | %s\n%s",
                sym, exc, traceback.format_exc(),
            )
            return self._neutral(sym, [f"compute_error:{exc}"], t0)

    # ─── Helpers ─────────────────────────────────────────────────────

    def get_signal(self, execute: bool = False) -> SignalResult:
        """Backward-compatible signal API with strict sniper confidence gating."""
        res = self.compute(execute=execute)
        if res.signal != "Neutral" and (float(res.confidence) / 100.0) < 0.80:
            res.signal = "Neutral"
            res.trade_blocked = True
            res.reasons.append("sniper_reject:<0.80")
        return res

    def _symbol(self) -> str:
        return self.sp.symbol

    def _tick_bid_ask(self, sym: str) -> Tuple[float, float]:
        try:
            if mt5 is not None:
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    return float(tick.bid), float(tick.ask)
            # Fallback to feed
            if hasattr(self._feed, "last_bid"):
                return float(self._feed.last_bid), float(self._feed.last_ask)
        except Exception:
            pass
        return 0.0, 0.0

    def _bar_key(self, df: pd.DataFrame) -> str:
        try:
            last = df.iloc[-1]
            t = last.get("time", 0)
            c = last.get("close", 0)
            return f"{t}_{c}"
        except Exception:
            return "no_bar"

    def _last_bar_age(self, df: pd.DataFrame) -> float:
        try:
            if "time" in df.columns:
                last_ts = float(df.iloc[-1]["time"])
                return time.time() - last_ts
        except Exception:
            pass
        return 0.0

    def _flash_crash_guard(self, dfp: pd.DataFrame) -> Tuple[bool, str]:
        """BTC flash-crash guard via return z-score (sigma event)."""
        try:
            if "BTC" not in str(self.sp.base).upper():
                return True, ""
            if dfp is None or len(dfp) < 10:
                return True, ""
            lookback = int(getattr(self.cfg, "flash_crash_lookback", 30) or 30)
            lookback = max(5, min(200, lookback))

            cols = {c.lower(): c for c in dfp.columns}
            c = dfp[cols.get("close", "close")].values
            if len(c) < lookback + 2:
                return True, ""

            window = c[-(lookback + 1):]
            rets = np.diff(window) / window[:-1]
            if len(rets) < 3:
                return True, ""
            std = float(np.std(rets, ddof=1))
            if std <= 0:
                return True, ""
            z = abs(float(rets[-1])) / std
            sigma = float(getattr(self.cfg, "flash_crash_sigma", 3.5) or 3.5)
            if z >= sigma:
                cooldown = float(getattr(self.cfg, "flash_crash_cooldown_sec", 300.0) or 300.0)
                self._rm.block_analysis(cooldown)
                return False, f"flash_crash:z={z:.2f}"
            return True, ""
        except Exception:
            return True, ""

    def _signal_id(self, sym: str, tf: str, bar_key: str, signal: str) -> str:
        raw = f"{sym}_{tf}_{bar_key}_{signal}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

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
        return SignalResult(
            signal="Neutral",
            symbol=sym,
            confidence=confidence,
            spread_pct=spread_pct or 0.0,
            regime=regime or "normal",
            signal_id="",
            bar_key=bar_key,
            reasons=reasons,
            latency_ms=(time.time() - t0) * 1000,
            trade_blocked=trade_blocked,
        )

    # ─── Session checks ──────────────────────────────────────────────

    def _in_active_session(self) -> bool:
        if self.cfg.ignore_sessions or self.sp.is_24_7:
            return True
        try:
            from datetime import datetime, timezone
            h = datetime.now(timezone.utc).hour
            for start, end in self.cfg.active_sessions:
                if start <= h < end:
                    return True
            return False
        except Exception:
            return True

    def _check_drawdown(self) -> bool:
        try:
            eq = self._rm._get_equity()
            peak = self._rm._peak_equity
            if peak <= 0:
                return False
            dd = (peak - eq) / peak
            return dd >= self.cfg.max_drawdown
        except Exception:
            return False

    # ─── Adaptive params ─────────────────────────────────────────────

    def _get_adaptive_params(
        self,
        indp: Dict[str, Any],
        indl: Dict[str, Any],
        atr_pct: float,
    ) -> Dict[str, Any]:
        """Gather adaptive parameters from indicators."""
        regime = str(indp.get("regime", "normal"))
        atr = float(indp.get("atr", 0.0))
        adx = float(indp.get("adx", 15.0))
        ker = float(indp.get("ker", 0.0))
        rvi = float(indp.get("rvi", 0.0))

        return {
            "regime": regime,
            "atr": atr,
            "atr_pct": atr_pct,
            "adx": adx,
            "ker": ker,
            "rvi": rvi,
            "confidence": 0.5,
            "trend": str(indp.get("trend", "flat")),
            "volatility": str(indp.get("volatility_regime", "normal")),
        }

    # ─── Scoring system ──────────────────────────────────────────────

    def _ensemble_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        book: Dict[str, Any],
        adapt: Dict[str, Any],
        spread_pct: float,
        tick_stats: Any,
    ) -> Dict[str, Any]:
        """
        INSTITUTIONAL 100-POINT SCORING SYSTEM

        Components:
          - Trend Score (30 pts): H1 slope alignment + EMA stack + ADX strength
          - Momentum Score (20 pts): RSI zone + MACD crossover + histogram direction
          - Volatility Score (15 pts): ATR regime + BB width
          - Structure Score (15 pts): FVG + liquidity sweep + order blocks
          - Flow Score (10 pts): Tick imbalance + book pressure
          - Mean-Reversion Score (10 pts): BB touch + RSI extremes
        """
        weights = self._get_base_weights()

        trend = self._trend_score(indp, indc, indl, adapt.get("regime", "normal"))
        momentum = self._momentum_score(indp, indc)
        meanrev = self._meanrev_score(indp, adapt.get("regime", "normal"))
        structure = self._structure_score(indp)

        # Volatility score
        vol_score = 0.0
        regime = adapt.get("regime", "normal")
        if regime == "explosive":
            vol_score = 0.7
        elif regime == "normal":
            vol_score = 0.5
        else:
            vol_score = 0.2

        # Flow score from tick stats
        flow_score = 0.0
        if tick_stats is not None:
            imb = getattr(tick_stats, "imbalance", 0.0)
            flow_score = abs(float(imb)) if _is_finite(imb) else 0.0
            flow_score = min(1.0, flow_score)

        # Combine
        buy_score = 0.0
        sell_score = 0.0

        # Trend component
        trend_val = trend.get("value", 0.0)
        buy_score += max(0, trend_val) * weights["trend"]
        sell_score += max(0, -trend_val) * weights["trend"]

        # Momentum component
        mom_val = momentum.get("value", 0.0)
        buy_score += max(0, mom_val) * weights["momentum"]
        sell_score += max(0, -mom_val) * weights["momentum"]

        # Structure
        struct_val = structure.get("value", 0.0)
        buy_score += max(0, struct_val) * weights["structure"]
        sell_score += max(0, -struct_val) * weights["structure"]

        # Mean reversion
        mr_val = meanrev.get("value", 0.0)
        buy_score += max(0, mr_val) * weights["mean_reversion"]
        sell_score += max(0, -mr_val) * weights["mean_reversion"]

        # Volatility (direction-neutral boost)
        buy_score += vol_score * weights["volatility"] * 0.5
        sell_score += vol_score * weights["volatility"] * 0.5

        # Flow
        if tick_stats is not None and hasattr(tick_stats, "imbalance"):
            imb = float(getattr(tick_stats, "imbalance", 0.0))
            if imb > 0:
                buy_score += flow_score * weights["flow"]
            else:
                sell_score += flow_score * weights["flow"]

        net = buy_score - sell_score
        direction = "Buy" if net > 0 else "Sell" if net < 0 else "Neutral"

        reasons = []
        reasons.append(f"trend={trend_val:.2f}")
        reasons.append(f"mom={mom_val:.2f}")
        reasons.append(f"struct={struct_val:.2f}")
        reasons.append(f"mr={mr_val:.2f}")
        reasons.append(f"vol={vol_score:.2f}")
        reasons.append(f"flow={flow_score:.2f}")
        reasons.append(f"net={net:+.1f}")

        return {
            "net_score": net,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "direction": direction,
            "reasons": reasons,
        }

    # ─── Sub-scores ──────────────────────────────────────────────────

    def _trend_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        regime: str,
    ) -> Dict[str, float]:
        """[-1, +1]: EMA stack + ADX + H1 slope alignment."""
        score = 0.0

        # EMA stack
        ema_s = float(indp.get("ema_short", 0))
        ema_m = float(indp.get("ema_medium", 0))
        ema_l = float(indp.get("ema_slow", 0))

        if ema_s > ema_m > ema_l > 0:
            score += 0.4
        elif ema_s < ema_m < ema_l and ema_l > 0:
            score -= 0.4

        # ADX strength
        adx = float(indp.get("adx", 15))
        if adx > self.cfg.adx_trend_min:
            score *= 1.3
        elif adx < self.cfg.adx_min:
            score *= 0.5

        # H1 slope
        h1_slope = float(indl.get("linreg_slope", 0))
        if h1_slope > 0.3:
            score += 0.2
        elif h1_slope < -0.3:
            score -= 0.2

        # Confirm TF alignment
        c_trend = str(indc.get("trend", "flat"))
        if c_trend == "bull" and score > 0:
            score += 0.1
        elif c_trend == "bear" and score < 0:
            score -= 0.1

        return {"value": max(-1.0, min(1.0, score))}

    def _momentum_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
    ) -> Dict[str, float]:
        """[-1, +1]: RSI zone + MACD crossover + histogram."""
        score = 0.0

        rsi = float(indp.get("rsi", 50))
        if rsi > 55:
            score += min(0.3, (rsi - 50) / 100)
        elif rsi < 45:
            score -= min(0.3, (50 - rsi) / 100)

        # MACD
        macd = float(indp.get("macd", 0))
        macd_sig = float(indp.get("macd_signal", 0))
        macd_hist = float(indp.get("macd_hist", 0))

        if macd > macd_sig:
            score += 0.2
        elif macd < macd_sig:
            score -= 0.2

        if macd_hist > 0:
            score += 0.15
        elif macd_hist < 0:
            score -= 0.15

        return {"value": max(-1.0, min(1.0, score))}

    def _meanrev_score(
        self, indp: Dict[str, Any], regime: str,
    ) -> Dict[str, float]:
        """[-1, +1]: BB touch + RSI extremes."""
        score = 0.0

        rsi = float(indp.get("rsi", 50))
        bb_pct = float(indp.get("bb_pctb", 0.5))

        # Oversold + near lower BB → buy signal
        if rsi < 30 and bb_pct < 0.1:
            score += 0.6
        elif rsi < 35 and bb_pct < 0.2:
            score += 0.3

        # Overbought + near upper BB → sell signal
        if rsi > 70 and bb_pct > 0.9:
            score -= 0.6
        elif rsi > 65 and bb_pct > 0.8:
            score -= 0.3

        return {"value": max(-1.0, min(1.0, score))}

    def _structure_score(self, indp: Dict[str, Any]) -> Dict[str, float]:
        """[-1, +1]: FVG + liquidity sweep + order blocks."""
        score = 0.0

        fvg = str(indp.get("fvg", ""))
        sweep = str(indp.get("sweep", ""))
        ob = str(indp.get("order_block", ""))

        if "bull" in fvg:
            score += 0.3
        elif "bear" in fvg:
            score -= 0.3

        if "bull" in sweep:
            score += 0.25
        elif "bear" in sweep:
            score -= 0.25

        if "bull" in ob:
            score += 0.2
        elif "bear" in ob:
            score -= 0.2

        # Round number proximity
        near_rn = bool(indp.get("near_round", False))
        if near_rn:
            # Round numbers attract price action
            score *= 1.1

        return {"value": max(-1.0, min(1.0, score))}

    # ─── Confirmation layers ─────────────────────────────────────────

    def _extreme_flag(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> bool:
        try:
            rsi = float(indp.get("rsi", 50))
            return rsi < 20 or rsi > 80
        except Exception:
            return False

    def _liquidity_sweep(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> str:
        return str(indp.get("sweep", ""))

    def _divergence(self, dfp: pd.DataFrame, indp: Dict[str, Any]) -> str:
        return str(indp.get("divergence", ""))

    def _near_round(self, price: float, indp: Dict[str, Any]) -> bool:
        return bool(indp.get("near_round", False))

    def _confirm_layer(
        self, ext: bool, sweep: str, div: str, near_rn: bool,
    ) -> int:
        bonus = 0
        if ext:
            bonus += 5
        if sweep:
            bonus += 4
        if div:
            bonus += 3
        if near_rn:
            bonus += 2
        return bonus

    # ─── Volume gate ─────────────────────────────────────────────────

    def _sniper_volume_gate(
        self, dfp: pd.DataFrame, *, last_age: float,
    ) -> bool:
        """Dynamic relative-volume gate using recent median."""
        try:
            if "tick_volume" not in dfp.columns:
                return True
            vols = dfp["tick_volume"].values[-50:]
            if len(vols) < 5:
                return True
            med = float(np.median(vols))
            if med <= 0:
                return True
            last_vol = float(vols[-1])
            ratio = last_vol / med
            return ratio >= 0.3
        except Exception:
            return True

    def _early_momentum_trigger(
        self,
        indp: Dict[str, Any],
        dfp: pd.DataFrame,
        net_norm: float,
    ) -> bool:
        """Fast move-start detector to avoid entering several candles late."""
        try:
            c = dfp["close"].values if "close" in dfp.columns else dfp["Close"].values
            if len(c) < 5:
                return False
            pct_move = abs(c[-1] - c[-3]) / c[-3] if c[-3] > 0 else 0.0
            atr_pct = float(indp.get("atr_pct", 0.005))
            return pct_move > atr_pct * 1.5 and abs(net_norm) > 0.4
        except Exception:
            return False

    # ─── Weights ─────────────────────────────────────────────────────

    def _get_base_weights(self) -> Dict[str, float]:
        if self._weights_cache is not None:
            return self._weights_cache
        raw = dict(self.cfg.weights)
        total = sum(raw.values())
        if total <= 0:
            total = 100.0
        self._weights_cache = {k: v / total * 100 for k, v in raw.items()}
        return self._weights_cache

    # ─── Confidence ──────────────────────────────────────────────────

    def _conf_from_strength(
        self, net_abs: float, spread_pct: float, tick_stats: Any,
    ) -> int:
        """Map net score absolute value to confidence [0, 100]."""
        if net_abs < 15:
            base = 30
        elif net_abs < 25:
            base = 50
        elif net_abs < 40:
            base = 65
        elif net_abs < 55:
            base = 75
        else:
            base = 85

        # Penalize wide spread
        if spread_pct > self.sp.spread_limit_pct:
            base -= 10

        return max(0, min(100, base))

    def _cap_conf_by_strength(self, conf: int, net_abs: float) -> int:
        """Prevents weak signals from showing high confidence."""
        if net_abs < 15:
            return min(conf, 45)
        if net_abs < 25:
            return min(conf, 60)
        if net_abs < 35:
            return min(conf, 75)
        if net_abs < 50:
            return min(conf, 88)
        return min(conf, 100)

    # ─── Meta gate ───────────────────────────────────────────────────

    def _conformal_ok(self, dfp: pd.DataFrame) -> bool:
        """Check if recent price action is consistent with signal."""
        try:
            c = dfp["close"].values if "close" in dfp.columns else dfp["Close"].values
            if len(c) < 10:
                return True
            # Check for extreme volatility that might indicate unreliable data
            pct_changes = np.abs(np.diff(c[-10:])) / c[-11:-1]
            median_pct = float(np.median(pct_changes))
            last_pct = float(pct_changes[-1]) if len(pct_changes) > 0 else 0.0
            return last_pct < median_pct * 5
        except Exception:
            return True

    def _gate_meta(
        self, side: str, ind: Dict[str, Any], dfp: pd.DataFrame,
    ) -> bool:
        """Meta gate: checks if recent price movements show edge for the direction."""
        try:
            c = dfp["close"].values if "close" in dfp.columns else dfp["Close"].values
            if len(c) < 10:
                return True
            recent_change = (c[-1] - c[-5]) / c[-5] if c[-5] > 0 else 0.0
            if _side_norm(side) == "Buy":
                return recent_change > -0.02
            else:
                return recent_change < 0.02
        except Exception:
            return True

    # ─── Filters ─────────────────────────────────────────────────────

    def _apply_filters(
        self,
        signal: str,
        conf: int,
        indp: Dict[str, Any],
        indc: Optional[Dict[str, Any]] = None,
        indl: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[str]] = None,
    ) -> Tuple[str, int]:
        """Apply final safety filters."""
        # ADX filter
        adx = float(indp.get("adx", 15))
        if adx < self.cfg.adx_min:
            conf = max(0, conf - 10)

        # MTF confluence penalty (M5/M15 slopes vs M1 direction)
        if bool(getattr(self.cfg, "mtf_penalty_enabled", True)) and signal in ("Buy", "Sell"):
            m5 = indc or {}
            m15 = indl or {}
            thresh = float(getattr(self.cfg, "mtf_slope_thresh", 0.10) or 0.10)

            def _slope_dir(v: float) -> str:
                if v > thresh:
                    return "bull"
                if v < -thresh:
                    return "bear"
                return "flat"

            m5_dir = _slope_dir(float(m5.get("linreg_slope", 0.0) or 0.0))
            m15_dir = _slope_dir(float(m15.get("linreg_slope", 0.0) or 0.0))

            conflict_m5 = (signal == "Buy" and m5_dir == "bear") or (signal == "Sell" and m5_dir == "bull")
            conflict_m15 = (signal == "Buy" and m15_dir == "bear") or (signal == "Sell" and m15_dir == "bull")

            mult = 1.0
            if conflict_m5:
                pen = clamp01(float(getattr(self.cfg, "mtf_m5_penalty", 0.20) or 0.20))
                mult *= (1.0 - pen)
                if reasons is not None:
                    reasons.append(f"mtf_m5_conflict:{m5_dir}")
            if conflict_m15:
                pen = clamp01(float(getattr(self.cfg, "mtf_m15_penalty", 0.50) or 0.50))
                mult *= (1.0 - pen)
                if reasons is not None:
                    reasons.append(f"mtf_m15_conflict:{m15_dir}")

            if mult < 1.0:
                conf = int(conf * mult)

        return signal, conf

    # ─── Finalize ────────────────────────────────────────────────────

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
        tick_stats: Any,
        book: Dict[str, Any],
        bar_key: str,
        reasons: List[str],
        dfp: Optional[pd.DataFrame] = None,
    ) -> SignalResult:
        """Build final SignalResult, optionally with execution plan."""
        sig_id = self._signal_id(sym, self.sp.tf_primary, bar_key, signal)
        latency = (time.time() - t0) * 1000

        result = SignalResult(
            signal=signal,
            symbol=sym,
            confidence=conf,
            spread_pct=spread_pct,
            regime=adapt.get("regime", "normal"),
            signal_id=sig_id,
            bar_key=bar_key,
            reasons=reasons,
            latency_ms=latency,
            timeframe=self.sp.tf_primary,
        )

        if execute and signal != "Neutral" and conf >= self.cfg.min_confidence:
            plan = self._rm.plan_order(
                side=signal,
                confidence=conf / 100.0,
                ind=indp,
                adapt=adapt,
                ticks=tick_stats,
                df=dfp,
            )

            if not plan.get("blocked", True):
                result.entry = plan["entry"]
                result.sl = plan["sl"]
                result.tp = plan["tp"]
                result.lot = plan["lot"]
            else:
                result.trade_blocked = True
                result.reasons.append(f"plan_blocked:{plan.get('reason', 'unknown')}")

        self._rm.register_signal_emitted()
        return result

    # ─── Position context ────────────────────────────────────────────

    def _position_context(self, sym: str) -> Dict[str, Any]:
        """Get current position info for the symbol."""
        try:
            if mt5 is not None:
                positions = mt5.positions_get(symbol=sym)
                if positions:
                    return {
                        "count": len(positions),
                        "total_volume": sum(p.volume for p in positions),
                        "total_profit": sum(p.profit for p in positions),
                    }
        except Exception:
            pass
        return {"count": 0, "total_volume": 0.0, "total_profit": 0.0}
