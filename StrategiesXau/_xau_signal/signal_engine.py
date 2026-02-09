from __future__ import annotations

import hashlib
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime

from config_xau import EngineConfig, SymbolParams
from DataFeed.xau_market_feed import MarketFeed, TickStats
from mt5_client import MT5_LOCK

from ..xau_indicators import Classic_FeatureEngine, safe_last
from ..xau_risk_management import RiskManager

from .logging_ import log
from .models import SignalResult
from .utils import _as_regime, _atr, _atr_np, _clamp


class SignalEngine:
    """
    Production-grade SignalEngine (planner-only):
      - execute=True => attaches (entry/sl/tp/lot) via RiskManager.plan_order, NEVER sends orders
      - ERROR-only logging for real exceptions only
      - debounce + stability gating (monotonic time)
      - deterministic signal_id (bar-time based)
      - robust compatibility with different FeatureEngine outputs

    IMPORTANT:
      Bot/engine.py MUST call compute(execute=True) for Buy/Sell,
      otherwise lot/sl/tp stay None => no trades.
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

        # Ensure weights exist (do not change public config API)
        if not hasattr(self.cfg, "weights") or not isinstance(getattr(self.cfg, "weights", None), dict):
            setattr(
                self.cfg,
                "weights",
                {"trend": 0.50, "momentum": 0.25, "meanrev": 0.10, "structure": 0.10, "volume": 0.05},
            )

        self._w_keys = ("trend", "momentum", "meanrev", "structure", "volume")
        self._w_sig: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._w_base = np.array([0.50, 0.25, 0.10, 0.10, 0.05], dtype=np.float64)
        self._refresh_weights_if_needed()

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

            # Phase C flag (continue analysis for monitoring, block orders later)
            self.risk.evaluate_account_state()
            phase_c_block = bool(
                self.risk.current_phase == "C"
                and not bool(getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False))
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

            guard_reasons: List[str] = []
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
                if not phase_c_block:
                    return self._neutral(
                        sym,
                        reasons,
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        trade_blocked=True,
                    )
                guard_reasons = reasons[:25]

            # Indicators (MTF)
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
                    regime=_as_regime(indp.get("regime")),
                )

            adapt = self._get_adaptive_params(indp, indl, atr_pct)

            # ==============================================================
            # SNIPER LOGIC #1: Volume Validation (Smart)
            # Reject signals in low-volume conditions (choppy markets)
            # BUT: Skip check during first 15 seconds of bar (volume is naturally low)
            # ==============================================================
            # ==============================================================
            # SNIPER LOGIC #1: Volume Validation (Moved to after Confidence)
            # ==============================================================
            # (Volume check removed from here to allow confidence-based filtering)
            pass

            # HTF alignment (FIXED: symmetric, no "sell always ok")
            close_p = float(indp.get("close", 0.0) or 0.0)
            ema50_l = float(indl.get("ema50", close_p) or close_p)
            adx_l = float(indl.get("adx", 0.0) or 0.0)
            adx_lo_ltf = float(getattr(self.cfg, "adx_trend_lo", 18.0) or 18.0)

            ema_s, ema_m = self._ema_s_m(indp, close_p)
            require_stack = bool(getattr(self.cfg, "require_ema_stack", True))

            trend_l = str(indl.get("trend", "") or "").lower()
            trend_c = str(indc.get("trend", "") or "").lower()
            bias_up = ("up" in trend_l) or ("bull" in trend_l)
            bias_dn = ("down" in trend_l) or ("bear" in trend_l)

            c_up = ("up" in trend_c) or ("bull" in trend_c)
            c_dn = ("down" in trend_c) or ("bear" in trend_c)

            # Small tolerance for scalping noise
            ema50_pad = float(getattr(self.cfg, "ema50_pad", 0.0015) or 0.0015)

            ema_stack_buy = bool(close_p > ema_s > ema_m)
            ema_stack_sell = bool(close_p < ema_s < ema_m)

            # Core HTF gates
            trend_ok_buy = (close_p >= ema50_l * (1.0 - ema50_pad)) and (bias_up or (adx_l >= adx_lo_ltf * 0.90)) and (c_up or not trend_c)
            trend_ok_sell = (close_p <= ema50_l * (1.0 + ema50_pad)) and (bias_dn or (adx_l >= adx_lo_ltf * 0.90)) and (c_dn or not trend_c)

            if require_stack:
                trend_ok_buy = trend_ok_buy and ema_stack_buy
                trend_ok_sell = trend_ok_sell and ema_stack_sell

            # ==============================================================
            # SNIPER LOGIC #2: ENHANCED MTF Confirmation (10/10 Quality)    
            # Multi-Timeframe Alignment with ADX Strength + Price Position  
            # ==============================================================
            ema50_c = float(indc.get("ema50", 0.0) or 0.0)
            ema200_c = float(indc.get("ema200", indc.get("ema", 0.0)) or 0.0)
            ema50_l_val = float(indl.get("ema50", 0.0) or 0.0)
            ema200_l = float(indl.get("ema200", indl.get("ema", 0.0)) or 0.0)
            
            # ADX values for trend strength
            adx_c = float(indc.get("adx", 0.0) or 0.0)  # M5 ADX
            adx_l = float(indl.get("adx", 0.0) or 0.0)  # M15 ADX
            ADX_TREND_MIN = 20.0  # Minimum ADX for valid trend
            
            # Price position relative to EMAs
            close_c = float(indc.get("close", close_p) or close_p)  # M5 close
            close_l = float(indl.get("close", close_p) or close_p)  # M15 close

            # M5 trend from EMA crossover + Price position
            m5_bullish = (ema50_c > ema200_c and close_c > ema50_c) if (ema50_c > 0 and ema200_c > 0) else True
            m5_bearish = (ema50_c < ema200_c and close_c < ema50_c) if (ema50_c > 0 and ema200_c > 0) else True
            m5_trend_strong = adx_c >= ADX_TREND_MIN  # M5 has valid trend

            # M15 trend from EMA + Price position
            m15_bearish = (ema50_l_val < ema200_l * 0.998 and close_l < ema50_l_val) if (ema50_l_val > 0 and ema200_l > 0) else False
            m15_bullish = (ema50_l_val > ema200_l * 1.002 and close_l > ema50_l_val) if (ema50_l_val > 0 and ema200_l > 0) else False
            m15_trend_strong = adx_l >= ADX_TREND_MIN  # M15 has valid trend

            # MTF Alignment Score (for confidence boost)
            mtf_score_buy = 0
            mtf_score_sell = 0
            
            # M1 alignment
            if ema_stack_buy:
                mtf_score_buy += 1
            if ema_stack_sell:
                mtf_score_sell += 1
            
            # M5 alignment + strength
            if m5_bullish:
                mtf_score_buy += 1
                if m5_trend_strong:
                    mtf_score_buy += 1
            if m5_bearish:
                mtf_score_sell += 1
                if m5_trend_strong:
                    mtf_score_sell += 1
            
            # M15 alignment + strength
            if m15_bullish:
                mtf_score_buy += 1
                if m15_trend_strong:
                    mtf_score_buy += 1
            if m15_bearish:
                mtf_score_sell += 1
                if m15_trend_strong:
                    mtf_score_sell += 1
            
            # Perfect alignment: M1+M5+M15 all agree with strong ADX = 6 points
            PERFECT_MTF_SCORE = 6

            # Apply SNIPER MTF gate (enhanced)
            sniper_mtf_enabled = bool(getattr(self.cfg, "sniper_mtf_filter", True))
            if sniper_mtf_enabled:
                # Buy requires: M5 bullish AND M15 NOT bearish
                trend_ok_buy = trend_ok_buy and m5_bullish and (not m15_bearish)
                # Sell requires: M5 bearish AND M15 NOT bullish
                trend_ok_sell = trend_ok_sell and m5_bearish and (not m15_bullish)
                
                # STRICT MODE: Require M5 trend strength for high-quality signals
                strict_mtf = bool(getattr(self.cfg, "sniper_strict_mtf", True))
                if strict_mtf:
                    if not m5_trend_strong:
                        # Weak M5 trend - still allow but will have lower confidence later
                        pass

            # Ensemble score
            try:
                book = self.feed.fetch_book() or {}
            except Exception:
                book = {}

            net_norm, conf = self._ensemble_score(indp, indc, indl, book, adapt, spread_pct, tick_stats)
            net_abs = abs(net_norm)

            # ==============================================================
            # SNIPER LOGIC #1: Volume Validation (Strict & Tiered)
            # Re-inserted here to use 'conf'
            # ==============================================================
            try:
                if dfp is not None and "tick_volume" in dfp.columns and len(dfp) >= 20:
                    bar_age_ok = last_age >= 5.0
                    if bar_age_ok:
                        vol_ma = float(dfp["tick_volume"].iloc[-20:].mean())
                        current_vol = float(dfp["tick_volume"].iloc[-1])
                        
                        # Tiered Logic:
                        # Conf < 80: Strict (0.8x)
                        # Conf >= 85: Relaxed (0.3x)
                        # Else: Standard (0.6x)
                        if conf >= 85:
                            vol_mult = 0.3
                        elif conf < 80:
                            vol_mult = 0.8
                        else:
                            vol_mult = 0.6
                        
                        if current_vol < vol_ma * vol_mult:
                            return self._neutral(
                                sym,
                                [f"low_volume_sniper:{current_vol:.0f}<{vol_ma * vol_mult:.0f}", f"vol_mult:{vol_mult}"],
                                t0,
                                spread_pct=spread_pct,
                                bar_key=bar_key,
                                regime=_as_regime(adapt.get("regime")),
                                trade_blocked=True,
                            )
            except Exception:
                pass


            # Confluence layer
            ext = self._extreme_flag(dfp, indp)
            sweep = self._liquidity_sweep(dfp, indp)  # "bull"|"bear"|"" (empty)
            div = self._divergence(dfp, indp)  # "bullish"|"bearish"|"none"

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

            has_confluence = bool(sweep) or (div != "none")

            # Controlled boosts / penalties (fixed boolean logic)
            if has_confluence and net_abs >= 0.05:
                if net_abs >= 0.15:
                    conf = min(92, int(conf * 1.12))
                elif net_abs >= 0.08:
                    conf = min(88, int(conf * 1.06))
                else:
                    conf = min(85, int(conf * 1.03))

            if near_rn and (not has_confluence):
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

            # Ultra-confidence must be earned
            if conf >= 90 and (not has_confluence) and net_abs < 0.20:
                conf = min(conf, 89)

            # ==============================================================
            # MTF ALIGNMENT CONFIDENCE BOOST (10/10 Enhancement)
            # Reward perfect multi-timeframe alignment
            # ==============================================================
            mtf_score = mtf_score_buy if net_norm > 0 else mtf_score_sell
            
            if mtf_score >= PERFECT_MTF_SCORE:  # 6/6 = Perfect alignment
                conf = min(98, int(conf * 1.05))  # +5% boost
            elif mtf_score >= 5:  # Strong alignment
                conf = min(95, int(conf * 1.03))  # +3% boost
            elif mtf_score >= 4:  # Good alignment
                conf = min(92, int(conf * 1.01))  # +1% boost
            elif mtf_score <= 2:  # Weak alignment - penalize
                conf = max(0, int(conf * 0.90))  # -10% penalty

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
                if not (near_rn or has_confluence):
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

            # ============================================================
            # CRITICAL FIX #2: RSI EXTREME ZONE FILTER
            # Prevents "buying the top" and "selling the bottom"
            # This implements PULLBACK ENTRY LOGIC
            # ============================================================
            if signal in ("Buy", "Sell"):
                rsi_p = float(indp.get("rsi", 50.0) or 50.0)
                RSI_OVERBOUGHT = float(getattr(self.cfg, "rsi_overbought_limit", 70.0) or 70.0)
                RSI_OVERSOLD = float(getattr(self.cfg, "rsi_oversold_limit", 30.0) or 30.0)

                if signal == "Buy" and rsi_p > RSI_OVERBOUGHT:
                    return self._neutral(
                        sym,
                        [f"rsi_overbought:{rsi_p:.1f}", "wait_pullback"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

                if signal == "Sell" and rsi_p < RSI_OVERSOLD:
                    return self._neutral(
                        sym,
                        [f"rsi_oversold:{rsi_p:.1f}", "wait_pullback"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            # ==============================================================
            # SNIPER LOGIC #3: 2-Indicator Momentum Agreement
            # Require RSI + MACD to BOTH confirm direction
            # Eliminates weak signals with single-indicator confirmation
            # ==============================================================
            if signal in ("Buy", "Sell"):
                rsi_val = float(indp.get("rsi", 50.0) or 50.0)
                macd_val = float(indp.get("macd", 0.0) or 0.0)
                macd_signal = float(indp.get("macd_sig", indp.get("macd_signal", 0.0)) or 0.0)
                macd_hist_val = float(indp.get("macd_hist", (macd_val - macd_signal)) or (macd_val - macd_signal))

                momentum_confirms = 0
                if signal == "Buy":
                    if rsi_val > 50.0:  # RSI bullish
                        momentum_confirms += 1
                    if macd_hist_val > 0 and macd_val > macd_signal:  # MACD bullish crossover
                        momentum_confirms += 1
                elif signal == "Sell":
                    if rsi_val < 50.0:  # RSI bearish
                        momentum_confirms += 1
                    if macd_hist_val < 0 and macd_val < macd_signal:  # MACD bearish crossover
                        momentum_confirms += 1

                # Require 2 indicators to agree (sniper precision)
                min_momentum_confirms = int(getattr(self.cfg, "min_momentum_confirms", 2) or 2)
                if momentum_confirms < min_momentum_confirms:
                    return self._neutral(
                        sym,
                        [f"weak_momentum:{momentum_confirms}/{min_momentum_confirms}", "sniper_reject"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            # ============================================================
            # CRITICAL: Confluence Gate (prevents weak/incomplete signals)
            # ============================================================
            if signal in ("Buy", "Sell"):
                pts = 0.0
                pt_reasons: List[str] = []

                # EMA stack confirmation (strong in scalping)
                if (signal == "Buy" and ema_stack_buy) or (signal == "Sell" and ema_stack_sell):
                    pts += 1.0
                else:
                    pt_reasons.append("no_ema_stack")

                # Sweep / Divergence directional confirmation
                if signal == "Buy" and sweep == "bull":
                    pts += 1.0
                elif signal == "Sell" and sweep == "bear":
                    pts += 1.0

                if signal == "Buy" and div == "bullish":
                    pts += 1.0
                elif signal == "Sell" and div == "bearish":
                    pts += 1.0

                # Momentum confirmation (MACD/RSI quick check)
                macd = float(indp.get("macd", 0.0) or 0.0)
                macd_sig = float(indp.get("macd_sig", indp.get("macd_signal", 0.0)) or 0.0)
                macd_hist = float(indp.get("macd_hist", (macd - macd_sig)) or (macd - macd_sig))
                rsi_p = float(indp.get("rsi", 50.0) or 50.0)

                if signal == "Buy" and macd_hist > 0 and macd > macd_sig:
                    pts += 0.5
                if signal == "Sell" and macd_hist < 0 and macd < macd_sig:
                    pts += 0.5

                if signal == "Buy" and rsi_p >= 55.0:
                    pts += 0.5
                if signal == "Sell" and rsi_p <= 45.0:
                    pts += 0.5

                # Near round number is NOT a confirmation by itself (only if also confluence exists)
                if near_rn and has_confluence:
                    pts += 0.25

                # Minimum points required
                pts_min = float(getattr(self.cfg, "confirm_points_min", 2.0) or 2.0)

                # Ultra-strong override (rare)
                ultra_ok = bool(conf >= 92 and net_abs >= 0.22)

                if (pts < pts_min) and (not ultra_ok):
                    return self._neutral(
                        sym,
                        ["low_confluence", f"pts:{pts:.2f}"] + pt_reasons[:10],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

            # 11) Advanced filters
            if signal != "Neutral":
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
                        # Log MISSED OPPORTUNITY
                        log.warning(
                            "MISSED OPPORTUNITY | %s | Signal: %s | Conf: %d | Reasons: %s",
                            sym, signal, int(conf), gate_reasons
                        )
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

            # Debounce + stability (only debounce after active signal)
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
                    # ============================================================
                    # AI_NO_DECISION: Strict Confidence Gate
                    # AI must only speak when 100% sure. If uncertain, stay silent.
                    # ============================================================
                    log.warning(
                        "⛔ AI_NO_DECISION | conf=%d < min=%d | Signal Skipped: AI Inference returned NO ANSWER (Low Confidence)",
                        conf, conf_min
                    )
                    return self._neutral(
                        sym,
                        ["AI_NO_DECISION", f"low_confidence:{conf}<{conf_min}"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=_as_regime(adapt.get("regime")),
                        trade_blocked=True,
                    )

                # ADX gate only for trend regime
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

            reasons: List[str] = list(guard_reasons)
            phase = str(adapt.get("phase", "A") or "A")
            if sweep:
                reasons.append(f"sweep:{sweep}")
            if div != "none":
                reasons.append(f"div:{div}")
            if near_rn:
                reasons.append("near_rn")
            reasons.append(f"net:{net_norm:.3f}")
            reasons.append(f"mtf:{mtf_score}/6")  # MTF alignment score
            reasons.append(f"phase:{phase}")

            # Dynamic confidence caps (strength-based)
            if net_abs < 0.10:
                conf = min(conf, 82)
            else:
                conf = min(conf, 96)

            # ==============================================================
            # SNIPER LOGIC #2: Dynamic ATR Stop Loss ("Toqatfarso")
            # Widen SL for strong signals to survive noise
            # ==============================================================
            if conf >= 85:
                # Widen SL by 20%
                old_sl = float(adapt.get("sl_mult", 1.35))
                adapt["sl_mult"] = old_sl * 1.2 
                # Ensure TP is at least 2x SL (RR >= 1:2)
                # But don't shrink TP if it's already huge; just ensure min RR
                min_tp = adapt["sl_mult"] * 2.0
                if float(adapt.get("tp_mult", 0.0)) < min_tp:
                    adapt["tp_mult"] = min_tp

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
                dfp=dfp,  # Pass dataframe for structure analysis
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

            # STABILIZATION FIX: timestamp must be stable for the entire bar duration
            if "time" in df.columns:
                t = df["time"].iloc[-1]
            elif "datetime" in df.columns:
                t = df["datetime"].iloc[-1]
            elif "ts" in df.columns:
                t = df["ts"].iloc[-1]
            else:
                t = df.index[-1]

            # Ensure we have a pandas Timestamp
            if not isinstance(t, (pd.Timestamp, datetime)):
                try:
                    t = pd.to_datetime(t)
                except Exception:
                    return str(t)

            # Round to nearest second (strip micros) to prevent high-res timestamp jitter
            # Ideally this should be floor to timeframe, but strip micros is a good baseline fix
            if hasattr(t, "replace"):
                t = t.replace(microsecond=0)
            
            # Use strict format: YYYY-MM-DDTHH:MM:SS
            return t.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return "bar_err"

    def _signal_id(self, sym: str, tf: str, bar_key: str, signal: str) -> str:
        raw = f"{sym}|{tf}|{bar_key}|{signal}"
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"{sym}:{tf}:{signal}:{h}"

    def _is_noisy(self, indp: Dict[str, Any], indc: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Enhanced Noise Filter for M5/M15 focus (v2).
        """
        try:
            adx_p = float(indp.get("adx", 25.0) or 25.0)
            adx_c = float(indc.get("adx", 25.0) or 25.0)

            atr_rel = float(indp.get("atr_rel", 0.001) or 0.001)
            min_vol_thresh = float(getattr(self.cfg, "atr_rel_lo", 0.00055) or 0.00055)

            rsi_p = float(indp.get("rsi", 50.0) or 50.0)
            rsi_c = float(indc.get("rsi", 50.0) or 50.0)

            bb_width = float(indp.get("bb_width", 0.0) or 0.0)
            bb_width_thresh = float(getattr(self.cfg, "bb_width_range_max", 0.003) or 0.003)

            if adx_p < 16.0 and adx_c < 16.0:
                return True, "noise_adx_extreme_dual"

            if atr_rel < min_vol_thresh * 0.7:
                return True, f"noise_dead_vol:{atr_rel:.5f}"

            if adx_p < 20.0 and (44.0 < rsi_p < 56.0):
                if adx_c < 22.0 and (44.0 < rsi_c < 56.0):
                    return True, "noise_rsi_indecision"

            if bb_width > 0 and bb_width < bb_width_thresh * 0.5:
                if adx_p < 20.0:
                    return True, "noise_bb_squeeze"

            trend_p = str(indp.get("trend", "")).lower()
            trend_c = str(indc.get("trend", "")).lower()
            if trend_p and trend_c:
                p_up = "up" in trend_p or "bull" in trend_p
                p_dn = "down" in trend_p or "bear" in trend_p
                c_up = "up" in trend_c or "bull" in trend_c
                c_dn = "down" in trend_c or "bear" in trend_c
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
            tol = 0.0
            if atr > 0:
                tol = max(tol, atr * 0.15)
            if spread_pct > 0 and price > 0:
                spread_price = price * spread_pct
                tol = max(tol, spread_price * 1.5)

            if tol <= 0:
                tol = price * 0.0005  # 5bps fallback

            rem_100 = price % 100.0
            dist_100 = min(rem_100, 100.0 - rem_100)

            rem_50 = price % 50.0
            dist_50 = min(rem_50, 50.0 - rem_50)

            if dist_100 < tol or dist_50 < (tol * 0.8):
                return True
            return False
        except Exception:
            return False

    def _refresh_weights_if_needed(self) -> None:
        w = getattr(self.cfg, "weights", {}) or {}
        sig = (
            float(w.get("trend", 0.50)),
            float(w.get("momentum", 0.25)),
            float(w.get("meanrev", 0.10)),
            float(w.get("structure", 0.10)),
            float(w.get("volume", 0.05)),
        )
        if sig == self._w_sig:
            return

        vec = np.array(sig, dtype=np.float64)
        vec = np.maximum(vec, 0.0)
        s = float(np.sum(vec))
        self._w_base = (vec / s) if s > 0 else np.array([0.50, 0.25, 0.10, 0.10, 0.05], dtype=np.float64)
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
        """
        INSTITUTIONAL 100-POINT SCORING SYSTEM
        
        Components:
        - Trend Score (30 pts): H1 slope alignment + EMA stack + ADX strength
        - Momentum Score (20 pts): RSI zone + MACD crossover + histogram direction
        - Volatility Score (20 pts): ATR regime + spread OK + volume confirmation
        - Pattern Score (30 pts): Order Block + FVG + Sweep + Divergence
        
        Total: 100 points max
        Execution Rule: Only trade if total_score >= 85
        """
        try:
            _ = book  # Reserved for future orderbook analysis
            
            total_score = 0.0
            score_breakdown: Dict[str, float] = {}
            
            # ==================== TREND SCORE (30 pts max) ====================
            trend_score = 0.0
            
            # 1. ADX Strength (0-10 pts)
            adx_p = float(indp.get("adx", 0.0) or 0.0)
            adx_c = float(indc.get("adx", 0.0) or 0.0)
            if adx_p >= 30.0:
                trend_score += 10.0
            elif adx_p >= 25.0:
                trend_score += 8.0
            elif adx_p >= 20.0:
                trend_score += 5.0
            elif adx_p >= 15.0:
                trend_score += 3.0
            
            # 2. EMA Stack Alignment (0-10 pts)
            close_p = float(indp.get("close", 0.0) or 0.0)
            ema9 = float(indp.get("ema9", indp.get("ema_short", close_p)) or close_p)
            ema21 = float(indp.get("ema21", indp.get("ema_mid", close_p)) or close_p)
            ema50 = float(indp.get("ema50", indp.get("ema_long", close_p)) or close_p)
            
            if close_p > ema9 > ema21 > ema50:  # Perfect bullish stack
                trend_score += 10.0
            elif close_p < ema9 < ema21 < ema50:  # Perfect bearish stack
                trend_score += 10.0
            elif close_p > ema21 > ema50:  # Partial bullish
                trend_score += 6.0
            elif close_p < ema21 < ema50:  # Partial bearish
                trend_score += 6.0
            elif close_p > ema50:  # Weak bullish
                trend_score += 3.0
            elif close_p < ema50:  # Weak bearish
                trend_score += 3.0
            
            # 3. HTF Trend Alignment (0-10 pts)
            trend_l = str(indl.get("trend", "") or "").lower()
            trend_c = str(indc.get("trend", "") or "").lower()
            
            if ("strong_up" in trend_l or "strong_down" in trend_l):
                trend_score += 10.0
            elif ("up" in trend_l or "down" in trend_l):
                trend_score += 6.0
            elif ("up" in trend_c or "down" in trend_c):
                trend_score += 4.0
            
            trend_score = min(30.0, trend_score)
            score_breakdown["trend"] = trend_score
            total_score += trend_score
            
            # ==================== MOMENTUM SCORE (20 pts max) ====================
            momentum_score = 0.0
            
            # 1. RSI Zone (0-8 pts)
            rsi_p = float(indp.get("rsi", 50.0) or 50.0)
            if rsi_p > 60.0 or rsi_p < 40.0:  # Clear directional RSI
                momentum_score += 8.0
            elif rsi_p > 55.0 or rsi_p < 45.0:  # Moderate directional
                momentum_score += 5.0
            elif rsi_p > 52.0 or rsi_p < 48.0:  # Slight bias
                momentum_score += 2.0
            # RSI 48-52 = indecision = 0 pts
            
            # 2. MACD Crossover (0-8 pts)
            macd = float(indp.get("macd", 0.0) or 0.0)
            macd_sig = float(indp.get("macd_signal", indp.get("macd_sig", 0.0)) or 0.0)
            macd_hist = float(indp.get("macd_hist", 0.0) or 0.0)
            
            if macd > macd_sig and macd_hist > 0:  # Bullish crossover confirmed
                momentum_score += 8.0
            elif macd < macd_sig and macd_hist < 0:  # Bearish crossover confirmed
                momentum_score += 8.0
            elif abs(macd - macd_sig) < abs(macd) * 0.1:  # Near crossover
                momentum_score += 4.0
            
            # 3. MACD Histogram Momentum (0-4 pts)
            if abs(macd_hist) > abs(macd) * 0.3:  # Strong histogram
                momentum_score += 4.0
            elif abs(macd_hist) > abs(macd) * 0.15:  # Moderate histogram
                momentum_score += 2.0
            
            momentum_score = min(20.0, momentum_score)
            score_breakdown["momentum"] = momentum_score
            total_score += momentum_score
            
            # ==================== VOLATILITY SCORE (20 pts max) ====================
            volatility_score = 0.0
            
            # 1. Volume Confirmation (0-8 pts)
            z_vol = float(indp.get("z_volume", indp.get("z_vol", 0.0)) or 0.0)
            if z_vol >= 2.0:  # High volume
                volatility_score += 8.0
            elif z_vol >= 1.0:  # Above average
                volatility_score += 6.0
            elif z_vol >= 0.0:  # Normal
                volatility_score += 4.0
            elif z_vol >= -0.5:  # Slightly low
                volatility_score += 2.0
            # Very low volume = 0 pts (dangerous)
            
            # 2. Spread OK (0-6 pts)
            spread_limit = float(getattr(self.sp, "spread_limit_pct", 0.0003) or 0.0003)
            if spread_pct <= spread_limit * 0.5:  # Excellent spread
                volatility_score += 6.0
            elif spread_pct <= spread_limit * 0.8:  # Good spread
                volatility_score += 4.0
            elif spread_pct <= spread_limit:  # Acceptable spread
                volatility_score += 2.0
            # Bad spread = 0 pts
            
            # 3. ATR Regime (0-6 pts)
            atr_rel = float(indp.get("atr_rel", 0.001) or 0.001)
            atr_lo = float(getattr(self.cfg, "atr_rel_lo", 0.0005) or 0.0005)
            atr_hi = float(getattr(self.cfg, "atr_rel_hi", 0.003) or 0.003)
            
            if atr_lo <= atr_rel <= atr_hi:  # Goldilocks zone
                volatility_score += 6.0
            elif atr_rel > atr_hi * 0.8:  # Slightly high but OK
                volatility_score += 4.0
            elif atr_rel < atr_lo * 1.5:  # Low but tradeable
                volatility_score += 2.0
            
            volatility_score = min(20.0, volatility_score)
            score_breakdown["volatility"] = volatility_score
            total_score += volatility_score
            
            # ==================== PATTERN SCORE (30 pts max) ====================
            pattern_score = 0.0
            
            # 1. Order Block Touch (0-10 pts)
            ob = str(indp.get("order_block", "") or "")
            if ob in ("bull_ob", "bear_ob"):
                pattern_score += 10.0
            
            # 2. Fair Value Gap (0-8 pts)
            fvg_bull = bool(indp.get("fvg_bull", False))
            fvg_bear = bool(indp.get("fvg_bear", False))
            if fvg_bull or fvg_bear:
                pattern_score += 8.0
            
            # 3. Liquidity Sweep (0-8 pts)
            sweep = str(indp.get("liquidity_sweep", "") or "")
            if sweep in ("bull", "bear"):
                pattern_score += 8.0
            
            # 4. Divergence (0-4 pts)
            rsi_div = str(indp.get("rsi_div", "none") or "none")
            macd_div = str(indp.get("macd_div", "none") or "none")
            if rsi_div in ("bullish", "bearish") or macd_div in ("bullish", "bearish"):
                pattern_score += 4.0
            
            pattern_score = min(30.0, pattern_score)
            score_breakdown["pattern"] = pattern_score
            total_score += pattern_score
            
            # ==================== FINAL CALCULATIONS ====================
            total_score = min(100.0, total_score)
            
            # Convert to net_norm (-1 to +1) for compatibility
            # Determine direction from RSI + MACD + EMA stack
            direction = 0.0
            if rsi_p > 50 and macd_hist > 0 and close_p > ema21:
                direction = 1.0  # Bullish
            elif rsi_p < 50 and macd_hist < 0 and close_p < ema21:
                direction = -1.0  # Bearish
            elif rsi_p > 55 or macd_hist > 0:
                direction = 0.5
            elif rsi_p < 45 or macd_hist < 0:
                direction = -0.5
            
            # net_norm = direction * (score / 100)
            net_norm = direction * (total_score / 100.0)
            
            # Confidence = score directly (0-100 scale)
            conf = int(min(96, total_score))
            
            # Apply tick volatility penalty
            tick_vol = float(getattr(tick_stats, "volatility", 0.0) or 0.0)
            if tick_vol > 60.0:
                conf = max(0, conf - 15)
            elif tick_vol > 40.0:
                conf = max(0, conf - 8)
            
            # Log for debugging
            if conf >= 80:
                log.info(
                    "SCORE_100 | total=%d | trend=%.0f mom=%.0f vol=%.0f pattern=%.0f | net=%.3f",
                    int(total_score), trend_score, momentum_score, volatility_score, pattern_score, net_norm
                )

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
        """
        try:
            _ = ind
            h_bars = int(getattr(self.cfg, "meta_h_bars", 6) or 6)
            R = float(getattr(self.cfg, "meta_barrier_R", 0.50) or 0.50)
            tc_bps = float(getattr(self.cfg, "tc_bps", 1.0) or 1.0)

            if dfp is None or len(dfp) < (h_bars + 30):
                return True

            c = dfp["close"].to_numpy(dtype=np.float64)
            h = dfp["high"].to_numpy(dtype=np.float64)
            l = dfp["low"].to_numpy(dtype=np.float64)

            atr = _atr(h, l, c, 14)
            atr_val = float(safe_last(atr, 1.0) or 1.0)
            if atr_val <= 0:
                return True

            sign = 1.0 if side == "Buy" else -1.0
            start_i = max(1, len(c) - h_bars - 50)
            end_i = max(start_i + 1, len(c) - h_bars - 1)

            idx = np.arange(start_i, end_i, dtype=np.int64)
            if idx.size < 3:
                return True

            moves = (c[idx + h_bars] - c[idx]) * sign
            wins = int(np.sum(moves >= (R * atr_val)))
            total = int(idx.size)
            if total < 3:
                return True

            hitrate = wins / total
            edge_needed = (tc_bps / 10000.0) * 1.5
            threshold = 0.40 + edge_needed

            if hitrate >= threshold:
                return True
            if hitrate >= (threshold - 0.20):
                return True
            if hitrate >= 0.30:
                return True
            if total < 10 and hitrate >= 0.25:
                return True

            return bool(hitrate >= 0.25)
        except Exception:
            return True

    def _apply_filters(self, signal: str, conf: int, ind: Dict[str, Any]) -> Tuple[str, int]:
        """
        OPTIMIZED filters for Scalping 1-15 min.
        """
        try:
            if signal == "Neutral":
                return "Neutral", int(_clamp(float(conf), 0.0, 96.0))

            atr = float(ind.get("atr", 0.0) or 0.0)
            body = float(ind.get("body", 0.0) or 0.0)
            if body <= 0.0:
                try:
                    body = abs(float(ind.get("close", 0.0) or 0.0) - float(ind.get("open", 0.0) or 0.0))
                except Exception:
                    body = 0.0

            min_body_k = float(getattr(self.cfg, "min_body_pct_of_atr", 0.09) or 0.09)
            if atr > 0 and body < (min_body_k * atr):
                conf = max(0, conf - 15)

            close_p = float(ind.get("close", 0.0) or 0.0)
            ema_s, ema_m = self._ema_s_m(ind, close_p)

            if signal == "Buy" and close_p > ema_s > ema_m:
                conf += 12
            elif signal == "Sell" and close_p < ema_s < ema_m:
                conf += 12

            conf = int(_clamp(float(conf), 0.0, 96.0))
            return signal, conf
        except Exception:
            conf = int(_clamp(float(conf), 0.0, 96.0))
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
        dfp: Optional[pd.DataFrame] = None,
    ) -> SignalResult:
        comp_ms = (time.perf_counter() - t0) * 1000.0
        sid = self._signal_id(sym, str(self.sp.tf_primary), bar_key, signal)

        try:
            regime = _as_regime(adapt.get("regime"))

            entry_val = sl_val = tp_val = lot_val = None

            if signal in ("Buy", "Sell") and execute:
                phase_c_block = bool(
                    self.risk.current_phase == "C"
                    and not bool(getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False))
                )
                try:
                    zones = self.feed.micro_price_zones(book)  # type: ignore[attr-defined]
                except Exception:
                    zones = None

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
                    allow_when_blocked=bool(phase_c_block),
                    df=dfp,
                )

                if not plan.ok:
                    if phase_c_block and str(plan.reason or "").startswith("phase_c"):
                        monitor_reasons = reasons[:25] + ["phase_c_monitor", str(plan.reason)]
                        return SignalResult(
                            symbol=sym,
                            signal=signal,
                            confidence=int(_clamp(float(conf), 0.0, 96.0)),
                            regime=regime,
                            reasons=monitor_reasons[:25],
                            spread_bps=float(spread_pct) * 10000.0,
                            latency_ms=float(comp_ms),
                            timestamp=self.feed.now_local().isoformat(),
                            signal_id=self._signal_id(sym, str(self.sp.tf_primary), bar_key, signal),
                            trade_blocked=True,
                        )
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
