from __future__ import annotations

import hashlib
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config_btc import EngineConfig, SymbolParams
from DataFeed.btc_market_feed import MarketFeed, TickStats
from mt5_client import MT5_LOCK, ensure_mt5

from ..btc_indicators import Classic_FeatureEngine, safe_last
from ..btc_risk_management import RiskManager

from .logging_ import log
from .models import SignalResult
from .utils import _atr, _atr_np, _clamp, _side_norm

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

        self._debounce_ms = float(getattr(self.cfg, "decision_debounce_ms", 200.0) or 200.0)
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

    def _unwrap_rates_df(self, ret: Any) -> Optional[pd.DataFrame]:
        """
        feed.get_rates() дар баъзе имплементацияҳо:
          - df
          - (df, meta)
          - (df, meta, extra)
        бармегардонад. Ин функсия ҳамаро устувор мекунад.
        """
        try:
            if ret is None:
                return None
            if isinstance(ret, pd.DataFrame):
                return ret
            if isinstance(ret, (tuple, list)) and ret:
                df = ret[0]
                return df if isinstance(df, pd.DataFrame) else None
            return None
        except Exception:
            return None

    def _sanitize_rates(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        BTC scalping: DataFrame must be usable:
          - columns: open/high/low/close
          - sorted by time (if time column exists)
          - numeric close
        """
        try:
            if df is None or df.empty:
                return None

            # Ensure required OHLC columns exist
            cols = set(df.columns)
            req = {"open", "high", "low", "close"}
            if not req.issubset(cols):
                return None

            # Sort by time if present
            if "time" in cols:
                df = df.sort_values("time")
            elif "datetime" in cols:
                df = df.sort_values("datetime")
            elif "ts" in cols:
                df = df.sort_values("ts")

            # Drop duplicates last bar stability
            if "time" in df.columns:
                df = df.drop_duplicates(subset=["time"], keep="last")
            elif "datetime" in df.columns:
                df = df.drop_duplicates(subset=["datetime"], keep="last")
            elif "ts" in df.columns:
                df = df.drop_duplicates(subset=["ts"], keep="last")

            # Numeric close safety
            c = pd.to_numeric(df["close"], errors="coerce")
            if c.isna().all():
                return None

            # Keep tail only (speed) if huge
            max_bars = int(getattr(self.sp, "bars_primary", 240) or 240)
            if len(df) > max(300, max_bars * 2):
                df = df.tail(max(300, max_bars * 2)).copy()

            return df
        except Exception:
            return None

    def _get_rates_df(self, sym: str, tf: str, *, bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Safe wrapper around feed.get_rates().
        Supports both signatures:
          get_rates(sym, tf) -> df or (df, meta)
          get_rates(sym, tf, bars=...) -> df or (df, meta)
        """
        try:
            # OPTION 1: Specialized BTC feed method
            if bars is not None and hasattr(self.feed, "fetch_rates"):
                # BtcMarketFeed.fetch_rates(tf, bars) uses internal symbol
                ret = self.feed.fetch_rates(tf, bars=int(bars))
            
            # OPTION 2: Generic get_rates logic
            else:
                if bars is None:
                    ret = self.feed.get_rates(sym, tf)
                else:
                    try:
                        ret = self.feed.get_rates(sym, tf, bars=int(bars))
                    except TypeError:
                        # Fallback for feeds that don't support bars arg
                        ret = self.feed.get_rates(sym, tf)

            df = self._unwrap_rates_df(ret)
            if df is None:
                return None
            
            # Manual slice if we got more data than requested (and bars was requested)
            if bars is not None and len(df) > int(bars):
                df = df.tail(int(bars)).copy()

            return self._sanitize_rates(df)
        except Exception:
            return None

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

            # 1) Rates (MTF) — HARDENED (handles tuple returns + sanitizes)
            bars_p = int(getattr(self.sp, "bars_primary", 240) or 240)
            bars_c = int(getattr(self.sp, "bars_confirm", bars_p) or bars_p)
            bars_l = int(getattr(self.sp, "bars_long", bars_p) or bars_p)

            dfp = self._get_rates_df(sym, str(self.sp.tf_primary), bars=bars_p)
            if dfp is None or dfp.empty:
                return self._neutral(sym, ["no_rates_primary"], t0)

            dfc = self._get_rates_df(sym, str(self.sp.tf_confirm), bars=bars_c)
            if dfc is None or dfc.empty:
                dfc = dfp

            dfl = self._get_rates_df(sym, str(self.sp.tf_long), bars=bars_l)
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

            # 3) Phase C flag (continue analysis for monitoring, block orders later)
            self.risk.evaluate_account_state()  # Update phase
            phase_c_block = bool(
                self.risk.current_phase == "C"
                and not bool(getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False))
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

            guard_reasons: List[str] = []
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
                if not phase_c_block:
                    return self._neutral(
                        sym,
                        reasons[:25],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        trade_blocked=True,
                    )
                guard_reasons = reasons[:25]

            # 4.1) BTC Spread Gate (USD-based)
            # BTC has wider spreads, especially on weekends
            # Scalping 1-15min requires higher tolerance
            if bid and ask:
                spread_usd = float(ask - bid)
                
                # ИСЛОҲ: Dynamic spread limit based on price level
                # For BTC at ~$87000-100000, $50 spread = 0.05% (realistic for crypto)
                # ИСЛОҲ: Dynamic spread limit based on mid price + configured spread_limit_pct
                # Prevents 'no signals' when BTC spread is briefly > $50.
                mid_price = float((bid + ask) / 2.0)
                sp_pct = float(getattr(self.sp, "spread_limit_pct", 0.0007) or 0.0007)
                # allow ~1.8x configured pct, clamp to sane USD range
                # HOTFIX: Allow up to 3000 points (approx 30 USD) for BTC
                max_spread_usd = 3000.0 if "BTC" in sym else max(20.0, min(250.0, mid_price * sp_pct * 1.8))
                
                if spread_usd > max_spread_usd:
                    return self._neutral(
                        sym,
                        [f"spread_gate_usd:{spread_usd:.2f}>{max_spread_usd:.2f}"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        trade_blocked=True,
                    )

            # 5) Indicators (MTF)
            tf_data: Dict[str, pd.DataFrame] = {}
            if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                tf_data[str(self.sp.tf_primary)] = dfp
            if isinstance(dfc, pd.DataFrame) and not dfc.empty:
                tf_data[str(self.sp.tf_confirm)] = dfc
            if isinstance(dfl, pd.DataFrame) and not dfl.empty:
                tf_data[str(self.sp.tf_long)] = dfl

            # Feature engine requires M1; if primary != M1, add M1 safely
            if "M1" not in tf_data:
                bars_m1 = int(getattr(self.sp, "bars_m1", bars_p) or bars_p)
                df_m1 = self._get_rates_df(sym, "M1", bars=bars_m1)
                if isinstance(df_m1, pd.DataFrame) and not df_m1.empty:
                    tf_data["M1"] = df_m1

            # SNIPER MODE: Use shift=0 (Real-Time Forming Candle)
            # This eliminates the 1-bar lag. Be careful of repainting, but allows instant entries.
            indicators = self.features.compute_indicators(tf_data, shift=0)
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
            net_abs = abs(net_norm)

            # ==============================================================
            # SNIPER LOGIC #1: Volume Validation (Strict & Tiered)
            # HOTFIX: Lower threshold to 15 (aggressive entry)
            # ==============================================================
            try:
                if dfp is not None and "tick_volume" in dfp.columns and len(dfp) >= 20:
                    bar_age_ok = last_age >= 5.0
                    if bar_age_ok:
                        now_vol = float(dfp["tick_volume"].iloc[-1])
                        # HOTFIX: Priority on Speed. Use hard floor of 5.
                        if now_vol < 5.0:
                             return self._neutral(
                                sym,
                                [f"low_volume_sniper:{now_vol:.0f}<5"],
                                t0,
                                spread_pct=spread_pct,
                                bar_key=bar_key,
                                regime=str(adapt.get("regime")),
                                trade_blocked=True,
                            )
            except Exception:
                pass

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
            
            # === ИСЛОҲ: RELAXED TREND CHECKS FOR CRYPTO VOLATILITY ===
            # BTC дар 24/7 савдо мешавад ва тезтар аз XAU ҳаракат мекунад
            ema_tolerance = 0.001  # 0.1% tolerance (higher for crypto)
            
            # Buy: Нархи ҷорӣ аз EMA боло бошад
            trend_ok_buy = close_p > ema_s * (1 - ema_tolerance)
            # Sell: Нархи ҷорӣ аз EMA поён бошад
            trend_ok_sell = close_p < ema_s * (1 + ema_tolerance)
            
            # Allow BREAKOUT trades when ADX is high (strong trend developing)
            adx_p = float(indp.get("adx", 0.0) or 0.0)
            
            # SNIPER OPTIMIZATION: Faster reaction for crypto
            # 1. Lower ADX threshold for breakout detection (was 25, now 18)
            # 2. Allow "Sniper" entry if net_norm is very strong (momentum), even if price is slightly lagging EMA
            
            is_sniper_momentum = abs(net_norm) > 0.15  # Strong momentum
            
            if adx_p > 18.0 or is_sniper_momentum:
                if net_norm > 0.08:  # Bullish breakout/momentum
                    trend_ok_buy = True
                elif net_norm < -0.08:  # Bearish breakout/momentum
                    trend_ok_sell = True

            net_norm_abs = abs(net_norm)

            # Ultra-confidence must be earned: require strong strength OR confluence.
            # Prevents "fake 96%" spam in choppy conditions.
            has_confluence = bool(sweep) or (div != "none")
            if conf >= 90 and (not has_confluence) and net_norm_abs < 0.20:
                conf = min(conf, 89)

            if conf >= conf_min and net_norm_abs >= min_net_norm_for_signal:
                if net_norm > net_thr:
                    if trend_ok_buy:    
                        signal = "Buy"
                elif net_norm < -net_thr:
                    if trend_ok_sell:
                        signal = "Sell"

            # 10.1) Confirmation gate (prevents weak/false signals in chop)
            if signal in ("Buy", "Sell"):
                adx_p = float(indp.get("adx", 0.0) or 0.0)
                
                # Side-aware confluence check
                # A BUY signal needs BULLISH confluence (not bearish)
                is_buy = (signal == "Buy")
                
                sweep_ok = (sweep == "bull") if is_buy else (sweep == "bear")
                div_ok = (div == "bullish") if is_buy else (div == "bearish")
                fvg_ok = bool(indp.get("fvg_bull", False)) if is_buy else bool(indp.get("fvg_bear", False))
                
                # Penalty for contradicton
                sweep_bad = (sweep == "bear") if is_buy else (sweep == "bull")
                div_bad = (div == "bearish") if is_buy else (div == "bullish")
                
                if sweep_bad or div_bad:
                    # Downgrade confidence significantly if trying to trade against structure
                    conf = max(0, int(conf) - 30)
                    # If contradiction is strong, kill the signal or force Neutral
                    if conf < conf_min:
                        signal = "Neutral"

                has_confluence = sweep_ok or div_ok or fvg_ok

                # If not enough market structure confirmation, require stronger strength
                if (not has_confluence) and adx_p < 18.0 and abs(net_norm) < 0.15:
                    return self._neutral(
                        sym,
                        ["no_confluence"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=str(adapt.get("regime")),
                        trade_blocked=True,
                    )
                
                # Retest signal validity after penalties
                if signal == "Neutral":
                     return self._neutral(
                        sym,
                        [f"contradiction_block:sweep={sweep},div={div}"],
                        t0,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        regime=str(adapt.get("regime")),
                        trade_blocked=True,
                    )

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

            # Single consistent cap (avoid double conflicting caps)
            conf = self._cap_conf_by_strength(int(conf), abs(net_norm))

            # Risk-level emission gate (Missed Opportunity Logging)
            if signal in ("Buy", "Sell") and hasattr(self.risk, "can_emit_signal"):
                try:
                    allowed, gate_reasons = self.risk.can_emit_signal(int(conf), getattr(self.feed, "tz", None))
                    if not bool(allowed):
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
                            regime=str(adapt.get("regime")),
                            trade_blocked=True,
                        )
                except Exception:
                    pass

            # ==============================================================
            # SNIPER LOGIC #2: Dynamic ATR Stop Loss ("Toqatfarso")
            # Widen SL for strong signals to survive noise
            # ==============================================================
            if conf >= 85:
                old_sl = float(adapt.get("sl_mult", 1.35))
                adapt["sl_mult"] = old_sl * 1.2 
                min_tp = adapt["sl_mult"] * 2.0
                if float(adapt.get("tp_mult", 0.0)) < min_tp:
                    adapt["tp_mult"] = min_tp

            # 14) Finalize (plan only)
            reasons: List[str] = list(guard_reasons)
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
                dfp=dfp,  # Pass dataframe for structure analysis
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
                # STABILIZATION FIX: round to nearest second
                tt = tt.replace(microsecond=0)
                return tt.strftime("%Y-%m-%dT%H:%M:%S")
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

        # Adaptive sizing only (no static lot fallbacks)
        fixed_vol = float(getattr(self.cfg, "fixed_volume", 0.0) or 0.0)
        use_dynamic = True

        return {
            "conf_min": int(conf_min),
            "sl_mult": float(getattr(self.cfg, "sl_atr_mult_trend", 1.35) or 1.35),
            "tp_mult": float(getattr(self.cfg, "tp_atr_mult_trend", 3.2) or 3.2),
            "trail_mult": float(getattr(self.cfg, "trail_atr_mult", 1.0) or 1.0),
            "w_mul": {"trend": 1.0, "momentum": 1.0, "meanrev": 1.0, "structure": 1.0, "volume": 1.0},
            "regime": "trend",
            "phase": phase,
            "use_dynamic_sizing": True,
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
        """
        INSTITUTIONAL 100-POINT SCORING SYSTEM (BTC)
        
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
            
            # 2. Spread OK (0-6 pts) - BTC has tighter spread requirements
            spread_limit = float(getattr(self.sp, "spread_limit_pct", 0.0005) or 0.0005)
            if spread_pct <= spread_limit * 0.5:  # Excellent spread
                volatility_score += 6.0
            elif spread_pct <= spread_limit * 0.8:  # Good spread
                volatility_score += 4.0
            elif spread_pct <= spread_limit:  # Acceptable spread
                volatility_score += 2.0
            
            # 3. ATR Regime (0-6 pts)
            atr_pct = float(indp.get("atr_pct", 0.001) or 0.001)
            atr_lo = float(getattr(self.cfg, "atr_pct_lo", 0.0008) or 0.0008)
            atr_hi = float(getattr(self.cfg, "atr_pct_hi", 0.005) or 0.005)
            
            if atr_lo <= atr_pct <= atr_hi:  # Goldilocks zone
                volatility_score += 6.0
            elif atr_pct > atr_hi * 0.8:  # Slightly high but OK
                volatility_score += 4.0
            elif atr_pct < atr_lo * 1.5:  # Low but tradeable
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
            direction = 0.0
            if rsi_p > 50 and macd_hist > 0 and close_p > ema21:
                direction = 1.0  # Bullish
            elif rsi_p < 50 and macd_hist < 0 and close_p < ema21:
                direction = -1.0  # Bearish
            elif rsi_p > 55 or macd_hist > 0:
                direction = 0.5
            elif rsi_p < 45 or macd_hist < 0:
                direction = -0.5
            
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
            
            # FAIL-OPEN: If ATR is invalid, don't block. Let RiskManager handle sizing/stops.
            if atr_val <= 1e-9:
                return True

            sign = 1.0 if _side_norm(side) == "Buy" else -1.0
            wins = 0
            total = 0

            # Look back for samples
            lookback_window = max(40, h_bars * 10)
            start_i = max(1, len(c) - lookback_window)
            end_i = max(start_i + 1, len(c) - h_bars - 1)
            
            if end_i <= start_i or (end_i - start_i) < 5:
                # FAIL-OPEN: Not enough samples to judge
                return True

            for i in range(start_i, end_i):
                if (i + h_bars) >= len(c):
                    continue
                move = (c[i + h_bars] - c[i]) * sign
                if move >= (R * atr_val):
                    wins += 1
                total += 1

            if total < 5:  # Reduced from 10
                return True

            hitrate = wins / total
            edge_needed = (tc_bps / 10000.0) * 2.0
            threshold = 0.45 + edge_needed
            
            # === OPTIMIZED FOR BTC SCALPING ===
            # Even wider margin (20%) for BTC
            if hitrate >= (threshold - 0.20):
                return True
            
            # Absolute floor: if > 30% hitrate in choppy BTC, allow it
            if hitrate >= 0.30:
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
        dfp: Optional[pd.DataFrame] = None,
    ) -> SignalResult:
        comp_ms = (time.perf_counter() - t0) * 1000.0
        sid = self._signal_id(sym, str(self.sp.tf_primary), bar_key, signal)

        try:
            regime = str(adapt.get("regime") or None)
            entry_val = sl_val = tp_val = lot_val = None

            if signal in ("Buy", "Sell") and execute:
                phase_c_block = bool(
                    self.risk.current_phase == "C"
                    and not bool(getattr(self.risk.cfg, "ignore_daily_stop_for_trading", False))
                )
                trade_dec = self.risk.can_trade(float(conf) / 100.0, signal)
                if not trade_dec.allowed:
                    if phase_c_block:
                        zones = self.feed.micro_price_zones(book)
                        tick_vol = float(getattr(tick_stats, "volatility", 0.0) or 0.0)
                        open_pos, unreal_pl = self._position_context(sym)

                        entry_val, sl_val, tp_val, lot_val, block_reason = self.risk.plan_order(
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
                            allow_when_blocked=True,
                            df=dfp,
                        )

                        block_reasons = list(trade_dec.reasons or [])
                        monitor_reasons = (reasons or []) + ["phase_c_monitor"] + block_reasons
                        return SignalResult(
                            symbol=sym,
                            signal=signal,
                            confidence=int(_clamp(float(conf), 0.0, 96.0)),
                            regime=regime,
                            reasons=monitor_reasons[:25],
                            spread_bps=float(spread_pct) * 10000.0,
                            latency_ms=float(comp_ms),
                            timestamp=self.feed.now_local().isoformat(),
                            entry=None if entry_val is None else float(entry_val),
                            sl=None if sl_val is None else float(sl_val),
                            tp=None if tp_val is None else float(tp_val),
                            lot=None if lot_val is None else float(lot_val),
                            signal_id=sid,
                            trade_blocked=True,
                        )

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

                entry_val, sl_val, tp_val, lot_val, plan_reason = self.risk.plan_order(
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
                    allow_when_blocked=False,
                    df=dfp,
                )

                if entry_val is None or sl_val is None or tp_val is None or lot_val is None:
                    fail_reason = str(plan_reason) if plan_reason else "plan_order_failed"
                    return self._neutral(
                        sym,
                        [fail_reason],
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

