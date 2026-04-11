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
from .data_integrity import (
    DATA_INCOMPLETE,
    DATA_VALID,
    Validator,
)
from .feature_engine import (
    FEATURES_VALID,
    FeatureIntegrityError,
)
from .models import SignalResult
from .risk_engine import RiskManager
from .utils import _is_finite, _side_norm, clamp01

log = logging.getLogger("core.signal_engine")

_DUMMY_LOCK = None


def _mt5_lock():
    global _DUMMY_LOCK
    # Lazy import avoids hard dependency and circular imports at module load.
    try:
        from mt5_client import MT5_LOCK as lock  # type: ignore
        return lock
    except Exception:
        if _DUMMY_LOCK is None:
            import threading
            _DUMMY_LOCK = threading.RLock()
        return _DUMMY_LOCK


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
        self._macro_symbol_cache: Dict[str, Optional[str]] = {}
        self._macro_ctx_cache_ts: float = 0.0
        self._macro_ctx_cache: Dict[str, float] = {}
        self._validator = Validator(cfg, sp)
        self._data_audits: Dict[Tuple[str, str], Any] = {}
        self._last_market_audit: Optional[Any] = None

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
        Normalize OHLCV column names and numeric dtypes without mutating
        chronology. The strict integrity audit decides whether ordering, gaps,
        timezone handling, or schema issues are acceptable.
        """
        if df is None or len(df) == 0:
            return None

        df = df.copy()

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

        for col in ("open", "high", "low", "close", "tick_volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if len(df) < 10:
            return None
        return df

    def _get_rates_df(
        self, sym: str, tf: str, *, bars: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Safe wrapper around feed.get_rates()."""
        try:
            if bars is not None:
                try:
                    ret = self._feed.get_rates(sym, tf, bars=bars)
                except TypeError as exc:
                    msg = str(exc)
                    if "unexpected keyword argument 'bars'" not in msg:
                        raise
                    ret = self._feed.get_rates(sym, tf)
            else:
                ret = self._feed.get_rates(sym, tf)
            df = self._unwrap_rates_df(ret)
            if df is not None and bars is not None and len(df) > bars:
                df = df.iloc[-bars:].copy()
            df = self._sanitize_rates(df)
            audit = self._validator.validate_ohlcv(df, tf, symbol=sym)
            self._data_audits[(sym, tf)] = audit
            if not audit.tradable:
                return None
            return audit.normalized_df
        except Exception as exc:
            log.error("_get_rates_df(%s, %s): %s", sym, tf, exc)
            self._data_audits[(sym, tf)] = self._validator.validate_ohlcv(
                None,
                tf,
                symbol=sym,
            )
            return None

    def _audit_for(self, sym: str, tf: str) -> Optional[Any]:
        return self._data_audits.get((sym, tf))

    def _audit_failure_result(
        self,
        sym: str,
        tf: str,
        t0: float,
        fallback_reason: str,
    ) -> SignalResult:
        audit = self._audit_for(sym, tf)
        reasons = [fallback_reason]
        data_state = DATA_INCOMPLETE
        if audit is not None:
            reasons = audit.reason_codes()[:10]
            data_state = str(getattr(audit, "state", DATA_INCOMPLETE) or DATA_INCOMPLETE)
            if fallback_reason not in reasons:
                reasons.append(fallback_reason)
        return self._neutral(
            sym,
            reasons,
            t0,
            trade_blocked=True,
            data_state=data_state,
        )

    def _quote_snapshot(self, sym: str) -> Tuple[float, float, Optional[Any]]:
        bid = ask = 0.0
        tick_timestamp: Optional[Any] = None

        try:
            snap_fn = getattr(self._feed, "last_quote_snapshot", None)
            if callable(snap_fn):
                snap = snap_fn()
                if isinstance(snap, (tuple, list)) and len(snap) >= 3:
                    bid = float(snap[0] or 0.0)
                    ask = float(snap[1] or 0.0)
                    tick_timestamp = snap[2]
                    return bid, ask, tick_timestamp
        except Exception:
            pass

        try:
            snap_fn = getattr(self._feed, "_last_quote_snapshot", None)
            if callable(snap_fn):
                snap = snap_fn()
                if isinstance(snap, (tuple, list)) and len(snap) >= 3:
                    bid = float(snap[0] or 0.0)
                    ask = float(snap[1] or 0.0)
                    tick_timestamp = snap[2]
                    return bid, ask, tick_timestamp
        except Exception:
            pass

        try:
            quote_fn = getattr(self._feed, "quote", None)
            if callable(quote_fn):
                snap = quote_fn()
                if isinstance(snap, (tuple, list)) and len(snap) >= 2:
                    bid = float(snap[0] or 0.0)
                    ask = float(snap[1] or 0.0)
        except Exception:
            bid = ask = 0.0

        try:
            if mt5 is not None:
                with _mt5_lock():
                    tick = mt5.symbol_info_tick(sym)
                if tick is not None:
                    bid = float(getattr(tick, "bid", bid) or bid)
                    ask = float(getattr(tick, "ask", ask) or ask)
                    tick_timestamp = (
                        getattr(tick, "time_msc", None)
                        or getattr(tick, "time", None)
                    )
        except Exception:
            pass

        return bid, ask, tick_timestamp

    def _symbol_info_snapshot(self, sym: str) -> Optional[Any]:
        try:
            info_fn = getattr(self._feed, "symbol_info", None)
            if callable(info_fn):
                info = info_fn(sym)
                if info is not None:
                    return info
        except Exception:
            pass

        try:
            if mt5 is not None:
                with _mt5_lock():
                    return mt5.symbol_info(sym)
        except Exception:
            pass
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
                return self._audit_failure_result(
                    sym,
                    self.sp.tf_primary,
                    t0,
                    "no_primary_data",
                )

            dfc = self._get_rates_df(sym, self.sp.tf_confirm)
            if dfc is None or len(dfc) < 30:
                return self._audit_failure_result(
                    sym,
                    self.sp.tf_confirm,
                    t0,
                    "no_confirm_data",
                )

            dfl = self._get_rates_df(sym, self.sp.tf_long)
            if dfl is None or len(dfl) < 30:
                return self._audit_failure_result(
                    sym,
                    self.sp.tf_long,
                    t0,
                    "no_long_data",
                )

            # D1 timeframe for daily trend confluence
            tf_daily = getattr(self.sp, "tf_daily", "D1")
            d1_bars = max(int(getattr(self.cfg, "d1_bars_required", 60) or 60), 60)
            dfd = self._get_rates_df(
                sym, tf_daily,
                bars=d1_bars,
            )
            if dfd is None or len(dfd) < 30:
                return self._audit_failure_result(
                    sym,
                    tf_daily,
                    t0,
                    "no_daily_data",
                )
            # Dedicated H1 stream for vector alignment math (M1/M15/H1/D1)
            if str(self.sp.tf_long).upper() == "H1":
                dfh = dfl
            else:
                dfh = self._get_rates_df(sym, "H1", bars=120)
                if dfh is None or len(dfh) < 30:
                    return self._audit_failure_result(
                        sym,
                        "H1",
                        t0,
                        "no_h1_data",
                    )
            # Dedicated H4 stream for institutional flow confluence gate.
            if str(self.sp.tf_long).upper() == "H4":
                dfh4 = dfl
            else:
                dfh4 = self._get_rates_df(sym, "H4", bars=120)
                if dfh4 is None or len(dfh4) < 30:
                    return self._audit_failure_result(
                        sym,
                        "H4",
                        t0,
                        "no_h4_data",
                    )

            bid, ask, tick_timestamp = self._quote_snapshot(sym)
            symbol_info = self._symbol_info_snapshot(sym)
            market_audit = self._validator.validate_market_context(
                df=dfp,
                timeframe=self.sp.tf_primary,
                symbol=sym,
                bid=bid,
                ask=ask,
                tick_timestamp=tick_timestamp,
                symbol_info=symbol_info,
                now=time.time(),
            )
            self._last_market_audit = market_audit
            if not market_audit.tradable:
                return self._neutral(
                    sym,
                    market_audit.reason_codes()[:10],
                    t0,
                    trade_blocked=True,
                    data_state=market_audit.state,
                )

            # ── 1b. Volatility circuit breaker (Black Swan) ──
            if self._rm.check_volatility_circuit_breaker(dfp):
                return self._neutral(
                    sym, ["vol_circuit_breaker"], t0,
                    trade_blocked=True,
                    data_state=DATA_VALID,
                )

            # ── 2. Compute indicators ──
            df_dict = {
                self.sp.tf_primary: dfp,
                self.sp.tf_confirm: dfc,
                self.sp.tf_long: dfl,
            }
            if dfh is not None and len(dfh) > 0:
                df_dict["H1"] = dfh
            if dfh4 is not None and len(dfh4) > 0:
                df_dict["H4"] = dfh4

            try:
                indicators = self._fe.compute_indicators(df_dict, shift=1)
            except FeatureIntegrityError as exc:
                audit = exc.audit
                return self._neutral(
                    sym,
                    audit.reason_codes()[:10],
                    t0,
                    trade_blocked=True,
                    data_state=DATA_VALID,
                    feature_state=audit.state,
                )
            except Exception as exc:
                return self._neutral(
                    sym,
                    [f"indicator_error:{exc}"],
                    t0,
                    feature_state=FEATURES_VALID,
                )

            if not indicators:
                return self._neutral(
                    sym,
                    ["no_indicators"],
                    t0,
                    feature_state=FEATURES_VALID,
                )

            indp = indicators.get(self.sp.tf_primary, {})
            indc = indicators.get(self.sp.tf_confirm, {})
            indl = indicators.get(self.sp.tf_long, {})
            ind_h1 = indicators.get("H1", {})
            ind_h4 = indicators.get("H4", {})
            regime_timestamp: Optional[Any] = None
            market_regime = "normal"

            if not indp:
                return self._neutral(
                    sym,
                    ["no_primary_indicators"],
                    t0,
                    feature_state=FEATURES_VALID,
                )

            try:
                if "time" in dfp.columns and len(dfp) >= 2:
                    regime_timestamp = dfp["time"].iloc[-2]
                elif len(dfp.index) >= 2:
                    regime_timestamp = dfp.index[-2]
            except Exception:
                regime_timestamp = None

            market_regime = self._fe.classify_market_regime(
                indicators,
                asset=self.sp.base,
                now=regime_timestamp,
                primary_tf=self.sp.tf_primary,
            )

            # ── 3. Market state ──
            bid, ask = self._tick_bid_ask(sym)
            if bid <= 0 or ask <= 0:
                return self._neutral(sym, ["no_tick"], t0, regime=market_regime)

            spread_pct = (ask - bid) / bid if bid > 0 else 0.0
            bar_key = self._bar_key(dfp)
            if bar_key != "no_bar":
                if bar_key == self._last_bar_key:
                    return self._neutral(
                        sym, ["same_bar_dedup"], t0,
                        spread_pct=spread_pct, bar_key=bar_key,
                        regime=market_regime,
                    )
                self._last_bar_key = bar_key

            # BTC flash-crash guard
            flash_ok, flash_reason = self._flash_crash_guard(dfp)
            if not flash_ok:
                return self._neutral(
                    sym, reasons + [flash_reason], t0,
                    spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                    regime=market_regime,
                )

            # ATR for regime detection
            atr_pct = float(indp.get("atr_pct", 0.0))
            atr_val = float(indp.get("atr", 0.0))

            # Tick stats from feed
            tick_stats = None
            tick_stats_fn = getattr(self._feed, "tick_stats", None)
            if callable(tick_stats_fn):
                try:
                    tick_stats = tick_stats_fn(dfp)
                except TypeError:
                    tick_stats = tick_stats_fn()
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
                    regime=market_regime,
                )

            # ── 5. Ensemble scoring ──
            book = {}
            fetch_book_fn = getattr(self._feed, "fetch_book", None)
            if callable(fetch_book_fn):
                try:
                    book = fetch_book_fn(levels=20) or {}
                except Exception:
                    book = {}
            elif isinstance(getattr(self._feed, "order_book", None), dict):
                book = getattr(self._feed, "order_book", {})

            score_result = self._ensemble_score(
                indp, indc, indl, book, adapt, spread_pct, tick_stats,
                dfd=dfd, dfp=dfp, dfl=dfl, dfh=dfh,
            )

            net_score = score_result.get("net_score", 0.0)
            net_abs = abs(net_score)
            signal_dir = score_result.get("direction", "Neutral")
            sub_reasons = score_result.get("reasons", [])
            reasons.extend(sub_reasons)
            entry_flags = self._entry_quality_flags(
                signal=signal_dir,
                indp=indp,
                dfp=dfp,
                net_score=float(net_score),
            )
            score_floor = float(self.cfg.signal_min_score)
            if signal_dir in ("Buy", "Sell") and bool(entry_flags.get("early_trigger", False)):
                score_floor = max(58.0, score_floor - 8.0)
                reasons.append(f"early_ignition:score_floor={score_floor:.1f}")

            # ── 6. Signal decision ──
            if net_abs < score_floor:
                return self._neutral(
                    sym, reasons + [f"weak_score:{net_abs:.1f}<{score_floor:.1f}"], t0,
                    spread_pct=spread_pct, bar_key=bar_key,
                    regime=market_regime,
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
            if signal_dir in ("Buy", "Sell") and bool(entry_flags.get("late_chase", False)):
                ext_atr = float(entry_flags.get("extension_atr", 0.0) or 0.0)
                if bool(entry_flags.get("hard_veto", False)):
                    return self._neutral(
                        sym,
                        reasons + [f"late_chase_veto:{ext_atr:.2f}atr"],
                        t0,
                        confidence=conf,
                        spread_pct=spread_pct,
                        bar_key=bar_key,
                        trade_blocked=True,
                        regime=market_regime,
                    )
                conf = max(0, conf - 12)
                reasons.append(f"late_chase_penalty:{ext_atr:.2f}atr")
            elif signal_dir in ("Buy", "Sell") and bool(entry_flags.get("early_trigger", False)):
                conf = min(100, conf + 4)
                reasons.append("early_ignition_bonus")

            # Stop-hunt dominance: trade with trap resolution, not against it.
            try:
                sh_side = str(indp.get("stop_hunt_side", "") or "")
                sh_strength = float(indp.get("stop_hunt_strength", 0.0) or 0.0)
                sh_min = float(getattr(self.cfg, "stop_hunt_min_strength", 0.30) or 0.30)
                if signal_dir in ("Buy", "Sell") and abs(sh_strength) >= sh_min:
                    aligned = (signal_dir == "Buy" and sh_strength > 0.0) or (
                        signal_dir == "Sell" and sh_strength < 0.0
                    )
                    if aligned:
                        bonus = int(max(0, int(getattr(self.cfg, "stop_hunt_align_bonus", 8) or 8)))
                        conf = min(100, conf + bonus)
                        reasons.append(f"stop_hunt_align:{sh_side}:{sh_strength:+.2f}")
                    else:
                        veto_thr = float(
                            getattr(self.cfg, "stop_hunt_conflict_veto_strength", 0.55) or 0.55
                        )
                        if abs(sh_strength) >= veto_thr:
                            return self._neutral(
                                sym,
                                reasons + [f"stop_hunt_conflict_veto:{sh_side}:{sh_strength:+.2f}"],
                                t0,
                                confidence=conf,
                                spread_pct=spread_pct,
                                bar_key=bar_key,
                                trade_blocked=True,
                                regime=market_regime,
                            )
                        pen = int(max(0, int(getattr(self.cfg, "stop_hunt_conflict_penalty", 16) or 16)))
                        conf = max(0, conf - pen)
                        reasons.append(f"stop_hunt_conflict:{sh_side}:{sh_strength:+.2f}")
            except Exception:
                pass

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
            signal_dir, conf = self._apply_filters(
                signal_dir,
                conf,
                indp,
                indc,
                indl,
                reasons,
                dfd=dfd,
                ind_h1=ind_h1,
                ind_h4=ind_h4,
                df_m15=(dfl if dfl is not None else dfp),
            )

            # Cap confidence by strength
            conf = self._cap_conf_by_strength(conf, net_abs)

            # ── SNIPER FILTER ──
            # Configurable confidence floor. Hard minimum 70% prevents noise trades.
            # Default 75% balances selectivity with signal rate.
            _SNIPER_HARD_MIN = 70
            sniper_floor = max(int(getattr(self.cfg, "min_confidence", 75) or 75), _SNIPER_HARD_MIN)
            if conf < sniper_floor:
                return self._neutral(
                    sym, reasons + [f"sniper_reject:{conf}<{sniper_floor}"], t0,
                    confidence=conf, spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                    regime=market_regime,
                )

            # Can emit?
            can_emit, emit_reason = self._rm.can_emit_signal(conf)
            if not can_emit:
                return self._neutral(
                    sym, reasons + [f"emit_blocked:{emit_reason}"], t0,
                    confidence=conf, spread_pct=spread_pct, bar_key=bar_key,
                    trade_blocked=True,
                    regime=market_regime,
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
                structure_frames=(dfp, dfc, dfl, dfh),
                market_regime=market_regime,
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
                with _mt5_lock():
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
            if df is None or len(df) == 0:
                return "no_bar"
            idx = -2 if len(df) >= 2 else -1
            last = df.iloc[idx]
            t = last.get("time", 0)
            ts = self._epoch_seconds(t)
            if ts is None:
                return "no_bar"
            return str(int(ts))
        except Exception:
            return "no_bar"

    def _last_bar_age(self, df: pd.DataFrame) -> float:
        try:
            if "time" in df.columns:
                last_ts = self._epoch_seconds(df.iloc[-1]["time"])
                if last_ts is not None:
                    return max(0.0, time.time() - last_ts)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _epoch_seconds(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return float(value.tz_localize("UTC").timestamp())
            return float(value.tz_convert("UTC").timestamp())
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except Exception:
                pass
        try:
            raw = float(value)
            if raw > 10**12:
                raw /= 1000.0
            return raw
        except Exception:
            return None

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
        data_state: str = DATA_VALID,
        feature_state: str = FEATURES_VALID,
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
            data_state=data_state,
            feature_state=feature_state,
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
            "stop_hunt_strength": float(indp.get("stop_hunt_strength", 0.0) or 0.0),
            "ob_touch_proximity": float(indp.get("ob_touch_proximity", 0.0) or 0.0),
            "ob_pretouch_bias": float(indp.get("ob_pretouch_bias", 0.0) or 0.0),
            "close": float(indp.get("close", 0.0) or 0.0),
            "vwap": float(indp.get("vwap", 0.0) or 0.0),
            "ema_medium": float(indp.get("ema_medium", 0.0) or 0.0),
            "bb_pctb": float(indp.get("bb_pctb", 0.5) or 0.5),
            "rsi": float(indp.get("rsi", 50.0) or 50.0),
        }

    # ─── Scoring system ──────────────────────────────────────────────

    @staticmethod
    def _book_imbalance(book: Dict[str, Any]) -> float:
        """Compute top-of-book pressure imbalance in [-1, +1]."""
        try:
            if not isinstance(book, dict):
                return 0.0
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            if not bids or not asks:
                return 0.0
            bid_vol = float(np.sum([float(v) for _, v in bids[:20]]))
            ask_vol = float(np.sum([float(v) for _, v in asks[:20]]))
            denom = bid_vol + ask_vol
            if denom <= 0.0:
                return 0.0
            imb = (bid_vol - ask_vol) / denom
            return max(-1.0, min(1.0, float(imb)))
        except Exception:
            return 0.0

    def _ensemble_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        book: Dict[str, Any],
        adapt: Dict[str, Any],
        spread_pct: float,
        tick_stats: Any,
        *,
        dfd: Optional[pd.DataFrame] = None,
        dfp: Optional[pd.DataFrame] = None,
        dfl: Optional[pd.DataFrame] = None,
        dfh: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        INSTITUTIONAL 100-POINT SCORING SYSTEM

        Components:
          - Trend Score (30 pts): H1 slope alignment + EMA stack + ADX strength
          - Momentum Score (20 pts): RSI zone + MACD crossover + histogram direction
          - Volatility Score (15 pts): ATR regime + BB width
          - Structure Score (15 pts): FVG + liquidity sweep + order blocks
          - Flow Score (10 pts): OFI-style blend from book/tick imbalance
          - Tick Momentum (10 pts): Micro-trend pressure from tick stream
          - Volume Delta (10 pts): Signed flow imbalance from delta statistics
          - Mean-Reversion Score (10 pts): BB touch + RSI extremes
        """
        weights = self._get_base_weights()

        h1_slope = 0.0
        try:
            h1_slope = self._linreg_slope_norm(self._close_array(dfh), window=20)
        except Exception:
            h1_slope = 0.0

        trend = self._trend_score(
            indp,
            indc,
            indl,
            adapt.get("regime", "normal"),
            h1_slope=h1_slope,
        )
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

        # Microstructure edge stack:
        # - flow_imb: OFI-style pressure from tick/book imbalance
        # - vol_delta: signed volume delta proxy from tick stream
        # - tick_mom: normalized micro-trend pressure
        tick_imb = 0.0
        tick_delta = 0.0
        aggr_delta = 0.0
        micro_trend = 0.0
        if tick_stats is not None:
            imb = getattr(tick_stats, "imbalance", 0.0)
            td = getattr(tick_stats, "tick_delta", 0.0)
            ad = getattr(tick_stats, "aggr_delta", td)
            mt = getattr(tick_stats, "micro_trend", 0.0)
            tick_imb = float(imb) if _is_finite(imb) else 0.0
            tick_delta = float(td) if _is_finite(td) else 0.0
            aggr_delta = float(ad) if _is_finite(ad) else 0.0
            micro_trend = float(mt) if _is_finite(mt) else 0.0
        tick_imb = max(-1.0, min(1.0, tick_imb))
        tick_delta = max(-1.0, min(1.0, tick_delta))
        aggr_delta = max(-1.0, min(1.0, aggr_delta))
        book_imb = self._book_imbalance(book)
        flow_imb = max(-1.0, min(1.0, 0.55 * tick_imb + 0.45 * book_imb))

        vol_delta = max(-1.0, min(1.0, 0.60 * tick_delta + 0.40 * aggr_delta))
        tstat_ref = float(getattr(self.sp, "micro_tstat_thresh", 0.5) or 0.5)
        tstat_ref = max(1e-6, tstat_ref)
        tick_mom = max(-1.0, min(1.0, micro_trend / tstat_ref))

        micro_edge = max(-1.0, min(1.0, 0.45 * flow_imb + 0.35 * vol_delta + 0.20 * tick_mom))
        flow_score = abs(micro_edge)

        # Momentum ignition (early move trigger)
        ignition_score = self._momentum_ignition_score(dfp)

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

        # Volatility — amplifies whichever side is already leading.
        # Previous code added identical amounts to both sides, which cancelled
        # in net = buy - sell (10% of weight budget was dead).
        if buy_score >= sell_score:
            buy_score += vol_score * weights["volatility"]
        else:
            sell_score += vol_score * weights["volatility"]

        # Flow (OFI-style combined edge)
        if micro_edge > 0.0:
            buy_score += flow_score * weights.get("flow", 0.0)
        elif micro_edge < 0.0:
            sell_score += flow_score * weights.get("flow", 0.0)

        # Explicit tick-momentum weighting
        tick_mom_w = float(weights.get("tick_momentum", 0.0) or 0.0)
        if tick_mom > 0.0:
            buy_score += abs(tick_mom) * tick_mom_w
        elif tick_mom < 0.0:
            sell_score += abs(tick_mom) * tick_mom_w

        # Explicit volume-delta weighting
        vol_delta_w = float(weights.get("volume_delta", 0.0) or 0.0)
        if vol_delta > 0.0:
            buy_score += abs(vol_delta) * vol_delta_w
        elif vol_delta < 0.0:
            sell_score += abs(vol_delta) * vol_delta_w

        # Ignition adds directional weight before lagging confirmation closes.
        if ignition_score > 0.0 and dfp is not None and len(dfp) >= 22:
            try:
                c_arr = self._close_array(dfp)
                if c_arr is not None and len(c_arr) >= 22:
                    breakout_hi = float(np.max(c_arr[-21:-1]))
                    breakout_lo = float(np.min(c_arr[-21:-1]))
                    last_close = float(c_arr[-1])
                    ign_weight = min(3.0, 3.0 * ignition_score)
                    if last_close > breakout_hi:
                        buy_score += ign_weight
                    elif last_close < breakout_lo:
                        sell_score += ign_weight
                    else:
                        # No clean directional break: add a small neutral boost.
                        buy_score += ign_weight * 0.25
                        sell_score += ign_weight * 0.25
            except Exception:
                pass

        net = buy_score - sell_score

        # ── D1 Confluence Bonus/Penalty ──
        d1_weight = float(getattr(self.cfg, 'd1_confluence_weight', 5.0) or 5.0)
        d1_result = self._d1_confluence_score(dfd)
        d1_val = d1_result.get("value", 0.0)
        d1_aligned = False
        if d1_val != 0.0:
            if (net > 0 and d1_val > 0) or (net < 0 and d1_val < 0):
                # D1 aligns with signal → bonus
                net += d1_val * d1_weight
                d1_aligned = True
            else:
                # D1 conflicts → penalty
                net -= abs(d1_val) * d1_weight * 0.6

        # ── Vector Alignment Bonus (MTF confluence) ──
        vec_score = self._vector_alignment(
            indp=indp,
            indc=indc,
            indl=indl,
            dfp=dfp,
            dfl=dfl,
            dfh=dfh,
            dfd=dfd,
        )
        if vec_score > 0.85:
            # Strong alignment across all timeframes → bonus
            net *= 1.08
        elif vec_score < 0.3:
            # Conflicting timeframes → dampen
            net *= 0.92

        direction = "Buy" if net > 0 else "Sell" if net < 0 else "Neutral"

        reasons = []
        reasons.append(f"trend={trend_val:.2f}")
        reasons.append(f"mom={mom_val:.2f}")
        reasons.append(f"struct={struct_val:.2f}")
        reasons.append(f"mr={mr_val:.2f}")
        reasons.append(f"vol={vol_score:.2f}")
        reasons.append(f"ofi={flow_imb:+.2f}")
        reasons.append(f"vd={vol_delta:+.2f}")
        reasons.append(f"tm={tick_mom:+.2f}")
        reasons.append(f"flow={micro_edge:+.2f}")
        reasons.append(f"sh={float(indp.get('stop_hunt_strength', 0.0) or 0.0):+.2f}")
        reasons.append(f"obp={float(indp.get('ob_pretouch_bias', 0.0) or 0.0):+.2f}")
        reasons.append(f"ign={ignition_score:.2f}")
        d1_tag = "aligned" if d1_aligned else "conflict_or_flat"
        reasons.append(f"d1={d1_val:+.2f}:{d1_tag}")
        reasons.append(f"vec={vec_score:.2f}")
        reasons.append(f"net={net:+.1f}")

        return {
            "net_score": net,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "direction": direction,
            "reasons": reasons,
            "d1_value": d1_val,
            "vec_alignment": vec_score,
        }

    # ─── Sub-scores ──────────────────────────────────────────────────

    def _trend_score(
        self,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        regime: str,
        *,
        h1_slope: float = 0.0,
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
        if not _is_finite(h1_slope):
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

    # ─── D1 Confluence (Daily Trend) ─────────────────────────────────

    def _d1_confluence_score(
        self, dfd: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Daily trend confluence score [-1, +1].

        Uses D1 bars to determine the primary trend direction via:
          1. Linear regression slope of last 20 daily closes
          2. Price position relative to EMA200 (if enough data)
          3. Daily RSI direction

        Returns positive for bullish daily trend, negative for bearish.
        """
        if dfd is None or len(dfd) < 20:
            return {"value": 0.0, "reason": "insufficient_d1_data"}

        try:
            cols = {c.lower(): c for c in dfd.columns}
            c = dfd[cols.get("close", "Close")].values.astype(np.float64)

            if len(c) < 20:
                return {"value": 0.0, "reason": "insufficient_d1_closes"}

            score = 0.0

            # 1. Linear regression slope of last 20 daily closes
            window = c[-20:]
            x = np.arange(len(window), dtype=np.float64)
            # Normalized slope: rise per bar / mean price
            mean_p = float(np.mean(window))
            if mean_p > 0:
                slope = float(np.polyfit(x, window, 1)[0])
                norm_slope = slope / mean_p  # dimensionless
                # Scale: ±0.001/bar on daily = moderate trend
                if norm_slope > 0.0005:
                    score += 0.4
                elif norm_slope > 0.0002:
                    score += 0.2
                elif norm_slope < -0.0005:
                    score -= 0.4
                elif norm_slope < -0.0002:
                    score -= 0.2

            # 2. Price vs EMA200 (if enough bars)
            if len(c) >= 200:
                # Simple exponential moving average approximation
                ema200 = float(pd.Series(c).ewm(span=200, min_periods=200).mean().iloc[-1])
                if ema200 > 0:
                    price_vs_ema = (float(c[-1]) - ema200) / ema200
                    if price_vs_ema > 0.01:   # >1% above EMA200
                        score += 0.3
                    elif price_vs_ema > 0:
                        score += 0.1
                    elif price_vs_ema < -0.01:  # <1% below
                        score -= 0.3
                    elif price_vs_ema < 0:
                        score -= 0.1

            # 3. Daily momentum (5-day vs 20-day)
            if len(c) >= 21:
                ret5 = (float(c[-1]) - float(c[-6])) / float(c[-6]) if float(c[-6]) > 0 else 0.0
                ret20 = (float(c[-1]) - float(c[-21])) / float(c[-21]) if float(c[-21]) > 0 else 0.0
                # Accelerating trend: short-term > long-term
                if ret5 > 0 and ret20 > 0 and ret5 > ret20:
                    score += 0.15
                elif ret5 < 0 and ret20 < 0 and ret5 < ret20:
                    score -= 0.15

            return {"value": max(-1.0, min(1.0, score))}

        except Exception as exc:
            log.debug("_d1_confluence_score error: %s", exc)
            return {"value": 0.0, "reason": f"error:{exc}"}

    @staticmethod
    def _close_array(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if df is None or len(df) == 0:
            return None
        cols = {c.lower(): c for c in df.columns}
        c_col = cols.get("close")
        if not c_col:
            return None
        arr = df[c_col].values.astype(np.float64)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _linreg_slope_norm(close_arr: Optional[np.ndarray], window: int = 20) -> float:
        if close_arr is None or len(close_arr) < window:
            return 0.0
        y = close_arr[-window:].astype(np.float64)
        if not np.all(np.isfinite(y)):
            return 0.0
        ref = float(np.mean(y))
        if abs(ref) < 1e-12:
            return 0.0
        x = np.arange(window, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return float(slope / ref)

    def _vector_alignment(
        self,
        *,
        indp: Dict[str, Any],
        indc: Dict[str, Any],
        indl: Dict[str, Any],
        dfp: Optional[pd.DataFrame],
        dfl: Optional[pd.DataFrame],
        dfh: Optional[pd.DataFrame],
        dfd: Optional[pd.DataFrame],
    ) -> float:
        """
        Multi-timeframe Vector Analysis using cosine alignment.

        Builds a normalized slope vector from M1, M15, H1 and D1 closes,
        then measures its cosine similarity against perfect bullish and
        perfect bearish vectors. Final score is in [0, 1].
        """
        m1 = self._linreg_slope_norm(self._close_array(dfp), window=20)
        m15 = self._linreg_slope_norm(self._close_array(dfl), window=20)
        h1 = self._linreg_slope_norm(self._close_array(dfh), window=20)
        d1 = self._linreg_slope_norm(self._close_array(dfd), window=20)

        # Fallback to indicator slopes if any dataframe is missing.
        if m1 == 0.0:
            m1 = float(indp.get("linreg_slope", 0.0) or 0.0)
        if m15 == 0.0:
            m15 = float(indl.get("linreg_slope", 0.0) or 0.0)
        if h1 == 0.0:
            h1 = float(indc.get("linreg_slope", 0.0) or 0.0)

        raw_vec = np.array([m1, m15, h1, d1], dtype=np.float64)
        eps = 1e-6
        vec = np.where(raw_vec > eps, 1.0, np.where(raw_vec < -eps, -1.0, 0.0))
        if not np.all(np.isfinite(vec)):
            return 0.5
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return 0.5

        bull = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        bear = -bull
        bull_cos = float(np.dot(vec, bull) / (norm * np.linalg.norm(bull)))
        bear_cos = float(np.dot(vec, bear) / (norm * np.linalg.norm(bear)))

        # Keep pure alignment magnitude in [0, 1].
        alignment = max(bull_cos, bear_cos)
        if alignment < 0.0:
            return 0.0
        if alignment > 1.0:
            return 1.0
        return alignment

    def _momentum_ignition_score(self, dfp: Optional[pd.DataFrame]) -> float:
        """
        Detect early-move ignition before lagging candle-close confirmations.
        Score range: [0, 1].
        """
        if dfp is None or len(dfp) < 30:
            return 0.0

        try:
            cols = {c.lower(): c for c in dfp.columns}
            close_col = cols.get("close")
            high_col = cols.get("high")
            low_col = cols.get("low")
            vol_col = cols.get("tick_volume") or cols.get("volume") or cols.get("real_volume")
            if not close_col or not high_col or not low_col:
                return 0.0

            c = dfp[close_col].values.astype(np.float64)
            h = dfp[high_col].values.astype(np.float64)
            l = dfp[low_col].values.astype(np.float64)
            v = dfp[vol_col].values.astype(np.float64) if vol_col else None

            score = 0.0
            lookback = 20

            # 1) Volume spike: current volume above mean + 2 sigma
            if v is not None and len(v) >= lookback + 1:
                prev_v = v[-(lookback + 1):-1]
                mu = float(np.mean(prev_v))
                sigma = float(np.std(prev_v))
                cur_v = float(v[-1])
                if sigma > 0 and cur_v > mu + 2.0 * sigma:
                    score += 0.40
                elif sigma > 0 and cur_v > mu + 1.0 * sigma:
                    score += 0.20
                elif sigma <= 1e-12 and mu > 0 and cur_v > 1.5 * mu:
                    # Constant-volume baseline: treat large step-up as ignition.
                    score += 0.40

            # 2) Price breakout from recent range
            if len(c) >= lookback + 1 and len(h) >= lookback + 1 and len(l) >= lookback + 1:
                range_hi = float(np.max(h[-(lookback + 1):-1]))
                range_lo = float(np.min(l[-(lookback + 1):-1]))
                if float(c[-1]) > range_hi:
                    score += 0.40
                elif float(c[-1]) < range_lo:
                    score += 0.40

            # 3) Impulse acceleration on the current bar
            if len(c) >= 3:
                prev = float(c[-2])
                prev2 = float(c[-3])
                if abs(prev) > 1e-12 and abs(prev2) > 1e-12:
                    ret1 = (float(c[-1]) - prev) / abs(prev)
                    ret0 = (prev - prev2) / abs(prev2)
                    if abs(ret1) > max(2.0 * abs(ret0), 3e-4):
                        score += 0.20

            return clamp01(score)
        except Exception:
            return 0.0

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

        # Stop-hunt signed strength: +bull trap resolution / -bear trap resolution.
        sh = float(indp.get("stop_hunt_strength", 0.0) or 0.0)
        score += float(np.clip(sh, -1.0, 1.0)) * 0.35

        # Pre-touch order-block bias for early anticipatory entries.
        ob_bias = float(indp.get("ob_pretouch_bias", 0.0) or 0.0)
        ob_touch = float(indp.get("ob_touch_proximity", 0.0) or 0.0)
        score += float(np.clip(ob_bias, -1.0, 1.0)) * (0.20 + 0.10 * float(np.clip(ob_touch, 0.0, 1.0)))

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
            thr = float(getattr(self.cfg, "low_volume_sniper", 0.15) or 0.15)
            thr = max(0.05, min(1.0, thr))
            return ratio >= thr
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

    def _entry_quality_flags(
        self,
        *,
        signal: str,
        indp: Dict[str, Any],
        dfp: pd.DataFrame,
        net_score: float,
    ) -> Dict[str, Any]:
        """
        Entry-timing profile:
        - allow genuine impulse ignition a little earlier
        - block stretched end-of-trend chasing
        """
        out: Dict[str, Any] = {
            "early_trigger": False,
            "late_chase": False,
            "hard_veto": False,
            "extension_atr": 0.0,
        }
        if signal not in ("Buy", "Sell"):
            return out
        try:
            net_norm = max(-1.0, min(1.0, float(net_score) / 100.0))
            out["early_trigger"] = bool(self._early_momentum_trigger(indp, dfp, net_norm))

            atr = float(indp.get("atr", 0.0) or 0.0)
            close = float(indp.get("close", 0.0) or 0.0)
            if atr <= 0.0 or close <= 0.0:
                return out

            anchors = []
            for key in ("vwap", "ema_medium"):
                val = float(indp.get(key, 0.0) or 0.0)
                if val > 0.0 and _is_finite(val):
                    anchors.append(val)
            if not anchors:
                return out

            bb_pctb = float(indp.get("bb_pctb", 0.5) or 0.5)
            rsi = float(indp.get("rsi", 50.0) or 50.0)
            sh = float(indp.get("stop_hunt_strength", 0.0) or 0.0)
            ob_bias = float(indp.get("ob_pretouch_bias", 0.0) or 0.0)
            regime = str(indp.get("regime", "normal") or "normal").lower()

            if signal == "Buy":
                ref = max(anchors)
                extension = max(0.0, (close - ref) / atr)
                late_zone = bb_pctb > 0.88 or rsi > 67.0
                trap_aligned = sh > 0.25 or ob_bias > 0.20
            else:
                ref = min(anchors)
                extension = max(0.0, (ref - close) / atr)
                late_zone = bb_pctb < 0.12 or rsi < 33.0
                trap_aligned = sh < -0.25 or ob_bias < -0.20

            out["extension_atr"] = float(extension)

            ext_limit = 1.45 if "BTC" in str(self.sp.base).upper() else 1.20
            if regime in ("volatile", "explosive"):
                ext_limit += 0.20
            if bool(out["early_trigger"]):
                ext_limit += 0.20

            late_chase = bool(late_zone and extension > ext_limit and (not trap_aligned))
            out["late_chase"] = late_chase
            out["hard_veto"] = bool(late_chase and extension > (ext_limit + 0.55))
            return out
        except Exception:
            return out

    def _get_base_weights(self) -> Dict[str, float]:
        if self._weights_cache is not None:
            return self._weights_cache
        raw = dict(self.cfg.weights)
        for k, v in {
            "trend": 24.0,
            "momentum": 15.0,
            "volatility": 10.0,
            "structure": 12.0,
            "flow": 15.0,
            "tick_momentum": 12.0,
            "volume_delta": 8.0,
            "mean_reversion": 4.0,
        }.items():
            if k not in raw:
                raw[k] = float(v)
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
            pct_changes = np.abs(np.diff(c[-10:])) / c[-10:-1]
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
            side_n = _side_norm(side)
            if side_n == "Buy":
                return recent_change > -0.02
            if side_n == "Sell":
                return recent_change < 0.02
            return False
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
        dfd: Optional[pd.DataFrame] = None,
        ind_h1: Optional[Dict[str, Any]] = None,
        ind_h4: Optional[Dict[str, Any]] = None,
        df_m15: Optional[pd.DataFrame] = None,
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

        # Hard MTF institutional flow gate (H1/H4 veto).
        if signal in ("Buy", "Sell"):
            signal, conf = self._apply_hard_mtf_confluence(
                signal=signal,
                conf=conf,
                ind_h1=(ind_h1 or {}),
                ind_h4=(ind_h4 or {}),
                reasons=reasons,
            )
            if signal == "Neutral":
                return signal, conf

        # XAU macro gate from DXY + US10Y pressure.
        if signal in ("Buy", "Sell"):
            signal, conf = self._apply_macro_gate_xau(
                signal=signal,
                conf=conf,
                reasons=reasons,
                df_m15=df_m15,
            )
            if signal == "Neutral":
                return signal, conf

        # D1 conflict penalty — REMOVED (2026-04-02 forensic audit).
        # D1 confluence is already applied in _ensemble_score (line ~861) where
        # it adjusts the net score. Applying it a second time here as a 15%
        # confidence multiplier caused double-penalization from the same data.

        return signal, conf

    def _institutional_flow_score(self, ind: Dict[str, Any]) -> float:
        """Directional flow proxy in [-1, +1] from trend, slope, structure, and trap context."""
        if not ind:
            return 0.0
        score = 0.0
        trend = str(ind.get("trend", "") or "").lower()
        if trend == "bull":
            score += 0.30
        elif trend == "bear":
            score -= 0.30

        slope = float(ind.get("linreg_slope", 0.0) or 0.0)
        score += float(np.tanh(slope * 10.0)) * 0.28

        mh = float(ind.get("macd_hist", 0.0) or 0.0)
        score += float(np.tanh(mh * 12.0)) * 0.18

        sweep = str(ind.get("sweep", "") or "")
        if "bull" in sweep:
            score += 0.10
        elif "bear" in sweep:
            score -= 0.10

        ob = str(ind.get("order_block", "") or "")
        if "bull" in ob:
            score += 0.08
        elif "bear" in ob:
            score -= 0.08

        sh = float(ind.get("stop_hunt_strength", 0.0) or 0.0)
        score += float(np.clip(sh, -1.0, 1.0)) * 0.16

        ob_bias = float(ind.get("ob_pretouch_bias", 0.0) or 0.0)
        score += float(np.clip(ob_bias, -1.0, 1.0)) * 0.10

        return float(np.clip(score, -1.0, 1.0))

    def _apply_hard_mtf_confluence(
        self,
        *,
        signal: str,
        conf: int,
        ind_h1: Dict[str, Any],
        ind_h4: Dict[str, Any],
        reasons: Optional[List[str]],
    ) -> Tuple[str, int]:
        if not bool(getattr(self.cfg, "mtf_hard_gate_enabled", True)):
            return signal, conf
        h1 = self._institutional_flow_score(ind_h1)
        h4 = self._institutional_flow_score(ind_h4)
        h1_gate = float(getattr(self.cfg, "mtf_h1_flow_gate", 0.18) or 0.18)
        h4_gate = float(getattr(self.cfg, "mtf_h4_flow_gate", 0.22) or 0.22)
        h4_veto = bool(getattr(self.cfg, "mtf_h4_veto_enabled", True))
        is_buy = signal == "Buy"

        h1_conflict = (is_buy and h1 < -h1_gate) or ((not is_buy) and h1 > h1_gate)
        h4_conflict = (is_buy and h4 < -h4_gate) or ((not is_buy) and h4 > h4_gate)

        if h4_conflict and h4_veto:
            if reasons is not None:
                reasons.append(f"mtf_h4_veto:h1={h1:+.2f}|h4={h4:+.2f}")
            return "Neutral", 0

        if h1_conflict and h4_conflict:
            if reasons is not None:
                reasons.append(f"mtf_h1_h4_conflict:h1={h1:+.2f}|h4={h4:+.2f}")
            return "Neutral", 0

        if h1_conflict or h4_conflict:
            conf = int(conf * 0.70)
            if reasons is not None:
                reasons.append(f"mtf_partial_conflict:h1={h1:+.2f}|h4={h4:+.2f}")
        else:
            if reasons is not None:
                reasons.append(f"mtf_aligned:h1={h1:+.2f}|h4={h4:+.2f}")
        return signal, conf

    @staticmethod
    def _tf_enum(tf: str) -> Optional[int]:
        if mt5 is None:
            return None
        t = str(tf or "").upper().strip()
        if t == "M1":
            return int(getattr(mt5, "TIMEFRAME_M1", 0) or 0)
        if t == "M5":
            return int(getattr(mt5, "TIMEFRAME_M5", 0) or 0)
        if t == "M15":
            return int(getattr(mt5, "TIMEFRAME_M15", 0) or 0)
        if t == "H1":
            return int(getattr(mt5, "TIMEFRAME_H1", 0) or 0)
        if t == "H4":
            return int(getattr(mt5, "TIMEFRAME_H4", 0) or 0)
        return None

    def _resolve_macro_symbol(self, key: str, candidates: List[str]) -> Optional[str]:
        k = str(key).upper().strip()
        if k in self._macro_symbol_cache:
            return self._macro_symbol_cache[k]
        resolved: Optional[str] = None
        try:
            with _mt5_lock():
                for sym in candidates:
                    info = mt5.symbol_info(sym) if mt5 is not None else None
                    if info is None:
                        continue
                    try:
                        if not bool(getattr(info, "visible", True)):
                            mt5.symbol_select(sym, True)
                    except Exception:
                        pass
                    resolved = sym
                    break
                if resolved is None and mt5 is not None:
                    symbols = mt5.symbols_get()
                    if symbols:
                        names = [str(getattr(s, "name", "")) for s in symbols]
                        for needle in candidates:
                            nu = needle.upper()
                            hit = next((n for n in names if nu in n.upper()), None)
                            if hit:
                                resolved = hit
                                break
        except Exception:
            resolved = None
        self._macro_symbol_cache[k] = resolved
        return resolved

    def _fetch_close_array(self, symbol: str, tf: str, bars: int) -> Optional[np.ndarray]:
        tf_id = self._tf_enum(tf)
        if tf_id is None or tf_id <= 0 or not symbol:
            return None
        try:
            with _mt5_lock():
                rates = mt5.copy_rates_from_pos(symbol, tf_id, 0, int(max(32, bars)))
            if rates is None or len(rates) < 16:
                return None
            raw = pd.DataFrame(rates)
            if "close" not in raw.columns:
                return None
            arr = pd.to_numeric(raw["close"], errors="coerce").to_numpy(dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size < 16:
                return None
            return arr
        except Exception:
            return None

    def _macro_context_xau(self, df_m15: Optional[pd.DataFrame]) -> Dict[str, float]:
        ttl = float(getattr(self.cfg, "macro_gate_ttl_sec", 15.0) or 15.0)
        now = time.time()
        if self._macro_ctx_cache and (now - self._macro_ctx_cache_ts) <= max(1.0, ttl):
            return dict(self._macro_ctx_cache)

        ctx = {
            "bias": 0.0,
            "dxy_z": 0.0,
            "us10y_z": 0.0,
            "corr_dxy": 0.0,
            "corr_us10y": 0.0,
            "ready": 0.0,
        }
        try:
            if mt5 is None:
                self._macro_ctx_cache = ctx
                self._macro_ctx_cache_ts = now
                return dict(ctx)

            dxy_sym = self._resolve_macro_symbol(
                "DXY", ["DXY", "DXYm", "USDX", "USDXm", "DX1!", "DOLLAR", "USDINDEX"]
            )
            us10y_sym = self._resolve_macro_symbol(
                "US10Y", ["US10Y", "US10Ym", "UST10Y", "US10YT", "TNX", "US10Y.cash"]
            )
            dxy_c = self._fetch_close_array(dxy_sym or "", "M15", 320) if dxy_sym else None
            us_c = self._fetch_close_array(us10y_sym or "", "M15", 320) if us10y_sym else None
            if dxy_c is None and us_c is None:
                self._macro_ctx_cache = ctx
                self._macro_ctx_cache_ts = now
                return dict(ctx)

            if df_m15 is not None and len(df_m15) >= 32:
                ccol = "close" if "close" in df_m15.columns else ("Close" if "Close" in df_m15.columns else None)
                if ccol is not None:
                    x = pd.to_numeric(df_m15[ccol], errors="coerce").to_numpy(dtype=np.float64)
                    x = x[np.isfinite(x)]
                else:
                    x = np.array([], dtype=np.float64)
            else:
                x = np.array([], dtype=np.float64)

            def _z_last(arr: Optional[np.ndarray]) -> float:
                if arr is None or arr.size < 70:
                    return 0.0
                r = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
                if r.size < 65:
                    return 0.0
                hist = r[-65:-1]
                mu = float(np.mean(hist))
                sd = float(np.std(hist))
                if sd <= 1e-12:
                    return 0.0
                return float((r[-1] - mu) / sd)

            dxy_z = _z_last(dxy_c)
            us_z = _z_last(us_c)
            w_dxy = float(getattr(self.cfg, "macro_dxy_weight", 0.60) or 0.60)
            w_us = float(getattr(self.cfg, "macro_us10y_weight", 0.40) or 0.40)
            raw = -(w_dxy * dxy_z + w_us * us_z)
            bias = float(np.tanh(raw))

            corr_dxy = 0.0
            corr_us = 0.0
            if x.size >= 80:
                xr = np.diff(x) / np.maximum(x[:-1], 1e-12)
                if dxy_c is not None and dxy_c.size >= 80:
                    dr = np.diff(dxy_c) / np.maximum(dxy_c[:-1], 1e-12)
                    m = min(len(xr), len(dr), 120)
                    if m >= 30:
                        corr_dxy = float(np.corrcoef(xr[-m:], dr[-m:])[0, 1])
                        if not np.isfinite(corr_dxy):
                            corr_dxy = 0.0
                if us_c is not None and us_c.size >= 80:
                    ur = np.diff(us_c) / np.maximum(us_c[:-1], 1e-12)
                    m = min(len(xr), len(ur), 120)
                    if m >= 30:
                        corr_us = float(np.corrcoef(xr[-m:], ur[-m:])[0, 1])
                        if not np.isfinite(corr_us):
                            corr_us = 0.0

            ctx = {
                "bias": float(np.clip(bias, -1.0, 1.0)),
                "dxy_z": float(np.clip(dxy_z, -5.0, 5.0)),
                "us10y_z": float(np.clip(us_z, -5.0, 5.0)),
                "corr_dxy": float(np.clip(corr_dxy, -1.0, 1.0)),
                "corr_us10y": float(np.clip(corr_us, -1.0, 1.0)),
                "ready": 1.0,
            }
        except Exception as exc:
            log.debug("_macro_context_xau failed: %s", exc)
            return dict(ctx)

        self._macro_ctx_cache = dict(ctx)
        self._macro_ctx_cache_ts = now
        return dict(ctx)

    def _apply_macro_gate_xau(
        self,
        *,
        signal: str,
        conf: int,
        reasons: Optional[List[str]],
        df_m15: Optional[pd.DataFrame],
    ) -> Tuple[str, int]:
        if "XAU" not in str(self.sp.base).upper():
            return signal, conf
        if not bool(getattr(self.cfg, "macro_gate_enabled", True)):
            return signal, conf
        ctx = self._macro_context_xau(df_m15)
        if float(ctx.get("ready", 0.0)) < 0.5:
            return signal, conf
        bias = float(ctx.get("bias", 0.0) or 0.0)
        block = float(getattr(self.cfg, "macro_bias_block", 0.28) or 0.28)
        pen = float(getattr(self.cfg, "macro_bias_penalty", 0.12) or 0.12)

        conflict = (signal == "Buy" and bias < 0.0) or (signal == "Sell" and bias > 0.0)
        hard_conflict = (signal == "Buy" and bias <= -abs(block)) or (signal == "Sell" and bias >= abs(block))

        if hard_conflict:
            if reasons is not None:
                reasons.append(
                    "macro_veto:"
                    f"bias={bias:+.2f}|dxy_z={float(ctx.get('dxy_z', 0.0)):+.2f}|"
                    f"us10y_z={float(ctx.get('us10y_z', 0.0)):+.2f}"
                )
            return "Neutral", 0
        if conflict:
            conf = int(conf * max(0.0, 1.0 - abs(pen)))
            if reasons is not None:
                reasons.append(
                    "macro_conflict:"
                    f"bias={bias:+.2f}|dxy_z={float(ctx.get('dxy_z', 0.0)):+.2f}|"
                    f"us10y_z={float(ctx.get('us10y_z', 0.0)):+.2f}"
                )
        else:
            if reasons is not None:
                reasons.append(f"macro_aligned:bias={bias:+.2f}")
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
        structure_frames: Tuple[Optional[pd.DataFrame], ...] = (),
        market_regime: Optional[str] = None,
    ) -> SignalResult:
        """Build final SignalResult, optionally with execution plan."""
        sig_id = self._signal_id(sym, self.sp.tf_primary, bar_key, signal)
        latency = (time.time() - t0) * 1000

        result = SignalResult(
            signal=signal,
            symbol=sym,
            confidence=conf,
            spread_pct=spread_pct,
            regime=market_regime or adapt.get("regime", "normal"),
            signal_id=sig_id,
            bar_key=bar_key,
            reasons=reasons,
            latency_ms=latency,
            timeframe=self.sp.tf_primary,
            data_state=DATA_VALID,
            feature_state=FEATURES_VALID,
        )

        if execute and signal != "Neutral" and conf >= self.cfg.min_confidence:
            plan = self._rm.plan_order(
                side=signal,
                confidence=conf / 100.0,
                ind=indp,
                adapt=adapt,
                ticks=tick_stats,
                df=dfp,
                structure_frames=structure_frames,
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
                with _mt5_lock():
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
