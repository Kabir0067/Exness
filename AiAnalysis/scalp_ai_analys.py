"""
sclaping_ai.py — Scalping Analysis Engine (M15 / M5 / M1)
Institutional-grade SMC-aware AI analysis with ranked provider chain.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from log_config import build_logger

from .analysis_common import (
    SCALPING_PROVIDERS,
    action_short_for_signal,
    clamp,
    run_provider_chain,
    run_provider_diagnostics,
    safe_float,
)
from .sym_news import attach_news_context

log = build_logger("ai.scalping.analysis", "sclaping_ai.log", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

AI_TIMEOUT_SEC = 16
AI_MAX_RETRIES = 1
AI_MIN_CONF = 0.70
AI_BUDGET_SEC = 22.0
AI_CACHE_TTL_SEC = 120.0
AI_CONFIRMATION_MODELS = 2
AI_CONFIRMATION_MIN_CONF = 0.55
AI_CONFIRMATION_SKIP_CONF = 0.85
MIN_RISK_REWARD = 1.25

_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _pick_atr(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("M5", "M15", "M1"):
        value = safe_float((market_data.get(tf) or {}).get("atr_14"))
        if value and value > 0:
            return value
    return None


def _pick_close(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("M1", "M5", "M15"):
        value = safe_float((market_data.get(tf) or {}).get("last_close"))
        if value and value > 0:
            return value
    return None


def _pick_m1_close(market_data: Dict[str, Any]) -> Optional[float]:
    value = safe_float((market_data.get("M1") or {}).get("last_close"))
    return value if value and value > 0 else None


def _news_cache_fingerprint(market_data: Dict[str, Any]) -> Tuple[Any, ...]:
    news = market_data.get("news_context") or {}
    return (
        str(news.get("bias") or "neutral").lower(),
        round(float(safe_float(news.get("avg_sentiment")) or 0.0), 3),
        int(news.get("high_impact_count") or 0),
        str(news.get("status") or "unknown").lower(),
        str(news.get("ai_summary") or "")[:160],
    )


def _build_cache_key(
    asset: str, ts_bar: int, market_data: Dict[str, Any]
) -> Tuple[Any, ...]:
    meta = market_data.get("meta") or {}
    return (
        str(asset),
        int(ts_bar),
        round(float(_pick_close(market_data) or 0.0), 4),
        int(meta.get("age_sec") or 0),
        str(market_data.get("summary_text") or "")[:160],
        _news_cache_fingerprint(market_data),
    )


def _spread_penalty(meta: Dict[str, Any], atr: Optional[float]) -> Tuple[float, str]:
    spread_pts = safe_float(meta.get("spread_points"))
    point = safe_float(meta.get("point")) or 0.0
    if not spread_pts or spread_pts <= 0 or point <= 0 or not atr or atr <= 0:
        return 0.0, "spread_penalty=0"
    spread_price = float(spread_pts) * float(point)
    if spread_price > 0.25 * float(atr):
        return (
            0.20,
            f"spread_penalty=0.20 spread_price={spread_price:.5f} atr={atr:.5f}",
        )
    return 0.0, f"spread_penalty=0 spread_price={spread_price:.5f} atr={atr:.5f}"


def _levels_clamp(
    symbol_meta: Dict[str, Any],
    signal: str,
    entry: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    atr: Optional[float],
) -> Tuple[float, float, float]:
    digits = int(symbol_meta.get("digits") or 2)
    point = float(symbol_meta.get("point") or (10 ** (-digits) if digits > 0 else 0.01))
    stops_level_pts = int(symbol_meta.get("stops_level_points") or 0)
    min_dist = float(stops_level_pts) * point
    if atr and atr > 0:
        min_dist = max(min_dist, atr * 0.6)

    def rnd(v: float) -> float:
        return float(round(v, digits))

    base_sl = atr * 0.9 if atr and atr > 0 else max(min_dist * 1.8, point * 50.0)
    base_tp = atr * 1.4 if atr and atr > 0 else max(min_dist * 2.5, point * 70.0)

    if signal == "BUY":
        sl_ok = stop_loss is not None and stop_loss < (entry - min_dist)
        tp_ok = take_profit is not None and take_profit > (entry + min_dist)
        stop_loss = stop_loss if sl_ok else entry - max(base_sl, min_dist)
        take_profit = take_profit if tp_ok else entry + max(base_tp, min_dist)
    elif signal == "SELL":
        sl_ok = stop_loss is not None and stop_loss > (entry + min_dist)
        tp_ok = take_profit is not None and take_profit < (entry - min_dist)
        stop_loss = stop_loss if sl_ok else entry + max(base_sl, min_dist)
        take_profit = take_profit if tp_ok else entry - max(base_tp, min_dist)
    else:
        stop_loss = entry
        take_profit = entry

    return rnd(entry), rnd(float(stop_loss)), rnd(float(take_profit))


def _risk_reward_ratio(
    signal: str,
    entry: Optional[float],
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> float:
    if signal not in ("BUY", "SELL"):
        return 0.0
    if entry is None or stop_loss is None or take_profit is None:
        return 0.0
    risk = abs(float(entry) - float(stop_loss))
    reward = abs(float(take_profit) - float(entry))
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def _confirm_direction(
    *,
    selected_rank: int,
    selected_signal: str,
    prompt: str,
    close: Optional[float],
) -> Tuple[bool, List[Dict[str, Any]]]:
    if selected_signal not in ("BUY", "SELL"):
        return True, []

    backups = [
        row
        for row in SCALPING_PROVIDERS
        if int(row.get("rank") or 0) > int(selected_rank or 0)
    ][:AI_CONFIRMATION_MODELS]
    if not backups:
        return True, []

    results = run_provider_diagnostics(
        providers=backups,
        prompt=prompt,
        schema_name="trade_signal_scalping",
        close=close,
        timeout_per_call_sec=min(AI_TIMEOUT_SEC, 10),
        max_retries=0,
    )
    successful = [row for row in results if row.get("ok")]
    for row in successful:
        if str(row.get("signal") or "").upper() != selected_signal:
            continue
        if float(row.get("confidence") or 0.0) >= AI_CONFIRMATION_MIN_CONF:
            return True, results
    for row in successful:
        other_signal = str(row.get("signal") or "").upper()
        if other_signal not in ("BUY", "SELL"):
            continue
        if other_signal == selected_signal:
            continue
        if float(row.get("confidence") or 0.0) >= AI_CONFIRMATION_MIN_CONF:
            return False, results
    return True, results


def _has_live_high_impact_news_conflict(news: Dict[str, Any], signal: str) -> bool:
    direction = str(signal or "").upper()
    if direction not in ("BUY", "SELL"):
        return False
    bias = str(news.get("bias") or "neutral").lower()
    high_impact = int(news.get("high_impact_count") or 0)
    mode = str(news.get("confidence_mode") or news.get("status") or "").lower()
    if mode != "live" or high_impact <= 0:
        return False
    return (direction == "BUY" and bias == "bearish") or (
        direction == "SELL" and bias == "bullish"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SMC context extractor
# ─────────────────────────────────────────────────────────────────────────────


def _format_smc_context(market_data: Dict[str, Any]) -> str:
    """Extract SMC data from payload and format for AI prompt."""
    lines: List[str] = []
    for tf in ("M15", "M5", "M1"):
        tf_data = market_data.get(tf) or {}
        struct = tf_data.get("structure") or {}
        fvgs = tf_data.get("fvgs") or []
        obs = tf_data.get("order_blocks") or []

        if not struct and not fvgs and not obs:
            continue

        lines.append(f"\n[SMC/{tf}]")

        if struct:
            trend = str(struct.get("trend") or "neutral").upper()
            bos = struct.get("last_bos")
            choch = struct.get("choch")
            zone = str(struct.get("zone") or "equilibrium").upper()
            sh = struct.get("swing_high")
            sl_level = struct.get("swing_low")
            parts = [f"TREND={trend}", f"ZONE={zone}"]
            if bos:
                parts.append(f"BOS={bos}")
            if choch:
                parts.append(f"CHoCH={choch}")
            if sh:
                parts.append(f"SwH={sh:.2f}")
            if sl_level:
                parts.append(f"SwL={sl_level:.2f}")
            lines.append("  STRUCTURE: " + " | ".join(parts))

        # Most relevant FVG
        if fvgs:
            close = safe_float(tf_data.get("last_close")) or 0.0
            # Sort by distance to close
            sorted_fvgs = sorted(
                fvgs, key=lambda f: abs(float(f.get("mid", 0) or 0) - close)
            )
            for fvg in sorted_fvgs[:2]:
                direction = str(fvg.get("direction") or "").upper()
                top = fvg.get("top", 0)
                bot = fvg.get("bottom", 0)
                mid = fvg.get("mid", 0)
                dist_pct = (
                    round((float(mid or 0) - close) / max(close, 1e-9) * 100, 2)
                    if close
                    else 0
                )
                lines.append(
                    f"  FVG: {direction} top={top:.2f} bot={bot:.2f} dist={dist_pct:+.2f}%"
                )

        # Closest order block
        if obs:
            close = safe_float(tf_data.get("last_close")) or 0.0
            sorted_obs = sorted(
                obs, key=lambda o: abs(float(o.get("mid", 0) or 0) - close)
            )
            for ob in sorted_obs[:2]:
                direction = str(ob.get("direction") or "").upper()
                top = ob.get("top", 0)
                bot = ob.get("bottom", 0)
                strength = ob.get("strength", 0)
                dist_pct = (
                    round(
                        (float(ob.get("mid", 0) or 0) - close) / max(close, 1e-9) * 100,
                        2,
                    )
                    if close
                    else 0
                )
                lines.append(
                    f"  OB: {direction} top={top:.2f} bot={bot:.2f} str={strength:.2f} dist={dist_pct:+.2f}%"
                )

    return "\n".join(lines) if lines else "SMC: No significant zones detected."


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────


def _advanced_bonus(
    direction: str, market_data: Dict[str, Any]
) -> Tuple[float, List[str]]:
    m1 = market_data.get("M1") or {}
    m5 = market_data.get("M5") or {}
    score = 0.0
    notes: List[str] = []

    m5_conf = safe_float((m5.get("confluence") or {}).get("score")) or 50.0
    if direction == "BUY":
        if m5_conf >= 65:
            score += 0.08
            notes.append("M5 confluence bullish")
        elif m5_conf <= 40:
            score -= 0.08
    else:
        if m5_conf <= 35:
            score += 0.08
            notes.append("M5 confluence bearish")
        elif m5_conf >= 60:
            score -= 0.08

    m5_hist = safe_float((m5.get("macd") or {}).get("histogram")) or 0.0
    if direction == "BUY" and m5_hist > 0:
        score += 0.08
        notes.append("M5 MACD > 0")
    elif direction == "SELL" and m5_hist < 0:
        score += 0.08
        notes.append("M5 MACD < 0")
    elif m5_hist != 0:
        score -= 0.06

    stoch = m1.get("stoch_rsi") or {}
    stoch_cross = str(stoch.get("cross") or "none")
    stoch_zone = str(stoch.get("zone") or "neutral")
    if direction == "BUY":
        if stoch_cross == "bullish_cross" or stoch_zone == "oversold":
            score += 0.06
            notes.append("M1 StochRSI recovery")
        elif stoch_cross == "bearish_cross" or stoch_zone == "overbought":
            score -= 0.06
    else:
        if stoch_cross == "bearish_cross" or stoch_zone == "overbought":
            score += 0.06
            notes.append("M1 StochRSI fade")
        elif stoch_cross == "bullish_cross" or stoch_zone == "oversold":
            score -= 0.06

    volume_ratio = safe_float((m1.get("volume") or {}).get("ratio")) or 1.0
    pattern = str((m1.get("price_action") or {}).get("pattern") or "none")
    if volume_ratio >= 1.2:
        if direction == "BUY" and pattern in {
            "bullish_engulfing",
            "hammer",
            "breakout_close",
        }:
            score += 0.06
            notes.append("M1 volume-backed bullish pattern")
        elif direction == "SELL" and pattern in {
            "bearish_engulfing",
            "shooting_star",
            "breakdown_close",
        }:
            score += 0.06
            notes.append("M1 volume-backed bearish pattern")

    bb_pct = safe_float((m1.get("bb") or {}).get("pct_b"))
    if bb_pct is not None:
        if direction == "BUY" and bb_pct >= 0.95:
            score -= 0.04
        elif direction == "SELL" and bb_pct <= 0.05:
            score -= 0.04

    return score, notes


def _heuristic_reason(
    direction: str, market_data: Dict[str, Any], notes: List[str]
) -> str:
    m15 = market_data.get("M15") or {}
    m5 = market_data.get("M5") or {}
    m1 = market_data.get("M1") or {}
    struct = m5.get("structure") or {}
    news = market_data.get("news_context") or {}
    trend_score = int((m15.get("confluence") or {}).get("score") or 50)
    confirm_score = int((m5.get("confluence") or {}).get("score") or 50)
    pattern = str((m1.get("price_action") or {}).get("pattern") or "none")
    volume_ratio = safe_float((m1.get("volume") or {}).get("ratio")) or 1.0
    macd_cross = str((m5.get("macd") or {}).get("cross") or "none")
    bos = str(struct.get("last_bos") or "нест")
    news_bias = str(news.get("bias") or "neutral")
    extra = ", ".join(notes[:2]) if notes else "microstructure миёна"

    if direction == "BUY":
        return (
            f"M15 боло мемонад (score={trend_score}), M5 тасдиқ медиҳад (score={confirm_score}), "
            f"MACD={macd_cross}, pattern={pattern}, ҳаҷм={volume_ratio:.2f}x, BOS={bos}, news={news_bias}; {extra}."
        )
    if direction == "SELL":
        return (
            f"M15 поён мемонад (score={trend_score}), M5 тасдиқ медиҳад (score={confirm_score}), "
            f"MACD={macd_cross}, pattern={pattern}, ҳаҷм={volume_ratio:.2f}x, BOS={bos}, news={news_bias}; {extra}."
        )
    return (
        f"Тасдиқи кофӣ нест: M15/M5 score={trend_score}/{confirm_score}, "
        f"MACD={macd_cross}, pattern={pattern}, news={news_bias}; вуруд ба таъхир гузошта шуд."
    )


def _build_prompt(asset: str, market_data: Dict[str, Any]) -> str:
    summary = str(market_data.get("summary_text") or "").strip()
    meta = market_data.get("meta") or {}
    news = market_data.get("news_context") or {}
    state = market_data.get("market_state") or {}
    atr = _pick_atr(market_data)
    close = _pick_close(market_data) or 0.0
    penalty, penalty_debug = _spread_penalty(meta, atr)
    smc_context = _format_smc_context(market_data)

    asset_label = (
        "Bitcoin (BTC/USD)" if "btc" in str(asset).lower() else "Gold (XAU/USD)"
    )

    return f"""You are a SENIOR INSTITUTIONAL SCALPING ANALYST specializing in {asset_label}.
You have mastered Smart Money Concepts (SMC): Order Blocks, Fair Value Gaps, Market Structure.
Return EXACTLY one JSON object. Use ONLY the data below. No invented data.

═══ ASSET & CONTEXT ════════════════════════════════════════
Asset         : {asset_label}
Style         : SCALPING (M15/M5/M1 multi-timeframe)
Venue         : Exness MT5 | Symbol: {market_data.get('symbol', asset)}
Current Price : {close:.2f}
Market State  : bias={state.get('directional_bias', 'balanced')} | regime={state.get('regime', 'balanced')} | execution={state.get('execution_state', 'normal')}
News Source   : status={news.get('status', 'unknown')} | confidence_mode={news.get('confidence_mode', 'unknown')}
News Policy   : Live MarketAux high-impact conflict can BLOCK. AI fallback news is SOFT context only and may adjust confidence, not force HOLD by itself.

═══ MARKET SNAPSHOT ════════════════════════════════════════
{summary}

═══ SMC INSTITUTIONAL CONTEXT ══════════════════════════════
{smc_context}

═══ SCORING FRAMEWORK ══════════════════════════════════════
Start at 0.00

TREND ALIGNMENT (+0.35):
  BUY  → M15: close > EMA20 > EMA50 > EMA200
  SELL → M15: close < EMA20 < EMA50 < EMA200

M5 CONFIRMATION (+0.20):
  BUY  → M5 close > M5 EMA20
  SELL → M5 close < M5 EMA20

VWAP FILTER (+0.15):
  BUY  → M5 close above VWAP
  SELL → M5 close below VWAP

MOMENTUM (+0.10 each):
  M1 RSI not adversarial (BUY: RSI≥45 | SELL: RSI≤55)
  M5 RSI not extended   (BUY: RSI≤55 | SELL: RSI≥45)

ADVANCED FLOW:
  +0.08 if M5 MACD histogram supports direction
  +0.06 if M1 StochRSI cross/zone supports direction
  +0.06 if M1 volume ratio >= 1.2x confirms the entry candle
  +0.05 if price action pattern supports direction
  +0.08 if M5 confluence score >= 65 in direction

SMC BONUS/PENALTY:
  +0.10 if price is at/above a BULLISH Order Block (BUY) or BEARISH OB (SELL)
  +0.08 if a relevant FVG supports the direction (price entering discount FVG for BUY, premium FVG for SELL)
  +0.05 if Market Structure BOS/CHoCH confirms direction
  -0.10 if price in PREMIUM zone for BUY (overextended) or DISCOUNT for SELL

NEWS OVERLAY:
  -0.20 if data stale >90s
  -{penalty:.2f} spread cost penalty
  -0.15 if high-impact news conflicts
  +0.05 if news confirms + no high-impact conflict
  Clamp confidence [0.00, 1.00]
  If confidence < {AI_MIN_CONF:.2f} → signal = HOLD
  If trend + confirmation + VWAP + SMC align strongly, prefer BUY/SELL over a passive HOLD.

═══ TRADE LEVELS ════════════════════════════════════════════
entry       = M1 last_close
stop_loss   = 0.9 × ATR14(M5), adjusted to nearest invalidation / Order Block
take_profit = at least 1.4 × ATR14(M5), adjusted to nearest FVG mid or swing level
Minimum R:R = {MIN_RISK_REWARD:.2f}; if lower → HOLD

═══ OUTPUT RULES ════════════════════════════════════════════
- reason: Tajik Cyrillic, senior concise style
- Mention: M15 trend, M5 confirmation, VWAP, advanced indicators, SMC zones, news impact
- action_short: EXACTLY one of: "Харид", "Фурӯш", "Интизор"
- No markdown. No text outside JSON.

═══ DEBUG ═══════════════════════════════════════════════════
spread_debug  : {penalty_debug}
news_bias     : {news.get('bias', 'unknown')}
news_sentiment: {news.get('avg_sentiment', 0.0)}
news_impact   : {news.get('high_impact_count', 0)}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristic fallback (when all AI providers fail)
# ─────────────────────────────────────────────────────────────────────────────


def _heuristic_signal(market_data: Dict[str, Any], asset: str) -> Dict[str, Any]:
    m1 = market_data.get("M1") or {}
    m5 = market_data.get("M5") or {}
    m15 = market_data.get("M15") or {}
    meta = market_data.get("meta") or {}
    news = market_data.get("news_context") or {}

    m1_close = safe_float(m1.get("last_close")) or 0.0
    m5_close = safe_float(m5.get("last_close")) or 0.0
    m15_close = safe_float(m15.get("last_close")) or 0.0

    if not any((m1_close, m5_close, m15_close)):
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Маълумоти кофии бозор дастрас нест.",
            "action_short": action_short_for_signal("HOLD"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_scalp_v3",
        }

    rsi1 = safe_float(m1.get("rsi_14")) or 50.0
    rsi5 = safe_float(m5.get("rsi_14")) or 50.0
    ema20_15 = safe_float(m15.get("ema_20")) or m15_close
    ema50_15 = safe_float(m15.get("ema_50")) or m15_close
    ema200_15 = safe_float(m15.get("ema_200")) or m15_close
    ema20_5 = safe_float(m5.get("ema_20")) or m5_close
    vwap5 = safe_float(m5.get("vwap")) or m5_close
    age_sec = int(meta.get("age_sec") or 0)
    atr = _pick_atr(market_data)
    spread_penalty_val, _ = _spread_penalty(meta, atr)
    news_bias = str(news.get("bias") or "neutral").lower()
    high_impact = int(news.get("high_impact_count") or 0)

    # SMC bonuses
    def smc_bonus(direction: str) -> float:
        bonus = 0.0
        m5_data = market_data.get("M5") or {}
        struct = m5_data.get("structure") or {}
        zone = str(struct.get("zone") or "equilibrium").lower()
        obs = m5_data.get("order_blocks") or []
        close5 = m5_close

        # Zone check
        if direction == "BUY" and zone == "premium":
            bonus -= 0.10
        elif direction == "SELL" and zone == "discount":
            bonus -= 0.10

        # Order Block check
        if close5 > 0:
            for ob in obs:
                ob_dir = str(ob.get("direction") or "").lower()
                ob_top = float(ob.get("top") or 0)
                ob_bot = float(ob.get("bottom") or 0)
                if (
                    direction == "BUY"
                    and ob_dir == "bullish"
                    and ob_bot <= close5 <= ob_top
                ):
                    bonus += 0.10
                    break
                if (
                    direction == "SELL"
                    and ob_dir == "bearish"
                    and ob_bot <= close5 <= ob_top
                ):
                    bonus += 0.10
                    break

        # BOS
        bos = struct.get("last_bos") or ""
        if direction == "BUY" and bos == "bullish_bos":
            bonus += 0.05
        if direction == "SELL" and bos == "bearish_bos":
            bonus += 0.05

        return bonus

    def score(direction: str) -> float:
        total = 0.0
        if direction == "BUY":
            if m15_close > ema20_15 > ema50_15 > ema200_15:
                total += 0.35
            if m5_close > ema20_5:
                total += 0.20
            if m5_close > vwap5:
                total += 0.15
            if rsi1 >= 45:
                total += 0.10
            if rsi5 <= 55:
                total += 0.10
            if news_bias == "bullish":
                total += 0.05
            if news_bias == "bearish" and high_impact > 0:
                total -= 0.15
        else:
            if m15_close < ema20_15 < ema50_15 < ema200_15:
                total += 0.35
            if m5_close < ema20_5:
                total += 0.20
            if m5_close < vwap5:
                total += 0.15
            if rsi1 <= 55:
                total += 0.10
            if rsi5 >= 45:
                total += 0.10
            if news_bias == "bearish":
                total += 0.05
            if news_bias == "bullish" and high_impact > 0:
                total -= 0.15
        if age_sec > 90:
            total -= 0.20
        total -= spread_penalty_val
        total += smc_bonus(direction)
        advanced_score, _ = _advanced_bonus(direction, market_data)
        total += advanced_score
        return float(clamp(total, 0.0, 1.0))

    buy_score = score("BUY")
    sell_score = score("SELL")
    buy_notes = _advanced_bonus("BUY", market_data)[1]
    sell_notes = _advanced_bonus("SELL", market_data)[1]
    entry = m1_close or m5_close or m15_close

    if buy_score >= sell_score and buy_score >= AI_MIN_CONF:
        return {
            "signal": "BUY",
            "confidence": round(buy_score, 2),
            "entry": float(entry),
            "stop_loss": None,
            "take_profit": None,
            "reason": _heuristic_reason("BUY", market_data, buy_notes),
            "action_short": action_short_for_signal("BUY"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_scalp_v3",
        }
    if sell_score > buy_score and sell_score >= AI_MIN_CONF:
        return {
            "signal": "SELL",
            "confidence": round(sell_score, 2),
            "entry": float(entry),
            "stop_loss": None,
            "take_profit": None,
            "reason": _heuristic_reason("SELL", market_data, sell_notes),
            "action_short": action_short_for_signal("SELL"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_scalp_v3",
        }
    return {
        "signal": "HOLD",
        "confidence": round(
            clamp(max(buy_score, sell_score), 0.0, AI_MIN_CONF - 0.01), 2
        ),
        "entry": float(entry),
        "stop_loss": None,
        "take_profit": None,
        "reason": _heuristic_reason("HOLD", market_data, []),
        "action_short": action_short_for_signal("HOLD"),
        "provider": "local",
        "provider_display": "Local Heuristic",
        "model": "heuristic_scalp_v3",
    }


def _cache_cleanup(now_ts: float) -> None:
    cutoff = now_ts - AI_CACHE_TTL_SEC
    for key, (saved_ts, _) in list(_CACHE.items()):
        if saved_ts < cutoff:
            _CACHE.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
#  Main analyse function
# ─────────────────────────────────────────────────────────────────────────────


def analyse(asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    if not market_data or not str(market_data.get("summary_text") or "").strip():
        log.error("scalping_payload_missing asset=%s", asset)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Payload-и scalping дастрас нест.",
            "action_short": action_short_for_signal("HOLD"),
            "provider": "none",
            "provider_display": "Unavailable",
            "model": "none",
        }

    enriched = attach_news_context(market_data, asset, "scalping") or market_data
    meta = enriched.get("meta") or {}
    age_sec = int(meta.get("age_sec") or 0)
    close = _pick_close(enriched) or 0.0
    atr = _pick_atr(enriched)
    ts_bar = int(((enriched.get("M1") or {}).get("ts_bar") or 0) or 0)
    cache_key = _build_cache_key(asset, ts_bar, enriched)

    if age_sec > 300:
        log.warning("scalping_stale_guard asset=%s age_sec=%s", asset, age_sec)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": float(close) if close > 0 else None,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"Маълумоти scalping кӯҳна аст (AGE_SEC={age_sec}).",
            "action_short": action_short_for_signal("HOLD"),
            "provider": "precheck",
            "provider_display": "Precheck",
            "model": "stale_guard",
            "news_context": enriched.get("news_context"),
        }

    now = time.time()
    if ts_bar > 0 and cache_key in _CACHE:
        saved_ts, saved_result = _CACHE[cache_key]
        if (now - saved_ts) <= AI_CACHE_TTL_SEC:
            log.info("scalping_cache_hit asset=%s ts_bar=%s", asset, ts_bar)
            return dict(saved_result)

    prompt = _build_prompt(asset, enriched)
    result = run_provider_chain(
        providers=SCALPING_PROVIDERS,
        prompt=prompt,
        schema_name="trade_signal_scalping",
        close=close,
        timeout_budget_sec=AI_BUDGET_SEC,
        timeout_per_call_sec=AI_TIMEOUT_SEC,
        max_retries=AI_MAX_RETRIES,
        actionable_confidence_floor=AI_MIN_CONF,
        hold_confidence_cutoff=AI_CONFIRMATION_SKIP_CONF,
    )
    if result is None:
        log.warning("scalping_chain_exhausted_using_heuristic asset=%s", asset)
        result = _heuristic_signal(enriched, asset)

    signal = str(result.get("signal") or "HOLD").upper()
    confidence = float(result.get("confidence") or 0.0)
    entry = _pick_m1_close(enriched) or safe_float(result.get("entry")) or close
    stop_loss = safe_float(result.get("stop_loss"))
    take_profit = safe_float(result.get("take_profit"))

    # Final news guard
    news = enriched.get("news_context") or {}
    news_conflict = _has_live_high_impact_news_conflict(news, signal)
    news_bias = (
        str(news.get("bias") or "neutral").lower() if news_conflict else "neutral"
    )
    high_impact = int(news.get("high_impact_count") or 0) if news_conflict else 0
    if signal == "BUY" and news_bias == "bearish" and high_impact > 0:
        signal = "HOLD"
        confidence = min(confidence, 0.74)
        result["reason"] = (
            f"{result.get('reason', '')} | Хабарҳои фаврии манфӣ BUY-ро мусдуд кард."
        )
    if signal == "SELL" and news_bias == "bullish" and high_impact > 0:
        signal = "HOLD"
        confidence = min(confidence, 0.74)
        result["reason"] = (
            f"{result.get('reason', '')} | Хабарҳои мусбат SELL-ро мусдуд кард."
        )

    if confidence < AI_MIN_CONF and signal in ("BUY", "SELL"):
        signal = "HOLD"
        result["reason"] = (
            f"Эътимод паст ({confidence:.2f} < {AI_MIN_CONF:.2f}); "
            f"интизори тасдиқи беҳтари M15+SMC."
        )
        stop_loss = None
        take_profit = None

    if signal in ("BUY", "SELL") and confidence < AI_CONFIRMATION_SKIP_CONF:
        confirmed, confirmation_rows = _confirm_direction(
            selected_rank=int(result.get("model_rank") or 0),
            selected_signal=signal,
            prompt=prompt,
            close=close,
        )
        if not confirmed:
            log.warning(
                "scalping_consensus_block asset=%s provider=%s model=%s signal=%s confirmations=%s",
                asset,
                result.get("provider"),
                result.get("model"),
                signal,
                confirmation_rows,
            )
            original_signal = signal
            signal = "HOLD"
            confidence = min(confidence, AI_MIN_CONF - 0.01)
            stop_loss = None
            take_profit = None
            result["reason"] = (
                f"{result.get('reason', '')} | Тасдиқи backup model барои {original_signal} гирифта нашуд; савдо рад шуд."
            ).strip(" |")

    if entry and entry > 0 and signal in ("BUY", "SELL"):
        entry, stop_loss, take_profit = _levels_clamp(
            symbol_meta=meta,
            signal=signal,
            entry=float(entry),
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
        )
        rr = _risk_reward_ratio(signal, entry, stop_loss, take_profit)
        if rr < MIN_RISK_REWARD:
            log.warning(
                "scalping_rr_block asset=%s provider=%s model=%s signal=%s rr=%.2f",
                asset,
                result.get("provider"),
                result.get("model"),
                signal,
                rr,
            )
            signal = "HOLD"
            confidence = min(confidence, AI_MIN_CONF - 0.01)
            stop_loss = None
            take_profit = None
            result["reason"] = (
                f"{result.get('reason', '')} | R:R={rr:.2f} аз ҳадди {MIN_RISK_REWARD:.2f} паст аст; савдо рад шуд."
            ).strip(" |")
    else:
        stop_loss = None
        take_profit = None

    result["signal"] = signal
    result["confidence"] = round(confidence, 2)
    result["entry"] = float(entry) if entry else None
    result["stop_loss"] = float(stop_loss) if stop_loss is not None else None
    result["take_profit"] = float(take_profit) if take_profit is not None else None
    if signal == "HOLD":
        result["action_short"] = action_short_for_signal("HOLD")
        result["stop_loss"] = None
        result["take_profit"] = None
    result["analysis_style"] = "scalping"
    result["asset"] = asset
    result["symbol"] = enriched.get("symbol")
    result["news_context"] = news

    log.info(
        "scalping_done asset=%s provider=%s model=%s rank=%s signal=%s conf=%.2f",
        asset,
        result.get("provider"),
        result.get("model"),
        result.get("model_rank"),
        signal,
        confidence,
    )

    if ts_bar > 0:
        _CACHE[cache_key] = (time.time(), dict(result))
        _cache_cleanup(time.time())
    return result


def diagnose_providers(asset: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    enriched = attach_news_context(market_data, asset, "scalping") or market_data
    prompt = _build_prompt(asset, enriched)
    close = _pick_close(enriched)
    return run_provider_diagnostics(
        providers=SCALPING_PROVIDERS,
        prompt=prompt,
        schema_name="trade_signal_scalping",
        close=close,
        timeout_per_call_sec=AI_TIMEOUT_SEC,
        max_retries=AI_MAX_RETRIES,
    )
