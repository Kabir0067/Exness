"""
intraday_ai.py — Intraday Analysis Engine (D1 / H4 / H1)
Institutional-grade SMC-aware AI analysis with ranked provider chain.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from log_config import build_logger

from .analysis_common import (
    INTRADAY_PROVIDERS,
    action_short_for_signal,
    clamp,
    run_provider_chain,
    run_provider_diagnostics,
    safe_float,
)
from .sym_news import attach_news_context

log = build_logger("ai.intraday.analysis", "intraday_ai.log", level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

AI_TIMEOUT_SEC = 20
AI_MAX_RETRIES = 1
AI_MIN_CONF = 0.70
AI_BUDGET_SEC = 28.0
AI_CACHE_TTL_SEC = 600.0
AI_CONFIRMATION_MODELS = 2
AI_CONFIRMATION_MIN_CONF = 0.55
AI_CONFIRMATION_SKIP_CONF = 0.85
INTRADAY_STALE_GUARD_SEC = 7200
MIN_RISK_REWARD = 1.50

_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _pick_atr(market_data: dict[str, Any]) -> float | None:
    for tf in ("H1", "H4", "D1"):
        value = safe_float((market_data.get(tf) or {}).get("atr_14"))
        if value and value > 0:
            return value
    return None


def _pick_close(market_data: dict[str, Any]) -> float | None:
    for tf in ("H1", "H4", "D1"):
        value = safe_float((market_data.get(tf) or {}).get("last_close"))
        if value and value > 0:
            return value
    return None


def _pick_h1_close(market_data: dict[str, Any]) -> float | None:
    value = safe_float((market_data.get("H1") or {}).get("last_close"))
    return value if value and value > 0 else None


def _news_cache_fingerprint(market_data: dict[str, Any]) -> tuple[Any, ...]:
    news = market_data.get("news_context") or {}
    return (
        str(news.get("bias") or "neutral").lower(),
        round(float(safe_float(news.get("avg_sentiment")) or 0.0), 3),
        int(news.get("high_impact_count") or 0),
        str(news.get("status") or "unknown").lower(),
        str(news.get("ai_summary") or "")[:160],
    )


def _build_cache_key(
    asset: str, ts_bar: int, market_data: dict[str, Any]
) -> tuple[Any, ...]:
    meta = market_data.get("meta") or {}
    return (
        str(asset),
        int(ts_bar),
        round(float(_pick_close(market_data) or 0.0), 4),
        int(meta.get("age_sec") or 0),
        str(market_data.get("summary_text") or "")[:160],
        _news_cache_fingerprint(market_data),
    )


def _spread_penalty(meta: dict[str, Any], atr: float | None) -> tuple[float, str]:
    spread_pts = safe_float(meta.get("spread_points"))
    point = safe_float(meta.get("point")) or 0.0
    if not spread_pts or spread_pts <= 0 or point <= 0 or not atr or atr <= 0:
        return 0.0, "spread_penalty=0"
    spread_price = float(spread_pts) * float(point)
    if spread_price > 0.20 * float(atr):
        return (
            0.20,
            f"spread_penalty=0.20 spread_price={spread_price:.5f} atr={atr:.5f}",
        )
    return 0.0, f"spread_penalty=0 spread_price={spread_price:.5f} atr={atr:.5f}"


def _levels_clamp(
    symbol_meta: dict[str, Any],
    signal: str,
    entry: float,
    stop_loss: float | None,
    take_profit: float | None,
    atr: float | None,
) -> tuple[float, float, float]:
    digits = int(symbol_meta.get("digits") or 2)
    point = float(symbol_meta.get("point") or (10 ** (-digits) if digits > 0 else 0.01))
    stops_level_pts = int(symbol_meta.get("stops_level_points") or 0)
    min_dist = float(stops_level_pts) * point
    if atr and atr > 0:
        min_dist = max(min_dist, atr * 0.5)

    def rnd(v: float) -> float:
        return float(round(v, digits))

    base_sl = atr * 1.0 if atr and atr > 0 else max(min_dist * 1.8, point * 120.0)
    base_tp = atr * 1.8 if atr and atr > 0 else max(min_dist * 2.8, point * 160.0)

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
    signal: str, entry: float | None, stop_loss: float | None, take_profit: float | None
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
    close: float | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if selected_signal not in ("BUY", "SELL"):
        return True, []

    backups = [
        row
        for row in INTRADAY_PROVIDERS
        if int(row.get("rank") or 0) > int(selected_rank or 0)
    ][:AI_CONFIRMATION_MODELS]
    if not backups:
        return True, []

    results = run_provider_diagnostics(
        providers=backups,
        prompt=prompt,
        schema_name="trade_signal_intraday",
        close=close,
        timeout_per_call_sec=min(AI_TIMEOUT_SEC, 12),
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


def _has_live_high_impact_news_conflict(news: dict[str, Any], signal: str) -> bool:
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


def _format_smc_context(market_data: dict[str, Any]) -> str:
    lines: list[str] = []
    for tf in ("D1", "H4", "H1"):
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

        close = safe_float(tf_data.get("last_close")) or 0.0

        if fvgs:
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

        if obs:
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


def _advanced_bonus(direction: str, market_data: dict[str, Any]) -> float:
    h4 = market_data.get("H4") or {}
    h1 = market_data.get("H1") or {}
    score = 0.0

    h4_conf = safe_float((h4.get("confluence") or {}).get("score")) or 50.0
    if direction == "BUY":
        if h4_conf >= 65:
            score += 0.10
        elif h4_conf <= 40:
            score -= 0.10
    else:
        if h4_conf <= 35:
            score += 0.10
        elif h4_conf >= 60:
            score -= 0.10

    h4_hist = safe_float((h4.get("macd") or {}).get("histogram")) or 0.0
    if (direction == "BUY" and h4_hist > 0) or (direction == "SELL" and h4_hist < 0):
        score += 0.08
    elif h4_hist != 0:
        score -= 0.06

    stoch = h1.get("stoch_rsi") or {}
    stoch_cross = str(stoch.get("cross") or "none")
    stoch_zone = str(stoch.get("zone") or "neutral")
    if direction == "BUY":
        if stoch_cross == "bullish_cross" or stoch_zone == "oversold":
            score += 0.05
        elif stoch_cross == "bearish_cross" or stoch_zone == "overbought":
            score -= 0.05
    else:
        if stoch_cross == "bearish_cross" or stoch_zone == "overbought":
            score += 0.05
        elif stoch_cross == "bullish_cross" or stoch_zone == "oversold":
            score -= 0.05

    volume_ratio = safe_float((h1.get("volume") or {}).get("ratio")) or 1.0
    pattern = str((h1.get("price_action") or {}).get("pattern") or "none")
    if volume_ratio >= 1.15:
        if (
            direction == "BUY"
            and pattern in {"bullish_engulfing", "hammer", "breakout_close"}
        ) or (
            direction == "SELL"
            and pattern in {"bearish_engulfing", "shooting_star", "breakdown_close"}
        ):
            score += 0.05

    return score


def _build_prompt(asset: str, market_data: dict[str, Any]) -> str:
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

    return f"""You are a SENIOR INSTITUTIONAL INTRADAY ANALYST specializing in {asset_label}.
You combine macro analysis with Smart Money Concepts (SMC): Order Blocks, FVGs, Market Structure.
Return EXACTLY one JSON object. Use ONLY the data below. No invented data.

═══ ASSET & CONTEXT ════════════════════════════════════════
Asset         : {asset_label}
Style         : INTRADAY (D1/H4/H1 multi-timeframe)
Venue         : Exness MT5 | Symbol: {market_data.get('symbol', asset)}
Current Price : {close:.2f}
Market State  : bias={state.get('directional_bias', 'balanced')} | regime={state.get('regime', 'balanced')} | execution={state.get('execution_state', 'normal')}
News Source   : status={news.get('status', 'unknown')} | confidence_mode={news.get('confidence_mode', 'unknown')}
News Policy   : Live MarketAux macro conflict can block. AI fallback news is SOFT directional context and should adjust confidence, not hard-block by itself.

═══ MARKET SNAPSHOT ════════════════════════════════════════
{summary}

═══ SMC INSTITUTIONAL CONTEXT ══════════════════════════════
{smc_context}

═══ SCORING FRAMEWORK ══════════════════════════════════════
Start at 0.00

D1 MACRO ALIGNMENT (+0.40):
  BUY  → D1: close > EMA20 > EMA50 > EMA200
  SELL → D1: close < EMA20 < EMA50 < EMA200

H4 CONFIRMATION (+0.25):
  BUY  → H4: close > EMA20 AND EMA20 > EMA50
  SELL → H4: close < EMA20 AND EMA20 < EMA50

H1 ENTRY QUALITY (+0.15):
  BUY  → H1 close above VWAP
  SELL → H1 close below VWAP

MOMENTUM (+0.10 each):
  H1 RSI direction-aligned (BUY: 45-65 | SELL: 35-55)
  H1 RSI not extended      (BUY: not >70 | SELL: not <30)

ADVANCED FLOW:
  +0.10 if H4 confluence score >= 65 in direction
  +0.08 if H4 MACD histogram supports direction
  +0.05 if H1 StochRSI cross/zone supports direction
  +0.05 if H1 price action plus volume confirms entry

SMC BONUS/PENALTY:
  +0.12 if D1 Market Structure BOS confirms direction (most important)
  +0.10 if price is respecting a key H4 Order Block
  +0.08 if H4/H1 FVG supports direction (entering discount for BUY)
  +0.05 if D1 CHoCH confirms trend change aligning with direction
  -0.12 if price in opposite zone (PREMIUM for BUY, DISCOUNT for SELL) on D1

NEWS & MACRO:
  -0.20 if data stale >5400s
  -{penalty:.2f} spread cost penalty
  -0.15 if high-impact macro news conflicts with direction
  +0.08 if macro news confirms direction
  Clamp confidence [0.00, 1.00]
  If confidence < {AI_MIN_CONF:.2f} → signal = HOLD
  When D1 + H4 + H1 + SMC align strongly, prefer a directional call over HOLD.

═══ TRADE LEVELS ════════════════════════════════════════════
entry       = H1 last_close
stop_loss   = 1.0 × ATR14(H1), placed below/above nearest invalidation / Order Block
take_profit = at least 1.8 × ATR14(H1), targeting next FVG mid or swing level
Minimum R:R = {MIN_RISK_REWARD:.2f}; if lower → HOLD

═══ OUTPUT RULES ════════════════════════════════════════════
- reason: Tajik Cyrillic, institutional senior style
- Mention D1 macro, H4 confirmation, H1 entry, advanced indicators, SMC zones, news impact
- action_short: EXACTLY one of: "Харид", "Фурӯш", "Интизор"
- No markdown. No text outside JSON.

═══ DEBUG ═══════════════════════════════════════════════════
spread_debug  : {penalty_debug}
news_bias     : {news.get('bias', 'unknown')}
news_sentiment: {news.get('avg_sentiment', 0.0)}
news_impact   : {news.get('high_impact_count', 0)}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristic fallback
# ─────────────────────────────────────────────────────────────────────────────


def _heuristic_signal(market_data: dict[str, Any], asset: str) -> dict[str, Any]:
    d1 = market_data.get("D1") or {}
    h4 = market_data.get("H4") or {}
    h1 = market_data.get("H1") or {}
    meta = market_data.get("meta") or {}
    news = market_data.get("news_context") or {}

    close = _pick_close(market_data) or 0.0
    if close <= 0:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Маълумоти intraday дастрас нест.",
            "action_short": action_short_for_signal("HOLD"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_intraday_v3",
        }

    d1_close = safe_float(d1.get("last_close")) or close
    d1_e20 = safe_float(d1.get("ema_20")) or d1_close
    d1_e50 = safe_float(d1.get("ema_50")) or d1_close
    d1_e200 = safe_float(d1.get("ema_200")) or d1_close
    h4_close = safe_float(h4.get("last_close")) or close
    h4_e20 = safe_float(h4.get("ema_20")) or h4_close
    h4_e50 = safe_float(h4.get("ema_50")) or h4_close
    h1_close = safe_float(h1.get("last_close")) or close
    h1_rsi = safe_float(h1.get("rsi_14")) or 50.0
    h1_vwap = safe_float(h1.get("vwap")) or h1_close

    age_sec = int(meta.get("age_sec") or 0)
    atr = _pick_atr(market_data)
    spread_pen, _ = _spread_penalty(meta, atr)
    news_bias = str(news.get("bias") or "neutral").lower()
    high_impact = int(news.get("high_impact_count") or 0)

    def smc_bonus(direction: str) -> float:
        bonus = 0.0
        d1_data = market_data.get("D1") or {}
        h4_data = market_data.get("H4") or {}
        d1_struct = d1_data.get("structure") or {}
        d1_zone = str(d1_struct.get("zone") or "equilibrium").lower()
        d1_bos = d1_struct.get("last_bos") or ""
        d1_choch = d1_struct.get("choch") or ""
        h4_obs = h4_data.get("order_blocks") or []

        if direction == "BUY":
            if d1_zone == "premium":
                bonus -= 0.12
            if d1_bos == "bullish_bos":
                bonus += 0.12
            if d1_choch == "bullish_choch":
                bonus += 0.05
        else:
            if d1_zone == "discount":
                bonus -= 0.12
            if d1_bos == "bearish_bos":
                bonus += 0.12
            if d1_choch == "bearish_choch":
                bonus += 0.05

        # H4 Order Block
        for ob in h4_obs:
            ob_dir = str(ob.get("direction") or "").lower()
            ob_top = float(ob.get("top") or 0)
            ob_bot = float(ob.get("bottom") or 0)
            if (
                direction == "BUY"
                and ob_dir == "bullish"
                and ob_bot <= h4_close <= ob_top
            ):
                bonus += 0.10
                break
            if (
                direction == "SELL"
                and ob_dir == "bearish"
                and ob_bot <= h4_close <= ob_top
            ):
                bonus += 0.10
                break

        return bonus

    def score(direction: str) -> float:
        total = 0.0
        if direction == "BUY":
            if d1_close > d1_e20 > d1_e50 > d1_e200:
                total += 0.40
            if h4_close > h4_e20 and h4_e20 > h4_e50:
                total += 0.25
            if h1_close > h1_vwap:
                total += 0.15
            if h1_rsi >= 45:
                total += 0.10
            if h1_rsi <= 65:
                total += 0.10
            if news_bias == "bullish":
                total += 0.08
            if news_bias == "bearish" and high_impact > 0:
                total -= 0.15
        else:
            if d1_close < d1_e20 < d1_e50 < d1_e200:
                total += 0.40
            if h4_close < h4_e20 and h4_e20 < h4_e50:
                total += 0.25
            if h1_close < h1_vwap:
                total += 0.15
            if h1_rsi <= 55:
                total += 0.10
            if h1_rsi >= 35:
                total += 0.10
            if news_bias == "bearish":
                total += 0.08
            if news_bias == "bullish" and high_impact > 0:
                total -= 0.15
        if age_sec > 5400:
            total -= 0.20
        total -= spread_pen
        total += smc_bonus(direction)
        total += _advanced_bonus(direction, market_data)
        return float(clamp(total, 0.0, 1.0))

    buy_score = score("BUY")
    sell_score = score("SELL")

    if buy_score >= sell_score and buy_score >= AI_MIN_CONF:
        return {
            "signal": "BUY",
            "confidence": round(buy_score, 2),
            "entry": float(h1_close),
            "stop_loss": None,
            "take_profit": None,
            "reason": "Heuristic: D1/H4 тренди боло, SMC зонаи мусоид, хабар пуштибонӣ мекунад.",
            "action_short": action_short_for_signal("BUY"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_intraday_v3",
        }
    if sell_score > buy_score and sell_score >= AI_MIN_CONF:
        return {
            "signal": "SELL",
            "confidence": round(sell_score, 2),
            "entry": float(h1_close),
            "stop_loss": None,
            "take_profit": None,
            "reason": "Heuristic: D1/H4 тренди поён, SMC тасдиқ кард.",
            "action_short": action_short_for_signal("SELL"),
            "provider": "local",
            "provider_display": "Local Heuristic",
            "model": "heuristic_intraday_v3",
        }
    return {
        "signal": "HOLD",
        "confidence": round(
            clamp(max(buy_score, sell_score), 0.0, AI_MIN_CONF - 0.01), 2
        ),
        "entry": float(h1_close),
        "stop_loss": None,
        "take_profit": None,
        "reason": "Heuristic: D1/H4/H1 ва SMC ҳанӯз ба таври пурра ҳамоҳанг нестанд.",
        "action_short": action_short_for_signal("HOLD"),
        "provider": "local",
        "provider_display": "Local Heuristic",
        "model": "heuristic_intraday_v3",
    }


def _cache_cleanup(now_ts: float) -> None:
    cutoff = now_ts - AI_CACHE_TTL_SEC
    for key, (saved_ts, _) in list(_CACHE.items()):
        if saved_ts < cutoff:
            _CACHE.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
#  Main analyse function
# ─────────────────────────────────────────────────────────────────────────────


def analyse_intraday(asset: str, market_data: dict[str, Any]) -> dict[str, Any]:
    if not market_data or not str(market_data.get("summary_text") or "").strip():
        log.error("intraday_payload_missing asset=%s", asset)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Payload-и intraday дастрас нест.",
            "action_short": action_short_for_signal("HOLD"),
            "provider": "none",
            "provider_display": "Unavailable",
            "model": "none",
        }

    enriched = attach_news_context(market_data, asset, "intraday") or market_data
    meta = enriched.get("meta") or {}
    age_sec = int(meta.get("age_sec") or 0)
    close = _pick_close(enriched) or 0.0
    atr = _pick_atr(enriched)
    ts_bar = int(((enriched.get("H1") or {}).get("ts_bar") or 0) or 0)
    cache_key = _build_cache_key(asset, ts_bar, enriched)

    if age_sec > INTRADAY_STALE_GUARD_SEC:
        log.warning("intraday_stale_guard asset=%s age_sec=%s", asset, age_sec)
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": float(close) if close > 0 else None,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"Маълумоти intraday кӯҳна аст (AGE_SEC={age_sec}).",
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
            log.info("intraday_cache_hit asset=%s ts_bar=%s", asset, ts_bar)
            return dict(saved_result)

    prompt = _build_prompt(asset, enriched)
    result = run_provider_chain(
        providers=INTRADAY_PROVIDERS,
        prompt=prompt,
        schema_name="trade_signal_intraday",
        close=close,
        timeout_budget_sec=AI_BUDGET_SEC,
        timeout_per_call_sec=AI_TIMEOUT_SEC,
        max_retries=AI_MAX_RETRIES,
        actionable_confidence_floor=AI_MIN_CONF,
        hold_confidence_cutoff=AI_CONFIRMATION_SKIP_CONF,
    )
    if result is None:
        log.warning("intraday_chain_exhausted_using_heuristic asset=%s", asset)
        result = _heuristic_signal(enriched, asset)

    signal = str(result.get("signal") or "HOLD").upper()
    confidence = float(result.get("confidence") or 0.0)
    entry = _pick_h1_close(enriched) or safe_float(result.get("entry")) or close
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
            f"{result.get('reason', '')} | Контексти macro барои BUY зид аст."
        )
    if signal == "SELL" and news_bias == "bullish" and high_impact > 0:
        signal = "HOLD"
        confidence = min(confidence, 0.74)
        result["reason"] = (
            f"{result.get('reason', '')} | Контексти macro барои SELL зид аст."
        )

    if confidence < AI_MIN_CONF and signal in ("BUY", "SELL"):
        signal = "HOLD"
        result["reason"] = (
            f"Эътимод паст ({confidence:.2f} < {AI_MIN_CONF:.2f}); "
            f"интизори ҳамоҳангии D1+H4+SMC."
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
                "intraday_consensus_block asset=%s provider=%s model=%s signal=%s confirmations=%s",
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
                "intraday_rr_block asset=%s provider=%s model=%s signal=%s rr=%.2f",
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
    result["analysis_style"] = "intraday"
    result["asset"] = asset
    result["symbol"] = enriched.get("symbol")
    result["news_context"] = news

    log.info(
        "intraday_done asset=%s provider=%s model=%s rank=%s signal=%s conf=%.2f",
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


def diagnose_providers(asset: str, market_data: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = attach_news_context(market_data, asset, "intraday") or market_data
    prompt = _build_prompt(asset, enriched)
    close = _pick_close(enriched)
    return run_provider_diagnostics(
        providers=INTRADAY_PROVIDERS,
        prompt=prompt,
        schema_name="trade_signal_intraday",
        close=close,
        timeout_per_call_sec=AI_TIMEOUT_SEC,
        max_retries=AI_MAX_RETRIES,
    )
