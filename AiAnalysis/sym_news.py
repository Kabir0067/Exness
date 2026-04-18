"""
sym_news.py — News context engine
Primary:  MarketAux API (live sentiment)
Fallback: AI-generated macro context when MarketAux is unavailable / rate-limited
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from log_config import build_logger

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


log = build_logger("ai.sym_news", "sym_news.log")


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

MARKETAUX_BASE_URL = "https://api.marketaux.com/v1"
MARKETAUX_API_KEY = (
    os.getenv("MARKETAUX_API_KEY") or os.getenv("MARKETAUX") or ""
).strip()

GOLD_SEARCH_QUERY = (
    '"gold" OR xau OR bullion OR "gold price" OR "Federal Reserve" '
    'OR "US dollar" OR "Treasury yields" OR "safe haven" OR "inflation"'
)

HIGH_IMPACT_KEYWORDS = (
    "cpi",
    "ppi",
    "nfp",
    "fomc",
    "fed",
    "powell",
    "inflation",
    "rates",
    "yield",
    "treasury",
    "etf",
    "sec",
    "tariff",
    "geopolitical",
    "war",
    "sanctions",
    "recession",
    "gdp",
    "jobs",
    "unemployment",
    "hawkish",
    "dovish",
    "rate cut",
    "rate hike",
)

NEWS_PRESETS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("btc", "scalping"): {"hours": 4, "limit": 8, "threshold": 0.18},
    ("btc", "intraday"): {"hours": 12, "limit": 14, "threshold": 0.12},
    ("gold", "scalping"): {"hours": 4, "limit": 8, "threshold": 0.18},
    ("gold", "intraday"): {"hours": 18, "limit": 16, "threshold": 0.12},
}

# Simple in-memory cache for AI fallback news (avoid hammering AI API)
_AI_NEWS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_AI_NEWS_CACHE_TTL = 1800.0  # 30 minutes
_AI_FALLBACK_DIRECTIONAL_SENTIMENT_MIN = 0.18
_NEWS_CONTEXT_CACHE: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
_NEWS_CACHE_TTL_SEC = {
    "scalping": 3600.0,
    "intraday": 10800.0,
}
_FALLBACK_CACHE_TTL_SEC = {
    "scalping": 3600.0,
    "intraday": 10800.0,
}
_NEWS_CONTEXT_CACHE_LOADED = False
_NEWS_CONTEXT_CACHE_PATH = (
    Path(__file__).resolve().parent / "cache" / "news_context_cache.json"
)
_NEWS_FETCH_LOCK_PATH = Path(__file__).resolve().parent / "cache" / "news_context.lock"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_asset(asset: str) -> str:
    value = str(asset or "").strip().lower()
    return "btc" if "btc" in value else "gold"


def _published_after(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M"
    )


def _http_get_json(
    path: str, params: Dict[str, Any], timeout: int = 20
) -> Dict[str, Any]:
    if not MARKETAUX_API_KEY:
        raise RuntimeError("marketaux_api_key_missing")
    query = urlencode({**params, "api_token": MARKETAUX_API_KEY})
    url = f"{MARKETAUX_BASE_URL}{path}?{query}"
    req = Request(url, headers={"User-Agent": "AiBot/1.0"})
    log.info("marketaux_request path=%s", path)
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_params(asset: str, trade_style: str) -> Dict[str, Any]:
    preset = NEWS_PRESETS[(asset, trade_style)]
    base: Dict[str, Any] = {
        "language": "en",
        "group_similar": "true",
        "published_after": _published_after(preset["hours"]),
        "limit": preset["limit"],
        "sort": "published_desc",
    }
    if asset == "btc":
        base.update(
            {
                "symbols": "CC:BTC",
                "filter_entities": "true",
                "must_have_entities": "true",
            }
        )
    else:
        base.update(
            {
                "search": GOLD_SEARCH_QUERY,
                "entity_types": "commodity,forex,equity,index,etf",
            }
        )
    return base


def _headline_bias(article: Dict[str, Any], asset: str) -> float:
    entity_scores: List[float] = []
    for entity in article.get("entities") or []:
        score = entity.get("sentiment_score")
        symbol = str(entity.get("symbol") or "")
        if score is None:
            continue
        if asset == "btc" and symbol == "CC:BTC":
            entity_scores.append(float(score))
        elif asset == "gold":
            entity_scores.append(float(score))
    if entity_scores:
        return float(mean(entity_scores))
    fallback = article.get("sentiment_score")
    if fallback is not None:
        try:
            return float(fallback)
        except Exception:
            pass
    return 0.0


def _headline_rows(news_json: Dict[str, Any], asset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for article in news_json.get("data", []) or []:
        title = str(article.get("title") or "").strip()
        source = str(article.get("source") or "").strip()
        published_at = str(article.get("published_at") or "").strip()
        score = _headline_bias(article, asset)
        headline_lc = title.lower()
        high_impact = any(kw in headline_lc for kw in HIGH_IMPACT_KEYWORDS)
        rows.append(
            {
                "title": title,
                "source": source,
                "published_at": published_at,
                "url": str(article.get("url") or "").strip(),
                "score": round(score, 4),
                "high_impact": high_impact,
            }
        )
    return rows


def _classify_bias(avg_sentiment: float, threshold: float) -> str:
    if avg_sentiment >= threshold:
        return "bullish"
    if avg_sentiment <= -threshold:
        return "bearish"
    return "neutral"


def _context_cache_key(asset: str, trade_style: str) -> str:
    return f"{asset}::{trade_style}"


def _context_cache_ttl(trade_style: str, status: str) -> float:
    style_key = (
        "intraday" if str(trade_style).strip().lower() == "intraday" else "scalping"
    )
    status_key = str(status or "").strip().lower()
    if status_key == "live":
        return float(_NEWS_CACHE_TTL_SEC[style_key])
    return float(_FALLBACK_CACHE_TTL_SEC[style_key])


def _load_news_context_cache(force: bool = False) -> None:
    global _NEWS_CONTEXT_CACHE_LOADED
    if _NEWS_CONTEXT_CACHE_LOADED and not force:
        return
    if force:
        _NEWS_CONTEXT_CACHE.clear()
    _NEWS_CONTEXT_CACHE_LOADED = True
    try:
        if not _NEWS_CONTEXT_CACHE_PATH.exists():
            return
        payload = json.loads(_NEWS_CONTEXT_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        now = time.time()
        for key, row in payload.items():
            if not isinstance(row, dict):
                continue
            saved_ts = float(row.get("saved_ts") or 0.0)
            ttl_sec = float(row.get("ttl_sec") or 0.0)
            data = row.get("data")
            if saved_ts <= 0 or ttl_sec <= 0 or not isinstance(data, dict):
                continue
            if (now - saved_ts) >= ttl_sec:
                continue
            _NEWS_CONTEXT_CACHE[str(key)] = (saved_ts, ttl_sec, data)
    except Exception:
        log.exception(
            "news_context_cache_load_failed path=%s", _NEWS_CONTEXT_CACHE_PATH
        )


def _save_news_context_cache() -> None:
    try:
        _NEWS_CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable: Dict[str, Dict[str, Any]] = {}
        for key, (saved_ts, ttl_sec, data) in _NEWS_CONTEXT_CACHE.items():
            serializable[str(key)] = {
                "saved_ts": saved_ts,
                "ttl_sec": ttl_sec,
                "data": data,
            }
        _NEWS_CONTEXT_CACHE_PATH.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        log.exception(
            "news_context_cache_save_failed path=%s", _NEWS_CONTEXT_CACHE_PATH
        )


def _get_cached_context(
    asset: str, trade_style: str, force_refresh: bool = False
) -> Optional[Dict[str, Any]]:
    _load_news_context_cache(force=force_refresh)
    key = _context_cache_key(asset, trade_style)
    cached = _NEWS_CONTEXT_CACHE.get(key)
    if not cached:
        return None
    saved_ts, ttl_sec, data = cached
    age = time.time() - saved_ts
    if age >= ttl_sec:
        _NEWS_CONTEXT_CACHE.pop(key, None)
        _save_news_context_cache()
        return None
    log.info(
        "news_context_cache_hit asset=%s style=%s status=%s age_sec=%d ttl_sec=%d",
        asset,
        trade_style,
        data.get("status"),
        int(age),
        int(ttl_sec),
    )
    return deepcopy(data)


def _put_cached_context(
    asset: str, trade_style: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    _load_news_context_cache()
    ttl_sec = _context_cache_ttl(trade_style, str(data.get("status") or ""))
    _NEWS_CONTEXT_CACHE[_context_cache_key(asset, trade_style)] = (
        time.time(),
        ttl_sec,
        deepcopy(data),
    )
    _save_news_context_cache()
    return data


def _acquire_news_fetch_lock(timeout_sec: float = 15.0) -> Optional[int]:
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            _NEWS_FETCH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                str(_NEWS_FETCH_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
            return fd
        except FileExistsError:
            time.sleep(0.2)
        except Exception:
            log.exception(
                "news_fetch_lock_acquire_failed path=%s", _NEWS_FETCH_LOCK_PATH
            )
            return None
    log.warning("news_fetch_lock_timeout path=%s", _NEWS_FETCH_LOCK_PATH)
    return None


def _release_news_fetch_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        if _NEWS_FETCH_LOCK_PATH.exists():
            _NEWS_FETCH_LOCK_PATH.unlink()
    except Exception:
        log.exception("news_fetch_lock_release_failed path=%s", _NEWS_FETCH_LOCK_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  AI Fallback — uses Gemini free tier to generate macro context
# ─────────────────────────────────────────────────────────────────────────────


def _http_post_json(
    url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            return int(getattr(resp, "status", 200)), resp.read().decode(
                "utf-8", errors="replace"
            )
    except urllib.error.HTTPError as exc:
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = ""
        return int(getattr(exc, "code", 0) or 0), body_text
    except Exception as exc:
        return 0, str(exc)


def _ai_news_fallback(
    asset: str, trade_style: str, preset: Dict[str, Any]
) -> Dict[str, Any]:
    """
    When MarketAux is unavailable, ask Gemini (free) for a quick macro sentiment
    based on general market knowledge. Result is cached for 30 minutes.
    """
    cache_key = f"{asset}::{trade_style}"
    now = time.time()
    cached = _AI_NEWS_CACHE.get(cache_key)
    if cached:
        ts, data = cached
        if (now - ts) < _AI_NEWS_CACHE_TTL:
            log.info(
                "ai_news_cache_hit asset=%s style=%s age_sec=%d",
                asset,
                trade_style,
                int(now - ts),
            )
            return data

    api_key = (
        os.getenv("GEMINI_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ).strip()
    if not api_key:
        # Try Groq as second fallback
        return _ai_news_groq_fallback(asset, trade_style, preset)

    asset_label = "Bitcoin (BTC/USD)" if asset == "btc" else "Gold (XAU/USD)"
    window_h = preset["hours"]
    style_label = trade_style.upper()

    prompt = f"""You are an institutional macro analyst. The date/time is approximately {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.

Provide a concise {style_label} macro/news sentiment context for {asset_label} based on current market conditions.
Focus on: central bank policy, USD strength, geopolitical risk, inflation data, safe-haven demand (for gold) or risk appetite (for BTC).
Window: last {window_h} hours.

Respond ONLY with this exact JSON structure (no markdown, no prose):
{{
  "bias": "bullish" | "bearish" | "neutral",
  "avg_sentiment": <float between -1.0 and 1.0>,
  "high_impact_count": <integer 0-5>,
  "key_themes": [<string>, <string>, <string>],
  "summary": "<1-2 sentence Tajik Cyrillic summary of macro context>"
}}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json", "User-Agent": "AiBot/1.0"}

    try:
        status, raw = _http_post_json(url, body, headers, timeout=15)
        if status != 200:
            log.warning("ai_news_gemini_error status=%s", status)
            return _ai_news_groq_fallback(asset, trade_style, preset)

        payload = json.loads(raw) if raw else {}
        candidates = payload.get("candidates") or []
        if not candidates:
            return _ai_news_groq_fallback(asset, trade_style, preset)
        parts_list = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = next(
            (
                str(item.get("text"))
                for item in parts_list
                if isinstance(item, dict) and item.get("text")
            ),
            "",
        )

        # Parse JSON from text
        text = text.strip().replace("```json", "").replace("```", "").strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            text = text[s : e + 1]
        obj = json.loads(text)

        bias = str(obj.get("bias") or "neutral").lower()
        avg_sentiment = float(obj.get("avg_sentiment") or 0.0)
        high_impact = int(obj.get("high_impact_count") or 0)
        themes = list(obj.get("key_themes") or [])
        summary_ai = str(obj.get("summary") or "")

        result = _build_ai_news_result(
            asset,
            trade_style,
            preset,
            bias,
            avg_sentiment,
            high_impact,
            themes,
            summary_ai,
            "gemini-ai-fallback",
        )
        _AI_NEWS_CACHE[cache_key] = (time.time(), result)
        log.info(
            "ai_news_gemini_fallback_ok asset=%s style=%s bias=%s",
            asset,
            trade_style,
            bias,
        )
        return result

    except Exception:
        log.exception(
            "ai_news_gemini_fallback_failed asset=%s style=%s", asset, trade_style
        )
        return _ai_news_groq_fallback(asset, trade_style, preset)


def _ai_news_groq_fallback(
    asset: str, trade_style: str, preset: Dict[str, Any]
) -> Dict[str, Any]:
    """Second-tier AI fallback using Groq."""
    cache_key = f"{asset}::{trade_style}::groq"
    now = time.time()
    cached = _AI_NEWS_CACHE.get(cache_key)
    if cached:
        ts, data = cached
        if (now - ts) < _AI_NEWS_CACHE_TTL:
            return data

    api_key = (os.getenv("GROQ_AI_API_KEY") or "").strip()
    if not api_key:
        log.warning("ai_news_groq_no_key asset=%s style=%s", asset, trade_style)
        return _neutral_fallback(asset, trade_style, preset, "no_ai_key")

    asset_label = "Bitcoin (BTC/USD)" if asset == "btc" else "Gold (XAU/USD)"
    window_h = preset["hours"]
    prompt = f"""Institutional macro analyst. Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.
{asset_label} macro sentiment for last {window_h}h ({trade_style.upper()} context).
Respond ONLY in JSON: {{"bias":"bullish|bearish|neutral","avg_sentiment":<-1.0 to 1.0>,"high_impact_count":<0-5>,"key_themes":["..",".."],"summary":"<Tajik Cyrillic 1 sentence>"}}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AiBot/1.0",
    }
    try:
        status, raw = _http_post_json(url, body, headers, timeout=10)
        if status != 200:
            log.warning("ai_news_groq_error status=%s", status)
            return _neutral_fallback(asset, trade_style, preset, f"groq_http_{status}")

        outer = json.loads(raw)
        choices = outer.get("choices") or []
        if not choices:
            return _neutral_fallback(asset, trade_style, preset, "groq_no_choices")
        content = (choices[0].get("message") or {}).get("content") or ""
        obj = json.loads(content.strip())

        bias = str(obj.get("bias") or "neutral").lower()
        avg_sentiment = float(obj.get("avg_sentiment") or 0.0)
        high_impact = int(obj.get("high_impact_count") or 0)
        themes = list(obj.get("key_themes") or [])
        summary_ai = str(obj.get("summary") or "")

        result = _build_ai_news_result(
            asset,
            trade_style,
            preset,
            bias,
            avg_sentiment,
            high_impact,
            themes,
            summary_ai,
            "groq-ai-fallback",
        )
        _AI_NEWS_CACHE[cache_key] = (time.time(), result)
        log.info(
            "ai_news_groq_fallback_ok asset=%s style=%s bias=%s",
            asset,
            trade_style,
            bias,
        )
        return result

    except Exception:
        log.exception(
            "ai_news_groq_fallback_failed asset=%s style=%s", asset, trade_style
        )
        return _neutral_fallback(asset, trade_style, preset, "ai_fallback_exception")


def _build_ai_news_result(
    asset: str,
    trade_style: str,
    preset: Dict[str, Any],
    bias: str,
    avg_sentiment: float,
    high_impact: int,
    themes: List[str],
    summary_ai: str,
    source_tag: str,
) -> Dict[str, Any]:
    theme_text = " | ".join(themes[:3]) if themes else ""
    summary = (
        f"NEWS_OVERLAY={trade_style.upper()} | STATUS=AI_FALLBACK | SOURCE={source_tag.upper()} | "
        f"ASSET={asset.upper()} | WINDOW_HOURS={preset['hours']} | "
        f"AVG_SENTIMENT={avg_sentiment:+.2f} | BIAS={bias.upper()} | "
        f"HIGH_IMPACT={high_impact} | THEMES={theme_text} | "
        f"RULE={'risk_filter_only' if trade_style == 'scalping' else 'directional_overlay'}"
    )
    if summary_ai:
        summary += f"\nAI_CONTEXT: {summary_ai}"

    return {
        "status": "ai_fallback",
        "asset": asset,
        "trade_style": trade_style,
        "bias": bias,
        "avg_sentiment": round(avg_sentiment, 4),
        "headline_count": 0,
        "high_impact_count": high_impact,
        "window_hours": preset["hours"],
        "headlines": [],
        "summary_text": summary,
        "ai_themes": themes,
        "ai_summary": summary_ai,
    }


def _neutral_fallback(
    asset: str, trade_style: str, preset: Dict[str, Any], reason: str
) -> Dict[str, Any]:
    return {
        "status": "neutral_fallback",
        "asset": asset,
        "trade_style": trade_style,
        "bias": "neutral",
        "avg_sentiment": 0.0,
        "headline_count": 0,
        "high_impact_count": 0,
        "window_hours": preset["hours"],
        "headlines": [],
        "summary_text": (
            f"NEWS_OVERLAY={trade_style.upper()} | STATUS=NEUTRAL_FALLBACK | "
            f"REASON={reason} | RULE={'risk_filter_only' if trade_style == 'scalping' else 'directional_overlay'}"
        ),
    }


def _effective_trading_context(context: Dict[str, Any]) -> Dict[str, Any]:
    effective = deepcopy(context or {})
    status = str(effective.get("status") or "").lower()
    if status == "live":
        effective["confidence_mode"] = "live"
        return effective

    if status == "ai_fallback":
        # Preserve raw AI output for logging/diagnostics only.
        effective["raw_bias"] = str(effective.get("bias") or "neutral").lower()
        effective["raw_avg_sentiment"] = float(effective.get("avg_sentiment") or 0.0)
        effective["raw_high_impact_count"] = int(
            effective.get("high_impact_count") or 0
        )
        # SAFETY: AI-generated sentiment is not market data — never let it
        # modify live trade confidence.  Always neutralize for trading.
        effective["bias"] = "neutral"
        effective["avg_sentiment"] = 0.0
        effective["high_impact_count"] = 0
        effective["confidence_mode"] = "ai_fallback_neutralized"
        summary = str(effective.get("summary_text") or "").rstrip()
        if summary:
            summary += "\nEFFECTIVE_RULE=AI_FALLBACK_NEUTRALIZED | REASON=synthetic_sentiment_blocked_from_trading"
        effective["summary_text"] = summary
        log.info(
            "ai_fallback_neutralized | raw_bias=%s raw_sentiment=%.4f raw_high_impact=%d",
            effective["raw_bias"],
            effective["raw_avg_sentiment"],
            effective["raw_high_impact_count"],
        )
        return effective

    # AI-generated news fallback is too weak to drive direction or high-impact blocking.
    effective["raw_bias"] = effective.get("bias")
    effective["raw_avg_sentiment"] = effective.get("avg_sentiment")
    effective["raw_high_impact_count"] = effective.get("high_impact_count")
    effective["bias"] = "neutral"
    effective["avg_sentiment"] = 0.0
    effective["high_impact_count"] = 0
    effective["confidence_mode"] = "neutralized_fallback"
    summary = str(effective.get("summary_text") or "").rstrip()
    if summary:
        summary += (
            "\nEFFECTIVE_RULE=NEUTRALIZED_FOR_TRADING | REASON=non_live_news_context"
        )
    effective["summary_text"] = summary
    return effective


# ─────────────────────────────────────────────────────────────────────────────
#  Main public interface
# ─────────────────────────────────────────────────────────────────────────────


def build_news_context(asset: str, trade_style: str) -> Dict[str, Any]:
    asset_key = _normalize_asset(asset)
    style_key = (
        "intraday" if str(trade_style).strip().lower() == "intraday" else "scalping"
    )
    preset = NEWS_PRESETS[(asset_key, style_key)]
    cached = _get_cached_context(asset_key, style_key)
    if cached is not None:
        return cached
    lock_fd = _acquire_news_fetch_lock()
    try:
        cached = _get_cached_context(asset_key, style_key, force_refresh=True)
        if cached is not None:
            return cached

        if not MARKETAUX_API_KEY:
            log.warning(
                "marketaux_key_missing_using_ai_fallback asset=%s style=%s",
                asset_key,
                style_key,
            )
            return _put_cached_context(
                asset_key, style_key, _ai_news_fallback(asset_key, style_key, preset)
            )

        try:
            payload = _http_get_json("/news/all", _request_params(asset_key, style_key))
            rows = _headline_rows(payload, asset_key)
            scores = [row["score"] for row in rows]
            avg_sentiment = round(mean(scores), 4) if scores else 0.0
            bias = _classify_bias(avg_sentiment, preset["threshold"])
            high_impact = sum(1 for row in rows if row["high_impact"])
            headline_lines: List[str] = []
            for row in rows[:5]:
                headline_lines.append(
                    f"- {row['published_at']} | {row['source']} | score={row['score']:+.2f} | {row['title']}"
                )

            summary = (
                f"NEWS_OVERLAY={style_key.upper()} | STATUS=LIVE | SOURCE=MARKETAUX | ASSET={asset_key.upper()} | "
                f"WINDOW_HOURS={preset['hours']} | ARTICLES={len(rows)} | AVG_SENTIMENT={avg_sentiment:+.2f} | "
                f"BIAS={bias.upper()} | HIGH_IMPACT={high_impact} | "
                f"RULE={'risk_filter_only' if style_key == 'scalping' else 'directional_overlay'}"
            )
            if headline_lines:
                summary += "\nTOP_HEADLINES:\n" + "\n".join(headline_lines)

            result = {
                "status": "live",
                "asset": asset_key,
                "trade_style": style_key,
                "bias": bias,
                "avg_sentiment": avg_sentiment,
                "headline_count": len(rows),
                "high_impact_count": high_impact,
                "window_hours": preset["hours"],
                "headlines": rows,
                "summary_text": summary,
            }
            log.info(
                "marketaux_ok asset=%s style=%s articles=%s bias=%s sentiment=%+.2f high_impact=%s",
                asset_key,
                style_key,
                len(rows),
                bias,
                avg_sentiment,
                high_impact,
            )
            return _put_cached_context(asset_key, style_key, result)

        except Exception as exc:
            log.warning(
                "marketaux_failed_using_ai_fallback asset=%s style=%s reason=%s",
                asset_key,
                style_key,
                exc,
            )
            # Gracefully fall back to AI-generated context
            return _put_cached_context(
                asset_key, style_key, _ai_news_fallback(asset_key, style_key, preset)
            )
    finally:
        _release_news_fetch_lock(lock_fd)


def attach_news_context(
    payload: Optional[Dict[str, Any]],
    asset: str,
    trade_style: str,
) -> Optional[Dict[str, Any]]:
    if payload is None:
        log.warning(
            "attach_news_context_empty_payload asset=%s style=%s", asset, trade_style
        )
        return None

    out = deepcopy(payload)
    raw_context = build_news_context(asset=asset, trade_style=trade_style)
    context = _effective_trading_context(raw_context)
    out["news_context"] = context
    out["summary_text"] = (
        f"{out.get('summary_text', '').rstrip()}\n{context['summary_text']}".strip()
    )
    out.setdefault("meta", {})["news_status"] = context.get("status")
    out["meta"]["news_confidence_mode"] = context.get("confidence_mode")
    out["meta"]["news_bias"] = context.get("bias")
    out["meta"]["news_avg_sentiment"] = context.get("avg_sentiment")
    out["meta"]["news_high_impact_count"] = context.get("high_impact_count")
    out["meta"]["news_raw_status"] = raw_context.get("status")
    out["meta"]["news_raw_bias"] = raw_context.get("bias")
    out["meta"]["news_raw_avg_sentiment"] = raw_context.get("avg_sentiment")
    out["meta"]["news_raw_high_impact_count"] = raw_context.get("high_impact_count")
    return out
