from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

import logging
import demjson3
from logging.handlers import RotatingFileHandler
from log_config import get_log_path

# =============================================================================
# Logging (ERROR-only, separate)
# =============================================================================
log = logging.getLogger("ai.intraday.analysis")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        str(get_log_path("ai_intraday_analysis.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)
    log.addHandler(fh)

# =============================================================================
# Intraday runtime knobs (separate env names)
# =============================================================================
AI_INTRA_TIMEOUT_SEC = 12
AI_INTRA_MAX_RETRIES = 1
AI_INTRA_MIN_CONF = 0.82

AI_INTRA_BUDGET_SEC = 8.0
AI_INTRA_CACHE_TTL_SEC = 3600  # H1 => cache up to 1h (safe)

GEMINI_INTRA_MODEL_PRIMARY = "gemini-3.0-flash"
GEMINI_INTRA_MODEL_FALLBACK = "gemini-2.5-flash-lite"

GROQ_INTRA_MODEL_PRIMARY = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_INTRA_MODEL_FALLBACK = "llama-3.1-8b-instant"

SIGNAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "entry": {"type": ["number", "null"]},
        "stop_loss": {"type": ["number", "null"]},
        "take_profit": {"type": ["number", "null"]},
        "reason": {"type": "string", "minLength": 1},
        "action_short": {"type": "string", "enum": ["Харид", "Фурӯш", "Интизор"]},
    },
    "required": ["signal", "confidence", "entry", "stop_loss", "take_profit", "reason", "action_short"],
    "additionalProperties": False,
}

_CACHE: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _clean_json(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return t[s : e + 1]
    return t


def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _clean_json(text)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        log.error("_parse_json_obj json.loads failed: %s", e)
        try:
            obj = demjson3.decode(cleaned, strict=False)
            return obj if isinstance(obj, dict) else None
        except Exception as e2:
            log.error("_parse_json_obj demjson3 failed: %s | text=%s", e2, cleaned[:200])
            return None


def _pick_close(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("H1", "H4", "D1"):
        v = _safe_float((market_data.get(tf) or {}).get("last_close"))
        if v and v > 0:
            return v
    return None


def _pick_atr(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("H1", "H4", "D1"):
        v = _safe_float((market_data.get(tf) or {}).get("atr_14"))
        if v and v > 0:
            return v
    return None


def _normalize_signal(obj: Dict[str, Any], close: Optional[float]) -> Dict[str, Any]:
    signal = str(obj.get("signal", "HOLD")).upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"

    conf = float(_clamp(_safe_float(obj.get("confidence")) or 0.0, 0.0, 1.0))
    reason = str(obj.get("reason") or "—").strip() or "—"

    action = str(obj.get("action_short") or "").strip()
    if action not in ("Харид", "Фурӯш", "Интизор"):
        action = "Харид" if signal == "BUY" else ("Фурӯш" if signal == "SELL" else "Интизор")

    entry = _safe_float(obj.get("entry"))
    sl = _safe_float(obj.get("stop_loss"))
    tp = _safe_float(obj.get("take_profit"))

    return {
        "signal": signal,
        "confidence": round(conf, 2),
        "entry": float(entry) if entry is not None else (float(close) if close else None),
        "stop_loss": float(sl) if sl is not None else None,
        "take_profit": float(tp) if tp is not None else None,
        "reason": reason,
        "action_short": action,
    }


def _spread_penalty(meta: Dict[str, Any], atr: Optional[float]) -> Tuple[float, str]:
    spread_pts = _safe_float(meta.get("spread_points"))
    point = _safe_float(meta.get("point")) or 0.0
    if not spread_pts or spread_pts <= 0 or point <= 0 or not atr or atr <= 0:
        return 0.0, "spread_penalty=0"

    spread_price = float(spread_pts) * float(point)
    # Intraday: threshold slightly tighter than swing, looser than scalping
    if spread_price > 0.20 * float(atr):
        return 0.20, f"spread_penalty=0.20 spread_price={spread_price:.6f} atr={atr:.6f}"
    return 0.0, f"spread_penalty=0 spread_price={spread_price:.6f} atr={atr:.6f}"


def _levels_clamp(
    symbol_meta: Dict[str, Any],
    signal: str,
    entry: float,
    sl: Optional[float],
    tp: Optional[float],
    atr: Optional[float],
) -> Tuple[float, float, float]:
    point = float(symbol_meta.get("point") or 0.0)
    digits = int(symbol_meta.get("digits") or 2)
    stops_pts = int(symbol_meta.get("stops_level_points") or 0)

    if point <= 0:
        point = 10 ** (-digits) if digits > 0 else 0.01

    min_dist = float(stops_pts) * point
    if atr and atr > 0:
        min_dist = max(min_dist, atr * 0.50)

    def rnd(x: float) -> float:
        return float(round(x, digits)) if digits >= 0 else float(x)

    # Intraday: SL wider than scalping
    if atr and atr > 0:
        base_sl = atr * 1.50
        base_tp = atr * 1.20
    else:
        base_sl = max(min_dist * 2.0, point * 120.0)
        base_tp = max(min_dist * 2.0, point * 120.0)

    if signal == "BUY":
        sl_ok = (sl is not None) and (sl < entry - min_dist)
        tp_ok = (tp is not None) and (tp > entry + min_dist)
        sl2 = sl if sl_ok else (entry - max(base_sl, min_dist))
        tp2 = tp if tp_ok else (entry + max(base_tp, min_dist))
    elif signal == "SELL":
        sl_ok = (sl is not None) and (sl > entry + min_dist)
        tp_ok = (tp is not None) and (tp < entry - min_dist)
        sl2 = sl if sl_ok else (entry + max(base_sl, min_dist))
        tp2 = tp if tp_ok else (entry - max(base_tp, min_dist))
    else:
        sl2 = sl if sl is not None else entry
        tp2 = tp if tp is not None else entry

    return rnd(entry), rnd(sl2), rnd(tp2)


def _build_prompt(asset: str, market_data: Dict[str, Any]) -> str:
    summary = (market_data.get("summary_text") or "").strip()
    meta = market_data.get("meta") or {}
    atr = _pick_atr(market_data)
    pen, pen_dbg = _spread_penalty(meta, atr)

    # PATCH: keep prompt threshold consistent with runtime hard gate
    min_conf = float(AI_INTRA_MIN_CONF)

    return f"""
You are a deterministic intraday trading analyst for {asset}. Use ONLY the provided data. Output MUST follow the JSON schema.

MARKET SUMMARY (authoritative):
{summary}

META (authoritative):
POINT={meta.get("point")} | DIGITS={meta.get("digits")} | {pen_dbg}

SCORING (intraday, strict):
- Start score = 0.00
- +0.40 if D1 trend is aligned:
  BUY: close>EMA20>EMA50>EMA200, SELL: close<EMA20<EMA50<EMA200
- +0.25 if H4 confirms direction:
  BUY close>EMA20 and EMA20>EMA50, SELL close<EMA20 and EMA20<EMA50
- +0.10 if H1 is not strongly opposite:
  BUY RSI>=45, SELL RSI<=55
- +0.15 if H1 close is on correct side of VWAP:
  BUY close>VWAP, SELL close<VWAP
- +0.10 if H1 RSI supports:
  BUY RSI<=60, SELL RSI>=40
- -0.20 if AGE_SEC > 5400 (data older than ~1.5h)
- -{pen:.2f} if spread too high relative to ATR (computed above)

Clamp confidence to [0,1]. If confidence < {min_conf:.2f} => signal MUST be HOLD.

LEVELS (intraday):
- entry = current close from H1 as number
- Prefer ATR14 from H1:
  BUY: SL = entry - 1.5*ATR, TP = entry + 1.2*ATR
  SELL: SL = entry + 1.5*ATR, TP = entry - 1.2*ATR

IMPORTANT:
- Output ONLY JSON object. No extra text.
- action_short must be exactly: "Харид" / "Фурӯш" / "Интизор".
- reason MUST be in Tajik language (Тоҷикӣ).
""".strip()


def _http_post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            status = int(getattr(resp, "status", 200))
            text = resp.read().decode("utf-8", errors="replace")
            return status, text
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode("utf-8", errors="replace")
        except Exception:
            txt = ""
        return int(getattr(e, "code", 0) or 0), txt
    except Exception as e:
        log.error("_http_post_json crash: %s", e)
        return 0, ""


def _call_gemini_structured(asset: str, market_data: Dict[str, Any], model: str, timeout: int) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GEMINI_AI_API_KEY") or "").strip()
    if not api_key:
        return None

    model_id = model.split("/")[-1].strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    prompt = _build_prompt(asset, market_data)

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 700,
            "responseMimeType": "application/json",
            "responseJsonSchema": SIGNAL_SCHEMA,
        },
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(AI_INTRA_MAX_RETRIES + 1):
        status, raw = _http_post_json(url, body, headers, timeout)
        if status in (429, 503):
            time.sleep(1.0 + attempt * 1.0)
            continue
        if status != 200 or not raw:
            return None

        resp = _parse_json_obj(raw)
        if not resp:
            return None

        candidates = resp.get("candidates") or []
        if not candidates:
            return None
        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = ""
        for p in parts:
            if isinstance(p, dict) and "text" in p:
                text = str(p["text"])
                break

        obj = _parse_json_obj(text)
        if not obj:
            return None

        return _normalize_signal(obj, _pick_close(market_data))

    return None


def _call_groq_structured(asset: str, market_data: Dict[str, Any], model: str, timeout: int) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GROQ_AI_API_KEY") or "").strip()
    if not api_key:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = _build_prompt(asset, market_data)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 700,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "trade_signal_intraday",
                "strict": True,
                "schema": SIGNAL_SCHEMA,
            },
        },
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(AI_INTRA_MAX_RETRIES + 1):
        status, raw = _http_post_json(url, body, headers, timeout)
        if status in (429, 503):
            time.sleep(1.0 + attempt * 1.0)
            continue
        if status != 200 or not raw:
            return None

        resp = _parse_json_obj(raw)
        if not resp:
            return None

        choices = resp.get("choices") or []
        if not choices:
            return None
        msg = (choices[0] or {}).get("message") or {}
        text = str(msg.get("content") or "")

        obj = _parse_json_obj(text)
        if not obj:
            return None

        return _normalize_signal(obj, _pick_close(market_data))

    return None


def _heuristic_signal(market_data: Dict[str, Any], asset: str) -> Dict[str, Any]:
    d1 = market_data.get("D1") or {}
    h4 = market_data.get("H4") or {}
    h1 = market_data.get("H1") or {}
    meta = market_data.get("meta") or {}

    close = _pick_close(market_data) or 0.0
    if close <= 0:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Маълумоти бозор нест",
            "action_short": "Интизор",
        }

    age = int(meta.get("age_sec") or 0)
    atr = _pick_atr(market_data)
    pen, _ = _spread_penalty(meta, atr)

    d1_close = _safe_float(d1.get("last_close")) or close
    d1_e20 = _safe_float(d1.get("ema_20")) or d1_close
    d1_e50 = _safe_float(d1.get("ema_50")) or d1_close
    d1_e200 = _safe_float(d1.get("ema_200")) or d1_close

    h4_close = _safe_float(h4.get("last_close")) or close
    h4_e20 = _safe_float(h4.get("ema_20")) or h4_close
    h4_e50 = _safe_float(h4.get("ema_50")) or h4_close

    h1_close = _safe_float(h1.get("last_close")) or close
    h1_rsi = _safe_float(h1.get("rsi_14")) or 50.0
    h1_vwap = _safe_float(h1.get("vwap")) or h1_close

    def score(direction: str) -> float:
        s = 0.0
        if direction == "BUY":
            if d1_close > d1_e20 > d1_e50 > d1_e200:
                s += 0.40
            if h4_close > h4_e20 and h4_e20 > h4_e50:
                s += 0.25
            if h1_rsi >= 45:
                s += 0.10
            if h1_close > h1_vwap:
                s += 0.15
            if h1_rsi <= 60:
                s += 0.10
        else:
            if d1_close < d1_e20 < d1_e50 < d1_e200:
                s += 0.40
            if h4_close < h4_e20 and h4_e20 < h4_e50:
                s += 0.25
            if h1_rsi <= 55:
                s += 0.10
            if h1_close < h1_vwap:
                s += 0.15
            if h1_rsi >= 40:
                s += 0.10

        if age > 5400:
            s -= 0.20
        s -= pen
        return float(_clamp(s, 0.0, 1.0))

    sb = score("BUY")
    ss = score("SELL")

    if sb >= ss and sb >= 0.70:
        return {
            "signal": "BUY",
            "confidence": round(sb, 2),
            "entry": float(h1_close),
            "stop_loss": None,
            "take_profit": None,
            "reason": "Рӯзона: D1 тамоюл боло, H4 тасдиқ, H1 болотар аз VWAP",
            "action_short": "Харид",
        }
    if ss > sb and ss >= 0.70:
        return {
            "signal": "SELL",
            "confidence": round(ss, 2),
            "entry": float(h1_close),
            "stop_loss": None,
            "take_profit": None,
            "reason": "Рӯзона: D1 тамоюл поён, H4 тасдиқ, H1 поёнтар аз VWAP",
            "action_short": "Фурӯш",
        }

    return {
        "signal": "HOLD",
        "confidence": round(_clamp(max(sb, ss), 0.0, 0.69), 2),
        "entry": float(h1_close),
        "stop_loss": None,
        "take_profit": None,
        "reason": "Рӯзона: шартҳо пурра нестанд, интизор",
        "action_short": "Интизор",
    }


def _cache_cleanup(now_ts: float) -> None:
    # remove expired
    cutoff = now_ts - AI_INTRA_CACHE_TTL_SEC
    for k, (ts_saved, _v) in list(_CACHE.items()):
        if ts_saved < cutoff:
            _CACHE.pop(k, None)

    # hard cap (safety)
    if len(_CACHE) > 512:
        items = sorted(_CACHE.items(), key=lambda kv: kv[1][0], reverse=True)
        _CACHE.clear()
        _CACHE.update(dict(items[:512]))


def analyse_intraday(asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intraday chain:
    Gemini(primary->fallback) -> Groq(primary->fallback) -> heuristic
    + clamp SL/TP
    + budget + cache (by H1 bar)
    """
    if not market_data or not (market_data.get("summary_text") or "").strip():
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "Payload нест",
            "action_short": "Интизор",
            "provider": "none",
            "model": "none",
        }

    meta = market_data.get("meta") or {}
    age = int(meta.get("age_sec") or 0)
    close = _pick_close(market_data) or 0.0
    atr = _pick_atr(market_data)

    # Intraday stale guard (H1 base)
    if age > 21600:  # >6h
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "entry": float(close) if close > 0 else None,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"Маълумот кӯҳна аст (AGE_SEC={age})",
            "action_short": "Интизор",
            "provider": "precheck",
            "model": "stale_guard",
        }

    ts_bar_h1 = int(((market_data.get("H1") or {}).get("ts_bar") or 0) or 0)
    cache_key = (asset, ts_bar_h1)
    now = time.time()

    if ts_bar_h1 > 0 and cache_key in _CACHE:
        ts_saved, res_saved = _CACHE[cache_key]
        if (now - ts_saved) <= AI_INTRA_CACHE_TTL_SEC:
            return dict(res_saved)

    deadline = now + AI_INTRA_BUDGET_SEC

    def remaining_timeout() -> int:
        return int(max(1.0, min(float(AI_INTRA_TIMEOUT_SEC), deadline - time.time())))

    res: Optional[Dict[str, Any]] = None

    # 1) Gemini
    for mdl in (GEMINI_INTRA_MODEL_PRIMARY, GEMINI_INTRA_MODEL_FALLBACK):
        if time.time() >= deadline:
            break
        res = _call_gemini_structured(asset, market_data, mdl, remaining_timeout())
        if res:
            res["provider"] = "gemini"
            res["model"] = mdl
            break

    # 2) Groq
    if res is None:
        for mdl in (GROQ_INTRA_MODEL_PRIMARY, GROQ_INTRA_MODEL_FALLBACK):
            if time.time() >= deadline:
                break
            res = _call_groq_structured(asset, market_data, mdl, remaining_timeout())
            if res:
                res["provider"] = "groq"
                res["model"] = mdl
                break

    # 3) heuristic
    if res is None:
        res = _heuristic_signal(market_data, asset)
        res["provider"] = "heuristic"
        res["model"] = "local"

    # Enforce levels
    signal = str(res.get("signal") or "HOLD").upper()
    conf = float(res.get("confidence") or 0.0)

    # PATCH: force entry from H1 close for BUY/SELL (deterministic execution)
    h1_close = _safe_float(((market_data.get("H1") or {}).get("last_close"))) or close

    if signal in ("BUY", "SELL") and h1_close and h1_close > 0:
        entry = float(h1_close)
    else:
        entry = _safe_float(res.get("entry")) or (float(close) if close > 0 else 0.0)

    sl = _safe_float(res.get("stop_loss"))
    tp = _safe_float(res.get("take_profit"))

    if entry > 0 and signal in ("BUY", "SELL"):
        entry2, sl2, tp2 = _levels_clamp(meta, signal, float(entry), sl, tp, atr)
        res["entry"] = entry2
        res["stop_loss"] = sl2
        res["take_profit"] = tp2
    else:
        res["entry"] = float(h1_close) if h1_close > 0 else (float(close) if close > 0 else None)
        res["stop_loss"] = None
        res["take_profit"] = None

    # Hard gate (PATCH: clear levels when gated)
    if conf < AI_INTRA_MIN_CONF:
        res["signal"] = "HOLD"
        res["action_short"] = "Интизор"
        res["stop_loss"] = None
        res["take_profit"] = None
        res["reason"] = f"{res.get('reason','—')} | gated: conf<{AI_INTRA_MIN_CONF}"

    if ts_bar_h1 > 0:
        _CACHE[cache_key] = (time.time(), dict(res))
        _cache_cleanup(time.time())

    return res
