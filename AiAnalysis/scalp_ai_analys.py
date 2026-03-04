from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
import ssl
from typing import Any, Dict, Optional, Tuple

import logging
import demjson3
from logging.handlers import RotatingFileHandler
from log_config import get_log_path

# =============================================================================
# Logging (ERROR-only)
# =============================================================================
log = logging.getLogger("ai.analysis")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = RotatingFileHandler(
        str(get_log_path("ai_analysis.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)
    log.addHandler(fh)

# =============================================================================
# Runtime knobs (scalping-safe defaults)
# =============================================================================
AI_TIMEOUT_SEC   = 14
AI_MAX_RETRIES   = 1
AI_MIN_CONF      = 0.75   # hard floor: confidence < 75% → HOLD

AI_BUDGET_SEC    = 10
AI_CACHE_TTL_SEC = 120    # M1 bar → 2 min cache

# =============================================================================
# Models — top free-tier, ranked by reasoning quality
# =============================================================================

# Gemini (Google AI Studio — free tier, 1 500 req/day)
GEMINI_MODEL_PRIMARY  = "gemini-2.5-flash"           # best free reasoning model
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"            # fast, reliable fallback

# Groq (free tier — ultra-low latency)
GROQ_MODEL_PRIMARY    = "deepseek-r1-distill-llama-70b"          # strong CoT reasoning
GROQ_MODEL_FALLBACK   = "meta-llama/llama-4-scout-17b-16e-instruct"  # fast & accurate

# Cerebras (free tier — fastest inference ~2 000 tok/s)
CEREBRAS_MODEL_PRIMARY  = "llama-3.3-70b"
CEREBRAS_MODEL_FALLBACK = "llama-3.1-8b"

# =============================================================================
# JSON schema (unchanged — keep downstream compatibility)
# =============================================================================
SIGNAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "signal":       {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "entry":        {"type": ["number", "null"]},
        "stop_loss":    {"type": ["number", "null"]},
        "take_profit":  {"type": ["number", "null"]},
        "reason":       {"type": "string", "minLength": 1},
        "action_short": {"type": "string", "enum": ["Харид", "Фурӯш", "Интизор"]},
    },
    "required": ["signal", "confidence", "entry", "stop_loss", "take_profit", "reason", "action_short"],
    "additionalProperties": False,
}

_CACHE: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}

# =============================================================================
# Helpers
# =============================================================================

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
    # Strip DeepSeek-R1 <think>…</think> reasoning block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    except Exception as err:
        log.error("_parse_json_obj json.loads failed: %s", err)
        try:
            obj = demjson3.decode(cleaned, strict=False)
            return obj if isinstance(obj, dict) else None
        except Exception as err2:
            log.error("_parse_json_obj demjson3 failed: %s | text=%s", err2, cleaned[:200])
            return None


def _normalize_signal(obj: Dict[str, Any], close: Optional[float]) -> Dict[str, Any]:
    signal = str(obj.get("signal", "HOLD")).upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"

    conf   = float(_clamp(_safe_float(obj.get("confidence")) or 0.0, 0.0, 1.0))
    reason = str(obj.get("reason") or "—").strip() or "—"

    action = str(obj.get("action_short") or "").strip()
    if action not in ("Харид", "Фурӯш", "Интизор"):
        action = "Харид" if signal == "BUY" else ("Фурӯш" if signal == "SELL" else "Интизор")

    entry = _safe_float(obj.get("entry"))
    sl    = _safe_float(obj.get("stop_loss"))
    tp    = _safe_float(obj.get("take_profit"))

    return {
        "signal":       signal,
        "confidence":   round(conf, 2),
        "entry":        float(entry) if entry is not None else (float(close) if close else None),
        "stop_loss":    float(sl) if sl is not None else None,
        "take_profit":  float(tp) if tp is not None else None,
        "reason":       reason,
        "action_short": action,
    }


def _pick_atr(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("M5", "M15", "M1"):
        v = _safe_float((market_data.get(tf) or {}).get("atr_14"))
        if v and v > 0:
            return v
    return None


def _pick_close(market_data: Dict[str, Any]) -> Optional[float]:
    for tf in ("M1", "M5", "M15"):
        v = _safe_float((market_data.get(tf) or {}).get("last_close"))
        if v and v > 0:
            return v
    return None


def _pick_m1_close(market_data: Dict[str, Any]) -> Optional[float]:
    v = _safe_float((market_data.get("M1") or {}).get("last_close"))
    return v if v and v > 0 else None


def _spread_penalty(meta: Dict[str, Any], atr: Optional[float]) -> Tuple[float, str]:
    spread_pts = _safe_float(meta.get("spread_points"))
    point      = _safe_float(meta.get("point")) or 0.0
    if not spread_pts or spread_pts <= 0 or point <= 0 or not atr or atr <= 0:
        return 0.0, "spread_penalty=0"
    spread_price = float(spread_pts) * float(point)
    if spread_price > 0.25 * float(atr):
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
    point     = float(symbol_meta.get("point") or 0.0)
    digits    = int(symbol_meta.get("digits") or 2)
    stops_pts = int(symbol_meta.get("stops_level_points") or 0)

    if point <= 0:
        point = 10 ** (-digits) if digits > 0 else 0.01

    min_dist = float(stops_pts) * point
    if atr and atr > 0:
        min_dist = max(min_dist, atr * 0.60)

    def rnd(x: float) -> float:
        return float(round(x, digits)) if digits >= 0 else float(x)

    if atr and atr > 0:
        base_sl = atr * 1.20
        base_tp = atr * 1.00
    else:
        base_sl = max(min_dist * 2.0, point * 50.0)
        base_tp = max(min_dist * 2.0, point * 50.0)

    if signal == "BUY":
        sl_ok = (sl is not None) and (sl < entry - min_dist)
        tp_ok = (tp is not None) and (tp > entry + min_dist)
        sl2   = sl if sl_ok else (entry - max(base_sl, min_dist))
        tp2   = tp if tp_ok else (entry + max(base_tp, min_dist))
    elif signal == "SELL":
        sl_ok = (sl is not None) and (sl > entry + min_dist)
        tp_ok = (tp is not None) and (tp < entry - min_dist)
        sl2   = sl if sl_ok else (entry + max(base_sl, min_dist))
        tp2   = tp if tp_ok else (entry - max(base_tp, min_dist))
    else:
        sl2 = sl if sl is not None else entry
        tp2 = tp if tp is not None else entry

    return rnd(entry), rnd(sl2), rnd(tp2)


# =============================================================================
# Prompt builder — institutional-grade, chain-of-thought enforced
# =============================================================================

def _build_prompt(asset: str, market_data: Dict[str, Any]) -> str:
    summary  = (market_data.get("summary_text") or "").strip()
    meta     = market_data.get("meta") or {}
    atr      = _pick_atr(market_data)
    pen, pen_dbg = _spread_penalty(meta, atr)
    min_conf = float(AI_MIN_CONF)

    return f"""You are a senior quantitative scalping analyst at an institutional trading desk.
Your role: produce a single deterministic trade signal for {asset} based ONLY on the data below.
No assumptions, no external knowledge. Every field is mandatory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET DATA SNAPSHOT  (authoritative)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{summary}

SYMBOL META:
  POINT={meta.get("point")} | DIGITS={meta.get("digits")} | {pen_dbg}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC  (apply in order, sum → confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start at 0.00.

[A] +0.35  M15 EMA stack fully aligned:
    BUY  → close > EMA20 > EMA50 > EMA200
    SELL → close < EMA20 < EMA50 < EMA200

[B] +0.20  M5 price confirms direction:
    BUY  → M5 close > M5 EMA20
    SELL → M5 close < M5 EMA20

[C] +0.15  M5 price on correct side of VWAP:
    BUY  → M5 close > VWAP
    SELL → M5 close < VWAP

[D] +0.10  M1 momentum not adversarial:
    BUY  → M1 RSI14 ≥ 45
    SELL → M1 RSI14 ≤ 55

[E] +0.10  M5 RSI not overbought/oversold:
    BUY  → M5 RSI14 ≤ 55  (not chasing)
    SELL → M5 RSI14 ≥ 45  (not chasing)

[F] −0.20  Data stale: AGE_SEC > 90
[G] −{pen:.2f}  Spread cost > 25% of ATR (execution risk)

→ Clamp final score to [0.00, 1.00].
→ If score < {min_conf:.2f}: signal MUST be "HOLD" (rule: never trade uncertain setups).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVEL CALCULATION  (scalping)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  entry     = M1 last_close  (exact price — do NOT round freely)
  ATR basis = M5 ATR14

  BUY  → stop_loss  = entry − 1.2 × ATR
          take_profit = entry + 1.0 × ATR

  SELL → stop_loss  = entry + 1.2 × ATR
          take_profit = entry − 1.0 × ATR

  HOLD → entry = M1 last_close, stop_loss = null, take_profit = null

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Return ONLY a valid JSON object. Zero prose, zero markdown.
• action_short MUST be exactly one of: "Харид" | "Фурӯш" | "Интизор"
• reason MUST be written in Tajik (Тоҷикӣ), ≥ 20 chars, explain which criteria passed/failed.
• All numeric fields: raw float (no strings).
""".strip()


# =============================================================================
# HTTP helper
# =============================================================================

def _http_post_json(
    url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx  = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=int(timeout), context=ctx) as resp:
            return int(getattr(resp, "status", 200)), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            txt = e.read().decode("utf-8", errors="replace")
        except Exception:
            txt = ""
        return int(getattr(e, "code", 0) or 0), txt
    except Exception as e:
        log.error("_http_post_json crash: %s", e)
        return 0, ""


# =============================================================================
# Provider: Gemini (Google AI Studio)
# =============================================================================

def _call_gemini_structured(
    asset: str, market_data: Dict[str, Any], model: str, timeout: int
) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GEMINI_AI_API_KEY") or "").strip()
    if not api_key:
        return None

    model_id = model.split("/")[-1].strip()
    url      = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    prompt   = _build_prompt(asset, market_data)

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":       0.0,
            "maxOutputTokens":   800,
            "responseMimeType":  "application/json",
            "responseJsonSchema": SIGNAL_SCHEMA,
        },
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(AI_MAX_RETRIES + 1):
        status, raw = _http_post_json(url, body, headers, timeout)
        if status in (429, 503):
            time.sleep(1.5 + attempt * 1.5)
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
        text  = next((str(p["text"]) for p in parts if isinstance(p, dict) and "text" in p), "")

        obj = _parse_json_obj(text)
        if not obj:
            return None
        return _normalize_signal(obj, _pick_close(market_data))

    return None


# =============================================================================
# Provider: Groq  (OpenAI-compatible)
# =============================================================================

def _call_groq_structured(
    asset: str, market_data: Dict[str, Any], model: str, timeout: int
) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GROQ_AI_API_KEY") or "").strip()
    if not api_key:
        return None

    url    = "https://api.groq.com/openai/v1/chat/completions"
    prompt = _build_prompt(asset, market_data)

    body = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens":  800,
        "response_format": {
            "type":        "json_schema",
            "json_schema": {
                "name":   "trade_signal_scalp",
                "strict": True,
                "schema": SIGNAL_SCHEMA,
            },
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(AI_MAX_RETRIES + 1):
        status, raw = _http_post_json(url, body, headers, timeout)
        if status in (429, 503):
            time.sleep(1.5 + attempt * 1.5)
            continue
        if status != 200 or not raw:
            return None

        resp    = _parse_json_obj(raw)
        choices = (resp or {}).get("choices") or []
        if not choices:
            return None
        text = str(((choices[0] or {}).get("message") or {}).get("content") or "")

        obj = _parse_json_obj(text)
        if not obj:
            return None
        return _normalize_signal(obj, _pick_close(market_data))

    return None


# =============================================================================
# Provider: Cerebras  (free tier — ~2 000 tok/s, OpenAI-compatible)
# =============================================================================

def _call_cerebras_structured(
    asset: str, market_data: Dict[str, Any], model: str, timeout: int
) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("CEREBRAS_AI_API_KEY") or "").strip()
    if not api_key:
        return None

    url    = "https://api.cerebras.ai/v1/chat/completions"
    prompt = _build_prompt(asset, market_data)

    body = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens":  800,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(AI_MAX_RETRIES + 1):
        status, raw = _http_post_json(url, body, headers, timeout)
        if status in (429, 503):
            time.sleep(1.5 + attempt * 1.5)
            continue
        if status != 200 or not raw:
            return None

        resp    = _parse_json_obj(raw)
        choices = (resp or {}).get("choices") or []
        if not choices:
            return None
        text = str(((choices[0] or {}).get("message") or {}).get("content") or "")

        obj = _parse_json_obj(text)
        if not obj:
            return None
        return _normalize_signal(obj, _pick_close(market_data))

    return None


# =============================================================================
# Heuristic fallback (local, deterministic)
# =============================================================================

def _heuristic_signal(market_data: Dict[str, Any], asset: str) -> Dict[str, Any]:
    m1   = market_data.get("M1")  or {}
    m5   = market_data.get("M5")  or {}
    m15  = market_data.get("M15") or {}
    meta = market_data.get("meta") or {}

    m1_close  = _safe_float(m1.get("last_close"))  or 0.0
    m5_close  = _safe_float(m5.get("last_close"))  or 0.0
    m15_close = _safe_float(m15.get("last_close")) or 0.0
    entry_ref = m1_close or m5_close or m15_close or 0.0

    if entry_ref <= 0:
        return {
            "signal": "HOLD", "confidence": 0.0,
            "entry": None, "stop_loss": None, "take_profit": None,
            "reason": "Маълумоти бозор нест", "action_short": "Интизор",
        }

    rsi1  = _safe_float(m1.get("rsi_14"))  or 50.0
    rsi5  = _safe_float(m5.get("rsi_14"))  or 50.0

    ema20_15  = _safe_float(m15.get("ema_20"))  or m15_close or entry_ref
    ema50_15  = _safe_float(m15.get("ema_50"))  or m15_close or entry_ref
    ema200_15 = _safe_float(m15.get("ema_200")) or m15_close or entry_ref
    ema20_5   = _safe_float(m5.get("ema_20"))   or m5_close  or entry_ref
    vwap5     = _safe_float(m5.get("vwap"))     or m5_close  or entry_ref

    age = int(meta.get("age_sec") or 0)
    atr = _pick_atr(market_data)
    pen, _ = _spread_penalty(meta, atr)

    def score(direction: str) -> float:
        s = 0.0
        if direction == "BUY":
            if m15_close > 0 and m15_close > ema20_15 > ema50_15 > ema200_15: s += 0.35
            if m5_close  > 0 and m5_close  > ema20_5:                         s += 0.20
            if m5_close  > 0 and m5_close  > vwap5:                           s += 0.15
            if m1_close  > 0 and rsi1 >= 45:                                  s += 0.10
            if rsi5 <= 55:                                                     s += 0.10
        else:
            if m15_close > 0 and m15_close < ema20_15 < ema50_15 < ema200_15: s += 0.35
            if m5_close  > 0 and m5_close  < ema20_5:                         s += 0.20
            if m5_close  > 0 and m5_close  < vwap5:                           s += 0.15
            if m1_close  > 0 and rsi1 <= 55:                                  s += 0.10
            if rsi5 >= 45:                                                     s += 0.10
        if age > 90:
            s -= 0.20
        s -= pen
        return float(_clamp(s, 0.0, 1.0))

    sb = score("BUY")
    ss = score("SELL")

    if sb >= ss and sb >= AI_MIN_CONF:
        return {
            "signal": "BUY", "confidence": round(sb, 2), "entry": float(entry_ref),
            "stop_loss": None, "take_profit": None,
            "reason": "Скалп: M15 EMA stack боло + M5 тасдиқ + VWAP",
            "action_short": "Харид",
        }
    if ss > sb and ss >= AI_MIN_CONF:
        return {
            "signal": "SELL", "confidence": round(ss, 2), "entry": float(entry_ref),
            "stop_loss": None, "take_profit": None,
            "reason": "Скалп: M15 EMA stack поён + M5 тасдиқ + VWAP",
            "action_short": "Фурӯш",
        }

    return {
        "signal": "HOLD",
        "confidence": round(_clamp(max(sb, ss), 0.0, AI_MIN_CONF - 0.01), 2),
        "entry": float(entry_ref), "stop_loss": None, "take_profit": None,
        "reason": "Скалп: шартҳо пурра тасдиқ нашуданд",
        "action_short": "Интизор",
    }


# =============================================================================
# Cache
# =============================================================================

def _cache_cleanup(now_ts: float) -> None:
    cutoff = now_ts - AI_CACHE_TTL_SEC
    for k, (ts_saved, _v) in list(_CACHE.items()):
        if ts_saved < cutoff:
            _CACHE.pop(k, None)
    if len(_CACHE) > 512:
        items = sorted(_CACHE.items(), key=lambda kv: kv[1][0], reverse=True)
        _CACHE.clear()
        _CACHE.update(dict(items[:512]))


# =============================================================================
# Public entry point
# =============================================================================

def analyse(asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scalping inference chain (fastest-first):
      1. Gemini  (primary → fallback)
      2. Groq    (primary → fallback)
      3. Cerebras(primary → fallback)
      ML unavailable → HOLD / AI_NO_DECISION
    + deterministic entry from M1 close
    + ATR-based SL/TP clamping
    + time budget + per-bar cache
    """
    if not market_data or not (market_data.get("summary_text") or "").strip():
        return {
            "signal": "HOLD", "confidence": 0.0,
            "entry": None, "stop_loss": None, "take_profit": None,
            "reason": "Payload нест", "action_short": "Интизор",
            "provider": "none", "model": "none",
        }

    meta  = market_data.get("meta") or {}
    age   = int(meta.get("age_sec") or 0)
    close = _pick_close(market_data) or 0.0
    atr   = _pick_atr(market_data)

    if age > 180:
        return {
            "signal": "HOLD", "confidence": 0.0,
            "entry": float(close) if close > 0 else None,
            "stop_loss": None, "take_profit": None,
            "reason": f"Маълумот кӯҳна аст (AGE_SEC={age})",
            "action_short": "Интизор",
            "provider": "precheck", "model": "stale_guard",
        }

    ts_bar    = int(((market_data.get("M1") or {}).get("ts_bar") or 0) or 0)
    cache_key = (asset, ts_bar)
    now       = time.time()

    if ts_bar > 0 and cache_key in _CACHE:
        ts_saved, res_saved = _CACHE[cache_key]
        if (now - ts_saved) <= AI_CACHE_TTL_SEC:
            return dict(res_saved)

    deadline = now + AI_BUDGET_SEC

    def remaining_timeout() -> int:
        return int(max(1.0, min(float(AI_TIMEOUT_SEC), deadline - time.time())))

    res: Optional[Dict[str, Any]] = None

    # ── 1. Gemini ─────────────────────────────────────────────────────────────
    for mdl in (GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK):
        if time.time() >= deadline:
            break
        res = _call_gemini_structured(asset, market_data, mdl, remaining_timeout())
        if res:
            res["provider"] = "gemini"
            res["model"]    = mdl
            break

    # ── 2. Groq ───────────────────────────────────────────────────────────────
    if res is None:
        for mdl in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
            if time.time() >= deadline:
                break
            res = _call_groq_structured(asset, market_data, mdl, remaining_timeout())
            if res:
                res["provider"] = "groq"
                res["model"]    = mdl
                break

    # ── 3. Cerebras ───────────────────────────────────────────────────────────
    if res is None:
        for mdl in (CEREBRAS_MODEL_PRIMARY, CEREBRAS_MODEL_FALLBACK):
            if time.time() >= deadline:
                break
            res = _call_cerebras_structured(asset, market_data, mdl, remaining_timeout())
            if res:
                res["provider"] = "cerebras"
                res["model"]    = mdl
                break

    # ── 4. ML unavailable ─────────────────────────────────────────────────────
    if res is None:
        res = {
            "signal": "HOLD", "confidence": 0.0,
            "entry": float(close) if close > 0 else None,
            "stop_loss": None, "take_profit": None,
            "reason": "ML inference unavailable",
            "action_short": "Интизор",
            "provider": "none", "model": "none",
            "status": "ML_UNAVAILABLE",
        }

    # ── Deterministic entry + level clamping ──────────────────────────────────
    signal   = str(res.get("signal") or "HOLD").upper()
    conf     = float(res.get("confidence") or 0.0)
    m1_close = _pick_m1_close(market_data) or 0.0

    entry = (
        float(m1_close) if (signal in ("BUY", "SELL") and m1_close > 0)
        else (_safe_float(res.get("entry")) or (float(close) if close > 0 else 0.0))
    )
    sl = _safe_float(res.get("stop_loss"))
    tp = _safe_float(res.get("take_profit"))

    if entry > 0 and signal in ("BUY", "SELL"):
        entry2, sl2, tp2    = _levels_clamp(meta, signal, entry, sl, tp, atr)
        res["entry"]        = entry2
        res["stop_loss"]    = sl2
        res["take_profit"]  = tp2
    else:
        res["entry"]       = float(m1_close) if m1_close > 0 else (float(close) if close > 0 else None)
        res["stop_loss"]   = None
        res["take_profit"] = None

    # ── Hard gate: non-ML provider ─────────────────────────────────────────────
    provider = str(res.get("provider") or "none").strip().lower()
    if provider not in ("gemini", "groq", "cerebras"):
        res["signal"]       = "HOLD"
        res["status"]       = "AI_NO_DECISION"
        res["action_short"] = "Интизор"
        res["stop_loss"]    = None
        res["take_profit"]  = None
        res["reason"]       = f"AI_NO_DECISION: non_ml_provider={provider}"

    # ── Hard gate: low confidence ──────────────────────────────────────────────
    if conf < AI_MIN_CONF:
        res["signal"]       = "HOLD"
        res["status"]       = "AI_NO_DECISION"
        res["action_short"] = "Интизор"
        res["stop_loss"]    = None
        res["take_profit"]  = None
        res["reason"]       = (
            f"⛔ AI_NO_DECISION: conf={conf:.2f} < min={AI_MIN_CONF} | {res.get('reason', '—')}"
        )

    # ── Cache ──────────────────────────────────────────────────────────────────
    if ts_bar > 0:
        _CACHE[cache_key] = (time.time(), dict(res))
        _cache_cleanup(time.time())

    return res
