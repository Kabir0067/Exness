"""
analysis_common.py — Provider invocation layer
Supports: Gemini, Groq, Cerebras, OpenRouter
Ranked chain: TOP-1 → TOP-2 → ... → heuristic fallback
"""
from __future__ import annotations

import ast
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from log_config import build_logger

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


log = build_logger("ai.analysis_common", "analysis_common.log")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

ACTION_BUY  = "Харид"
ACTION_SELL = "Фурӯш"
ACTION_HOLD = "Интизор"

_DISABLED_CANDIDATES: Dict[str, Tuple[float, str]] = {}
_DISABLE_TTL_SEC = 1800.0

SIGNAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "signal":       {"type": "string",  "enum": ["BUY", "SELL", "HOLD"]},
        "confidence":   {"type": "number",  "minimum": 0.0, "maximum": 1.0},
        "entry":        {"type": ["number", "null"]},
        "stop_loss":    {"type": ["number", "null"]},
        "take_profit":  {"type": ["number", "null"]},
        "reason":       {"type": "string",  "minLength": 1},
        "action_short": {"type": "string",  "enum": [ACTION_BUY, ACTION_SELL, ACTION_HOLD]},
    },
    "required": ["signal", "confidence", "entry", "stop_loss", "take_profit", "reason", "action_short"],
    "additionalProperties": False,
}

PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "gemini":     "Gemini",
    "openrouter": "OpenRouter",
    "groq":       "Groq",
    "cerebras":   "Cerebras",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Provider model lists (live-verified 2026-03-11)
#  Ordered by practical production value for each trading style.
# ─────────────────────────────────────────────────────────────────────────────

SCALPING_PROVIDERS: List[Dict[str, Any]] = [
    # Fast path first for scalping. Every model below passed a live API check.
    {"rank": 1, "provider": "groq",     "model": "meta-llama/llama-4-scout-17b-16e-instruct"},
    {"rank": 2, "provider": "groq",     "model": "moonshotai/kimi-k2-instruct-0905"},
    {"rank": 3, "provider": "groq",     "model": "llama-3.3-70b-versatile"},
    {"rank": 4, "provider": "gemini",   "model": "gemini-2.5-flash-lite"},
]

INTRADAY_PROVIDERS: List[Dict[str, Any]] = [
    # Reasoning-heavy path first for intraday. Every model below passed a live API check.
    {"rank": 1, "provider": "groq",     "model": "moonshotai/kimi-k2-instruct-0905"},
    {"rank": 2, "provider": "groq",     "model": "llama-3.3-70b-versatile"},
    {"rank": 3, "provider": "gemini",   "model": "gemini-2.5-flash-lite"},
]


# ─────────────────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def expand_model_candidates(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    next_rank = 1
    for item in items:
        provider_name = str(item.get("provider") or "").strip().lower()
        if not provider_name:
            continue
        if item.get("model"):
            candidate = {
                "rank":     int(item.get("rank") or next_rank),
                "provider": provider_name,
                "model":    str(item.get("model") or ""),
            }
            expanded.append(candidate)
            next_rank = max(next_rank, int(candidate["rank"]) + 1)
            continue
        for model in list(item.get("models") or []):
            expanded.append({"rank": next_rank, "provider": provider_name, "model": str(model)})
            next_rank += 1

    expanded.sort(key=lambda row: (int(row.get("rank") or 9999), str(row.get("provider") or ""), str(row.get("model") or "")))
    return expanded


def _candidate_key(provider_name: str, model: str) -> str:
    return f"{provider_name.strip().lower()}::{model.strip()}"


def _candidate_disabled_reason(provider_name: str, model: str) -> Optional[str]:
    key      = _candidate_key(provider_name, model)
    disabled = _DISABLED_CANDIDATES.get(key)
    if not disabled:
        return None
    disabled_until, reason = disabled
    if time.time() >= disabled_until:
        _DISABLED_CANDIDATES.pop(key, None)
        return None
    return reason


def _mark_candidate_unavailable(provider_name: str, model: str, reason: str) -> None:
    _DISABLED_CANDIDATES[_candidate_key(provider_name, model)] = (time.time() + _DISABLE_TTL_SEC, reason)
    log.warning("candidate_disabled provider=%s model=%s reason=%s", provider_name, model, reason)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def clean_json(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = clean_json(text)
    candidates = [
        cleaned,
        cleaned.replace("\r", " ").replace("\n", " "),
        re.sub(r",\s*([}\]])", r"\1", cleaned.replace("\r", " ").replace("\n", " ")),
    ]
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    log.debug("json_parse_failed text_prefix=%s", clean_json(text)[:400])
    return None


def _strip_unsupported_schema_keys(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_unsupported_schema_keys(v) for k, v in node.items() if k not in {"minLength"}}
    if isinstance(node, list):
        return [_strip_unsupported_schema_keys(item) for item in node]
    return node


def action_short_for_signal(signal: str) -> str:
    normalized = str(signal or "").upper().strip()
    if normalized == "BUY":
        return ACTION_BUY
    if normalized == "SELL":
        return ACTION_SELL
    return ACTION_HOLD


def normalize_signal(result: Dict[str, Any], close: Optional[float]) -> Dict[str, Any]:
    signal     = str(result.get("signal") or "HOLD").upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"
    confidence  = clamp(float(safe_float(result.get("confidence")) or 0.0), 0.0, 1.0)
    entry       = safe_float(result.get("entry"))
    stop_loss   = safe_float(result.get("stop_loss"))
    take_profit = safe_float(result.get("take_profit"))
    reason      = str(result.get("reason") or "AI reasoning unavailable").strip()
    action_short = str(result.get("action_short") or action_short_for_signal(signal)).strip()
    if action_short not in (ACTION_BUY, ACTION_SELL, ACTION_HOLD):
        action_short = action_short_for_signal(signal)
    return {
        "signal":       signal,
        "confidence":   round(confidence, 2),
        "entry":        float(entry)       if entry       is not None else (float(close) if close is not None else None),
        "stop_loss":    float(stop_loss)   if stop_loss   is not None else None,
        "take_profit":  float(take_profit) if take_profit is not None else None,
        "reason":       reason,
        "action_short": action_short,
    }


def http_post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Tuple[int, str]:
    data    = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=int(timeout), context=context) as response:
            return int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = ""
        return int(getattr(exc, "code", 0) or 0), body_text
    except Exception as exc:
        log.exception("http_post_json_failed url=%s", url)
        return 0, str(exc)


def _env_first(*keys: str) -> str:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

def _parse_openai_compatible_response(raw: str) -> Optional[Dict[str, Any]]:
    payload = parse_json_obj(raw)
    if not payload:
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts)
    else:
        text = str(content or "")
    return parse_json_obj(text)


def _invoke_gemini(prompt: str, model: str, timeout: int, max_retries: int, close: Optional[float]) -> Dict[str, Any]:
    api_key = _env_first("GEMINI_AI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        log.warning("gemini_missing_api_key")
        return {"ok": False, "error": "missing_api_key"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":   0.0,
            "maxOutputTokens": 2048,
            "thinkingConfig":  {"thinkingBudget": 0},
            "responseMimeType": "application/json",
            "responseJsonSchema": SIGNAL_SCHEMA,
        },
    }
    headers    = {"Content-Type": "application/json", "User-Agent": "AiBot/1.0"}
    last_error = "retry_exhausted"

    for attempt in range(max_retries + 1):
        status, raw = http_post_json(url, body, headers, timeout)
        if status in (429, 500, 503):
            time.sleep(1.5 + attempt)
            continue
        if status != 200:
            log.debug("gemini_http_error model=%s status=%s", model, status)
            return {"ok": False, "error": f"http_{status}: {clean_json(raw)[:220]}"}
        payload    = parse_json_obj(raw)
        if not payload:
            return {"ok": False, "error": "invalid_json_payload"}
        candidates = payload.get("candidates") or []
        if not candidates:
            return {"ok": False, "error": "empty_candidates"}
        parts_list = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = next((str(item.get("text")) for item in parts_list if isinstance(item, dict) and item.get("text")), "")
        obj  = parse_json_obj(text)
        if not obj:
            last_error = f"invalid_json_message: {clean_json(text)[:220]}"
            log.debug("gemini_invalid_json model=%s", model)
            continue
        return {"ok": True, "result": normalize_signal(obj, close)}

    return {"ok": False, "error": last_error}


def _invoke_openrouter(prompt: str, model: str, timeout: int, max_retries: int, close: Optional[float]) -> Dict[str, Any]:
    api_key = _env_first("OPENROUTER_API_KEY", "OPEN_ROUTER")
    if not api_key:
        log.warning("openrouter_missing_api_key")
        return {"ok": False, "error": "missing_api_key"}

    url  = "https://openrouter.ai/api/v1/chat/completions"
    body = {
        "model":           model,
        "messages":        [{"role": "user", "content": prompt}],
        "temperature":     0.0,
        "max_tokens":      1200,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://localhost",
        "X-Title":       "InstitutionalAiBot",
        "User-Agent":    "AiBot/1.0",
    }
    last_error = "retry_exhausted"
    for attempt in range(max_retries + 1):
        status, raw = http_post_json(url, body, headers, timeout)
        if status in (429, 500, 503):
            time.sleep(1.5 + attempt)
            continue
        if status != 200:
            log.debug("openrouter_http_error model=%s status=%s", model, status)
            return {"ok": False, "error": f"http_{status}: {clean_json(raw)[:220]}"}
        obj = _parse_openai_compatible_response(raw)
        if not obj:
            last_error = "invalid_json_message"
            log.debug("openrouter_invalid_json model=%s", model)
            continue
        return {"ok": True, "result": normalize_signal(obj, close)}

    return {"ok": False, "error": last_error}


def _invoke_groq(
    prompt: str, model: str, timeout: int, max_retries: int,
    close: Optional[float], schema_name: str,
) -> Dict[str, Any]:
    api_key = _env_first("GROQ_AI_API_KEY")
    if not api_key:
        log.warning("groq_missing_api_key")
        return {"ok": False, "error": "missing_api_key"}

    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "User-Agent":    "AiBot/1.0",
    }
    # Try strict JSON schema first, then plain JSON object
    bodies = [
        {
            "model": model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  1200,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": SIGNAL_SCHEMA},
            },
        },
        {
            "model": model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  1200,
            "response_format": {"type": "json_object"},
        },
    ]
    last_error = "retry_exhausted"
    for body in bodies:
        for attempt in range(max_retries + 1):
            status, raw = http_post_json(url, body, headers, timeout)
            if status in (429, 500, 503):
                time.sleep(1.5 + attempt)
                continue
            if status != 200:
                last_error = f"http_{status}: {clean_json(raw)[:220]}"
                log.debug("groq_http_error model=%s status=%s", model, status)
                if status == 400 and "does not support response format `json_schema`" in raw:
                    break   # fall through to json_object body
                return {"ok": False, "error": last_error}
            obj = _parse_openai_compatible_response(raw)
            if not obj:
                last_error = "invalid_json_message"
                log.debug("groq_invalid_json model=%s", model)
                continue
            return {"ok": True, "result": normalize_signal(obj, close)}

    return {"ok": False, "error": last_error}


def _invoke_cerebras(prompt: str, model: str, timeout: int, max_retries: int, close: Optional[float]) -> Dict[str, Any]:
    api_key = _env_first("CEREBRAS_AI_API_KEY")
    if not api_key:
        log.warning("cerebras_missing_api_key")
        return {"ok": False, "error": "missing_api_key"}

    url     = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "User-Agent":    "AiBot/1.0",
    }
    bodies = [
        {
            "model": model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  1400,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "trade_signal", "strict": True, "schema": _strip_unsupported_schema_keys(SIGNAL_SCHEMA)},
            },
        },
        {
            "model": model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens":  900,
            "response_format": {"type": "json_object"},
        },
    ]
    last_error = "retry_exhausted"
    for body in bodies:
        for attempt in range(max_retries + 1):
            status, raw = http_post_json(url, body, headers, timeout)
            if status in (429, 500, 503):
                time.sleep(1.5 + attempt)
                continue
            if status != 200:
                log.debug("cerebras_http_error model=%s status=%s", model, status)
                last_error = f"http_{status}: {clean_json(raw)[:220]}"
                if status in (400, 422) and ("response format" in raw.lower() or "wrong_api_format" in raw.lower()):
                    break
                if status == 429:
                    break
                return {"ok": False, "error": last_error}
            obj = _parse_openai_compatible_response(raw)
            if not obj:
                last_error = "invalid_json_message"
                log.debug("cerebras_invalid_json model=%s", model)
                continue
            return {"ok": True, "result": normalize_signal(obj, close)}

    return {"ok": False, "error": last_error}


# ─────────────────────────────────────────────────────────────────────────────
#  Provider dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def invoke_provider(
    provider_name: str,
    model:         str,
    prompt:        str,
    timeout:       int,
    max_retries:   int,
    close:         Optional[float],
    schema_name:   str,
) -> Dict[str, Any]:
    started       = time.perf_counter()
    provider_name = str(provider_name or "").strip().lower()
    disabled_reason = _candidate_disabled_reason(provider_name, model)
    if disabled_reason:
        return {
            "ok":               False,
            "error":            f"candidate_disabled: {disabled_reason}",
            "provider":         provider_name,
            "provider_display": PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title()),
            "model":            model,
            "latency_ms":       0,
        }

    if provider_name == "gemini":
        response = _invoke_gemini(prompt, model, timeout, max_retries, close)
    elif provider_name == "openrouter":
        response = _invoke_openrouter(prompt, model, timeout, max_retries, close)
    elif provider_name == "groq":
        response = _invoke_groq(prompt, model, timeout, max_retries, close, schema_name)
    elif provider_name == "cerebras":
        response = _invoke_cerebras(prompt, model, timeout, max_retries, close)
    else:
        response = {"ok": False, "error": f"unsupported_provider[{provider_name}]"}

    response["provider"]         = provider_name
    response["provider_display"] = PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name.title())
    response["model"]            = model
    response["latency_ms"]       = int((time.perf_counter() - started) * 1000)

    error_text = str(response.get("error") or "")
    if (
        error_text.startswith("missing_api_key")
        or error_text.startswith("http_401")
        or "User not found" in error_text
        or "model_not_found" in error_text
        or "does not exist" in error_text
    ):
        _mark_candidate_unavailable(provider_name, model, error_text)

    if response.get("ok"):
        log.info(
            "provider_ok provider=%s model=%s latency_ms=%s",
            provider_name, model, response["latency_ms"],
        )
    else:
        log.warning(
            "provider_fail provider=%s model=%s error=%s latency_ms=%s",
            provider_name, model, error_text[:120], response["latency_ms"],
        )
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  Chain runner
# ─────────────────────────────────────────────────────────────────────────────

def run_provider_chain(
    providers:           Sequence[Dict[str, Any]],
    prompt:              str,
    schema_name:         str,
    close:               Optional[float],
    timeout_budget_sec:  float,
    timeout_per_call_sec: int,
    max_retries:         int,
    actionable_confidence_floor: float = 0.70,
    hold_confidence_cutoff: float = 0.85,
) -> Optional[Dict[str, Any]]:
    deadline = time.time() + float(timeout_budget_sec)
    best_hold: Optional[Dict[str, Any]] = None
    best_directional: Optional[Dict[str, Any]] = None
    for candidate in expand_model_candidates(providers):
        remaining = int(max(1.0, min(float(timeout_per_call_sec), deadline - time.time())))
        if time.time() >= deadline:
            log.warning("provider_chain_deadline_exceeded schema=%s", schema_name)
            break
        attempt = invoke_provider(
            provider_name=str(candidate.get("provider") or ""),
            model=str(candidate.get("model") or ""),
            prompt=prompt,
            timeout=remaining,
            max_retries=max_retries,
            close=close,
            schema_name=schema_name,
        )
        if attempt.get("ok"):
            result = dict(attempt["result"])
            result["provider"]         = str(candidate.get("provider") or "")
            result["provider_display"] = attempt["provider_display"]
            result["model"]            = str(candidate.get("model") or "")
            result["model_rank"]       = int(candidate.get("rank") or 0)
            result["latency_ms"]       = int(attempt["latency_ms"])
            signal = str(result.get("signal") or "HOLD").upper()
            confidence = float(result.get("confidence") or 0.0)

            if signal in ("BUY", "SELL"):
                if best_directional is None or confidence > float(best_directional.get("confidence") or 0.0):
                    best_directional = result
                if confidence >= float(actionable_confidence_floor):
                    log.info(
                        "chain_selected rank=%s provider=%s model=%s latency_ms=%s signal=%s conf=%.2f",
                        result["model_rank"], result["provider"], result["model"],
                        result["latency_ms"], signal, confidence,
                    )
                    return result
                log.info(
                    "chain_directional_below_floor rank=%s provider=%s model=%s signal=%s conf=%.2f floor=%.2f",
                    result["model_rank"], result["provider"], result["model"],
                    signal, confidence, float(actionable_confidence_floor),
                )
                continue

            if best_hold is None or confidence > float(best_hold.get("confidence") or 0.0):
                best_hold = result
            if confidence >= float(hold_confidence_cutoff):
                log.info(
                    "chain_selected_hold rank=%s provider=%s model=%s latency_ms=%s conf=%.2f",
                    result["model_rank"], result["provider"], result["model"],
                    result["latency_ms"], confidence,
                )
                return result
            log.info(
                "chain_soft_hold_continue rank=%s provider=%s model=%s conf=%.2f cutoff=%.2f",
                result["model_rank"], result["provider"], result["model"],
                confidence, float(hold_confidence_cutoff),
            )

    if best_hold is not None and float(best_hold.get("confidence") or 0.0) >= float(hold_confidence_cutoff):
        log.info(
            "chain_selected_hold_fallback rank=%s provider=%s model=%s conf=%.2f",
            best_hold.get("model_rank"), best_hold.get("provider"),
            best_hold.get("model"), float(best_hold.get("confidence") or 0.0),
        )
        return best_hold
    if best_directional is not None:
        log.info(
            "chain_selected_best_directional rank=%s provider=%s model=%s signal=%s conf=%.2f",
            best_directional.get("model_rank"), best_directional.get("provider"),
            best_directional.get("model"), best_directional.get("signal"),
            float(best_directional.get("confidence") or 0.0),
        )
        return best_directional
    if best_hold is not None:
        log.info(
            "chain_selected_best_hold rank=%s provider=%s model=%s conf=%.2f",
            best_hold.get("model_rank"), best_hold.get("provider"),
            best_hold.get("model"), float(best_hold.get("confidence") or 0.0),
        )
        return best_hold

    log.error("provider_chain_exhausted schema=%s", schema_name)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostics runner
# ─────────────────────────────────────────────────────────────────────────────

def run_provider_diagnostics(
    providers:            Sequence[Dict[str, Any]],
    prompt:               str,
    schema_name:          str,
    close:                Optional[float],
    timeout_per_call_sec: int,
    max_retries:          int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for candidate in expand_model_candidates(providers):
        attempt = invoke_provider(
            provider_name=str(candidate.get("provider") or ""),
            model=str(candidate.get("model") or ""),
            prompt=prompt,
            timeout=timeout_per_call_sec,
            max_retries=max_retries,
            close=close,
            schema_name=schema_name,
        )
        if attempt.get("ok"):
            results.append({
                "rank":             int(candidate.get("rank") or 0),
                "provider":         str(candidate.get("provider") or ""),
                "provider_display": attempt["provider_display"],
                "ok":               True,
                "model":            str(candidate.get("model") or ""),
                "latency_ms":       int(attempt["latency_ms"]),
                "signal":           attempt["result"]["signal"],
                "confidence":       attempt["result"]["confidence"],
                "failures":         [],
            })
        else:
            results.append({
                "rank":             int(candidate.get("rank") or 0),
                "provider":         str(candidate.get("provider") or ""),
                "provider_display": PROVIDER_DISPLAY_NAMES.get(
                    str(candidate.get("provider") or ""),
                    str(candidate.get("provider") or "").title(),
                ),
                "ok":         False,
                "model":      str(candidate.get("model") or ""),
                "latency_ms": int(attempt.get("latency_ms") or 0),
                "failures":   [{
                    "model":      str(candidate.get("model") or ""),
                    "error":      str(attempt.get("error") or "unknown_error"),
                    "latency_ms": int(attempt.get("latency_ms") or 0),
                }],
            })
    return results
