# Bot/bot_utils.py
from __future__ import annotations

import html
import logging
import socket
import time
import traceback
from functools import wraps
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from queue import Queue, Full
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

import MetaTrader5 as mt5
import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectTimeout, ReadTimeout, RequestException
from telebot import apihelper
from telebot.apihelper import ApiException, ApiTelegramException
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from telebot import types

from core.config import get_config_from_env
from ExnessAPI.functions import market_is_open
from Bot.portfolio_engine import engine
from log_config import LOG_DIR as LOG_ROOT, get_log_path
from mt5_client import MT5_LOCK, mt5_status

# =============================================================================
# Logging (production-grade: rotate + no dup handlers)
# =============================================================================
LOG_DIR = LOG_ROOT

log = logging.getLogger("telegram.bot")
log.setLevel(logging.ERROR)
log.propagate = False

if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(
        str(get_log_path("telegram.log")),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)

    # Silence TeleBot internal exception spam (network/DNS issues)
    tb_log = logging.getLogger("TeleBot")
    tb_log.setLevel(logging.CRITICAL)
    tb_log.propagate = False

# =============================================================================
# Config / Session
# =============================================================================
cfg = get_config_from_env()
ADMIN = int(getattr(cfg, "admin_id", 0) or 0)

# =============================================================================
# Non-blocking notification queue (fire-and-forget for engine notifications)
# =============================================================================
_NOTIFY_QUEUE: "Queue[Tuple[int, str]]" = Queue(maxsize=200)


def notify_async(chat_id: int, message: str) -> None:
    """
    Fire-and-forget Telegram message. NEVER blocks the calling thread.
    Used by engine notifiers to avoid blocking trading loop during Telegram outages.
    """
    try:
        _NOTIFY_QUEUE.put_nowait((int(chat_id), str(message)))
    except Full:
        pass  # Drop if queue full - never block engine


def get_notify_queue() -> "Queue[Tuple[int, str]]":
    """Access the notification queue from main.py worker thread."""
    return _NOTIFY_QUEUE
PAGE_SIZE = 1

TP_USD_MIN = 1
TP_USD_MAX = 10
TP_CALLBACK_PREFIX = "tp_usd:"

SL_USD_MIN = 1
SL_USD_MAX = 10
SL_CALLBACK_PREFIX = "sl_usd:"

AI_CALLBACK_PREFIX = "ai:"

# Helpers menu: TP/SL + manual emergency order (single safe order only)
HELPER_CALLBACK_PREFIX = "hlp:"
HELPER_ORDER_COUNTS = (1,)

_session = requests.Session()
_adapter = HTTPAdapter(
    max_retries=3,
    pool_connections=int(getattr(cfg, "http_pool_conn", 20) or 20),
    pool_maxsize=int(getattr(cfg, "http_pool_max", 20) or 20),
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

apihelper.SESSION = _session
apihelper.READ_TIMEOUT = int(getattr(cfg, "telegram_read_timeout", 60) or 60)
apihelper.CONNECT_TIMEOUT = int(getattr(cfg, "telegram_connect_timeout", 60) or 60)

# =============================================================================
# Git / README Button (inline URL button)
# =============================================================================
GIT_README_URL = "https://github.com/Kabir0067/Exness/blob/main/README.md"
GIT_README_TEXT = "🔗 Маълумоти система"


def build_git_readme_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=1)
    kb.add(InlineKeyboardButton(text=GIT_README_TEXT, url=GIT_README_URL))
    return kb


# =============================================================================
# Reliability: single retry/backoff layer (no double wrapping)
# =============================================================================
TG_LOCK = Lock()


@dataclass(frozen=True)
class Backoff:
    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0

    def delay(self, attempt: int) -> float:
        return min(self.max_delay, self.base * (self.factor ** max(0, attempt - 1)))


_NETWORK_EXC = (
    ReadTimeout,
    ConnectTimeout,
    RequestException,
    urllib3.exceptions.ReadTimeoutError,
    socket.timeout,
    socket.gaierror,
    OSError,
)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, _NETWORK_EXC):
        return True

    if isinstance(exc, ApiTelegramException):
        # Retry only on transient Telegram errors
        try:
            code = int(getattr(exc, "error_code", 0) or 0)
        except Exception:
            code = 0
        return code in (429, 500, 502, 503, 504)

    if isinstance(exc, ApiException):
        # ApiException sometimes wraps network errors; heuristic:
        msg = str(exc).lower()
        return any(x in msg for x in ("timed out", "timeout", "connection", "read timed"))

    return False


def _is_callback_query_expired(exc: Exception) -> bool:
    """Check if error is a benign 'callback query expired' (user clicked button >30s ago)."""
    if isinstance(exc, ApiTelegramException):
        try:
            code = int(getattr(exc, "error_code", 0) or 0)
            desc = str(getattr(exc, "description", "") or "").lower()
            # Telegram returns 400 "query is too old" when callback expires
            return code == 400 and ("query is too old" in desc or "query id is invalid" in desc)
        except Exception:
            pass
    return False


def _is_message_not_modified(exc: Exception) -> bool:
    """Check if error is 'message is not modified' (content is same)."""
    if isinstance(exc, ApiTelegramException):
        try:
            code = int(getattr(exc, "error_code", 0) or 0)
            desc = str(getattr(exc, "description", "") or "").lower()
            return code == 400 and "message is not modified" in desc
        except Exception:
            pass
    return False


def tg_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 6,
    backoff: Backoff = Backoff(),
    on_permanent_failure: Optional[Callable[[Exception], bool]] = None,
    **kwargs: Any,
) -> Any:
    fn_name = getattr(fn, "__name__", "call")
    for attempt in range(1, max_retries + 1):
        try:
            with TG_LOCK:
                return fn(*args, **kwargs)
        except Exception as exc:
            # Silently ignore benign callback query timeout errors
            if _is_callback_query_expired(exc):
                return None  # Silent fail - this is expected when user clicks old buttons

            # Silently ignore "message is not modified"
            if _is_message_not_modified(exc):
                return None
            
            should_retry = _should_retry(exc) and attempt < max_retries
            if not should_retry:
                # Fallback: if HTML parsing failed, retry plain text
                try:
                    if isinstance(exc, ApiTelegramException):
                        desc = str(getattr(exc, "description", "")).lower()
                        code = int(getattr(exc, "error_code", 0) or 0)
                        if code == 400 and ("parse entities" in desc or "start tag" in desc):
                            pm = kwargs.get("parse_mode")
                            if pm and str(pm).lower() == "html":
                                log.warning("TG HTML parse error, retrying plain text... | err=%s", exc)
                                kwargs["parse_mode"] = None
                                continue
                except Exception:
                    pass

                suppress_log = False
                if on_permanent_failure:
                    try:
                        suppress_log = bool(on_permanent_failure(exc))
                    except Exception as hook_exc:
                        log.error(
                            "TG failure hook error fn=%s err=%s | tb=%s",
                            fn_name,
                            hook_exc,
                            traceback.format_exc(),
                        )
                if not suppress_log:
                    if fn_name == "send_chat_action" and _should_retry(exc):
                        log.warning("TG call failed fn=%s err=%s", fn_name, exc)
                    else:
                        log.error(
                            "TG call failed fn=%s err=%s | tb=%s",
                            fn_name,
                            exc,
                            traceback.format_exc(),
                        )
                return None

            d = backoff.delay(attempt)
            if attempt == 1 or attempt % 3 == 0:
                log.warning(
                    "TG retry fn=%s attempt=%d/%d err=%s sleep=%.1fs",
                    fn_name,
                    attempt,
                    max_retries,
                    exc,
                    d,
                )
            time.sleep(d)
    return None


# =============================================================================
# Bounded TTL cache (no leaks; supports caching None + pop)
# =============================================================================
class TTLCache:
    def __init__(self, *, maxsize: int, ttl_sec: float) -> None:
        self.maxsize = int(maxsize)
        self.ttl = float(ttl_sec)
        self._d: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()

    def _get_raw(self, key: Any) -> Tuple[bool, Any]:
        now = time.time()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return False, None
            ts, val = item
            if (now - ts) > self.ttl:
                self._d.pop(key, None)
                return False, None
            self._d.move_to_end(key)
            return True, val

    def get(self, key: Any) -> Any:
        found, val = self._get_raw(key)
        return val if found else None

    def has(self, key: Any) -> bool:
        found, _ = self._get_raw(key)
        return found

    def set(self, key: Any, val: Any) -> None:
        now = time.time()
        with self._lock:
            self._d[key] = (now, val)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            item = self._d.pop(key, None)
        if item is None:
            return default
        _, val = item
        return val


# Cache of chats that blocked the bot to avoid repeated 403 spam.
_blocked_chat_cache = TTLCache(maxsize=512, ttl_sec=12 * 3600)

# Typing throttle (prevents chat_action storms)
_typing_cache = TTLCache(maxsize=2048, ttl_sec=1.2)
_typing_pause_cache = TTLCache(maxsize=2048, ttl_sec=20.0)

# Prevent notification spam for blocked signals
_notify_skip_cache = TTLCache(maxsize=500, ttl_sec=300)


def _format_datetime(dt: Optional[datetime] = None, show_date: bool = False, show_time: bool = True) -> str:
    """Форматкунии сана ва вақт барои паёмҳои бот."""
    if dt is None:
        dt = datetime.now()

    parts: list[str] = []
    if show_date:
        parts.append(f"<b>{dt.strftime('%Y-%m-%d')}</b>")
    if show_time:
        parts.append(f"<code>{dt.strftime('%H:%M:%S')}</code>")

    return " ".join(parts) if parts else ""


def _format_time_only() -> str:
    """Танҳо вақтро дар шакли зебо бармегардонад."""
    return f"<code>{datetime.now().strftime('%H:%M:%S')}</code>"


def _format_date_time() -> str:
    """Сана ва вақтро дар шакли зебо бармегардонад."""
    now = datetime.now()
    return f"<b>{now.strftime('%Y-%m-%d')}</b> <code>{now.strftime('%H:%M:%S')}</code>"


def _tajik_signal_label(signal: Any) -> str:
    sig = str(signal or "").strip().upper()
    mapping = {
        "BUY": "ХАРИД",
        "STRONG BUY": "ХАРИДИ ҚАВӢ",
        "SELL": "ФУРӮШ",
        "STRONG SELL": "ФУРӮШИ ҚАВӢ",
        "HOLD": "ИНТИЗОР",
        "WAIT": "ИНТИЗОР",
        "NEUTRAL": "ИНТИЗОР",
    }
    return mapping.get(sig, sig or "-")


def _extract_chat_id_from_call(
    fn_name: str,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Optional[Any]:
    chat_id = kwargs.get("chat_id")
    if chat_id is not None:
        return chat_id

    if fn_name in {"send_message", "send_chat_action"}:
        if args:
            candidate = args[0]
            if isinstance(candidate, (int, str)):
                return candidate

    if fn_name == "edit_message_text":
        # telebot: edit_message_text(text, chat_id, message_id, ...)
        if len(args) >= 2:
            candidate = args[1]
            if isinstance(candidate, (int, str)):
                return candidate

    return None


def _handle_permanent_telegram_failure(
    fn_name: str,
    exc: Exception,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> bool:
    if isinstance(exc, ApiTelegramException):
        try:
            code = int(getattr(exc, "error_code", 0) or 0)
        except Exception:
            code = 0
        description = (getattr(exc, "description", "") or str(exc)).lower()
        chat_id = _extract_chat_id_from_call(fn_name, args, kwargs)

        if code == 403 and "blocked" in description and chat_id is not None:
            already_blocked = bool(_blocked_chat_cache.get(chat_id))
            _blocked_chat_cache.set(chat_id, True)
            if not already_blocked:
                log.warning(
                    "Telegram chat %s blocked the bot; suppressing further messages",
                    chat_id,
                )
            return True
    return False


# Global reference to bot instance (set from bot.py)
_bot_ref: Optional[Any] = None

# Global reference to original send_chat_action (set from bot.py)
_orig_send_chat_action_ref: Optional[Callable[..., Any]] = None


def set_bot_instance(bot_instance: Any) -> None:
    """Set the bot instance from bot.py"""
    global _bot_ref
    _bot_ref = bot_instance


def set_orig_send_chat_action(func: Callable[..., Any]) -> None:
    """Set the original send_chat_action function from bot.py"""
    global _orig_send_chat_action_ref
    _orig_send_chat_action_ref = func


def _maybe_send_typing(chat_id: Optional[Any]) -> None:
    # Global policy: every outgoing message triggers typing, but throttled.
    # Hard requirement: always emit typing for outgoing UX.
    if chat_id is None:
        return
    if _blocked_chat_cache.get(chat_id):
        return
    if _typing_cache.has(chat_id):
        return
    if _typing_pause_cache.has(chat_id):
        return
    _typing_cache.set(chat_id, True)

    # Use the original function if available
    if _orig_send_chat_action_ref is not None:
        send_func = _orig_send_chat_action_ref
    else:
        try:
            send_func = getattr(apihelper, "send_chat_action", None)
            if send_func is None:
                log.warning("_maybe_send_typing: send_chat_action not available, skipping typing")
                return
        except Exception:
            log.warning("_maybe_send_typing: failed to get send_chat_action, skipping typing")
            return

    result = tg_call(
        send_func,
        chat_id,
        action="typing",
        max_retries=1,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "send_chat_action",
            exc,
            (chat_id,),
            {"action": "typing"},
        ),
    )
    if result is None:
        _typing_pause_cache.set(chat_id, True)


def _extract_chat_id_from_handler_args(*args: Any, **kwargs: Any) -> Optional[Any]:
    if args:
        obj = args[0]
        try:
            chat = getattr(obj, "chat", None)
            if chat is not None:
                cid = getattr(chat, "id", None)
                if cid is not None:
                    return cid
        except Exception:
            pass
        try:
            msg = getattr(obj, "message", None)
            if msg is not None:
                chat = getattr(msg, "chat", None)
                cid = getattr(chat, "id", None) if chat is not None else None
                if cid is not None:
                    return cid
        except Exception:
            pass
    try:
        if "chat_id" in kwargs:
            return kwargs.get("chat_id")
    except Exception:
        pass
    return None


def send_action(action: str = "typing") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for Telegram handlers.
    Sends chat action before handler logic to make the bot feel alive.
    """
    action_name = str(action or "typing").strip().lower() or "typing"

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            chat_id = _extract_chat_id_from_handler_args(*args, **kwargs)
            if chat_id is None:
                return fn(*args, **kwargs)
            if action_name == "typing":
                _maybe_send_typing(chat_id)
            else:
                tg_call(
                    getattr(apihelper, "send_chat_action"),
                    chat_id,
                    action=action_name,
                    max_retries=1,
                )
            return fn(*args, **kwargs)

        return _wrapped

    return _decorator


def _rk_remove() -> types.ReplyKeyboardRemove:
    # Only policy: remove reply keyboard only via ReplyKeyboardRemove
    return types.ReplyKeyboardRemove()


def _fmt_price(val: float) -> str:
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def _normalize_confidence(val: Any, *, cap: Optional[float] = None) -> float:
    try:
        conf = float(val or 0.0)
    except Exception:
        conf = 0.0
    if conf > 1.0:
        conf = conf / 100.0
    conf = max(0.0, conf)
    if cap is not None:
        conf = min(float(cap), conf)
    return conf


def _fmt_metric(val: Any, digits: int = 6) -> str:
    try:
        out = f"{float(val):.{int(digits)}f}".rstrip("0").rstrip(".")
        return out or "0"
    except Exception:
        return str(val or "-")


def _humanize_ml_reason(raw: str) -> str:
    fields: Dict[str, str] = {}
    for chunk in str(raw or "").split("|"):
        part = chunk.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key.strip()] = value.strip()

    pred = fields.get("catboost_pred") or fields.get("pred")
    thr = fields.get("thr") or fields.get("model_q_thr")
    quantile = fields.get("model_q")

    if pred and thr:
        try:
            relation = ">" if float(pred) >= float(thr) else "<"
        except Exception:
            relation = "vs"
        tail = f" | q={_fmt_metric(quantile, 1)}" if quantile else ""
        return (
            f"CatBoost: pred {_fmt_metric(pred)} {relation} thr {_fmt_metric(thr)}"
            f"{tail}"
        )

    compact = str(raw or "").replace("|", " | ").strip()
    return compact or "Тафсил дастрас нест"


def _humanize_signal_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""

    key, sep, value = text.partition(":")
    if not sep:
        return text.replace("_", " ")

    value = value.strip()
    if key == "ml_provider" and value:
        return f"Провайдери ML: {value}"
    if key == "ml_model" and value:
        return f"Модели ML: {value}"
    if key == "ml_reason" and value:
        return _humanize_ml_reason(value)
    if key == "catboost_below_threshold" and value:
        return f"CatBoost аз ҳадд поён аст: {value}"

    return f"{key.replace('_', ' ')}: {value}" if value else key.replace("_", " ")


def _format_reason_block(reasons_raw: Any, *, max_items: int = 3) -> str:
    if isinstance(reasons_raw, (list, tuple, set)):
        raw_items = list(reasons_raw)
    elif reasons_raw:
        raw_items = [reasons_raw]
    else:
        raw_items = []

    lines: list[str] = []
    for item in raw_items:
        label = _humanize_signal_reason(item)
        if label:
            lines.append(f"• {html.escape(label)}")
        if len(lines) >= int(max_items):
            break

    if not lines:
        lines.append("• Сабаб дастрас нест")

    return "\n".join(lines)


def _is_interim_execution_reason(reason: Any) -> bool:
    text = str(reason or "").strip().lower()
    return (
        text.startswith("order_send_none")
        or text.startswith("retry_retcode_")
        or text.startswith("exception_")
    )


def _humanize_execution_reason(reason: Any, retcode: Any = 0) -> str:
    text = str(reason or "").strip()
    lower = text.lower()

    if lower.startswith("sync_timeout_no_broker_state"):
        return "Брокер дар муҳлати интизорӣ иҷрои фармонро тасдиқ накард"
    if lower.startswith("order_send_none"):
        return "MT5 фармонро қабул накард ё ҷавоби равшан надод"
    if lower.startswith("retry_retcode_"):
        return "Ирсоли фармон ба сабаби ҷавоби ноустувори MT5 тасдиқ нашуд"
    if lower.startswith("fail_retcode_"):
        code = text.rsplit("_", 1)[-1]
        return f"Фармон аз тарафи брокер рад шуд (retcode={code})"
    if lower.startswith("dispatch_gate:"):
        detail = text.split(":", 1)[1].replace("_", " ").strip()
        return f"Қоидаи иҷро фармонро рад кард: {detail or '-'}"
    if lower == "tick_none":
        return "Нархи ҷорӣ аз MT5 гирифта нашуд"
    if lower == "bad_req_price":
        return "Нархи бозор барои ирсол нодуруст буд"
    if lower == "mt5_not_ready":
        return "Пайвасти MT5 барои ирсол омода нест"
    if lower == "symbol_meta_none":
        return "Маълумоти символ барои ирсол дастрас нест"
    if lower.startswith("exception_"):
        return "Ҳангоми ирсоли фармон хатои дохилӣ рух дод"

    if int(retcode or 0):
        return f"{text or 'Хатои иҷро'} (retcode={int(retcode)})"
    return text.replace("_", " ") or "Хатои номаълум"


def _normalize_signal_value(signal: Any) -> str:
    sig_raw = str(signal or "").strip()
    sig_norm = sig_raw.upper()
    if sig_norm in ("BUY", "STRONG BUY"):
        return "Buy"
    if sig_norm in ("SELL", "STRONG SELL"):
        return "Sell"
    return sig_raw


def _build_signal_message(asset: str, result: Any) -> str:
    def _field(name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    sig = _normalize_signal_value(_field("signal", "Hold") or "Hold")
    conf = _normalize_confidence(_field("confidence", 0.0))
    if conf < 0.50:
        return ""

    direction_emoji = "🟢" if sig.lower() == "buy" else "🔴"
    time_str = _format_time_only()
    phase = str(_field("phase", "") or "").strip().upper()
    blocked = bool(_field("blocked", False))
    lot = float(_field("lot", 0.0) or 0.0)
    order_id = str(_field("order_id", "") or "").strip()

    extra_lines: list[str] = []
    if lot > 0.0:
        extra_lines.append(f"📦 Ҳаҷм: <b>{lot:.4f}</b>")
    if order_id:
        extra_lines.append(f"🧷 ID-и навбат: <code>{html.escape(order_id)}</code>")
    extra_block = "".join(f"{line}\n" for line in extra_lines)

    title = "📡 <b>СИГНАЛ БА НАВБАТ ГУЗОШТА ШУД</b>"
    footer = "<i>Дар навбати иҷрои broker/MT5...</i>"
    if blocked or phase == "C":
        title = "⚠️ <b>СИГНАЛИ PHASE C</b>"
        footer = "<i>Сигнал маҳдуд шуд ва иҷро намегардад.</i>"

    signal_label = _tajik_signal_label(sig)
    reasons_block = _format_reason_block(_field("reasons", []))

    return (
        f"{title} | {direction_emoji} <b>{signal_label}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💎 <b>{html.escape(str(asset))}</b>\n"
        f"🧠 Боварӣ: <b>{conf * 100:.1f}%</b>\n"
        f"🔍 Асос:\n"
        f"{reasons_block}\n"
        f"{extra_block}"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{footer}\n"
        f"{time_str}"
    )


def _build_order_update_message(intent: Any, result: Any) -> str:
    ok = bool(getattr(result, "ok", False))
    reason = str(getattr(result, "reason", "") or "")
    if (not ok) and _is_interim_execution_reason(reason):
        return ""

    asset = html.escape(str(getattr(intent, "asset", "-") or "-"))
    symbol = html.escape(str(getattr(intent, "symbol", "-") or "-"))
    order_id = html.escape(str(getattr(intent, "order_id", "") or ""))
    lot = float(getattr(intent, "lot", 0.0) or 0.0)
    sl = float(getattr(intent, "sl", 0.0) or 0.0)
    tp = float(getattr(intent, "tp", 0.0) or 0.0)
    conf = _normalize_confidence(getattr(intent, "confidence", 0.0), cap=0.96)
    conf_pct = conf * 100.0
    direction_emoji = "🟢" if str(getattr(intent, "signal", "")).lower() == "buy" else "🔴"
    signal_label = _tajik_signal_label(getattr(intent, "signal", ""))
    time_str = _format_time_only()

    id_line = f"🧷 ID-и фармон: <code>{order_id}</code>\n" if order_id else ""

    if ok:
        sltp = f"{_fmt_price(sl)} / {_fmt_price(tp)}" if (sl > 0.0 and tp > 0.0) else "-"
        note = ""
        if reason.startswith("sync_recovered_"):
            note = "<i>Баъди санҷиши брокер тасдиқ шуд.</i>\n"
        return (
            f"✅ <b>ОРДЕР КУШОДА ШУД</b> | {direction_emoji} <b>{signal_label}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💎 <b>{asset}</b> | {symbol}\n"
            f"📦 Ҳаҷм: <b>{lot:.4f}</b>\n"
            f"{id_line}"
            f"🏷 Нарх: <b>{_fmt_price(getattr(result, 'exec_price', 0.0))}</b>\n"
            f"🛡 SL / 🎯 TP: <b>{sltp}</b>\n"
            f"🧠 Боварӣ: <b>{conf_pct:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{note}{time_str}"
        )

    retcode = int(getattr(result, "retcode", 0) or 0)
    fail_reason = html.escape(_humanize_execution_reason(reason, retcode))
    code_line = f"📟 Retcode: <code>{retcode}</code>\n" if retcode else ""
    return (
        f"❌ <b>ИҶРО НАШУД</b> | {direction_emoji} <b>{signal_label}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💎 <b>{asset}</b> | {symbol}\n"
        f"📦 Ҳаҷм: <b>{lot:.4f}</b>\n"
        f"{id_line}"
        f"🚫 Сабаб: <b>{fail_reason}</b>\n"
        f"{code_line}"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>Сигнал омад, аммо брокер иҷрои онро тасдиқ накард.</i>\n"
        f"{time_str}"
    )


def _send_clean(chat_id: int, text: str, *, parse_mode: str = "HTML") -> None:
    # One-shot message that guarantees reply keyboard is removed.
    if _bot_ref is None:
        log.error("_send_clean: bot instance not set")
        return
    _bot_ref.send_message(chat_id, text, parse_mode=parse_mode, reply_markup=_rk_remove())


def _notify_order_update(intent: Any, result: Any) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        msg = _build_order_update_message(intent, result)
        if msg:
            notify_async(ADMIN, msg)
    except Exception:
        return


_notify_order_opened = _notify_order_update


def _notify_order_skipped(candidate: Any) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        
        # Debounce spam
        if candidate and hasattr(candidate, "signal_id"):
             sid = str(candidate.signal_id)
             if _notify_skip_cache.get(sid):
                 return
             _notify_skip_cache.set(sid, True)
        
        # Only notify if explicitly blocked or has reasons
        if not getattr(candidate, "blocked", False) and not getattr(candidate, "reasons", None):
            return

        conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
        if conf > 1.0:
            conf = conf / 100.0
        # Filter weak signals to avoid spam - only notify Strong signals that got blocked
        if conf < 0.80:
            return

        sltp = f"{_fmt_price(float(getattr(candidate, 'sl', 0)))} / {_fmt_price(float(getattr(candidate, 'tp', 0)))}"
        lot = float(getattr(candidate, "lot", 0.0))
        
        reasons_list = getattr(candidate, "reasons", []) or []
        reasons_str = ", ".join(reasons_list) if reasons_list else "Маҳдудияти сахт / филтр"

        direction_emoji = "🟢" if str(candidate.signal).lower() == "buy" else "🔴"
        signal_label = _tajik_signal_label(candidate.signal)
        time_str = _format_time_only()

        # User requested actionable message even if blocked
        msg = (
            f"⚠️ <b>СИГНАЛ МАНЪ ШУД</b> | {direction_emoji} <b>{signal_label}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💎 <b>{html.escape(str(candidate.asset))}</b> | {html.escape(str(candidate.symbol))}\n"
            f"📦 Ҳаҷм: <b>{lot:.2f}</b> | 🧠 Боварӣ: <b>{conf*100:.1f}%</b>\n"
            f"🛡 SL/TP: <b>{sltp}</b>\n"
            f"🚫 Сабаб: <b>{html.escape(reasons_str)}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"<i>Барои иҷрои дастӣ метавонед аз меню истифода баред.</i>\n"
            f"{time_str}"
        )
        # Non-blocking: push to queue
        notify_async(ADMIN, msg)
    except Exception:
        return


def _notify_signal(asset: str, result: Any) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        signal_id = ""
        signal_value = ""
        if isinstance(result, dict):
            signal_id = str(result.get("signal_id", "") or "").strip()
            signal_value = _normalize_signal_value(result.get("signal", ""))
        else:
            signal_id = str(getattr(result, "signal_id", "") or "").strip()
            signal_value = _normalize_signal_value(getattr(result, "signal", ""))
        if signal_value not in ("Buy", "Sell"):
            return
        if signal_id:
            cache_key = f"signal_notify:{signal_id}"
            if _notify_skip_cache.get(cache_key):
                return
            _notify_skip_cache.set(cache_key, True)
        msg = _build_signal_message(asset, result)
        if msg:
            notify_async(ADMIN, msg)
    except Exception:
        return

def _notify_phase_change(asset: str, old_phase: str, new_phase: str, reason: str = "") -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        reason_line = f" | <b>{reason}</b>" if reason else ""
        time_str = _format_time_only()
        msg = (
            f"🔄 <b>{asset}</b>: <b>{old_phase}</b> → <b>{new_phase}</b>{reason_line}\n"
            f"{time_str}"
        )
        # Non-blocking: push to queue, don't wait for Telegram
        notify_async(ADMIN, msg)
    except Exception:
        return


def _notify_engine_stopped(asset: str, reason: str = "") -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        reason_line = f" | <b>{reason}</b>" if reason else ""
        time_str = _format_time_only()
        msg = (
            f"🛑 <b>{asset}</b> қатъ шуд{reason_line}\n"
            f"✅ Барои оғоз: /buttons → «🚀 Оғози Тиҷорат»\n"
            f"{time_str}"
        )
        # Non-blocking: push to queue, don't wait for Telegram
        notify_async(ADMIN, msg)
    except Exception:
        return


def _notify_daily_start(asset: str, day: str) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        time_str = _format_time_only()
        msg = (
            f"🌅 <b>{asset}</b> | Рӯзи нав: <b>{day}</b>\n"
            f"✅ Лимитҳо ва омор аз нав ҳисоб шуданд\n"
            f"{time_str}"
        )
        # Non-blocking: push to queue, don't wait for Telegram
        notify_async(ADMIN, msg)
    except Exception:
        return


def is_admin_chat(chat_id: int) -> bool:
    return bool(ADMIN and int(chat_id) == int(ADMIN))


def deny(message: types.Message) -> None:
    """Unauthorized access message + inline README button."""
    if _bot_ref is None:
        log.error("deny: bot instance not set")
        return

    msg = (
        "🔒 <b>ДАСТРАСИИ ХУСУСӢ</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Ин бот танҳо барои соҳиби худ дастрас аст.\n"
        "Дастрасӣ маҳдуд мебошад.\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "👤 Соҳиб: @kabir_0067"
    )

    _bot_ref.send_message(
        message.chat.id,
        msg,
        parse_mode="HTML",
        reply_markup=build_git_readme_keyboard(),
    )


def fetch_external_correlation(cfg_obj: Any) -> Optional[float]:
    if not bool(getattr(cfg_obj, "enable_telemetry", False)):
        return None

    vs = str(getattr(cfg_obj, "correlation_vs_currency", "usd") or "usd").lower()
    key = ("coingecko", "pax-gold", vs)

    found, cached = _corr_cache._get_raw(key)
    if found:
        return cached  # may be float or None (cached None)

    url = f"https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies={vs}"
    timeout_sec = float(getattr(cfg_obj, "correlation_timeout_sec", 1.2) or 1.2)
    try:
        resp = _session.get(url, timeout=timeout_sec, headers={"User-Agent": "xau-bot/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("pax-gold", {}).get(vs)
            val = float(price) if price is not None else None
            _corr_cache.set(key, val)
            return val
        _corr_cache.set(key, None)
        return None
    except Exception as exc:
        log.warning("fetch_external_correlation error: %s", exc)
        _corr_cache.set(key, None)
        return None


def build_health_ribbon(status: Any, compact: bool = True) -> str:
    try:
        active_str = str(getattr(status, "active_asset", "NONE"))
        if active_str == "NONE" and getattr(status, "trading", False):
            active_str = "ҶУСТУҶӮ"

        segments: list[str] = [
            f"DD {float(getattr(status, 'dd_pct', 0.0)) * 100:.1f}%",
            f"P&L {float(getattr(status, 'today_pnl', 0.0)):+.2f}",
            f"Ҳолат {active_str}",
            f"XAU {int(getattr(status, 'open_trades_xau', 0))}",
            f"BTC {int(getattr(status, 'open_trades_btc', 0))}",
        ]

        last_xau = _tajik_signal_label(getattr(status, "last_signal_xau", "Neutral"))
        last_btc = _tajik_signal_label(getattr(status, "last_signal_btc", "Neutral"))
        segments.append(f"Сигнал XAU:{last_xau} BTC:{last_btc}")

        # Keep compact ribbons instant (status command path).
        if not compact:
            corr = fetch_external_correlation(cfg)
            if corr is not None:
                segments.append(f"Ҳамбастагии GOLD {corr:+.2f}")

        ribbon = " | ".join(segments)
        prefix = "\n" if compact else ""
        return f"{prefix}<b>🧭 Мониторинг:</b> {ribbon}"
    except Exception as exc:
        log.error("build_health_ribbon error: %s", exc)
        return ""


def _format_status_message(status: Any) -> str:
    active_label = str(getattr(status, "active_asset", "NONE"))
    if active_label == "NONE" and getattr(status, "trading", False):
        active_label = "✅ Дар ҷустуҷӯ (XAU + BTC)"

    active_icon = "🟢" if getattr(status, "trading", False) else "🔴"
    trading_status = "ФАЪОЛ" if getattr(status, "trading", False) else "ХОМӮШ"
    mt5_state = "ПАЙВАСТ" if getattr(status, "connected", False) else "ҶУДО"
    balance = float(getattr(status, "balance", 0.0))
    equity = float(getattr(status, "equity", 0.0))
    today_pnl = float(getattr(status, "today_pnl", 0.0))
    dd_pct = float(getattr(status, "dd_pct", 0.0))
    open_xau = int(getattr(status, "open_trades_xau", 0))
    open_btc = int(getattr(status, "open_trades_btc", 0))
    sig_xau = _tajik_signal_label(getattr(status, "last_signal_xau", "Neutral"))
    sig_btc = _tajik_signal_label(getattr(status, "last_signal_btc", "Neutral"))
    queue_size = int(getattr(status, "exec_queue_size", 0))

    lines = [
        f"{active_icon} <b>Система</b> | Савдо: <b>{trading_status}</b> | MT5: <b>{mt5_state}</b>",
        f"💰 Баланс: <b>{balance:.2f}$</b> | Эквити: <b>{equity:.2f}$</b>",
        f"🧭 Ҳолати фаъол: <b>{active_label}</b>",
    ]
    if today_pnl != 0:
        lines[1] += f" | P&L-и рӯз: <b>{today_pnl:+.2f}$</b>"
    if dd_pct > 0:
        lines[2] += f" | 📉 DD: <b>{dd_pct:.2%}</b>"

    signal_line = f"🔹 XAU: <b>{open_xau}</b> | {sig_xau} | 🔸 BTC: <b>{open_btc}</b> | {sig_btc}"
    if queue_size > 0:
        signal_line += f" | 📥 Навбат: <b>{queue_size}</b>"
    lines.append(signal_line)
    return "\n".join(lines) + "\n"


def _build_daily_summary_text(summary: Dict[str, Any]) -> str:
    text = (
        "📜 <b>Хулосаи савдои рӯзона</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Сана: <b>{summary.get('date', '-')}</b>\n"
    )

    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed > 0:
        net = float(summary.get("net", 0.0) or 0.0)
        wins = int(summary.get("wins", 0) or 0)
        losses = int(summary.get("losses", 0) or 0)
        profit = float(summary.get("profit", 0.0) or 0.0)
        loss = float(summary.get("loss", 0.0) or 0.0)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
        pnl_emoji = "🟢" if net >= 0 else "🔴"

        text += (
            f"\n{pnl_emoji} <b>P&L: {net:+.2f}$</b>\n"
            f"📊 Бурд: <b>{wins}</b> | Бохт: <b>{losses}</b> | WR: <b>{win_rate:.1f}%</b>\n"
            f"💹 Фоида: +{profit:.2f}$ | 📉 Зиён: -{loss:.2f}$\n"
        )
    else:
        text += "\n🚫 Ордерҳои басташуда: 0\n"

    if total_open > 0:
        unrealized = float(summary.get("unrealized_pnl", 0.0) or 0.0)
        text += f"🕒 Ордерҳои кушода: <b>{total_open}</b> | P&L: <b>{unrealized:+.2f}$</b>\n"

    balance = float(summary.get("balance", 0.0) or 0.0)
    text += f"\n💰 Баланс: <b>{balance:.2f}$</b>\n"
    return text


def _build_tp_usd_keyboard(min_usd: int = TP_USD_MIN, max_usd: int = TP_USD_MAX, row_width: int = 5) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=row_width)
    kb.add(
        *[
            InlineKeyboardButton(text=f"{i}$", callback_data=f"{TP_CALLBACK_PREFIX}{i}")
            for i in range(int(min_usd), int(max_usd) + 1)
        ]
    )
    kb.add(InlineKeyboardButton(text="❌ Бекор", callback_data=f"{TP_CALLBACK_PREFIX}cancel"))
    return kb


def _format_tp_result(usd: float, res: dict) -> str:
    total = int(res.get("total", 0) or 0)
    updated = int(res.get("updated", 0) or 0)
    skipped = int(res.get("skipped", 0) or 0)
    ok = bool(res.get("ok", False))
    errors = res.get("errors") or []

    status_emoji = "✅" if ok else "⚠️"
    lines = [f"{status_emoji} <b>TP: {usd:.0f}$</b> | Навсозӣ: <b>{updated}/{total}</b>"]
    if skipped > 0:
        lines.append(f"⏭️ Гузаронда шуд: <b>{skipped}</b>")
    if errors:
        preview = " | ".join(html.escape(str(e))[:30] for e in errors[:3])
        lines.append(f"⚠️ <code>{preview}</code>")
    return "\n".join(lines)


def _build_sl_usd_keyboard(min_usd: int = SL_USD_MIN, max_usd: int = SL_USD_MAX, row_width: int = 5) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=row_width)
    kb.add(
        *[
            InlineKeyboardButton(text=f"{i}$", callback_data=f"{SL_CALLBACK_PREFIX}{i}")
            for i in range(int(min_usd), int(max_usd) + 1)
        ]
    )
    kb.add(InlineKeyboardButton(text="❌ Бекор", callback_data=f"{SL_CALLBACK_PREFIX}cancel"))
    return kb


def build_ai_keyboard() -> InlineKeyboardMarkup:
    """Тугмаҳои менюи AI барои XAU ва BTC."""
    kb = InlineKeyboardMarkup(row_width=2)
    kb.row(
        InlineKeyboardButton(text="🥇 AI XAU", callback_data=f"{AI_CALLBACK_PREFIX}xau_scalp"),
        InlineKeyboardButton(text="₿ AI BTC", callback_data=f"{AI_CALLBACK_PREFIX}btc_scalp"),
    )
    kb.row(
        InlineKeyboardButton(text="📈 XAU рӯзона", callback_data=f"{AI_CALLBACK_PREFIX}xau_intraday"),
        InlineKeyboardButton(text="📉 BTC рӯзона", callback_data=f"{AI_CALLBACK_PREFIX}btc_intraday"),
    )
    return kb


def build_helpers_keyboard() -> InlineKeyboardMarkup:
    """Ёвариҳо: TP/SL + харид/фурӯш (монанд ба TP/SL — аввал интихоб, баъд шумора)."""
    kb = InlineKeyboardMarkup(row_width=2)
    kb.row(
        InlineKeyboardButton(text="📈 Тейк-профит", callback_data=f"{HELPER_CALLBACK_PREFIX}tp"),
        InlineKeyboardButton(text="🛡 Стоп-лосс", callback_data=f"{HELPER_CALLBACK_PREFIX}sl"),
    )
    kb.row(
        InlineKeyboardButton(text="🟢 BTC ↑ Харид", callback_data=f"{HELPER_CALLBACK_PREFIX}buy_btc"),
        InlineKeyboardButton(text="🔴 BTC ↓ Фурӯш", callback_data=f"{HELPER_CALLBACK_PREFIX}sell_btc"),
    )
    kb.row(
        InlineKeyboardButton(text="🟢 XAU ↑ Харид", callback_data=f"{HELPER_CALLBACK_PREFIX}buy_xau"),
        InlineKeyboardButton(text="🔴 XAU ↓ Фурӯш", callback_data=f"{HELPER_CALLBACK_PREFIX}sell_xau"),
    )
    return kb


def build_helper_order_count_keyboard(action: str) -> InlineKeyboardMarkup:
    """Тугмаҳои шумора барои кушодани ордерҳо."""
    kb = InlineKeyboardMarkup(row_width=4)
    kb.add(
        *[
            InlineKeyboardButton(text=str(c), callback_data=f"{HELPER_CALLBACK_PREFIX}{action}:{c}")
            for c in HELPER_ORDER_COUNTS
        ]
    )
    kb.add(InlineKeyboardButton(text="❌ Бекор", callback_data=f"{HELPER_CALLBACK_PREFIX}{action}:cancel"))
    return kb


def format_close_by_profit_result(res: Dict[str, Any]) -> str:
    """Формати натиҷаи «бастани фақат фоидадор» барои пайғоми бот."""
    closed = int(res.get("closed", 0) or 0)
    ok = bool(res.get("ok", False))
    errors = res.get("errors") or []
    status_emoji = "✅" if ok else "⚠️"
    lines = [f"{status_emoji} <b>Фоидадорҳои баста:</b> <b>{closed}</b>"]
    if errors:
        preview = " | ".join(html.escape(str(e))[:25] for e in errors[:2])
        lines.append(f"⚠️ <code>{preview}</code>")
    return "\n".join(lines)


def _format_sl_result(usd: float, res: dict) -> str:
    total = int(res.get("total", 0) or 0)
    updated = int(res.get("updated", 0) or 0)
    skipped = int(res.get("skipped", 0) or 0)
    ok = bool(res.get("ok", False))
    errors = res.get("errors") or []

    status_emoji = "✅" if ok else "⚠️"
    lines = [f"{status_emoji} <b>SL: {usd:.0f}$</b> | Навсозӣ: <b>{updated}/{total}</b>"]
    if skipped > 0:
        lines.append(f"⏭️ Гузаронда шуд: <b>{skipped}</b>")
    if errors:
        preview = " | ".join(html.escape(str(e))[:30] for e in errors[:3])
        lines.append(f"⚠️ <code>{preview}</code>")
    return "\n".join(lines)


_summary_cache = TTLCache(maxsize=4, ttl_sec=3.0)
_corr_cache = TTLCache(maxsize=8, ttl_sec=float(getattr(cfg, "correlation_refresh_sec", 60) or 60))
_orders_kb_cache = TTLCache(maxsize=256, ttl_sec=120.0)


def format_order(order_data: Dict[str, Any]) -> str:
    direction_emoji = "🟢" if order_data.get("type") == "BUY" else "🔴"
    direction_text = "ХАРИД" if order_data.get("type") == "BUY" else "ФУРӮШ"
    ticket = order_data.get("ticket", "-")
    symbol = order_data.get("symbol", "-")
    volume = float(order_data.get("volume", 0.0) or 0.0)
    price = float(order_data.get("price", 0.0) or 0.0)
    profit = float(order_data.get("profit", 0.0) or 0.0)
    profit_emoji = "🟢" if profit >= 0 else "🔴"

    return (
        f"{direction_emoji} <b>{direction_text}</b> | <b>{html.escape(str(symbol))}</b> | #{ticket}\n"
        f"📦 Ҳаҷм: <b>{volume:.2f}</b> | 🏷 Нарх: <b>{price:.5f}</b> | {profit_emoji} <b>{profit:+.2f}$</b>"
    )


def order_keyboard(index: int, total: int, ticket: int) -> InlineKeyboardMarkup:
    key = (index, total, ticket)
    cached = _orders_kb_cache.get(key)
    if cached is not None:
        return cached

    kb = InlineKeyboardMarkup(row_width=3)
    row: list[InlineKeyboardButton] = []

    if index > 0:
        row.append(InlineKeyboardButton("⬅️ Пеш", callback_data=f"orders:nav:{index-1}"))
    row.append(InlineKeyboardButton(f"{index+1}/{total}", callback_data="noop"))
    if index < total - 1:
        row.append(InlineKeyboardButton("Баъд ➡️", callback_data=f"orders:nav:{index+1}"))

    if row:
        kb.row(*row)
    kb.row(InlineKeyboardButton("❌ Бастани ин ордер", callback_data=f"orders:close:{ticket}:{index}"))
    kb.row(InlineKeyboardButton("🔒 Пӯшидан", callback_data="orders:close_view"))

    _orders_kb_cache.set(key, kb)
    return kb


def _format_full_report(report: Dict[str, Any], period_name: str) -> str:
    """Формати паймонаи ҳисобот барои бот."""
    try:
        date_from = report.get("date_from", "")
        date_to = report.get("date_to", "")
        date_str = report.get("date", "")

        if not date_from and period_name == "Имрӯза":
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_from = datetime.now().strftime("%Y-%m-%d 00:00:00")
            date_to = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        period_map = {
            "Имрӯза": "📊 ИМРӮЗА",
            "Ҳафтаина": "📊 ҲАФТАИНА",
            "Моҳона": "📊 МОҲОНА",
            "Пурра (Аз ибтидо)": "📊 АЗ ОҒОЗ",
        }
        title = period_map.get(period_name, f"📊 {period_name.upper()}")

        text = f"<b>{title}</b>\n"

        if date_from and date_to:
            try:
                if period_name == "Пурра (Аз ибтидо)":
                    df = date_from.split()[0] if " " in date_from else date_from
                    dt = date_to.split()[0] if " " in date_to else date_to
                    text += f"<b>{df}</b> → <b>{dt}</b>\n"
                else:
                    df = date_from.split()[0] if " " in date_from else date_from
                    dt_time = date_to.split()[1] if " " in date_to else ""
                    dt_date = date_to.split()[0] if " " in date_to else date_to
                    if dt_time:
                        df_time = date_from.split()[1] if " " in date_from else ""
                        text += f"<b>{df}</b> <code>{df_time}</code> → <b>{dt_date}</b> <code>{dt_time}</code>\n"
                    else:
                        text += f"<b>{df}</b> → <b>{dt_date}</b>\n"
            except Exception:
                text += f"<b>{date_from}</b> → <b>{date_to}</b>\n"
        elif date_str:
            text += f"<b>{date_str}</b>\n"

        total_closed = int(report.get("total_closed", 0) or 0)
        total_open = int(report.get("total_open", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        losses = int(report.get("losses", 0) or 0)
        profit = float(report.get("profit", 0.0) or 0.0)
        loss = float(report.get("loss", 0.0) or 0.0)
        net_pnl = float(report.get("net", 0.0) or 0.0)
        unrealized_pnl = float(report.get("unrealized_pnl", 0.0) or 0.0)

        if total_closed > 0:
            pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            profit_factor = profit / loss if loss > 0 else (profit if profit > 0 else 0.0)

            text += (
                f"\n{pnl_emoji} <b>P&L: {net_pnl:+.2f}$</b>\n"
                f"📊 Бурд: <b>{wins}</b> | Бохт: <b>{losses}</b> | WR: <b>{win_rate:.1f}%</b>"
            )
            if profit_factor > 0:
                text += f" | PF: <b>{profit_factor:.2f}</b>"
            text += f"\n💹 Фоида: +{profit:.2f}$ | 📉 Зиён: -{loss:.2f}$\n"
        else:
            text += "\n🚫 Ордерҳои басташуда: 0\n"

        if total_open > 0:
            text += f"🔓 Ордерҳои кушода: <b>{total_open}</b> | P&L: <b>{unrealized_pnl:+.2f}$</b>\n"

        return text
    except Exception as exc:
        return f"⚠️ <code>{exc}</code>"


def check_full_program() -> tuple[bool, str]:
    issues: list[str] = []

    # 1. MT5 Connectivity
    ok_mt5, last_reason = mt5_status()
    reason_s = str(last_reason or "номаълум")
    if ok_mt5:
        try:
            with MT5_LOCK:
                acc = mt5.account_info()
            if not acc:
                issues.append("Маълумоти ҳисоби MT5 дастрас нест.")
        except Exception as exc:
            issues.append(f"Пайвастшавӣ ба MT5 ноком шуд: {exc}")
    else:
        # Fast diagnostic mode: do not trigger reconnect attempts from chat command.
        issues.append(f"Пайвастшавӣ ба MT5 ноком шуд: {reason_s}")

    # 2. Check Portfolio Engine Pipelines
    xau_pipe = getattr(engine, "_xau", None)
    btc_pipe = getattr(engine, "_btc", None)
    gate_reason = str(getattr(engine, "_gate_last_reason", "") or "")
    blocked_assets: list[str] = []
    pending_count = 0
    queue_backlog = 0
    state_lock = getattr(engine, "_order_state_lock", None)
    try:
        if state_lock:
            with state_lock:
                blocked_assets = sorted(set(getattr(engine, "_blocked_assets", []) or []))
                pending_count = len(getattr(engine, "_pending_order_meta", {}) or {})
        else:
            blocked_assets = sorted(set(getattr(engine, "_blocked_assets", []) or []))
            pending_count = len(getattr(engine, "_pending_order_meta", {}) or {})
    except Exception:
        blocked_assets = sorted(set(getattr(engine, "_blocked_assets", []) or []))
        pending_count = 0
    try:
        queue_backlog = int(getattr(getattr(engine, "_order_q", None), "qsize", lambda: 0)() or 0)
    except Exception:
        queue_backlog = 0

    if not bool(getattr(engine, "_model_loaded", False)) or not bool(getattr(engine, "_backtest_passed", False)):
        issues.append(f"Gatekeeper live-ready нест: {gate_reason or 'unknown'}")
    if blocked_assets:
        issues.append(f"Asset gate block: {', '.join(blocked_assets)}")
    if pending_count > 25:
        issues.append(f"Навбати order sync калон аст: pending={pending_count}")
    if queue_backlog > 20:
        issues.append(f"Навбати execution калон аст: queue={queue_backlog}")

    if not xau_pipe or not btc_pipe:
        issues.append("Модулҳои портфели XAU/BTC ҳанӯз оғоз нашудаанд.")
    else:
        # XAU: игнорируем ошибки в выходные дни (суббота/воскресенье)
        if not xau_pipe.last_market_ok:
            if market_is_open("XAU"):
                reason = str(xau_pipe.last_market_reason or "")
                issues.append(f"Хатои маълумоти бозори XAU: {reason}")

        # BTC: always expected (24/7)
        if not btc_pipe.last_market_ok:
            issues.append(f"Хатои маълумоти бозори BTC: {btc_pipe.last_market_reason}")

        try:
            if xau_pipe.risk:
                xau_pipe.risk.evaluate_account_state()
        except Exception as e:
            issues.append(f"Хатои ҳисобкунии риск барои XAU: {e}")

        try:
            if btc_pipe.risk:
                btc_pipe.risk.evaluate_account_state()
        except Exception as e:
            issues.append(f"Хатои ҳисобкунии риск барои BTC: {e}")

    telemetry = ""
    try:
        telemetry = build_health_ribbon(engine.status(), compact=False)
    except Exception:
        telemetry = ""

    if issues:
        summary = "⚠️ <b>Мушкилот ёфт шуд</b>:\n" + "\n".join(f"• {i}" for i in issues)
        return False, summary + ("\n" + telemetry if telemetry else "")

    ok_note = "✅ <b>Санҷиш анҷом ёфт</b>\nҲамаи модулҳо (XAU + BTC) дуруст фаъоланд."
    return True, ok_note + ("\n" + telemetry if telemetry else "")

