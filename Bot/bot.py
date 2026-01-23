from __future__ import annotations

"""Telegram control plane (Exness Portfolio Bot)

Ин файл UI/Control қисми система мебошад.
- Танҳо ADMIN истифода мебарад.
- Engine/Strategy-ро идора мекунад (start/stop/status)
- Амалҳои идоракунӣ: close_all, TP/SL (USD) барои ҳамаи позицияҳо

Эзоҳ:
- Логикаи тиҷорат дар portfolio_engine ва Strategies аст.
- Ин ҷо танҳо Telegram ва даъват ба ExnessAPI/orders.py.
"""

import logging
import re
import socket
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

import MetaTrader5 as mt5
import requests
import telebot
import urllib3
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectTimeout, ReadTimeout, RequestException
from telebot import apihelper, types
from telebot.apihelper import ApiException, ApiTelegramException
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

from config_xau import get_config_from_env
from ExnessAPI.history import (
    view_all_history_dict,
    format_usdt,
)
from ExnessAPI.functions import (
    close_all_position,
    get_balance,
    get_account_info,
    get_order_by_index,
    get_positions_summary,
    close_order,
    set_takeprofit_all_positions_usd,
    set_stoploss_all_positions_usd,
    get_full_report_day,
    get_full_report_week,
    get_full_report_month,
    get_full_report_all,
)
from Bot.portfolio_engine import engine
from mt5_client import ensure_mt5, MT5_LOCK
from log_config import LOG_DIR as LOG_ROOT, get_log_path


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
PAGE_SIZE = 1

TP_USD_MIN = 1
TP_USD_MAX = 10
TP_CALLBACK_PREFIX = "tp_usd:"

SL_USD_MIN = 1
SL_USD_MAX = 10
SL_CALLBACK_PREFIX = "sl_usd:"

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
# Bot instance
# =============================================================================
bot = telebot.TeleBot(cfg.telegram_token)


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


def tg_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 6,
    backoff: Backoff = Backoff(),
    on_permanent_failure: Optional[Callable[[Exception], bool]] = None,
    **kwargs: Any,
) -> Any:
    for attempt in range(1, max_retries + 1):
        try:
            with TG_LOCK:
                return fn(*args, **kwargs)
        except Exception as exc:
            should_retry = _should_retry(exc) and attempt < max_retries
            if not should_retry:
                suppress_log = False
                if on_permanent_failure:
                    try:
                        suppress_log = bool(on_permanent_failure(exc))
                    except Exception as hook_exc:
                        log.error(
                            "TG failure hook error fn=%s err=%s | tb=%s",
                            getattr(fn, "__name__", "call"),
                            hook_exc,
                            traceback.format_exc(),
                        )
                if not suppress_log:
                    log.error(
                        "TG call failed fn=%s err=%s | tb=%s",
                        getattr(fn, "__name__", "call"),
                        exc,
                        traceback.format_exc(),
                    )
                return None

            d = backoff.delay(attempt)
            # log.warning disabled by ERROR level, kept for optional future
            if attempt == 1 or attempt % 3 == 0:
                log.warning(
                    "TG retry fn=%s attempt=%d/%d err=%s sleep=%.1fs",
                    getattr(fn, "__name__", "call"),
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


def _format_datetime(dt: Optional[datetime] = None, show_date: bool = False, show_time: bool = True) -> str:
    """Форматирование даты и времени в красивом HTML формате."""
    if dt is None:
        dt = datetime.now()
    
    parts = []
    if show_date:
        parts.append(f"<b>{dt.strftime('%Y-%m-%d')}</b>")
    if show_time:
        parts.append(f"<code>{dt.strftime('%H:%M:%S')}</code>")
    
    return " ".join(parts) if parts else ""


def _format_time_only() -> str:
    """Только время в красивом формате."""
    return f"<code>{datetime.now().strftime('%H:%M:%S')}</code>"


def _format_date_time() -> str:
    """Дата и время в красивом формате."""
    now = datetime.now()
    return f"<b>{now.strftime('%Y-%m-%d')}</b> <code>{now.strftime('%H:%M:%S')}</code>"


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


def _maybe_send_typing(chat_id: Optional[Any]) -> None:
    # Global policy: every outgoing message triggers typing, but throttled.
    if chat_id is None:
        return
    if _blocked_chat_cache.get(chat_id):
        return
    if _typing_cache.has(chat_id):
        return
    _typing_cache.set(chat_id, True)

    tg_call(
        _orig_send_chat_action,
        chat_id,
        action="typing",
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("send_chat_action", exc, (chat_id,), {"action": "typing"}),
    )


# =============================================================================
# Patch critical bot methods ONCE (keeps your old code calls working)
# + Adds "typing" for every outgoing message/edit
# =============================================================================
_orig_send_message = bot.send_message
_orig_edit_message_text = bot.edit_message_text
_orig_answer_callback_query = bot.answer_callback_query
_orig_send_chat_action = bot.send_chat_action
_orig_set_my_commands = bot.set_my_commands


def _safe_send_message(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("send_message", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None

    # default: no link previews (clean UX)
    kw.setdefault("disable_web_page_preview", True)

    _maybe_send_typing(chat_id)

    return tg_call(
        _orig_send_message,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("send_message", exc, a, kw),
        **kw,
    )


def _safe_edit_message_text(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("edit_message_text", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None

    _maybe_send_typing(chat_id)

    return tg_call(
        _orig_edit_message_text,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("edit_message_text", exc, a, kw),
        **kw,
    )


def _safe_answer_callback_query(*a: Any, **kw: Any) -> Any:
    return tg_call(
        _orig_answer_callback_query,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("answer_callback_query", exc, a, kw),
        **kw,
    )


def _safe_send_chat_action(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("send_chat_action", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None
    return tg_call(
        _orig_send_chat_action,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("send_chat_action", exc, a, kw),
        **kw,
    )


def _safe_set_my_commands(*a: Any, **kw: Any) -> Any:
    return tg_call(
        _orig_set_my_commands,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure("set_my_commands", exc, a, kw),
        **kw,
    )


bot.send_message = _safe_send_message
bot.edit_message_text = _safe_edit_message_text
bot.answer_callback_query = _safe_answer_callback_query
bot.send_chat_action = _safe_send_chat_action
bot.set_my_commands = _safe_set_my_commands


# =============================================================================
# Small utilities (security + maintainability)
# =============================================================================
def is_admin_chat(chat_id: int) -> bool:
    return bool(ADMIN and int(chat_id) == int(ADMIN))


def _rk_remove() -> types.ReplyKeyboardRemove:
    # Only policy: remove reply keyboard only via ReplyKeyboardRemove
    return types.ReplyKeyboardRemove()


def _fmt_price(val: float) -> str:
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def _send_clean(chat_id: int, text: str, *, parse_mode: str = "HTML") -> None:
    # One-shot message that guarantees reply keyboard is removed.
    bot.send_message(chat_id, text, parse_mode=parse_mode, reply_markup=_rk_remove())


def _notify_order_opened(intent: Any, result: Any) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        if not getattr(result, "ok", False):
            return

        sl = float(getattr(intent, "sl", 0.0) or 0.0)
        tp = float(getattr(intent, "tp", 0.0) or 0.0)
        conf = float(getattr(intent, "confidence", 0.0) or 0.0)

        # Normalize confidence to 0-1 range if needed, then cap at 0.96 (96%) to prevent showing 100%
        if conf > 1.0:
            conf = conf / 100.0
        conf = max(0.0, min(0.96, conf))
        conf_pct = conf * 100.0

        sltp = f"{_fmt_price(sl)} / {_fmt_price(tp)}" if (sl > 0 and tp > 0) else "-"
        direction_emoji = "🟢" if str(intent.signal).lower() == "buy" else "🔴"
        currency = "USD"
        time_str = _format_time_only()
        
        msg = (
            f"{direction_emoji} <b>{intent.signal.upper()}</b> | <b>{intent.asset}</b>\n"
            f"📌 {intent.symbol} | #{intent.order_id} | Lot: <b>{float(intent.lot):.4f}</b>\n"
            f"🏷 <b>{_fmt_price(getattr(result, 'exec_price', 0.0))}</b> | 🛡 {sltp}\n"
            f"🧠 <b>{conf_pct:.1f}%</b> | {time_str}"
        )
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_order_notifier(_notify_order_opened)


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
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_phase_notifier(_notify_phase_change)


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
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_engine_stop_notifier(_notify_engine_stopped)


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
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_daily_start_notifier(_notify_daily_start)


def deny(message: types.Message) -> None:
    bot.send_message(
        message.chat.id,
        "❌ Шумо ҳуқуқи истифодабарии ин ботро надоред.",
        reply_markup=_rk_remove(),
    )


def admin_only_message(fn: Callable[[types.Message], None]) -> Callable[[types.Message], None]:
    def wrapper(message: types.Message) -> None:
        if not is_admin_chat(message.chat.id):
            deny(message)
            return
        fn(message)
    return wrapper


def admin_only_callback(fn: Callable[[types.CallbackQuery], None]) -> Callable[[types.CallbackQuery], None]:
    def wrapper(call: types.CallbackQuery) -> None:
        try:
            chat_id = int(call.message.chat.id) if call.message else 0
            user_id = int(call.from_user.id) if call.from_user else 0
        except Exception:
            bot.answer_callback_query(call.id, "❌ Дастрасӣ нест")
            return
        # Strong guard: both chat and user must be ADMIN
        if not (is_admin_chat(chat_id) and ADMIN and user_id == ADMIN):
            bot.answer_callback_query(call.id, "❌ Дастрасӣ нест")
            return
        fn(call)
    return wrapper


# =============================================================================
# Telemetry / Correlation (safe + cached; prevents rate limit storms)
# =============================================================================
_corr_cache = TTLCache(maxsize=8, ttl_sec=float(getattr(cfg, "correlation_refresh_sec", 60) or 60))


def fetch_external_correlation(cfg_obj: Any) -> Optional[float]:
    if not bool(getattr(cfg_obj, "enable_telemetry", False)):
        return None

    vs = str(getattr(cfg_obj, "correlation_vs_currency", "usd") or "usd").lower()
    key = ("coingecko", "pax-gold", vs)

    found, cached = _corr_cache._get_raw(key)
    if found:
        return cached  # may be float or None (cached None)

    url = f"https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies={vs}"
    try:
        resp = _session.get(url, timeout=5, headers={"User-Agent": "xau-bot/1.0"})
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
            active_str = "SCANNING"

        segments: list[str] = [
            f"DD {float(getattr(status, 'dd_pct', 0.0)) * 100:.1f}%",
            f"PnL {float(getattr(status, 'today_pnl', 0.0)):+.2f}",
            f"Mode {active_str}",
            f"XAU {int(getattr(status, 'open_trades_xau', 0))}",
            f"BTC {int(getattr(status, 'open_trades_btc', 0))}",
        ]

        last_xau = str(getattr(status, "last_signal_xau", "Neutral"))
        last_btc = str(getattr(status, "last_signal_btc", "Neutral"))
        segments.append(f"Sig XAU:{last_xau} BTC:{last_btc}")

        corr = fetch_external_correlation(cfg)
        if corr is not None:
            segments.append(f"Corr GOLD {corr:+.2f}")

        ribbon = " | ".join(segments)
        prefix = "\n" if compact else ""
        return f"{prefix}<b>🧭 Мониторинг:</b> {ribbon}"
    except Exception as exc:
        log.error("build_health_ribbon error: %s", exc)
        return ""


def _format_status_message(status: Any) -> str:
    active_label = str(getattr(status, "active_asset", "NONE"))
    if active_label == "NONE" and getattr(status, "trading", False):
        active_label = "✅ SCANNING (XAU + BTC)"

    active_icon = "🟢" if getattr(status, "trading", False) else "🔴"
    trading_status = "ON" if getattr(status, "trading", False) else "OFF"
    mt5_status = "✓" if getattr(status, "connected", False) else "✗"
    balance = float(getattr(status, "balance", 0.0))
    equity = float(getattr(status, "equity", 0.0))
    today_pnl = float(getattr(status, "today_pnl", 0.0))
    dd_pct = float(getattr(status, "dd_pct", 0.0))
    open_xau = int(getattr(status, "open_trades_xau", 0))
    open_btc = int(getattr(status, "open_trades_btc", 0))
    sig_xau = str(getattr(status, "last_signal_xau", "Neutral"))
    sig_btc = str(getattr(status, "last_signal_btc", "Neutral"))
    queue = int(getattr(status, "exec_queue_size", 0))
    
    return (
        f"{active_icon} <b>Система</b> | Trading: <b>{trading_status}</b> | MT5: <b>{mt5_status}</b>\n"
        f"💰 <b>{balance:.2f}$</b> | Equity: <b>{equity:.2f}$</b>"
    ) + (
        f" | PnL: <b>{today_pnl:+.2f}$</b>\n" if today_pnl != 0 else "\n"
    ) + (
        f"📉 DD: <b>{dd_pct:.2%}</b>\n" if dd_pct > 0 else ""
    ) + (
        f"🔹 XAU: <b>{open_xau}</b> | {sig_xau} | "
        f"🔸 BTC: <b>{open_btc}</b> | {sig_btc}"
    ) + (
        f" | 📥 <b>{queue}</b>\n" if queue > 0 else "\n"
    )


# =============================================================================
# Commands
# =============================================================================
def bot_commands() -> None:
    commands = [
        types.BotCommand("/start", "🚀 Барои оғози бот"),
        types.BotCommand("/history", "📜 Дидани таърихи ордерҳо"),
        types.BotCommand("/balance", "💰 Дидани баланси худ"),
        types.BotCommand("/buttons", "🎛️ Тугмаҳои асосӣ"),
        types.BotCommand("/status", "⚙️ Статус оператсия"),
        types.BotCommand("/tek_prof", "💰 Гузоштани тек профит"),
        types.BotCommand("/stop_ls", "🛡 SL: гузоштан (USD 1..10)"),
    ]
    ok = bot.set_my_commands(commands)
    if not ok:
        log.warning("set_my_commands failed (non-fatal)")


# =============================================================================
# Menu
# =============================================================================
BTN_START = "🚀 Оғози Тиҷорат"
BTN_STOP = "🛑 Қатъи Тиҷорат"
BTN_CLOSE_ALL = "❌ Баста кардани Ҳама Ордерҳо"
BTN_OPEN_ORDERS = "📋 Дидани Ордерҳои Кушода"
BTN_PROFIT_D = "📈 Фоидаи Имрӯза"
BTN_PROFIT_W = "📊 Фоидаи Ҳафтаина"
BTN_PROFIT_M = "💹 Фоидаи Моҳона"
BTN_BALANCE = "💰 Баланс"
BTN_POS = "📊 Хулосаи Позицияҳо"
BTN_ENGINE = "🔍 Санҷиши Муҳаррик"
BTN_FULL = "🛠 Санҷиши Пурраи Барнома"


def buttons_func(message: types.Message) -> None:
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row(KeyboardButton(BTN_START), KeyboardButton(BTN_STOP))
    markup.row(KeyboardButton(BTN_CLOSE_ALL), KeyboardButton(BTN_OPEN_ORDERS))
    markup.row(KeyboardButton(BTN_BALANCE), KeyboardButton(BTN_POS))
    markup.row(KeyboardButton(BTN_ENGINE), KeyboardButton(BTN_FULL))
    markup.row(KeyboardButton(BTN_PROFIT_D), KeyboardButton(BTN_PROFIT_W), KeyboardButton(BTN_PROFIT_M))

    bot.send_message(
        message.chat.id,
        "🎛 <b>Бот Control Panel</b>\nЛутфан амалиётро интихоб кунед ⬇️",
        reply_markup=markup,
        parse_mode="HTML",
    )


def _build_tp_usd_keyboard(min_usd: int = TP_USD_MIN, max_usd: int = TP_USD_MAX, row_width: int = 5) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=row_width)
    kb.add(*[
        InlineKeyboardButton(text=f"{i}$", callback_data=f"{TP_CALLBACK_PREFIX}{i}")
        for i in range(int(min_usd), int(max_usd) + 1)
    ])
    kb.add(InlineKeyboardButton(text="❌ Бекор", callback_data=f"{TP_CALLBACK_PREFIX}cancel"))
    return kb


def _format_tp_result(usd: float, res: dict) -> str:
    total = int(res.get("total", 0) or 0)
    updated = int(res.get("updated", 0) or 0)
    skipped = int(res.get("skipped", 0) or 0)
    ok = bool(res.get("ok", False))
    errors = res.get("errors") or []

    status_emoji = "✅" if ok else "⚠️"
    lines = [
        f"{status_emoji} <b>TP: {usd:.0f}$</b> | Навсозӣ: <b>{updated}/{total}</b>"
    ]
    if skipped > 0:
        lines.append(f"⏭️ Skip: <b>{skipped}</b>")
    if errors:
        preview = " | ".join(e[:30] for e in errors[:3])
        lines.append(f"⚠️ <code>{preview}</code>")
    return "\n".join(lines)


@bot.message_handler(commands=["tek_prof"])
@admin_only_message
def tek_profit_put(message: types.Message) -> None:
    # Fix: remove reply keyboard before showing inline keyboard
    _send_clean(message.chat.id, "⌨️ <b>Меню пӯшида шуд</b>\n🎛 Ҳоло TP-ро интихоб мекунем.")
    kb = _build_tp_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🎛 <b>Take Profit (USD)</b>\nБарои <b>ҳамаи позицияҳои кушода</b> интихоб кунед:",
        reply_markup=kb,
        parse_mode="HTML",
    )


@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith(TP_CALLBACK_PREFIX))
@admin_only_callback
def on_tp_usd_click(call: types.CallbackQuery) -> None:
    data = (call.data or "").split(":", 1)[-1].strip().lower()

    if data == "cancel":
        bot.answer_callback_query(call.id, "Бекор шуд")
        try:
            bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
        except Exception:
            pass
        return

    try:
        usd = float(data)
        if not (TP_USD_MIN <= usd <= TP_USD_MAX):
            bot.answer_callback_query(call.id, "Диапазон: 1..10", show_alert=True)
            return

        bot.answer_callback_query(call.id, f"⏳ TP={usd:.0f}$ ...")
        res = set_takeprofit_all_positions_usd(usd_profit=usd)

        text = _format_tp_result(usd, res)
        try:
            bot.edit_message_text(
                text,
                call.message.chat.id,
                call.message.message_id,
                reply_markup=None,
                parse_mode="HTML",
            )
        except Exception:
            bot.send_message(call.message.chat.id, text, parse_mode="HTML")

    except Exception as exc:
        bot.answer_callback_query(call.id, "Хато дар обработчик", show_alert=True)
        bot.send_message(call.message.chat.id, f"⚠️ Handler error: <code>{exc}</code>", parse_mode="HTML")


# =============================================================================
# SL (USD) — interactive keyboard (1..10$)
# =============================================================================
def _build_sl_usd_keyboard(min_usd: int = SL_USD_MIN, max_usd: int = SL_USD_MAX, row_width: int = 5) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=row_width)
    kb.add(*[
        InlineKeyboardButton(text=f"{i}$", callback_data=f"{SL_CALLBACK_PREFIX}{i}")
        for i in range(int(min_usd), int(max_usd) + 1)
    ])
    kb.add(InlineKeyboardButton(text="❌ Бекор", callback_data=f"{SL_CALLBACK_PREFIX}cancel"))
    return kb


def _format_sl_result(usd: float, res: dict) -> str:
    total = int(res.get("total", 0) or 0)
    updated = int(res.get("updated", 0) or 0)
    skipped = int(res.get("skipped", 0) or 0)
    ok = bool(res.get("ok", False))
    errors = res.get("errors") or []

    status_emoji = "✅" if ok else "⚠️"
    lines = [
        f"{status_emoji} <b>SL: {usd:.0f}$</b> | Навсозӣ: <b>{updated}/{total}</b>"
    ]
    if skipped > 0:
        lines.append(f"⏭️ Skip: <b>{skipped}</b>")
    if errors:
        preview = " | ".join(e[:30] for e in errors[:3])
        lines.append(f"⚠️ <code>{preview}</code>")
    return "\n".join(lines)


@bot.message_handler(commands=["stop_ls"])
@admin_only_message
def tek_stoploss_put(message: types.Message) -> None:
    # Fix: remove reply keyboard before showing inline keyboard
    _send_clean(message.chat.id, "⌨️ <b>Меню пӯшида шуд</b>\n🛡 Ҳоло SL-ро интихоб мекунем.")
    kb = _build_sl_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🛡 <b>Stop Loss (USD)</b>\nБарои <b>ҳамаи позицияҳои кушода</b> интихоб кунед (1..10$):",
        reply_markup=kb,
        parse_mode="HTML",
    )


@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith(SL_CALLBACK_PREFIX))
@admin_only_callback
def on_sl_usd_click(call: types.CallbackQuery) -> None:
    data = (call.data or "").split(":", 1)[-1].strip().lower()

    if data == "cancel":
        bot.answer_callback_query(call.id, "Бекор шуд")
        try:
            bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
        except Exception:
            pass
        return

    try:
        usd = float(data)
        if not (SL_USD_MIN <= usd <= SL_USD_MAX):
            bot.answer_callback_query(call.id, "Диапазон: 1..10", show_alert=True)
            return

        bot.answer_callback_query(call.id, f"⏳ SL={usd:.0f}$ ...")
        res = set_stoploss_all_positions_usd(usd_loss=usd)

        text = _format_sl_result(usd, res)
        try:
            bot.edit_message_text(
                text,
                call.message.chat.id,
                call.message.message_id,
                reply_markup=None,
                parse_mode="HTML",
            )
        except Exception:
            bot.send_message(call.message.chat.id, text, parse_mode="HTML")

    except Exception as exc:
        bot.answer_callback_query(call.id, "Хато дар обработчик", show_alert=True)
        bot.send_message(call.message.chat.id, f"⚠️ Handler error: <code>{exc}</code>", parse_mode="HTML")


# =============================================================================
# Daily summary (single source; no duplication)
# =============================================================================
_summary_cache = TTLCache(maxsize=4, ttl_sec=3.0)


def _build_daily_summary_text(summary: Dict[str, Any]) -> str:
    text = (
        "📜 <b>DAILY TRADING SUMMARY</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Date: <b>{summary.get('date', '-')}</b>\n"
    )

    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed > 0:
        net = float(summary.get('net', 0.0) or 0.0)
        wins = int(summary.get('wins', 0) or 0)
        losses = int(summary.get('losses', 0) or 0)
        profit = float(summary.get('profit', 0.0) or 0.0)
        loss = float(summary.get('loss', 0.0) or 0.0)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
        pnl_emoji = "🟢" if net >= 0 else "🔴"
        
        text += (
            f"\n{pnl_emoji} <b>P&L: {net:+.2f}$</b>\n"
            f"📊 {wins}W/{losses}L | WR: <b>{win_rate:.1f}%</b>\n"
            f"💹 +{profit:.2f}$ | 📉 -{loss:.2f}$\n"
        )
    else:
        text += "\n🚫 Ордерҳои басташуда: 0\n"

    if total_open > 0:
        unrealized = float(summary.get('unrealized_pnl', 0.0) or 0.0)
        text += f"🔓 Кушода: <b>{total_open}</b> | P&L: <b>{unrealized:+.2f}$</b>\n"

    balance = float(summary.get('balance', 0.0) or 0.0)
    text += f"\n💰 <b>{balance:.2f}$</b>\n"
    return text


def send_daily_summary(chat_id: int, *, force_refresh: bool = True) -> None:
    cache_key = ("daily", chat_id)

    if not force_refresh:
        cached = _summary_cache.get(cache_key)
        if cached is not None:
            bot.send_message(chat_id, cached, parse_mode="HTML", reply_markup=_rk_remove())
            return
    else:
        _summary_cache.pop(cache_key, None)

    summary = view_all_history_dict(force_refresh=force_refresh)
    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed == 0 and total_open == 0:
        bot.send_message(chat_id, "📅 Имрӯз ҳеҷ ордер (кушода ё баста) вуҷуд надорад.", parse_mode="HTML", reply_markup=_rk_remove())
        return

    text = _build_daily_summary_text(summary)
    _summary_cache.set(cache_key, text)
    bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=_rk_remove())


# =============================================================================
# /start /history /balance /buttons /status
# =============================================================================
@bot.message_handler(commands=["start"])
def start_handler(message: types.Message) -> None:
    if not is_admin_chat(message.chat.id):
        deny(message)
        return
        bot.send_message(
        message.chat.id,
        "👋 <b>Хуш омадед!</b>\nБарои идоракунӣ менюро истифода баред: /buttons",
        parse_mode="HTML",
    )
    buttons_func(message)


@bot.message_handler(commands=["history"])
@admin_only_message
def history_handler(message: types.Message) -> None:
    """
    /history - ҳисоботи пурра + маълумоти аккаунт (сенёр формат)
    """
    try:
        # Remove reply keyboard for clean report reading
        _send_clean(message.chat.id, "📥 <b>Гирифтани ҳисобот...</b>")

        report = get_full_report_all(force_refresh=True)
        acc_info = get_account_info()

        text = _format_full_report(report, "Пурра (Аз ибтидо)")

        # Добавляем детальную информацию об открытых позициях
        open_positions = report.get("open_positions", [])
        if open_positions and len(open_positions) > 0:
            text += "\n<b>Кушода:</b> "
            for i, pos in enumerate(open_positions[:5]):  # Показываем максимум 5
                if i > 0:
                    text += " | "
                ticket = pos.get("ticket", 0)
                symbol = pos.get("symbol", "")
                profit = pos.get("profit", 0.0)
                text += f"#{ticket} {symbol} <b>{profit:+.2f}$</b>"
            if len(open_positions) > 5:
                text += f" | +{len(open_positions) - 5}"
            text += "\n"

        if acc_info:
            login = acc_info.get("login", 0)
            balance = acc_info.get("balance", 0.0)
            equity = acc_info.get("equity", 0.0)
            profit = acc_info.get("profit", 0.0)
            margin_level = acc_info.get("margin_level", 0.0)

            text += f"\n💰 <b>{balance:.2f}$</b> | Equity: <b>{equity:.2f}$</b>"
            if profit != 0:
                text += f" | P&L: <b>{profit:+.2f}$</b>"
            if margin_level:
                text += f" | ML: <b>{margin_level:.1f}%</b>"
            text += "\n"

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        losses = int(report.get("losses", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)
            text += f"📊 WR: <b>{win_rate:.1f}%</b>"
            if profit_factor:
                text += f" | PF: <b>{profit_factor:.2f}</b>"
            text += "\n"

        text += f"\n{_format_time_only()}\n"

        bot.send_message(message.chat.id, text, parse_mode="HTML", reply_markup=_rk_remove())
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            f"⚠️ Хатогӣ ҳангоми гирифтани таърих: <code>{exc}</code>",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )


@bot.message_handler(commands=["balance"])
@admin_only_message
def balance_handler(message: types.Message) -> None:
    bal = get_balance()
    if bal is None:
        bot.send_message(message.chat.id, "⚠️ Хатогӣ ҳангоми гирифтани баланс.", parse_mode="HTML", reply_markup=_rk_remove())
        return
    bot.send_message(message.chat.id, f"💰 <b>Баланс</b>\n{format_usdt(bal)}", parse_mode="HTML", reply_markup=_rk_remove())


@bot.message_handler(commands=["buttons"])
@admin_only_message
def buttons_handler(message: types.Message) -> None:
    buttons_func(message)


@bot.message_handler(commands=["status"])
@admin_only_message
def status_handler(message: types.Message) -> None:
    try:
        # Status is usually read-only -> keep clean, remove reply keyboard
        status = engine.status()
        ribbon = build_health_ribbon(status)
        bot.send_message(message.chat.id, _format_status_message(status) + ribbon, parse_mode="HTML", reply_markup=_rk_remove())
    except Exception as exc:
        log.error("/status handler error: %s | tb=%s", exc, traceback.format_exc())
        bot.send_message(
            message.chat.id,
            "⚠️ Ҳангоми дархости статус мушкил пеш омад. Пайвастшавӣ ба MT5-ро санҷед.",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )


# =============================================================================
# Open orders (format + keyboard)
# =============================================================================
_orders_kb_cache = TTLCache(maxsize=256, ttl_sec=120.0)


def format_order(order_data: Dict[str, Any]) -> str:
    direction_emoji = "🟢" if order_data.get("type") == "BUY" else "🔴"
    direction_text = "BUY" if order_data.get("type") == "BUY" else "SELL"
    ticket = order_data.get('ticket', '-')
    symbol = order_data.get('symbol', '-')
    volume = float(order_data.get('volume', 0.0) or 0.0)
    price = float(order_data.get('price', 0.0) or 0.0)
    profit = float(order_data.get("profit", 0.0) or 0.0)
    profit_emoji = "🟢" if profit >= 0 else "🔴"
    
    return (
        f"{direction_emoji} <b>{direction_text}</b> | <b>{symbol}</b> | #{ticket}\n"
        f"📦 <b>{volume:.2f}</b> | 🏷 <b>{price:.5f}</b> | {profit_emoji} <b>{profit:+.2f}$</b>"
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
    kb.row(InlineKeyboardButton("❌ Бастан ин ордер", callback_data=f"orders:close:{ticket}:{index}"))
    kb.row(InlineKeyboardButton("🔒 Пӯшидан", callback_data="orders:close_view"))

    _orders_kb_cache.set(key, kb)
    return kb


def start_view_open_orders(message: types.Message) -> None:
    if not is_admin_chat(message.chat.id):
        return

    # Clean UX: remove reply keyboard before inline navigation
    _send_clean(message.chat.id, "📋 <b>Ордерҳои кушода</b>")

    order_data, total = get_order_by_index(0)

    if not order_data or int(total or 0) == 0:
        bot.send_message(message.chat.id, "📭 Ордерҳои кушода нестанд.", parse_mode="HTML", reply_markup=_rk_remove())
        return

    text = format_order(order_data)
    kb = order_keyboard(0, int(total), int(order_data.get("ticket", 0) or 0))
    bot.send_message(message.chat.id, text, reply_markup=kb, parse_mode="HTML")


# =============================================================================
# Callback router (no monolith)
# =============================================================================
_CALLBACK_ROUTES: list[tuple[re.Pattern[str], Callable[[types.CallbackQuery, re.Match[str]], None]]] = []


def callback_route(pattern: str) -> Callable[[Callable[[types.CallbackQuery, re.Match[str]], None]], Callable]:
    rx = re.compile(pattern)

    def deco(fn: Callable[[types.CallbackQuery, re.Match[str]], None]) -> Callable:
        _CALLBACK_ROUTES.append((rx, fn))
        return fn

    return deco


@bot.callback_query_handler(func=lambda call: True)
@admin_only_callback
def callback_dispatch(call: types.CallbackQuery) -> None:
    data = str(call.data or "")
    if data == "noop":
        bot.answer_callback_query(call.id)
        return

    for rx, fn in _CALLBACK_ROUTES:
        m = rx.match(data)
        if m:
            try:
                fn(call, m)
            except Exception as exc:
                log.error("Callback error data=%s err=%s | tb=%s", data, exc, traceback.format_exc())
                bot.answer_callback_query(call.id, "❌ Хатогӣ рух дод")
            return

    bot.answer_callback_query(call.id)  # unknown callback -> silent


@callback_route(r"^orders:nav:(\d+)$")
def cb_orders_nav(call: types.CallbackQuery, m: re.Match[str]) -> None:
    idx = int(m.group(1))
    order_data, total = get_order_by_index(idx)

    if not order_data or int(total or 0) == 0:
        bot.answer_callback_query(call.id, "⚠️ Ордер дастрас нест.")
        return

    text = format_order(order_data)
    kb = order_keyboard(idx, int(total), int(order_data.get("ticket", 0) or 0))
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text=text,
        parse_mode="HTML",
        reply_markup=kb,
    )
    bot.answer_callback_query(call.id)


@callback_route(r"^orders:close:(\d+):(\d+)$")
def cb_orders_close(call: types.CallbackQuery, m: re.Match[str]) -> None:
    ticket = int(m.group(1))
    idx = int(m.group(2))

    ok = close_order(ticket)
    bot.answer_callback_query(call.id, "✅ Баста шуд" if ok else "❌ Хатогӣ")

    order_data, total = get_order_by_index(idx)
    if order_data and int(total or 0) > 0:
        text = format_order(order_data)
        kb = order_keyboard(idx, int(total), int(order_data.get("ticket", 0) or 0))
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=text,
            parse_mode="HTML",
            reply_markup=kb,
        )
    else:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="📭 Ордерҳои кушода нестанд.",
            parse_mode="HTML",
        )


@callback_route(r"^orders:close_view$")
def cb_orders_close_view(call: types.CallbackQuery, m: re.Match[str]) -> None:
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
            text="🔒 Намоиш пӯшида шуд. Барои дидани дубора: /buttons",
        parse_mode="HTML",
    )
    bot.answer_callback_query(call.id, "Намоиш пӯшида шуд.")


# =============================================================================
# Button dispatcher (maintainable; no huge if-elif)
# =============================================================================
def _format_full_report(report: Dict[str, Any], period_name: str) -> str:
    """Компактный профессиональный формат отчета."""
    try:
        date_from = report.get("date_from", "")
        date_to = report.get("date_to", "")
        date_str = report.get("date", "")

        if not date_from and period_name == "Имрӯза":
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_from = datetime.now().strftime("%Y-%m-%d 00:00:00")
            date_to = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        period_map = {"Имрӯза": "📊 РӮЗОНА", "Ҳафтаина": "📊 ҲАФТАИНА", "Моҳона": "📊 МОҲОНА", "Пурра (Аз ибтидо)": "📊 ПУРРА"}
        title = period_map.get(period_name, f"📊 {period_name.upper()}")
        
        text = f"<b>{title}</b>\n"

        if date_from and date_to:
            # Форматируем даты красиво
            try:
                if period_name == "Пурра (Аз ибтидо)":
                    # Для полного отчета показываем только даты
                    df = date_from.split()[0] if ' ' in date_from else date_from
                    dt = date_to.split()[0] if ' ' in date_to else date_to
                    text += f"<b>{df}</b> → <b>{dt}</b>\n"
                else:
                    # Для других периодов показываем дату и время
                    df = date_from.split()[0] if ' ' in date_from else date_from
                    dt_time = date_to.split()[1] if ' ' in date_to else ""
                    dt_date = date_to.split()[0] if ' ' in date_to else date_to
                    if dt_time:
                        text += f"<b>{df}</b> <code>{date_from.split()[1] if ' ' in date_from else ''}</code> → <b>{dt_date}</b> <code>{dt_time}</code>\n"
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
                f"📊 {wins}W/{losses}L | WR: <b>{win_rate:.1f}%</b>"
            )
            if profit_factor > 0:
                text += f" | PF: <b>{profit_factor:.2f}</b>"
            text += f"\n💹 +{profit:.2f}$ | 📉 -{loss:.2f}$\n"
        else:
            text += "\n🚫 Ордерҳои басташуда: 0\n"

        if total_open > 0:
            text += f"🔓 Кушода: <b>{total_open}</b> | P&L: <b>{unrealized_pnl:+.2f}$</b>\n"

        return text
    except Exception as exc:
        return f"⚠️ <code>{exc}</code>"


def handle_profit_day(message: types.Message) -> None:
    try:
        report = get_full_report_day(force_refresh=True)
        text = _format_full_report(report, "Имрӯза")
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")


def handle_profit_week(message: types.Message) -> None:
    try:
        report = get_full_report_week(force_refresh=True)
        text = _format_full_report(report, "Ҳафтаина")

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)

            text += f"<b>WR:</b> {win_rate:.1f}%"
            if profit_factor > 0:
                text += f" | <b>PF:</b> {profit_factor:.2f}"
            text += "\n"

        text += f"\n{_format_time_only()}\n"

        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")


def handle_profit_month(message: types.Message) -> None:
    try:
        report = get_full_report_month(force_refresh=True)
        text = _format_full_report(report, "Моҳона")

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)

            text += f"<b>WR:</b> {win_rate:.1f}%"
            if profit_factor > 0:
                text += f" | <b>PF:</b> {profit_factor:.2f}"
            text += "\n"

        text += f"\n{_format_time_only()}\n"

        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")


def handle_open_orders(message: types.Message) -> None:
    start_view_open_orders(message)


def handle_close_all(message: types.Message) -> None:
    res = close_all_position()
    closed = int(res.get('closed', 0) or 0)
    canceled = int(res.get('canceled', 0) or 0)
    ok = res.get('ok', False)
    status_emoji = "✅" if ok else "⚠️"
    
    lines = [
        f"{status_emoji} <b>Баста: {closed}</b>"
    ]
    if canceled > 0:
        lines.append(f"🗑️ Бекор: <b>{canceled}</b>")
    
    errs = list(res.get("errors") or [])
    if errs:
        preview = " | ".join(e[:25] for e in errs[:2])
        lines.append(f"⚠️ <code>{preview}</code>")

    bot.send_message(message.chat.id, "\n".join(lines), parse_mode="HTML")


def handle_positions_summary(message: types.Message) -> None:
    summary = get_positions_summary()
    bot.send_message(message.chat.id, f"📊 <b>{format_usdt(summary)}</b>", parse_mode="HTML")


def handle_balance(message: types.Message) -> None:
    balance = get_balance()
    bot.send_message(message.chat.id, f"💰 <b>Баланс</b>\n{format_usdt(balance)}", parse_mode="HTML")


def handle_trade_start(message: types.Message) -> None:
    try:
        st = engine.status()
        if bool(getattr(st, "trading", False)) and not bool(getattr(st, "manual_stop", False)):
            bot.send_message(message.chat.id, "ℹ️ Система аллакай фаъол аст.", parse_mode="HTML")
            return

        if engine.manual_stop_active():
            engine.clear_manual_stop()

        engine.start()

        st_after = engine.status()
        if bool(getattr(st_after, "manual_stop", False)):
            bot.send_message(message.chat.id, "⚠️ Manual stop фаъол аст. Аввал онро хомӯш кунед.", parse_mode="HTML")
        elif bool(getattr(st_after, "trading", False)):
            bot.send_message(message.chat.id, "🚀 <b>Система оғоз шуд</b> | ✅ Фаъол", parse_mode="HTML")
        else:
            bot.send_message(message.chat.id, "⚠️ Оғоз нашуд. MT5-ро санҷед.", parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")


def handle_trade_stop(message: types.Message) -> None:
    try:
        st = engine.status()
        was_active = engine.request_manual_stop()
        if was_active:
            bot.send_message(message.chat.id, "🛑 <b>Система қатъ шуд</b> | ⛔ Manual stop", parse_mode="HTML")
        elif bool(getattr(st, "manual_stop", False)):
            bot.send_message(message.chat.id, "ℹ️ Manual stop аллакай фаъол аст.", parse_mode="HTML")
        else:
            bot.send_message(message.chat.id, "ℹ️ Система аллакай қатъ аст.", parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")


def handle_engine_check(message: types.Message) -> None:
    status = engine.status()
    bot.send_message(
        message.chat.id,
        (
            "⚙️ <b>Статуси Муҳаррик</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 Пайваст: {'✅' if status.connected else '❌'}\n"
            f"📈 Trading: {'✅' if status.trading else '❌'}\n"
            f"⛔ Manual stop: {'✅' if status.manual_stop else '❌'}\n"
            f"🎯 Актив: <b>{status.active_asset}</b>\n"
            f"📉 DD: <b>{status.dd_pct * 100:.2f}%</b>\n"
            f"📆 Today PnL: <b>{status.today_pnl:+.2f}$</b>\n"
            f"📂 Позицияҳо: XAU <b>{status.open_trades_xau}</b> | BTC <b>{status.open_trades_btc}</b>\n"
            f"🛎 Сигналҳо: XAU <b>{status.last_signal_xau}</b> | BTC <b>{status.last_signal_btc}</b>\n"
            f"📥 Queue: <b>{status.exec_queue_size}</b>\n"
        ),
        parse_mode="HTML",
    )


def handle_full_check(message: types.Message) -> None:
    bot.send_message(message.chat.id, "🔄 <b>Санҷиши пурраи барнома оғоз шуд...</b>", parse_mode="HTML")
    ok, detail = check_full_program()
    bot.send_message(message.chat.id, detail, parse_mode="HTML")
    if not ok:
        log.warning("Full check found issues")


BUTTONS: Dict[str, Callable[[types.Message], None]] = {
    BTN_PROFIT_D: handle_profit_day,
    BTN_PROFIT_W: handle_profit_week,
    BTN_PROFIT_M: handle_profit_month,
    BTN_OPEN_ORDERS: handle_open_orders,
    BTN_CLOSE_ALL: handle_close_all,
    BTN_POS: handle_positions_summary,
    BTN_BALANCE: handle_balance,
    BTN_START: handle_trade_start,
    BTN_STOP: handle_trade_stop,
    BTN_ENGINE: handle_engine_check,
    BTN_FULL: handle_full_check,
}


@bot.message_handler(func=lambda m: True)
def message_dispatcher(message: types.Message) -> None:
    if not is_admin_chat(message.chat.id):
        deny(message)
        return

    text = message.text
    if not isinstance(text, str):
        return
    if text.startswith("/"):
        return

    handler = BUTTONS.get(text)
    if handler:
        try:
            handler(message)
        except Exception as exc:
            log.error("handler error text=%s err=%s | tb=%s", text, exc, traceback.format_exc())
            bot.send_message(message.chat.id, "⚠️ Хатогӣ рух дод. Баъдтар дубора санҷед.", parse_mode="HTML")
        return

    bot.send_message(message.chat.id, "❓ Амали номаълум. /buttons → меню.", parse_mode="HTML")


# =============================================================================
# Full program check (production safe)
# =============================================================================
def check_full_program() -> tuple[bool, str]:
    issues: list[str] = []

    # 1. MT5 Connectivity
    try:
        ensure_mt5()
        with MT5_LOCK:
            acc = mt5.account_info()
        if not acc:
            issues.append("MT5 Account Info дастрас нест.")
    except Exception as exc:
        issues.append(f"MT5 Connection failed: {exc}")

    # 2. Check Portfolio Engine Pipelines
    xau_pipe = getattr(engine, "_xau", None)
    btc_pipe = getattr(engine, "_btc", None)

    if not xau_pipe or not btc_pipe:
        issues.append("Portfolio Pipelines (XAU/BTC) ҳанӯз сохта нашудаанд (Engine not started?).")
    else:
        if not xau_pipe.last_market_ok:
            issues.append(f"XAU Market Data Error: {xau_pipe.last_market_reason}")
        if not btc_pipe.last_market_ok:
            issues.append(f"BTC Market Data Error: {btc_pipe.last_market_reason}")

        try:
            if xau_pipe.risk:
                xau_pipe.risk.evaluate_account_state()
        except Exception as e:
            issues.append(f"XAU Risk Calc Error: {e}")

        try:
            if btc_pipe.risk:
                btc_pipe.risk.evaluate_account_state()
        except Exception as e:
            issues.append(f"BTC Risk Calc Error: {e}")

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
