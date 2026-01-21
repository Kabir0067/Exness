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
from DataFeed.xau_market_feed import MarketFeed
from ExnessAPI.history import (
    view_all_history_dict,
    get_day_loss,
    get_day_profit,
    get_month_loss,
    get_month_profit,
    get_week_loss,
    get_week_profit,
    format_usdt,
    
)
from ExnessAPI.functions import (
    close_all_position,
    get_balance,
    get_order_by_index,
    get_positions_summary,
    close_order,
    set_takeprofit_all_positions_usd,
    set_stoploss_all_positions_usd,
)
from Bot.portfolio_engine import engine
from StrategiesXau.indicators import Classic_FeatureEngine
from StrategiesXau.risk_management import RiskManager
from StrategiesXau.signal_engine import SignalEngine
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
            if attempt == 1 or attempt % 3 == 0:
                log.warning("TG retry fn=%s attempt=%d/%d err=%s sleep=%.1fs",
                            getattr(fn, "__name__", "call"), attempt, max_retries, exc, d)
            time.sleep(d)
    return None

# Patch critical bot methods ONCE (keeps your old code calls working)
_orig_send_message = bot.send_message
_orig_edit_message_text = bot.edit_message_text
_orig_answer_callback_query = bot.answer_callback_query
_orig_send_chat_action = bot.send_chat_action
_orig_set_my_commands = bot.set_my_commands

def _safe_send_message(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("send_message", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None
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


def _fmt_price(val: float) -> str:
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def _notify_order_opened(intent: Any, result: Any) -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        if not getattr(result, "ok", False):
            return

        sl = float(getattr(intent, "sl", 0.0) or 0.0)
        tp = float(getattr(intent, "tp", 0.0) or 0.0)
        conf = float(getattr(intent, "confidence", 0.0) or 0.0)
        conf_pct = max(0.0, min(1.0, conf)) * 100.0

        sltp = f"{_fmt_price(sl)} / {_fmt_price(tp)}" if (sl > 0 and tp > 0) else "-"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = (
            "✅ <b>Ордер кушода шуд</b>\n"
            f"Ассет: <b>{intent.asset}</b> | Символ: <b>{intent.symbol}</b>\n"
            f"Самт: <b>{intent.signal}</b>\n"
            f"Лот: <b>{float(intent.lot):.4f}</b>\n"
            f"Нарх: <b>{_fmt_price(getattr(result, 'exec_price', 0.0))}</b>\n"
            f"SL/TP: <b>{sltp}</b>\n"
            f"Дақиқӣ: <b>{conf_pct:.1f}%</b>\n"
            f"ID: <code>{intent.order_id}</code>\n"
            f"Вақт: {ts}"
        )
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_order_notifier(_notify_order_opened)

def _notify_phase_change(asset: str, old_phase: str, new_phase: str, reason: str = "") -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason_line = f"Сабаб: <b>{reason}</b>\n" if reason else ""
        msg = (
            "🔔 <b>Тағйири режим</b>\n"
            f"Ассет: <b>{asset}</b>\n"
            f"Аз <b>{old_phase}</b> → <b>{new_phase}</b>\n"
            f"{reason_line}"
            f"Вақт: {ts}"
        )
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_phase_notifier(_notify_phase_change)

def _notify_engine_stopped(asset: str, reason: str = "") -> None:
    try:
        if not is_admin_chat(ADMIN):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason_line = f"Сабаб: <b>{reason}</b>\n" if reason else ""
        msg = (
            "🛑 <b>Трейд стоп шуд</b>\n"
            f"Ассет: <b>{asset}</b>\n"
            f"{reason_line}"
            "Агар хоҳед, метавонед аз нав оғоз кунед.\n"
            f"Вақт: {ts}"
        )
        bot.send_message(ADMIN, msg, parse_mode="HTML")
    except Exception:
        return


engine.set_engine_stop_notifier(_notify_engine_stopped)

def deny(message: types.Message) -> None:
    bot.send_message(
        message.chat.id,
        "❌ Шумо ҳуқуқи истифодабарии ин ботро надоред.",
        reply_markup=types.ReplyKeyboardRemove(),
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
# Bounded TTL cache (no leaks; caches None too)
# =============================================================================
class TTLCache:
    def __init__(self, *, maxsize: int, ttl_sec: float) -> None:
        self.maxsize = int(maxsize)
        self.ttl = float(ttl_sec)
        self._d: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Any) -> Any:
        now = time.time()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            ts, val = item
            if (now - ts) > self.ttl:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return val

    def set(self, key: Any, val: Any) -> None:
        now = time.time()
        with self._lock:
            self._d[key] = (now, val)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

# Cache of chats that blocked the bot to avoid repeated 403 spam.
_blocked_chat_cache = TTLCache(maxsize=512, ttl_sec=12 * 3600)

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



# =============================================================================
# Telemetry / Correlation (safe + cached; prevents rate limit storms)
# =============================================================================
_corr_cache = TTLCache(maxsize=8, ttl_sec=float(getattr(cfg, "correlation_refresh_sec", 60) or 60))

def fetch_external_correlation(cfg_obj: Any) -> Optional[float]:
    if not bool(getattr(cfg_obj, "enable_telemetry", False)):
        return None

    vs = str(getattr(cfg_obj, "correlation_vs_currency", "usd") or "usd").lower()
    key = ("coingecko", "pax-gold", vs)

    cached = _corr_cache.get(key)
    if cached is not None or key in _corr_cache._d:
        return cached

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
        # Portfolio engine status fields
        active_str = str(getattr(status, 'active_asset', 'NONE'))
        if active_str == "NONE" and getattr(status, 'trading', False):
            active_str = "SCANNING"

        segments: list[str] = [
            f"DD {float(getattr(status, 'dd_pct', 0.0)) * 100:.1f}%",
            f"PnL {float(getattr(status, 'today_pnl', 0.0)):+.2f}",
            f"Mode {active_str}",
            f"XAU {int(getattr(status, 'open_trades_xau', 0))}",
            f"BTC {int(getattr(status, 'open_trades_btc', 0))}",
        ]

        # Add last signals for both assets
        last_xau = str(getattr(status, 'last_signal_xau', 'Neutral'))
        last_btc = str(getattr(status, 'last_signal_btc', 'Neutral'))
        segments.append(f"Sig XAU:{last_xau} BTC:{last_btc}")

        # Portfolio engine doesn't expose _risk/_feed directly
        # These are internal to each asset pipeline

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
    active_label = str(getattr(status, 'active_asset', 'NONE'))
    if active_label == "NONE" and getattr(status, 'trading', False):
        active_label = "✅ SCANNING (XAU + BTC)"

    return (
        "⚙️ Статуси Portfolio Bot (XAU + BTC)\n"
        f"- 🔗 Пайваст ба MT5: {'✅' if getattr(status, 'connected', False) else '❌'}\n"
        f"- 📈 Тиҷорат фаъол аст: {'✅' if getattr(status, 'trading', False) else '❌'}\n"
        f"- ⛔ Реҷаи дастӣ: {'✅' if getattr(status, 'manual_stop', False) else '❌'}\n"
        f"- 🎯 Реҷаи ҷорӣ: {active_label}\n"
        f"- 💰 Баланс: {float(getattr(status, 'balance', 0.0)):.2f}$\n"
        f"- 📊 Арзиш: {float(getattr(status, 'equity', 0.0)):.2f}$\n"
        f"- 📉 Коҳиш: {float(getattr(status, 'dd_pct', 0.0)):.2%}\n"
        f"- 📆 Фоида/Зарари Имрӯза: {float(getattr(status, 'today_pnl', 0.0)):+.2f}$\n"
        f"- 📂 Позицияҳои XAU: {int(getattr(status, 'open_trades_xau', 0))}\n"
        f"- 📂 Позицияҳои BTC: {int(getattr(status, 'open_trades_btc', 0))}\n"
        f"- 📊 Ҷамъи позицияҳо: {int(getattr(status, 'open_trades_total', 0))}\n"
        f"- 🛎 Сигнали XAU: {str(getattr(status, 'last_signal_xau', 'Neutral'))}\n"
        f"- 🛎 Сигнали BTC: {str(getattr(status, 'last_signal_btc', 'Neutral'))}\n"
        f"- 🎲 Охирин интихоб: {str(getattr(status, 'last_selected_asset', 'NONE'))}\n"
        f"- 📥 Навбати иҷро: {int(getattr(status, 'exec_queue_size', 0))}\n"
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
BTN_LOSS_D = "📉 Зарари Имрӯза"
BTN_LOSS_W = "📉 Зарари Ҳафтаина"
BTN_LOSS_M = "📉 Зарари Моҳона"
BTN_BALANCE = "💰 Баланс"
BTN_POS = "📊 Хулосаи Позицияҳо"
BTN_DAILY = "📋 Хулосаи руз"
BTN_ENGINE = "🔍 Санҷиши Муҳаррик"
BTN_FULL = "🛠 Санҷиши Пурраи Барнома"

def buttons_func(message: types.Message) -> None:
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row(KeyboardButton(BTN_START), KeyboardButton(BTN_STOP))
    markup.row(KeyboardButton(BTN_CLOSE_ALL), KeyboardButton(BTN_OPEN_ORDERS))
    markup.row(KeyboardButton(BTN_BALANCE), KeyboardButton(BTN_POS))
    markup.row(KeyboardButton(BTN_DAILY))
    markup.row(KeyboardButton(BTN_ENGINE), KeyboardButton(BTN_FULL))
    markup.row(KeyboardButton(BTN_PROFIT_D), KeyboardButton(BTN_PROFIT_W), KeyboardButton(BTN_PROFIT_M))
    markup.row(KeyboardButton(BTN_LOSS_D), KeyboardButton(BTN_LOSS_W), KeyboardButton(BTN_LOSS_M))

    bot.send_message(message.chat.id, "📋 Менюи Асосӣ: Як амалиётро интихоб кунед ⬇️", reply_markup=markup)

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

    status = "✅ ИҶРО ШУД" if ok else "⚠️ ҚИСМАН / ХАТО"
    lines = [
        f"{status}",
        f"🎯 TP барои ҳамаи позицияҳо: {usd:.0f}$",
        "",
        f"📌 Ҳамагӣ позицияҳо: {total}",
        f"✅ Навсозӣ шуд: {updated}",
        f"⏭️ Skip: {skipped}",
    ]

    if errors:
        preview = "\n".join(f"• {e}" for e in errors[:10])
        lines += ["", "🧾 Хатоҳо (10-тои аввал):", preview]

    return "\n".join(lines)

@bot.message_handler(commands=["tek_prof"])
@admin_only_message
def tek_profit_put(message):
    kb = _build_tp_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🎛 *Take Profit (USD)* интихоб кунед (барои *ҳамаи* позицияҳои кушода):",
        reply_markup=kb,
        parse_mode="Markdown",
    )

@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith(TP_CALLBACK_PREFIX))
@admin_only_callback
def on_tp_usd_click(call):
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

        bot.answer_callback_query(call.id, f"⏳ Гузоштани TP={usd:.0f}$ ...")
        res = set_takeprofit_all_positions_usd(usd_profit=usd)

        # Update message text (clean UX)
        text = _format_tp_result(usd, res)
        try:
            bot.edit_message_text(
                text,
                call.message.chat.id,
                call.message.message_id,
                reply_markup=None,
            )
        except Exception:
            bot.send_message(call.message.chat.id, text)

    except Exception as exc:
        bot.answer_callback_query(call.id, "Хато дар обработчик", show_alert=True)
        bot.send_message(call.message.chat.id, f"Handler error: {exc}")



# =============================================================================
# SL (USD) — interactive keyboard (1..10$)
# =============================================================================
def _build_sl_usd_keyboard(min_usd: int = SL_USD_MIN, max_usd: int = SL_USD_MAX, row_width: int = 5):
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

    status = "✅ ИҶРО ШУД" if ok else "⚠️ ҚИСМАН / ХАТО"
    lines = [
        f"{status}",
        f"🛡 SL барои ҳамаи позицияҳо: {usd:.0f}$",
        "",
        f"📌 Ҳамагӣ позицияҳо: {total}",
        f"✅ Навсозӣ шуд: {updated}",
        f"⏭️ Skip: {skipped}",
    ]
    if errors:
        preview = "\n".join(f"• {e}" for e in errors[:10])
        lines += ["", "🧾 Хатоҳо (10-тои аввал):", preview]
    return "\n".join(lines)

@bot.message_handler(commands=["stop_ls"])
@admin_only_message
def tek_stoploss_put(message):
    kb = _build_sl_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🛡 *Stop Loss (USD)* интихоб кунед (барои *ҳамаи* позицияҳои кушода, 1..10$):",
        reply_markup=kb,
        parse_mode="Markdown",
    )

@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith(SL_CALLBACK_PREFIX))
@admin_only_callback
def on_sl_usd_click(call):
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

        bot.answer_callback_query(call.id, f"⏳ Гузоштани SL={usd:.0f}$ ...")
        res = set_stoploss_all_positions_usd(usd_loss=usd)

        text = _format_sl_result(usd, res)
        try:
            bot.edit_message_text(text, call.message.chat.id, call.message.message_id, reply_markup=None)
        except Exception:
            bot.send_message(call.message.chat.id, text)

    except Exception as exc:
        bot.answer_callback_query(call.id, "Хато дар обработчик", show_alert=True)
        bot.send_message(call.message.chat.id, f"Handler error: {exc}")



# =============================================================================
# Daily summary (single source; no duplication)
# =============================================================================
_summary_cache = TTLCache(maxsize=4, ttl_sec=3.0)

def _build_daily_summary_text(summary: Dict[str, Any]) -> str:
    text = (
        "📜 <b>Ҳисоботи Имрӯза</b>\n"
        f"📅 Рӯз: <code>{summary.get('date', '-')}</code>\n"
        "———————————————\n"
    )

    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed > 0:
        text += (
            f"✅ Ордерҳои Бурд: <b>{int(summary.get('wins', 0) or 0)}</b>\n"
            f"❌ Ордерҳои Бохт: <b>{int(summary.get('losses', 0) or 0)}</b>\n"
            f"📋 Ҷамъ Ордерҳои Басташуда: <b>{total_closed}</b>\n\n"
            f"💹 Фоидаи Басташуда: <b>{float(summary.get('profit', 0.0) or 0.0):.2f}$</b>\n"
            f"📉 Зарари Басташуда: <b>{float(summary.get('loss', 0.0) or 0.0):.2f}$</b>\n"
            f"📊 Нетто P&L (баста): <b>{float(summary.get('net', 0.0) or 0.0):.2f}$</b>\n\n"
        )
    else:
        text += "📋 Ордерҳои басташуда: <b>0</b>\n\n"

    if total_open > 0:
        text += (
            f"🔓 Ордерҳои Кушода: <b>{total_open}</b>\n"
            f"💰 P&L Нореалӣ: <b>{float(summary.get('unrealized_pnl', 0.0) or 0.0):.2f}$</b>\n\n"
        )

    text += f"💰 Баланси Ҳозира: <b>{float(summary.get('balance', 0.0) or 0.0):.2f}$</b>\n"
    text += "———————————————\n"
    return text

def send_daily_summary(chat_id: int, *, force_refresh: bool = True) -> None:
    bot.send_chat_action(chat_id, "typing")

    cache_key = ("daily", chat_id)
    cached = _summary_cache.get(cache_key)
    if cached is not None:
        bot.send_message(chat_id, cached, parse_mode="HTML")
        return

    summary = view_all_history_dict(force_refresh=force_refresh)
    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed == 0 and total_open == 0:
        bot.send_message(chat_id, "📅 Имрӯз ҳеҷ ордер (кушода ё баста) вуҷуд надорад.", parse_mode="HTML")
        return

    text = _build_daily_summary_text(summary)
    _summary_cache.set(cache_key, text)
    bot.send_message(chat_id, text, parse_mode="HTML")



# =============================================================================
# /start /history /balance /buttons /status
# =============================================================================
@bot.message_handler(commands=["start"])
def start_handler(message: types.Message) -> None:
    if not is_admin_chat(message.chat.id):
        deny(message)
        return
    bot.send_message(message.chat.id, "👋 Хуш омадед ба бот! Барои идома тугмаҳои зерро истифода баред.")
    buttons_func(message)

@bot.message_handler(commands=["history"])
@admin_only_message
def history_handler(message: types.Message) -> None:
    send_daily_summary(message.chat.id, force_refresh=True)

@bot.message_handler(commands=["balance"])
@admin_only_message
def balance_handler(message: types.Message) -> None:
    bal = get_balance()
    if bal is None:
        bot.send_message(message.chat.id, "⚠️ Хатогӣ ҳангоми гирифтани баланс рӯй дод.")
        return
    bot.send_message(message.chat.id, f"💰 Баланс:\n {format_usdt(bal)}", parse_mode="HTML")

@bot.message_handler(commands=["buttons"])
@admin_only_message
def buttons_handler(message: types.Message) -> None:
    buttons_func(message)

@bot.message_handler(commands=["status"])
@admin_only_message
def status_handler(message: types.Message) -> None:
    try:
        status = engine.status()
        ribbon = build_health_ribbon(status)
        bot.send_message(message.chat.id, _format_status_message(status) + ribbon, parse_mode="HTML")
    except Exception as exc:
        log.error("/status handler error: %s | tb=%s", exc, traceback.format_exc())
        bot.send_message(
            message.chat.id,
            "⚠️ Ҳангоми дархости статус мушкил пеш омад. Эҳтимолан пайвастшавӣ ба MT5 вуҷуд надорад.",
            parse_mode="HTML",
        )



# =============================================================================
# Open orders (format + keyboard)
# =============================================================================
_orders_kb_cache = TTLCache(maxsize=256, ttl_sec=120.0)

def format_order(order_data: Dict[str, Any]) -> str:
    direction = "🟢 BUY" if order_data.get("type") == "BUY" else "🔴 SELL"
    profit = float(order_data.get("profit", 0.0) or 0.0)
    profit_sign = "+" if profit >= 0 else ""
    return (
        f"<b>🎫 Ордери Кушода</b>\n\n"
        f"🔹 <b>Ticket:</b> <code>{order_data.get('ticket', '-')}</code>\n"
        f"🔹 <b>Символ:</b> <code>{order_data.get('symbol', '-')}</code>\n"
        f"🔹 <b>Самт:</b> {direction}\n"
        f"🔹 <b>Ҳаҷм:</b> <code>{float(order_data.get('volume', 0.0) or 0.0):.2f}</code>\n"
        f"🔹 <b>Нархи Кушодан:</b> <code>{float(order_data.get('price', 0.0) or 0.0):.5f}</code>\n"
        f"🔹 <b>P&L (нореалӣ):</b> <code>{profit_sign}{profit:.2f}$</code>"
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

    bot.send_chat_action(message.chat.id, "typing")
    order_data, total = get_order_by_index(0)

    if not order_data or int(total or 0) == 0:
        bot.send_message(message.chat.id, "📭 Ҳоло ордерҳои кушода вуҷуд надоранд.")
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
    bot.answer_callback_query(call.id, "✅ Ордер баста шуд." if ok else "❌ Хатогӣ ҳангоми бастан.")

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
            text="📭 Ҳама ордерҳои кушода баста шуданд.",
            parse_mode="HTML",
        )


@callback_route(r"^orders:close_view$")
def cb_orders_close_view(call: types.CallbackQuery, m: re.Match[str]) -> None:
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="🔒 Намоиши ордерҳои кушода пӯшида шуд.\nБарои дидани дубора тугмаи '📋 Дидани Ордерҳои Кушода'-ро пахш кунед.",
        parse_mode="HTML",
    )
    bot.answer_callback_query(call.id, "Намоиш пӯшида шуд.")



# =============================================================================
# Button dispatcher (maintainable; no huge if-elif)
# =============================================================================
def handle_profit_day(message: types.Message) -> None:
    profit = get_day_profit()
    bot.send_message(
        message.chat.id,
        f"📈 Фоидаи Имрӯза:\n {format_usdt(profit)}\n\n🕒 {datetime.now().strftime('%H:%M')}",
        parse_mode="HTML",
    )

def handle_daily(message: types.Message) -> None:
    send_daily_summary(message.chat.id, force_refresh=True)

def handle_profit_week(message: types.Message) -> None:
    profit = get_week_profit()
    bot.send_message(message.chat.id, f"📊 Фоидаи Ҳафтаина:\n {format_usdt(profit)}", parse_mode="HTML")

def handle_profit_month(message: types.Message) -> None:
    profit = get_month_profit()
    bot.send_message(message.chat.id, f"💹 Фоидаи Моҳона:\n {format_usdt(profit)}", parse_mode="HTML")

def handle_loss_day(message: types.Message) -> None:
    loss = get_day_loss()
    bot.send_message(message.chat.id, f"📉 Зарари Имрӯза:\n {format_usdt(loss)}", parse_mode="HTML")

def handle_loss_week(message: types.Message) -> None:
    loss = get_week_loss()
    bot.send_message(message.chat.id, f"📉 Зарари Ҳафтаина:\n {format_usdt(loss)}", parse_mode="HTML")

def handle_loss_month(message: types.Message) -> None:
    loss = get_month_loss()
    bot.send_message(message.chat.id, f"📉 Зарари Моҳона:\n {format_usdt(loss)}", parse_mode="HTML")

def handle_open_orders(message: types.Message) -> None:
    start_view_open_orders(message)

def handle_close_all(message: types.Message) -> None:
    res = close_all_position()
    lines = [
        "🧹 <b>Натиҷаи «Ҳамаи ордерҳоро бастан»</b>",
        f"✅ Муваффақ: <b>{'ҳа' if res.get('ok') else 'не'}</b>",
        f"🔒 Ордерҳои баста: <b>{int(res.get('closed', 0) or 0)}</b>",
        f"🗑️ Ордерҳои бекоршуда: <b>{int(res.get('canceled', 0) or 0)}</b>",
    ]

    errs = list(res.get('errors') or [])
    if errs:
        err_lines = '\n'.join(f"• {e}" for e in errs)
        lines.append(f"⚠️ Хатогиҳо:\n{err_lines}")
    else:
        lines.append("⚠️ Хатогиҳо: <b>нест</b>")

    last_err = res.get('last_error')
    if last_err:
        lines.append(f"🛠️ last_error: <code>{last_err}</code>")

    bot.send_message(message.chat.id, "\n".join(lines), parse_mode="HTML")

def handle_positions_summary(message: types.Message) -> None:
    summary = get_positions_summary()
    bot.send_message(message.chat.id, f"📊 Хулосаи Позицияҳо:\n {format_usdt(summary)}", parse_mode="HTML")

def handle_balance(message: types.Message) -> None:
    balance = get_balance()
    bot.send_message(message.chat.id, f"💰 Баланс:\n {format_usdt(balance)}", parse_mode="HTML")

def handle_trade_start(message: types.Message) -> None:
    try:
        st = engine.status()
        if bool(getattr(st, "trading", False)) and not bool(getattr(st, "manual_stop", False)):
            bot.send_message(message.chat.id, "ℹ️ Мотор аллакай фаъол аст.")
            return

        if engine.manual_stop_active():
            engine.clear_manual_stop()

        engine.start()

        st_after = engine.status()
        if bool(getattr(st_after, "manual_stop", False)):
            bot.send_message(
                message.chat.id,
                "⚠️ Тиҷорат ҳоло дар реҷаи дастӣ қатъ аст. Аввал аз режими дастӣ бароед.",
            )
        elif bool(getattr(st_after, "trading", False)):
            bot.send_message(message.chat.id, "🚀 Тиҷорат бомуваффақият оғоз шуд! (Мотор фаъол аст)")
        else:
            bot.send_message(message.chat.id, "⚠️ Мотор оғоз нашуд. Лутфан пайвастшавӣ ба MT5-ро санҷед.")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ ҳангоми оғози тиҷорат: {exc}")

def handle_trade_stop(message: types.Message) -> None:
    try:
        st = engine.status()
        was_active = engine.request_manual_stop()
        if was_active:
            bot.send_message(message.chat.id, "🛑 Тиҷорат қатъ карда шуд ва ба реҷаи дастӣ гузашт.")
        elif bool(getattr(st, "manual_stop", False)):
            bot.send_message(message.chat.id, "ℹ️ Реҷаи дастӣ аллакай фаъол аст.")
        else:
            bot.send_message(message.chat.id, "ℹ️ Мотор аллакай қатъ буд.")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ ҳангоми қатъи тиҷорат: {exc}")

def handle_engine_check(message: types.Message) -> None:
    status = engine.status()
    bot.send_message(
        message.chat.id,
        (
            "⚙️ Статуси Муҳаррик\n"
            "———————————————\n"
            f"🔗 Пайваст шудааст: {'✅' if status.connected else '❌'}\n"
            f"📈 Тиҷорат фаъол аст: {'✅' if status.trading else '❌'}\n"
            f"⛔ Реҷаи дастӣ: {'✅' if status.manual_stop else '❌'}\n"
            f"🎯 Активи ҷорӣ: {status.active_asset}\n"
            f"📉 Коҳиш: {status.dd_pct * 100:.2f}%\n"
            f"📆 PnL Имрӯза: {status.today_pnl:+.2f}$\n"
            f"📂 Позицияҳо → XAU: {status.open_trades_xau} | BTC: {status.open_trades_btc}\n"
            f"🛎 Сигналҳо → XAU: {status.last_signal_xau} | BTC: {status.last_signal_btc}\n"
            f"📥 Навбати иҷро: {status.exec_queue_size}\n"
        ),
        parse_mode="HTML",
    )

def handle_full_check(message: types.Message) -> None:
    bot.send_message(message.chat.id, "🔄 Санҷиши пурраи барнома оғоз шуд...")
    ok, detail = check_full_program()
    bot.send_message(message.chat.id, detail, parse_mode="HTML")
    if not ok:
        log.warning("Full check found issues")

BUTTONS: Dict[str, Callable[[types.Message], None]] = {
    BTN_PROFIT_D: handle_profit_day,
    BTN_DAILY: handle_daily,
    BTN_PROFIT_W: handle_profit_week,
    BTN_PROFIT_M: handle_profit_month,
    BTN_LOSS_D: handle_loss_day,
    BTN_LOSS_W: handle_loss_week,
    BTN_LOSS_M: handle_loss_month,
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
            bot.send_message(message.chat.id, "⚠️ Хатогӣ рух дод. Лутфан баъдтар дубора санҷед.")
        return

    bot.send_message(message.chat.id, "❓ Амали номаълум. Лутфан аз менюи асосӣ интихоб кунед.")



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
    # We inspect the live engine instance directly
    xau_pipe = getattr(engine, "_xau", None)
    btc_pipe = getattr(engine, "_btc", None)

    if not xau_pipe or not btc_pipe:
        issues.append("Portfolio Pipelines (XAU/BTC) ҳанӯз сохта нашудаанд (Engine not started?).")
    else:
        # Check XAU
        if not xau_pipe.last_market_ok:
            issues.append(f"XAU Market Data Error: {xau_pipe.last_market_reason}")
        
        # Check BTC
        if not btc_pipe.last_market_ok:
            issues.append(f"BTC Market Data Error: {btc_pipe.last_market_reason}")

        # Check Risk Managers (Basic)
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
        summary = "⚠️ Омор нишон медиҳад, ки мушкилот пайдо шуданд:\n" + "\n".join(f"• {i}" for i in issues)
        return False, summary + ("\n" + telemetry if telemetry else "")

    ok_note = "✅ Санҷиши пурраи барнома анҷом ёфт. Ҳамаи модулҳо (XAU + BTC) ба таври дуруст фаъоланд."
    return True, ok_note + ("\n" + telemetry if telemetry else "")



