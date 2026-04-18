"""
Bot/bot.py - Telegram control plane (Exness Portfolio Bot)

Ин файл UI/Control қисми система мебошад.
- Танҳо ADMIN истифода мебарад.
- Engine/Strategy-ро идора мекунад (start/stop/status)
- Амалҳои идоракунӣ: close_all, TP/SL (USD) барои ҳамаи позицияҳо

Логикаи тиҷорат дар portfolio_engine ва Strategies аст.
Ин ҷо танҳо Telegram ва даъватҳо ба ExnessAPI/orders.py.
"""

from __future__ import annotations

import html
import re
import threading
import time
import traceback
from typing import Any, Callable, Dict

import telebot
from telebot.types import KeyboardButton, ReplyKeyboardMarkup

from AiAnalysis.intrd_ai_analys import analyse_intraday
from AiAnalysis.scalp_ai_analys import analyse
from Bot.bot_utils import (
    ADMIN,
    AI_CALLBACK_PREFIX,
    HELPER_CALLBACK_PREFIX,
    HELPER_ORDER_COUNTS,
    SL_CALLBACK_PREFIX,
    SL_USD_MAX,
    SL_USD_MIN,
    TP_CALLBACK_PREFIX,
    TP_USD_MAX,
    TP_USD_MIN,
    _blocked_chat_cache,
    _build_daily_summary_text,
    _build_sl_usd_keyboard,
    _build_tp_usd_keyboard,
    _extract_chat_id_from_call,
    _fmt_price,
    _format_full_report,
    _format_sl_result,
    _format_status_message,
    _format_time_only,
    _format_tp_result,
    _handle_permanent_telegram_failure,
    _maybe_send_typing,
    _notify_daily_start,
    _notify_engine_stopped,
    _notify_order_skipped,
    _notify_order_update,
    _notify_phase_change,
    _notify_signal,
    _rk_remove,
    _send_clean,
    _summary_cache,
    build_ai_keyboard,
    build_health_ribbon,
    build_helper_order_count_keyboard,
    build_helpers_keyboard,
    cfg,
    check_full_program,
    deny,
    engine,
    format_close_by_profit_result,
    format_order,
    is_admin_chat,
    log,
    order_keyboard,
    send_action,
    set_bot_instance,
    set_orig_send_chat_action,
    tg_call,
)
from DataFeed.ai_day_market_feed import (
    get_ai_payload_btc_intraday,
    get_ai_payload_xau_intraday,
)
from DataFeed.ai_scalp_market_feed import get_ai_payload_btc, get_ai_payload_xau
from ExnessAPI.functions import (
    close_all_position,
    close_all_position_by_profit,
    close_order,
    get_account_info,
    get_balance,
    get_full_report_all,
    get_full_report_day,
    get_full_report_month,
    get_full_report_week,
    get_order_by_index,
    get_positions_summary,
    manual_open_capacity,
    open_buy_order_btc,
    open_buy_order_xau,
    open_sell_order_btc,
    open_sell_order_xau,
    set_stoploss_all_positions_usd,
    set_takeprofit_all_positions_usd,
)
from ExnessAPI.history import *  # noqa: F401, F403
from ExnessAPI.history import view_all_history_dict
from mt5_client import mt5_status

# =============================================================================
# Global Constants & Meta Definitions
# =============================================================================
_HANDLER_DEBOUNCE_SEC = 1.5

BTN_START = "🚀 Оғози Тиҷорат"
BTN_STOP = "🛑 Қатъи Тиҷорат"
BTN_CLOSE_ALL = "❌ Бастани ҳама ордерҳо"
BTN_CLOSE_PROFIT = "💰 Бастани фоидадорҳо"
BTN_OPEN_ORDERS = "📋 Дидани Ордерҳои Кушода"
BTN_PROFIT_D = "📈 Фоидаи Имрӯза"
BTN_PROFIT_W = "📊 Фоидаи Ҳафтаина"
BTN_PROFIT_M = "💹 Фоидаи Моҳона"
BTN_BALANCE = "💳 Баланс"
BTN_POS = "📊 Хулосаи Позицияҳо"
BTN_ENGINE = "🔍 Санҷиши Муҳаррик"
BTN_FULL = "🛠 Санҷиши Пурраи мотор"


# =============================================================================
# Global Mutex and State Variables
# =============================================================================
_HANDLER_LAST_TS: Dict[str, float] = {}
_HANDLER_LOCK = threading.Lock()
_CALLBACK_ROUTES = []


bot = telebot.TeleBot(cfg.telegram_token)

# Provide bot reference to low-level notification utilities
set_bot_instance(bot)

# Hook Telegram API core methods for telemetry/UI feedback
_orig_send_message = bot.send_message
_orig_edit_message_text = bot.edit_message_text
_orig_answer_callback_query = bot.answer_callback_query
_orig_send_chat_action = bot.send_chat_action
_orig_set_my_commands = bot.set_my_commands

set_orig_send_chat_action(_orig_send_chat_action)


# =============================================================================
# Formatter Functions & Core Tools
# =============================================================================
def format_usdt(val: Any) -> str:
    """Format balance symmetrically."""
    try:
        return f"💰 <b>{float(val):.2f}$</b>"
    except Exception:
        return f"💰 <b>{val}</b>"


def _debounce_check(key: str) -> bool:
    """Validate if command complies with rate limits (buttons spams)."""
    now = time.time()
    with _HANDLER_LOCK:
        last = _HANDLER_LAST_TS.get(key, 0.0)
        if now - last < _HANDLER_DEBOUNCE_SEC:
            return False
        _HANDLER_LAST_TS[key] = now
    return True


def _safe_send_message(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("send_message", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None

    kw.setdefault("disable_web_page_preview", True)
    _maybe_send_typing(chat_id)

    return tg_call(
        _orig_send_message,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "send_message", exc, a, kw
        ),
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
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "edit_message_text", exc, a, kw
        ),
        **kw,
    )


def _safe_answer_callback_query(*a: Any, **kw: Any) -> Any:
    return tg_call(
        _orig_answer_callback_query,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "answer_callback_query", exc, a, kw
        ),
        **kw,
    )


def _safe_send_chat_action(*a: Any, **kw: Any) -> Any:
    chat_id = _extract_chat_id_from_call("send_chat_action", a, kw)
    if chat_id is not None and _blocked_chat_cache.get(chat_id):
        return None
    return tg_call(
        _orig_send_chat_action,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "send_chat_action", exc, a, kw
        ),
        **kw,
    )


def _safe_set_my_commands(*a: Any, **kw: Any) -> Any:
    return tg_call(
        _orig_set_my_commands,
        *a,
        on_permanent_failure=lambda exc: _handle_permanent_telegram_failure(
            "set_my_commands", exc, a, kw
        ),
        **kw,
    )


bot.send_message = _safe_send_message
bot.edit_message_text = _safe_edit_message_text
bot.answer_callback_query = _safe_answer_callback_query
bot.send_chat_action = _safe_send_chat_action
bot.set_my_commands = _safe_set_my_commands

_orig_message_handler = bot.message_handler
_orig_callback_query_handler = bot.callback_query_handler


def _with_action_wrapper(
    registrar: Callable[..., Any], action: str = "typing"
) -> Callable[..., Any]:
    def _decorator_factory(
        *dargs: Any, **dkwargs: Any
    ) -> Callable[[Callable[..., Any]], Any]:
        base_deco = registrar(*dargs, **dkwargs)

        def _apply(fn: Callable[..., Any]) -> Any:
            return base_deco(send_action(action)(fn))

        return _apply

    return _decorator_factory


bot.message_handler = _with_action_wrapper(_orig_message_handler, "typing")
bot.callback_query_handler = _with_action_wrapper(
    _orig_callback_query_handler, "typing"
)

engine.set_order_notifier(_notify_order_update)
engine.set_skip_notifier(_notify_order_skipped)
engine.set_phase_notifier(_notify_phase_change)
engine.set_engine_stop_notifier(_notify_engine_stopped)
engine.set_daily_start_notifier(_notify_daily_start)


# =============================================================================
# Permission Decorators
# =============================================================================
def admin_only_message(fn: Callable) -> Callable:
    def wrapper(message: telebot.types.Message) -> None:
        try:
            chat_id = int(message.chat.id)
        except Exception:
            chat_id = 0
        try:
            user_id = int(message.from_user.id) if message.from_user else 0
        except Exception:
            user_id = 0
        is_admin = bool((ADMIN and user_id == int(ADMIN)) or is_admin_chat(chat_id))
        if not is_admin:
            deny(message)
            return
        fn(message)

    return wrapper


def admin_only_callback(fn: Callable) -> Callable:
    def wrapper(call: telebot.types.CallbackQuery) -> None:
        try:
            chat_id = int(call.message.chat.id) if call.message else 0
            user_id = int(call.from_user.id) if call.from_user else 0
        except Exception:
            bot.answer_callback_query(call.id, "❌ Дастрасӣ нест")
            return
        if not bool((ADMIN and user_id == int(ADMIN)) or is_admin_chat(chat_id)):
            bot.answer_callback_query(call.id, "❌ Дастрасӣ нест")
            return
        fn(call)

    return wrapper


def callback_route(pattern: str) -> Callable:
    rx = re.compile(pattern)

    def deco(fn: Callable) -> Callable:
        _CALLBACK_ROUTES.append((rx, fn))
        return fn

    return deco


def _ai_style_label(style: str) -> str:
    return "Рӯзона" if str(style).lower() == "intraday" else "Скалп"


def _ai_cached_banner(payload: Dict[str, Any]) -> str:
    meta = payload.get("meta") or {}
    if not bool(meta.get("cached_fallback")):
        return ""
    age = int(meta.get("cached_age_sec", 0) or 0)
    mt5_reason = str(meta.get("mt5_reason", "") or "").strip()
    suffix = f" | сабаб: <code>{html.escape(mt5_reason)}</code>" if mt5_reason else ""
    return f"⚠️ <b>МАЪЛУМОТИ КЭШШУДА</b> | синну сол={age}с{suffix}"


def _decorate_ai_text(text: str, payload: Dict[str, Any]) -> str:
    banner = _ai_cached_banner(payload)
    return f"{banner}\n\n{text}" if banner else text


def _mt5_unavailable_message() -> str:
    _, reason = mt5_status()
    reason_raw = str(reason or "unknown")
    reason_s = reason_raw
    if reason_s.startswith("blocked_cooldown:"):
        reason_s = reason_s.split(":", 1)[1]
    hint = ""
    if "algo_trading_disabled_in_terminal" in reason_s:
        hint = (
            "\n\nMT5: Tools -> Options -> Expert Advisors -> "
            "Allow Algo Trading-ро фаъол кунед ва тугмаи AutoTrading-ро сабз гардонед."
        )
    elif "ipc_disconnected" in reason_s or "No IPC connection" in reason_s:
        hint = "\n\nПайвасти Python ба MT5 канда шудааст. Терминали MT5 ва ботро аз нав оғоз кунед."
    return f"⚠️ <b>MT5 ҳоло омода нест</b>\n<code>{html.escape(reason_s)}</code>{hint}"


# =============================================================================
# Main Telegram Commands Logic
# =============================================================================
def bot_commands() -> None:
    commands = [
        telebot.types.BotCommand("/start", "🚀 Оғоз / Менюи Асосӣ"),
        telebot.types.BotCommand("/buttons", "🎛️ Панели Идоракунӣ"),
        telebot.types.BotCommand("/status", "🟢 Ҳолати Система (Live)"),
        telebot.types.BotCommand("/ai", "🧠 Таҳлили Бозор (AI)"),
        telebot.types.BotCommand("/balance", "💳 Баланс ва Сармоя"),
        telebot.types.BotCommand("/history", "📜 Таърихи Савдо (Логҳо)"),
        telebot.types.BotCommand("/helpers", "📚 Роҳнамо ва Дастурҳо"),
    ]

    ok = bot.set_my_commands(commands)
    if not ok:
        log.warning("⚠️ Failed to update bot commands (API Error).")
    else:
        log.info("✅ Bot commands updated successfully with Premium UI.")


def buttons_func(message: telebot.types.Message) -> None:
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row(KeyboardButton(BTN_START), KeyboardButton(BTN_STOP))
    markup.row(KeyboardButton(BTN_CLOSE_ALL), KeyboardButton(BTN_CLOSE_PROFIT))
    markup.row(KeyboardButton(BTN_OPEN_ORDERS))
    markup.row(KeyboardButton(BTN_BALANCE), KeyboardButton(BTN_POS))
    markup.row(KeyboardButton(BTN_ENGINE), KeyboardButton(BTN_FULL))
    markup.row(
        KeyboardButton(BTN_PROFIT_D),
        KeyboardButton(BTN_PROFIT_W),
        KeyboardButton(BTN_PROFIT_M),
    )

    bot.send_message(
        message.chat.id,
        "🎛 <b>Панели идоракунии бот</b>\nЛутфан амалиётро интихоб кунед ⬇️",
        reply_markup=markup,
        parse_mode="HTML",
    )


@bot.message_handler(commands=["tek_prof"])
@admin_only_message
def tek_profit_put(message: telebot.types.Message) -> None:
    _send_clean(
        message.chat.id, "⌨️ <b>Меню пӯшида шуд</b>\n🎛 Ҳоло TP-ро интихоб мекунем."
    )
    kb = _build_tp_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🎛 <b>Тейк-профит (бо ATR ва ҳадди ақал дар USD)</b>\nБарои <b>ҳамаи позицияҳои кушода</b> интихоб кунед:",
        reply_markup=kb,
        parse_mode="HTML",
    )


@bot.callback_query_handler(
    func=lambda call: bool(call.data) and call.data.startswith(TP_CALLBACK_PREFIX)
)
@admin_only_callback
def on_tp_usd_click(call: telebot.types.CallbackQuery) -> None:
    data = (call.data or "").split(":", 1)[-1].strip().lower()

    if data == "cancel":
        bot.answer_callback_query(call.id, "Бекор шуд")
        try:
            bot.edit_message_reply_markup(
                call.message.chat.id, call.message.message_id, reply_markup=None
            )
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
        bot.answer_callback_query(call.id, "Хато ҳангоми коркард", show_alert=True)
        bot.send_message(
            call.message.chat.id,
            f"⚠️ Хато ҳангоми коркард: <code>{exc}</code>",
            parse_mode="HTML",
        )


@bot.message_handler(commands=["helpers"])
@admin_only_message
def helpers_handler(message: telebot.types.Message) -> None:
    _send_clean(message.chat.id, "⌨️ <b>Менюи helper боз шуд</b>\n🛠 Ёвариҳо.")
    bot.send_message(
        message.chat.id,
        "🛠 <b>Ёвариҳо</b>\n\n"
        "📈 <b>TP</b> / 🛡 <b>SL</b> — барои ҳамаи позицияҳои кушода.\n"
        "🟢 <b>Харид</b> / 🔴 <b>Фурӯш</b> — helper-и дастӣ акнун танҳо "
        "<b>1 safe order</b> мекушояд ва bulk stacking иҷозат нест.",
        reply_markup=build_helpers_keyboard(),
        parse_mode="HTML",
    )


@bot.callback_query_handler(
    func=lambda call: bool(call.data) and call.data.startswith(HELPER_CALLBACK_PREFIX)
)
@admin_only_callback
def on_helper_click(call: telebot.types.CallbackQuery) -> None:
    data = (call.data or "").replace(HELPER_CALLBACK_PREFIX, "", 1).strip().lower()
    if not data:
        bot.answer_callback_query(call.id, "Бекор")
        return

    if data == "tp":
        bot.answer_callback_query(call.id, "📈 TP …")
        kb = _build_tp_usd_keyboard()
        bot.send_message(
            call.message.chat.id,
            "📈 <b>Тейк-профит (бо ATR ва ҳадди ақал дар USD)</b>\nБарои ҳамаи позицияҳои кушода интихоб кунед:",
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    if data == "sl":
        bot.answer_callback_query(call.id, "🛡 SL …")
        kb = _build_sl_usd_keyboard()
        bot.send_message(
            call.message.chat.id,
            "🛡 <b>Стоп-лосс (USD)</b>\nБарои ҳамаи позицияҳои кушода интихоб кунед (1..10$):",
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    if data in ("buy_btc", "sell_btc", "buy_xau", "sell_xau"):
        if not _debounce_check(f"helper:{call.message.chat.id}:{data}"):
            bot.answer_callback_query(
                call.id, "⏳ Дар ҳоли коркард...", show_alert=False
            )
            return
        try:
            st = engine.status()
            if bool(getattr(st, "trading", False)) and not bool(
                getattr(st, "manual_stop", False)
            ):
                bot.answer_callback_query(
                    call.id, "Manual helper blocked", show_alert=True
                )
                bot.send_message(
                    call.message.chat.id,
                    "⚠️ <b>Manual helper баста шуд</b>\nАввал auto-trading-ро ба monitoring гузаронед, баъд helper-и дастиро истифода баред.",
                    parse_mode="HTML",
                )
                return
        except Exception:
            pass
        bot.answer_callback_query(call.id, "Шумораро интихоб кунед")
        titles = {
            "buy_btc": "🟢 <b>Хариди BTC</b> — шумораи ордерҳо",
            "sell_btc": "🔴 <b>Фурӯши BTC</b> — шумораи ордерҳо",
            "buy_xau": "🟢 <b>Хариди XAU</b> — шумораи ордерҳо",
            "sell_xau": "🔴 <b>Фурӯши XAU</b> — шумораи ордерҳо",
        }
        bot.send_message(
            call.message.chat.id,
            titles.get(data, "Шумора:"),
            reply_markup=build_helper_order_count_keyboard(data),
            parse_mode="HTML",
        )
        return

    parts = data.split(":", 1)
    if len(parts) != 2:
        bot.answer_callback_query(call.id, "Формат нодуруст", show_alert=True)
        return
    action, count_str = parts[0].strip(), parts[1].strip()

    if count_str == "cancel":
        bot.answer_callback_query(call.id, "Бекор шуд")
        try:
            bot.edit_message_reply_markup(
                call.message.chat.id, call.message.message_id, reply_markup=None
            )
        except Exception:
            pass
        return

    try:
        count = int(count_str)
    except ValueError:
        bot.answer_callback_query(call.id, "Адад нодуруст", show_alert=True)
        return
    if count not in tuple(int(x) for x in HELPER_ORDER_COUNTS):
        allowed_txt = ",".join(str(int(x)) for x in HELPER_ORDER_COUNTS)
        bot.answer_callback_query(call.id, f"Адад: {allowed_txt}", show_alert=True)
        return

    symbol_map = {
        "buy_btc": "BTCUSDm",
        "sell_btc": "BTCUSDm",
        "buy_xau": "XAUUSDm",
        "sell_xau": "XAUUSDm",
    }
    symbol = symbol_map.get(action, "")
    if symbol:
        try:
            remaining, reason = manual_open_capacity(symbol)
        except Exception:
            remaining, reason = 0, "guard_error"
        if remaining <= 0:
            bot.answer_callback_query(call.id, "Manual helper blocked", show_alert=True)
            bot.send_message(
                call.message.chat.id,
                f"⚠️ <b>Manual helper blocked</b>\n<code>{html.escape(str(reason))}</code>",
                parse_mode="HTML",
            )
            return

    if action == "buy_btc":
        bot.answer_callback_query(call.id, f"⏳ Хариди BTC ×{count} …")
        n = open_buy_order_btc(count)
        bot.send_message(
            call.message.chat.id,
            f"🟢 <b>Хариди BTC</b> ×{count}\n✅ Фиристода шуд: <b>{n}</b>",
            parse_mode="HTML",
        )
        return
    if action == "sell_btc":
        bot.answer_callback_query(call.id, f"⏳ Фурӯши BTC ×{count} …")
        n = open_sell_order_btc(count)
        bot.send_message(
            call.message.chat.id,
            f"🔴 <b>Фурӯши BTC</b> ×{count}\n✅ Фиристода шуд: <b>{n}</b>",
            parse_mode="HTML",
        )
        return
    if action == "buy_xau":
        bot.answer_callback_query(call.id, f"⏳ Хариди XAU ×{count} …")
        n = open_buy_order_xau(count)
        bot.send_message(
            call.message.chat.id,
            f"🟢 <b>Хариди XAU</b> ×{count}\n✅ Фиристода шуд: <b>{n}</b>",
            parse_mode="HTML",
        )
        return
    if action == "sell_xau":
        bot.answer_callback_query(call.id, f"⏳ Фурӯши XAU ×{count} …")
        n = open_sell_order_xau(count)
        bot.send_message(
            call.message.chat.id,
            f"🔴 <b>Фурӯши XAU</b> ×{count}\n✅ Фиристода шуд: <b>{n}</b>",
            parse_mode="HTML",
        )
        return

    bot.answer_callback_query(call.id, "Амал номаълум", show_alert=True)


@bot.message_handler(commands=["stop_ls"])
@admin_only_message
def tek_stoploss_put(message: telebot.types.Message) -> None:
    _send_clean(
        message.chat.id, "⌨️ <b>Меню пӯшида шуд</b>\n🛡 Ҳоло SL-ро интихоб мекунем."
    )
    kb = _build_sl_usd_keyboard()
    bot.send_message(
        message.chat.id,
        "🛡 <b>Стоп-лосс (USD)</b>\nБарои <b>ҳамаи позицияҳои кушода</b> интихоб кунед (1..10$):",
        reply_markup=kb,
        parse_mode="HTML",
    )


@bot.callback_query_handler(
    func=lambda call: bool(call.data) and call.data.startswith(SL_CALLBACK_PREFIX)
)
@admin_only_callback
def on_sl_usd_click(call: telebot.types.CallbackQuery) -> None:
    data = (call.data or "").split(":", 1)[-1].strip().lower()

    if data == "cancel":
        bot.answer_callback_query(call.id, "Бекор шуд")
        try:
            bot.edit_message_reply_markup(
                call.message.chat.id, call.message.message_id, reply_markup=None
            )
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
        bot.answer_callback_query(call.id, "Хато ҳангоми коркард", show_alert=True)
        bot.send_message(
            call.message.chat.id,
            f"⚠️ Хато ҳангоми коркард: <code>{exc}</code>",
            parse_mode="HTML",
        )


def send_daily_summary(chat_id: int, *, force_refresh: bool = True) -> None:
    cache_key = ("daily", chat_id)

    if not force_refresh:
        cached = _summary_cache.get(cache_key)
        if cached is not None:
            bot.send_message(
                chat_id, cached, parse_mode="HTML", reply_markup=_rk_remove()
            )
            return
    else:
        _summary_cache.pop(cache_key, None)

    summary = view_all_history_dict(force_refresh=force_refresh)
    total_closed = int(summary.get("total_closed", 0) or 0)
    total_open = int(summary.get("total_open", 0) or 0)

    if total_closed == 0 and total_open == 0:
        bot.send_message(
            chat_id,
            "📅 Имрӯз ҳеҷ ордер (кушода ё баста) вуҷуд надорад.",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return

    text = _build_daily_summary_text(summary)
    _summary_cache.set(cache_key, text)
    bot.send_message(chat_id, text, parse_mode="HTML", reply_markup=_rk_remove())


@bot.message_handler(commands=["start"])
def start_handler(message: telebot.types.Message) -> None:
    try:
        chat_id = int(message.chat.id)
    except Exception:
        chat_id = 0
    try:
        user_id = int(message.from_user.id) if message.from_user else 0
    except Exception:
        user_id = 0
    if not bool((ADMIN and user_id == int(ADMIN)) or is_admin_chat(chat_id)):
        deny(message)
        try:
            user_id = int(message.from_user.id) if message.from_user else 0
            username = (
                str(message.from_user.username or "N/A") if message.from_user else "N/A"
            )
            chat_id = int(message.chat.id)
            first_name = (
                str(message.from_user.first_name or "N/A")
                if message.from_user
                else "N/A"
            )
            last_name = (
                str(message.from_user.last_name or "") if message.from_user else ""
            )

            alert_msg = (
                "⚠️ <b>Кӯшиши дастрасии ғайриқонунӣ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 ID корбар: <code>{user_id}</code>\n"
                f"💬 ID чат: <code>{chat_id}</code>\n"
                f"📛 Номи корбар: @{html.escape(username)}\n"
                f"👨‍💼 Ном: {html.escape(first_name)} {html.escape(last_name)}\n"
                f"⏰ Вақт: {_format_time_only()}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔒 Дастрасӣ рад карда шуд."
            )
            bot.send_message(ADMIN, alert_msg, parse_mode="HTML")
        except Exception as exc:
            log.error("Failed to send unauthorized access alert: %s", exc)
        return

    bot.send_message(
        message.chat.id,
        "👋 <b>Хуш омадед!</b>\nБарои идоракунӣ менюро истифода баред: /buttons",
        parse_mode="HTML",
        reply_markup=_rk_remove(),
    )
    buttons_func(message)


@bot.message_handler(commands=["history"])
@admin_only_message
def history_handler(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(
            message.chat.id,
            _mt5_unavailable_message(),
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return

    try:
        _send_clean(message.chat.id, "📥 <b>Гирифтани ҳисобот...</b>")

        report = get_full_report_all(force_refresh=True)
        acc_info = get_account_info()

        text = _format_full_report(report, "Пурра (Аз ибтидо)")

        open_positions = report.get("open_positions", [])
        if open_positions and len(open_positions) > 0:
            text += "\n<b>Кушода:</b> "
            for i, pos in enumerate(open_positions[:5]):
                if i > 0:
                    text += " | "
                ticket = pos.get("ticket", 0)
                symbol = pos.get("symbol", "")
                profit = pos.get("profit", 0.0)
                text += f"#{ticket} {html.escape(str(symbol))} <b>{profit:+.2f}$</b>"
            if len(open_positions) > 5:
                text += f" | +{len(open_positions) - 5}"
            text += "\n"

        if acc_info:
            login = acc_info.get("login", 0)
            balance = acc_info.get("balance", 0.0)
            equity = acc_info.get("equity", 0.0)
            profit = acc_info.get("profit", 0.0)
            margin_level = acc_info.get("margin_level", 0.0)

            text += f"\n👤 Login: <b>{login}</b>\n"
            text += f"💰 <b>{balance:.2f}$</b> | Equity: <b>{equity:.2f}$</b>"
            if profit != 0:
                text += f" | P&L: <b>{profit:+.2f}$</b>"
            if margin_level:
                text += f" | ML: <b>{margin_level:.1f}%</b>"
            text += "\n"

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = (
                total_profit / total_loss
                if total_loss > 0
                else (total_profit if total_profit > 0 else 0.0)
            )
            text += f"📊 WR: <b>{win_rate:.1f}%</b>"
            if profit_factor:
                text += f" | PF: <b>{profit_factor:.2f}</b>"
            text += "\n"

        text += f"\n{_format_time_only()}\n"

        bot.send_message(
            message.chat.id, text, parse_mode="HTML", reply_markup=_rk_remove()
        )
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            f"⚠️ Хатогӣ ҳангоми гирифтани таърих: <code>{exc}</code>",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )


@bot.message_handler(commands=["buttons"])
@admin_only_message
def buttons_handler(message: telebot.types.Message) -> None:
    buttons_func(message)


@bot.message_handler(commands=["balance"])
@admin_only_message
def balance_handler(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(
            message.chat.id,
            _mt5_unavailable_message(),
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return

    bal = get_balance()
    if bal is None:
        bot.send_message(
            message.chat.id,
            "⚠️ Хатогӣ ҳангоми гирифтани баланс.",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return
    bot.send_message(
        message.chat.id, format_usdt(bal), parse_mode="HTML", reply_markup=_rk_remove()
    )


@bot.message_handler(commands=["ai"])
@admin_only_message
def ai_menu_handler(message: telebot.types.Message) -> None:
    bot.send_message(
        message.chat.id,
        "🤖 <b>Менюи таҳлили ИИ</b>\n\nНавъи таҳлилро интихоб кунед:",
        reply_markup=build_ai_keyboard(),
        parse_mode="HTML",
    )


def _run_ai_panel(
    chat_id: int,
    *,
    asset: str,
    style: str,
    payload_loader: Callable[[], Any],
    analyzer: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> None:
    style_label = _ai_style_label(style)
    symbol = "XAUUSDm" if str(asset).upper() == "XAU" else "BTCUSDm"

    try:
        ok_mt5, _ = mt5_status()
        if not ok_mt5:
            payload = payload_loader()
            if not payload:
                bot.send_message(
                    chat_id,
                    _mt5_unavailable_message(),
                    parse_mode="HTML",
                    reply_markup=_rk_remove(),
                )
                return
            result = analyzer(asset, payload)
            text = _decorate_ai_text(_format_ai_signal(asset, result), payload)
            bot.send_message(
                chat_id, text, parse_mode="HTML", reply_markup=build_helpers_keyboard()
            )
            return

        loading = bot.send_message(
            chat_id,
            f"🔄 <b>AI {asset} | {style_label}</b>\n⏳ Дар ҳоли гирифтани маълумот ва таҳлил...",
            parse_mode="HTML",
        )

        payload = payload_loader()
        if not payload:
            bot.edit_message_text(
                f"⚠️ <b>{asset} | {style_label}</b>\n\nМаълумоти бозор дастрас нест.\nMT5 ва {symbol}-ро дар Market Watch санҷед.",
                chat_id,
                loading.message_id,
                parse_mode="HTML",
            )
            return

        result = analyzer(asset, payload)
        text = _decorate_ai_text(_format_ai_signal(asset, result), payload)
        bot.edit_message_text(
            text,
            chat_id,
            loading.message_id,
            parse_mode="HTML",
            reply_markup=build_helpers_keyboard(),
        )
    except Exception as exc:
        log.error(
            "AI panel error asset=%s style=%s err=%s | tb=%s",
            asset,
            style,
            exc,
            traceback.format_exc(),
        )
        bot.send_message(
            chat_id,
            f"⚠️ Хатои таҳлили {asset} ({style_label}): <code>{html.escape(str(exc))}</code>",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )


def _format_ai_signal(asset: str, result: Dict[str, Any]) -> str:
    signal = str(result.get("signal", "HOLD")).upper()
    confidence = float(result.get("confidence", 0.0) or 0.0)
    reason = html.escape(
        str(result.get("reason", "")).strip() or "Тафсил дастрас нест."
    )
    action = str(result.get("action_short", "")).strip() or (
        "Харид" if signal == "BUY" else ("Фурӯш" if signal == "SELL" else "Интизор")
    )
    entry = result.get("entry")
    stop_loss = result.get("stop_loss")
    take_profit = result.get("take_profit")
    style = str(result.get("analysis_style", "scalping") or "scalping").lower()
    style_label = _ai_style_label(style)
    provider_display = str(
        result.get("provider_display") or result.get("provider") or "Local Heuristic"
    )
    model = str(result.get("model") or "-")
    symbol = str(result.get("symbol") or (f"{asset}USDm")).strip()
    model_rank = int(result.get("model_rank", 0) or 0)
    latency_ms = int(result.get("latency_ms", 0) or 0)
    news = result.get("news_context") or {}

    def _news_bias_label(value: str) -> str:
        mapping = {
            "bullish": "Мусбат",
            "bearish": "Манфӣ",
            "neutral": "Бетараф",
        }
        return mapping.get(str(value or "").lower(), "Бетараф")

    rr_value = None
    try:
        if (
            signal in ("BUY", "SELL")
            and entry is not None
            and stop_loss is not None
            and take_profit is not None
        ):
            risk = abs(float(entry) - float(stop_loss))
            reward = abs(float(take_profit) - float(entry))
            if risk > 0 and reward > 0:
                rr_value = reward / risk
    except Exception:
        rr_value = None

    conf_pct = f"{confidence * 100:.1f}%"

    if signal == "BUY":
        icon, label = "🟢", "ХАРИД"
        direction = "📈"
    elif signal == "SELL":
        icon, label = "🔴", "ФУРӮШ"
        direction = "📉"
    else:
        icon, label = "⚪", "ИНТИЗОР"
        direction = "⏸️"

    lines = [
        f"{icon} <b>AI {asset} | {style_label} | {label}</b>",
        f"🎯 Боварӣ: <b>{conf_pct}</b> {direction}",
        f"🧠 Муҳаррик: <b>{html.escape(provider_display)}</b> | <code>{html.escape(model)}</code>"
        + (f" | TOP-{model_rank}" if model_rank > 0 else ""),
        f"📌 Символ: <b>{html.escape(symbol)}</b>"
        + (f" | ⏱ {latency_ms}ms" if latency_ms > 0 else ""),
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if signal in ("BUY", "SELL"):
        if entry is not None:
            lines.append(f"🔹 Вуруд: <code>{_fmt_price(entry)}</code>")

        sl_tp_line = []
        if stop_loss is not None:
            sl_tp_line.append(f"🛡 SL: <code>{_fmt_price(stop_loss)}</code>")
        if take_profit is not None:
            sl_tp_line.append(f"💰 TP: <code>{_fmt_price(take_profit)}</code>")

        if sl_tp_line:
            lines.append(" | ".join(sl_tp_line))
        if rr_value is not None:
            lines.append(f"⚖️ R:R: <b>{rr_value:.2f}</b>")

        lines.append("━━━━━━━━━━━━━━━━━━━━")

    if news:
        sentiment = float(news.get("avg_sentiment", 0.0) or 0.0)
        high_impact = int(news.get("high_impact_count", 0) or 0)
        news_status = str(news.get("status", "") or "").strip()
        lines.append(
            f"📰 Хабарҳо: <b>{_news_bias_label(str(news.get('bias', 'neutral')))}</b>"
            f" | Sent={sentiment:+.2f} | High={high_impact}"
            + (f" | {html.escape(news_status)}" if news_status else "")
        )
        summary = str(news.get("ai_summary") or news.get("summary_text") or "").strip()
        if summary:
            summary = html.escape(summary.splitlines()[0][:220])
            lines.append(f"🗞 Хулоса: {summary}")
        lines.append("━━━━━━━━━━━━━━━━━━━━")

    lines.append(f"📝 <b>Таҳлил:</b>\n{reason}")
    lines.append("")
    lines.append(f"✅ <b>Тавсия: {action}</b>")

    if signal in ("BUY", "SELL") and confidence >= 0.75:
        lines.append("")
        lines.append("💡 <i>Барои иҷро аз тугмаҳои ёрирасони поён истифода кунед.</i>")
    elif signal == "HOLD":
        lines.append("")
        lines.append(
            "⏸ <i>Ҳоло беҳтар аст интизор шавед то тасдиқи қавитар пайдо шавад.</i>"
        )

    return "\n".join(lines)


def handle_xau_ai_intraday(chat_id: int, message_id: int) -> None:
    _run_ai_panel(
        chat_id,
        asset="XAU",
        style="intraday",
        payload_loader=get_ai_payload_xau_intraday,
        analyzer=analyse_intraday,
    )


def handle_btc_ai_intraday(chat_id: int, message_id: int) -> None:
    _run_ai_panel(
        chat_id,
        asset="BTC",
        style="intraday",
        payload_loader=get_ai_payload_btc_intraday,
        analyzer=analyse_intraday,
    )


def handle_xau_ai(message: telebot.types.Message) -> None:
    _run_ai_panel(
        message.chat.id,
        asset="XAU",
        style="scalping",
        payload_loader=get_ai_payload_xau,
        analyzer=analyse,
    )


def handle_btc_ai(message: telebot.types.Message) -> None:
    _run_ai_panel(
        message.chat.id,
        asset="BTC",
        style="scalping",
        payload_loader=get_ai_payload_btc,
        analyzer=analyse,
    )


@bot.callback_query_handler(
    func=lambda call: bool(call.data) and call.data.startswith(AI_CALLBACK_PREFIX)
)
@admin_only_callback
def ai_callback_handler(call: telebot.types.CallbackQuery) -> None:
    action = call.data[len(AI_CALLBACK_PREFIX) :]
    chat_id = call.message.chat.id
    message_id = call.message.message_id

    if action == "xau_scalp":
        mock_message = telebot.types.Message(
            message_id=0,
            from_user=call.from_user,
            date=0,
            chat=call.message.chat,
            content_type="text",
            options={},
            json_string="",
        )
        handle_xau_ai(mock_message)
        bot.answer_callback_query(call.id)
        return
    if action == "btc_scalp":
        mock_message = telebot.types.Message(
            message_id=0,
            from_user=call.from_user,
            date=0,
            chat=call.message.chat,
            content_type="text",
            options={},
            json_string="",
        )
        handle_btc_ai(mock_message)
        bot.answer_callback_query(call.id)
        return
    if action == "xau_intraday":
        handle_xau_ai_intraday(chat_id, message_id)
    elif action == "btc_intraday":
        handle_btc_ai_intraday(chat_id, message_id)

    bot.answer_callback_query(call.id)


@bot.message_handler(commands=["status"])
@admin_only_message
def status_handler(message: telebot.types.Message) -> None:
    try:
        status = engine.status()
        ribbon = build_health_ribbon(status)
        bot.send_message(
            message.chat.id,
            _format_status_message(status) + ribbon,
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
    except Exception as exc:
        log.error("/status handler error: %s | tb=%s", exc, traceback.format_exc())
        bot.send_message(
            message.chat.id,
            "⚠️ Ҳангоми дархости статус мушкил пеш омад. Пайвастшавӣ ба MT5-ро санҷед.",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )


def start_view_open_orders(message: telebot.types.Message) -> None:
    try:
        chat_id = int(message.chat.id)
    except Exception:
        chat_id = 0
    try:
        user_id = int(message.from_user.id) if message.from_user else 0
    except Exception:
        user_id = 0
    if not bool((ADMIN and user_id == int(ADMIN)) or is_admin_chat(chat_id)):
        return

    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    _send_clean(message.chat.id, "📋 <b>Ордерҳои кушода</b>")
    order_data, total = get_order_by_index(0)

    if not order_data or int(total or 0) == 0:
        bot.send_message(
            message.chat.id,
            "📭 Ордерҳои кушода нестанд.",
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return

    text = format_order(order_data)
    kb = order_keyboard(0, int(total), int(order_data.get("ticket", 0) or 0))
    bot.send_message(message.chat.id, text, reply_markup=kb, parse_mode="HTML")


@bot.callback_query_handler(func=lambda call: True)
@admin_only_callback
def callback_dispatch(call: telebot.types.CallbackQuery) -> None:
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
                log.error(
                    "Callback error data=%s err=%s | tb=%s",
                    data,
                    exc,
                    traceback.format_exc(),
                )
                bot.answer_callback_query(call.id, "❌ Хатогӣ рух дод")
            return

    bot.answer_callback_query(call.id)


@callback_route(r"^orders:nav:(\d+)$")
def cb_orders_nav(call: telebot.types.CallbackQuery, m: re.Match[str]) -> None:
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
def cb_orders_close(call: telebot.types.CallbackQuery, m: re.Match[str]) -> None:
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
def cb_orders_close_view(call: telebot.types.CallbackQuery, m: re.Match[str]) -> None:
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="🔒 Намоиш пӯшида шуд. Барои дидани дубора: /buttons",
        parse_mode="HTML",
    )
    bot.answer_callback_query(call.id, "Намоиш пӯшида шуд.")


def handle_profit_day(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    try:
        report = get_full_report_day(force_refresh=True)
        text = _format_full_report(report, "Имрӯза")
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(
            message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML"
        )


def handle_profit_week(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    try:
        report = get_full_report_week(force_refresh=True)
        text = _format_full_report(report, "Ҳафтаина")

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = (
                total_profit / total_loss
                if total_loss > 0
                else (total_profit if total_profit > 0 else 0.0)
            )

            text += f"<b>WR:</b> {win_rate:.1f}%"
            if profit_factor > 0:
                text += f" | <b>PF:</b> {profit_factor:.2f}"
            text += "\n"

        text += f"\n{_format_time_only()}\n"
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(
            message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML"
        )


def handle_profit_month(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    try:
        report = get_full_report_month(force_refresh=True)
        text = _format_full_report(report, "Моҳона")

        total_closed = int(report.get("total_closed", 0) or 0)
        wins = int(report.get("wins", 0) or 0)
        total_profit = float(report.get("profit", 0.0) or 0.0)
        total_loss = float(report.get("loss", 0.0) or 0.0)

        if total_closed > 0:
            win_rate = (wins / total_closed) * 100.0
            profit_factor = (
                total_profit / total_loss
                if total_loss > 0
                else (total_profit if total_profit > 0 else 0.0)
            )

            text += f"<b>WR:</b> {win_rate:.1f}%"
            if profit_factor > 0:
                text += f" | <b>PF:</b> {profit_factor:.2f}"
            text += "\n"

        text += f"\n{_format_time_only()}\n"
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(
            message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML"
        )


def handle_open_orders(message: telebot.types.Message) -> None:
    start_view_open_orders(message)


def handle_close_all(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    res = close_all_position()
    closed = int(res.get("closed", 0) or 0)
    canceled = int(res.get("canceled", 0) or 0)
    ok = bool(res.get("ok", False))
    status_emoji = "✅" if ok else "⚠️"

    lines = [f"{status_emoji} <b>Баста: {closed}</b>"]
    if canceled > 0:
        lines.append(f"🗑️ Бекор: <b>{canceled}</b>")

    errs = list(res.get("errors") or [])
    if errs:
        preview = " | ".join(str(e)[:25] for e in errs[:2])
        lines.append(f"⚠️ <code>{html.escape(preview)}</code>")

    bot.send_message(message.chat.id, "\n".join(lines), parse_mode="HTML")


def handle_close_by_profit(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    res = close_all_position_by_profit()
    bot.send_message(
        message.chat.id, format_close_by_profit_result(res), parse_mode="HTML"
    )


def handle_positions_summary(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    summary = get_positions_summary()
    bot.send_message(
        message.chat.id, f"📊 <b>{format_usdt(summary)}</b>", parse_mode="HTML"
    )


def handle_balance(message: telebot.types.Message) -> None:
    ok_mt5, _ = mt5_status()
    if not ok_mt5:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return

    acc = get_account_info()
    if not acc:
        bot.send_message(message.chat.id, _mt5_unavailable_message(), parse_mode="HTML")
        return
    balance = float(acc.get("balance", 0.0) or 0.0)
    bot.send_message(message.chat.id, format_usdt(balance), parse_mode="HTML")


def handle_trade_start(message: telebot.types.Message) -> None:
    if not _debounce_check(f"start:{message.chat.id}"):
        bot.send_message(
            message.chat.id,
            "⏳ Команда аллакай дар ҳоли коркард аст...",
            parse_mode="HTML",
        )
        return
    try:
        st = engine.status()
        if bool(getattr(st, "trading", False)) and not bool(
            getattr(st, "manual_stop", False)
        ):
            bot.send_message(
                message.chat.id,
                "ℹ️ <b>Система аллакай фаъол аст.</b>\nСавдо идома дорад.",
                parse_mode="HTML",
            )
            return

        engine.clear_manual_stop()
        st_after_clear = engine.status()
        if bool(getattr(st_after_clear, "manual_stop", False)):
            bot.send_message(
                message.chat.id,
                "Trading is still paused by active protection checks.",
                parse_mode="HTML",
            )
            return
        started = bool(engine.start())
        if started:
            bot.send_message(
                message.chat.id,
                "✅ <b>Система оғоз шуд (Савдо фаъол)</b>\n\nСавдои автоматӣ давом мекунад.",
                parse_mode="HTML",
            )
            return

        _, reason = mt5_status()
        gate_reason = str(getattr(engine, "_gate_last_reason", "") or "")
        blocked_assets = sorted(set(getattr(engine, "_blocked_assets", []) or []))
        model_ready = bool(getattr(engine, "_model_loaded", False)) and bool(
            getattr(engine, "_backtest_passed", False)
        )
        reason_raw = str(reason or "unknown")
        if (not model_ready) or blocked_assets:
            reason_s = gate_reason or "gatekeeper_failed"
            if blocked_assets:
                reason_s = f"{reason_s} | blocked={','.join(blocked_assets)}"
        else:
            reason_s = reason_raw
        if reason_s.startswith("blocked_cooldown:"):
            reason_s = reason_s.split(":", 1)[1]
        hint = ""
        if "algo_trading_disabled_in_terminal" in reason_s:
            hint = (
                "\n\nMT5: Tools -> Options -> Expert Advisors -> "
                "Allow Algo Trading, ва тугмаи AutoTrading-ро фаъол кунед."
            )
        bot.send_message(
            message.chat.id,
            f"⚠️ <b>Start иҷро нашуд</b>\n<code>{html.escape(reason_s)}</code>{hint}",
            parse_mode="HTML",
        )
    except Exception as exc:
        bot.send_message(
            message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML"
        )


def handle_trade_stop(message: telebot.types.Message) -> None:
    if not _debounce_check(f"stop:{message.chat.id}"):
        bot.send_message(
            message.chat.id,
            "⏳ Команда аллакай дар ҳоли коркард аст...",
            parse_mode="HTML",
        )
        return
    try:
        st = engine.status()
        was_active = engine.request_manual_stop()
        if was_active:
            bot.send_message(
                message.chat.id,
                "🛑 <b>Система қатъ шуд (Мониторинг)</b>\n\n👁️ <i>Ҳолати мушоҳида фаъол шуд. Савдо қатъ гашт, аммо сигналҳои AI меоянд.</i>",
                parse_mode="HTML",
            )
        elif bool(getattr(st, "manual_stop", False)):
            bot.send_message(
                message.chat.id,
                "ℹ️ Manual stop аллакай фаъол аст (Мониторинг).",
                parse_mode="HTML",
            )
        else:
            bot.send_message(
                message.chat.id, "ℹ️ Система аллакай қатъ аст.", parse_mode="HTML"
            )
    except Exception as exc:
        bot.send_message(
            message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML"
        )


def handle_engine_check(message: telebot.types.Message) -> None:
    status = engine.status()
    bot.send_message(
        message.chat.id,
        (
            "⚙️ <b>Статуси Муҳаррик</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 Пайваст: {'✅' if status.connected else '❌'}\n"
            f"📈 Trading: {'✅' if status.trading else '❌'}\n"
            f"⛔ Manual stop: {'✅' if status.manual_stop else '❌'}\n"
            f"🎯 Актив: <b>{html.escape(str(status.active_asset))}</b>\n"
            f"📉 DD: <b>{status.dd_pct * 100:.2f}%</b>\n"
            f"📆 Today PnL: <b>{status.today_pnl:+.2f}$</b>\n"
            f"📂 Позицияҳо: XAU <b>{status.open_trades_xau}</b> | BTC <b>{status.open_trades_btc}</b>\n"
            f"🛎 Сигналҳо: XAU <b>{html.escape(str(status.last_signal_xau))}</b> | BTC <b>{html.escape(str(status.last_signal_btc))}</b>\n"
            f"📥 Queue: <b>{status.exec_queue_size}</b>\n"
            f"🧭 Controller: <b>{html.escape(str(getattr(status, 'controller_state', 'unknown')))}</b>\n"
            f"🧪 Chaos: <b>{html.escape(str(getattr(status, 'chaos_state', 'unknown')))}</b>\n"
            f"🚫 Gate: <code>{html.escape(str(getattr(status, 'gate_reason', '') or '-'))}</code>\n"
            f"🛑 Halt: <code>{html.escape(str(getattr(status, 'risk_halt_reason', '') or '-'))}</code>\n"
        ),
        parse_mode="HTML",
    )


def handle_full_check(message: telebot.types.Message) -> None:
    bot.send_message(
        message.chat.id,
        "🔄 <b>Санҷиши пурраи барнома оғоз шуд...</b>",
        parse_mode="HTML",
    )
    ok, detail = check_full_program()
    bot.send_message(message.chat.id, detail, parse_mode="HTML")
    if not ok:
        log.warning("Full check found issues")


BUTTONS: Dict[str, Callable[[telebot.types.Message], None]] = {
    BTN_PROFIT_D: handle_profit_day,
    BTN_PROFIT_W: handle_profit_week,
    BTN_PROFIT_M: handle_profit_month,
    BTN_OPEN_ORDERS: handle_open_orders,
    BTN_CLOSE_ALL: handle_close_all,
    BTN_CLOSE_PROFIT: handle_close_by_profit,
    BTN_POS: handle_positions_summary,
    BTN_BALANCE: handle_balance,
    BTN_START: handle_trade_start,
    BTN_STOP: handle_trade_stop,
    BTN_ENGINE: handle_engine_check,
    BTN_FULL: handle_full_check,
}


@bot.message_handler(func=lambda m: True)
def message_dispatcher(message: telebot.types.Message) -> None:
    try:
        chat_id = int(message.chat.id)
    except Exception:
        chat_id = 0
    try:
        user_id = int(message.from_user.id) if message.from_user else 0
    except Exception:
        user_id = 0
    if not bool((ADMIN and user_id == int(ADMIN)) or is_admin_chat(chat_id)):
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
            log.error(
                "handler error text=%s err=%s | tb=%s",
                text,
                exc,
                traceback.format_exc(),
            )
            bot.send_message(
                message.chat.id,
                "⚠️ Хатогӣ рух дод. Баъдтар дубора санҷед.",
                parse_mode="HTML",
            )
        return

    bot.send_message(
        message.chat.id, "❓ Амали номаълум. /buttons → меню.", parse_mode="HTML"
    )


def send_signal_notification(asset: str, result: Dict[str, Any]) -> None:
    """Callback for engine to notify Telegram about a new signal."""
    _notify_signal(asset, result)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "bot",
    "bot_commands",
    "send_daily_summary",
    "send_signal_notification",
]
