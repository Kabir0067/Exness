# bot.py
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

import re
import traceback
from typing import Any, Dict, Callable

import telebot
from telebot.types import KeyboardButton, ReplyKeyboardMarkup
from ExnessAPI.functions import *
from ExnessAPI.history import *
from .utils import (
    log,
    cfg,
    ADMIN,
    TP_USD_MIN,
    TP_USD_MAX,
    TP_CALLBACK_PREFIX,
    SL_USD_MIN,
    SL_USD_MAX,
    SL_CALLBACK_PREFIX,
    tg_call,
    _blocked_chat_cache,
    _format_time_only,
    _extract_chat_id_from_call,
    _handle_permanent_telegram_failure,
    _maybe_send_typing,
    _rk_remove,
    _send_clean,
    _notify_order_opened,
    engine,
    _notify_phase_change,
    _notify_engine_stopped,
    _notify_daily_start,
    is_admin_chat,
    deny,
    build_health_ribbon,
    _format_status_message,
    _build_daily_summary_text,
    _build_tp_usd_keyboard,
    _format_tp_result,
    _build_sl_usd_keyboard,
    _format_sl_result,
    _summary_cache,
    format_order,
    order_keyboard,
    _format_full_report,
    check_full_program,
    set_bot_instance,
    set_orig_send_chat_action,
)

# =============================================================================
# Bot instance
# =============================================================================
bot = telebot.TeleBot(cfg.telegram_token)

# Set bot instance in utils for helper functions
set_bot_instance(bot)

# Patch critical bot methods ONCE (keeps your old code calls working)
# + Adds "typing" for every outgoing message/edit
# =============================================================================
_orig_send_message = bot.send_message
_orig_edit_message_text = bot.edit_message_text
_orig_answer_callback_query = bot.answer_callback_query
_orig_send_chat_action = bot.send_chat_action
_orig_set_my_commands = bot.set_my_commands

# Set the original send_chat_action reference in utils for _maybe_send_typing
set_orig_send_chat_action(_orig_send_chat_action)

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

engine.set_order_notifier(_notify_order_opened)
engine.set_phase_notifier(_notify_phase_change)
engine.set_engine_stop_notifier(_notify_engine_stopped)
engine.set_daily_start_notifier(_notify_daily_start)

def admin_only_message(fn):
    def wrapper(message):
        if not is_admin_chat(message.chat.id):
            deny(message)
            return
        fn(message)
    return wrapper

def admin_only_callback(fn):
    def wrapper(call):
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
# Commands
# =============================================================================
def bot_commands() -> None:
    commands = [
        telebot.types.BotCommand("/start", "🚀 Барои оғози бот"),
        telebot.types.BotCommand("/history", "📜 Дидани таърихи ордерҳо"),
        telebot.types.BotCommand("/balance", "💰 Дидани баланси худ"),
        telebot.types.BotCommand("/buttons", "🎛️ Тугмаҳои асосӣ"),
        telebot.types.BotCommand("/status", "⚙️ Статус оператсия"),
        telebot.types.BotCommand("/tek_prof", "💰 Гузоштани тек профит"),
        telebot.types.BotCommand("/stop_ls", "🛡 Гузоштани Стоп Лосс"),
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

def buttons_func(message: telebot.types.Message) -> None:
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

@bot.message_handler(commands=["tek_prof"])
@admin_only_message
def tek_profit_put(message: telebot.types.Message) -> None:
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
def on_tp_usd_click(call: telebot.types.CallbackQuery) -> None:
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

@bot.message_handler(commands=["stop_ls"])
@admin_only_message
def tek_stoploss_put(message: telebot.types.Message) -> None:
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
def on_sl_usd_click(call: telebot.types.CallbackQuery) -> None:
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
def send_daily_summary(chat_id: int, *, force_refresh: bool = True) -> None:
    from ExnessAPI.history import view_all_history_dict
    
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
def start_handler(message: telebot.types.Message) -> None:
    if not is_admin_chat(message.chat.id):
        deny(message)
        # Notify admin about unauthorized access attempt
        try:
            user_id = int(message.from_user.id) if message.from_user else 0
            username = str(message.from_user.username or "N/A") if message.from_user else "N/A"
            chat_id = int(message.chat.id)
            first_name = str(message.from_user.first_name or "N/A") if message.from_user else "N/A"
            last_name = str(message.from_user.last_name or "") if message.from_user else ""
            
            alert_msg = (
                "⚠️ <b>Unauthorized Access Attempt</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 User ID: <code>{user_id}</code>\n"
                f"💬 Chat ID: <code>{chat_id}</code>\n"
                f"📛 Username: @{username}\n"
                f"👨‍💼 Name: {first_name} {last_name}\n"
                f"⏰ Time: {_format_time_only()}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔒 Access denied."
            )
            bot.send_message(ADMIN, alert_msg, parse_mode="HTML")
        except Exception as exc:
            log.error("Failed to send unauthorized access alert: %s", exc)
        return
    
    # Admin access - show welcome message and menu
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
def balance_handler(message: telebot.types.Message) -> None:
    bal = get_balance()
    if bal is None:
        bot.send_message(message.chat.id, "⚠️ Хатогӣ ҳангоми гирифтани баланс.", parse_mode="HTML", reply_markup=_rk_remove())
        return
    bot.send_message(message.chat.id, f"💰 <b>Баланс</b>\n{format_usdt(bal)}", parse_mode="HTML", reply_markup=_rk_remove())

@bot.message_handler(commands=["buttons"])
@admin_only_message
def buttons_handler(message: telebot.types.Message) -> None:
    buttons_func(message)

@bot.message_handler(commands=["status"])
@admin_only_message
def status_handler(message: telebot.types.Message) -> None:
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
def start_view_open_orders(message: telebot.types.Message) -> None:
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
_CALLBACK_ROUTES = []

def callback_route(pattern: str):
    rx = re.compile(pattern)

    def deco(fn):
        _CALLBACK_ROUTES.append((rx, fn))
        return fn

    return deco

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
                log.error("Callback error data=%s err=%s | tb=%s", data, exc, traceback.format_exc())
                bot.answer_callback_query(call.id, "❌ Хатогӣ рух дод")
            return

    bot.answer_callback_query(call.id)  # unknown callback -> silent

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

# =============================================================================
# Button dispatcher (maintainable; no huge if-elif)
# =============================================================================
def handle_profit_day(message: telebot.types.Message) -> None:
    try:
        report = get_full_report_day(force_refresh=True)
        text = _format_full_report(report, "Имрӯза")
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(message.chat.id, f"⚠️ Хатогӣ: <code>{exc}</code>", parse_mode="HTML")

def handle_profit_week(message: telebot.types.Message) -> None:
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

def handle_profit_month(message: telebot.types.Message) -> None:
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

def handle_open_orders(message: telebot.types.Message) -> None:
    start_view_open_orders(message)

def handle_close_all(message: telebot.types.Message) -> None:
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

def handle_positions_summary(message: telebot.types.Message) -> None:
    summary = get_positions_summary()
    bot.send_message(message.chat.id, f"📊 <b>{format_usdt(summary)}</b>", parse_mode="HTML")

def handle_balance(message: telebot.types.Message) -> None:
    balance = get_balance()
    bot.send_message(message.chat.id, f"💰 <b>Баланс</b>\n{format_usdt(balance)}", parse_mode="HTML")

def handle_trade_start(message: telebot.types.Message) -> None:
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

def handle_trade_stop(message: telebot.types.Message) -> None:
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
            f"🎯 Актив: <b>{status.active_asset}</b>\n"
            f"📉 DD: <b>{status.dd_pct * 100:.2f}%</b>\n"
            f"📆 Today PnL: <b>{status.today_pnl:+.2f}$</b>\n"
            f"📂 Позицияҳо: XAU <b>{status.open_trades_xau}</b> | BTC <b>{status.open_trades_btc}</b>\n"
            f"🛎 Сигналҳо: XAU <b>{status.last_signal_xau}</b> | BTC <b>{status.last_signal_btc}</b>\n"
            f"📥 Queue: <b>{status.exec_queue_size}</b>\n"
        ),
        parse_mode="HTML",
    )

def handle_full_check(message: telebot.types.Message) -> None:
    bot.send_message(message.chat.id, "🔄 <b>Санҷиши пурраи барнома оғоз шуд...</b>", parse_mode="HTML")
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
    BTN_POS: handle_positions_summary,
    BTN_BALANCE: handle_balance,
    BTN_START: handle_trade_start,
    BTN_STOP: handle_trade_stop,
    BTN_ENGINE: handle_engine_check,
    BTN_FULL: handle_full_check,
}

@bot.message_handler(func=lambda m: True)
def message_dispatcher(message: telebot.types.Message) -> None:
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
            bot.send_message(message.chat.id,  "⚠️ Хатогӣ рух дод. Баъдтар дубора санҷед.", parse_mode="HTML")
        return

    bot.send_message(message.chat.id, "❓ Амали номаълум. /buttons → меню.", parse_mode="HTML")