"""
Telegram control plane for the trading system.

Exposes the administrator bot interface for runtime controls, status
inspection, and order-management actions.
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
    """Format a USD amount in a compact, styled way."""
    try:
        return f"<b>{float(val):.2f}$</b>"
    except Exception:
        return f"<b>{val}</b>"


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


# ═════════════════════════════════════════════════════════════════════════
# GUARDIAN — autonomous watchtower (heartbeats, trade alerts, disconnects)
# ═════════════════════════════════════════════════════════════════════════
try:
    from Bot.guardian import install_guardian as _install_guardian

    def _guardian_notify(text: str) -> None:
        try:
            if not ADMIN:
                return
            bot.send_message(int(ADMIN), text, parse_mode="HTML")
        except Exception as _gn_exc:  # pragma: no cover — best-effort
            log.warning("GUARDIAN_SEND_FAILED | err=%s", _gn_exc)

    _guardian = _install_guardian(
        engine_provider=lambda: engine,
        notify=_guardian_notify,
    )

    _orig_order_notifier = _notify_order_update

    def _notify_order_update_with_guardian(intent: Any, result: Any) -> None:
        try:
            _orig_order_notifier(intent, result)
        except Exception:
            pass
        try:
            _guardian.on_trade_result(intent, result)
        except Exception:
            pass

    engine.set_order_notifier(_notify_order_update_with_guardian)
except Exception as _g_exc:  # pragma: no cover — guardian is optional
    log.warning("GUARDIAN_INSTALL_FAILED | err=%s", _g_exc)


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
    suffix = f"\n💬 Сабаб: <code>{html.escape(mt5_reason)}</code>" if mt5_reason else ""
    return (
        "⚠️ <b>ДИҚҚАТ: МАЪЛУМОТИ ЗАХИРАШУДА ИСТИФОДА ШУД</b>\n"
        f"🕐 Синни маълумот: <b>{age} сония</b>"
        f"{suffix}"
    )


def _decorate_ai_text(text: str, payload: Dict[str, Any]) -> str:
    banner = _ai_cached_banner(payload)
    return f"{banner}\n\n{text}" if banner else text


def _mt5_unavailable_message() -> str:
    _, reason = mt5_status()
    reason_raw = str(reason or "сабаб номаълум")
    reason_s = reason_raw
    if reason_s.startswith("blocked_cooldown:"):
        reason_s = reason_s.split(":", 1)[1]
    hint = ""
    if "algo_trading_disabled_in_terminal" in reason_s:
        hint = (
            "\n\n💡 <b>Роҳи ҳалли мушкилот:</b>\n"
            "1️⃣ Терминали MT5-ро кушоед\n"
            "2️⃣ Ба менюи: <code>Tools → Options → Expert Advisors</code>\n"
            "3️⃣ Сатри «Allow Algo Trading»-ро фаъол созед\n"
            "4️⃣ Тугмаи «AutoTrading»-и дар болои терминал будаашро сабз гардонед"
        )
    elif "ipc_disconnected" in reason_s or "No IPC connection" in reason_s:
        hint = (
            "\n\n💡 <b>Роҳи ҳалли мушкилот:</b>\n"
            "Пайвасти бот ба терминали MT5 қатъ шудааст.\n"
            "Лутфан аввал терминали MT5-ро, сипас ботро аз нав оғоз кунед."
        )
    return (
        "⚠️ <b>ТЕРМИНАЛИ MT5 ҲОЛО ОМОДА НЕСТ</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 Тафсилот: <code>{html.escape(reason_s)}</code>"
        f"{hint}"
    )


# =============================================================================
# Main Telegram Commands Logic
# =============================================================================
def bot_commands() -> None:
    commands = [
        telebot.types.BotCommand("/start", "🚀 Оғоз ва менюи асосӣ"),
        telebot.types.BotCommand("/buttons", "🎛 Панели идоракунии савдо"),
        telebot.types.BotCommand("/status", "📊 Ҳолати ҷории система"),
        telebot.types.BotCommand("/ai", "🧠 Таҳлили бозор бо AI"),
        telebot.types.BotCommand("/balance", "💳 Баланс ва сармоя"),
        telebot.types.BotCommand("/history", "📜 Ҳисоботи пурра аз оғоз"),
        telebot.types.BotCommand("/helpers", "🛠 Ёвариҳои савдои дастӣ"),
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
        (
            "🎛 <b>ПАНЕЛИ ИДОРАКУНИИ САВДО</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Лутфан амалиёти дилхоҳро аз тугмаҳои поён интихоб кунед ⬇️"
        ),
        reply_markup=markup,
        parse_mode="HTML",
    )


@bot.message_handler(commands=["tek_prof"])
@admin_only_message
def tek_profit_put(message: telebot.types.Message) -> None:
    _send_clean(
        message.chat.id,
        "⌨️ <b>Менюи асосӣ муваққатан пӯшида шуд</b>\n🎯 Ҳоло Тейк-Профитро танзим мекунем...",
    )
    kb = _build_tp_usd_keyboard()
    bot.send_message(
        message.chat.id,
        (
            "🎯 <b>ТАНЗИМИ ТЕЙК-ПРОФИТ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Маблағи Тейк-Профитро барои <b>ҳамаи позицияҳои кушода</b> "
            "интихоб кунед:\n"
            "<i>(ҳисобот бо назардошти ATR ва ҳадди ақали USD)</i>"
        ),
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
        bot.answer_callback_query(call.id, "Амал бекор карда шуд")
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
            bot.answer_callback_query(
                call.id,
                f"Диапазон: {TP_USD_MIN}..{TP_USD_MAX}$",
                show_alert=True,
            )
            return

        bot.answer_callback_query(call.id, f"⏳ Дар ҳоли танзими TP = {usd:.0f}$ ...")
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
        bot.answer_callback_query(
            call.id, "Ҳангоми иҷро хатогӣ рух дод", show_alert=True
        )
        bot.send_message(
            call.message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАНЗИМИ ТЕЙК-ПРОФИТ ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
        )


@bot.message_handler(commands=["helpers"])
@admin_only_message
def helpers_handler(message: telebot.types.Message) -> None:
    _send_clean(
        message.chat.id,
        "⌨️ <b>Менюи асосӣ муваққатан пӯшида шуд</b>\n🛠 Менюи ёвариҳо кушода мешавад...",
    )
    bot.send_message(
        message.chat.id,
        (
            "🛠 <b>МЕНЮИ ЁВАРИҲОИ ДАСТӢ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🎯 <b>Тейк-Профит / Стоп-Лосс</b>\n"
            "    барои ҳамаи позицияҳои кушода\n\n"
            "🟢 <b>Харид</b> / 🔴 <b>Фурӯш</b>\n"
            "    кушодани ордери дастӣ (танҳо 1 ордери ҳифзшуда)\n\n"
            "ℹ️ <i>Эзоҳ: кушодани якбораи бисёр ордерҳо манъ аст.</i>"
        ),
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
        bot.answer_callback_query(call.id, "Амал бекор карда шуд")
        return

    if data == "tp":
        bot.answer_callback_query(call.id, "🎯 Танзими Тейк-Профит...")
        kb = _build_tp_usd_keyboard()
        bot.send_message(
            call.message.chat.id,
            (
                "🎯 <b>ТАНЗИМИ ТЕЙК-ПРОФИТ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Маблағи Тейк-Профитро барои <b>ҳамаи позицияҳои кушода</b> "
                "интихоб кунед:\n"
                "<i>(ҳисобот бо назардошти ATR ва ҳадди ақали USD)</i>"
            ),
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    if data == "sl":
        bot.answer_callback_query(call.id, "🛡 Танзими Стоп-Лосс...")
        kb = _build_sl_usd_keyboard()
        bot.send_message(
            call.message.chat.id,
            (
                "🛡 <b>ТАНЗИМИ СТОП-ЛОСС</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"Ҳадди зиёнро барои <b>ҳамаи позицияҳои кушода</b> интихоб кунед\n"
                f"<i>(диапазон: {SL_USD_MIN}$ то {SL_USD_MAX}$)</i>"
            ),
            reply_markup=kb,
            parse_mode="HTML",
        )
        return

    if data in ("buy_btc", "sell_btc", "buy_xau", "sell_xau"):
        if not _debounce_check(f"helper:{call.message.chat.id}:{data}"):
            bot.answer_callback_query(
                call.id, "⏳ Лутфан интизор шавед, амал ҳанӯз коркард мешавад...",
                show_alert=False,
            )
            return
        try:
            st = engine.status()
            if bool(getattr(st, "trading", False)) and not bool(
                getattr(st, "manual_stop", False)
            ):
                bot.answer_callback_query(
                    call.id,
                    "Кушодани ордери дастӣ ҳоло иҷозат дода намешавад",
                    show_alert=True,
                )
                bot.send_message(
                    call.message.chat.id,
                    (
                        "⚠️ <b>КУШОДАНИ ОРДЕРИ ДАСТӢ МАНЪ АСТ</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "Ҳоло системаи савдои худкор фаъол аст.\n\n"
                        "💡 <b>Барои истифодаи ёвариҳои дастӣ:</b>\n"
                        "1️⃣ Аввал тугмаи «🛑 Қатъи Тиҷорат»-ро пахш кунед\n"
                        "2️⃣ Пас аз гузаштан ба ҳолати мониторинг\n"
                        "    ёвариҳои дастӣ дастрас мешаванд"
                    ),
                    parse_mode="HTML",
                )
                return
        except Exception:
            pass
        bot.answer_callback_query(call.id, "Шумораи ордерҳоро интихоб кунед")
        titles = {
            "buy_btc": (
                "🟢 <b>ХАРИДИ БИТКОИН (BTC)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Шумораи ордерҳоро барои кушодан интихоб кунед:"
            ),
            "sell_btc": (
                "🔴 <b>ФУРӮШИ БИТКОИН (BTC)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Шумораи ордерҳоро барои кушодан интихоб кунед:"
            ),
            "buy_xau": (
                "🟢 <b>ХАРИДИ ТИЛЛО (XAU)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Шумораи ордерҳоро барои кушодан интихоб кунед:"
            ),
            "sell_xau": (
                "🔴 <b>ФУРӮШИ ТИЛЛО (XAU)</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Шумораи ордерҳоро барои кушодан интихоб кунед:"
            ),
        }
        bot.send_message(
            call.message.chat.id,
            titles.get(data, "Шумораи ордерҳоро интихоб кунед:"),
            reply_markup=build_helper_order_count_keyboard(data),
            parse_mode="HTML",
        )
        return

    parts = data.split(":", 1)
    if len(parts) != 2:
        bot.answer_callback_query(
            call.id, "Формати дархост нодуруст аст", show_alert=True
        )
        return
    action, count_str = parts[0].strip(), parts[1].strip()

    if count_str == "cancel":
        bot.answer_callback_query(call.id, "Амал бекор карда шуд")
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
        bot.answer_callback_query(
            call.id, "Адади интихобшуда нодуруст аст", show_alert=True
        )
        return
    if count not in tuple(int(x) for x in HELPER_ORDER_COUNTS):
        allowed_txt = ", ".join(str(int(x)) for x in HELPER_ORDER_COUNTS)
        bot.answer_callback_query(
            call.id,
            f"Танҳо ададҳои {allowed_txt} иҷозат дода шудаанд",
            show_alert=True,
        )
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
            bot.answer_callback_query(
                call.id,
                "Кушодани ордери дастӣ ҳоло иҷозат дода намешавад",
                show_alert=True,
            )
            bot.send_message(
                call.message.chat.id,
                (
                    "⚠️ <b>КУШОДАНИ ОРДЕРИ ДАСТӢ МАНЪ ШУД</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"💬 Сабаб: <code>{html.escape(str(reason))}</code>"
                ),
                parse_mode="HTML",
            )
            return

    def _send_manual_result(
        asset_label: str, side_emoji: str, side_label: str, count: int, n: int
    ) -> None:
        bot.send_message(
            call.message.chat.id,
            (
                f"{side_emoji} <b>{side_label.upper()}И {asset_label}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"📦 Дархости кушодан: <b>{count}</b> ордер\n"
                f"✅ Бо муваффақият кушода шуд: <b>{n}</b> ордер"
                + (
                    f"\n⚠️ Кушода нашуд: <b>{count - n}</b> ордер"
                    if n < count
                    else ""
                )
            ),
            parse_mode="HTML",
        )

    if action == "buy_btc":
        bot.answer_callback_query(call.id, f"⏳ Хариди BTC × {count} ...")
        n = open_buy_order_btc(count)
        _send_manual_result("БИТКОИН (BTC)", "🟢", "Харид", count, n)
        return
    if action == "sell_btc":
        bot.answer_callback_query(call.id, f"⏳ Фурӯши BTC × {count} ...")
        n = open_sell_order_btc(count)
        _send_manual_result("БИТКОИН (BTC)", "🔴", "Фурӯш", count, n)
        return
    if action == "buy_xau":
        bot.answer_callback_query(call.id, f"⏳ Хариди XAU × {count} ...")
        n = open_buy_order_xau(count)
        _send_manual_result("ТИЛЛО (XAU)", "🟢", "Харид", count, n)
        return
    if action == "sell_xau":
        bot.answer_callback_query(call.id, f"⏳ Фурӯши XAU × {count} ...")
        n = open_sell_order_xau(count)
        _send_manual_result("ТИЛЛО (XAU)", "🔴", "Фурӯш", count, n)
        return

    bot.answer_callback_query(call.id, "Амали интихобшуда номаълум аст", show_alert=True)


@bot.message_handler(commands=["stop_ls"])
@admin_only_message
def tek_stoploss_put(message: telebot.types.Message) -> None:
    _send_clean(
        message.chat.id,
        "⌨️ <b>Менюи асосӣ муваққатан пӯшида шуд</b>\n🛡 Ҳоло Стоп-Лоссро танзим мекунем...",
    )
    kb = _build_sl_usd_keyboard()
    bot.send_message(
        message.chat.id,
        (
            "🛡 <b>ТАНЗИМИ СТОП-ЛОСС</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"Ҳадди зиёнро барои <b>ҳамаи позицияҳои кушода</b> интихоб кунед\n"
            f"<i>(диапазон: {SL_USD_MIN}$ то {SL_USD_MAX}$)</i>"
        ),
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
        bot.answer_callback_query(call.id, "Амал бекор карда шуд")
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
            bot.answer_callback_query(
                call.id,
                f"Диапазон: {SL_USD_MIN}..{SL_USD_MAX}$",
                show_alert=True,
            )
            return

        bot.answer_callback_query(call.id, f"⏳ Дар ҳоли танзими SL = {usd:.0f}$ ...")
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
        bot.answer_callback_query(
            call.id, "Ҳангоми иҷро хатогӣ рух дод", show_alert=True
        )
        bot.send_message(
            call.message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАНЗИМИ СТОП-ЛОСС ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
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
            (
                "📅 <b>ХУЛОСАИ ИМРӮЗА</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "ℹ️ Имрӯз ҳеҷ ордер (на кушода, на баста) мавҷуд нест.\n"
                "Система дар ҳолати мушоҳида қарор дорад."
            ),
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
                "🚨 <b>КӮШИШИ ДАСТРАСИИ БЕИҶОЗАТ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 ID-и корбар: <code>{user_id}</code>\n"
                f"💬 ID-и чат: <code>{chat_id}</code>\n"
                f"📛 Номи корбарӣ: @{html.escape(username)}\n"
                f"🧑 Ном: {html.escape(first_name)} {html.escape(last_name)}\n"
                f"⏰ Вақт: {_format_time_only()}\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "🔒 Дастрасӣ автоматӣ рад карда шуд."
            )
            bot.send_message(ADMIN, alert_msg, parse_mode="HTML")
        except Exception as exc:
            log.error("Failed to send unauthorized access alert: %s", exc)
        return

    bot.send_message(
        message.chat.id,
        (
            "👋 <b>ХУШ ОМАДЕД!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🤖 Ин боти савдои автоматии Exness аст.\n"
            "🎛 Барои идоракунӣ аз менюи поён истифода баред\n"
            "    ё фармони <code>/buttons</code>-ро пахш кунед."
        ),
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
        _send_clean(
            message.chat.id,
            "📥 <b>ДАР ҲОЛИ ТАЙЁР КАРДАНИ ҲИСОБОТ...</b>\nЛутфан чанд сония интизор шавед.",
        )

        report = get_full_report_all(force_refresh=True)
        acc_info = get_account_info()

        text = _format_full_report(report, "Пурра (Аз ибтидо)")

        open_positions = report.get("open_positions", [])
        if open_positions and len(open_positions) > 0:
            text += "\n🔓 <b>Позицияҳои кушода:</b>\n"
            for i, pos in enumerate(open_positions[:5]):
                ticket = pos.get("ticket", 0)
                symbol = pos.get("symbol", "")
                profit = pos.get("profit", 0.0)
                profit_icon = "🟢" if float(profit) >= 0 else "🔴"
                text += (
                    f"   {profit_icon} #{ticket} "
                    f"{html.escape(str(symbol))} "
                    f"<b>{float(profit):+.2f}$</b>\n"
                )
            if len(open_positions) > 5:
                text += f"   ➕ ва боз <b>{len(open_positions) - 5}</b> ордери дигар\n"

        if acc_info:
            login = acc_info.get("login", 0)
            balance = acc_info.get("balance", 0.0)
            equity = acc_info.get("equity", 0.0)
            profit = acc_info.get("profit", 0.0)
            margin_level = acc_info.get("margin_level", 0.0)

            text += (
                "━━━━━━━━━━━━━━━━━━━━\n"
                "👤 <b>МАЪЛУМОТИ ҲИСОБ</b>\n"
                f"🔢 Логин: <b>{login}</b>\n"
                f"💰 Баланс: <b>{balance:.2f}$</b>\n"
                f"📈 Сармоя (Equity): <b>{equity:.2f}$</b>\n"
            )
            if profit != 0:
                text += f"💵 Фоидаи кушодаҳо: <b>{profit:+.2f}$</b>\n"
            if margin_level:
                text += f"🛡 Сатҳи маржа: <b>{margin_level:.1f}%</b>\n"

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
            text += (
                "━━━━━━━━━━━━━━━━━━━━\n"
                "📊 <b>САМАРАНОКИИ УМУМӢ</b>\n"
                f"🎯 Сатҳи муваффақият: <b>{win_rate:.1f}%</b>\n"
            )
            if profit_factor:
                text += f"📐 Омили фоида (PF): <b>{profit_factor:.2f}</b>\n"

        text += f"\n⏰ Вақти ҳисобот: {_format_time_only()}\n"

        bot.send_message(
            message.chat.id, text, parse_mode="HTML", reply_markup=_rk_remove()
        )
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАЙЁР КАРДАНИ ҲИСОБОТ ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>\n"
                "💡 Баъди чанд сония дубора кӯшиш кунед."
            ),
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
            (
                "⚠️ <b>БАЛАНС ДАСТРАС НАШУД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Ҳоло маълумоти ҳисоб аз MT5 гирифта нашуд.\n"
                "💡 Лутфан баъди чанд сония дубора кӯшиш кунед."
            ),
            parse_mode="HTML",
            reply_markup=_rk_remove(),
        )
        return
    bot.send_message(
        message.chat.id,
        (
            "💳 <b>БАЛАНСИ ҲИСОБ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Маблағи ҷорӣ: {format_usdt(bal)}"
        ),
        parse_mode="HTML",
        reply_markup=_rk_remove(),
    )


@bot.message_handler(commands=["ai"])
@admin_only_message
def ai_menu_handler(message: telebot.types.Message) -> None:
    bot.send_message(
        message.chat.id,
        (
            "🤖 <b>МЕНЮИ ТАҲЛИЛИ СУНЪӢ (AI)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Намуди таҳлилро интихоб кунед:\n\n"
            "⚡ <b>Скалп</b> — таҳлили фаврӣ барои савдоҳои кӯтоҳ\n"
            "📊 <b>Рӯзона</b> — таҳлили васеъ барои савдоҳои рӯзона"
        ),
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
            (
                f"🔄 <b>ТАҲЛИЛИ AI — {asset} ({style_label})</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "⏳ Дар ҳоли гирифтани маълумоти бозор...\n"
                "🧠 Таҳлил бо системаи сунъӣ оғоз шуд...\n"
                "<i>Лутфан чанд сония интизор шавед.</i>"
            ),
            parse_mode="HTML",
        )

        payload = payload_loader()
        if not payload:
            bot.edit_message_text(
                (
                    f"⚠️ <b>{asset} — {style_label}</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "❌ Маълумоти ҷории бозор дастрас нест.\n\n"
                    "💡 <b>Роҳи ҳалли мушкилот:</b>\n"
                    "• Боварӣ ҳосил кунед, ки MT5 кушода аст\n"
                    f"• Символи <code>{symbol}</code>-ро ба "
                    "Market Watch илова кунед\n"
                    "• Баъдтар дубора кӯшиш кунед"
                ),
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
            (
                f"⚠️ <b>ҲАНГОМИ ТАҲЛИЛИ {asset} ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Навъи таҳлил: <b>{style_label}</b>\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>\n"
                "💡 Лутфан баъди якчанд сония дубора кӯшиш кунед."
            ),
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
            "bullish": "📈 Мусбат",
            "bearish": "📉 Манфӣ",
            "neutral": "⚖️ Бетараф",
        }
        return mapping.get(str(value or "").lower(), "⚖️ Бетараф")

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

    asset_name = "ТИЛЛО" if asset.upper() == "XAU" else ("БИТКОИН" if asset.upper() == "BTC" else asset)

    lines = [
        f"{icon} <b>ТАҲЛИЛИ AI — {asset_name} ({asset})</b>",
        f"📊 Намуди таҳлил: <b>{style_label}</b>",
        f"🎯 Тавсия: {direction} <b>{label}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🧠 Сатҳи боварӣ: <b>{conf_pct}</b>",
        f"🤖 Манбаи таҳлил: <b>{html.escape(provider_display)}</b>"
        + (f" (ҷой: #{model_rank})" if model_rank > 0 else ""),
        f"💻 Модели истифодашуда: <code>{html.escape(model)}</code>",
        f"💎 Символ: <b>{html.escape(symbol)}</b>"
        + (f" | ⏱ {latency_ms} мс" if latency_ms > 0 else ""),
    ]

    if signal in ("BUY", "SELL"):
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("📌 <b>НУҚТАҲОИ ТАВСИЯШАВАНДА</b>")
        if entry is not None:
            lines.append(f"🔹 Нархи вуруд: <code>{_fmt_price(entry)}</code>")
        if stop_loss is not None:
            lines.append(f"🛡 Стоп-Лосс: <code>{_fmt_price(stop_loss)}</code>")
        if take_profit is not None:
            lines.append(f"🎯 Тейк-Профит: <code>{_fmt_price(take_profit)}</code>")
        if rr_value is not None:
            lines.append(f"⚖️ Таносуби Риск/Фоида: <b>{rr_value:.2f}</b>")

    if news:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("📰 <b>ФАЗОИ ХАБАРҲО</b>")
        sentiment = float(news.get("avg_sentiment", 0.0) or 0.0)
        high_impact = int(news.get("high_impact_count", 0) or 0)
        news_status = str(news.get("status", "") or "").strip()
        lines.append(
            f"   Самт: <b>{_news_bias_label(str(news.get('bias', 'neutral')))}</b>"
        )
        lines.append(f"   Эҳсосот: <b>{sentiment:+.2f}</b>")
        lines.append(f"   Хабарҳои муҳим: <b>{high_impact}</b>")
        if news_status:
            lines.append(f"   Ҳолат: <i>{html.escape(news_status)}</i>")
        summary = str(news.get("ai_summary") or news.get("summary_text") or "").strip()
        if summary:
            summary = html.escape(summary.splitlines()[0][:220])
            lines.append(f"🗞 Хулоса: <i>{summary}</i>")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"📝 <b>Таҳлили муфассал:</b>\n{reason}")
    lines.append("")
    lines.append(f"✅ <b>Хулосаи ниҳоӣ: {action}</b>")

    if signal in ("BUY", "SELL") and confidence >= 0.75:
        lines.append("")
        lines.append(
            "💡 <i>Барои кушодани ордер аз тугмаҳои ёрирасони поён истифода кунед.</i>"
        )
    elif signal == "HOLD":
        lines.append("")
        lines.append(
            "⏸ <i>Ҳоло беҳтар аст интизор шавед то сигнали қавитар пайдо шавад.</i>"
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
            (
                "⚠️ <b>ҲАНГОМИ ГИРИФТАНИ ҲОЛАТИ СИСТЕМА ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "💡 Лутфан пайвасти MT5-ро санҷед ва баъди якчанд сония "
                "дубора кӯшиш кунед."
            ),
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

    _send_clean(
        message.chat.id,
        "📋 <b>ОРДЕРҲОИ ҲОЛО КУШОДА</b>\n<i>Дар ҳоли гирифтани маълумот...</i>",
    )
    order_data, total = get_order_by_index(0)

    if not order_data or int(total or 0) == 0:
        bot.send_message(
            message.chat.id,
            (
                "📭 <b>ҲОЛО ЯГОН ОРДЕРИ КУШОДА НЕСТ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "ℹ️ Система дар ҳолати мушоҳида ё оромӣ қарор дорад.\n"
                "Ҳамин ки сигнали мувофиқ пайдо шавад, ордер кушода хоҳад шуд."
            ),
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
                bot.answer_callback_query(
                    call.id, "❌ Ҳангоми иҷро хатогӣ рух дод"
                )
            return

    bot.answer_callback_query(call.id)


@callback_route(r"^orders:nav:(\d+)$")
def cb_orders_nav(call: telebot.types.CallbackQuery, m: re.Match[str]) -> None:
    idx = int(m.group(1))
    order_data, total = get_order_by_index(idx)

    if not order_data or int(total or 0) == 0:
        bot.answer_callback_query(call.id, "⚠️ Ин ордер ҳоло дастрас нест")
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
    bot.answer_callback_query(
        call.id,
        "✅ Ордер бо муваффақият баста шуд" if ok else "❌ Бастан ноком шуд",
    )

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
            text=(
                "📭 <b>ҲАМАИ ОРДЕРҲО БАСТА ШУДАНД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "ℹ️ Ҳоло ягон ордери кушода боқӣ намондааст."
            ),
            parse_mode="HTML",
        )


@callback_route(r"^orders:close_view$")
def cb_orders_close_view(call: telebot.types.CallbackQuery, m: re.Match[str]) -> None:
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text=(
            "🔒 <b>НАМОИШИ ОРДЕРҲО ПӮШИДА ШУД</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "💡 Барои дидани дубора фармони\n"
            "<code>/buttons</code>-ро истифода баред."
        ),
        parse_mode="HTML",
    )
    bot.answer_callback_query(call.id, "Намоиш пӯшида шуд")


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
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАЙЁР КАРДАНИ ҲИСОБОТИ ИМРӮЗА ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
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

            text += f"🎯 Сатҳи муваффақият: <b>{win_rate:.1f}%</b>"
            if profit_factor > 0:
                text += f" | 📐 Омили фоида: <b>{profit_factor:.2f}</b>"
            text += "\n"

        text += f"\n⏰ Вақти ҳисобот: {_format_time_only()}\n"
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАЙЁР КАРДАНИ ҲИСОБОТИ ҲАФТАИНА ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
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

            text += f"🎯 Сатҳи муваффақият: <b>{win_rate:.1f}%</b>"
            if profit_factor > 0:
                text += f" | 📐 Омили фоида: <b>{profit_factor:.2f}</b>"
            text += "\n"

        text += f"\n⏰ Вақти ҳисобот: {_format_time_only()}\n"
        bot.send_message(message.chat.id, text, parse_mode="HTML")
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ТАЙЁР КАРДАНИ ҲИСОБОТИ МОҲОНА ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
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

    lines = [
        f"{status_emoji} <b>БАСТАНИ ҲАМАИ ОРДЕРҲО</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🔒 Баста шуд: <b>{closed}</b> ордер",
    ]
    if canceled > 0:
        lines.append(f"🗑 Бекор карда шуд: <b>{canceled}</b> ордер")
    if closed == 0 and canceled == 0 and ok:
        lines.append("ℹ️ Ҳоло ягон ордери кушода вуҷуд надошт")

    errs = list(res.get("errors") or [])
    if errs:
        preview = " • ".join(str(e)[:25] for e in errs[:2])
        lines.append(f"⚠️ Огоҳиҳо: <code>{html.escape(preview)}</code>")

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
    try:
        summary_val = float(summary)
    except Exception:
        summary_val = 0.0
    emoji = "🟢" if summary_val >= 0 else "🔴"
    bot.send_message(
        message.chat.id,
        (
            "📊 <b>ХУЛОСАИ ПОЗИЦИЯҲОИ КУШОДА</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} Фоида/Зиёни ҷамъи ҳамаи позицияҳо: "
            f"{format_usdt(summary_val)}"
        ),
        parse_mode="HTML",
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
    bot.send_message(
        message.chat.id,
        (
            "💳 <b>БАЛАНСИ ҲИСОБ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Маблағи ҷорӣ: {format_usdt(balance)}"
        ),
        parse_mode="HTML",
    )


def handle_trade_start(message: telebot.types.Message) -> None:
    if not _debounce_check(f"start:{message.chat.id}"):
        bot.send_message(
            message.chat.id,
            (
                "⏳ <b>Амали қаблӣ ҳанӯз дар ҳоли коркард аст</b>\n"
                "Лутфан чанд сония интизор шавед..."
            ),
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
                (
                    "ℹ️ <b>СИСТЕМА АЛЛАКАЙ ФАЪОЛ АСТ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "✅ Савдои автоматӣ дар ҳолати кор қарор дорад.\n"
                    "Ҳеҷ амали иловагӣ лозим нест."
                ),
                parse_mode="HTML",
            )
            return

        engine.clear_manual_stop()
        st_after_clear = engine.status()
        if bool(getattr(st_after_clear, "manual_stop", False)):
            bot.send_message(
                message.chat.id,
                (
                    "🛡 <b>САВДО АЗ ТАРАФИ СИСТЕМАИ ҲИФЗ МАНЪ АСТ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Системаи муҳофизати риск ҳоло оғози савдоро иҷозат намедиҳад.\n"
                    "💡 Лутфан баъди чанд дақиқа дубора кӯшиш кунед."
                ),
                parse_mode="HTML",
            )
            return
        started = bool(engine.start())
        if started:
            bot.send_message(
                message.chat.id,
                (
                    "✅ <b>САВДОИ АВТОМАТӢ БО МУВАФФАҚИЯТ ОҒОЗ ШУД</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "🚀 Система акнун фаъол аст ва ҳолати бозорро назорат мекунад.\n"
                    "📡 Ҳангоми пайдо шудани сигнали дуруст ордер кушода хоҳад шуд.\n\n"
                    "💡 <i>Барои қатъ кардани савдо тугмаи "
                    "«🛑 Қатъи Тиҷорат»-ро истифода баред.</i>"
                ),
                parse_mode="HTML",
            )
            return

        _, reason = mt5_status()
        gate_reason = str(getattr(engine, "_gate_last_reason", "") or "")
        blocked_assets = sorted(set(getattr(engine, "_blocked_assets", []) or []))
        model_ready = bool(getattr(engine, "_model_loaded", False)) and bool(
            getattr(engine, "_backtest_passed", False)
        )
        reason_raw = str(reason or "сабаб номаълум")
        if (not model_ready) or blocked_assets:
            reason_s = gate_reason or "санҷиши омодагии модел ноком шуд"
            if blocked_assets:
                reason_s = (
                    f"{reason_s} | активҳои манъшуда: {', '.join(blocked_assets)}"
                )
        else:
            reason_s = reason_raw
        if reason_s.startswith("blocked_cooldown:"):
            reason_s = reason_s.split(":", 1)[1]
        hint = ""
        if "algo_trading_disabled_in_terminal" in reason_s:
            hint = (
                "\n\n💡 <b>Роҳи ҳалли мушкилот:</b>\n"
                "1️⃣ Терминали MT5-ро кушоед\n"
                "2️⃣ Ба менюи: <code>Tools → Options → Expert Advisors</code>\n"
                "3️⃣ Сатри «Allow Algo Trading»-ро фаъол созед\n"
                "4️⃣ Тугмаи «AutoTrading»-ро сабз гардонед"
            )
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>САВДО ОҒОЗ НАШУД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Сабаб: <code>{html.escape(reason_s)}</code>"
                f"{hint}"
            ),
            parse_mode="HTML",
        )
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ОҒОЗИ САВДО ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
        )


def handle_trade_stop(message: telebot.types.Message) -> None:
    if not _debounce_check(f"stop:{message.chat.id}"):
        bot.send_message(
            message.chat.id,
            (
                "⏳ <b>Амали қаблӣ ҳанӯз дар ҳоли коркард аст</b>\n"
                "Лутфан чанд сония интизор шавед..."
            ),
            parse_mode="HTML",
        )
        return
    try:
        st = engine.status()
        was_active = engine.request_manual_stop()
        if was_active:
            bot.send_message(
                message.chat.id,
                (
                    "🛑 <b>САВДОИ АВТОМАТӢ ҚАТЪ ГАРДИД</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "👁 Система ба <b>ҳолати мушоҳида</b> гузашт.\n\n"
                    "ℹ️ <i>Савдо муваққатан манъ аст, вале таҳлили AI ва "
                    "сигналҳо омада истодаанд. Барои оғози дубора тугмаи "
                    "«🚀 Оғози Тиҷорат»-ро пахш кунед.</i>"
                ),
                parse_mode="HTML",
            )
        elif bool(getattr(st, "manual_stop", False)):
            bot.send_message(
                message.chat.id,
                (
                    "ℹ️ <b>САВДО АЛЛАКАЙ ҚАТЪ АСТ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "👁 Система дар ҳолати мушоҳида қарор дорад."
                ),
                parse_mode="HTML",
            )
        else:
            bot.send_message(
                message.chat.id,
                (
                    "ℹ️ <b>СИСТЕМА АЛЛАКАЙ ХОМӮШ АСТ</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "Савдои автоматӣ ҳоло фаъол нест."
                ),
                parse_mode="HTML",
            )
    except Exception as exc:
        bot.send_message(
            message.chat.id,
            (
                "⚠️ <b>ҲАНГОМИ ҚАТЪИ САВДО ХАТОГӢ РУХ ДОД</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💬 Тафсилот: <code>{html.escape(str(exc))}</code>"
            ),
            parse_mode="HTML",
        )


def handle_engine_check(message: telebot.types.Message) -> None:
    status = engine.status()
    mt5_conn = "🟢 Пайваст" if status.connected else "🔴 Қатъ шудааст"
    trading_icon = "✅ Фаъол" if status.trading else "⏸ Хомӯш"
    manual_stop_icon = "🛑 Фаъол" if status.manual_stop else "⚪ Ғайрифаъол"
    gate_val = str(getattr(status, "gate_reason", "") or "").strip() or "мушкилот нест"
    halt_val = (
        str(getattr(status, "risk_halt_reason", "") or "").strip() or "мушкилот нест"
    )
    controller = str(getattr(status, "controller_state", "номаълум"))
    chaos = str(getattr(status, "chaos_state", "номаълум"))
    bot.send_message(
        message.chat.id,
        (
            "⚙️ <b>ҲОЛАТИ МУФАССАЛИ МУҲАРРИК</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 Пайваст ба MT5: <b>{mt5_conn}</b>\n"
            f"📈 Савдои худкор: <b>{trading_icon}</b>\n"
            f"🛑 Қатъи дастӣ: <b>{manual_stop_icon}</b>\n"
            f"🎯 Активи фаъол: <b>{html.escape(str(status.active_asset))}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"📉 Коҳиши сармоя (DD): <b>{status.dd_pct * 100:.2f}%</b>\n"
            f"💵 Фоидаи имрӯза: <b>{status.today_pnl:+.2f}$</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🥇 Позицияҳои XAU: <b>{status.open_trades_xau}</b>\n"
            f"₿ Позицияҳои BTC: <b>{status.open_trades_btc}</b>\n"
            f"📡 Сигнали охирини XAU: <b>"
            f"{html.escape(str(status.last_signal_xau))}</b>\n"
            f"📡 Сигнали охирини BTC: <b>"
            f"{html.escape(str(status.last_signal_btc))}</b>\n"
            f"📥 Навбати иҷро: <b>{status.exec_queue_size}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🧭 Ҳолати контроллер: <b>{html.escape(controller)}</b>\n"
            f"🧪 Ҳолати тестӣ: <b>{html.escape(chaos)}</b>\n"
            f"🚫 Филтри сигнал: <code>{html.escape(gate_val)}</code>\n"
            f"🛡 Филтри риск: <code>{html.escape(halt_val)}</code>"
        ),
        parse_mode="HTML",
    )


def handle_full_check(message: telebot.types.Message) -> None:
    bot.send_message(
        message.chat.id,
        (
            "🔄 <b>САНҶИШИ ПУРРАИ СИСТЕМА ОҒОЗ ШУД...</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⏳ Лутфан чанд сония интизор шавед.\n"
            "🔍 Санҷиши модулҳо, пайваст ва риск..."
        ),
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
                (
                    "⚠️ <b>ҲАНГОМИ ИҶРОИ АМАЛ ХАТОГӢ РУХ ДОД</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "💡 Лутфан баъди чанд сония дубора кӯшиш кунед."
                ),
                parse_mode="HTML",
            )
        return

    bot.send_message(
        message.chat.id,
        (
            "❓ <b>АМАЛИ НОМАЪЛУМ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Ман ин паёмро нафаҳмидам.\n"
            "💡 Барои кушодани менюи асосӣ фармони\n"
            "<code>/buttons</code>-ро истифода баред."
        ),
        parse_mode="HTML",
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
