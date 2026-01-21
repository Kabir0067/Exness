from typing import Literal, Dict, Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

import MetaTrader5 as mt5

from mt5_client import ensure_mt5, MT5_LOCK


_cache: Dict[str, Any] | None = None
_cache_time: datetime | None = None
_cache_mono: float | None = None

_conn_ok: bool = False
_conn_mono: float = 0.0
_last_trade_disabled_print_mono: float = 0.0

_TZ_NAME = "Asia/Dushanbe"
_TZ: ZoneInfo | None = None

_CACHE_TTL_SEC = 5.0
_CONN_TTL_SEC = 1.0
_PRINT_THROTTLE_SEC = 60.0


def _local_now() -> datetime:
    global _TZ
    if _TZ is None:
        try:
            _TZ = ZoneInfo(_TZ_NAME)
        except Exception:
            _TZ = None
    try:
        return datetime.now(_TZ) if _TZ else datetime.now()
    except Exception:
        return datetime.now()


def _day_start_local(now: datetime) -> datetime:
    try:
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        return datetime(now.year, now.month, now.day, 0, 0, 0)


def _naive_local(dt: datetime) -> datetime:
    # MT5 history APIs accept naive datetime (terminal local time).
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def _connect() -> bool:
    global _conn_ok, _conn_mono, _last_trade_disabled_print_mono

    now_mono = time.monotonic()
    if (now_mono - _conn_mono) < _CONN_TTL_SEC:
        return bool(_conn_ok)

    ok = False
    try:
        ensure_mt5()
        with MT5_LOCK:
            term = mt5.terminal_info()
            ok = bool(term and getattr(term, "trade_allowed", False))
    except Exception:
        ok = False

    if not ok:
        # Throttle prints (fast + non-spam)
        if (now_mono - _last_trade_disabled_print_mono) >= _PRINT_THROTTLE_SEC:
            _last_trade_disabled_print_mono = now_mono
            try:
                print("АвтоТорговля дар MT5 хомӯш аст! Лутфан онро фаъол кунед.")
            except Exception:
                pass

    _conn_ok = bool(ok)
    _conn_mono = now_mono
    return bool(ok)


def get_profit_period(period: Literal["day", "week", "month"]) -> dict:
    if not _connect():
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    local_now = _local_now()

    if period == "day":
        from_date = _day_start_local(local_now)
    elif period == "week":
        from_date = local_now - timedelta(days=int(local_now.weekday()))
        from_date = _day_start_local(from_date)
    elif period == "month":
        try:
            from_date = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            from_date = datetime(local_now.year, local_now.month, 1, 0, 0, 0)
    else:
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    with MT5_LOCK:
        deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(local_now))

    total_profit = 0.0
    total_loss = 0.0

    if deals:
        entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
        for d in deals:
            try:
                if getattr(d, "entry", None) != entry_out:
                    continue
                p = float(getattr(d, "profit", 0.0) or 0.0)
                if p > 0.0:
                    total_profit += p
                elif p < 0.0:
                    total_loss += p  # keep negative (compat)
            except Exception:
                continue

    net = round(total_profit + total_loss, 2)
    total_profit = round(total_profit, 2)
    total_loss = round(total_loss, 2)

    return {"profit": total_profit, "loss": total_loss, "net": net}


def get_day_profit() -> float:
    return float(get_profit_period("day")["profit"])


def get_day_loss() -> float:
    return float(get_profit_period("day")["loss"])


def get_day_net() -> float:
    return float(get_profit_period("day")["net"])


def get_week_profit() -> float:
    return float(get_profit_period("week")["profit"])


def get_week_loss() -> float:
    return float(get_profit_period("week")["loss"])


def get_week_net() -> float:
    return float(get_profit_period("week")["net"])


def get_month_profit() -> float:
    return float(get_profit_period("month")["profit"])


def get_month_loss() -> float:
    return float(get_profit_period("month")["loss"])


def get_month_net() -> float:
    return float(get_profit_period("month")["net"])


def format_usdt(amount: float, show_plus: bool = False) -> str:
    try:
        if amount is None or float(amount) == 0.0:
            return "0.00 USDT"
        amt = float(amount)
    except Exception:
        return "0.00 USDT"

    prefix = ""
    if show_plus and amt > 0:
        prefix = "+"
    elif amt < 0:
        prefix = "-"

    abs_amount = abs(amt)
    return f"{prefix}{abs_amount:,.2f} USDT".replace(",", " ")


def view_all_history_dict(force_refresh: bool = False) -> Dict[str, Any]:
    global _cache, _cache_time, _cache_mono

    now = _local_now()
    now_mono = time.monotonic()

    need_refresh = (
        bool(force_refresh)
        or _cache is None
        or _cache_time is None
        or _cache_mono is None
        or (now_mono - float(_cache_mono)) > _CACHE_TTL_SEC
    )

    if not need_refresh:
        return _cache  # type: ignore[return-value]

    if not _connect():
        empty_summary = {
            "date": now.strftime("%Y-%m-%d"),
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": 0.0,
            "records": [],
        }
        _cache = empty_summary
        _cache_time = now
        _cache_mono = now_mono
        return empty_summary

    today_start = _day_start_local(now)
    from_date = today_start
    to_date = now

    with MT5_LOCK:
        deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(to_date))
        open_positions = mt5.positions_get() or []
        acc = mt5.account_info()

    balance = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0

    if not deals and not open_positions:
        empty_summary = {
            "date": now.strftime("%Y-%m-%d"),
            "wins": 0,
            "losses": 0,
            "total_closed": 0,
            "total_open": 0,
            "unrealized_pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "net": 0.0,
            "balance": round(balance, 2),
            "records": [],
        }
        _cache = empty_summary
        _cache_time = now
        _cache_mono = now_mono
        return empty_summary

    # Map open positions by position_id (fallback to ticket)
    open_map: Dict[int, Any] = {}
    for p in open_positions:
        try:
            pid = int(getattr(p, "position_id", 0) or 0)
            if pid == 0:
                pid = int(getattr(p, "ticket", 0) or 0)
            if pid:
                open_map[pid] = p
        except Exception:
            continue

    # Build trades by position_id: keep first IN, last OUT (best-effort)
    trades: Dict[int, Dict[str, Any]] = {}
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)

    if deals:
        for d in deals:
            try:
                pid = int(getattr(d, "position_id", 0) or 0)
                if pid == 0:
                    continue
                e = getattr(d, "entry", None)
                if e not in (entry_in, entry_out):
                    continue
                rec = trades.get(pid)
                if rec is None:
                    rec = {"entry": None, "exit": None, "entry_t": None, "exit_t": None}
                    trades[pid] = rec

                t_msc = int(getattr(d, "time_msc", 0) or 0)
                if e == entry_in:
                    if rec["entry"] is None or (rec["entry_t"] is not None and t_msc < int(rec["entry_t"])):
                        rec["entry"] = d
                        rec["entry_t"] = t_msc
                else:
                    if rec["exit"] is None or (rec["exit_t"] is not None and t_msc > int(rec["exit_t"])):
                        rec["exit"] = d
                        rec["exit_t"] = t_msc
            except Exception:
                continue

    wins = 0
    losses = 0
    total_closed = 0
    total_profit = 0.0
    total_loss = 0.0
    net_total = 0.0

    # Unrealized PnL from open positions
    unrealized_pnl = 0.0
    for p in open_positions:
        try:
            unrealized_pnl += float(getattr(p, "profit", 0.0) or 0.0)
        except Exception:
            continue

    records = []
    total_open = int(len(open_positions))

    for pid, t in trades.items():
        entry = t.get("entry")
        if entry is None:
            continue

        try:
            entry_volume = float(getattr(entry, "volume", 0.0) or 0.0)
        except Exception:
            entry_volume = 0.0

        open_pos = open_map.get(int(pid))
        if open_pos is not None:
            try:
                profit = float(getattr(open_pos, "profit", 0.0) or 0.0)
            except Exception:
                profit = 0.0
            status = "open"
        else:
            exit_deal = t.get("exit")
            if exit_deal is None:
                continue
            try:
                profit = float(getattr(exit_deal, "profit", 0.0) or 0.0)
            except Exception:
                profit = 0.0
            status = "closed"
            total_closed += 1
            net_total += profit
            if profit > 0:
                wins += 1
                total_profit += profit
            elif profit < 0:
                losses += 1
                total_loss += abs(profit)

        records.append(
            {
                "id": int(pid),
                "status": status,
                "profit": profit if profit != 0 else None,
                "lot": entry_volume,
            }
        )

    summary = {
        "date": now.strftime("%Y-%m-%d"),
        "wins": int(wins),
        "losses": int(losses),
        "total_closed": int(total_closed),
        "total_open": int(total_open),
        "unrealized_pnl": round(float(unrealized_pnl), 2),
        "profit": round(float(total_profit), 2),
        "loss": round(float(total_loss), 2),
        "net": round(float(net_total), 2),
        "balance": round(float(balance), 2),
        "records": records[:50],
    }

    _cache = summary
    _cache_time = now
    _cache_mono = now_mono
    return summary

