from typing import Literal, Dict, Any, List
from datetime import datetime, timedelta
import os
from zoneinfo import ZoneInfo
import MetaTrader5 as mt5

from mt5_client import ensure_mt5, MT5_LOCK


_cache = None
_cache_time = None


def _local_now() -> datetime:
    tz_name = os.getenv("TIMEZONE", "Asia/Dushanbe")
    try:
        return datetime.now(ZoneInfo(tz_name))
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
    try:
        ensure_mt5()
        with MT5_LOCK:
            term = mt5.terminal_info()
            if not term or not term.trade_allowed:
                print("АвтоТорговля дар MT5 хомӯш аст! Лутфан онро фаъол кунед.")
                return False
        return True
    except Exception:
        return False

def get_profit_period(period: Literal["day", "week", "month"]) -> dict:
    if not _connect():
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    local_now = _local_now()

    if period == "day":
        from_date = _day_start_local(local_now)
    elif period == "week":
        from_date = local_now - timedelta(days=local_now.weekday())
        from_date = _day_start_local(from_date)
    elif period == "month":
        from_date = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    with MT5_LOCK:
        deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(local_now))

    total_profit = 0.0
    total_loss = 0.0

    if deals:
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT: 
                if deal.profit > 0:
                    total_profit += deal.profit
                elif deal.profit < 0:
                    total_loss += deal.profit  

    net = round(total_profit + total_loss, 2)
    total_profit = round(total_profit, 2)
    total_loss = round(total_loss, 2)

    return {
        "profit": total_profit,
        "loss": total_loss,
        "net": net
    }

def get_day_profit() -> float:
    return get_profit_period("day")["profit"]

def get_day_loss() -> float:
    return get_profit_period("day")["loss"]

def get_day_net() -> float:
    return get_profit_period("day")["net"]

def get_week_profit() -> float:
    return get_profit_period("week")["profit"]

def get_week_loss() -> float:
    return get_profit_period("week")["loss"]

def get_week_net() -> float:
    return get_profit_period("week")["net"]

def get_month_profit() -> float:
    return get_profit_period("month")["profit"]

def get_month_loss() -> float:
    return get_profit_period("month")["loss"]

def get_month_net() -> float:
    return get_profit_period("month")["net"]


def format_usdt(amount: float, show_plus: bool = False) -> str:
    if amount is None or amount == 0:
        return "0.00 USDT"

    prefix = ""
    if show_plus and amount > 0:
        prefix = "+"
    elif amount < 0:
        prefix = "-"

    abs_amount = abs(amount)
    return f"{prefix}{abs_amount:,.2f} USDT".replace(",", " ")




def view_all_history_dict(force_refresh: bool = False) -> Dict[str, Any]:
    global _cache, _cache_time

    now = _local_now()

    if force_refresh or not _cache or not _cache_time or (now - _cache_time).seconds > 5:
        if not _connect():
            empty_summary = {
                "date": now.strftime("%Y-%m-%d"),
                "wins": 0, "losses": 0, "total_closed": 0,
                "total_open": 0, "unrealized_pnl": 0.0,
                "profit": 0.0, "loss": 0.0, "net": 0.0,
                "balance": 0.0,
                "records": [],
            }
            _cache = empty_summary
            _cache_time = now
            return empty_summary

        today_start = _day_start_local(now)
        from_date = today_start
        to_date = now

        with MT5_LOCK:
            deals = mt5.history_deals_get(_naive_local(from_date), _naive_local(to_date))
            open_positions = mt5.positions_get() or []
            acc = mt5.account_info()
        balance = float(acc.balance) if acc else 0.0

        if not deals and not open_positions:
            empty_summary = {
                "date": now.strftime("%Y-%m-%d"),
                "wins": 0, "losses": 0, "total_closed": 0,
                "total_open": 0, "unrealized_pnl": 0.0,
                "profit": 0.0, "loss": 0.0, "net": 0.0,
                "balance": round(balance, 2),
                "records": [],
            }
            _cache = empty_summary
            _cache_time = now
            return empty_summary

        trades: Dict[int, Dict] = {}
        for d in deals:
            pid = d.position_id
            if pid not in trades:
                trades[pid] = {"entry": None, "exit": None}
            if d.entry == mt5.DEAL_ENTRY_IN:
                trades[pid]["entry"] = d
            elif d.entry == mt5.DEAL_ENTRY_OUT:
                trades[pid]["exit"] = d

        wins = losses = total_closed = 0
        total_profit = total_loss = net_total = 0.0
        unrealized_pnl = sum(float(p.profit) for p in open_positions)

        records = []

        for pid, t in trades.items():
            entry = t["entry"]
            exit_deal = t["exit"]
            if not entry:
                continue

            open_pos = next((p for p in open_positions if p.position_id == pid), None)
            if open_pos:
                profit = float(open_pos.profit)
                status = "open"
                total_open = len(open_positions)
            else:
                if not exit_deal:
                    continue  
                profit = float(exit_deal.profit)
                status = "closed"
                total_closed += 1
                net_total += profit
                if profit > 0:
                    wins += 1
                    total_profit += profit
                elif profit < 0:
                    losses += 1
                    total_loss += abs(profit)

            records.append({
                "id": pid,
                "status": status,
                "profit": profit if profit != 0 else None,
                "lot": float(entry.volume),
            })

        summary = {
            "date": now.strftime("%Y-%m-%d"),
            "wins": wins,
            "losses": losses,
            "total_closed": total_closed,
            "total_open": len(open_positions),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "profit": round(total_profit, 2),
            "loss": round(total_loss, 2),
            "net": round(net_total, 2),
            "balance": round(balance, 2),
            "records": records[:50],  
        }

        _cache = summary
        _cache_time = now
        return summary

    if _cache and _cache_time and (now - _cache_time).seconds < 5:
        return _cache

    return view_all_history_dict(force_refresh=True) 




