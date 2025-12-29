from typing import Literal, Dict, Any, List
from datetime import datetime, timedelta
import MetaTrader5 as mt5


_cache = None
_cache_time = None


def _connect() -> bool:
    if not mt5.terminal_info() or not mt5.terminal_info().trade_allowed:
        print("АвтоТорговля дар MT5 хомӯш аст! Лутфан онро фаъол кунед.")
        return False
    return True

def get_profit_period(period: Literal["day", "week", "month"]) -> dict:
    if not _connect():
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    utc_now = datetime.utcnow() + timedelta(hours=5)

    if period == "day":
        from_date = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        from_date = utc_now - timedelta(days=utc_now.weekday())
        from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        from_date = utc_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return {"profit": 0.0, "loss": 0.0, "net": 0.0}

    deals = mt5.history_deals_get(from_date, datetime.utcnow() + timedelta(hours=10)) 

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
    """
    Маълумоти дақиқи ҳаррӯзаро аз ҳисоби Exness (MT5) мегирад.
    - force_refresh=True → ҳатман аз MT5 навсозӣ мекунад (барои /history).
    - Кэш фақат 5 сония нигоҳ дошта мешавад, то маълумот ҳамеша тоза бошад.
    - Ҳамаи ордерҳои имрӯз (кушода + баста) + P&L нореалӣ.
    - Хатоҳо лог мешаванд, аммо функсия ҳамеша dict бармегардонад.
    """
    global _cache, _cache_time

    now = datetime.now()

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

        # Диапозони имрӯз (аз нисфишаб то ҳозир + 2 соат барои ҳамаи минтақаҳо)
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
        from_date = today_start - timedelta(hours=3)  # Барои бехатарӣ
        to_date = now + timedelta(hours=2)

        # Гирифтани таърих
        deals = mt5.history_deals_get(from_date.timestamp(), to_date.timestamp())

        # Ордерҳои кушода
        open_positions = mt5.positions_get() or []

        # Баланс
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

        # Гурӯҳбандии deal-ҳо аз рӯи position_id
        trades: Dict[int, Dict] = {}
        for d in deals:
            pid = d.position_id
            if pid not in trades:
                trades[pid] = {"entry": None, "exit": None}
            if d.entry == mt5.DEAL_ENTRY_IN:
                trades[pid]["entry"] = d
            elif d.entry == mt5.DEAL_ENTRY_OUT:
                trades[pid]["exit"] = d

        # Маълумот барои ҳисобот
        wins = losses = total_closed = 0
        total_profit = total_loss = net_total = 0.0
        unrealized_pnl = sum(float(p.profit) for p in open_positions)

        records = []

        for pid, t in trades.items():
            entry = t["entry"]
            exit_deal = t["exit"]
            if not entry:
                continue

            # Агар ордер кушода бошад
            open_pos = next((p for p in open_positions if p.position_id == pid), None)
            if open_pos:
                profit = float(open_pos.profit)
                status = "open"
                total_open = len(open_positions)
            else:
                if not exit_deal:
                    continue  # Нодуруст — набояд рӯй диҳад
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

        # Ҷамъбаст
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
            "records": records[:50],  # Фақат охирин 50 барои нигоҳдорӣ
        }

        _cache = summary
        _cache_time = now
        return summary

    # Баргардонидани кэш (фақат агар хеле тоза бошад)
    if _cache and _cache_time and (now - _cache_time).seconds < 5:
        return _cache

    return view_all_history_dict(force_refresh=True)  # Агар чизе хато шавад — навсозӣ




