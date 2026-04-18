"""Public package API for ``ExnessAPI``.

This module intentionally keeps package import side effects close to zero:

- no eager MetaTrader/IO imports at package import time
- no wildcard re-exports from implementation modules
- a single, explicit public surface for stable downstream usage

Heavy modules are imported lazily on first attribute access via ``__getattr__``.
That keeps startup fast, reduces the chance of circular imports, and gives us a
clear place to control the public contract as the package grows.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final


_PUBLIC_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    # Execution primitives
    "OrderExecutor": (".order_execution", "OrderExecutor"),
    "OrderRequest": (".order_execution", "OrderRequest"),
    "OrderResult": (".order_execution", "OrderResult"),
    # Broker/account helpers
    "enable_trading": (".functions", "enable_trading"),
    "get_balance": (".functions", "get_balance"),
    "get_account_info": (".functions", "get_account_info"),
    "get_positions_summary": (".functions", "get_positions_summary"),
    "get_order_by_index": (".functions", "get_order_by_index"),
    "get_all_open_positions": (".functions", "get_all_open_positions"),
    "has_open_positions": (".functions", "has_open_positions"),
    "market_is_open": (".functions", "market_is_open"),
    "close_order": (".functions", "close_order"),
    "close_all_position": (".functions", "close_all_position"),
    "close_all_position_by_profit": (".functions", "close_all_position_by_profit"),
    "set_takeprofit_all_positions_usd": (
        ".functions",
        "set_takeprofit_all_positions_usd",
    ),
    "set_stoploss_all_positions_usd": (
        ".functions",
        "set_stoploss_all_positions_usd",
    ),
    "manual_open_capacity": (".functions", "manual_open_capacity"),
    "open_buy_order_btc": (".functions", "open_buy_order_btc"),
    "open_buy_order_xau": (".functions", "open_buy_order_xau"),
    "open_sell_order_btc": (".functions", "open_sell_order_btc"),
    "open_sell_order_xau": (".functions", "open_sell_order_xau"),
    # Reporting/history helpers
    "format_usdt": (".history", "format_usdt"),
    "get_profit_period": (".history", "get_profit_period"),
    "get_day_profit": (".history", "get_day_profit"),
    "get_day_loss": (".history", "get_day_loss"),
    "get_day_net": (".history", "get_day_net"),
    "get_week_profit": (".history", "get_week_profit"),
    "get_week_loss": (".history", "get_week_loss"),
    "get_week_net": (".history", "get_week_net"),
    "get_month_profit": (".history", "get_month_profit"),
    "get_month_loss": (".history", "get_month_loss"),
    "get_month_net": (".history", "get_month_net"),
    "view_all_history_dict": (".history", "view_all_history_dict"),
    "get_full_report_day": (".functions", "get_full_report_day"),
    "get_full_report_week": (".functions", "get_full_report_week"),
    "get_full_report_month": (".functions", "get_full_report_month"),
    "get_full_report_all": (".functions", "get_full_report_all"),
    # Daily balance / phase anchors
    "DailyRow": (".daily_balance", "DailyRow"),
    "initialize_daily_balance": (".daily_balance", "initialize_daily_balance"),
    "get_peak_balance": (".daily_balance", "get_peak_balance"),
    "update_peak_equity": (".daily_balance", "update_peak_equity"),
}

__all__ = list(_PUBLIC_EXPORTS)


if TYPE_CHECKING:
    from .daily_balance import (
        DailyRow as DailyRow,
        get_peak_balance as get_peak_balance,
        initialize_daily_balance as initialize_daily_balance,
        update_peak_equity as update_peak_equity,
    )
    from .functions import (
        close_all_position as close_all_position,
        close_all_position_by_profit as close_all_position_by_profit,
        close_order as close_order,
        enable_trading as enable_trading,
        get_account_info as get_account_info,
        get_all_open_positions as get_all_open_positions,
        get_balance as get_balance,
        get_full_report_all as get_full_report_all,
        get_full_report_day as get_full_report_day,
        get_full_report_month as get_full_report_month,
        get_full_report_week as get_full_report_week,
        get_order_by_index as get_order_by_index,
        get_positions_summary as get_positions_summary,
        has_open_positions as has_open_positions,
        manual_open_capacity as manual_open_capacity,
        market_is_open as market_is_open,
        open_buy_order_btc as open_buy_order_btc,
        open_buy_order_xau as open_buy_order_xau,
        open_sell_order_btc as open_sell_order_btc,
        open_sell_order_xau as open_sell_order_xau,
        set_stoploss_all_positions_usd as set_stoploss_all_positions_usd,
        set_takeprofit_all_positions_usd as set_takeprofit_all_positions_usd,
    )
    from .history import (
        format_usdt as format_usdt,
        get_day_loss as get_day_loss,
        get_day_net as get_day_net,
        get_day_profit as get_day_profit,
        get_month_loss as get_month_loss,
        get_month_net as get_month_net,
        get_month_profit as get_month_profit,
        get_profit_period as get_profit_period,
        get_week_loss as get_week_loss,
        get_week_net as get_week_net,
        get_week_profit as get_week_profit,
        view_all_history_dict as view_all_history_dict,
    )
    from .order_execution import (
        OrderExecutor as OrderExecutor,
        OrderRequest as OrderRequest,
        OrderResult as OrderResult,
    )


def __getattr__(name: str):
    """Lazily resolve public exports from their implementation modules."""
    try:
        module_name, attr_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to IDEs, REPLs, and introspection tools."""
    return sorted(set(globals()) | set(__all__))
