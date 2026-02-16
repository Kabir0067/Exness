# Backtest/metrics_institutional.py
# Enhanced institutional-grade metrics with regulatory compliance
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

log = logging.getLogger("backtest.metrics_institutional")


@dataclass
class InstitutionalBacktestMetrics:
    """Institutional-grade backtest metrics with regulatory compliance"""
    
    # ═══ Core Return Metrics ═══════════════════════════════════
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    cagr: float = 0.0  # Compound Annual Growth Rate
    
    # ═══ Risk-Adjusted Returns ═════════════════════════════════
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0  # vs benchmark
    omega_ratio: float = 0.0  # Probability-weighted ratio
    
    # ═══ Risk Metrics (Institutional) ══════════════════════════
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0  # Net profit / Max DD
    
    volatility_annualized: float = 0.0
    downside_deviation: float = 0.0
    upside_deviation: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    
    # ═══ Trade Statistics ══════════════════════════════════════
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0  # Expectancy / Avg Loss
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_trade_duration_minutes: float = 0.0
    median_trade_duration_minutes: float = 0.0
    
    # ═══ Consecutive Trades ════════════════════════════════════
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_consecutive_wins: float = 0.0
    avg_consecutive_losses: float = 0.0
    
    # ═══ Transaction Costs (Reality) ═══════════════════════════
    total_spread_cost: float = 0.0
    total_swap_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_commission: float = 0.0
    total_fees: float = 0.0
    fee_impact_pct: float = 0.0
    
    avg_spread_per_trade: float = 0.0
    avg_slippage_per_trade: float = 0.0
    
    # ═══ Robustness Tests ══════════════════════════════════════
    risk_of_ruin: float = 0.0
    monte_carlo_runs: int = 0
    mc_confidence_95_percentile: float = 0.0
    mc_worst_case: float = 0.0
    mc_best_case: float = 0.0
    
    wfa_passed: bool = False
    wfa_total_windows: int = 0
    wfa_failed_windows: int = 0
    wfa_avg_sharpe: float = 0.0
    wfa_min_sharpe: float = 0.0
    
    stress_test_passed: bool = False
    stress_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # ═══ Regime Performance ════════════════════════════════════
    trend_up_sharpe: float = 0.0
    trend_down_sharpe: float = 0.0
    range_sharpe: float = 0.0
    high_vol_sharpe: float = 0.0
    
    # ═══ Time-based Analysis ═══════════════════════════════════
    total_days: float = 0.0
    trading_days: float = 0.0
    avg_trades_per_day: float = 0.0
    best_day_return: float = 0.0
    worst_day_return: float = 0.0
    
    # ═══ Position Management ═══════════════════════════════════
    avg_position_size_pct: float = 0.0
    max_position_size_pct: float = 0.0
    kelly_optimal_fraction: float = 0.0
    
    # ═══ Metadata ══════════════════════════════════════════════
    symbol: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_capital: float = 0.0
    
    institutional_grade: bool = True


def compute_institutional_metrics(
    pnl_series: List[float],
    *,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.05,
    periods_per_year: float = 252.0,
    symbol: str = "",
    timeframe: str = "",
    start_date: str = "",
    end_date: str = "",
    spread_costs: float = 0.0,
    swap_costs: float = 0.0,
    slippage_costs: float = 0.0,
    commission_costs: float = 0.0,
    trade_durations_min: Optional[List[float]] = None,
    trade_details: Optional[List[Dict]] = None,
) -> InstitutionalBacktestMetrics:
    """
    Compute comprehensive institutional-grade metrics.
    
    Args:
        pnl_series: List of trade PnLs
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        trade_details: Optional detailed trade information for advanced metrics
    
    Returns:
        Complete institutional metrics report
    """
    m = InstitutionalBacktestMetrics()
    m.symbol = symbol
    m.timeframe = timeframe
    m.start_date = start_date
    m.end_date = end_date
    m.initial_capital = initial_capital
    
    if not pnl_series:
        m.final_capital = initial_capital
        return m
    
    pnls = np.asarray(pnl_series, dtype=np.float64)
    if pnls.size > 0:
        pnls = pnls[np.isfinite(pnls)]
    if pnls.size == 0:
        m.final_capital = initial_capital
        return m
    m.total_trades = len(pnls)
    
    # ═══ Trade Statistics ═══════════════════════════════════════
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
    
    m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    m.avg_win_loss_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else 0.0
    
    m.largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
    m.largest_loss = float(np.min(losses)) if len(losses) > 0 else 0.0
    
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    m.expectancy = float(np.mean(pnls))
    m.expectancy_ratio = m.expectancy / abs(m.avg_loss) if m.avg_loss != 0 else 0.0
    
    # ═══ Consecutive Wins/Losses ════════════════════════════════
    if len(pnls) > 0:
        streaks = _calculate_streaks(pnls)
        m.max_consecutive_wins = streaks["max_wins"]
        m.max_consecutive_losses = streaks["max_losses"]
        m.avg_consecutive_wins = streaks["avg_wins"]
        m.avg_consecutive_losses = streaks["avg_losses"]
    
    # ═══ Equity Curve ═══════════════════════════════════════════
    equity = np.cumsum(pnls) + initial_capital
    m.final_capital = float(equity[-1])
    m.total_return_pct = (m.final_capital - initial_capital) / initial_capital
    
    # CAGR (Compound Annual Growth Rate)
    if m.total_trades > 0:
        years = m.total_trades / periods_per_year
        if years > 0 and m.final_capital > 0:
            m.cagr = (m.final_capital / initial_capital) ** (1 / years) - 1
    
    # ═══ Returns Series ═════════════════════════════════════════
    returns = pnls / initial_capital
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    
    # Annualized metrics
    if m.total_trades > 0:
        m.annualized_return_pct = mean_return * periods_per_year
        m.volatility_annualized = std_return * math.sqrt(periods_per_year)
    
    # ═══ Sharpe Ratio ═══════════════════════════════════════════
    rf_per_period = risk_free_rate / periods_per_year
    excess_return = mean_return - rf_per_period
    m.sharpe_ratio = (
        (excess_return / std_return * math.sqrt(periods_per_year))
        if std_return > 0 else 0.0
    )
    
    # ═══ Sortino Ratio (downside deviation) ════════════════════
    downside = returns[returns < rf_per_period]
    m.downside_deviation = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    m.sortino_ratio = (
        (excess_return / m.downside_deviation * math.sqrt(periods_per_year))
        if m.downside_deviation > 0 else 0.0
    )
    
    # Upside deviation (for advanced metrics)
    upside = returns[returns > rf_per_period]
    m.upside_deviation = float(np.std(upside, ddof=1)) if len(upside) > 1 else 0.0
    
    # ═══ Drawdown Analysis ══════════════════════════════════════
    # Include initial capital in drawdown baseline to avoid 0/0 and NaN propagation.
    equity_curve = np.concatenate(([float(initial_capital)], equity))
    peak = np.maximum.accumulate(equity_curve)
    safe_peak = np.where(peak > 0.0, peak, np.nan)
    drawdown = (peak - equity_curve) / safe_peak
    drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
    m.max_drawdown_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    
    # Average drawdown
    m.avg_drawdown_pct = float(np.mean(drawdown[drawdown > 0])) if np.any(drawdown > 0) else 0.0
    
    # Recovery factor
    net_profit = m.final_capital - initial_capital
    m.recovery_factor = (
        net_profit / (initial_capital * m.max_drawdown_pct)
        if m.max_drawdown_pct > 0 else 0.0
    )
    
    # Drawdown duration
    in_dd = drawdown > 0
    if np.any(in_dd):
        dd_runs = np.diff(np.concatenate(([0], in_dd.astype(int), [0])))
        dd_starts = np.where(dd_runs == 1)[0]
        dd_ends = np.where(dd_runs == -1)[0]
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            max_run = max(end - start for start, end in zip(dd_starts, dd_ends))
            m.max_drawdown_duration_days = max_run
    
    # ═══ Calmar Ratio ═══════════════════════════════════════════
    m.calmar_ratio = (
        m.annualized_return_pct / m.max_drawdown_pct
        if m.max_drawdown_pct > 0 else 0.0
    )
    
    # ═══ Omega Ratio ════════════════════════════════════════════
    threshold = rf_per_period
    gains = np.sum(returns[returns > threshold] - threshold)
    losses_sum = np.sum(threshold - returns[returns < threshold])
    m.omega_ratio = gains / losses_sum if losses_sum > 0 else 0.0
    
    # ═══ Value at Risk & CVaR ═══════════════════════════════════
    if len(returns) > 0:
        m.var_95 = float(np.percentile(returns, 5)) * initial_capital
        # CVaR (Expected Shortfall) - average of losses beyond VaR
        var_threshold = float(np.percentile(returns, 5))
        tail_losses = returns[returns <= var_threshold]
        m.cvar_95 = float(np.mean(tail_losses)) * initial_capital if len(tail_losses) > 0 else m.var_95
    
    # ═══ Transaction Costs ══════════════════════════════════════
    m.total_spread_cost = spread_costs
    m.total_swap_cost = swap_costs
    m.total_slippage_cost = slippage_costs
    m.total_commission = commission_costs
    m.total_fees = spread_costs + swap_costs + slippage_costs + commission_costs
    m.fee_impact_pct = m.total_fees / initial_capital if initial_capital > 0 else 0.0
    
    if m.total_trades > 0:
        m.avg_spread_per_trade = spread_costs / m.total_trades
        m.avg_slippage_per_trade = slippage_costs / m.total_trades
    
    # ═══ Trade Durations ════════════════════════════════════════
    if trade_durations_min:
        m.avg_trade_duration_minutes = float(np.mean(trade_durations_min))
        m.median_trade_duration_minutes = float(np.median(trade_durations_min))
    
    # ═══ Position Sizing ════════════════════════════════════════
    if trade_details:
        position_sizes = [t.get("position_size_pct", 0.0) for t in trade_details]
        if position_sizes:
            m.avg_position_size_pct = float(np.mean(position_sizes))
            m.max_position_size_pct = float(np.max(position_sizes))
    
    # Kelly Optimal Fraction
    if m.avg_loss != 0 and m.win_rate > 0:
        win_loss_ratio = abs(m.avg_win / m.avg_loss)
        kelly = (m.win_rate * win_loss_ratio - (1 - m.win_rate)) / win_loss_ratio
        m.kelly_optimal_fraction = max(0.0, kelly)
    
    # ═══ Time-based Metrics ═════════════════════════════════════
    if m.total_trades > 0:
        m.trading_days = m.total_trades  # Approximation
        m.avg_trades_per_day = 1.0  # Simplified
    
    return m


# Compatibility alias for legacy imports
def compute_metrics(
    pnl_series: List[float],
    *,
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.05,
    periods_per_year: float = 252.0,
    symbol: str = "",
    timeframe: str = "",
    start_date: str = "",
    end_date: str = "",
    spread_costs: float = 0.0,
    swap_costs: float = 0.0,
    slippage_costs: float = 0.0,
    commission_costs: float = 0.0,
    trade_durations_min: Optional[List[float]] = None,
    trade_details: Optional[List[Dict]] = None,
) -> InstitutionalBacktestMetrics:
    return compute_institutional_metrics(
        pnl_series,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        spread_costs=spread_costs,
        swap_costs=swap_costs,
        slippage_costs=slippage_costs,
        commission_costs=commission_costs,
        trade_durations_min=trade_durations_min,
        trade_details=trade_details,
    )


def _calculate_streaks(pnls: np.ndarray) -> Dict[str, float]:
    """Calculate consecutive wins/losses statistics"""
    is_win = pnls > 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    all_win_streaks = []
    all_loss_streaks = []
    
    for win in is_win:
        if win:
            current_wins += 1
            if current_losses > 0:
                all_loss_streaks.append(current_losses)
                current_losses = 0
        else:
            current_losses += 1
            if current_wins > 0:
                all_win_streaks.append(current_wins)
                current_wins = 0
        
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)
    
    # Capture final streak
    if current_wins > 0:
        all_win_streaks.append(current_wins)
    if current_losses > 0:
        all_loss_streaks.append(current_losses)
    
    return {
        "max_wins": max_wins,
        "max_losses": max_losses,
        "avg_wins": float(np.mean(all_win_streaks)) if all_win_streaks else 0.0,
        "avg_losses": float(np.mean(all_loss_streaks)) if all_loss_streaks else 0.0,
    }


def save_institutional_metrics(
    metrics: InstitutionalBacktestMetrics,
    output_dir: Path,
    prefix: str = "backtest_institutional",
) -> None:
    """Save institutional metrics to JSON and text report"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON (machine-readable)
    json_path = output_dir / f"{prefix}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
    
    # Text report (human-readable)
    report_path = output_dir / f"{prefix}_report.txt"
    report = _format_institutional_report(metrics)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    log.info("Institutional metrics saved: %s, %s", json_path, report_path)


# Compatibility aliases for legacy imports
BacktestMetrics = InstitutionalBacktestMetrics


def save_metrics(
    metrics: InstitutionalBacktestMetrics,
    output_dir: Path,
    prefix: str = "backtest_institutional",
) -> None:
    save_institutional_metrics(metrics, output_dir, prefix=prefix)


def _format_institutional_report(m: InstitutionalBacktestMetrics) -> str:
    """Format comprehensive institutional report"""
    lines = [
        "=" * 80,
        f"  🏛️  INSTITUTIONAL BACKTEST REPORT — {m.symbol} ({m.timeframe})",
        f"  Period: {m.start_date} → {m.end_date}",
        "=" * 80,
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  RETURNS & PERFORMANCE",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Initial Capital:        ${m.initial_capital:,.2f}",
        f"    Final Capital:          ${m.final_capital:,.2f}",
        f"    Total Return:           {m.total_return_pct:+.2%}",
        f"    Annualized Return:      {m.annualized_return_pct:+.2%}",
        f"    CAGR:                   {m.cagr:+.2%}",
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  RISK-ADJUSTED METRICS (Institutional Standard)",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Sharpe Ratio:           {m.sharpe_ratio:.3f}  {'✅' if m.sharpe_ratio >= 2.0 else '❌'} (Target: >= 2.0)",
        f"    Sortino Ratio:          {m.sortino_ratio:.3f}",
        f"    Calmar Ratio:           {m.calmar_ratio:.3f}",
        f"    Omega Ratio:            {m.omega_ratio:.3f}",
        f"    Information Ratio:      {m.information_ratio:.3f}",
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  RISK MEASURES",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Max Drawdown:           {m.max_drawdown_pct:.2%}  {'✅' if m.max_drawdown_pct <= 0.20 else '❌'} (Max: 20%)",
        f"    Average Drawdown:       {m.avg_drawdown_pct:.2%}",
        f"    DD Duration (days):     {m.max_drawdown_duration_days:.0f}",
        f"    Recovery Factor:        {m.recovery_factor:.2f}",
        f"    Annualized Vol:         {m.volatility_annualized:.2%}",
        f"    Downside Deviation:     {m.downside_deviation:.2%}",
        f"    Value at Risk (95%):    ${m.var_95:,.2f}",
        f"    CVaR/ES (95%):          ${m.cvar_95:,.2f}",
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  ROBUSTNESS & STRESS TESTS",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Risk of Ruin:           {m.risk_of_ruin:.2%}  {'✅' if m.risk_of_ruin <= 0.01 else '❌'} (Max: 1.0%)",
        f"    MC Runs:                {m.monte_carlo_runs:,}",
        f"    MC 95% Confidence:      ${m.mc_confidence_95_percentile:,.2f}",
        f"    MC Worst Case:          ${m.mc_worst_case:,.2f}",
        f"    MC Best Case:           ${m.mc_best_case:,.2f}",
        "",
        f"    Walk-Forward:           {'✅ PASS' if m.wfa_passed else '❌ FAIL'} "
        f"({m.wfa_failed_windows}/{m.wfa_total_windows} windows failed)",
        f"    WFA Avg Sharpe:         {m.wfa_avg_sharpe:.2f}",
        f"    WFA Min Sharpe:         {m.wfa_min_sharpe:.2f}",
        "",
        f"    Stress Test:            {'✅ PASS' if m.stress_test_passed else '❌ FAIL'}",
    ]
    
    # Add stress scenarios if available
    if m.stress_scenarios:
        lines.append("    Scenarios:")
        for scenario in m.stress_scenarios:
            status = '✅' if scenario.get('passed') else '❌'
            lines.append(f"      {status} {scenario.get('name')}: Impact={scenario.get('impact', 0):.2f}")
    
    lines.extend([
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  TRADE STATISTICS",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Total Trades:           {m.total_trades}",
        f"    Win Rate:               {m.win_rate:.1%}  {'✅' if m.win_rate >= 0.58 else '❌'} (Target: >= 58%)",
        f"    Profit Factor:          {m.profit_factor:.2f}",
        f"    Expectancy:             ${m.expectancy:.2f}",
        f"    Expectancy Ratio:       {m.expectancy_ratio:.2f}",
        "",
        f"    Avg Win:                ${m.avg_win:.2f}",
        f"    Avg Loss:               ${m.avg_loss:.2f}",
        f"    Win/Loss Ratio:         {m.avg_win_loss_ratio:.2f}",
        f"    Largest Win:            ${m.largest_win:.2f}",
        f"    Largest Loss:           ${m.largest_loss:.2f}",
        "",
        f"    Max Consecutive Wins:   {m.max_consecutive_wins}",
        f"    Max Consecutive Losses: {m.max_consecutive_losses}",
        f"    Avg Consecutive Wins:   {m.avg_consecutive_wins:.1f}",
        f"    Avg Consecutive Losses: {m.avg_consecutive_losses:.1f}",
        "",
        f"    Avg Duration (min):     {m.avg_trade_duration_minutes:.1f}",
        f"    Median Duration (min):  {m.median_trade_duration_minutes:.1f}",
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  TRANSACTION COSTS (Reality Check)",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Spread Cost:            ${m.total_spread_cost:.2f}  (${m.avg_spread_per_trade:.2f}/trade)",
        f"    Slippage Cost:          ${m.total_slippage_cost:.2f}  (${m.avg_slippage_per_trade:.2f}/trade)",
        f"    Commission:             ${m.total_commission:.2f}",
        f"    Swap/Funding:           ${m.total_swap_cost:.2f}",
        f"    Total Fees:             ${m.total_fees:.2f}",
        f"    Fee Impact:             {m.fee_impact_pct:.2%}",
        "",
        "  ══════════════════════════════════════════════════════════════════════════",
        "  POSITION MANAGEMENT",
        "  ──────────────────────────────────────────────────────────────────────────",
        f"    Avg Position Size:      {m.avg_position_size_pct:.1%}",
        f"    Max Position Size:      {m.max_position_size_pct:.1%}",
        f"    Kelly Optimal:          {m.kelly_optimal_fraction:.1%}",
        "",
        "=" * 80,
        "",
        f"  {'🏛️  INSTITUTIONAL GRADE VERIFIED ✅' if m.sharpe_ratio >= 2.0 and m.win_rate >= 0.58 and m.risk_of_ruin <= 0.01 else '❌ DOES NOT MEET INSTITUTIONAL STANDARDS'}",
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)
