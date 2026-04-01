"""Performance metrics for backtested equity curves and trades."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity: pd.Series) -> float:
    """Total return as a fraction (e.g., 1.5 = 150% gain)."""
    return equity.iloc[-1] / equity.iloc[0] - 1


def annual_return(equity: pd.Series) -> float:
    """Compound annual growth rate (CAGR)."""
    n_days = len(equity)
    if n_days < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = n_days / TRADING_DAYS_PER_YEAR
    return total ** (1 / years) - 1


def annual_volatility(equity: pd.Series) -> float:
    """Annualized volatility of daily returns."""
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    vol = annual_volatility(equity)
    if vol == 0:
        return 0.0
    return (annual_return(equity) - risk_free_rate) / vol


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std == 0:
        return 0.0
    return (annual_return(equity) - risk_free_rate) / downside_std


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    dd = drawdown_series(equity)
    if dd.empty:
        return 0.0
    return dd.min()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Daily drawdown from peak as a fraction (0 = at peak, -0.2 = 20% below)."""
    peak = equity.cummax()
    return (equity - peak) / peak


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar ratio: annual return / abs(max drawdown)."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return annual_return(equity) / abs(mdd)


def win_rate(trades_df: pd.DataFrame) -> float | None:
    """Fraction of sell trades with positive P&L."""
    sells = trades_df[trades_df["action"] == "SELL"]
    if sells.empty:
        return None
    pnl = sells["pnl"].dropna()
    if pnl.empty:
        return None
    return (pnl > 0).sum() / len(pnl)


def profit_factor(trades_df: pd.DataFrame) -> float | None:
    """Gross profit / gross loss."""
    sells = trades_df[trades_df["action"] == "SELL"]
    if sells.empty:
        return None
    pnl = sells["pnl"].dropna()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else None
    return gross_profit / gross_loss


def compute_all_metrics(
    equity: pd.Series, trades_df: pd.DataFrame | None = None
) -> dict:
    """Compute all performance metrics and return as a dict."""
    if equity.empty:
        return {}

    metrics = {
        "Total Return": total_return(equity),
        "Annual Return (CAGR)": annual_return(equity),
        "Annual Volatility": annual_volatility(equity),
        "Sharpe Ratio": sharpe_ratio(equity),
        "Sortino Ratio": sortino_ratio(equity),
        "Max Drawdown": max_drawdown(equity),
        "Calmar Ratio": calmar_ratio(equity),
    }

    if trades_df is not None and not trades_df.empty:
        wr = win_rate(trades_df)
        pf = profit_factor(trades_df)
        if wr is not None:
            metrics["Win Rate"] = wr
        if pf is not None:
            metrics["Profit Factor"] = pf

    return metrics
