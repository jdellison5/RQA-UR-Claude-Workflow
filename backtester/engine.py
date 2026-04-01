"""Backtester engine: runs strategies against historical price data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.metrics import compute_all_metrics
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    strategy_name: str
    equity_curve: pd.Series
    trades: pd.DataFrame
    positions_history: pd.DataFrame
    metrics: dict


class Backtester:
    """Run a strategy against historical price data and track portfolio performance."""

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100_000,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.prices = prices
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.start_date = (
            pd.Timestamp(start_date) if start_date else prices.index[0]
        )
        self.end_date = pd.Timestamp(end_date) if end_date else prices.index[-1]

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        portfolio = Portfolio(cash=self.initial_capital)
        trading_dates = self.prices.loc[self.start_date : self.end_date].index

        # Let strategy precompute indicators
        self.strategy.on_backtest_start(self.prices)

        for date in trading_dates:
            # Historical prices: everything strictly before today (no look-ahead)
            historical = self.prices.loc[:date].iloc[:-1]

            # Today's prices for trade execution
            today_prices = self.prices.loc[date].to_dict()

            # Get signals from strategy
            signals = self.strategy.generate_signals(date, historical, portfolio)

            # Execute signals at today's prices
            self._execute_signals(signals, today_prices, portfolio, date)

            # Record daily snapshot
            portfolio.snapshot(date, today_prices)

        # Compute performance metrics
        equity_curve = portfolio.get_equity_curve()
        trades_df = portfolio.get_trades_df()
        metrics = compute_all_metrics(equity_curve, trades_df)

        return BacktestResult(
            strategy_name=self.strategy.name,
            equity_curve=equity_curve,
            trades=trades_df,
            positions_history=portfolio.get_positions_df(),
            metrics=metrics,
        )

    def _execute_signals(
        self,
        signals: list[dict],
        today_prices: dict[str, float],
        portfolio: Portfolio,
        date: pd.Timestamp,
    ) -> None:
        """Convert weight-based signals into buy/sell orders.

        Processes sells first to free up cash, then buys.
        """
        if not signals:
            return

        total_equity = portfolio.get_equity(today_prices)

        # Separate and process sells before buys
        sells = [s for s in signals if s["action"] == "SELL"]
        buys = [s for s in signals if s["action"] == "BUY"]

        for signal in sells:
            asset = signal["asset"]
            price = today_prices.get(asset)
            if price is None or pd.isna(price):
                continue
            current_units = portfolio.positions.get(asset, 0)
            if current_units <= 0:
                continue
            target_value = signal["weight"] * total_equity
            current_value = current_units * price
            sell_value = current_value - target_value
            if sell_value > 0:
                sell_units = sell_value / price
                sell_units = min(sell_units, current_units)
                portfolio.sell(asset, sell_units, price, date)

        for signal in buys:
            asset = signal["asset"]
            price = today_prices.get(asset)
            if price is None or pd.isna(price):
                continue
            target_value = signal["weight"] * total_equity
            current_value = portfolio.positions.get(asset, 0) * price
            buy_value = target_value - current_value
            if buy_value > 0 and portfolio.cash >= buy_value:
                buy_units = buy_value / price
                portfolio.buy(asset, buy_units, price, date)
