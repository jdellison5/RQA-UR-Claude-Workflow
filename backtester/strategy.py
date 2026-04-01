"""Base strategy class and reference Buy & Hold implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backtester.portfolio import Portfolio


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement generate_signals(). The backtester calls it once
    per trading day with historical data up to (but NOT including) the current
    date, ensuring no look-ahead bias.
    """

    def __init__(self, name: str, assets: list[str]):
        self.name = name
        self.assets = assets

    @abstractmethod
    def generate_signals(
        self,
        date: pd.Timestamp,
        historical_prices: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[dict]:
        """Return a list of signal dicts for the current date.

        Each signal: {"asset": str, "action": "BUY"|"SELL", "weight": float}
        - weight is the target fraction of total equity to allocate.
        - Return an empty list for "do nothing."
        """
        pass

    def on_backtest_start(self, prices: pd.DataFrame) -> None:
        """Optional hook called once before the backtest loop begins.

        Useful for precomputing indicators on the full price history.
        """
        pass


class BuyAndHoldStrategy(Strategy):
    """Buy target assets equally on the first day and hold forever."""

    def __init__(
        self,
        name: str = "Buy & Hold",
        assets: list[str] | None = None,
    ):
        if assets is None:
            assets = ["S&P 500"]
        super().__init__(name=name, assets=assets)
        self._bought = False

    def generate_signals(
        self,
        date: pd.Timestamp,
        historical_prices: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[dict]:
        if self._bought:
            return []

        # Buy equal weight across all target assets on the first trading day
        self._bought = True
        weight_per_asset = 1.0 / len(self.assets)
        return [
            {"asset": asset, "action": "BUY", "weight": weight_per_asset}
            for asset in self.assets
        ]
