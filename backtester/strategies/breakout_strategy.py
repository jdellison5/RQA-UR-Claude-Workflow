"""252-day high breakout strategy with trailing stop."""

from __future__ import annotations

import pandas as pd

from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


class BreakoutStrategy(Strategy):
    """Buy on a new 252-day high, exit on a trailing stop.

    - Buy when price makes a new 252-day (1-year) high.
    - Set a trailing stop at (1 - stop_pct) * highest price since entry.
    - Sell when price drops below the trailing stop.
    - Only re-enter when price makes another new 252-day high.
    """

    def __init__(
        self,
        name: str = "252-Day Breakout (5% Trail)",
        assets: list[str] | None = None,
        lookback: int = 252,
        stop_pct: float = 0.05,
    ):
        if assets is None:
            assets = ["S&P 500"]
        super().__init__(name=name, assets=assets)
        self.lookback = lookback
        self.stop_pct = stop_pct

        # Track highest price since entry for trailing stop
        self._high_since_entry: dict[str, float] = {}
        self._in_position: dict[str, bool] = {}

    def on_backtest_start(self, prices: pd.DataFrame) -> None:
        for asset in self.assets:
            self._in_position[asset] = False
            self._high_since_entry[asset] = 0.0

    def generate_signals(
        self,
        date: pd.Timestamp,
        historical_prices: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[dict]:
        signals = []
        weight_per_asset = 1.0 / len(self.assets)

        for asset in self.assets:
            if asset not in historical_prices.columns:
                continue

            hist = historical_prices[asset].dropna()
            if len(hist) < self.lookback:
                continue

            yesterday_price = hist.iloc[-1]
            rolling_high = hist.iloc[-self.lookback:].max()

            if self._in_position.get(asset, False):
                # Update trailing stop high-water mark
                if yesterday_price > self._high_since_entry[asset]:
                    self._high_since_entry[asset] = yesterday_price

                # Check trailing stop
                stop_level = self._high_since_entry[asset] * (1 - self.stop_pct)
                if yesterday_price < stop_level:
                    self._in_position[asset] = False
                    self._high_since_entry[asset] = 0.0
                    signals.append({
                        "asset": asset,
                        "action": "SELL",
                        "weight": 0.0,
                    })
            else:
                # Buy on new 252-day high
                if yesterday_price >= rolling_high:
                    self._in_position[asset] = True
                    self._high_since_entry[asset] = yesterday_price
                    signals.append({
                        "asset": asset,
                        "action": "BUY",
                        "weight": weight_per_asset,
                    })

        return signals
