"""Moving average crossover strategy (20/50 SMA)."""

from __future__ import annotations

import pandas as pd

from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


class MACrossoverStrategy(Strategy):
    """Simple moving average crossover strategy.

    - Buy when the fast MA crosses above the slow MA (golden cross).
    - Sell when the fast MA crosses below the slow MA (death cross).
    """

    def __init__(
        self,
        name: str = "MA Crossover (20/50)",
        assets: list[str] | None = None,
        fast_period: int = 20,
        slow_period: int = 50,
    ):
        if assets is None:
            assets = ["S&P 500"]
        super().__init__(name=name, assets=assets)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Precomputed indicators
        self._fast_ma: dict[str, pd.Series] = {}
        self._slow_ma: dict[str, pd.Series] = {}

    def on_backtest_start(self, prices: pd.DataFrame) -> None:
        """Precompute moving averages for each asset."""
        for asset in self.assets:
            if asset not in prices.columns:
                continue
            series = prices[asset].dropna()
            self._fast_ma[asset] = series.rolling(window=self.fast_period).mean()
            self._slow_ma[asset] = series.rolling(window=self.slow_period).mean()

    def generate_signals(
        self,
        date: pd.Timestamp,
        historical_prices: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[dict]:
        signals = []
        weight_per_asset = 1.0 / len(self.assets)

        for asset in self.assets:
            if asset not in self._fast_ma:
                continue

            fast = self._fast_ma[asset]
            slow = self._slow_ma[asset]

            # Only use data available up to yesterday (no look-ahead)
            hist_fast = fast.loc[:date].iloc[:-1] if date in fast.index else fast.loc[:date]
            hist_slow = slow.loc[:date].iloc[:-1] if date in slow.index else slow.loc[:date]

            if len(hist_fast) < 2 or len(hist_slow) < 2:
                continue

            # Skip if MAs aren't fully formed yet
            if pd.isna(hist_fast.iloc[-1]) or pd.isna(hist_slow.iloc[-1]):
                continue
            if pd.isna(hist_fast.iloc[-2]) or pd.isna(hist_slow.iloc[-2]):
                continue

            fast_today = hist_fast.iloc[-1]
            fast_yesterday = hist_fast.iloc[-2]
            slow_today = hist_slow.iloc[-1]
            slow_yesterday = hist_slow.iloc[-2]

            has_position = asset in portfolio.positions and portfolio.positions[asset] > 0

            # Golden cross: fast MA crosses above slow MA
            if fast_yesterday <= slow_yesterday and fast_today > slow_today:
                if not has_position:
                    signals.append({
                        "asset": asset,
                        "action": "BUY",
                        "weight": weight_per_asset,
                    })

            # Death cross: fast MA crosses below slow MA
            elif fast_yesterday >= slow_yesterday and fast_today < slow_today:
                if has_position:
                    signals.append({
                        "asset": asset,
                        "action": "SELL",
                        "weight": 0.0,
                    })

        return signals
