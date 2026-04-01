"""MACD crossover strategy for mean reversion on S&P 500."""

from __future__ import annotations

import pandas as pd

from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


class MACDStrategy(Strategy):
    """Traditional MACD crossover strategy.

    - Buy when MACD line crosses above the signal line (bullish crossover).
    - Sell when MACD line crosses below the signal line (bearish crossover).

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD line, signal_period)
    """

    def __init__(
        self,
        name: str = "MACD Crossover",
        assets: list[str] | None = None,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        if assets is None:
            assets = ["S&P 500"]
        super().__init__(name=name, assets=assets)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # Precomputed indicators (populated in on_backtest_start)
        self._macd: dict[str, pd.Series] = {}
        self._signal: dict[str, pd.Series] = {}

    def on_backtest_start(self, prices: pd.DataFrame) -> None:
        """Precompute MACD and signal line for each asset."""
        for asset in self.assets:
            if asset not in prices.columns:
                continue
            series = prices[asset].dropna()
            ema_fast = series.ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = series.ewm(span=self.slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
            self._macd[asset] = macd_line
            self._signal[asset] = signal_line

    def generate_signals(
        self,
        date: pd.Timestamp,
        historical_prices: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[dict]:
        signals = []
        weight_per_asset = 1.0 / len(self.assets)

        for asset in self.assets:
            if asset not in self._macd:
                continue

            macd = self._macd[asset]
            signal = self._signal[asset]

            # Only use data available up to yesterday (no look-ahead)
            hist_macd = macd.loc[:date].iloc[:-1] if date in macd.index else macd.loc[:date]
            hist_signal = signal.loc[:date].iloc[:-1] if date in signal.index else signal.loc[:date]

            if len(hist_macd) < 2 or len(hist_signal) < 2:
                continue

            # Check for crossover using the last two available values
            macd_today = hist_macd.iloc[-1]
            macd_yesterday = hist_macd.iloc[-2]
            signal_today = hist_signal.iloc[-1]
            signal_yesterday = hist_signal.iloc[-2]

            has_position = asset in portfolio.positions and portfolio.positions[asset] > 0

            # Bullish crossover: MACD crosses above signal
            if macd_yesterday <= signal_yesterday and macd_today > signal_today:
                if not has_position:
                    signals.append({
                        "asset": asset,
                        "action": "BUY",
                        "weight": weight_per_asset,
                    })

            # Bearish crossover: MACD crosses below signal
            elif macd_yesterday >= signal_yesterday and macd_today < signal_today:
                if has_position:
                    signals.append({
                        "asset": asset,
                        "action": "SELL",
                        "weight": 0.0,
                    })

        return signals
