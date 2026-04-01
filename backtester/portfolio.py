"""Portfolio tracking: cash, positions, trades, and equity snapshots."""

from __future__ import annotations

import pandas as pd


class Portfolio:
    """Tracks cash, positions, trade history, and daily equity snapshots."""

    def __init__(self, cash: float = 100_000):
        self.cash: float = cash
        self.positions: dict[str, float] = {}  # asset -> units held
        self._cost_basis: dict[str, float] = {}  # asset -> avg cost per unit
        self.trades: list[dict] = []
        self.history: list[dict] = []

    def buy(self, asset: str, units: float, price: float, date: pd.Timestamp) -> None:
        """Buy units of an asset at the given price."""
        cost = units * price
        if cost > self.cash + 1e-6:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
            )
        # Update average cost basis
        old_units = self.positions.get(asset, 0)
        old_cost = self._cost_basis.get(asset, 0) * old_units
        new_units = old_units + units
        if new_units > 0:
            self._cost_basis[asset] = (old_cost + cost) / new_units

        self.cash -= cost
        self.positions[asset] = new_units
        self.trades.append(
            {
                "date": date,
                "asset": asset,
                "action": "BUY",
                "units": units,
                "price": price,
                "value": cost,
                "pnl": None,
            }
        )

    def sell(self, asset: str, units: float, price: float, date: pd.Timestamp) -> None:
        """Sell units of an asset at the given price."""
        current = self.positions.get(asset, 0)
        if units > current + 1e-6:
            raise ValueError(
                f"Cannot sell {units:.4f} units of {asset}, only hold {current:.4f}"
            )
        proceeds = units * price
        avg_cost = self._cost_basis.get(asset, 0)
        pnl = (price - avg_cost) * units

        self.cash += proceeds
        self.positions[asset] = current - units
        if self.positions[asset] < 1e-10:
            self.positions.pop(asset)
            self._cost_basis.pop(asset, None)

        self.trades.append(
            {
                "date": date,
                "asset": asset,
                "action": "SELL",
                "units": units,
                "price": price,
                "value": proceeds,
                "pnl": pnl,
            }
        )

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value: cash + market value of all positions."""
        market_value = sum(
            units * current_prices.get(asset, 0)
            for asset, units in self.positions.items()
            if pd.notna(current_prices.get(asset, 0))
        )
        return self.cash + market_value

    def snapshot(self, date: pd.Timestamp, current_prices: dict[str, float]) -> None:
        """Record a daily snapshot of portfolio state."""
        equity = self.get_equity(current_prices)

        # Compute position weights
        weights = {}
        for asset, units in self.positions.items():
            price = current_prices.get(asset, 0)
            if pd.notna(price) and equity > 0:
                weights[asset] = (units * price) / equity

        self.history.append(
            {
                "date": date,
                "cash": self.cash,
                "equity": equity,
                "positions": dict(self.positions),
                "weights": weights,
            }
        )

    def get_equity_curve(self) -> pd.Series:
        """Return equity over time as a Series with DatetimeIndex."""
        if not self.history:
            return pd.Series(dtype=float)
        dates = [h["date"] for h in self.history]
        values = [h["equity"] for h in self.history]
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def get_trades_df(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=["date", "asset", "action", "units", "price", "value", "pnl"]
            )
        return pd.DataFrame(self.trades)

    def get_positions_df(self) -> pd.DataFrame:
        """Return daily position weights over time as a DataFrame."""
        if not self.history:
            return pd.DataFrame()
        records = []
        for h in self.history:
            row = {"date": h["date"]}
            row.update(h["weights"])
            records.append(row)
        df = pd.DataFrame(records).set_index("date")
        df = df.fillna(0)
        return df
