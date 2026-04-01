"""Load and preprocess historical price data."""

import pandas as pd


def load_prices(
    filepath: str = "prices.csv",
    assets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load price data from CSV with DatetimeIndex.

    Forward-fills missing values (no backfill to avoid look-ahead bias).
    Optionally filters to specific assets and date range.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)

    if assets is not None:
        missing = [a for a in assets if a not in df.columns]
        if missing:
            raise ValueError(f"Assets not found in data: {missing}")
        df = df[assets]

    # Forward-fill only — no backfill to prevent look-ahead bias
    df = df.ffill()

    if start_date is not None:
        df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]

    return df


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from price levels."""
    return prices.pct_change().dropna(how="all")


def get_available_assets(filepath: str = "prices.csv") -> list[str]:
    """Return all asset column names in the CSV."""
    df = pd.read_csv(filepath, nrows=0)
    return [col for col in df.columns if col != "Date"]
