"""Run backtest strategies and generate an interactive HTML dashboard."""

from backtester.data_loader import load_prices
from backtester.engine import Backtester
from backtester.strategy import BuyAndHoldStrategy
from backtester.strategies.macd_strategy import MACDStrategy
from backtester.strategies.ma_crossover_strategy import MACrossoverStrategy
from backtester.strategies.breakout_strategy import BreakoutStrategy
from backtester.dashboard import generate_dashboard


def main():
    # Load price data — use a start date where S&P 500 data is available
    prices = load_prices("prices.csv", start_date="1990-01-01")

    # Define strategies to backtest
    strategies = [
        BuyAndHoldStrategy(name="Buy & Hold S&P 500", assets=["S&P 500"]),
        MACDStrategy(name="MACD Crossover (S&P 500)", assets=["S&P 500"]),
        MACrossoverStrategy(name="MA Crossover 20/50 (S&P 500)", assets=["S&P 500"]),
        BreakoutStrategy(name="252-Day Breakout 5% Trail (S&P 500)", assets=["S&P 500"]),
    ]

    # Run each strategy
    results = []
    for strategy in strategies:
        bt = Backtester(prices, strategy, initial_capital=100_000)
        result = bt.run()
        results.append(result)

        # Print summary
        print(f"\n{'='*60}")
        print(f"  {result.strategy_name}")
        print(f"{'='*60}")
        for key, val in result.metrics.items():
            if isinstance(val, float):
                if "Return" in key or "Volatility" in key or "Drawdown" in key or "Rate" in key:
                    print(f"  {key:.<30} {val:>10.2%}")
                else:
                    print(f"  {key:.<30} {val:>10.2f}")
            else:
                print(f"  {key:.<30} {val!s:>10}")

    # Generate dashboard
    output_path = "dashboard.html"
    generate_dashboard(results, output_path)
    print(f"\nDashboard saved to {output_path}")


if __name__ == "__main__":
    main()
