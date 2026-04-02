"""Generate an interactive HTML dashboard from backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.engine import BacktestResult
from backtester.metrics import drawdown_series

# Dark professional color palette
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(family="Segoe UI, Roboto, sans-serif", color="#e0e0e0"),
    hovermode="x unified",
    margin=dict(l=60, r=30, t=50, b=40),
)


def generate_dashboard(
    results: list[BacktestResult],
    output_path: str = "dashboard.html",
) -> None:
    """Build and save a standalone interactive HTML dashboard."""
    include_js = True  # Bundle Plotly JS inline so it works offline

    sections = []
    sections.append(_html_header())
    sections.append(_performance_table(results))
    sections.append(_section_divider("Equity Curves"))
    sections.append(_equity_chart(results, include_js))
    include_js = False  # Subsequent charts reuse the CDN script

    sections.append(_section_divider("Drawdowns"))
    sections.append(_drawdown_chart(results, include_js))

    sections.append(_section_divider("Rolling Sharpe Ratio (252-day)"))
    sections.append(_rolling_sharpe_chart(results, include_js))

    for r in results:
        sections.append(_section_divider(f"Positions — {r.strategy_name}"))
        sections.append(_positions_chart(r, include_js))

    sections.append(_section_divider("Return Correlations"))
    sections.append(_correlation_matrix(results, include_js))

    sections.append(_section_divider("Trade Log"))
    sections.append(_trade_log_table(results))

    sections.append(_html_footer())

    with open(output_path, "w") as f:
        f.write("\n".join(sections))


# ---------------------------------------------------------------------------
# HTML scaffolding
# ---------------------------------------------------------------------------

def _html_header() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f0f23;
    color: #e0e0e0;
    font-family: 'Segoe UI', Roboto, sans-serif;
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  h1 { text-align: center; margin-bottom: 32px; color: #636EFA; font-size: 28px; }
  h2 {
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a4a;
    color: #aab;
    font-size: 18px;
    font-weight: 500;
  }
  .metrics-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }
  .metrics-table th, .metrics-table td {
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid #2a2a4a;
  }
  .metrics-table th { color: #8888aa; font-weight: 500; text-align: right; }
  .metrics-table th:first-child, .metrics-table td:first-child { text-align: left; }
  .metrics-table tr:hover { background: #1a1a3e; }
  .positive { color: #00CC96; }
  .negative { color: #EF553B; }
  .trade-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 13px;
  }
  .trade-table th, .trade-table td {
    padding: 6px 12px;
    text-align: right;
    border-bottom: 1px solid #1a1a3a;
  }
  .trade-table th { color: #8888aa; font-weight: 500; }
  .trade-table th:first-child, .trade-table td:first-child { text-align: left; }
  .trade-table td:nth-child(2) { text-align: left; }
  .chart-container { margin: 16px 0; }
</style>
</head>
<body>
<h1>Strategy Backtest Dashboard</h1>
"""


def _html_footer() -> str:
    return """
</body>
</html>"""


def _section_divider(title: str) -> str:
    return f"<h2>{title}</h2>"


# ---------------------------------------------------------------------------
# Performance summary table
# ---------------------------------------------------------------------------

def _performance_table(results: list[BacktestResult]) -> str:
    metric_keys = [
        "Total Return", "Annual Return (CAGR)", "Annual Volatility",
        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
        "Win Rate", "Profit Factor",
    ]

    header = "<th>Strategy</th>" + "".join(f"<th>{k}</th>" for k in metric_keys)
    rows = []
    for r in results:
        cells = [f"<td>{r.strategy_name}</td>"]
        for k in metric_keys:
            val = r.metrics.get(k)
            if val is None:
                cells.append("<td>—</td>")
            elif k in ("Total Return", "Annual Return (CAGR)", "Annual Volatility", "Max Drawdown", "Win Rate"):
                css = "positive" if val > 0 and k != "Max Drawdown" else "negative" if val < 0 else ""
                if k == "Win Rate":
                    css = "positive" if val >= 0.5 else "negative"
                cells.append(f'<td class="{css}">{val:.2%}</td>')
            else:
                css = "positive" if val > 0 else "negative" if val < 0 else ""
                cells.append(f'<td class="{css}">{val:.2f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<table class="metrics-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _fig_to_html(fig: go.Figure, include_plotlyjs) -> str:
    return (
        '<div class="chart-container">'
        + fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
        + "</div>"
    )


def _equity_chart(results: list[BacktestResult], include_plotlyjs) -> str:
    fig = go.Figure()
    for i, r in enumerate(results):
        # Normalize to 100
        normalized = r.equity_curve / r.equity_curve.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=r.strategy_name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Equity Curves (Normalized to 100)",
        yaxis_title="Portfolio Value",
        yaxis_type="log",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _fig_to_html(fig, include_plotlyjs)


def _drawdown_chart(results: list[BacktestResult], include_plotlyjs) -> str:
    fig = go.Figure()
    for i, r in enumerate(results):
        dd = drawdown_series(r.equity_curve)
        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            name=r.strategy_name,
            fill="tozeroy",
            line=dict(color=COLORS[i % len(COLORS)], width=1),
            fillcolor=COLORS[i % len(COLORS)].replace(")", ", 0.3)").replace("rgb", "rgba")
            if "rgb" in COLORS[i % len(COLORS)] else None,
        ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Drawdown from Peak",
        yaxis_title="Drawdown (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _fig_to_html(fig, include_plotlyjs)


def _rolling_sharpe_chart(results: list[BacktestResult], include_plotlyjs) -> str:
    fig = go.Figure()
    for i, r in enumerate(results):
        daily_returns = r.equity_curve.pct_change().dropna()
        if len(daily_returns) < 252:
            continue
        rolling_mean = daily_returns.rolling(252).mean()
        rolling_std = daily_returns.rolling(252).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name=r.strategy_name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Rolling 252-Day Sharpe Ratio",
        yaxis_title="Sharpe Ratio",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _fig_to_html(fig, include_plotlyjs)


def _positions_chart(result: BacktestResult, include_plotlyjs) -> str:
    pos = result.positions_history
    if pos.empty:
        return "<p>No position data available.</p>"

    fig = go.Figure()
    for i, col in enumerate(pos.columns):
        fig.add_trace(go.Scatter(
            x=pos.index,
            y=pos[col].values * 100,
            name=col,
            stackgroup="one",
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
        ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Portfolio Allocation — {result.strategy_name}",
        yaxis_title="Allocation (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _fig_to_html(fig, include_plotlyjs)


def _correlation_matrix(results: list[BacktestResult], include_plotlyjs) -> str:
    # If multiple strategies, correlate their returns; otherwise correlate asset returns
    if len(results) > 1:
        returns_dict = {}
        for r in results:
            returns_dict[r.strategy_name] = r.equity_curve.pct_change().dropna()
        returns_df = pd.DataFrame(returns_dict).dropna()
        title = "Strategy Return Correlations"
    else:
        # Correlate the positions/assets within the single strategy
        pos = results[0].positions_history
        if pos.empty or len(pos.columns) < 2:
            return "<p>Not enough assets for correlation matrix.</p>"
        returns_df = pos.pct_change().dropna()
        returns_df = returns_df.loc[:, (returns_df != 0).any()]
        title = "Asset Allocation Correlations"

    if returns_df.shape[1] < 2:
        return "<p>Not enough series for correlation matrix.</p>"

    corr = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        height=max(400, 50 * len(corr)),
        width=max(500, 60 * len(corr)),
    )
    return _fig_to_html(fig, include_plotlyjs)


# ---------------------------------------------------------------------------
# Trade log table
# ---------------------------------------------------------------------------

def _trade_log_table(results: list[BacktestResult]) -> str:
    all_trades = []
    for r in results:
        if r.trades.empty:
            continue
        df = r.trades.copy()
        df.insert(0, "strategy", r.strategy_name)
        all_trades.append(df)

    if not all_trades:
        return "<p>No trades executed.</p>"

    trades = pd.concat(all_trades, ignore_index=True)

    header_cols = ["Strategy", "Date", "Asset", "Action", "Units", "Price", "Value", "P&L"]
    header = "".join(f"<th>{c}</th>" for c in header_cols)

    rows = []
    for _, t in trades.iterrows():
        pnl = t.get("pnl")
        pnl_str = "—"
        pnl_class = ""
        if pnl is not None and not pd.isna(pnl):
            pnl_class = "positive" if pnl >= 0 else "negative"
            pnl_str = f"${pnl:,.2f}"

        date_str = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])
        rows.append(
            f"<tr>"
            f"<td>{t['strategy']}</td>"
            f"<td>{date_str}</td>"
            f"<td>{t['asset']}</td>"
            f"<td>{t['action']}</td>"
            f"<td>{t['units']:.4f}</td>"
            f"<td>${t['price']:,.2f}</td>"
            f"<td>${t['value']:,.2f}</td>"
            f'<td class="{pnl_class}">{pnl_str}</td>'
            f"</tr>"
        )

    return (
        '<table class="trade-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
