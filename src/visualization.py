"""Plotly charts for the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .monte_carlo import path_percentiles
from .optimize import OptimizedPortfolio

NAVY = "#1B4F72"
GOLD = "#C4A35A"
TEAL = "#148F77"
RED = "#C0392B"
SLATE = "#5D6D7E"
BLUE = "#2E86AB"


def _layout(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=NAVY),
            x=0.0,
            xanchor="left",
            pad=dict(t=0, b=12),
        ),
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0.0,
            xanchor="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=50, r=24, t=56, b=96),
        font=dict(color="#1C2833"),
    )
    return fig


def plot_normalized_prices(prices: pd.DataFrame, title: str = "Growth of $1") -> go.Figure:
    growth = prices / prices.iloc[0]
    fig = go.Figure()
    for col in growth.columns:
        fig.add_trace(
            go.Scatter(x=growth.index, y=growth[col], name=col, mode="lines", line=dict(width=1.6))
        )
    fig.update_yaxes(title="Indexed price (start = 1.0)")
    fig.update_xaxes(title="Date", title_standoff=8)
    fig = _layout(fig, title, height=520)
    # 12 tickers wrap to 2–3 legend rows; keep them under the x-axis, not on the title.
    fig.update_layout(
        legend=dict(y=-0.28, yanchor="top"),
        margin=dict(t=58, b=120),
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_coloraxes(colorbar_title="ρ")
    return _layout(fig, "Return correlation matrix", height=520)


def plot_efficient_frontier(
    random_df: pd.DataFrame,
    frontier: pd.DataFrame,
    highlights: list[tuple[str, float, float, str]],
) -> go.Figure:
    """
    highlights: list of (label, vol, ret, color)
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=random_df["ann_vol"],
            y=random_df["ann_return"],
            mode="markers",
            name="Random portfolios",
            marker=dict(
                size=6,
                color=random_df["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                opacity=0.55,
            ),
            hovertemplate="Vol=%{x:.1%}<br>Return=%{y:.1%}<br>Sharpe=%{marker.color:.2f}<extra></extra>",
        )
    )
    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["ann_vol"],
                y=frontier["ann_return"],
                mode="lines",
                name="Efficient frontier",
                line=dict(color=NAVY, width=3),
            )
        )
    symbols = ["star", "diamond", "hexagram", "x"]
    for i, (label, vol, ret, color) in enumerate(highlights):
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers+text",
                name=label,
                marker=dict(size=14, color=color, symbol=symbols[i % len(symbols)], line=dict(width=1, color="white")),
                text=[label],
                textposition="top center",
            )
        )
    fig.update_xaxes(title="Annualized volatility", tickformat=".0%")
    fig.update_yaxes(title="Annualized return", tickformat=".0%")
    return _layout(fig, "Efficient frontier (long-only)", height=540)


def plot_weights(weights: pd.Series, title: str = "Portfolio weights") -> go.Figure:
    ordered = weights[weights > 0.001].sort_values(ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ordered.index,
            y=ordered.values,
            marker_color=NAVY,
            text=[f"{v:.1%}" for v in ordered.values],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Weight", tickformat=".0%", range=[0, max(0.4, float(ordered.max()) * 1.25)])
    return _layout(fig, title, height=380)


def plot_var_histogram(returns: pd.Series, var_95: float, cvar_95: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=60,
            name="Daily returns",
            marker_color=BLUE,
            opacity=0.75,
        )
    )
    fig.add_vline(x=-var_95, line_dash="dash", line_color=RED, annotation_text="VaR 95%", annotation_position="top left")
    fig.add_vline(x=-cvar_95, line_dash="dot", line_color=GOLD, annotation_text="CVaR 95%", annotation_position="top right")
    fig.update_xaxes(title="Daily return", tickformat=".1%")
    fig.update_yaxes(title="Frequency")
    return _layout(fig, "Historical return distribution with VaR / CVaR", height=420)


def plot_monte_carlo_fan(
    paths: pd.DataFrame,
    initial_value: float,
    n_sample_paths: int = 25,
) -> go.Figure:
    bands = path_percentiles(paths)
    years = bands.index / 252.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(years) + list(years[::-1]),
            y=list(bands["p95"]) + list(bands["p5"][::-1]),
            fill="toself",
            fillcolor="rgba(27, 79, 114, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th–95th percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(years) + list(years[::-1]),
            y=list(bands["p75"]) + list(bands["p25"][::-1]),
            fill="toself",
            fillcolor="rgba(20, 143, 119, 0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th–75th percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=bands["p50"],
            mode="lines",
            name="Median path",
            line=dict(color=NAVY, width=3),
        )
    )
    sample_idx = np.linspace(0, paths.shape[1] - 1, n_sample_paths, dtype=int)
    for i, col in enumerate(sample_idx):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=paths.iloc[:, col],
                mode="lines",
                line=dict(color=SLATE, width=0.7),
                opacity=0.35,
                showlegend=i == 0,
                name="Sample paths",
                hoverinfo="skip",
            )
        )
    fig.add_hline(y=initial_value, line_dash="dash", line_color=GOLD, annotation_text="Start")
    fig.update_xaxes(title="Years ahead")
    fig.update_yaxes(title="Portfolio value ($)", tickformat="$,.0f")
    return _layout(fig, "Monte Carlo fan chart of future portfolio value", height=520)


def plot_benchmark_bars(table: pd.DataFrame, metric: str = "sharpe") -> go.Figure:
    ordered = table[metric].sort_values(ascending=False)
    colors = [TEAL if i == 0 else NAVY for i in range(len(ordered))]
    fig = go.Figure(
        go.Bar(
            x=ordered.index,
            y=ordered.values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in ordered.values],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Sharpe ratio")
    return _layout(fig, "Risk-adjusted return vs. benchmarks", height=400)


def metrics_to_display_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Format risk tables for Streamlit."""
    pct_cols = ["ann_return", "ann_vol", "var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown"]
    out = table.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    if "sharpe" in out.columns:
        out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    for col in ("sharpe_vs_spy", "sharpe_vs_60_40"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    rename = {
        "sector": "Sector",
        "ann_return": "Ann. Return",
        "ann_vol": "Ann. Vol",
        "sharpe": "Sharpe",
        "var_95": "VaR 95% (1d)",
        "var_99": "VaR 99% (1d)",
        "cvar_95": "CVaR 95% (1d)",
        "cvar_99": "CVaR 99% (1d)",
        "max_drawdown": "Max Drawdown",
        "sharpe_vs_spy": "Sharpe vs SPY",
        "sharpe_vs_60_40": "Sharpe vs 60/40",
        "obs": "N",
    }
    return out.rename(columns=rename)
