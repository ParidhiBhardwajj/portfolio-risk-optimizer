"""
Portfolio Risk & Optimization Dashboard

Interactive Streamlit app: pull prices, compute risk, optimize a long-only
book, simulate future paths, and benchmark against SPY and a 60/40 mix.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.benchmark import build_benchmark_comparison, equal_weight_series
from src.data import BENCHMARKS, DEFAULT_UNIVERSE, load_price_history
from src.monte_carlo import path_summary, simulate_portfolio_paths
from src.optimize import (
    OptimizedPortfolio,
    efficient_frontier,
    interpolate_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    random_portfolios,
)
from src.risk import (
    align_weights,
    asset_risk_table,
    compute_returns,
    portfolio_metrics,
    portfolio_returns,
)
from src.visualization import (
    metrics_to_display_frame,
    plot_benchmark_bars,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_monte_carlo_fan,
    plot_normalized_prices,
    plot_var_histogram,
    plot_weights,
)

st.set_page_config(
    page_title="Portfolio Risk & Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.35rem;
        font-weight: 700;
        color: #1B4F72;
        text-align: center;
        padding: 0.4rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #5D6D7E;
        margin-bottom: 1.2rem;
    }
    .section-title {
        color: #1B4F72;
        font-size: 1.25rem;
        font-weight: 650;
        margin: 0.4rem 0 0.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Downloading daily prices from Yahoo Finance...")
def cached_prices(tickers: tuple[str, ...], years: int, refresh: bool) -> pd.DataFrame:
    return load_price_history(tickers, years=years, refresh=refresh)


@st.cache_data(show_spinner="Solving mean-variance portfolios...")
def cached_optimizer(
    mean: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
    n_random: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    min_var = min_variance_portfolio(mean, cov, risk_free_rate, max_weight)
    max_sharpe = max_sharpe_portfolio(mean, cov, risk_free_rate, max_weight)
    frontier = efficient_frontier(mean, cov, risk_free_rate, n_points=36, max_weight=max_weight)
    random_df = random_portfolios(
        mean, cov, n_portfolios=n_random, risk_free_rate=risk_free_rate, max_weight=max_weight
    )
    return (
        frontier,
        random_df,
        {
            "weights": min_var.weights.to_dict(),
            "ann_return": min_var.ann_return,
            "ann_vol": min_var.ann_vol,
            "sharpe": min_var.sharpe,
            "label": min_var.label,
        },
        {
            "weights": max_sharpe.weights.to_dict(),
            "ann_return": max_sharpe.ann_return,
            "ann_vol": max_sharpe.ann_vol,
            "sharpe": max_sharpe.sharpe,
            "label": max_sharpe.label,
        },
    )


def pack_portfolio(payload: dict, index: pd.Index) -> OptimizedPortfolio:
    return OptimizedPortfolio(
        weights=pd.Series(payload["weights"]).reindex(index).fillna(0.0),
        ann_return=payload["ann_return"],
        ann_vol=payload["ann_vol"],
        sharpe=payload["sharpe"],
        label=payload["label"],
    )


st.markdown('<div class="main-header">Portfolio Risk & Optimization Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Mean-variance optimization, tail risk, and Monte Carlo wealth paths vs. SPY and 60/40</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Universe & settings")
    years = st.slider("Lookback (years)", min_value=3, max_value=5, value=5, step=1)
    refresh = st.button("Refresh prices from Yahoo Finance")
    risk_free = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0,
        max_value=0.12,
        value=0.043,
        step=0.001,
        format="%.3f",
        help="Used for Sharpe. Default is ~the 3-month T-bill level.",
    )
    max_weight = st.slider(
        "Max single-name weight",
        min_value=0.15,
        max_value=1.00,
        value=0.30,
        step=0.05,
        help="Mandate cap so max-Sharpe does not collapse into one winner.",
    )

    sector_options = {}
    for ticker, sector in DEFAULT_UNIVERSE.items():
        sector_options.setdefault(sector, []).append(ticker)

    selected = []
    st.subheader("Holdings")
    for sector, names in sector_options.items():
        chosen = st.multiselect(sector, names, default=names)
        selected.extend(chosen)

    if len(selected) < 2:
        st.warning("Select at least two holdings.")
        st.stop()

    st.subheader("Portfolio construction")
    mode = st.radio(
        "Weighting method",
        (
            "Max Sharpe",
            "Min Variance",
            "Risk tolerance (frontier)",
            "Equal weight",
            "Custom mix",
        ),
        index=0,
    )
    risk_tolerance = 55
    if mode == "Risk tolerance (frontier)":
        risk_tolerance = st.slider(
            "Risk tolerance",
            min_value=0,
            max_value=100,
            value=55,
            help="0 = lowest-vol book on the frontier; 100 = highest-return book.",
        )

    custom_weights = {}
    if mode == "Custom mix":
        st.caption("Weights are renormalized to 100% on every rerun.")
        for ticker in selected:
            custom_weights[ticker] = st.slider(f"{ticker}", 0.0, 100.0, value=round(100.0 / len(selected), 1), step=0.5)

    st.subheader("Monte Carlo")
    mc_years = st.slider("Horizon (years)", min_value=1, max_value=5, value=5)
    n_sims = st.select_slider("Simulated paths", options=[1000, 2000, 4000, 8000], value=4000)
    start_wealth = st.number_input("Starting wealth ($)", min_value=1000, value=100000, step=1000)

tickers = selected + [t for t in BENCHMARKS if t not in selected]
try:
    if refresh:
        cached_prices.clear()
        prices = load_price_history(tickers, years=years, refresh=True)
    else:
        prices = cached_prices(tuple(tickers), years, False)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

holdings_px = prices[selected].dropna(how="any")
if holdings_px.shape[0] < 200:
    st.error("Not enough overlapping price history for the selected names.")
    st.stop()

simple, _log_ret = compute_returns(holdings_px)
mean = simple.mean()
cov = simple.cov()
equal = equal_weight_series(selected)

frontier, random_df, min_var_payload, max_sharpe_payload = cached_optimizer(
    mean, cov, float(risk_free), float(max_weight), 3500
)
min_var = pack_portfolio(min_var_payload, mean.index)
max_sharpe = pack_portfolio(max_sharpe_payload, mean.index)

if mode == "Max Sharpe":
    active_weights = max_sharpe.weights
    active_label = "Max Sharpe"
elif mode == "Min Variance":
    active_weights = min_var.weights
    active_label = "Min Variance"
elif mode == "Equal weight":
    active_weights = equal
    active_label = "Equal Weight"
elif mode == "Custom mix":
    active_weights = align_weights(selected, custom_weights)
    active_label = "Custom Mix"
else:
    active_weights = interpolate_frontier(frontier, risk_tolerance)
    active_label = f"Frontier ({risk_tolerance:.0f} risk)"

active_weights = align_weights(selected, active_weights)
port_rets = portfolio_returns(simple, active_weights)
metrics = portfolio_metrics(port_rets, float(risk_free), name=active_label)
active_port = OptimizedPortfolio(
    weights=active_weights,
    ann_return=metrics["ann_return"],
    ann_vol=metrics["ann_vol"],
    sharpe=metrics["sharpe"],
    label=active_label,
)

spy_rets = prices["SPY"].pct_change().reindex(simple.index).dropna()
bnd_rets = prices["BND"].pct_change().reindex(simple.index) if "BND" in prices.columns else None
comparison = build_benchmark_comparison(
    asset_returns=simple,
    spy_returns=prices["SPY"].pct_change(),
    bnd_returns=prices["BND"].pct_change() if "BND" in prices.columns else None,
    selected=active_port,
    equal_weight=equal,
    min_var=min_var,
    max_sharpe=max_sharpe,
    risk_free_rate=float(risk_free),
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ann. return", f"{metrics['ann_return']:.1%}")
k2.metric("Ann. volatility", f"{metrics['ann_vol']:.1%}")
k3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
k4.metric("VaR 95% (1-day)", f"{metrics['var_95']:.2%}")
k5.metric("CVaR 95% (1-day)", f"{metrics['cvar_95']:.2%}")

if "S&P 500 (SPY)" in comparison.index:
    spy_sharpe = comparison.loc["S&P 500 (SPY)", "sharpe"]
    lift = metrics["sharpe"] / spy_sharpe - 1 if spy_sharpe else np.nan
    st.caption(
        f"{active_label} Sharpe vs. SPY: **{lift:+.1%}**  ·  "
        f"Max drawdown {metrics['max_drawdown']:.1%}  ·  "
        f"{holdings_px.index.min().date()} → {holdings_px.index.max().date()}  ·  "
        f"{len(simple):,} daily observations"
    )

overview, risk_tab, frontier_tab, mc_tab, bench_tab = st.tabs(
    ["Overview", "Risk analytics", "Efficient frontier", "Monte Carlo", "Benchmark"]
)

with overview:
    st.markdown('<div class="section-title">Holdings and sectors</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(
            plot_normalized_prices(holdings_px, "Growth of $1 — selected holdings"),
            use_container_width=True,
            key="overview_prices",
        )
    with right:
        st.plotly_chart(
            plot_weights(active_weights, f"{active_label} weights"),
            use_container_width=True,
            key="overview_weights",
        )
        meta = pd.DataFrame(
            {"Ticker": selected, "Sector": [DEFAULT_UNIVERSE.get(t, "") for t in selected]}
        )
        st.dataframe(meta, hide_index=True, width="stretch")

with risk_tab:
    st.markdown('<div class="section-title">Per-asset risk, correlation, VaR and CVaR</div>', unsafe_allow_html=True)
    asset_table = asset_risk_table(simple, float(risk_free), DEFAULT_UNIVERSE)
    st.dataframe(metrics_to_display_frame(asset_table), width="stretch")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(
            plot_correlation_heatmap(simple.corr()),
            use_container_width=True,
            key="risk_corr",
        )
    with c2:
        st.plotly_chart(
            plot_var_histogram(port_rets, metrics["var_95"], metrics["cvar_95"]),
            use_container_width=True,
            key="risk_var_hist",
        )
        st.markdown(
            f"""
**How to read the tail numbers**

- **VaR 95%** = {metrics['var_95']:.2%} of portfolio value: historically, 1-day losses were worse than this only 5% of the time.
- **CVaR 95%** = {metrics['cvar_95']:.2%}: the *average* loss on those worst 5% of days (expected shortfall).
- **VaR 99%** = {metrics['var_99']:.2%} · **CVaR 99%** = {metrics['cvar_99']:.2%}
            """
        )

with frontier_tab:
    st.markdown('<div class="section-title">Modern Portfolio Theory</div>', unsafe_allow_html=True)
    spy_ann_ret = float(spy_rets.mean() * 252)
    spy_ann_vol = float(spy_rets.std(ddof=1) * np.sqrt(252))
    highlights = [
        (max_sharpe.label, max_sharpe.ann_vol, max_sharpe.ann_return, "#148F77"),
        (min_var.label, min_var.ann_vol, min_var.ann_return, "#C4A35A"),
        (active_label, active_port.ann_vol, active_port.ann_return, "#C0392B"),
        ("S&P 500", spy_ann_vol, spy_ann_ret, "#1B4F72"),
    ]
    st.plotly_chart(
        plot_efficient_frontier(random_df, frontier, highlights),
        use_container_width=True,
        key="frontier_scatter",
    )
    w1, w2, w3 = st.columns(3)
    with w1:
        st.plotly_chart(
            plot_weights(max_sharpe.weights, "Max Sharpe"),
            use_container_width=True,
            key="frontier_max_sharpe_w",
        )
        st.caption(f"Sharpe {max_sharpe.sharpe:.2f} · vol {max_sharpe.ann_vol:.1%}")
    with w2:
        st.plotly_chart(
            plot_weights(min_var.weights, "Min Variance"),
            use_container_width=True,
            key="frontier_min_var_w",
        )
        st.caption(f"Sharpe {min_var.sharpe:.2f} · vol {min_var.ann_vol:.1%}")
    with w3:
        st.plotly_chart(
            plot_weights(active_weights, active_label),
            use_container_width=True,
            key="frontier_active_w",
        )
        st.caption(f"Sharpe {active_port.sharpe:.2f} · vol {active_port.ann_vol:.1%}")

with mc_tab:
    st.markdown('<div class="section-title">Forward wealth paths from historical mean / covariance</div>', unsafe_allow_html=True)
    paths = simulate_portfolio_paths(
        mean=mean,
        cov=cov,
        weights=active_weights,
        n_years=int(mc_years),
        n_sims=int(n_sims),
        initial_value=float(start_wealth),
        seed=42,
    )
    summary = path_summary(paths, float(start_wealth))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median terminal value", f"${summary['median_terminal']:,.0f}")
    m2.metric("5th percentile (stress)", f"${summary['p5_terminal']:,.0f}")
    m3.metric("Prob. of loss", f"{summary['prob_loss']:.1%}")
    m4.metric("Prob. of doubling", f"{summary['prob_double']:.1%}")
    st.plotly_chart(
        plot_monte_carlo_fan(paths, float(start_wealth)),
        use_container_width=True,
        key="mc_fan",
    )
    st.caption(
        "Shocks are i.i.d. draws from a normal distribution with the portfolio's historical daily mean and "
        "volatility. This is a risk illustration, not a forecast — fat tails and regime shifts are not modeled."
    )

with bench_tab:
    st.markdown('<div class="section-title">Risk-adjusted performance vs. SPY and 60/40</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_benchmark_bars(comparison), use_container_width=True, key="bench_sharpe_bars")
    show_cols = [
        c
        for c in [
            "ann_return",
            "ann_vol",
            "sharpe",
            "var_95",
            "cvar_95",
            "max_drawdown",
            "sharpe_vs_spy",
            "sharpe_vs_60_40",
        ]
        if c in comparison.columns
    ]
    st.dataframe(metrics_to_display_frame(comparison[show_cols]), width="stretch")

    if active_label in comparison.index and "S&P 500 (SPY)" in comparison.index:
        a = comparison.loc[active_label]
        s = comparison.loc["S&P 500 (SPY)"]
        sixty = comparison.loc["60/40 (SPY/BND)"] if "60/40 (SPY/BND)" in comparison.index else None
        spy_lift = a["sharpe"] / s["sharpe"] - 1 if s["sharpe"] else np.nan
        sixty_lift = (a["sharpe"] / sixty["sharpe"] - 1) if sixty is not None and sixty["sharpe"] else np.nan
        st.success(
            f"**Headline:** {active_label} delivered a Sharpe of **{a['sharpe']:.2f}** vs. "
            f"SPY **{s['sharpe']:.2f}** ({spy_lift:+.0%} risk-adjusted) "
            + (f"and vs. 60/40 **{sixty['sharpe']:.2f}** ({sixty_lift:+.0%})." if sixty is not None else ".")
        )

    csv = comparison.to_csv()
    st.download_button(
        "Download benchmark table (CSV)",
        csv,
        file_name="benchmark_comparison.csv",
        key="bench_csv",
    )

st.markdown("---")
st.caption("Built by Paridhi Bhardwaj · prices via Yahoo Finance (yfinance) · not investment advice.")
