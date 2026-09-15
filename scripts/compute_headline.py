"""Print headline Sharpe / VaR numbers used in the README."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import build_benchmark_comparison, equal_weight_series
from src.data import BENCHMARKS, DEFAULT_UNIVERSE, load_price_history
from src.optimize import max_sharpe_portfolio, min_variance_portfolio
from src.risk import compute_returns, portfolio_metrics, portfolio_returns


def main() -> None:
    tickers = list(DEFAULT_UNIVERSE) + list(BENCHMARKS)
    prices = load_price_history(tickers, years=5, refresh=False)
    holdings = list(DEFAULT_UNIVERSE)
    px = prices[holdings].dropna(how="any")
    simple, _ = compute_returns(px)
    rf = 0.043
    max_w = 0.30
    min_var = min_variance_portfolio(simple.mean(), simple.cov(), rf, max_w)
    max_sharpe = max_sharpe_portfolio(simple.mean(), simple.cov(), rf, max_w)
    equal = equal_weight_series(holdings)
    table = build_benchmark_comparison(
        asset_returns=simple,
        spy_returns=prices["SPY"].pct_change(),
        bnd_returns=prices["BND"].pct_change(),
        selected=max_sharpe,
        equal_weight=equal,
        min_var=min_var,
        max_sharpe=max_sharpe,
        risk_free_rate=rf,
    )
    print(f"Window: {px.index.min().date()} → {px.index.max().date()}  ({len(simple)} days)")
    print(table[["ann_return", "ann_vol", "sharpe", "var_95", "cvar_95", "max_drawdown", "sharpe_vs_spy", "sharpe_vs_60_40"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMax Sharpe weights")
    print(max_sharpe.weights[max_sharpe.weights > 0.01].sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.1%}"))
    print("\nMin Variance weights")
    print(min_var.weights[min_var.weights > 0.01].sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.1%}"))

    ms = table.loc["Max Sharpe"]
    spy = table.loc["S&P 500 (SPY)"]
    sixty = table.loc["60/40 (SPY/BND)"]
    print("\nHEADLINE")
    print(
        f"Max-Sharpe book Sharpe {ms['sharpe']:.2f} vs SPY {spy['sharpe']:.2f} "
        f"({ms['sharpe']/spy['sharpe']-1:+.0%}) and vs 60/40 {sixty['sharpe']:.2f} "
        f"({ms['sharpe']/sixty['sharpe']-1:+.0%}) over 5 years, "
        f"1-day 95% CVaR {ms['cvar_95']:.2%}."
    )


if __name__ == "__main__":
    main()
