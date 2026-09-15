"""Benchmark books: S&P 500, 60/40, equal-weight, and the optimized mix."""

from __future__ import annotations

import pandas as pd

from .optimize import OptimizedPortfolio
from .risk import align_weights, portfolio_metrics, portfolio_returns


def sixty_forty_returns(spy: pd.Series, bnd: pd.Series) -> pd.Series:
    """Daily-rebalanced 60% SPY / 40% BND."""
    aligned = pd.concat([spy, bnd], axis=1).dropna()
    aligned.columns = ["SPY", "BND"]
    return 0.60 * aligned["SPY"] + 0.40 * aligned["BND"]


def build_benchmark_comparison(
    asset_returns: pd.DataFrame,
    spy_returns: pd.Series,
    bnd_returns: pd.Series | None,
    selected: OptimizedPortfolio,
    equal_weight: pd.Series,
    min_var: OptimizedPortfolio,
    max_sharpe: OptimizedPortfolio,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Side-by-side risk-adjusted performance vs. SPY and 60/40."""
    series = {
        selected.label: portfolio_returns(asset_returns, selected.weights),
        "Equal Weight": portfolio_returns(asset_returns, equal_weight),
        min_var.label: portfolio_returns(asset_returns, min_var.weights),
        max_sharpe.label: portfolio_returns(asset_returns, max_sharpe.weights),
        "S&P 500 (SPY)": spy_returns.reindex(asset_returns.index).dropna(),
    }
    if bnd_returns is not None:
        series["60/40 (SPY/BND)"] = sixty_forty_returns(spy_returns, bnd_returns)

    rows = []
    for name, rets in series.items():
        aligned = rets.reindex(asset_returns.index).dropna()
        if aligned.empty:
            continue
        rows.append(portfolio_metrics(aligned, risk_free_rate, name=name))

    table = pd.DataFrame(rows).set_index("name")
    spy_sharpe = table.loc["S&P 500 (SPY)", "sharpe"] if "S&P 500 (SPY)" in table.index else None
    if spy_sharpe and spy_sharpe != 0:
        table["sharpe_vs_spy"] = table["sharpe"] / spy_sharpe - 1.0
    else:
        table["sharpe_vs_spy"] = pd.NA

    sixty_sharpe = table.loc["60/40 (SPY/BND)", "sharpe"] if "60/40 (SPY/BND)" in table.index else None
    if sixty_sharpe and sixty_sharpe != 0:
        table["sharpe_vs_60_40"] = table["sharpe"] / sixty_sharpe - 1.0
    else:
        table["sharpe_vs_60_40"] = pd.NA

    return table


def equal_weight_series(tickers: list[str]) -> pd.Series:
    return align_weights(tickers, {t: 1.0 for t in tickers})
