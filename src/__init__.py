"""Portfolio risk analytics and mean-variance optimization."""

from .data import DEFAULT_UNIVERSE, BENCHMARKS, load_price_history
from .risk import (
    compute_returns,
    asset_risk_table,
    portfolio_returns,
    portfolio_metrics,
    correlation_matrix,
)
from .optimize import (
    random_portfolios,
    min_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier,
    interpolate_frontier,
)
from .monte_carlo import simulate_portfolio_paths, path_summary
from .benchmark import build_benchmark_comparison

__all__ = [
    "DEFAULT_UNIVERSE",
    "BENCHMARKS",
    "load_price_history",
    "compute_returns",
    "asset_risk_table",
    "portfolio_returns",
    "portfolio_metrics",
    "correlation_matrix",
    "random_portfolios",
    "min_variance_portfolio",
    "max_sharpe_portfolio",
    "efficient_frontier",
    "interpolate_frontier",
    "simulate_portfolio_paths",
    "path_summary",
    "build_benchmark_comparison",
]
