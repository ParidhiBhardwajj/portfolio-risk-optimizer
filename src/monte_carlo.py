"""
Monte Carlo wealth-path simulation.

Draws correlated daily shocks from the historical mean / covariance of the
selected book and compounds them into a fan chart of possible future values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import TRADING_DAYS


def simulate_portfolio_paths(
    mean: pd.Series,
    cov: pd.DataFrame,
    weights: pd.Series,
    n_years: int = 5,
    n_sims: int = 4000,
    initial_value: float = 100_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate future portfolio values under a multivariate-normal return model.

    Parameters
    ----------
    mean, cov : historical *daily* mean vector and covariance matrix
    weights : long-only weights aligned to mean.index
    n_years : forecast horizon
    n_sims : number of paths
    initial_value : starting wealth
    seed : RNG seed for reproducibility

    Returns
    -------
    DataFrame of shape (n_days, n_sims) with simulated wealth paths.
    """
    tickers = list(mean.index)
    w = weights.reindex(tickers).fillna(0.0).astype(float)
    w = w / w.sum()

    port_mean = float(w.values @ mean.values)
    port_var = float(w.values @ cov.loc[tickers, tickers].values @ w.values)
    port_std = np.sqrt(max(port_var, 0.0))

    n_days = int(n_years * TRADING_DAYS)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=port_mean, scale=port_std, size=(n_days, n_sims))
    growth = np.cumprod(1.0 + shocks, axis=0)
    paths = initial_value * growth
    idx = pd.RangeIndex(1, n_days + 1, name="day")
    return pd.DataFrame(paths, index=idx)


def path_percentiles(
    paths: pd.DataFrame,
    bands: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """Percentile fan used by the chart (columns named p5, p25, ...)."""
    out = pd.DataFrame(index=paths.index)
    for q in bands:
        out[f"p{int(q)}"] = paths.quantile(q / 100.0, axis=1)
    return out


def path_summary(paths: pd.DataFrame, initial_value: float) -> dict:
    """Terminal-wealth statistics an analyst would quote from the simulation."""
    terminal = paths.iloc[-1]
    return {
        "median_terminal": float(terminal.median()),
        "mean_terminal": float(terminal.mean()),
        "p5_terminal": float(terminal.quantile(0.05)),
        "p95_terminal": float(terminal.quantile(0.95)),
        "prob_loss": float((terminal < initial_value).mean()),
        "prob_double": float((terminal >= 2 * initial_value).mean()),
        "n_sims": int(paths.shape[1]),
        "n_days": int(paths.shape[0]),
    }
