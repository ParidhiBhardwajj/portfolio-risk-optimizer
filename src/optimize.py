"""
Mean-variance optimization (Modern Portfolio Theory).

Uses scipy.optimize (SLSQP) to recover the minimum-variance and maximum-Sharpe
portfolios, traces the efficient frontier, and samples random long-only books
for the scatter plot behind the frontier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk import TRADING_DAYS


@dataclass
class OptimizedPortfolio:
    weights: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    label: str


def _portfolio_moments(
    weights: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    ret = float(weights @ mean * TRADING_DAYS)
    vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(TRADING_DAYS))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe


def _bounds(n: int, max_weight: float) -> tuple:
    cap = min(max(max_weight, 1.0 / n), 1.0)
    return tuple((0.0, cap) for _ in range(n))


def _long_only(n: int, max_weight: float) -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def min_variance_portfolio(
    mean: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    max_weight: float = 1.0,
) -> OptimizedPortfolio:
    """Global minimum-variance long-only portfolio."""
    n = len(mean)
    mu = mean.values
    sigma = cov.values
    bounds = _bounds(n, max_weight)
    constraints = [_long_only(n, max_weight)]

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w)

    starts = [np.ones(n) / n]
    rng = np.random.default_rng(7)
    for _ in range(4):
        raw = rng.random(n)
        starts.append(raw / raw.sum())

    best = None
    for w0 in starts:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-12},
        )
        if not result.success:
            continue
        candidate = result.x / result.x.sum()
        score = objective(candidate)
        if best is None or score < best[0]:
            best = (score, candidate)

    weights = best[1] if best is not None else np.ones(n) / n
    ret, vol, sharpe = _portfolio_moments(weights, mu, sigma, risk_free_rate)
    return OptimizedPortfolio(
        weights=pd.Series(weights, index=mean.index),
        ann_return=ret,
        ann_vol=vol,
        sharpe=sharpe,
        label="Min Variance",
    )


def max_sharpe_portfolio(
    mean: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    max_weight: float = 1.0,
) -> OptimizedPortfolio:
    """Maximum Sharpe ratio long-only portfolio."""
    n = len(mean)
    mu = mean.values
    sigma = cov.values
    rf_daily = risk_free_rate / TRADING_DAYS
    bounds = _bounds(n, max_weight)
    constraints = [_long_only(n, max_weight)]

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ sigma @ w))
        if vol < 1e-12:
            return 0.0
        return -(ret - rf_daily) / vol

    starts = [np.ones(n) / n]
    rng = np.random.default_rng(11)
    for _ in range(6):
        raw = rng.random(n)
        starts.append(raw / raw.sum())

    best = None
    for w0 in starts:
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success and result.x is None:
            continue
        candidate = np.clip(result.x, 0, None)
        if candidate.sum() <= 0:
            continue
        candidate = candidate / candidate.sum()
        score = neg_sharpe(candidate)
        if best is None or score < best[0]:
            best = (score, candidate)

    weights = best[1] if best is not None else np.ones(n) / n
    ret, vol, sharpe = _portfolio_moments(weights, mu, sigma, risk_free_rate)
    return OptimizedPortfolio(
        weights=pd.Series(weights, index=mean.index),
        ann_return=ret,
        ann_vol=vol,
        sharpe=sharpe,
        label="Max Sharpe",
    )


def efficient_frontier(
    mean: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.043,
    n_points: int = 40,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Trace the long-only frontier by minimizing variance at target returns.

    Returns a DataFrame with columns: ann_return, ann_vol, sharpe, plus a
    weight column per ticker. Sorted by volatility.
    """
    n = len(mean)
    mu = mean.values
    sigma = cov.values
    bounds = _bounds(n, max_weight)
    min_ret = float(mean.min())
    max_ret = float(mean.max())
    # Stay inside the feasible long-only return range, with a small pad.
    targets = np.linspace(min_ret, max_ret, n_points)

    rows = []
    w0 = np.ones(n) / n
    for target in targets:
        constraints = [
            _long_only(n, max_weight),
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]

        def objective(w: np.ndarray) -> float:
            return float(w @ sigma @ w)

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 400, "ftol": 1e-12},
        )
        if not result.success:
            continue
        w = np.clip(result.x, 0, None)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        w0 = w
        ret, vol, sharpe = _portfolio_moments(w, mu, sigma, risk_free_rate)
        row = {"ann_return": ret, "ann_vol": vol, "sharpe": sharpe}
        for ticker, weight in zip(mean.index, w):
            row[ticker] = float(weight)
        rows.append(row)

    if not rows:
        fallback = min_variance_portfolio(mean, cov, risk_free_rate, max_weight)
        row = {
            "ann_return": fallback.ann_return,
            "ann_vol": fallback.ann_vol,
            "sharpe": fallback.sharpe,
            **fallback.weights.to_dict(),
        }
        return pd.DataFrame([row])

    frontier = pd.DataFrame(rows).sort_values("ann_vol").drop_duplicates(subset=["ann_vol"])
    return frontier.reset_index(drop=True)


def random_portfolios(
    mean: pd.Series,
    cov: pd.DataFrame,
    n_portfolios: int = 4000,
    risk_free_rate: float = 0.043,
    max_weight: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Dirichlet-sampled long-only books used as the cloud behind the frontier."""
    n = len(mean)
    rng = np.random.default_rng(seed)
    alpha = np.ones(n)
    weights = rng.dirichlet(alpha, size=n_portfolios)
    if max_weight < 1.0:
        # Reject samples that breach the mandate cap, then refill.
        keep = []
        attempts = 0
        while len(keep) < n_portfolios and attempts < n_portfolios * 20:
            batch = rng.dirichlet(alpha, size=n_portfolios)
            valid = batch[batch.max(axis=1) <= max_weight + 1e-9]
            keep.extend(valid.tolist())
            attempts += 1
        if keep:
            weights = np.array(keep[:n_portfolios])
        else:
            weights = np.ones((n_portfolios, n)) / n

    mu = mean.values
    sigma = cov.values
    rets = weights @ mu * TRADING_DAYS
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights)) * np.sqrt(TRADING_DAYS)
    sharpe = (rets - risk_free_rate) / np.where(vols > 0, vols, np.nan)
    frame = pd.DataFrame({"ann_return": rets, "ann_vol": vols, "sharpe": sharpe})
    for i, ticker in enumerate(mean.index):
        frame[ticker] = weights[:, i]
    return frame


def interpolate_frontier(frontier: pd.DataFrame, risk_tolerance: float) -> pd.Series:
    """
    Map a 0–100 risk-tolerance slider onto the frontier.

    0 = lowest-volatility book on the frontier; 100 = highest-return book.
    """
    if frontier.empty:
        raise ValueError("Frontier is empty.")
    ordered = frontier.sort_values("ann_vol").reset_index(drop=True)
    t = float(np.clip(risk_tolerance, 0.0, 100.0)) / 100.0
    idx = t * (len(ordered) - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    skip = {"ann_return", "ann_vol", "sharpe"}
    cols = [c for c in ordered.columns if c not in skip]
    if lo == hi:
        return ordered.loc[lo, cols].astype(float)
    frac = idx - lo
    blended = (1 - frac) * ordered.loc[lo, cols].astype(float) + frac * ordered.loc[hi, cols].astype(float)
    return blended / blended.sum()
