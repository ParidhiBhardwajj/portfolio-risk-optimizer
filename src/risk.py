"""
Return and risk analytics.

Daily / log returns, annualized volatility, Sharpe, historical VaR / CVaR,
max drawdown, and the correlation matrix that feeds the optimizer.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (simple daily returns, log returns) aligned on the same index."""
    simple = prices.pct_change().dropna(how="all")
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    return simple, log_ret


def annualize_return(daily_mean: float, periods: int = TRADING_DAYS) -> float:
    return float(daily_mean * periods)


def annualize_vol(daily_std: float, periods: int = TRADING_DAYS) -> float:
    return float(daily_std * np.sqrt(periods))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.043) -> float:
    """Annualized Sharpe using arithmetic mean and sample volatility."""
    excess = daily_returns.mean() * TRADING_DAYS - risk_free_rate
    vol = annualize_vol(daily_returns.std(ddof=1))
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess / vol)


def historical_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """1-day historical VaR as a positive loss number (e.g. 0.02 = 2%)."""
    alpha = 1.0 - confidence
    return float(-np.quantile(daily_returns.dropna(), alpha))


def historical_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """1-day historical CVaR / expected shortfall (positive loss number)."""
    alpha = 1.0 - confidence
    cutoff = np.quantile(daily_returns.dropna(), alpha)
    tail = daily_returns.dropna()[daily_returns.dropna() <= cutoff]
    if tail.empty:
        return float(-cutoff)
    return float(-tail.mean())


def max_drawdown(daily_returns: pd.Series) -> float:
    """Peak-to-trough decline of a wealth index built from daily returns."""
    wealth = (1 + daily_returns.fillna(0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series | np.ndarray) -> pd.Series:
    """Daily rebalanced portfolio returns."""
    w = pd.Series(weights, index=asset_returns.columns, dtype=float)
    w = w / w.sum()
    return asset_returns.fillna(0.0).dot(w)


def portfolio_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.043,
    name: str = "Portfolio",
) -> dict:
    """Analyst-style risk snapshot for a return series."""
    clean = daily_returns.dropna()
    ann_ret = annualize_return(clean.mean())
    ann_vol = annualize_vol(clean.std(ddof=1))
    return {
        "name": name,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe_ratio(clean, risk_free_rate),
        "var_95": historical_var(clean, 0.95),
        "var_99": historical_var(clean, 0.99),
        "cvar_95": historical_cvar(clean, 0.95),
        "cvar_99": historical_cvar(clean, 0.99),
        "max_drawdown": max_drawdown(clean),
        "obs": int(clean.shape[0]),
    }


def asset_risk_table(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.043,
    sectors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-asset return / risk / Sharpe / VaR / CVaR table."""
    rows = []
    for ticker in returns.columns:
        m = portfolio_metrics(returns[ticker], risk_free_rate, name=ticker)
        m["sector"] = (sectors or {}).get(ticker, "")
        rows.append(m)
    table = pd.DataFrame(rows).set_index("name")
    ordered = [
        "sector",
        "ann_return",
        "ann_vol",
        "sharpe",
        "var_95",
        "cvar_95",
        "var_99",
        "cvar_99",
        "max_drawdown",
        "obs",
    ]
    return table[[c for c in ordered if c in table.columns]]


def align_weights(tickers: Iterable[str], weights: dict | pd.Series | np.ndarray) -> pd.Series:
    """Map a weight vector onto `tickers` and renormalize to 1."""
    tickers = list(tickers)
    if isinstance(weights, dict):
        series = pd.Series({t: float(weights.get(t, 0.0)) for t in tickers})
    else:
        series = pd.Series(np.asarray(weights, dtype=float), index=tickers)
    total = series.sum()
    if total <= 0:
        series = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        series = series / total
    return series
