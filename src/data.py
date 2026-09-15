"""
Market data loader.

Pulls daily adjusted close prices from Yahoo Finance via yfinance and caches
them locally so the dashboard can run offline after the first download.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf


DEFAULT_UNIVERSE = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "LLY": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
}

BENCHMARKS = {
    "SPY": "S&P 500 ETF",
    "BND": "Total Bond Market ETF",
}

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "prices.csv"


def _normalize_close(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Return a clean Adj Close / auto-adjusted Close panel."""
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError(f"Unexpected yfinance columns: {level0.unique().tolist()}")
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        if prices.shape[1] == 1 and len(tickers) == 1:
            prices.columns = [tickers[0]]

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices.dropna(how="all")


def download_prices(
    tickers: Sequence[str],
    start: str | datetime,
    end: str | datetime | None = None,
) -> pd.DataFrame:
    """Download daily auto-adjusted close prices for `tickers`."""
    unique = list(dict.fromkeys(tickers))
    raw = yf.download(
        tickers=unique,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    prices = _normalize_close(raw, unique)
    missing = [t for t in unique if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns for: {', '.join(missing)}")
    return prices[unique]


def load_price_history(
    tickers: Iterable[str],
    years: int = 5,
    refresh: bool = False,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load 3–5 years of daily prices, using a local CSV cache when possible.

    Parameters
    ----------
    tickers : iterable of str
        Symbols to include (holdings plus any benchmarks).
    years : int
        Lookback window in years.
    refresh : bool
        If True, ignore the cache and pull a fresh download.
    cache_path : path, optional
        Override the default `data/prices.csv` location.
    """
    tickers = list(dict.fromkeys(tickers))
    path = Path(cache_path) if cache_path else CACHE_PATH
    end = datetime.today().date()
    start = end - timedelta(days=int(years * 365.25) + 7)

    def _from_yahoo() -> pd.DataFrame:
        prices = download_prices(tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)
        return prices

    if refresh or not path.exists():
        return _from_yahoo()

    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    have_all = all(t in cached.columns for t in tickers)
    span_ok = not cached.empty and cached.index.min().date() <= start + timedelta(days=10)
    if have_all and span_ok:
        sliced = cached.loc[cached.index >= pd.Timestamp(start), tickers].dropna(how="all")
        if len(sliced) > 200:
            return sliced

    return _from_yahoo()
