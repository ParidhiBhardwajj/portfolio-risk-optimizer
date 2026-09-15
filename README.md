# Portfolio Risk & Optimization Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A long-only portfolio risk dashboard that pulls live market data, reports analyst-style tail-risk numbers, and lets you watch the efficient frontier, Sharpe ratio, and Monte Carlo wealth paths update as you change the mix.

**Headline (Sep 2021 – Sep 2026, in-sample):** the max-Sharpe book (30% single-name cap) delivered a **Sharpe of 1.59 vs. 0.53 for SPY (+201% risk-adjusted)** and **0.33 for a 60/40 SPY/BND mix (+388%)**, with 1-day 95% CVaR of 2.42%.

## What it does

1. **Universe & prices** — 12 names across four sectors plus SPY (equity benchmark) and BND (for 60/40). Five years of daily adjusted closes via `yfinance`.
2. **Returns & risk** — simple and log returns, annualized volatility, and a correlation heatmap.
3. **Sharpe, VaR, CVaR** — per asset and at the portfolio level (historical 95% / 99% VaR and expected shortfall).
4. **Efficient frontier** — mean-variance optimization with `scipy.optimize` (min-variance, max-Sharpe, and a 3,500-point random cloud).
5. **Monte Carlo** — thousands of forward wealth paths from the book’s historical mean and covariance, shown as a fan chart with 5–95% bands.
6. **Live controls** — change holdings, risk tolerance, the single-name cap, or a custom mix and the frontier / Sharpe / simulation rerun.
7. **Benchmarks** — optimized book vs. equal-weight, SPY, and a daily-rebalanced 60/40.

## Universe

| Sector | Tickers |
| --- | --- |
| Technology | AAPL, MSFT, NVDA, GOOGL |
| Financials | JPM, BAC, GS |
| Healthcare | UNH, JNJ, LLY |
| Energy | XOM, CVX |
| Benchmarks | SPY, BND |

Default mandate: long-only, weights sum to 100%, **max 30% in any one name** so the optimizer cannot collapse into a single winner.

## 5-year snapshot

Window: **2021-09-07 → 2026-09-11** (1,258 trading days). Risk-free rate 4.3%.

| Book | Ann. return | Ann. vol | Sharpe | VaR 95% (1d) | CVaR 95% (1d) | Max DD | Sharpe vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Max Sharpe | 33.1% | 18.1% | **1.59** | 1.61% | 2.42% | −16.7% | **+201%** |
| Equal weight | 23.6% | 16.8% | 1.15 | 1.63% | 2.37% | −18.0% | +118% |
| Min variance | 18.5% | 13.4% | 1.06 | 1.28% | 1.88% | −10.5% | +101% |
| S&P 500 (SPY) | 13.4% | 17.2% | 0.53 | 1.66% | 2.47% | −24.5% | — |
| 60/40 (SPY/BND) | 7.9% | 11.1% | 0.33 | 1.06% | 1.58% | −20.6% | −38% |

Max-Sharpe weights in this window: XOM 30%, LLY 24%, JNJ 22%, NVDA 20% (energy + healthcare + a capped NVDA sleeve). That mix is **in-sample** — the same returns used to estimate means/covariances are used to score the book. It is a demonstration of the optimizer, not a live track record.

## Tech stack

- **Python 3.9+**
- **Streamlit** — interactive BI dashboard
- **yfinance** — daily prices
- **pandas / numpy** — returns, covariance, risk tables
- **scipy.optimize** — SLSQP mean-variance (min-variance, max-Sharpe, frontier)
- **Plotly** — heatmap, frontier scatter, Monte Carlo fan chart

## Quick start

```bash
git clone https://github.com/ParidhiBhardwajj/portfolio-risk-optimizer.git
cd portfolio-risk-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run scripts/app.py
```

Or from the project root:

```bash
./run.sh
```

Then open [http://localhost:8501](http://localhost:8501).

A cached price file lives in `data/prices.csv` so the app runs without Yahoo on the first load. Click **Refresh prices from Yahoo Finance** in the sidebar to pull a new window.

## Project structure

```
portfolio-risk-optimizer/
├── data/prices.csv            # cached daily adjusted closes
├── scripts/
│   ├── app.py                 # Streamlit dashboard
│   └── compute_headline.py    # reprints Sharpe / VaR vs. benchmarks
├── src/
│   ├── data.py                # yfinance download + cache
│   ├── risk.py                # returns, Sharpe, VaR, CVaR, drawdown
│   ├── optimize.py            # MPT: min-var, max-Sharpe, frontier
│   ├── monte_carlo.py         # wealth-path simulation
│   ├── benchmark.py           # SPY and 60/40 comparison
│   └── visualization.py       # Plotly charts
├── requirements.txt
├── run.sh
└── README.md
```

## Dashboard tabs

1. **Overview** — growth-of-$1 chart and the active weights
2. **Risk analytics** — per-name Sharpe / VaR / CVaR, correlation heatmap, return histogram with VaR and expected-shortfall markers
3. **Efficient frontier** — random long-only cloud, frontier curve, min-variance / max-Sharpe / SPY highlighted
4. **Monte Carlo** — fan chart of 1–5 year wealth paths, median, 5th percentile, probability of loss / doubling
5. **Benchmark** — risk-adjusted scoreboard vs. SPY and 60/40, CSV export

## Method notes

- **Returns:** daily simple returns; means and vols annualized with 252 trading days.
- **Sharpe:** (annualized return − risk-free) / annualized volatility.
- **VaR / CVaR:** historical, reported as 1-day positive loss figures. CVaR is the mean loss beyond VaR (expected shortfall).
- **Optimizer:** long-only SLSQP. The frontier is the minimum-variance book at a grid of target returns. Risk tolerance slides along that frontier.
- **Monte Carlo:** i.i.d. normal shocks using the *portfolio* historical daily mean and volatility. Fat tails and regime shifts are not modeled — the fan chart is a risk illustration, not a forecast.
- **60/40:** 60% SPY + 40% BND, daily rebalanced, over the same dates as the equity book.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Paridhi Bhardwaj** — Data Analyst / Business Analyst portfolio project. Not investment advice.

---

*This project is meant to show an end-to-end market-risk workflow: data → risk statistics an analyst would actually quote → mean-variance construction → simulation → an interactive dashboard.*
