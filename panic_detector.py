"""
Market Panic Detector — data + systemic-risk metrics.

Pipeline
--------
1. Download ~5 years of daily adjusted closes for 50 S&P 500 stocks (yfinance).
2. Daily log returns.
3. Rolling W-day (default 60) correlation matrix for every trading day.
4. Per window: mean pairwise correlation and the largest eigenvalue (lambda_1).

lambda_1 of an N x N correlation matrix ranges from 1 (no common factor) to N
(every stock moves as one). lambda_1 / N is the share of total variance explained
by the "market mode" — it spikes when diversification disappears, i.e. in panics.

Run directly for a text summary:  python panic_detector.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 50 large caps spread across all 11 GICS sectors, all listed for the full 5 years.
SECTORS: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "ADBE"],
    "Communication Services": ["GOOGL", "META", "VZ", "DIS", "NFLX"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "TSLA"],
    "Consumer Staples": ["PG", "KO", "WMT", "COST", "PEP"],
    "Health Care": ["JNJ", "UNH", "PFE", "MRK", "ABT"],
    "Financials": ["JPM", "BAC", "GS", "BRK-B", "V"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "LMT"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Real Estate": ["AMT", "PLD", "O"],
    "Materials": ["LIN", "NEM", "FCX", "SHW"],
}

# 11 sectors is more than a chart can tell apart by colour, so the network
# colours nodes by 8 economically related groups (hover still shows the sector).
SECTOR_GROUP: dict[str, str] = {
    "Information Technology": "Tech & Media",
    "Communication Services": "Tech & Media",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Health Care": "Health Care",
    "Financials": "Financials",
    "Industrials": "Industrials & Materials",
    "Materials": "Industrials & Materials",
    "Energy": "Energy",
    "Utilities": "Utilities & Real Estate",
    "Real Estate": "Utilities & Real Estate",
}

TICKERS: list[str] = [t for ts in SECTORS.values() for t in ts]
TICKER_SECTOR: dict[str, str] = {t: s for s, ts in SECTORS.items() for t in ts}
assert len(TICKERS) == 50, len(TICKERS)

CACHE_DIR = Path(__file__).parent / "data"

# Mean pairwise correlation at which each regime starts.
#   Calm / Dispersed        < 0.25   stocks mostly follow their own news
#   Elevated Stress   0.25 – 0.40   a common driver is taking over
#   Panic / Clustering    >= 0.40   the market moves as one herd
REGIME_BANDS = {"Elevated": 0.25, "Panic": 0.40}

# Named market episodes: (label, search start, search end, one-line description).
# A preset jumps to the most-clustered window (max lambda_1) inside its search range.
EPISODES = {
    "covid": ("COVID Crash", "2020-02-20", "2020-06-30",
              "Feb–Mar 2020: the pandemic sell-off, the fastest bear market on record."),
    "rates": ("Rate Shock", "2022-05-01", "2022-12-31",
              "2022: the Fed's fastest hiking cycle in decades sent stocks and bonds down together."),
    "banks": ("Bank Scare", "2023-03-08", "2023-05-15",
              "Mar 2023: Silicon Valley Bank failed and regional-bank fears spread."),
    "yen": ("Yen Unwind", "2024-08-01", "2024-10-15",
            "Aug 2024: a sudden unwind of yen-funded trades caused a sharp one-day global drop."),
    "tariff": ("Tariff Shock", "2025-04-01", "2025-07-15",
               "Apr 2025: sweeping new US tariffs triggered a fast, broad sell-off."),
}


# --------------------------------------------------------------------------- data
def fetch_prices(years: int = 5, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices (rows = dates, cols = tickers), cached on disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"prices_{years}y.csv"
    if cache.exists() and not refresh:
        age_h = (pd.Timestamp.now().timestamp() - cache.stat().st_mtime) / 3600
        if age_h < 24:
            return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(
        TICKERS,
        start=(pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,  # 'Close' is split- and dividend-adjusted
        progress=False,
        threads=True,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.reindex(columns=TICKERS).sort_index()

    # Drop tickers that failed to download or are mostly empty; fill tiny gaps.
    good = prices.columns[prices.notna().mean() > 0.95]
    missing = sorted(set(TICKERS) - set(good))
    if missing:
        print(f"[panic_detector] dropping tickers with insufficient data: {missing}")
    prices = prices[good].ffill(limit=3).dropna()
    if prices.empty:
        raise RuntimeError("No price data returned from Yahoo Finance.")
    prices.to_csv(cache)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ------------------------------------------------------------------------ metrics
@dataclass
class RiskResult:
    dates: pd.DatetimeIndex          # end date of each window
    tickers: list[str]
    corr: np.ndarray                 # (T, N, N) float32 correlation matrices
    metrics: pd.DataFrame            # per-date systemic-risk metrics
    window: int


def rolling_risk(returns: pd.DataFrame, window: int = 60) -> RiskResult:
    """Rolling correlation matrices + mean pairwise correlation + lambda_1."""
    X = returns.to_numpy(dtype=np.float64)
    T, N = X.shape
    if T < window:
        raise ValueError(f"Need at least {window} return observations, got {T}.")
    iu = np.triu_indices(N, k=1)

    n_win = T - window + 1
    corrs = np.empty((n_win, N, N), dtype=np.float32)
    mean_corr = np.empty(n_win)
    lam1 = np.empty(n_win)
    lam2 = np.empty(n_win)

    for k in range(n_win):
        C = np.corrcoef(X[k : k + window], rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        corrs[k] = C
        mean_corr[k] = C[iu].mean()
        ev = np.linalg.eigvalsh(C)  # ascending, symmetric -> real
        lam1[k], lam2[k] = ev[-1], ev[-2]

    dates = returns.index[window - 1 :]
    m = pd.DataFrame(
        {
            "mean_corr": mean_corr,
            "lambda1": lam1,
            "lambda1_share": lam1 / N,   # fraction of variance in the market mode
            "lambda2": lam2,
        },
        index=dates,
    )
    # Percentile of each day's mean correlation within the whole sample (context only).
    m["corr_pctile"] = m["mean_corr"].rank(pct=True)
    # Regime from fixed, interpretable mean-correlation bands (stable across lookbacks).
    m["regime"] = pd.cut(
        m["mean_corr"],
        bins=[-1.0, REGIME_BANDS["Elevated"], REGIME_BANDS["Panic"], 1.0],
        labels=["Calm", "Elevated", "Panic"],
        right=False,
    )
    return RiskResult(dates=dates, tickers=list(returns.columns), corr=corrs,
                      metrics=m, window=window)


def equal_weight_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Drawdown of an equal-weight, daily-rebalanced portfolio of the universe."""
    ret = prices.pct_change().mean(axis=1).fillna(0.0)
    idx = (1 + ret).cumprod()
    return idx / idx.cummax() - 1.0


def stress_peaks(series: pd.Series, n: int = 6, min_gap_days: int = 90) -> list[pd.Timestamp]:
    """Top-n local maxima at least `min_gap_days` apart (greedy)."""
    picked: list[pd.Timestamp] = []
    for d, _ in series.sort_values(ascending=False).items():
        if all(abs((d - p).days) >= min_gap_days for p in picked):
            picked.append(d)
        if len(picked) == n:
            break
    return sorted(picked)


if __name__ == "__main__":
    px = fetch_prices()
    res = rolling_risk(log_returns(px))
    m = res.metrics
    print(f"{len(res.tickers)} tickers, {px.index[0].date()} -> {px.index[-1].date()}, "
          f"{len(m)} rolling windows of {res.window} days")
    print(m[["mean_corr", "lambda1"]].describe().round(3))
    print("\nHighest-stress windows (by lambda_1):")
    for d in stress_peaks(m["lambda1"]):
        r = m.loc[d]
        print(f"  {d.date()}  lambda1={r.lambda1:5.2f}  mean_corr={r.mean_corr:.3f}  ({r.regime})")
