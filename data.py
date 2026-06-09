"""Market data ingestion layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_INTERVAL, DATA_PERIOD, UNIVERSE_FILE, WATCHLIST


def get_all_watchlist_tickers() -> dict[str, str]:
    """Return {ticker: cap_category} for default + universe file tickers."""
    mapping: dict[str, str] = {}
    for cap, tickers in WATCHLIST.items():
        for t in tickers:
            mapping[t.upper()] = cap

    universe_path = Path(UNIVERSE_FILE)
    if universe_path.exists():
        for line in universe_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ticker = line.split(",")[0].strip().upper()
            cap = line.split(",")[1].strip() if "," in line else "Universe"
            mapping[ticker] = cap

    return mapping


def fetch_ticker_df(
    ticker: str,
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
) -> pd.DataFrame:
    data = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data.dropna()


def fetch_multiple(
    tickers: list[str],
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers, period=period, interval=interval,
        group_by="ticker", progress=False, auto_adjust=True,
    )
    return data


def extract_ticker_df(data: pd.DataFrame, ticker: str, num_tickers: int) -> pd.DataFrame:
    # fetch_multiple uses group_by="ticker", so the symbol sits on column
    # level 0 even when only one ticker was requested; selecting by symbol
    # (rather than dropping a hardcoded level) handles both layouts.
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].dropna()
        return data.droplevel(1, axis=1).dropna()
    return data.dropna()


def get_current_price(ticker: str) -> float:
    df = fetch_ticker_df(ticker, period="5d", interval="1d")
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    return round(float(df["Close"].iloc[-1]), 2)