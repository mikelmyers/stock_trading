"""Context agent: market and sector tailwinds."""

import pandas as pd

from config import SECTOR_ETFS
from data import fetch_ticker_df


def _return_over(df: pd.DataFrame, days: int) -> float:
    if len(df) < days + 1:
        return 0.0
    start = df["Close"].iloc[-days - 1]
    end = df["Close"].iloc[-1]
    if start == 0:
        return 0.0
    return (end - start) / start


def resolve_sector_etf(ticker: str) -> str:
    try:
        info = __import__("yfinance").Ticker(ticker).info
        sector = info.get("sector", "")
        return SECTOR_ETFS.get(sector, "SPY")
    except Exception:
        return "SPY"


def analyze_context(ticker: str, df: pd.DataFrame) -> dict:
    """Measure whether context supports the breakout."""
    spy_df = fetch_ticker_df("SPY", period="60d")
    sector_etf = resolve_sector_etf(ticker)

    try:
        sector_df = fetch_ticker_df(sector_etf, period="60d")
    except Exception:
        sector_df = spy_df

    stock_20d = _return_over(df, 20)
    spy_20d = _return_over(spy_df, 20)
    sector_20d = _return_over(sector_df, 20)

    rel_vs_spy = stock_20d - spy_20d
    rel_vs_sector = stock_20d - sector_20d

    score = 0
    if rel_vs_spy > 0:
        score += 25
    if rel_vs_sector > 0:
        score += 25
    if spy_20d > 0:
        score += 25
    if sector_20d > 0:
        score += 25

    tailwind = score >= 50

    return {
        "sector_etf": sector_etf,
        "stock_20d_return": round(stock_20d * 100, 2),
        "spy_20d_return": round(spy_20d * 100, 2),
        "sector_20d_return": round(sector_20d * 100, 2),
        "relative_strength_vs_spy": round(rel_vs_spy * 100, 2),
        "relative_strength_vs_sector": round(rel_vs_sector * 100, 2),
        "context_score": score,
        "has_tailwind": tailwind,
        "summary": (
            "Tailwind: stock outperforming market and sector."
            if tailwind
            else "Headwind: weak relative strength or soft market."
        ),
    }