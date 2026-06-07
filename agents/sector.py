"""Sector rotation: rank ticker vs sector peers."""

from __future__ import annotations

from agents.context import resolve_sector_etf
from data import fetch_ticker_df


def analyze_sector(ticker: str, df) -> dict:
    etf = resolve_sector_etf(ticker)
    try:
        sector_df = fetch_ticker_df(etf, period="60d")
    except Exception:
        return {"sector_etf": etf, "sector_rank": "UNKNOWN", "summary": "Sector data N/A"}

    stock_ret = (df["Close"].iloc[-1] - df["Close"].iloc[-21]) / df["Close"].iloc[-21]
    sector_ret = (sector_df["Close"].iloc[-1] - sector_df["Close"].iloc[-21]) / sector_df["Close"].iloc[-21]
    rel = stock_ret - sector_ret

    if rel > 0.05:
        rank = "LEADER"
        score = 85
        summary = f"Sector leader vs {etf} (+{rel*100:.1f}%)"
    elif rel > 0:
        rank = "OUTPERFORMER"
        score = 65
        summary = f"Outperforming {etf} (+{rel*100:.1f}%)"
    elif rel > -0.05:
        rank = "INLINE"
        score = 45
        summary = f"Inline with {etf} ({rel*100:+.1f}%)"
    else:
        rank = "LAGGARD"
        score = 20
        summary = f"Lagging {etf} ({rel*100:+.1f}%) — weak sector pick"

    return {
        "sector_etf": etf,
        "sector_rank": rank,
        "sector_score": score,
        "relative_vs_sector": round(rel * 100, 2),
        "summary": summary,
    }