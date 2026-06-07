"""Earnings filter: warn when trade is near an earnings catalyst."""

from __future__ import annotations

import yfinance as yf


def analyze_earnings(ticker: str) -> dict:
    try:
        from providers.finnhub import days_to_next_earnings

        fh_days = days_to_next_earnings(ticker)
        if fh_days is not None:
            return _from_days(fh_days, source="Finnhub")

        t = yf.Ticker(ticker)
        dates = t.get_earnings_dates(limit=4)
        if dates is None or dates.empty:
            return _clear("No earnings dates available")

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        future = []
        for idx in dates.index[:4]:
            try:
                d = idx.to_pydatetime()
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                days = (d - now).days
                if -3 <= days <= 14:
                    future.append(days)
            except Exception:
                pass

        if not future:
            return _clear("No earnings within 14 days")

        nearest = min(future, key=abs)
        return _from_days(nearest, source="Yahoo")
    except Exception:
        return _clear("Earnings data unavailable")


def _from_days(nearest: int, source: str = "Yahoo") -> dict:
    if not (-3 <= nearest <= 14):
        return _clear(f"No earnings within 14 days ({source})")
    if abs(nearest) <= 3:
        return {
            "earnings_risk": "HIGH",
            "days_to_earnings": nearest,
            "allow_trade": False,
            "summary": f"Earnings in {nearest} days ({source}) — avoid new swing entries",
        }
    return {
        "earnings_risk": "MEDIUM",
        "days_to_earnings": nearest,
        "allow_trade": True,
        "summary": f"Earnings in {nearest} days ({source}) — size down or use defined risk",
    }


def _clear(msg: str) -> dict:
    return {
        "earnings_risk": "LOW",
        "days_to_earnings": None,
        "allow_trade": True,
        "summary": msg,
    }