"""Optional Finnhub enrichment (set FINNHUB_API_KEY in environment)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone


def _api_key() -> str | None:
    return os.environ.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_TOKEN")


def _get(path: str, params: dict | None = None) -> dict | list | None:
    key = _api_key()
    if not key:
        return None
    params = dict(params or {})
    params["token"] = key
    url = f"https://finnhub.io/api/v1{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "trading-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def days_to_next_earnings(ticker: str) -> int | None:
    """Return days until next earnings (negative if just reported)."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=21)).strftime("%Y-%m-%d")
    data = _get("/calendar/earnings", {"from": start, "to": end, "symbol": ticker.upper()})
    if not data or not isinstance(data, dict):
        return None

    earnings = data.get("earningsCalendar") or []
    best: int | None = None
    for row in earnings:
        if row.get("symbol", "").upper() != ticker.upper():
            continue
        raw_date = row.get("date")
        if not raw_date:
            continue
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days = (dt - now).days
            if best is None or abs(days) < abs(best):
                best = days
        except ValueError:
            continue
    return best