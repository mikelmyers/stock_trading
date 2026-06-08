"""Load OHLCV from repo-local EODHD and Arandkei files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATA_DIR

EODHD_DIR = DATA_DIR / "eodhd"
ARANDKEI_DIR = DATA_DIR / "arandkei" / "delisted"

_EODHD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
_ARANDKEI_RENAME = {
    "DATE": "Date",
    "OPEN": "Open",
    "HIGH": "High",
    "LOW": "Low",
    "CLOSE": "Close",
    "ADJ_CLOSE": "Adj Close",
    "VOLUME": "Volume",
}


def _eodhd_path(ticker: str, asset_class: str = "stocks") -> Path | None:
    folder = EODHD_DIR / ("Stocks" if asset_class == "stocks" else "ETFs")
    path = folder / f"{ticker.lower()}.us.txt"
    return path if path.exists() else None


def load_eodhd_ticker(ticker: str, asset_class: str = "stocks") -> pd.DataFrame:
    """Return daily OHLCV for a US ticker from the local EODHD extract."""
    path = _eodhd_path(ticker, asset_class=asset_class)
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def list_eodhd_tickers(asset_class: str = "stocks") -> list[str]:
    folder = EODHD_DIR / ("Stocks" if asset_class == "stocks" else "ETFs")
    if not folder.exists():
        return []
    return sorted(p.name.replace(".us.txt", "").upper() for p in folder.glob("*.us.txt"))


def load_arandkei_ticker(ticker: str) -> pd.DataFrame:
    """Return delisted OHLCV for a ticker from the Arandkei archive."""
    matches = list(ARANDKEI_DIR.glob(f"{ticker.upper()}_*_ARANDKEI.csv"))
    if not matches:
        return pd.DataFrame()

    df = pd.read_csv(matches[0])
    df = df.rename(columns=_ARANDKEI_RENAME)
    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df


def list_arandkei_tickers() -> list[str]:
    if not ARANDKEI_DIR.exists():
        return []
    tickers: list[str] = []
    for path in ARANDKEI_DIR.glob("*_ARANDKEI.csv"):
        tickers.append(path.name.split("_", 1)[0].upper())
    return sorted(set(tickers))