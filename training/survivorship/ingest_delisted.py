"""Ingest delisted-name price history into the per-ticker cache (Rung 2).

Sources (bundled in the repo under ``data/``, see ``data/arandkei/delisted`` and
``data/eodhd``):

  * Arandkei historical delisted-assets archive — has a real ``Adj Close``;
    we rescale OHLC by the adjustment ratio to match our split+dividend
    -adjusted cache convention (mirrors yfinance ``auto_adjust=True``).
  * EODHD / "Huge Stock Market Dataset" (Boris Marjanovic) — already
    split-adjusted (verified against GME's 2007 2:1 split showing no price
    discontinuity), but has no separate dividend-adjusted column. Used as-is;
    this under-adjusts slightly for dividends (a small, conservative effect).

Only ingests names that are (a) ever-S&P-member, (b) exited the index, and
(c) currently absent from ``training/cache/tickers``. Writes
``{TICKER}_max.pkl`` in the exact format ``training.history`` expects, so the
existing pipeline picks them up with zero code changes.

CLI
---
    python -m training.survivorship.ingest_delisted            # do it
    python -m training.survivorship.ingest_delisted --dry-run  # just report
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SURV_DIR = Path(__file__).resolve().parent
MEMBERSHIP_CSV = SURV_DIR / "sp500_ticker_start_end.csv"
CACHE_DIR = REPO_ROOT / "training" / "cache" / "tickers"
ARANDKEI_DIR = REPO_ROOT / "data" / "arandkei" / "delisted"
EODHD_STOCKS = REPO_ROOT / "data" / "eodhd" / "Stocks"
EODHD_ETFS = REPO_ROOT / "data" / "eodhd" / "ETFs"

MIN_BARS = 252


def _yahoo_symbol(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def _to_cache_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to Open/High/Low/Close/Volume float64, DatetimeIndex 'Date',
    columns Index named 'Price' — bit-identical shape to the yfinance cache."""
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None).floor("s")
    df.columns.name = "Price"
    return df.dropna()


def _load_arandkei(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    rename = {
        "DATE": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low",
        "CLOSE": "Close", "ADJ_CLOSE": "Adj Close", "VOLUME": "Volume",
    }
    df = df.rename(columns=rename)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    if "Adj Close" in df.columns and df["Close"].gt(0).all():
        ratio = df["Adj Close"] / df["Close"]
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * ratio
    return _to_cache_format(df)


def _load_eodhd(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return _to_cache_format(df)


def _find_source(ticker: str) -> tuple[str, Path] | None:
    matches = list(ARANDKEI_DIR.glob(f"{ticker}_*_ARANDKEI.csv"))
    if matches:
        return ("arandkei", matches[0])
    p = EODHD_STOCKS / f"{ticker.lower()}.us.txt"
    if p.exists():
        return ("eodhd", p)
    p = EODHD_ETFS / f"{ticker.lower()}.us.txt"
    if p.exists():
        return ("eodhd", p)
    return None


def candidates() -> list[str]:
    m = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["start_date", "end_date"])
    cache = {p.name.replace("_max.pkl", "") for p in CACHE_DIR.glob("*.pkl")}
    ever = set(m["ticker"])
    still_in = set(m.loc[m["end_date"].isna(), "ticker"])
    exited_missing = sorted((ever - cache) - still_in)
    return [_yahoo_symbol(t) for t in exited_missing]


def ingest(dry_run: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = candidates()
    found = skipped_short = errors = written = 0
    by_source = {"arandkei": 0, "eodhd": 0}
    written_list: list[str] = []

    for t in tickers:
        src = _find_source(t)
        if src is None:
            continue
        found += 1
        kind, path = src
        try:
            df = _load_arandkei(path) if kind == "arandkei" else _load_eodhd(path)
        except Exception as exc:
            print(f"  [!] {t}: failed to load {path.name}: {exc}")
            errors += 1
            continue
        if df is None or len(df) < MIN_BARS:
            skipped_short += 1
            continue
        by_source[kind] += 1
        written += 1
        written_list.append(t)
        if not dry_run:
            out = CACHE_DIR / f"{t}_max.pkl"
            with open(out, "wb") as f:
                pickle.dump(df, f)

    print(f"Candidates (exited & missing ex-S&P names): {len(tickers)}")
    print(f"Found in delisted archives:                  {found}")
    print(f"  -> Arandkei: {by_source['arandkei']}   EODHD/Boris: {by_source['eodhd']}")
    print(f"Skipped (< {MIN_BARS} usable bars):           {skipped_short}")
    print(f"Load errors:                                  {errors}")
    print(f"{'Would write' if dry_run else 'Wrote'} cache files:               {written}")
    return {"candidates": len(tickers), "found": found, "written": written,
            "tickers": written_list}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest delisted-name OHLCV into the price cache")
    p.add_argument("--dry-run", action="store_true", help="Report only, don't write cache files")
    args = p.parse_args(argv)
    ingest(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
