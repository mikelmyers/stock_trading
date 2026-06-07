"""Historical OHLCV download with per-ticker cache, retries, and resume."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    TRAINING_BATCH_DELAY_SEC,
    TRAINING_BATCH_SIZE,
    TRAINING_CACHE_TTL_HOURS,
    TRAINING_MIN_BARS,
    TRAINING_USE_MAX_HISTORY,
    TRAINING_YEARS,
)
from training.universe import yahoo_symbol

TRAINING_DIR = Path(__file__).resolve().parent
CACHE_DIR = TRAINING_DIR / "cache"
TICKER_CACHE_DIR = CACHE_DIR / "tickers"
MANIFEST_DIR = CACHE_DIR / "manifests"


def _period_for_years(years: int | str | None = None) -> str:
    if TRAINING_USE_MAX_HISTORY or years in ("max", 0, None):
        return "max"
    years = int(years)
    if years >= 10:
        return "max"
    return f"{years}y"


def _universe_hash(tickers: list[str], period: str) -> str:
    payload = period + "|" + ",".join(sorted(tickers))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if len(df.columns) == 0:
        return pd.DataFrame()
    return df


def _extract_ticker_df(raw: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    sym = yahoo_symbol(ticker)
    if batch_len == 1:
        return _normalize_df(raw)
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym in raw.columns.get_level_values(0):
                return _normalize_df(raw[sym])
            if ticker in raw.columns.get_level_values(0):
                return _normalize_df(raw[ticker])
        return _normalize_df(raw[sym])
    except (KeyError, TypeError):
        return pd.DataFrame()


def _ticker_cache_path(ticker: str, period: str) -> Path:
    return TICKER_CACHE_DIR / f"{yahoo_symbol(ticker)}_{period}.pkl"


def _cache_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < ttl_hours


def _save_ticker_cache(ticker: str, period: str, df: pd.DataFrame) -> None:
    TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ticker_cache_path(ticker, period), "wb") as f:
        pickle.dump(df, f)


def _load_ticker_cache(ticker: str, period: str) -> pd.DataFrame | None:
    path = _ticker_cache_path(ticker, period)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _download_batch(
    tickers: list[str],
    period: str,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    syms = [yahoo_symbol(t) for t in tickers]
    for attempt in range(retries):
        try:
            raw = yf.download(
                syms if len(syms) > 1 else syms[0],
                period=period,
                interval="1d",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            out: dict[str, pd.DataFrame] = {}
            for orig, sym in zip(tickers, syms):
                df = _extract_ticker_df(raw, sym, len(syms))
                if len(df) >= TRAINING_MIN_BARS:
                    out[orig] = df
            if out or attempt == retries - 1:
                return out
        except Exception:
            if attempt == retries - 1:
                return {}
        time.sleep(TRAINING_BATCH_DELAY_SEC * (attempt + 1))
    return {}


def _download_single(ticker: str, period: str, retries: int = 3) -> pd.DataFrame | None:
    sym = yahoo_symbol(ticker)
    for attempt in range(retries):
        try:
            raw = yf.download(
                sym, period=period, interval="1d",
                progress=False, auto_adjust=True, threads=False,
            )
            df = _normalize_df(raw)
            if len(df) >= TRAINING_MIN_BARS:
                return df
            if attempt == retries - 1:
                return None
        except Exception:
            if attempt == retries - 1:
                return None
        time.sleep(0.75 * (attempt + 1))
    return None


def download_history(
    tickers: list[str],
    years: int | str | None = None,
    refresh: bool = False,
    min_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for training.

    Uses per-ticker disk cache so interrupted runs resume without re-fetching.
    Default period is ``max`` (full Yahoo history, often 15-25 years).
    """
    min_bars = min_bars or TRAINING_MIN_BARS
    period = _period_for_years(years if years is not None else TRAINING_YEARS)
    ttl = 0 if refresh else TRAINING_CACHE_TTL_HOURS

    TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    uhash = _universe_hash(tickers, period)
    manifest_path = MANIFEST_DIR / f"{uhash}.json"
    bundle_path = CACHE_DIR / f"bundle_{uhash}.pkl"

    if not refresh and bundle_path.exists() and manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        age_hours = (time.time() - meta.get("ts", 0)) / 3600
        if age_hours < TRAINING_CACHE_TTL_HOURS:
            with open(bundle_path, "rb") as f:
                cached = pickle.load(f)
            if len(cached) >= meta.get("loaded", 0) * 0.9:
                print(f"  Using cached bundle ({len(cached)} tickers, period={period})")
                return cached

    data: dict[str, pd.DataFrame] = {}
    pending: list[str] = []

    for ticker in tickers:
        sym = yahoo_symbol(ticker)
        cache_path = _ticker_cache_path(sym, period)
        if not refresh and _cache_fresh(cache_path, ttl):
            df = _load_ticker_cache(sym, period)
            if df is not None and len(df) >= min_bars:
                data[ticker] = df
                continue
        pending.append(ticker)

    if pending:
        print(f"  Fetching {len(pending)} tickers (cached: {len(data)}, period={period})...")
    else:
        print(f"  All {len(data)} tickers loaded from per-ticker cache (period={period})")

    failed: list[str] = []
    total = len(pending)

    for i in range(0, total, TRAINING_BATCH_SIZE):
        batch = pending[i : i + TRAINING_BATCH_SIZE]
        batch_data = _download_batch(batch, period)

        for ticker in batch:
            df = batch_data.get(ticker)
            if df is None or len(df) < min_bars:
                df = _download_single(ticker, period)
            if df is not None and len(df) >= min_bars:
                data[ticker] = df
                _save_ticker_cache(ticker, period, df)
            else:
                failed.append(ticker)

        done = min(i + TRAINING_BATCH_SIZE, total)
        if done % 50 == 0 or done == total:
            print(f"    Downloaded {done}/{total} — {len(data)} usable tickers so far")
        if i + TRAINING_BATCH_SIZE < total:
            time.sleep(TRAINING_BATCH_DELAY_SEC)

    if failed:
        print(f"  [!] {len(failed)} tickers skipped (insufficient data): "
              f"{', '.join(failed[:8])}{'...' if len(failed) > 8 else ''}")

    if data:
        with open(bundle_path, "wb") as f:
            pickle.dump(data, f)
        manifest_path.write_text(json.dumps({
            "ts": time.time(),
            "period": period,
            "requested": len(tickers),
            "loaded": len(data),
            "failed": len(failed),
            "min_bars": min_bars,
        }))

    return data