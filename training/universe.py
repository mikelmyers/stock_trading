"""Training universe: S&P 500 + ETFs + watchlist extras."""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from config import (
    BASE_DIR,
    SECTOR_ETFS,
    SP500_CSV_URL,
    TRAINING_UNIVERSE_DIR,
    WATCHLIST,
)

EXTRAS_FILE = TRAINING_UNIVERSE_DIR / "extras.txt"
SP500_FILE = TRAINING_UNIVERSE_DIR / "sp500.txt"


def yahoo_symbol(symbol: str) -> str:
    """Normalize exchange symbols for Yahoo Finance (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def _read_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(yahoo_symbol(line.split(",")[0]))
    return tickers


def fetch_sp500_symbols(refresh: bool = False) -> list[str]:
    """Download current S&P 500 constituents and cache locally."""
    TRAINING_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    if SP500_FILE.exists() and not refresh:
        return _read_ticker_file(SP500_FILE)

    req = urllib.request.Request(
        SP500_CSV_URL,
        headers={"User-Agent": "trading-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    symbols = sorted({
        yahoo_symbol(row["Symbol"])
        for row in reader
        if row.get("Symbol")
    })

    SP500_FILE.write_text(
        "# S&P 500 constituents (auto-fetched from datasets/s-and-p-500-companies)\n"
        + "\n".join(symbols)
        + "\n",
        encoding="utf-8",
    )
    return symbols


def default_extras() -> list[str]:
    """Broad-market ETFs, vol benchmarks, and swing names outside the S&P 500."""
    extras = [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "MDY", "IJH", "IJR",
        "XLG", "RSP", "VUG", "VTV", "ARKK",
        "^VIX", "^GSPC",
        "PLTR", "CELH", "ELF", "DUOL", "VKTX", "PATH", "SOUN", "BBAI",
        "RIG", "HIMS", "IONQ", "RKLB", "SOFI", "MARA", "RIOT", "SMCI",
        "ARM", "SNAP", "SHOP", "SNOW", "ZS", "PANW", "MDB", "TTD", "RBLX",
        "U", "DKNG", "HOOD", "RIVN", "LCID", "NIO", "XPEV", "LI", "MRNA",
        "BNTX", "SQ", "ROKU", "ASTS", "IREN", "CLSK", "WULF",
    ]
    extras.extend(SECTOR_ETFS.values())
    for tickers in WATCHLIST.values():
        extras.extend(tickers)
    return [yahoo_symbol(t) for t in extras]


def ensure_extras_file() -> None:
    TRAINING_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRAS_FILE.exists():
        return
    EXTRAS_FILE.write_text(
        "# Extra training tickers: ETFs, vol indices, retail swing names\n"
        + "\n".join(default_extras())
        + "\n",
        encoding="utf-8",
    )


def load_training_universe(refresh_sp500: bool = False) -> list[str]:
    """
    Full training universe: S&P 500 + extras file + watchlist merge.

    Deduplicated and sorted. Typically 520-560 symbols.
    """
    ensure_extras_file()
    sp500 = fetch_sp500_symbols(refresh=refresh_sp500)
    extras = _read_ticker_file(EXTRAS_FILE) or default_extras()

    merged = sorted(set(sp500) | set(extras))
    return merged


def universe_stats(tickers: list[str] | None = None) -> dict:
    tickers = tickers or load_training_universe()
    sp500 = set(_read_ticker_file(SP500_FILE))
    return {
        "total": len(tickers),
        "sp500_cached": len(sp500),
        "extras_file": str(EXTRAS_FILE),
        "sp500_file": str(SP500_FILE),
    }