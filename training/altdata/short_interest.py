"""FINRA consolidated equity short interest — normalized, point-in-time safe.

Source (free):
  * Files UI:  https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files
  * API:       https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest
               (free credentials: https://developer.finra.org)
Download the biweekly CSVs into data/short_interest/ (any filenames), then:

    python -m training.altdata.short_interest --build   # -> tidy parquet/pkl

POINT-IN-TIME RULE: positions settle on settlement_date but FINRA publishes
~8 business days later. A backtest may only see a record from its PUBLICATION
date onward — joining on settlement_date leaks ~2 weeks of future knowledge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW_DIR = DATA_DIR / "short_interest"
OUT = Path(__file__).resolve().parents[1] / "ml" / "datasets" / "short_interest.pkl"
PUBLICATION_LAG_BDAYS = 9        # conservative: settlement -> public dissemination

# header aliases across FINRA file vintages
COLUMN_ALIASES = {
    "symbol": ["symbolCode", "symbol", "Symbol", "issueSymbolIdentifier"],
    "settlement_date": ["settlementDate", "settlement_date", "Settlement Date"],
    "short_interest": ["currentShortPositionQuantity", "shortInterest",
                       "Current Short Position"],
    "prev_short_interest": ["previousShortPositionQuantity", "Previous Short Position"],
    "adv": ["averageDailyVolumeQuantity", "Average Daily Volume"],
    "days_to_cover": ["daysToCoverQuantity", "Days to Cover"],
}

FEATURES = ["si_dtc", "si_chg", "si_log"]


def _pick(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Any FINRA short-interest file vintage -> tidy
    [symbol, settlement_date, public_date, short_interest, adv, days_to_cover, si_chg]."""
    out = pd.DataFrame()
    for canon, aliases in COLUMN_ALIASES.items():
        col = _pick(raw, aliases)
        if col is not None:
            out[canon] = col
    missing = {"symbol", "settlement_date", "short_interest"} - set(out.columns)
    if missing:
        raise ValueError(f"unrecognized short-interest file: missing {sorted(missing)}; "
                         f"columns were {list(raw.columns)[:12]}")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["settlement_date"] = pd.to_datetime(out["settlement_date"])
    out["short_interest"] = pd.to_numeric(out["short_interest"], errors="coerce")
    # the date the market could first KNOW this number
    out["public_date"] = pd.DatetimeIndex(np.busday_offset(
        out["settlement_date"].values.astype("datetime64[D]"),
        PUBLICATION_LAG_BDAYS, roll="forward")).astype("datetime64[ns]")
    if "days_to_cover" in out.columns:
        out["days_to_cover"] = pd.to_numeric(out["days_to_cover"], errors="coerce")
    elif "adv" in out.columns:
        adv = pd.to_numeric(out["adv"], errors="coerce")
        out["days_to_cover"] = out["short_interest"] / adv.replace(0, np.nan)
    if "prev_short_interest" in out.columns:
        prev = pd.to_numeric(out["prev_short_interest"], errors="coerce")
        out["si_chg"] = (out["short_interest"] - prev) / prev.replace(0, np.nan)
    out = out.dropna(subset=["symbol", "settlement_date", "short_interest"])
    return out.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)


def build_from_dir(raw_dir: Path | None = None, out: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.txt")))
    if not files:
        raise SystemExit(
            f"no files in {raw_dir} — download the biweekly CSVs from\n"
            "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files")
    frames = []
    for f in files:
        sep = "|" if f.suffix == ".txt" else ","
        frames.append(normalize(pd.read_csv(f, sep=sep)))
    si = pd.concat(frames, ignore_index=True)
    si = si.drop_duplicates(["symbol", "settlement_date"], keep="last")
    si = si.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)
    if "si_chg" not in si.columns or si["si_chg"].isna().all():
        si["si_chg"] = si.groupby("symbol")["short_interest"].pct_change()
    out = Path(out) if out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    si.to_pickle(out)
    print(f"  {len(si):,} short-interest records, {si['symbol'].nunique():,} symbols "
          f"({si['settlement_date'].min().date()} .. {si['settlement_date'].max().date()}) -> {out}")
    return si


def attach_short_interest(ev: pd.DataFrame, si: pd.DataFrame) -> pd.DataFrame:
    """Join the latest PUBLISHED record per ticker onto each setup row.
    Adds: si_dtc (days to cover), si_chg (period-over-period change),
    si_log (log short interest)."""
    si = si.copy()
    si["si_dtc"] = si.get("days_to_cover")
    si["si_log"] = np.log1p(si["short_interest"])
    if "si_chg" not in si.columns:
        si["si_chg"] = si.groupby("symbol")["short_interest"].pct_change()
    right = (si[["symbol", "public_date", "si_dtc", "si_chg", "si_log"]]
             .rename(columns={"symbol": "ticker"})
             .sort_values("public_date"))
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"]).astype("datetime64[ns]")
    joined = pd.merge_asof(
        ev.sort_values("date"), right,
        left_on="date", right_on="public_date", by="ticker",
        direction="backward")
    return joined


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Normalize FINRA short-interest files")
    p.add_argument("--build", action="store_true")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.build:
        build_from_dir(a.raw_dir, a.out)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
