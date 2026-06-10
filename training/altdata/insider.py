"""SEC insider transactions (Form 3/4/5 structured data) — cluster-buy signal.

Source (free): https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
Quarterly zips; each contains SUBMISSION.tsv and NONDERIV_TRANS.tsv. Unzip into
data/insider/<anything>/ and run:

    python -m training.altdata.insider --build

POINT-IN-TIME RULE: the knowable date is FILING_DATE (the Form 4 hits EDGAR),
not TRANS_DATE (when the insider actually traded, up to 2 business days
earlier). We join on filing_date + 1 day so a same-day backtest can't act on
a filing that landed after the close.

Signal: open-market purchases (code P) net of sales (code S), in dollars, and
the count of distinct buying insiders — clustered buying by several insiders
is the variant of this anomaly with academic support.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW_DIR = DATA_DIR / "insider"
OUT = Path(__file__).resolve().parents[1] / "ml" / "datasets" / "insider.pkl"
FILING_LAG_DAYS = 1              # filed intraday/after close -> usable next day
WINDOW_DAYS = 90                 # trailing window for the features

FEATURES = ["ins_netbuy_90d", "ins_buyers_90d", "ins_selldays_90d"]


def normalize(submissions: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    """SUBMISSION + NONDERIV_TRANS -> tidy per (ticker, public_date) daily
    aggregate [net_buy_usd, buyers, sell_events]."""
    sub = submissions.rename(columns=str.upper)
    tr = trans.rename(columns=str.upper)
    need_s = {"ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"}
    need_t = {"ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"}
    if missing := (need_s - set(sub.columns)):
        raise ValueError(f"SUBMISSION.tsv missing {sorted(missing)}")
    if missing := (need_t - set(tr.columns)):
        raise ValueError(f"NONDERIV_TRANS.tsv missing {sorted(missing)}")

    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])].copy()
    tr["usd"] = (pd.to_numeric(tr["TRANS_SHARES"], errors="coerce")
                 * pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce"))
    tr = tr.dropna(subset=["usd"])
    tr["signed_usd"] = np.where(tr["TRANS_CODE"] == "P", tr["usd"], -tr["usd"])

    m = tr.merge(sub[["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"]],
                 on="ACCESSION_NUMBER", how="inner")
    m["ticker"] = m["ISSUERTRADINGSYMBOL"].astype(str).str.upper().str.strip()
    m = m[m["ticker"].ne("") & m["ticker"].ne("NAN") & m["ticker"].ne("NONE")]
    m["filing_date"] = pd.to_datetime(m["FILING_DATE"], format="mixed")
    m["public_date"] = (m["filing_date"]
                        + pd.Timedelta(days=FILING_LAG_DAYS)).astype("datetime64[ns]")

    g = m.groupby(["ticker", "public_date"])
    out = pd.DataFrame({
        "net_buy_usd": g["signed_usd"].sum(),
        "buyers": g.apply(
            lambda x: x.loc[x["TRANS_CODE"] == "P", "ACCESSION_NUMBER"].nunique(),
            include_groups=False),
        "sell_events": g.apply(
            lambda x: int((x["TRANS_CODE"] == "S").sum()), include_groups=False),
    }).reset_index()
    return out.sort_values(["ticker", "public_date"]).reset_index(drop=True)


def build_from_dir(raw_dir: Path | None = None, out: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    sub_files = sorted(raw_dir.rglob("SUBMISSION.tsv"))
    if not sub_files:
        raise SystemExit(
            f"no SUBMISSION.tsv under {raw_dir} — unzip the quarterly sets from\n"
            "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets")
    frames = []
    for sf in sub_files:
        tf = sf.parent / "NONDERIV_TRANS.tsv"
        if not tf.exists():
            print(f"  [!] {sf.parent.name}: no NONDERIV_TRANS.tsv — skipped")
            continue
        frames.append(normalize(pd.read_csv(sf, sep="\t", low_memory=False),
                                pd.read_csv(tf, sep="\t", low_memory=False)))
        print(f"  {sf.parent.name}: {len(frames[-1]):,} ticker-days")
    ins = (pd.concat(frames, ignore_index=True)
           .groupby(["ticker", "public_date"], as_index=False).sum())
    out = Path(out) if out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    ins.to_pickle(out)
    print(f"  {len(ins):,} ticker-days, {ins['ticker'].nunique():,} tickers -> {out}")
    return ins


def attach_insider(ev: pd.DataFrame, ins: pd.DataFrame,
                   window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Trailing-window insider features per setup row, point-in-time on
    public_date. Adds: ins_netbuy_90d (signed $), ins_buyers_90d (distinct
    buying filings — the cluster signal), ins_selldays_90d."""
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"]).astype("datetime64[ns]")
    parts = []
    for col, src in [("ins_netbuy_90d", "net_buy_usd"),
                     ("ins_buyers_90d", "buyers"),
                     ("ins_selldays_90d", "sell_events")]:
        s = (ins.set_index("public_date")
                .groupby("ticker")[src]
                .apply(lambda x: x.sort_index().rolling(f"{window_days}D").sum())
                .rename(col))
        parts.append(s.reset_index())
    feat = parts[0]
    for p in parts[1:]:
        feat = feat.merge(p, on=["ticker", "public_date"], how="outer")
    feat["public_date"] = feat["public_date"].astype("datetime64[ns]")
    feat = feat.sort_values("public_date")
    joined = pd.merge_asof(
        ev.sort_values("date"), feat,
        left_on="date", right_on="public_date", by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=window_days))   # stale beyond window -> NaN
    for c in FEATURES:
        if c in ("ins_netbuy_90d",):
            continue
        joined[c] = joined[c].fillna(0.0)
    joined["ins_netbuy_90d"] = joined["ins_netbuy_90d"].fillna(0.0)
    return joined


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Normalize SEC insider-transaction sets")
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
