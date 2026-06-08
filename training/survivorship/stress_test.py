"""Rung 1 — FREE survivorship-bias stress test.

We cannot fully *correct* the bias without delisted price data (that's Rung 2),
but we can *bound* it for $0:

  1. Coverage audit: how much of historical S&P 500 membership our data misses.
  2. Sensitivity: inject "ghost" trades for the missing (mostly exited) names at
     assumed outcomes, proportional to the membership-time we're missing, and see
     how the measured edge holds up — plus the break-even assumption that erases it.

Membership: fja05680/sp500 (MIT) sp500_ticker_start_end.csv.
Setups/labels: the ML dataset built from this run's checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MEMBERSHIP_CSV = BASE / "sp500_ticker_start_end.csv"
DATASET = BASE.parent / "ml" / "datasets" / "full.parquet"

# Reliability floor (pre-2001 membership is unreliable) and our data's last bar.
START_FLOOR = pd.Timestamp("2001-01-01")
DATA_END = pd.Timestamp("2026-06-05")
YEAR = 365.25


def _overlap_years(start, end) -> float:
    s = max(start, START_FLOOR)
    e = min(end if pd.notna(end) else DATA_END, DATA_END)
    return max(0.0, (e - s).days / YEAR)


def main() -> None:
    m = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["start_date", "end_date"])
    cache = {p.name.replace("_max.pkl", "") for p in (BASE.parent / "cache" / "tickers").glob("*.pkl")}

    ever = set(m["ticker"])
    still_in = set(m.loc[m["end_date"].isna(), "ticker"])
    exited = set(m.loc[m["end_date"].notna(), "ticker"]) - still_in
    missing = ever - cache

    # Member-years (2001+) covered by our data vs missing.
    covered_my = missing_my = 0.0
    intervals: dict[str, list[tuple]] = {}
    for _, r in m.iterrows():
        yrs = _overlap_years(r["start_date"], r["end_date"])
        intervals.setdefault(r["ticker"], []).append((r["start_date"], r["end_date"]))
        if r["ticker"] in cache:
            covered_my += yrs
        else:
            missing_my += yrs
    miss_frac = missing_my / (covered_my + missing_my)

    print("=" * 66)
    print("  SURVIVORSHIP COVERAGE AUDIT (S&P 500, 2001-2026)")
    print("=" * 66)
    print(f"  Unique ever-members:        {len(ever):>6}")
    print(f"  Currently in index:         {len(still_in):>6}")
    print(f"  Exited (left for good):     {len(exited):>6}")
    print(f"  Ever-members in our data:   {len(ever & cache):>6}")
    print(f"  Ever-members MISSING:       {len(missing):>6}  "
          f"({len(missing & exited)} of them are exited names)")
    print(f"  Member-years covered:       {covered_my:>8.0f}")
    print(f"  Member-years MISSING:       {missing_my:>8.0f}")
    print(f"  --> Missing fraction of S&P member-time: {miss_frac:.1%}")

    # --- Setups restricted to genuine S&P-membership windows --------------
    df = pd.read_parquet(DATASET, columns=["ticker", "date", "y_r", "y_win", "setup_type"])
    df["date"] = pd.to_datetime(df["date"])

    def in_membership(row) -> bool:
        for s, e in intervals.get(row.ticker, ()):
            if s <= row.date <= (e if pd.notna(e) else DATA_END):
                return True
        return False

    df["is_member"] = [in_membership(r) for r in df.itertuples(index=False)]
    cov = df[df["is_member"] & df["ticker"].isin(cache)]
    n_cov = len(cov)
    e_cov = float(cov["y_r"].mean())
    e_all = float(df["y_r"].mean())

    setup_rate = n_cov / covered_my
    ghost = setup_rate * missing_my

    print()
    print("=" * 66)
    print("  EDGE & GHOST-TRADE VOLUME")
    print("=" * 66)
    print(f"  Measured edge, ALL our setups:        {e_all:+.4f} R  (n={len(df):,})")
    print(f"  Measured edge, S&P-member setups only:{e_cov:+.4f} R  (n={n_cov:,})")
    print(f"  Setup rate (member setups / yr):      {setup_rate:,.0f}")
    print(f"  Implied GHOST trades on missing names:{ghost:,.0f}")

    # --- Sensitivity: blend ghosts at assumed mean outcome ---------------
    print()
    print("=" * 66)
    print("  SENSITIVITY: blended edge if missing-name trades averaged e_ghost")
    print("=" * 66)
    print(f"  {'e_ghost (R)':>12} | {'blended edge (R)':>16} | survives?")
    print("  " + "-" * 48)
    for e_ghost in (0.0, -0.25, -0.50, -1.00, -2.00):
        blended = (n_cov * e_cov + ghost * e_ghost) / (n_cov + ghost)
        print(f"  {e_ghost:>12.2f} | {blended:>16.4f} | {'yes' if blended > 0 else 'NO'}")

    breakeven = -(n_cov * e_cov) / ghost
    print("  " + "-" * 48)
    print(f"  Break-even: the {ghost:,.0f} missing-name trades would each have to")
    print(f"  average worse than {breakeven:+.3f} R to erase the +{e_cov:.3f} R member edge.")
    print()
    # Translate to a failure-tail interpretation (non-failures assumed neutral 0R)
    print("  Failure-tail view (non-failure ghosts = 0R):")
    print(f"  {'fail frac':>10} | {'fail @ -1R':>12} | {'fail @ -2R':>12} | {'fail @ -3R':>12}")
    print("  " + "-" * 52)
    for f in (0.10, 0.25, 0.40):
        cells = []
        for fr in (-1.0, -2.0, -3.0):
            e_ghost = f * fr  # rest at 0
            blended = (n_cov * e_cov + ghost * e_ghost) / (n_cov + ghost)
            cells.append(f"{blended:+.4f}")
        print(f"  {f:>10.0%} | {cells[0]:>12} | {cells[1]:>12} | {cells[2]:>12}")
    print("=" * 66)


if __name__ == "__main__":
    main()
