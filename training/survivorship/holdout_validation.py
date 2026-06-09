"""Survivorship-free holdout validation: train through 2018, trade 2019+ untouched.

This is the clean forward test the meta-labeling result was missing — no walk-
forward folds peeking near the boundary, no re-tried selection schemes. One
split, decided in advance: everything before 2019-01-01 trains the model;
everything from 2019-01-01 onward is scored and "traded" exactly once.

Pipeline:
  1. Load the survivorship-free dataset (built from the expanded — survivors +
     delisted — universe) and restrict to point-in-time S&P member setups
     (``training/survivorship/sp500_ticker_start_end.csv`` intervals), so the
     "book" reflects the index as it actually existed, not as it looks today.
  2. Purge the training set of any row whose forward-return window
     (days_held, capped at MAX_HOLDING_DAYS) could leak into the holdout.
  3. Fit once on pre-2019, score 2019+ out-of-sample.
  4. Build the same capacity-limited, selective top-K% books as
     ``training.ml.metalabel_test`` and report realized R, net of cost, and
     under survivorship-haircut scenarios.

CLI
---
    python -m training.survivorship.holdout_validation \
        --dataset training/ml/datasets/survivorship_free.parquet
"""

from __future__ import annotations

import argparse
import collections
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from training.backtester import MAX_HOLDING_DAYS
from training.ml.features import FEATURE_COLUMNS
from training.ml.model import _make_classifier
from training.simutil import trading_close_dates, trading_to_calendar_days

SURV_DIR = Path(__file__).resolve().parent
MEMBERSHIP_CSV = SURV_DIR / "sp500_ticker_start_end.csv"
HOLDOUT_START = pd.Timestamp("2019-01-01")
EMBARGO_DAYS = 3
K, PER = 5, 2  # capacity book: max concurrent positions, max per setup type


def _membership_intervals() -> dict[str, list[tuple]]:
    m = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["start_date", "end_date"])
    intervals: dict[str, list[tuple]] = {}
    for _, r in m.iterrows():
        intervals.setdefault(r["ticker"], []).append((r["start_date"], r["end_date"]))
    return intervals


def _filter_to_members(df: pd.DataFrame) -> pd.DataFrame:
    intervals = _membership_intervals()
    data_end = pd.Timestamp("2026-06-05")

    def in_membership(row) -> bool:
        for s, e in intervals.get(row.ticker, ()):
            if s <= row.date <= (e if pd.notna(e) else data_end):
                return True
        return False

    mask = [in_membership(r) for r in df.itertuples(index=False)]
    return df[mask].reset_index(drop=True)


def _book(df: pd.DataFrame, score_col: str = "pwin") -> np.ndarray:
    """Capacity-limited book: at most K concurrent, max PER per setup type,
    greedily filled in score order each day."""
    df = df.copy()
    df["close"] = trading_close_dates(df["date"], df["days_held"])
    df["date"] = pd.to_datetime(df["date"])
    openh: list = []
    taken = []
    for day, grp in df.groupby("date"):
        while openh and openh[0][0] <= day:
            heapq.heappop(openh)
        tc = collections.Counter(t for _, t in openh)
        for r in grp.sort_values(score_col, ascending=False).itertuples():
            if len(openh) >= K:
                break
            if tc[r.setup_type] >= PER:
                continue
            heapq.heappush(openh, (r.close, r.setup_type))
            tc[r.setup_type] += 1
            taken.append(r.y_r)
    return np.array(taken)


def run(dataset_path: str | Path, members_only: bool = True) -> dict:
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} setups from {dataset_path}")

    if members_only:
        df = _filter_to_members(df)
        print(f"Point-in-time S&P-member setups: {len(df):,}")

    df = df.sort_values("date").reset_index(drop=True)
    df["setup_code"] = df["setup_type"].astype("category").cat.codes
    feat_cols = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    X = df[feat_cols].to_numpy("float64")
    y = df["y_win"].to_numpy("int64")

    # Purge: drop training rows whose forward window could overlap the holdout.
    # MAX_HOLDING_DAYS is TRADING days; the purge gap must cover it in calendar days.
    purge_cutoff = HOLDOUT_START - pd.Timedelta(
        days=trading_to_calendar_days(MAX_HOLDING_DAYS) + EMBARGO_DAYS)
    train_mask = (df["date"] < purge_cutoff).to_numpy()
    test_mask = (df["date"] >= HOLDOUT_START).to_numpy()
    print(f"\nTrain: {train_mask.sum():,} rows (< {purge_cutoff.date()}, "
          f"purged {EMBARGO_DAYS}d embargo + {MAX_HOLDING_DAYS}d label horizon before holdout)")
    print(f"Holdout: {test_mask.sum():,} rows (>= {HOLDOUT_START.date()}, untouched until scored)")

    fit, predict_proba, backend = _make_classifier()
    print(f"\nBackend: {backend}")
    model = fit(X[train_mask], y[train_mask])
    pwin = predict_proba(model, X[test_mask])

    oos = df[test_mask].copy()
    oos["pwin"] = pwin
    yrs = (oos["date"].max() - oos["date"].min()).days / 365.25

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(oos["y_win"], oos["pwin"]) if oos["y_win"].nunique() > 1 else float("nan")
    print(f"\nHoldout (2019+): {len(oos):,} setups over {yrs:.1f} yrs  |  AUC={auc:.4f}")
    print(f"  Base (untouched) holdout edge: mean R {oos['y_r'].mean():+.4f}  "
          f"win {oos['y_win'].mean()*100:.1f}%")

    print("\nRealized mean R by model-P(win) decile (holdout, cost-free):")
    oos["bucket"] = pd.qcut(oos["pwin"], 10, labels=False, duplicates="drop")
    g = oos.groupby("bucket").agg(meanR=("y_r", "mean"), win=("y_win", "mean"), n=("y_r", "size"))
    for b, row in g.iterrows():
        print(f"  decile {int(b)+1:>2}: meanR {row.meanR:+.4f}  win {row.win*100:4.1f}%  n={int(row.n):,}")

    print(f"\nSelective top-K%% books (holdout, K={K} concurrent, max {PER}/type):")
    print(f"  {'cutoff':>10} | {'trades/yr':>9} | {'mean R':>8} | {'net -0.015':>10} | "
          f"{'-0.03 surv':>10} | {'-0.06 surv':>10} | {'win%':>6}")
    print("  " + "-" * 78)
    results = {}
    for q, lbl in [(0.90, "top 10%"), (0.95, "top 5%"), (0.99, "top 1%")]:
        thr = oos["pwin"].quantile(q)
        sub = oos[oos["pwin"] >= thr]
        taken = _book(sub)
        if len(taken) == 0:
            print(f"  {lbl:>10} |       n/a — no trades survived capacity limits")
            continue
        m = float(taken.mean())
        win = float(np.mean(taken > 0))
        results[lbl] = {"n": len(taken), "trades_per_yr": len(taken) / yrs, "mean_r": m,
                        "net_cost": m - 0.015, "surv_03": m - 0.03, "surv_06": m - 0.06, "win": win}
        print(f"  {lbl:>10} | {len(taken)/yrs:>9.0f} | {m:>+8.4f} | {m-0.015:>+10.4f} | "
              f"{m-0.03:>+10.4f} | {m-0.06:>+10.4f} | {win*100:>5.1f}%")

    print("\nVERDICT (top-10% book, net of cost):")
    if "top 10%" in results and results["top 10%"]["net_cost"] > 0:
        print(f"  Survives the clean train-through-2018/trade-2019+ holdout: "
              f"+{results['top 10%']['net_cost']:.4f}R net "
              f"({results['top 10%']['trades_per_yr']:.0f} trades/yr).")
    else:
        print("  Does NOT survive the clean holdout — the OOS-fold edge from "
              "metalabel_test.py was likely an artifact of trying multiple "
              "selection schemes / folds.")

    return {"auc": float(auc), "n_holdout": len(oos), "yrs": yrs, "books": results}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Survivorship-free holdout validation (2019+)")
    p.add_argument("--dataset", default="training/ml/datasets/survivorship_free.parquet")
    p.add_argument("--all-setups", action="store_true",
                   help="Skip point-in-time S&P membership filter (use every setup)")
    args = p.parse_args(argv)
    run(args.dataset, members_only=not args.all_setups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
