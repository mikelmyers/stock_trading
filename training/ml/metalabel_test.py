"""Meta-labeling test: can the ML model SELECT a tradeable (positive) subset?

True meta-labeling: the rules generate candidates; the model decides which to
take. We score every setup OUT-OF-SAMPLE (purged walk-forward — train on past,
predict future, never peek), then ask:

  1. Do the highest-P(win) trades actually have positive realized R? (top decile/5%)
  2. If we build the capacity-limited book by model P(win) instead of hand score,
     is the realized edge positive?

This is the last free card before concluding the chart-pattern signal family is a
dead end on large-cap equities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.ml.features import FEATURE_COLUMNS
from training.ml.model import _make_classifier
from training.ml.splits import purged_walkforward_splits

DATASET = "training/ml/datasets/full.parquet"


def main():
    df = pd.read_parquet(DATASET)
    df = df.sort_values("date").reset_index(drop=True)
    df["setup_code"] = df["setup_type"].astype("category").cat.codes
    feat = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    X = df[feat].to_numpy("float64")
    y = df["y_win"].to_numpy("int64")

    fit, predict_proba, backend = _make_classifier()
    print(f"Backend: {backend} | rows {len(df):,}")
    splits = purged_walkforward_splits(df["date"], n_splits=5)

    pwin = np.full(len(df), np.nan)
    for i, (tr, te) in enumerate(splits, 1):
        m = fit(X[tr], y[tr])
        pwin[te] = predict_proba(m, X[te])
        print(f"  fold {i}: trained {len(tr):,} -> scored {len(te):,}")

    df["pwin"] = pwin
    oos = df[df["pwin"].notna()].copy()
    print(f"\nOOS-scored trades: {len(oos):,}")

    # 1) Does ranking by model P(win) sort on realized R? (the core question)
    print("\n  Realized mean R by model-P(win) bucket (cost-free):")
    oos["bucket"] = pd.qcut(oos["pwin"], 10, labels=False, duplicates="drop")
    g = oos.groupby("bucket").agg(meanR=("y_r", "mean"), win=("y_win", "mean"),
                                  n=("y_r", "size"))
    for b, row in g.iterrows():
        print(f"    decile {int(b)+1:>2}: meanR {row.meanR:+.4f}  win {row.win*100:4.1f}%  n={int(row.n):,}")
    top5 = oos[oos["pwin"] >= oos["pwin"].quantile(0.95)]
    top1 = oos[oos["pwin"] >= oos["pwin"].quantile(0.99)]
    print(f"  Top 5%% P(win): meanR {top5['y_r'].mean():+.4f} (n={len(top5):,})")
    print(f"  Top 1%% P(win): meanR {top1['y_r'].mean():+.4f} (n={len(top1):,})")

    # 2) Capacity-limited book selected by P(win) instead of hand score
    import collections, heapq
    oos["close"] = pd.to_datetime(oos["date"]) + pd.to_timedelta(oos["days_held"], "D")
    oos["date"] = pd.to_datetime(oos["date"])
    K, PER = 5, 2
    openh: list = []
    taken = []
    for day, grp in oos.groupby("date"):
        while openh and openh[0][0] <= day:
            heapq.heappop(openh)
        tc = collections.Counter(t for _, t in openh)
        for r in grp.sort_values("pwin", ascending=False).itertuples():
            if len(openh) >= K:
                break
            if tc[r.setup_type] >= PER:
                continue
            heapq.heappush(openh, (r.close, r.setup_type))
            tc[r.setup_type] += 1
            taken.append(r.y_r)
    taken = np.array(taken)
    print(f"\n  Realized book selected by MODEL P(win) (K={K}, max {PER}/type):")
    print(f"    n={len(taken):,}  mean R {taken.mean():+.4f}  win {np.mean(taken>0)*100:.1f}%")
    print(f"    after -0.015 cost: {taken.mean()-0.015:+.4f} | after -0.06 surv: {taken.mean()-0.06:+.4f}")

    print("\n  VERDICT:", "model finds a positive tradeable subset"
          if taken.mean() - 0.015 > 0 else
          "even model-selected book is NEGATIVE -> signal family is a dead end here")


if __name__ == "__main__":
    main()
