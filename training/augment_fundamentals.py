"""Does adding point-in-time fundamentals as FEATURES sharpen the timing model?

Point-in-time joins the SEC-XBRL fundamentals onto the event-driven setup dataset
(by ticker, latest snapshot <= setup date) + price (for value factors), then runs
the SAME 2019+ holdout head-to-head: baseline 33 features vs baseline + fundamental
factors. We restrict to 2013+ (fundamentals era) so it's apples-to-apples.

    SNAP=/tmp/snapshots.csv.gz python -m training.augment_fundamentals
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from training.history import download_history
from training.ml.features import FEATURE_COLUMNS
from training.universe import load_training_universe

SNAP = os.environ.get("SNAP", "/tmp/snapshots.csv.gz")
DATASET = "training/ml/datasets/survivorship_free_v2.parquet"
FUND_FACTORS = ["ep", "bp", "sp", "cfp", "gp_assets", "roe", "margin", "lev",
                "accruals", "rev_growth", "asset_growth", "f_size"]


def build():
    ev = pd.read_parquet(DATASET)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["date"] >= "2013-06-01"].copy()           # fundamentals era only
    ev["setup_code"] = ev["setup_type"].astype("category").cat.codes

    f = pd.read_csv(SNAP, parse_dates=["snapshot"]).dropna(subset=["ticker"])
    keep = ["ticker", "snapshot", "Earnings_ttm", "Equity", "Revenue_ttm",
            "NetCashOperating_ttm", "Assets", "GrossProfit_ttm", "Liabilities", "Shares"]
    f = f[keep].sort_values(["ticker", "snapshot"])
    f["rev_growth"] = f.groupby("ticker")["Revenue_ttm"].pct_change(12)
    f["asset_growth"] = f.groupby("ticker")["Assets"].pct_change(12)

    # price at setup date (for value factors)
    hist = download_history(load_training_universe())
    px = pd.concat({tk: d["Close"] for tk, d in hist.items()
                    if tk not in ("^GSPC", "^VIX")}, names=["ticker", "date"]).reset_index()
    px.columns = ["ticker", "date", "close"]

    ev = pd.merge_asof(ev.sort_values("date"), f.sort_values("snapshot"),
                       left_on="date", right_on="snapshot", by="ticker", direction="backward")
    ev = pd.merge_asof(ev.sort_values("date"), px.sort_values("date"),
                       on="date", by="ticker", direction="backward")

    mcap = ev["close"] * ev["Shares"]
    ev["ep"] = ev["Earnings_ttm"] / mcap
    ev["bp"] = ev["Equity"] / mcap
    ev["sp"] = ev["Revenue_ttm"] / mcap
    ev["cfp"] = ev["NetCashOperating_ttm"] / mcap
    ev["gp_assets"] = ev["GrossProfit_ttm"] / ev["Assets"]
    ev["roe"] = ev["Earnings_ttm"] / ev["Equity"]
    ev["margin"] = ev["GrossProfit_ttm"] / ev["Revenue_ttm"]
    ev["lev"] = ev["Liabilities"] / ev["Assets"]
    ev["accruals"] = (ev["Earnings_ttm"] - ev["NetCashOperating_ttm"]) / ev["Assets"]
    ev["f_size"] = np.log(mcap.where(mcap > 0))
    ev = ev.replace([np.inf, -np.inf], np.nan)
    cov = ev["ep"].notna().mean()
    print(f"  joined: {len(ev):,} setups (2013+), fundamental coverage {cov*100:.0f}%")
    return ev


def holdout_edge(ev, feats, label):
    import lightgbm as lgb
    tr = ev[ev["date"] < "2019-01-01"]
    te = ev[ev["date"] >= "2019-01-01"].copy()
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                           n_jobs=-1, verbosity=-1)
    m.fit(tr[feats].to_numpy("float64"), tr["y_win"].to_numpy("int64"))
    te["p"] = m.predict_proba(te[feats].to_numpy("float64"))[:, 1]
    cut = te["p"].quantile(0.90)
    book = te[te["p"] >= cut]
    net = book["y_r"].mean() - 0.019
    auc = _auc(te["y_win"].to_numpy(), te["p"].to_numpy())
    print(f"  [{label:<22}] top-10% net R {net:+.4f}  win {(book['y_r']>0).mean()*100:.1f}%  "
          f"OOS-AUC {auc:.4f}  ({len(book):,} trades)")
    if "ep" in feats[-1] or label.startswith("+"):
        imp = pd.Series(m.feature_importances_, index=feats)
        fund_imp = imp[[c for c in FUND_FACTORS if c in feats]].sort_values(ascending=False)
        print(f"     fundamental importance: " + ", ".join(f"{k}={v}" for k, v in fund_imp.head(5).items()))
    return net


def _auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (np.sum(np.where(y == 1, np.arange(1, len(y) + 1), 0)) - n1 * (n1 + 1) / 2) / (n0 * n1)


def run():
    ev = build()
    base = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    aug = base + FUND_FACTORS
    print("=" * 70)
    print("  DOES ADDING FUNDAMENTALS SHARPEN THE TIMING MODEL?  (2019+ holdout)")
    print("=" * 70)
    b = holdout_edge(ev, base, "baseline (33 feats)")
    a = holdout_edge(ev, aug, "+ fundamentals")
    print("\n  " + "-" * 50)
    delta = a - b
    print(f"  delta from fundamentals: {delta:+.4f}R  "
          f"({'HELPS' if delta > 0.005 else 'NEUTRAL' if delta > -0.005 else 'HURTS'})")
    print(f"  (baseline-on-this-subset {b:+.4f}; full-history baseline was +0.0765)")


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
