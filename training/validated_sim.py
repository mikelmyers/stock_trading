"""Validated-system simulator: model-selected, regime-sized realized book.

Unlike portfolio_sim.py (which selects by the hand setup_score on the old
survivorship-biased data), this:
  * uses the survivorship-free dataset + the regime-aware ML model to RANK and
    select the book (top conviction by model P(win));
  * excludes structurally-dead setups (bear_breakdown);
  * sizes each trade by REGIME (size down / step aside when VIX is elevated and
    the tape is below its 200-day) -- the validated way to handle down years.

Development is on 2019-2023; 2024+ is reserved untouched for the one-shot test.
Run:  python -m training.validated_sim                 # dev (2019-2023)
      python -m training.validated_sim --clean-test    # spend the 2024+ bullet
"""
from __future__ import annotations

import argparse
import collections
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from training.ml.features import FEATURE_COLUMNS

DATASET = Path(__file__).resolve().parent / "ml" / "datasets" / "survivorship_free_v2.parquet"
DEAD_SETUPS = {"bear_breakdown"}          # Layer-2 audit: -0.22R, no edge any regime
COST_R = 0.019                            # Layer-1 honest all-in cost


def _fit_model(train: pd.DataFrame):
    import lightgbm as lgb
    fc = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                           n_jobs=-1, verbosity=-1)
    m.fit(train[fc].to_numpy("float64"), train["y_win"].to_numpy("int64"))
    return m, fc


def _regime_mult(row) -> float:
    """Risk multiplier in [~0.3, 1.0]: step aside in dangerous regimes."""
    danger = 0
    if row.mkt_above_sma200 < 0.5:   # tape below its 200-day
        danger += 1
    if row.vix_z_252 > 1.0:          # VIX stretched vs its own year
        danger += 1
    elif row.vix_z_252 > 0.3:
        danger += 0.5
    return {0: 1.0, 0.5: 0.8, 1.0: 0.6, 1.5: 0.45, 2.0: 0.35}.get(danger, 0.35)


def select_and_size(df: pd.DataFrame, k: int, per_type: int, p_floor: float,
                    regime_size: bool):
    """Each day, fill K free slots with highest-P signals (max per_type/type,
    only if P>=floor). Record entry/close and a per-trade risk multiplier."""
    open_heap: list = []
    rows = []
    for day, grp in df.groupby("date"):
        while open_heap and open_heap[0][0] <= day:
            heapq.heappop(open_heap)
        tcount = collections.Counter(t for _, t in open_heap)
        for r in grp.sort_values("p", ascending=False).itertuples():
            if len(open_heap) >= k:
                break
            if r.p < p_floor or tcount[r.setup_type] >= per_type:
                continue
            heapq.heappush(open_heap, (r.close, r.setup_type))
            tcount[r.setup_type] += 1
            mult = _regime_mult(r) if regime_size else 1.0
            rows.append((r.date, r.close, r.y_r - COST_R, mult))
    return pd.DataFrame(rows, columns=["date", "close", "R", "mult"])


def equity_curve(book: pd.DataFrame, seed: float, base_frac: float):
    """Compound with concurrency; risk base_frac*mult of equity per trade."""
    openh: list = []
    equity = seed
    curve = [(book["date"].min(), seed)]
    for t in book.sort_values("date").itertuples():
        while openh and openh[0][0] <= t.date:
            _, rd, r = heapq.heappop(openh)
            equity += rd * r
            curve.append((t.date, equity))
        if equity <= 0:
            equity = 0.0
            break
        rd = base_frac * t.mult * equity
        heapq.heappush(openh, (t.close, rd, t.R))
    while openh:
        d, rd, r = heapq.heappop(openh)
        equity += rd * r
        curve.append((d, equity))
    cur = pd.DataFrame(curve, columns=["date", "eq"]).groupby("date")["eq"].last()
    eq = cur.values
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    yrs = (book["date"].max() - book["date"].min()).days / 365.25
    cagr = (equity / seed) ** (1 / yrs) - 1 if equity > 0 and yrs > 0 else -1.0
    # daily-ish Sharpe from the equity steps
    rets = pd.Series(eq).pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    return {"final": equity, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe,
            "yrs": yrs, "trades_yr": len(book) / yrs}


def walk_forward_blind(seed: float, k: int, per_type: int, base_frac: float,
                       p_floor_q: float, start_year: int = 2019):
    """True blind money test: for each year, train ONLY on prior years, set the
    entry threshold from the TRAINING distribution (no peeking), trade that year
    blind, and compound one continuous account. Mirrors real deployment
    (retrain annually, trade forward)."""
    df = pd.read_parquet(DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["setup_type"].isin(DEAD_SETUPS)].copy()
    df["setup_code"] = df["setup_type"].astype("category").cat.codes
    df["close"] = df["date"] + pd.to_timedelta(df["days_held"], "D")
    last_year = df["date"].dt.year.max()

    parts = []
    print("=" * 70)
    print("  VALIDATED SYSTEM -- WALK-FORWARD BLIND MONEY TEST")
    print("=" * 70)
    for Y in range(start_year, last_year + 1):
        train = df[df["date"] < f"{Y}-01-01"]
        if len(train) < 50_000:
            continue
        model, fc = _fit_model(train)
        # blind threshold: 90th pct of model P on the TRAINING set (known in advance)
        p_tr = model.predict_proba(train[fc].to_numpy("float64"))[:, 1]
        floor = float(np.quantile(p_tr, p_floor_q))
        yr = df[(df["date"] >= f"{Y}-01-01") & (df["date"] < f"{Y+1}-01-01")].copy()
        if yr.empty:
            continue
        yr["p"] = model.predict_proba(yr[fc].to_numpy("float64"))[:, 1]
        yr = yr[yr["p"] >= floor]
        parts.append(yr)
        print(f"  {Y}: trained on {len(train):,} prior rows  |  blind P-floor={floor:.3f}  "
              f"|  {len(yr):,} signals passed")

    pool = pd.concat(parts).sort_values("date").reset_index(drop=True)
    for label, regime in [("flat sizing", False), ("regime-scaled sizing", True)]:
        book = select_and_size(pool, k, per_type, -1.0, regime)  # floor already applied
        m = equity_curve(book, seed, base_frac)
        peryr = book.assign(yr=book["date"].dt.year).groupby("yr")["R"].mean()
        print(f"\n  [{label}]  {len(book):,} trades ({m['trades_yr']:.0f}/yr), "
              f"mean R {book['R'].mean():+.4f}")
        print(f"    CAGR {m['cagr']*100:+.1f}%  | maxDD {m['max_dd']*100:.0f}%  | "
              f"Sharpe {m['sharpe']:.2f}  | final ${m['final']:,.0f} on ${seed:,.0f}")
        print("    per-year mean R: " + "  ".join(
            f"{y}:{r:+.3f}" for y, r in peryr.items()))


def run(clean_test: bool, seed: float, k: int, per_type: int, base_frac: float,
        p_floor_q: float):
    df = pd.read_parquet(DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["setup_type"].isin(DEAD_SETUPS)].copy()
    df["setup_code"] = df["setup_type"].astype("category").cat.codes

    train = df[df["date"] < "2019-01-01"]
    model, fc = _fit_model(train)

    eval_lo, eval_hi = ("2024-01-01", "2099-01-01") if clean_test else ("2019-01-01", "2024-01-01")
    ev = df[(df["date"] >= eval_lo) & (df["date"] < eval_hi)].copy()
    ev["p"] = model.predict_proba(ev[fc].to_numpy("float64"))[:, 1]
    ev["close"] = ev["date"] + pd.to_timedelta(ev["days_held"], "D")
    p_floor = ev["p"].quantile(p_floor_q)

    tag = "CLEAN TEST (2024+, one-shot)" if clean_test else "DEV (2019-2023)"
    print("=" * 70)
    print(f"  VALIDATED SYSTEM -- {tag}")
    print("=" * 70)
    print(f"  model-selected book, exclude {DEAD_SETUPS}, P-floor q={p_floor_q} "
          f"(P>={p_floor:.3f}), K={k} max {per_type}/type, cost {COST_R}R")
    for label, regime in [("flat sizing", False), ("regime-scaled sizing", True)]:
        book = select_and_size(ev, k, per_type, p_floor, regime)
        m = equity_curve(book, seed, base_frac)
        peryr = book.assign(yr=book["date"].dt.year).groupby("yr")["R"].mean()
        y22 = f"  2022={peryr.get(2022):.3f}" if 2022 in peryr.index else ""
        print(f"\n  [{label}]  {len(book):,} trades ({m['trades_yr']:.0f}/yr), "
              f"mean R {book['R'].mean():+.4f}")
        print(f"    CAGR {m['cagr']*100:+.1f}%  | maxDD {m['max_dd']*100:.0f}%  | "
              f"Sharpe {m['sharpe']:.2f}  | final ${m['final']:,.0f} on ${seed:,.0f}")
        print("    per-year mean R: " + "  ".join(
            f"{y}:{r:+.3f}" for y, r in peryr.items()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clean-test", action="store_true", help="evaluate 2024+ (one-shot)")
    p.add_argument("--seed", type=float, default=10000.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--per-type", type=int, default=2)
    p.add_argument("--base-frac", type=float, default=0.01, help="base risk/trade")
    p.add_argument("--p-floor-q", type=float, default=0.90, help="only trade top (1-q) by model P")
    p.add_argument("--walk-forward", action="store_true",
                   help="blind money test: retrain annually, trade forward, compound one account")
    a = p.parse_args(argv)
    if a.walk_forward:
        walk_forward_blind(a.seed, a.k, a.per_type, a.base_frac, a.p_floor_q)
    else:
        run(a.clean_test, a.seed, a.k, a.per_type, a.base_frac, a.p_floor_q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
