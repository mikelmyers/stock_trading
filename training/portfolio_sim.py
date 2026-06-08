"""Bankroll / position-sizing / risk-of-ruin simulator (Rung A).

Answers the only question that matters for "can this pay for itself?":
given the trades we could *realistically take* (capacity-limited, diversified),
honest costs, and a survivorship haircut — does the account grow, how hard does
it draw down, and what's the chance of ruin?

Crucial subtlety this version fixes: the backtest pool has ~1M overlapping
signals you can never all trade. What matters is the **realized book** — the
~150-200 trades/yr you can actually hold with K slots. We select that book the
way a real trader would (each day, take the highest-conviction signals, capped
per setup type for diversification) and evaluate *that*.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET = BASE / "ml" / "datasets" / "full.parquet"
LEARNED = BASE.parent / "learned_params.json"
START = pd.Timestamp("2001-01-01")


def load_stream(cost_r: float) -> pd.DataFrame:
    lp = json.loads(LEARNED.read_text())
    enabled, min_score = set(lp["enabled_setups"]), lp["min_setup_score"]
    df = pd.read_parquet(DATASET, columns=["date", "setup_type", "setup_score", "y_r", "days_held"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= START) & df["setup_type"].isin(enabled) & (df["setup_score"] >= min_score)]
    df = df.sort_values("date").reset_index(drop=True)
    df["R_cost"] = df["y_r"] - cost_r
    df["close"] = df["date"] + pd.to_timedelta(df["days_held"], "D")
    return df


def select_book(df: pd.DataFrame, k: int, per_type: int) -> pd.DataFrame:
    """Realistic capacity-limited selection: each day fill free slots with the
    highest-score signals, max `per_type` per setup type (diversification)."""
    import heapq
    open_heap: list = []          # (close_date, setup_type)
    rows = []
    for day, grp in df.groupby("date"):
        while open_heap and open_heap[0][0] <= day:
            heapq.heappop(open_heap)
        tcount = collections.Counter(t for _, t in open_heap)
        for r in grp.sort_values("setup_score", ascending=False).itertuples():
            if len(open_heap) >= k:
                break
            if tcount[r.setup_type] >= per_type:
                continue
            heapq.heappush(open_heap, (r.close, r.setup_type))
            tcount[r.setup_type] += 1
            rows.append((r.date, r.close, r.R_cost))
    return pd.DataFrame(rows, columns=["date", "close", "R"])


def event_equity(book: pd.DataFrame, seed: float, risk_frac: float, edge_shift: float):
    """Compound the realized book with concurrency: size risk_frac of realized
    equity at entry, realize P&L at close."""
    ev = []
    for t in book.itertuples():
        ev.append((t.date, "open", t.R + edge_shift))
        ev.append((t.close, "close", None))
    ev.sort(key=lambda x: (x[0], x[1] == "open"))  # closes before opens same day
    equity = seed
    pending: list = []
    pidx = 0
    curve = [seed]
    # process opens in order, closes via a queue keyed by close date
    import heapq
    openh: list = []  # (close_date, risk_dollars, R)
    for t in book.sort_values("date").itertuples():
        while openh and openh[0][0] <= t.date:
            _, rd, r = heapq.heappop(openh)
            equity += rd * r
            curve.append(equity)
        if equity <= 0:
            equity = 0.0
            break
        rd = risk_frac * equity
        heapq.heappush(openh, (t.close, rd, t.R + edge_shift))
    while openh:
        _, rd, r = heapq.heappop(openh)
        equity += rd * r
        curve.append(equity)
    eq = np.array(curve)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    yrs = (book["date"].max() - book["date"].min()).days / 365.25
    cagr = (equity / seed) ** (1 / yrs) - 1 if equity > 0 and yrs > 0 else -1.0
    return {"final": equity, "cagr": cagr, "max_dd": max_dd, "yrs": yrs,
            "trades_yr": len(book) / yrs}


def monte_carlo(R: np.ndarray, seed: float, risk_frac: float, n_trades: int,
                paths: int = 4000, ruin_dd: float = 0.5):
    rng = np.random.default_rng(7)
    draws = rng.choice(R, size=(paths, n_trades), replace=True)
    eq = seed * np.cumprod(np.clip(1 + risk_frac * draws, 1e-6, None), axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    worst_dd = ((peak - eq) / peak).max(axis=1)
    final = eq[:, -1]
    return {"p10": float(np.percentile(final, 10)), "p50": float(np.percentile(final, 50)),
            "p90": float(np.percentile(final, 90)), "ruin": float(np.mean(worst_dd >= ruin_dd))}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=float, default=1000.0)
    p.add_argument("--k", type=int, default=5, help="max concurrent positions")
    p.add_argument("--per-type", type=int, default=2, help="max concurrent per setup type")
    p.add_argument("--cost-r", type=float, default=0.015,
                   help="per-trade cost in R (~0.13%% round-trip on 8.75%% stop)")
    args = p.parse_args(argv)

    gross_pool = load_stream(0.0)["y_r"].values
    df = load_stream(args.cost_r)
    book = select_book(df, args.k, args.per_type)
    R = book["R"].values
    yrs = (book["date"].max() - book["date"].min()).days / 365.25

    print("=" * 76)
    print("  REALIZED-BOOK BANKROLL / RISK-OF-RUIN  (calibrated rule, 2001+)")
    print("=" * 76)
    print(f"  Signal pool: {len(df):,} (mean R {load_stream(args.cost_r)['R_cost'].mean():+.4f}) "
          f"-- but you can't trade them all.")
    print(f"  REALIZED BOOK (K={args.k}, max {args.per_type}/type): {len(book):,} trades "
          f"({len(book)/yrs:.0f}/yr)")
    print(f"  Realized mean R: {R.mean():+.4f}  win {np.mean(R>0)*100:.1f}%  "
          f"<-- this is what actually hits your account")
    print(f"  Pool gross mean R: {gross_pool.mean():+.4f}  "
          f"(the pool edge does NOT survive capacity + selection)")

    edges = {"realized (optimistic)": 0.0,
             "+ survivorship -0.03R": -0.03,
             "+ survivorship -0.06R": -0.06}
    sizings = {"0.5%/trade": 0.005, "1%/trade": 0.01, "2%/trade": 0.02}
    n_yr = int(max(len(book) / yrs, 1))

    for ename, esh in edges.items():
        print("\n" + "-" * 76)
        print(f"  EDGE: {ename}   (realized mean R = {R.mean()+esh:+.4f})")
        print(f"  {'risk/trade':<11}|{'CAGR':>8}|{'final $':>11}|{'maxDD':>7}|"
              f"{'P(>=50%DD)':>11}|{'median $/mo':>12}")
        print("  " + "-" * 72)
        for sname, rf in sizings.items():
            sim = event_equity(book, args.seed, rf, esh)
            mc = monte_carlo(R + esh, args.seed, rf, n_yr)
            monthly = (mc["p50"] - args.seed) / 12.0
            print(f"  {sname:<11}|{sim['cagr']*100:>7.1f}%|{sim['final']:>11,.0f}|"
                  f"{sim['max_dd']*100:>6.0f}%|{mc['ruin']*100:>10.0f}%|{monthly:>12,.2f}")

    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    viable = R.mean() > 0
    print(f"  Realized book is {'POSITIVE' if viable else 'NEGATIVE'} even before survivorship/haircut: "
          f"{R.mean():+.4f}R/trade.")
    if not viable:
        print("  => Not self-funding at any seed or size. The backtest's positive pool")
        print("     edge was an artifact of un-tradeable overlapping signals + survivorship.")
        print("     Position sizing cannot rescue a negative realized edge; it only changes")
        print("     how fast you lose. This is the $0 lesson Rung A exists to deliver.")
    print("=" * 76)


if __name__ == "__main__":
    raise SystemExit(main())
