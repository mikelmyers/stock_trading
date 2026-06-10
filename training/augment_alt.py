"""Do short-interest / insider features sharpen the timing model?

Same protocol that correctly killed the fundamentals idea: join the candidate
features point-in-time onto the event dataset, run the 2019+ holdout
head-to-head (baseline vs +features), then the per-year walk-forward delta as
the tiebreaker. The verdict is logged to the experiment registry either way —
failures count as trials too.

    # after building the inputs locally:
    #   python -m training.altdata.short_interest --build
    #   python -m training.altdata.insider --build
    python -m training.augment_alt --source si             # holdout head-to-head
    python -m training.augment_alt --source si --wf        # per-year tiebreaker
    python -m training.augment_alt --source insider --wf
    python -m training.augment_alt --source both --wf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training import experiments
from training.altdata import insider as ins_mod
from training.altdata import short_interest as si_mod
from training.augment_fundamentals import _net
from training.ml.features import FEATURE_COLUMNS

DATASET = "training/ml/datasets/survivorship_free_v2.parquet"
COST_R = 0.019

SOURCES = {
    "si": (si_mod.OUT, si_mod.FEATURES),
    "insider": (ins_mod.OUT, ins_mod.FEATURES),
}


def load_events(dataset: str | Path = DATASET) -> pd.DataFrame:
    ev = pd.read_parquet(dataset) if str(dataset).endswith(".parquet") \
        else pd.read_pickle(dataset)
    ev["date"] = pd.to_datetime(ev["date"])
    ev["setup_code"] = ev["setup_type"].astype("category").cat.codes
    return ev


def attach(ev: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    feats: list[str] = []
    if source in ("si", "both"):
        path, cols = SOURCES["si"]
        if not Path(path).exists():
            raise SystemExit(f"{path} missing — run: python -m training.altdata.short_interest --build")
        ev = si_mod.attach_short_interest(ev, pd.read_pickle(path))
        feats += cols
    if source in ("insider", "both"):
        path, cols = SOURCES["insider"]
        if not Path(path).exists():
            raise SystemExit(f"{path} missing — run: python -m training.altdata.insider --build")
        ev = ins_mod.attach_insider(ev, pd.read_pickle(path))
        feats += cols
    cov = ev[feats[0]].notna().mean() if feats else 0.0
    print(f"  joined {source}: {len(ev):,} setups, coverage {cov*100:.0f}% on {feats[0]}")
    return ev, feats


def holdout(ev: pd.DataFrame, extra: list[str], label: str) -> tuple[float, float]:
    base = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    tr = ev[ev["date"] < "2019-01-01"]
    te = ev[ev["date"] >= "2019-01-01"]
    b = _net(base, tr, te)
    a = _net(base + extra, tr, te)
    print(f"  [baseline          ] top-10% net R {b:+.4f}")
    print(f"  [+ {label:<16}] top-10% net R {a:+.4f}   delta {a-b:+.4f}")
    return b, a


def walk_forward(ev: pd.DataFrame, extra: list[str]) -> tuple[float, int, int]:
    base = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    aug = base + extra
    ev = ev.copy()
    ev["yr"] = ev["date"].dt.year
    print(f"  {'year':<6}{'baseline':>10}{'+feats':>10}{'delta':>10}{'trades':>9}")
    print("  " + "-" * 45)
    deltas = []
    for Y in range(2019, int(ev["yr"].max()) + 1):
        tr = ev[ev["yr"] < Y]
        te = ev[ev["yr"] == Y]
        if len(tr) < 50_000 or len(te) < 5_000:
            continue
        b = _net(base, tr, te)
        a = _net(aug, tr, te)
        deltas.append(a - b)
        print(f"  {Y:<6}{b:>+10.4f}{a:>+10.4f}{a-b:>+10.4f}{len(te):>9,}")
    md = float(np.mean(deltas))
    pos = sum(d > 0 for d in deltas)
    print("  " + "-" * 45)
    print(f"  mean delta {md:+.4f}R  |  positive in {pos}/{len(deltas)} years")
    return md, pos, len(deltas)


def verdict_of(mean_delta: float, pos: int, n: int) -> str:
    if mean_delta > 0.003 and pos >= n * 0.7:
        return "adopt"
    if abs(mean_delta) < 0.003 or pos <= n * 0.5:
        return "reject"
    return "inconclusive"


def run(source: str, wf: bool, dataset: str = DATASET) -> None:
    ev = load_events(dataset)
    ev, feats = attach(ev, source)
    print("=" * 64)
    if wf:
        print(f"  WALK-FORWARD: baseline vs +{source} (top-10% net R, per year)")
        print("=" * 64)
        md, pos, n = walk_forward(ev, feats)
        v = verdict_of(md, pos, n)
        print(f"  VERDICT: {v.upper()}")
        experiments.log_experiment(
            name=f"altdata {source} as features (walk-forward)",
            config={"source": source, "features": feats, "protocol": "per-year WF delta"},
            dataset=str(dataset),
            metric=f"mean_delta={md:+.4f}R positive {pos}/{n} years",
            baseline="baseline 35-feat timing model",
            verdict=v,
            notes="auto-logged by augment_alt")
    else:
        print(f"  HOLDOUT 2019+: baseline vs +{source}")
        print("=" * 64)
        b, a = holdout(ev, feats, source)
        print("  (single holdout is suggestive only — run --wf for the verdict)")
        experiments.log_experiment(
            name=f"altdata {source} as features (single holdout)",
            config={"source": source, "features": feats, "protocol": "2019+ holdout"},
            dataset=str(dataset),
            metric=f"top10_net_r={a:+.4f}", baseline=f"top10_net_r={b:+.4f}",
            verdict="running",
            notes="single holdout — walk-forward is the tiebreaker")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Alt-data feature experiments")
    p.add_argument("--source", choices=["si", "insider", "both"], required=True)
    p.add_argument("--wf", action="store_true", help="per-year walk-forward (the verdict)")
    p.add_argument("--dataset", default=DATASET)
    a = p.parse_args(argv)
    run(a.source, a.wf, a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
