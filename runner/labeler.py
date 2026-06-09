"""Auto-labeler — turn logged candidates into training labels.

For each logged candidate not yet labeled (and old enough that its window is
complete), pull the post-signal bars and compute max-favorable-excursion: did the
stock run? That MFE is the clean, management-independent `y` the classifier learns
from (it labels the SETUP, not how a trade was managed). Idempotent — safe to run
repeatedly (e.g., end of each day). Runs locally (needs Alpaca bars).

    python -m runner.labeler
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from runner.logger import CANDIDATES, OUTCOMES, RunnerLog

HORIZON_MIN = 120          # measure MFE over the 2h after the signal
MIN_AGE_MIN = 130          # only label candidates whose window has fully elapsed
MONSTER_MFE, GOOD_MFE, SCRATCH_MFE = 20.0, 5.0, 3.0


def bucket(mfe: float) -> str:
    if mfe >= MONSTER_MFE:
        return "monster"
    if mfe >= GOOD_MFE:
        return "good"
    if mfe >= SCRATCH_MFE:
        return "scratch"
    return "loss"


def compute_mfe(entry_price, bars) -> float | None:
    """Max favorable excursion (%) — the highest high after the signal vs entry."""
    if not bars or not entry_price:
        return None
    hi = max(float(b["high"]) for b in bars)
    return (hi - float(entry_price)) / float(entry_price) * 100.0


def _alpaca_bars(symbol, start_iso, end_iso):
    import os
    import requests
    h = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
         "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    j = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars", headers=h,
                     params={"timeframe": "1Min", "start": start_iso, "end": end_iso,
                             "limit": 400}, timeout=20).json()
    return [dict(high=b["h"], low=b["l"], close=b["c"]) for b in j.get("bars", [])]


def label_pending(bars_provider, candidates_path=CANDIDATES, outcomes_path=OUTCOMES,
                  horizon_min=HORIZON_MIN, min_age_min=MIN_AGE_MIN, now=None) -> dict:
    if not Path(candidates_path).exists():
        return {"labeled": 0, "reason": "no candidates logged yet"}
    cand = pd.read_csv(candidates_path).drop_duplicates(["symbol", "asof"])
    done = set()
    if Path(outcomes_path).exists():
        o = pd.read_csv(outcomes_path)
        done = set(zip(o["symbol"], o["asof"].astype(str)))
    now = now or dt.datetime.now(dt.timezone.utc)
    labeled = skipped_young = 0
    for r in cand.itertuples():
        if (r.symbol, str(r.asof)) in done:
            continue
        try:
            sig = pd.to_datetime(r.asof, utc=True)
        except Exception:
            continue
        if (now - sig.to_pydatetime()).total_seconds() / 60 < min_age_min:
            skipped_young += 1
            continue
        bars = bars_provider(r.symbol, sig.isoformat(),
                             (sig + pd.Timedelta(minutes=horizon_min)).isoformat())
        mfe = compute_mfe(getattr(r, "price", None), bars)
        if mfe is None:
            continue
        RunnerLog.log_outcome(r.symbol, str(r.asof), round(mfe, 2),
                              bucket=bucket(mfe), path=outcomes_path)
        labeled += 1
    return {"labeled": labeled, "skipped_too_young": skipped_young}


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    print(label_pending(_alpaca_bars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
