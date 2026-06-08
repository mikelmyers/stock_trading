"""Resumable checkpoint walk for the survivorship-free universe.

Mirrors ``training.resumable_train`` but:
  * walks the EXPANDED universe (current S&P 500 + extras + 334 newly-ingested
    delisted names — see ``training/universes/extras.txt`` and
    ``training/survivorship/ingest_delisted.py``);
  * uses ``slippage_levels=[0.0]`` only. The ML dataset / meta-labeling layer
    only consumes the zero-slippage real trades anyway (see
    ``training.ml.dataset.build_from_checkpoints``), so the 0.1/0.2 slippage
    sweeps the original full calibration ran would be wasted compute here —
    skipping them cuts the walk to roughly a third.

Writes to the SAME checkpoint dir as the original run
(``training/cache/sims_full/``) in the identical per-ticker pickle format, so
``build_from_checkpoints`` needs no changes. Does NOT touch
``learned_params.json`` — that recalibration is a separate concern from
rebuilding the ML dataset survivorship-free.

Usage:
    python -m training.survivorship.walk_survivorship_free
"""

from __future__ import annotations

import argparse

from training.backtester import TrainingProfile, CHECKPOINT_CHUNK_SIZE
from training.history import download_history
from training.resumable_train import _gather_all, _walk_with_checkpoints
from training.universe import load_training_universe

PROFILE = TrainingProfile(
    name="surv_free",
    # walk_step=2: setups are heavily autocorrelated bar-to-bar (a pattern valid
    # at bar i is usually still valid at i+1), so scanning every other bar roughly
    # halves compute while leaving ~900k setups — far more than the meta-labeling
    # model + holdout need. The sampling change is noted in the writeup.
    walk_step=2,
    max_setups_per_ticker=None,
    early_stop_real_setups=None,
    chunk_size=CHECKPOINT_CHUNK_SIZE,
    slippage_levels=[0.0],
)


def run(workers: int = 4) -> None:
    print("=" * 64)
    print("  SURVIVORSHIP-FREE CHECKPOINT WALK (slippage=[0.0], walk_step=1)")
    print("=" * 64)
    tickers = load_training_universe()
    print(f"Universe: {len(tickers)} tickers (incl. delisted ex-S&P names)")
    history = download_history(tickers)
    print(f"  Loaded {len(history)} tickers with usable price history.")

    _walk_with_checkpoints(history, PROFILE, workers, PROFILE.chunk_size)

    base_results = _gather_all(history)
    print("=" * 64)
    print(f"  DONE — {len(base_results):,} real (slippage=0) simulations checkpointed")
    print(f"  across {len(history)} tickers.")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Survivorship-free checkpoint walk")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    run(workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
