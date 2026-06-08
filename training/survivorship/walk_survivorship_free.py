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
import hashlib

from training.backtester import TrainingProfile, CHECKPOINT_CHUNK_SIZE
from training.history import download_history
from training.resumable_train import _gather_all, _walk_with_checkpoints
from training.universe import load_training_universe

PROFILE = TrainingProfile(
    name="surv_free",
    walk_step=1,
    max_setups_per_ticker=None,
    early_stop_real_setups=None,
    chunk_size=CHECKPOINT_CHUNK_SIZE,
    slippage_levels=[0.0],
)


def _in_shard(ticker: str, shard: int, num_shards: int) -> bool:
    """Deterministic, stable assignment of a ticker to one of ``num_shards``
    buckets via a content hash. Lets independent containers each take a
    disjoint slice of the universe and merge their per-ticker checkpoint
    pickles via git with no filename collisions."""
    h = int(hashlib.sha256(ticker.encode()).hexdigest(), 16)
    return h % num_shards == shard


def run(workers: int = 4, shard: int = 0, num_shards: int = 1,
        limit: int | None = None) -> None:
    print("=" * 64)
    print("  SURVIVORSHIP-FREE CHECKPOINT WALK (slippage=[0.0], walk_step=1)")
    print("=" * 64)
    tickers = load_training_universe()
    print(f"Universe: {len(tickers)} tickers (incl. delisted ex-S&P names)")
    history = download_history(tickers)
    print(f"  Loaded {len(history)} tickers with usable price history.")

    if num_shards > 1:
        history = {t: df for t, df in history.items()
                   if _in_shard(t, shard, num_shards)}
        print(f"  Shard {shard}/{num_shards}: {len(history)} tickers in this slice.")

    if limit is not None:
        history = dict(list(history.items())[:limit])
        print(f"  --limit {limit}: capped to {len(history)} tickers (smoke test).")

    _walk_with_checkpoints(history, PROFILE, workers, PROFILE.chunk_size)

    base_results = _gather_all(history)
    print("=" * 64)
    print(f"  DONE — {len(base_results):,} real (slippage=0) simulations checkpointed")
    print(f"  across {len(history)} tickers.")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Survivorship-free checkpoint walk")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", default="0/1",
                   help="Process only this shard, formatted 'k/N' (0 <= k < N). "
                        "Run k=0..N-1 in separate containers to parallelize the "
                        "universe; each writes a disjoint set of per-ticker "
                        "checkpoints that merge cleanly via git.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap to the first N tickers of this shard (smoke test).")
    args = p.parse_args(argv)
    shard_str, _, num_str = args.shard.partition("/")
    shard, num_shards = int(shard_str), int(num_str or "1")
    if not (0 <= shard < num_shards):
        p.error(f"--shard k/N requires 0 <= k < N (got {args.shard})")
    run(workers=args.workers, shard=shard, num_shards=num_shards, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
