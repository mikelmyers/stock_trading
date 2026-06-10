"""Resumable full training run with per-ticker disk checkpoints.

The standard ``run_training`` holds all results in memory, so if the process is
killed mid-run (e.g. an ephemeral container is suspended) everything is lost.
This driver does the identical Phase 1-2 work but writes each ticker's
simulations to disk the moment they finish. Re-running skips tickers that are
already checkpointed, so a kill costs at most the one in-flight ticker.

Phases 3-5 (bootstrap floor, options backtest, calibration, report) run once all
tickers are checkpointed, reusing the exact same functions as ``run_training`` —
so the calibrated output is identical to an uninterrupted run.

Usage:
    python -m training.resumable_train                 # full 10k-floor run
    python -m training.resumable_train --finalize-only # just (re)build params
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents.quant import summarize_backtest
from agents.setups.registry import SETUP_REGISTRY
from training.backtester import (
    RESULTS_DIR,
    _process_ticker,
    _profile_for_simulations,
    bootstrap_expand,
    summarize_results,
)
from training.calibrator import calibrate, save_learned_params
from training.history import download_history
from training.options_backtest import backtest_options_from_equity_results
from training.universe import load_training_universe

CKPT_DIR = Path(__file__).resolve().parent / "cache" / "sims_full"


def set_ckpt_dir(path: str | Path) -> Path:
    """Redirect checkpoints (e.g. the realism re-walk writes to sims_realism/
    so the historical sims_full labels stay reproducible)."""
    global CKPT_DIR
    CKPT_DIR = Path(path)
    return CKPT_DIR


def _ckpt_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace(".", "_")
    return CKPT_DIR / f"{safe}.pkl"


def _load_ckpt(ticker: str) -> list[dict] | None:
    path = _ckpt_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # corrupt/partial write -> recompute


def _save_ckpt(ticker: str, results: list[dict]) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _ckpt_path(ticker).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    tmp.replace(_ckpt_path(ticker))  # atomic: never leave a half-written .pkl


def _walk_with_checkpoints(history, profile, workers, chunk_size):
    items = list(history.items())
    rng = np.random.default_rng(42)
    rng.shuffle(items)

    pending = [(t, df) for t, df in items if _load_ckpt(t) is None]
    done = len(items) - len(pending)
    print(f"  Checkpoints: {done} done, {len(pending)} to compute "
          f"(dir: {CKPT_DIR})")

    t0 = time.time()
    completed = done
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        tasks = []
        for ticker, df in chunk:
            df_reset = df.reset_index()
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "date"})
            tasks.append((
                ticker, df_reset.to_dict(), profile.slippage_levels,
                profile.walk_step, profile.max_setups_per_ticker,
                profile.entry_fill, profile.gap_fills,
            ))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_ticker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    ticker, results, raw = fut.result()
                    _save_ckpt(ticker, results)
                    completed += 1
                    print(f"  [{completed}/{len(items)}] {ticker}: "
                          f"+{len(results):,} sims (raw {raw:,}) "
                          f"({time.time() - t0:.0f}s)", flush=True)
                except Exception as e:  # never lose the rest of the run to one ticker
                    _save_ckpt(name, [])
                    print(f"  [!] {name}: {e} (checkpointed empty)", flush=True)


def _gather_all(history) -> list[dict]:
    results: list[dict] = []
    for ticker in history:
        ck = _load_ckpt(ticker)
        if ck:
            results.extend(ck)
    return results


def finalize(history, base_results, simulations=10_000, profile_name="full",
             walk_step=1) -> dict:
    print(f"  Gathered {len(base_results):,} real simulations from checkpoints.")
    if len(base_results) < simulations:
        print(f"Phase 3: Bootstrapping to {simulations:,}...")
        all_results = bootstrap_expand(base_results, simulations)
    else:
        print(f"Phase 3: Real ({len(base_results):,}) >= target ({simulations:,}) "
              f"— keeping all real, no bootstrap.")
        all_results = base_results

    summary = summarize_results(all_results)
    summary["quant_metrics"] = summarize_backtest(all_results)
    summary["data_source"] = "yfinance_real_ohlcv"
    summary["history_period"] = "max"
    summary["training_profile"] = profile_name
    summary["walk_step"] = walk_step
    summary["tickers_scanned"] = len(history)
    summary["target_simulations"] = simulations

    print("Phase 4: Options spread backtest...")
    options_bt = backtest_options_from_equity_results(history, base_results)
    summary["options_backtest"] = options_bt
    print(f"  {options_bt.get('summary', 'no options results')}")

    print("Phase 5: Calibrating...")
    learned = calibrate(all_results)
    learned["options_backtest"] = options_bt
    save_learned_params(learned)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"training_{simulations}_{ts}.json"

    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_safe(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    result_path.write_text(json.dumps(_safe({
        "summary": summary,
        "learned_params": learned,
        "sample_trades": all_results[:50],
    }), indent=2), encoding="utf-8")
    summary["learned_params"] = learned
    summary["result_file"] = str(result_path)
    return summary


def run(simulations=10_000, workers=4, finalize_only=False) -> dict:
    profile = _profile_for_simulations(simulations)
    print("=" * 64)
    print(f"  RESUMABLE TRAINING [{profile.name}] — target {simulations:,}")
    print("=" * 64)
    tickers = load_training_universe()
    print(f"Loading history for {len(tickers)} tickers...")
    history = download_history(tickers)
    print(f"  Loaded {len(history)} tickers.")
    print(f"  Patterns: {', '.join(SETUP_REGISTRY.keys())}")

    if not finalize_only:
        print(f"Phase 1-2: Walking history [{profile.name}] "
              f"walk_step={profile.walk_step} cap={profile.max_setups_per_ticker} "
              f"slippage={profile.slippage_levels}")
        _walk_with_checkpoints(history, profile, workers, profile.chunk_size)

    base_results = _gather_all(history)
    summary = finalize(history, base_results, simulations=simulations,
                       profile_name=profile.name, walk_step=profile.walk_step)

    print("=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Tickers scanned:  {summary['tickers_scanned']}")
    print(f"  Total simulations:  {summary['count']:,}")
    print(f"  Real historical:  {summary.get('real_trades', 0):,}")
    print(f"  Bootstrapped:     {summary.get('bootstrapped', 0):,}")
    print(f"  Win rate:         {summary['win_rate']}%")
    print(f"  Expectancy:       {summary['expectancy']}R")
    print(f"  Saved params:     learned_params.json")
    print(f"  Full report:      {summary.get('result_file')}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resumable full training run")
    p.add_argument("--simulations", "-n", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--finalize-only", action="store_true",
                   help="Skip the walk; just (re)build params from checkpoints")
    args = p.parse_args(argv)
    run(simulations=args.simulations, workers=args.workers,
        finalize_only=args.finalize_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
