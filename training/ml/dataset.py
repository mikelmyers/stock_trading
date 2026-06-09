"""Build a supervised (features -> forward outcome) dataset from the price cache.

This reuses the *same* candidate generator and labeler that drive calibration:

  * ``find_historical_setups`` finds every bar where a rule-based pattern fired.
  * ``simulate_trade_forward`` plays the trade out and produces the label.

For each setup we emit one row: the causal feature vector at that bar, the setup
type/score, and the realized outcome (win flag + R multiple). The result is a
flat table ready for any tabular ML model.

CLI
---
    python -m training.ml.dataset --limit 50 --out training/ml/datasets/sample.parquet
    python -m training.ml.dataset                 # full universe
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from agents.indicators import calculate_atr
from training.backtester import find_historical_setups, simulate_trade_forward
from training.history import download_history
from training.ml.features import (
    FEATURE_COLUMNS,
    compute_feature_frame,
    compute_market_frame,
)
from training.universe import load_training_universe

OUT_DIR = Path(__file__).resolve().parent / "datasets"

META_COLUMNS = [
    "ticker", "date", "setup_type", "setup_score",
    "y_win", "y_r", "exit_reason", "days_held",
]


def build_for_ticker(ticker: str, df: pd.DataFrame, slippage: float = 0.0) -> list[dict]:
    """Emit one (features + label) row per rule-based setup found in ``df``."""
    feats = compute_feature_frame(df)
    atr14 = calculate_atr(df, 14)
    rows: list[dict] = []
    for idx, setup in find_historical_setups(df):
        sim = simulate_trade_forward(df, idx, setup, slippage_pct=slippage, atr14=atr14)
        if sim is None:
            continue
        row = feats.iloc[idx].to_dict()
        row.update({
            "ticker": ticker,
            "date": str(df.index[idx].date()),
            "setup_type": setup["setup_type"],
            "setup_score": setup["confidence_score"],
            "y_win": int(sim.won),
            "y_r": sim.pnl_r,
            "exit_reason": sim.exit_reason,
            "days_held": sim.days_held,
        })
        rows.append(row)
    return rows


def build_dataset(
    tickers: list[str] | None = None,
    limit: int | None = None,
    slippage: float = 0.0,
    out: str | Path | None = None,
) -> pd.DataFrame:
    tickers = tickers or load_training_universe()
    if limit:
        tickers = tickers[:limit]

    history = download_history(tickers)
    items = list(history.items())
    print(f"Building dataset from {len(items)} tickers...")

    all_rows: list[dict] = []
    t0 = time.time()
    for i, (ticker, df) in enumerate(items, 1):
        all_rows.extend(build_for_ticker(ticker, df, slippage=slippage))
        if i % 25 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {ticker}: {len(all_rows):,} rows "
                  f"({time.time() - t0:.0f}s)")

    frame = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS + META_COLUMNS)
    frame = frame.sort_values("date").reset_index(drop=True)

    out_path = Path(out) if out else OUT_DIR / "dataset.parquet"
    _save(frame, out_path)
    win = frame["y_win"].mean() if len(frame) else 0
    print(f"\nSaved {len(frame):,} rows x {frame.shape[1]} cols -> {out_path}")
    print(f"  Base win rate: {win:.1%}  |  mean R: {frame['y_r'].mean():.3f}")
    print(f"  Setup mix: {frame['setup_type'].value_counts().to_dict()}")
    return frame


def _build_market_frame(history: dict):
    """Locate the broad-market proxy (S&P 500) and VIX in the loaded history and
    build the shared causal market-context frame. Returns None if no proxy is
    available (market features then fall back to NaN)."""
    def _find(*names):
        for n in names:
            if n in history and len(history[n]):
                return history[n]
        return None

    spy = _find("^GSPC", "GSPC", "SPY", "^SPX")
    vix = _find("^VIX", "VIX")
    if spy is None:
        print("  [market] no S&P proxy in history — market features left NaN")
        return None
    print(f"  [market] proxy ok (rows={len(spy)}), vix={'ok' if vix is not None else 'missing'}")
    return compute_market_frame(spy, vix)


def build_from_checkpoints(
    out: str | Path | None = None,
    slippage_label: float = 0.0,
) -> pd.DataFrame:
    """Fast path: reuse the per-ticker sim checkpoints (labels already computed)
    and just attach the feature vector at each setup's bar.

    Avoids re-running the expensive ``find_historical_setups`` walk. Keeps one row
    per setup by filtering to a single slippage level (default 0.0).
    """
    from training.resumable_train import _load_ckpt

    history = download_history(load_training_universe())
    items = list(history.items())
    market = _build_market_frame(history)
    print(f"Joining checkpoints + features for {len(items)} tickers "
          f"(slippage={slippage_label}, market_ctx={'yes' if market is not None else 'no'})...")

    all_rows: list[dict] = []
    t0 = time.time()
    for i, (ticker, df) in enumerate(items, 1):
        ckpt = _load_ckpt(ticker)
        if not ckpt:
            continue
        feats = compute_feature_frame(df, market=market)
        pos_by_date = {str(d.date()): p for p, d in enumerate(df.index)}
        for r in ckpt:
            if r.get("slippage_pct", 0.0) != slippage_label:
                continue
            pos = pos_by_date.get(r.get("entry_date"))
            if pos is None:
                continue
            row = feats.iloc[pos].to_dict()
            row.update({
                "ticker": ticker,
                "date": r["entry_date"],
                "setup_type": r["setup_type"],
                "setup_score": r["setup_score"],
                "y_win": int(r["won"]),
                "y_r": r["pnl_r"],
                "exit_reason": r["exit_reason"],
                "days_held": r["days_held"],
            })
            all_rows.append(row)
        if i % 50 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {len(all_rows):,} rows "
                  f"({time.time() - t0:.0f}s)")

    frame = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS + META_COLUMNS)
    frame = frame.sort_values("date").reset_index(drop=True)
    out_path = Path(out) if out else OUT_DIR / "dataset.parquet"
    _save(frame, out_path)
    print(f"\nSaved {len(frame):,} rows x {frame.shape[1]} cols -> {out_path}")
    if len(frame):
        print(f"  Base win rate: {frame['y_win'].mean():.1%}  "
              f"|  mean R: {frame['y_r'].mean():.3f}")
        print(f"  Setup mix: {frame['setup_type'].value_counts().to_dict()}")
    return frame


def _save(frame: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(out_path, index=False)
    except Exception as exc:  # pyarrow/fastparquet missing -> pickle fallback
        fallback = out_path.with_suffix(".pkl")
        print(f"  (parquet unavailable: {exc}; writing pickle {fallback.name})")
        frame.to_pickle(fallback)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build ML dataset from price cache")
    p.add_argument("--limit", type=int, default=None, help="Use only first N tickers")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage %% for labels")
    p.add_argument("--out", default=None, help="Output path (.parquet)")
    p.add_argument("--from-checkpoints", action="store_true",
                   help="Reuse training/cache/sims_full checkpoints (fast, no re-walk)")
    args = p.parse_args(argv)
    if args.from_checkpoints:
        build_from_checkpoints(out=args.out, slippage_label=args.slippage)
    else:
        build_dataset(limit=args.limit, slippage=args.slippage, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
