# ML layer (`training/ml/`)

Turns the existing rule-based backtest into a supervised-learning pipeline. The
hand-coded setups stay as the **candidate generator**; an ML model learns to
**rank/filter** those candidates by predicted edge.

```
OHLCV (.pkl cache)
      │  features.compute_feature_frame   (causal, point-in-time)
      ▼
features  ──┐
            ├─ dataset.build_dataset ──►  flat table (parquet)
labels  ────┘     (label = simulate_trade_forward outcome)
      │  model.train  (purged walk-forward CV vs shuffled baseline)
      ▼
   model.joblib  +  report.json
```

## Quick start

```bash
pip install -r requirements-ml.txt        # lightgbm / sklearn / pyarrow

# 1. Build a dataset (start small to sanity-check)
python -m training.ml.dataset --limit 50 --out training/ml/datasets/sample.parquet

# 2. Train + honestly validate
python -m training.ml.model --data training/ml/datasets/sample.parquet
```

## What to look at

`report.json` reports **out-of-sample AUC under a purged walk-forward split**,
next to a **shuffled-label baseline**. The only number that matters is the
*edge* (real AUC − shuffled AUC):

- `edge > 0.02` → some signal.
- `edge ≈ 0` → the model found nothing; do **not** build a strategy on it.
- AUC ≈ 0.9 → you have a leak, guaranteed. Find it.

## Design notes / honesty checklist

- **Causal features only.** Everything in `features.py` uses data up to bar `i`.
- **Purged + embargoed CV** (`splits.py`) prevents overlapping-label leakage and
  enforces time order. Never use random k-fold here.
- **Labels = the existing simulator**, so the ML target is consistent with how
  trades are actually managed (stops, scale-outs, trailing, time stop).
- **Known limitations not yet handled:** survivorship bias (universe is *today's*
  S&P 500, so delisted losers are missing) and point-in-time index membership.
  Treat results as optimistic until those are addressed.

Nothing in this package is imported by the live agent or the calibration run, so
it is safe to iterate on independently.
