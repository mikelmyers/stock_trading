"""Train and *honestly* validate a gradient-boosted model on the ML dataset.

The headline number is not accuracy — it is out-of-sample AUC under a purged
walk-forward split, compared against a shuffled-label baseline. If real AUC is
not clearly above the shuffled baseline, the model has found nothing and any
backtest built on it is fooling you.

Backend: prefers LightGBM, falls back to scikit-learn's HistGradientBoosting.
Both handle NaN features natively. Install either via ``requirements-ml.txt``.

CLI
---
    python -m training.ml.model --data training/ml/datasets/sample.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.ml.features import FEATURE_COLUMNS
from training.ml.splits import purged_walkforward_splits

OUT_DIR = Path(__file__).resolve().parent / "models"


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def _make_classifier():
    """Return (fit_fn, predict_proba_fn, name) for the best available backend.
    fit accepts an optional third arg of per-row sample weights."""
    try:
        import lightgbm as lgb

        def fit(X, y, w=None):
            m = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                n_jobs=-1, verbosity=-1,
            )
            m.fit(X, y, sample_weight=w)
            return m

        return fit, (lambda m, X: m.predict_proba(X)[:, 1]), "lightgbm"
    except ImportError:
        pass

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def fit(X, y, w=None):
            m = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                min_samples_leaf=200, l2_regularization=1.0,
            )
            m.fit(X, y, sample_weight=w)
            return m

        return fit, (lambda m, X: m.predict_proba(X)[:, 1]), "sklearn_histgbm"
    except ImportError as exc:
        raise SystemExit(
            "No ML backend available. Install one with:\n"
            "    pip install -r requirements-ml.txt\n"
            f"(import error: {exc})"
        )


def _make_regressor():
    """R-magnitude objective: predict expected y_r and rank by it. A win-prob
    objective is indifferent to a +0.5R vs +5R winner; the edge lives in the
    right tail, so rank on magnitude. Same (fit, score, name) contract."""
    try:
        import lightgbm as lgb

        def fit(X, y, w=None):
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                n_jobs=-1, verbosity=-1,
            )
            m.fit(X, y, sample_weight=w)
            return m

        return fit, (lambda m, X: m.predict(X)), "lightgbm_reg"
    except ImportError:
        pass

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        def fit(X, y, w=None):
            m = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                min_samples_leaf=200, l2_regularization=1.0,
            )
            m.fit(X, y, sample_weight=w)
            return m

        return fit, (lambda m, X: m.predict(X)), "sklearn_histgbm_reg"
    except ImportError as exc:
        raise SystemExit(
            "No ML backend available. Install one with:\n"
            "    pip install -r requirements-ml.txt\n"
            f"(import error: {exc})"
        )


def _auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def train(data_path: str | Path, n_splits: int = 5, objective: str = "win",
          use_weights: bool = False) -> dict:
    """objective: 'win' ranks by P(win); 'r' ranks by predicted R magnitude.
    use_weights: weight samples by average uniqueness (overlapping events of
    the same ticker stop masquerading as independent observations)."""
    if objective not in ("win", "r"):
        raise ValueError(f"objective must be 'win' or 'r', got {objective!r}")
    df = _load(Path(data_path))
    # setup_type as an extra categorical feature (integer-coded for trees).
    df = df.copy()
    df["setup_code"] = df["setup_type"].astype("category").cat.codes
    feat_cols = FEATURE_COLUMNS + ["setup_score", "setup_code"]

    X = df[feat_cols].to_numpy(dtype="float64")
    y_win = df["y_win"].to_numpy(dtype="int64")
    y_r = df["y_r"].to_numpy(dtype="float64")
    y = y_win if objective == "win" else y_r

    w = None
    if use_weights:
        from training.ml.weights import uniqueness_weights
        w = uniqueness_weights(df["date"], df["days_held"],
                               df["ticker"] if "ticker" in df else None)
        print(f"Uniqueness weights: mean={w.mean():.3f}  "
              f"effective N≈{w.sum():,.0f} of {len(w):,} rows")

    if objective == "win":
        fit, predict_score, backend = _make_classifier()
    else:
        fit, predict_score, backend = _make_regressor()
    print(f"Backend: {backend}  |  objective={objective}  weights={use_weights}  "
          f"|  {len(df):,} rows, {len(feat_cols)} features")

    splits = purged_walkforward_splits(df["date"], n_splits=n_splits)
    fold_auc, fold_auc_shuffled, fold_top_r = [], [], []
    rng = np.random.default_rng(42)

    for i, (tr, te) in enumerate(splits, 1):
        model = fit(X[tr], y[tr], w[tr] if w is not None else None)
        scores = predict_score(model, X[te])
        # AUC vs y_win is a valid ranking metric for either objective; the
        # decile-10 mean R is the number the realized book actually eats.
        a = _auc(y_win[te], scores)
        top = scores >= np.quantile(scores, 0.9)
        top_r = float(y_r[te][top].mean()) if top.any() else float("nan")
        # baseline: same model on shuffled labels -> should score ~0.5
        shuffled = fit(X[tr], rng.permutation(y[tr]),
                       w[tr] if w is not None else None)
        a0 = _auc(y_win[te], predict_score(shuffled, X[te]))
        fold_auc.append(a)
        fold_auc_shuffled.append(a0)
        fold_top_r.append(top_r)
        print(f"  fold {i}: train={len(tr):,} test={len(te):,}  "
              f"AUC={a:.4f}  (shuffled baseline={a0:.4f})  top-decile R={top_r:+.4f}")

    mean_auc = float(np.nanmean(fold_auc))
    mean_base = float(np.nanmean(fold_auc_shuffled))
    edge = mean_auc - mean_base
    print(f"\nMean OOS AUC: {mean_auc:.4f}  |  shuffled: {mean_base:.4f}  |  edge: {edge:+.4f}")
    print(f"Mean OOS top-decile R: {float(np.nanmean(fold_top_r)):+.4f}")
    verdict = (
        "signal present" if edge > 0.02 else
        "marginal" if edge > 0.005 else
        "NO measurable edge — do not trade this"
    )
    print(f"Verdict: {verdict}")

    # Final model on all data, plus importances if the backend exposes them.
    final = fit(X, y, w)
    importances = {}
    raw = getattr(final, "feature_importances_", None)
    if raw is not None:
        importances = dict(sorted(
            zip(feat_cols, (float(v) for v in raw)),
            key=lambda kv: kv[1], reverse=True,
        ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": backend,
        "objective": objective,
        "uniqueness_weights": use_weights,
        "rows": int(len(df)),
        "effective_rows": float(w.sum()) if w is not None else int(len(df)),
        "features": feat_cols,
        "mean_oos_auc": mean_auc,
        "shuffled_baseline_auc": mean_base,
        "edge": edge,
        "mean_top_decile_r": float(np.nanmean(fold_top_r)),
        "verdict": verdict,
        "fold_auc": fold_auc,
        "fold_auc_shuffled": fold_auc_shuffled,
        "fold_top_decile_r": fold_top_r,
        "feature_importances": importances,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    try:
        import joblib
        joblib.dump(final, OUT_DIR / "model.joblib")
        print(f"Saved model -> {OUT_DIR / 'model.joblib'}")
    except Exception:
        print("(joblib unavailable; model not serialized — report.json written)")
    print(f"Report -> {OUT_DIR / 'report.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train/validate ML model on dataset")
    p.add_argument("--data", required=True, help="Path to dataset (.parquet/.pkl)")
    p.add_argument("--splits", type=int, default=5, help="Walk-forward folds")
    p.add_argument("--objective", choices=["win", "r"], default="win",
                   help="win = rank by P(win); r = rank by predicted R magnitude")
    p.add_argument("--weights", action="store_true",
                   help="weight samples by average uniqueness (de-duplicate overlap)")
    args = p.parse_args(argv)
    train(args.data, n_splits=args.splits, objective=args.objective,
          use_weights=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
