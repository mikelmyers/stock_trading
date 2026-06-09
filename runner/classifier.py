"""Runner classifier — learns A+ / blow-up conditions from the system's OWN trades.

Dormant by design: until the ledger holds enough labeled outcomes it refuses to
train, and `score()` returns None so the system runs on the seed rules
(green_light / blowup_flags). Once data accumulates, it learns two heads from the
condition-vector:
  * P(monster) — did the setup have big upside (max-favorable-excursion >= 20%)?
  * P(loss)    — did it fail to go (MFE < 3%)?
Labels use the setup's POTENTIAL (MFE), not realized P&L, so the classifier learns
which *conditions* are good independent of how the trade was managed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from runner.conditions import ConditionVector
from runner.logger import load_training_frame

MODEL_PATH = Path(__file__).resolve().parent / "data" / "runner_classifier.pkl"
MONSTER_MFE = 20.0          # % max-favorable-excursion = "monster potential"
SCRATCH_MFE = 3.0           # % below which the setup didn't go
MIN_SAMPLES = 200
MIN_PER_CLASS = 15
HOLDOUT_FRAC = 0.25         # most-recent quarter held out, time-ordered
MIN_HOLDOUT_AUC = 0.55      # below this the model has no skill -> stay on rules
MONSTER_TOP_FRAC = 0.30     # gate passes the top 30% of holdout p_monster
LOSS_VETO_FRAC = 0.20       # gate vetoes the worst 20% of holdout p_loss

CATALYST_CODES = {None: 0, "news": 1, "partnership": 2, "earnings": 3, "fda": 4, "offering": 5}
REGIME_CODES = {None: 0, "risk_off": 1, "risk_on": 2}
NUMERIC = ["price", "float_shares", "market_cap", "avg_vol_20d", "rvol", "gap_pct",
           "premarket_vol", "vol_today", "vol_to_float", "gap_atr", "pct_change",
           "dist_vwap_pct", "vwap_slope", "dist_pm_high_pct", "dist_pm_low_pct",
           "extension_pct", "spread_pct", "halts_today", "atr_pct", "minutes_since_open"]
FEATURES = NUMERIC + ["catalyst_code", "regime_code", "has_news_i"]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["catalyst_code"] = x.get("catalyst_type").map(CATALYST_CODES).fillna(0) if "catalyst_type" in x else 0
    x["regime_code"] = x.get("market_regime").map(REGIME_CODES).fillna(0) if "market_regime" in x else 0
    x["has_news_i"] = x.get("has_news", False).astype(float) if "has_news" in x else 0.0
    for c in NUMERIC:
        if c not in x:
            x[c] = np.nan
    return x[FEATURES].astype("float64")


def status() -> dict:
    try:
        df = load_training_frame()
    except FileNotFoundError:
        return {"labeled": 0, "trained": MODEL_PATH.exists(), "ready": False}
    lab = df.dropna(subset=["max_favorable_pct"]) if "max_favorable_pct" in df else df.iloc[0:0]
    monsters = int((lab["max_favorable_pct"] >= MONSTER_MFE).sum()) if len(lab) else 0
    return {"labeled": len(lab), "monsters": monsters, "trained": MODEL_PATH.exists(),
            "ready": len(lab) >= MIN_SAMPLES and monsters >= MIN_PER_CLASS}


def train() -> dict:
    """Train both heads with a time-ordered holdout. The model is only shipped
    if it shows out-of-sample skill (holdout AUC), and the gate thresholds are
    calibrated from the holdout score distribution — a fixed p>=0.50 on a ~7%
    base-rate event would veto everything the day the model went live."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    df = load_training_frame()
    lab = df.dropna(subset=["max_favorable_pct"])
    sort_col = "asof" if "asof" in lab.columns else "logged_at"
    lab = lab.sort_values(sort_col).reset_index(drop=True)
    y_monster = (lab["max_favorable_pct"] >= MONSTER_MFE).astype(int)
    y_loss = (lab["max_favorable_pct"] < SCRATCH_MFE).astype(int)
    for name, y in [("monster", y_monster), ("loss", y_loss)]:
        if len(lab) < MIN_SAMPLES or y.sum() < MIN_PER_CLASS or (1 - y).sum() < MIN_PER_CLASS:
            return {"trained": False, "reason": f"not enough labeled data "
                    f"({len(lab)} rows, {int(y.sum())} positive {name}; "
                    f"need >= {MIN_SAMPLES} and >= {MIN_PER_CLASS}/class) — using rules"}
    X = _encode(lab)

    def fit(X_, y_):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                               min_child_samples=20, n_jobs=-1, verbosity=-1)
        m.fit(X_.to_numpy(), y_.to_numpy()); return m

    cut = int(len(lab) * (1 - HOLDOUT_FRAC))
    aucs, holdout_p = {}, {}
    for name, y in [("monster", y_monster), ("loss", y_loss)]:
        if y.iloc[:cut].nunique() < 2 or y.iloc[cut:].nunique() < 2:
            return {"trained": False, "reason": f"{name} head: one class missing "
                    "from the time split — need more data"}
        m = fit(X.iloc[:cut], y.iloc[:cut])
        p = m.predict_proba(X.iloc[cut:].to_numpy())[:, 1]
        aucs[name] = float(roc_auc_score(y.iloc[cut:], p))
        holdout_p[name] = p
    if aucs["monster"] < MIN_HOLDOUT_AUC:
        return {"trained": False, "holdout_auc": aucs,
                "reason": f"holdout AUC {aucs['monster']:.3f} < {MIN_HOLDOUT_AUC} — "
                          "no out-of-sample skill, staying on rules"}

    thr_monster = float(np.quantile(holdout_p["monster"], 1 - MONSTER_TOP_FRAC))
    thr_loss = float(np.quantile(holdout_p["loss"], 1 - LOSS_VETO_FRAC))
    bundle = {"monster": fit(X, y_monster), "loss": fit(X, y_loss), "features": FEATURES,
              "thr_monster": thr_monster, "thr_loss": thr_loss,
              "holdout_auc": aucs, "trained_rows": len(lab)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return {"trained": True, "samples": len(lab), "monsters": int(y_monster.sum()),
            "holdout_auc": aucs, "thr_monster": round(thr_monster, 4),
            "thr_loss": round(thr_loss, 4)}


_bundle_cache: tuple | None = None    # (mtime, bundle)


def _load_bundle():
    global _bundle_cache
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except OSError:
        return None
    if _bundle_cache and _bundle_cache[0] == mtime:
        return _bundle_cache[1]
    with open(MODEL_PATH, "rb") as f:
        b = pickle.load(f)
    _bundle_cache = (mtime, b)
    return b


def score(cv: ConditionVector) -> dict | None:
    """P(monster)/P(loss) + calibrated gate thresholds for one candidate, or
    None if untrained / unloadable (-> seed rules). Never raises: a corrupt or
    version-mismatched pickle must not crash the intraday loop mid-cycle."""
    try:
        b = _load_bundle()
        if b is None:
            return None
        X = _encode(pd.DataFrame([cv.to_row()]))[b["features"]]
        return {"p_monster": float(b["monster"].predict_proba(X.to_numpy())[:, 1][0]),
                "p_loss": float(b["loss"].predict_proba(X.to_numpy())[:, 1][0]),
                "thr_monster": b.get("thr_monster"), "thr_loss": b.get("thr_loss")}
    except Exception as e:
        print(f"    classifier unusable ({e!r}) — falling back to rules")
        return None
