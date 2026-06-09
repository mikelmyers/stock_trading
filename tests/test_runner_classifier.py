"""Runner classifier regression tests: the model must validate on a
time-ordered holdout before it can gate live trades, ship calibrated
(base-rate-aware) thresholds instead of fixed 0.50s, and never crash the
intraday loop when the pickle is unusable."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

import runner.classifier as clf


def _frame(n=400, signal=True, seed=0):
    """Labeled candidate frame; if signal, rvol drives the MFE label."""
    rng = np.random.default_rng(seed)
    rvol = rng.uniform(1, 10, n)
    noise = rng.normal(0, 3, n)
    mfe = (rvol * 3 + noise) if signal else rng.uniform(0, 25, n)
    df = pd.DataFrame({
        "symbol": [f"S{i % 40}" for i in range(n)],
        "asof": pd.date_range("2026-01-02", periods=n, freq="30min").astype(str),
        "price": rng.uniform(2, 10, n), "rvol": rvol,
        "gap_pct": rng.uniform(0, 30, n), "vol_today": rng.uniform(1e6, 9e6, n),
        "max_favorable_pct": np.clip(mfe, 0, None),
    })
    return df


def _patch_io(monkeypatch, tmp_path, df):
    monkeypatch.setattr(clf, "load_training_frame", lambda: df)
    monkeypatch.setattr(clf, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(clf, "_bundle_cache", None)


def test_refuses_to_train_on_too_little_data(monkeypatch, tmp_path):
    _patch_io(monkeypatch, tmp_path, _frame(n=50))
    out = clf.train()
    assert not out["trained"]
    assert not (tmp_path / "model.pkl").exists()


def test_trains_with_holdout_auc_and_calibrated_thresholds(monkeypatch, tmp_path):
    _patch_io(monkeypatch, tmp_path, _frame(n=600, signal=True))
    out = clf.train()
    assert out["trained"], out
    assert out["holdout_auc"]["monster"] >= clf.MIN_HOLDOUT_AUC
    # thresholds come from the holdout distribution, not a fixed 0.50
    assert 0.0 < out["thr_monster"] < 1.0
    assert (tmp_path / "model.pkl").exists()


def test_no_skill_model_is_not_shipped(monkeypatch, tmp_path):
    # Labels independent of features -> holdout AUC ~0.5 -> stay on rules.
    _patch_io(monkeypatch, tmp_path, _frame(n=600, signal=False))
    out = clf.train()
    assert not out["trained"]
    assert "no out-of-sample skill" in out["reason"]
    assert not (tmp_path / "model.pkl").exists()


def test_score_returns_none_when_untrained(monkeypatch, tmp_path):
    _patch_io(monkeypatch, tmp_path, _frame())
    from runner.conditions import ConditionVector
    cv = ConditionVector(symbol="X", asof="2026-06-09T14:00:00Z", price=5.0)
    assert clf.score(cv) is None


def test_corrupt_model_falls_back_to_rules(monkeypatch, tmp_path):
    _patch_io(monkeypatch, tmp_path, _frame())
    (tmp_path / "model.pkl").write_bytes(b"not a pickle")
    from runner.conditions import ConditionVector
    cv = ConditionVector(symbol="X", asof="2026-06-09T14:00:00Z", price=5.0)
    assert clf.score(cv) is None  # must not raise mid-cycle
