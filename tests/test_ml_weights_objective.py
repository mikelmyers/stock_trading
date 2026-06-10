"""Tests for uniqueness weights and the R-magnitude training objective."""

import numpy as np
import pandas as pd
import pytest

from training.ml.weights import uniqueness_weights


class TestUniquenessWeights:
    def test_lone_event_gets_full_weight(self):
        w = uniqueness_weights(["2024-01-02"], [10], ["AAA"])
        assert w[0] == pytest.approx(1.0)

    def test_identical_overlap_splits_weight(self):
        # Two events, same ticker, same window -> concurrency 2 everywhere.
        w = uniqueness_weights(["2024-01-02", "2024-01-02"], [10, 10],
                               ["AAA", "AAA"])
        assert w == pytest.approx([0.5, 0.5])

    def test_different_tickers_do_not_interact(self):
        w = uniqueness_weights(["2024-01-02", "2024-01-02"], [10, 10],
                               ["AAA", "BBB"])
        assert w == pytest.approx([1.0, 1.0])

    def test_partial_overlap_lands_between(self):
        # Second event starts halfway through the first's window.
        w = uniqueness_weights(["2024-01-02", "2024-01-09"], [10, 10],
                               ["AAA", "AAA"])
        assert 0.5 < w[0] < 1.0 and 0.5 < w[1] < 1.0

    def test_disjoint_events_full_weight(self):
        w = uniqueness_weights(["2024-01-02", "2024-03-01"], [5, 5],
                               ["AAA", "AAA"])
        assert w == pytest.approx([1.0, 1.0])

    def test_heavy_stacking_collapses_effective_n(self):
        # 30 consecutive-day near-duplicates: effective N must be far below 30.
        dates = pd.date_range("2024-01-02", periods=30, freq="B")
        w = uniqueness_weights(dates, [14] * 30, ["AAA"] * 30)
        assert w.sum() < 10


@pytest.fixture
def dataset(tmp_path):
    pytest.importorskip("lightgbm")
    from training.ml.features import FEATURE_COLUMNS
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))),
                      columns=FEATURE_COLUMNS)
    # plant signal: first feature drives R
    df["date"] = pd.date_range("2018-01-02", periods=n, freq="B").astype(str)
    # consecutive blocks per ticker -> overlapping same-ticker events, so the
    # uniqueness weights have something to de-duplicate
    df["ticker"] = [f"T{i // 32}" for i in range(n)]
    df["setup_type"] = np.where(np.arange(n) % 2 == 0, "breakout", "gap_fill")
    df["setup_score"] = rng.integers(50, 100, n)
    df["days_held"] = rng.integers(2, 14, n)
    df["y_r"] = df[FEATURE_COLUMNS[0]] * 0.5 + rng.normal(0, 0.5, n)
    df["y_win"] = (df["y_r"] > 0).astype(int)
    path = tmp_path / "ds.pkl"
    df.to_pickle(path)
    return path


def test_train_r_objective_with_weights(dataset, tmp_path, monkeypatch):
    import training.ml.model as mm
    monkeypatch.setattr(mm, "OUT_DIR", tmp_path / "models")
    report = mm.train(dataset, n_splits=2, objective="r", use_weights=True)
    assert report["objective"] == "r" and report["uniqueness_weights"]
    assert report["effective_rows"] < report["rows"]
    # planted signal must be found: top decile R clearly positive OOS
    assert report["mean_top_decile_r"] > 0.2
    assert report["mean_oos_auc"] > report["shuffled_baseline_auc"]


def test_train_win_objective_backward_compatible(dataset, tmp_path, monkeypatch):
    import training.ml.model as mm
    monkeypatch.setattr(mm, "OUT_DIR", tmp_path / "models")
    report = mm.train(dataset, n_splits=2)
    assert report["objective"] == "win"
    assert report["rows"] == 800


def test_invalid_objective_rejected(dataset):
    import training.ml.model as mm
    with pytest.raises(ValueError):
        mm.train(dataset, objective="sharpe")
