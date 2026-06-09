"""Tests for shared indicators, focused on the rolling VWAP that replaced the
frame-anchored cumulative version (training and live must compute the same
quantity, causally)."""

import numpy as np
import pandas as pd
import pytest

from agents.indicators import calculate_atr, calculate_rolling_vwap


def _ohlcv(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    vol = rng.integers(1_000, 5_000, n).astype(float)
    idx = pd.date_range("2026-01-02", periods=n, freq="B")
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def test_rolling_vwap_matches_manual_window():
    df = _ohlcv()
    vwap = calculate_rolling_vwap(df, 20)
    w = df.iloc[-20:]
    tp = (w["High"] + w["Low"] + w["Close"]) / 3
    expected = (tp * w["Volume"]).sum() / w["Volume"].sum()
    assert vwap.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_rolling_vwap_is_window_invariant():
    # The value at the last bar must not depend on how much history precedes
    # the window — the defect that made the old cumsum "VWAP" mean different
    # things in training (full history) vs live (60d frame).
    df = _ohlcv(120)
    full = calculate_rolling_vwap(df, 20).iloc[-1]
    suffix = calculate_rolling_vwap(df.iloc[-30:], 20).iloc[-1]
    assert full == suffix


def test_rolling_vwap_is_causal():
    df = _ohlcv(80)
    vwap = calculate_rolling_vwap(df, 20)
    truncated = calculate_rolling_vwap(df.iloc[:50], 20)
    pd.testing.assert_series_equal(vwap.iloc[:50], truncated)


def test_atr_true_range_uses_prior_close_gap():
    # A huge overnight gap must show up in true range even with a tiny H-L bar.
    df = pd.DataFrame({
        "High":  [10.0, 20.2],
        "Low":   [9.0, 20.0],
        "Close": [10.0, 20.1],
        "Volume": [1.0, 1.0],
    })
    atr = calculate_atr(df, window=1)
    assert atr.iloc[1] == np.float64(20.2 - 10.0)
