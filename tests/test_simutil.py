"""Regression tests for the measurement bugs fixed in training/simutil.py:
trading-day close dates (was calendar days) and daily-resampled Sharpe
(was sqrt(252) over per-trade events)."""

import numpy as np
import pandas as pd
import pytest

from training.simutil import (
    event_curve_sharpe,
    trading_close_dates,
    trading_to_calendar_days,
)


class TestTradingCloseDates:
    def test_skips_weekends(self):
        # Friday 2024-01-05 + 1 trading day = Monday 2024-01-08, not Saturday
        out = trading_close_dates(["2024-01-05"], [1])
        assert out[0] == pd.Timestamp("2024-01-08")

    def test_14_trading_days_spans_about_20_calendar(self):
        # The old bug: 14 trading days modeled as 14 calendar days freed
        # capacity slots ~30% early.
        out = trading_close_dates(["2024-01-02"], [14])
        span = (out[0] - pd.Timestamp("2024-01-02")).days
        assert span == 20  # 2024-01-02 (Tue) + 14 business days = 2024-01-22

    def test_vectorized(self):
        dates = ["2024-01-02", "2024-01-05"]
        out = trading_close_dates(dates, [2, 2])
        assert list(out) == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-09")]


def test_trading_to_calendar_days_covers_horizon():
    # 14 trading days is ~20 calendar days; the cover must be >= that.
    assert trading_to_calendar_days(14) >= 20


class TestEventCurveSharpe:
    def test_undefined_cases_return_zero(self):
        assert event_curve_sharpe(pd.Series(dtype=float)) == 0.0
        flat = pd.Series([100.0, 100.0],
                         index=pd.to_datetime(["2024-01-02", "2024-06-03"]))
        assert event_curve_sharpe(flat) == 0.0

    def test_sparse_events_not_annualized_as_daily(self):
        # ~50 trade events/yr with i.i.d. noise returns. Naive sqrt(252) over
        # the event series inflates Sharpe by ~sqrt(252/50) ~ 2.2x; the daily
        # resample must come out materially below the naive number.
        rng = np.random.default_rng(0)
        n = 250  # ~5 years at 50 events/yr
        dates = pd.date_range("2019-01-01", periods=n, freq="5B")
        rets = rng.normal(0.004, 0.02, n)
        eq = pd.Series(100 * np.cumprod(1 + rets), index=dates)

        naive = rets.mean() / rets.std() * np.sqrt(252)
        honest = event_curve_sharpe(eq)
        # event spacing is 5 business days -> inflation factor ~sqrt(5)
        assert honest == pytest.approx(naive / np.sqrt(5), rel=0.25)
        assert honest < naive / 1.8

    def test_daily_curve_matches_direct_computation(self):
        # On an already-daily curve the helper equals the textbook formula.
        rng = np.random.default_rng(1)
        dates = pd.date_range("2022-01-03", periods=504, freq="B")
        rets = rng.normal(0.0005, 0.01, 504)
        eq = pd.Series(100 * np.cumprod(1 + rets), index=dates)
        direct = eq.pct_change().dropna()
        expected = direct.mean() / direct.std() * np.sqrt(252)
        assert event_curve_sharpe(eq) == pytest.approx(expected, rel=1e-9)
