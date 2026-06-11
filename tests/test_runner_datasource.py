"""Runner data-source helpers — avg volume fallback and warrant filter."""

from runner.datasource import _avg_vol_20d, _is_warrant_or_unit
from runner import conditions as C
from runner.datasource import _build_cv


def test_avg_vol_20d_uses_prior_bars():
    daily = [{"v": 100}, {"v": 200}, {"v": 300}]
    assert _avg_vol_20d(daily, {}) == 150.0


def test_avg_vol_20d_falls_back_to_prev_daily_when_one_bar():
    daily = [{"v": 50000}]
    prev = {"v": 1206}
    assert _avg_vol_20d(daily, prev) == 1206.0


def test_avg_vol_20d_none_when_no_history():
    assert _avg_vol_20d([], {}) is None
    assert _avg_vol_20d([{"v": 1}], {}) is None


def test_warrant_filter():
    assert _is_warrant_or_unit("HLLY.WS")
    assert _is_warrant_or_unit("SPKLW")
    assert _is_warrant_or_unit("DAICW")
    assert not _is_warrant_or_unit("AAPL")
    assert not _is_warrant_or_unit("EDHL")
    assert not _is_warrant_or_unit("RKDA")


def test_build_cv_computes_rvol_with_single_bar_history():
    raw = dict(
        asof="2026-06-11T14:00:00Z",
        price=7.37,
        prev_close=3.26,
        day_open=16.5,
        vol_today=66802,
        avg_vol_20d=1206.0,
        bid=7.36,
        ask=7.38,
        news=[{"headline": "FDA clearance"}],
        catalyst_type="fda",
        halts_today=0,
        bars=[dict(high=7.5, low=7.2, close=7.37, volume=10000)],
    )
    cv = _build_cv("EDHL", raw, minutes_since_open=120.0, regime=None)
    assert cv.rvol is not None
    assert cv.rvol >= C.CAND_RVOL_MIN
    assert cv.is_candidate