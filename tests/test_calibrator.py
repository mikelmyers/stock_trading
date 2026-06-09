"""Calibrator regression tests: thresholds must be learned from REAL trades at
the base slippage level — not bootstrap resamples (zero information) and not
each setup triple-counted across slippage variants."""

import json

import pytest

import training.calibrator as cal


def _trade(pnl_r, setup="breakout", boot=0, slip=0.0, score=80, ticker="AAA"):
    return {"pnl_r": pnl_r, "won": pnl_r > 0, "setup_type": setup,
            "bootstrap_id": boot, "slippage_pct": slip, "setup_score": score,
            "ticker": ticker}


@pytest.fixture(autouse=True)
def _isolate_params(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "LEARNED_PARAMS_FILE", tmp_path / "learned.json")


def test_real_only_drops_bootstrap_and_slippage_variants():
    results = ([_trade(1.0)] +                       # the one real base trade
               [_trade(1.0, slip=0.1), _trade(1.0, slip=0.2)] +  # variants
               [_trade(1.0, boot=i) for i in range(1, 50)])      # bootstrap
    assert len(cal._real_only(results)) == 1


def test_bootstrap_cannot_buy_confidence_upgrades():
    # 40 real winners + 5000 bootstrap winners: the old code saw 5040 rows
    # with a 100% win rate and upgraded confidence to HIGH on synthetic data.
    real = [_trade(0.5) for _ in range(40)]
    boot = [_trade(0.5, boot=i) for i in range(1, 5001)]
    params = cal.calibrate(real + boot)
    assert params["min_probability_confidence"] == "LOW"
    assert params["trained_real_trades"] == 40


def test_setup_needs_min_real_trades_to_enable():
    few = [_trade(1.0, setup="gap_fill") for _ in range(10)]
    many = [_trade(0.5, setup="breakout")
            for _ in range(cal.MIN_REAL_TRADES_TO_ENABLE)]
    params = cal.calibrate(few + many)
    perf = params["setup_performance"]
    assert not perf["gap_fill"]["enabled"]      # 10 < MIN_REAL_TRADES_TO_ENABLE
    assert perf["breakout"]["enabled"]


def test_losing_setup_disabled():
    losers = [_trade(-1.0, setup="bear_breakdown") for _ in range(50)]
    winners = [_trade(0.5, setup="breakout") for _ in range(50)]
    params = cal.calibrate(losers + winners)
    assert "bear_breakdown" not in params["enabled_setups"]
    assert "breakout" in params["enabled_setups"]


def test_all_loss_subset_does_not_nan():
    # np.mean([]) is NaN and NaN is truthy — the old `or 1` guards were dead.
    losers = [_trade(-1.0) for _ in range(40)]
    params = cal.calibrate(losers)
    assert params["trained_expectancy"] == pytest.approx(-1.0)
    assert json.dumps(params)  # serializable, no NaN poisoning
