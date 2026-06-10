"""Tests for the honest-label modes: next-open entry fills and
gap-through-stop fills (the live path market-buys the next morning, and a
stock that opens 20% below the stop does not fill at the stop)."""

import pandas as pd
import pytest

from training.backtester import _process_ticker, simulate_trade_forward

WARMUP = [(100.0, 100.5, 99.5, 100.0)] * 20
ENTRY_IDX = len(WARMUP) - 1


def _df(rows):
    idx = pd.date_range("2026-01-02", periods=len(rows), freq="B")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": [1e6] * len(rows)}, index=idx)


def _setup(entry=100.0, stop=95.0):
    return {"is_valid_setup": True, "bias": "bullish", "current_price": entry,
            "stop_loss": stop, "resistance_level": entry,
            "confidence_score": 80, "setup_type": "breakout"}


def test_next_open_fills_at_next_bars_open():
    rows = WARMUP + [(103.0, 110.0, 102.0, 109.0)] + \
        [(109.0, 110.0, 108.0, 109.0)] * 14
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(),
                                 atr14=None, entry_fill="next_open")
    assert sim.entry_price == 103.0  # next bar's open, not the signal close


def test_next_open_gap_below_stop_skips_the_trade():
    # Signal at 100 with stop 95; next morning opens at 90 -> risk <= 0,
    # a sane system does not enter.
    rows = WARMUP + [(90.0, 92.0, 89.0, 91.0)] * 15
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(),
                                 atr14=None, entry_fill="next_open")
    assert sim is None


def test_gap_through_stop_fills_at_the_open():
    # Stop 95; the bar OPENS at 80 (gap through). Legacy mode books -1R at
    # 95; honest mode books the open: (80-100)/5 = -4R.
    crash = [(80.0, 82.0, 78.0, 79.0)] + [(79.0, 80.0, 78.0, 79.0)] * 14
    df = _df(WARMUP + crash)
    legacy = simulate_trade_forward(df, ENTRY_IDX, _setup(), atr14=None)
    honest = simulate_trade_forward(df, ENTRY_IDX, _setup(), atr14=None,
                                    gap_fills=True)
    assert legacy.exit_price == 95.0 and legacy.pnl_r == pytest.approx(-1.0)
    assert honest.exit_price == 80.0 and honest.pnl_r == pytest.approx(-4.0)
    assert honest.exit_reason == "HARD_STOP"


def test_intraday_stop_touch_still_fills_at_stop():
    # Opens above the stop, trades down through it intraday -> stop fills.
    rows = WARMUP + [(99.0, 100.0, 92.0, 93.0)] + [(93.0, 94.0, 92.0, 93.0)] * 14
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(),
                                 atr14=None, gap_fills=True)
    assert sim.exit_price == 95.0
    assert sim.pnl_r == pytest.approx(-1.0)


def test_defaults_reproduce_legacy_labels():
    # The historical sims_full checkpoints must stay reproducible: default
    # kwargs == legacy behavior.
    rows = WARMUP + [(100.0, 101.0, 94.0, 96.0)] + [(96.0, 97.0, 95.5, 96.0)] * 14
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(), atr14=None)
    assert (sim.entry_price, sim.exit_price, sim.pnl_r) == (100.0, 95.0, -1.0)


def test_invalid_entry_fill_rejected():
    with pytest.raises(ValueError):
        simulate_trade_forward(_df(WARMUP * 2), ENTRY_IDX, _setup(),
                               atr14=None, entry_fill="vwap")


def test_process_ticker_accepts_legacy_and_realism_tuples():
    # Old 5-tuple task args (pre-realism) must still work for resumability.
    df = _df(WARMUP + [(100.0, 101.0, 99.0, 100.0)] * 30).reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    legacy_args = ("TST", df.to_dict(), [0.0], 5, None)
    realism_args = ("TST", df.to_dict(), [0.0], 5, None, "next_open", True)
    for args in (legacy_args, realism_args):
        ticker, results, raw = _process_ticker(args)
        assert ticker == "TST"
        assert isinstance(results, list)
