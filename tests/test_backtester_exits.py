"""Regression tests for simulate_trade_forward's trailing stop: the level
tested against a bar's low must be the level armed BEFORE that bar (raising
the trail with the same bar's close was intrabar lookahead)."""

import numpy as np
import pandas as pd

from training.backtester import simulate_trade_forward


def _df(rows):
    """rows: list of (open, high, low, close); constant volume."""
    idx = pd.date_range("2026-01-02", periods=len(rows), freq="B")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": [1e6] * len(rows)}, index=idx)


def _setup(entry=100.0, stop=95.0):
    return {"is_valid_setup": True, "bias": "bullish", "current_price": entry,
            "stop_loss": stop, "resistance_level": entry,
            "confidence_score": 80, "setup_type": "breakout"}


# 20 flat warmup bars so ATR(14) is defined (~1.0) when the trade starts.
WARMUP = [(100.0, 100.5, 99.5, 100.0)] * 20
ENTRY_IDX = len(WARMUP) - 1


def test_trail_not_triggered_by_same_bar_that_raises_it():
    # Entry 100, stop 95 (risk 5). Day 1 closes at 105 (t1) -> trail arms
    # around 105 - 2*ATR (~102). Day 2 closes way up at 120 with low 104:
    # a trail computed FROM day 2's own close (120 - 2*ATR ~ 115) would be
    # "hit" by the 104 low, but the level actually in force during day 2
    # came from day 1 and is never touched.
    rows = WARMUP + [
        (100.0, 106.0, 99.0, 105.0),    # day 1: tags +1R, arms trail
        (105.0, 121.0, 104.0, 120.0),   # day 2: big up bar, low 104
    ] + [(120.0, 121.0, 119.0, 120.0)] * 13
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(), atr14=None)
    # Old lookahead code exited TRAILING_STOP on day 1 (the arming bar's own
    # pre-breakout low breached the trail computed from that bar's close).
    # No bar's low ever touches the previously-armed trail, so the trade
    # must ride to MAX_HOLD.
    assert sim.exit_reason == "MAX_HOLD", (
        f"trail used same-bar close: {sim.exit_reason} day {sim.days_held}")


def test_trail_still_exits_on_later_breach():
    # After a run-up, a genuine breach of YESTERDAY's trail must still exit.
    rows = WARMUP + [
        (100.0, 106.0, 99.0, 105.0),     # day 1: arms trail
        (105.0, 113.0, 104.9, 112.0),    # day 2: ratchets trail to ~108
        (112.0, 112.5, 96.0, 97.0),      # day 3: crashes through it
    ] + [(97.0, 98.0, 96.0, 97.0)] * 12
    sim = simulate_trade_forward(_df(rows), ENTRY_IDX, _setup(), atr14=None)
    assert sim.exit_reason == "TRAILING_STOP"
    assert sim.days_held == 3


def test_hard_stop_unaffected():
    rows = [(100.0, 100.5, 99.5, 100.0),
            (100.0, 101.0, 94.0, 96.0)]      # straight to the stop
    rows += [(96.0, 97.0, 95.5, 96.0)] * 14
    df = _df(rows)
    sim = simulate_trade_forward(df, 0, _setup(), atr14=None)
    assert sim.exit_reason == "HARD_STOP"
    assert sim.pnl_r == -1.0
