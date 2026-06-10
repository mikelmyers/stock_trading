"""Regression tests for trade accounting in state.py: a scaled-out winner must
not be recorded as a loss, and totals must not double-count scale-out P&L."""

import json

import pytest

from state import ActivePosition, StateManager


def _manager(tmp_path):
    return StateManager(path=tmp_path / "state.json")


def _open_position(mgr, entry=100.0, stop=95.0, shares=30.0):
    mgr.open_position(ActivePosition(
        ticker="TEST", cap_category="Mid Cap", entry_price=entry, stop_loss=stop,
        shares=shares, shares_remaining=shares, entry_date="2026-01-02",
        confidence_score=80, resistance_level=99.0, max_risk=shares * (entry - stop),
        high_water_mark=entry, trailing_stop=stop,
    ))


def test_scaled_out_winner_is_a_win(tmp_path):
    # 30 shares at $100, stop $95 (risk $5/share, $150 total).
    # Scale 10 at +1R ($105) and 10 at +2R ($110), trail out the rest at $99.
    mgr = _manager(tmp_path)
    _open_position(mgr)
    mgr.partial_scale_out("TEST", 10, 105.0, "Target 1")
    mgr.partial_scale_out("TEST", 10, 110.0, "Target 2")
    closed = mgr.close_position("TEST", 99.0, "TRAILING_STOP")

    # +50 +100 -10 = +140 whole-trade; the old code reported -10 (a "loss")
    assert closed.pnl == pytest.approx(140.0)
    assert closed.r_multiple == pytest.approx(140.0 / 150.0, abs=0.01)
    assert mgr.state.wins == 1
    assert mgr.state.total_pnl == pytest.approx(140.0)


def test_plain_loss_unchanged(tmp_path):
    mgr = _manager(tmp_path)
    _open_position(mgr)
    closed = mgr.close_position("TEST", 95.0, "STOP_LOSS")
    assert closed.pnl == pytest.approx(-150.0)
    assert closed.r_multiple == pytest.approx(-1.0)
    assert mgr.state.wins == 0
    assert mgr.state.total_pnl == pytest.approx(-150.0)


def test_total_pnl_not_double_counted(tmp_path):
    mgr = _manager(tmp_path)
    _open_position(mgr)
    mgr.partial_scale_out("TEST", 10, 105.0, "Target 1")
    assert mgr.state.total_pnl == pytest.approx(50.0)  # banked at scale time
    mgr.close_position("TEST", 100.0, "TIME_STOP")     # remainder flat
    assert mgr.state.total_pnl == pytest.approx(50.0)  # not added twice


def test_state_roundtrip_with_legacy_position(tmp_path):
    # Positions saved before scaled_pnl existed must still load.
    path = tmp_path / "state.json"
    mgr = StateManager(path=path)
    _open_position(mgr)
    raw = json.loads(path.read_text())
    del raw["active_positions"][0]["scaled_pnl"]
    path.write_text(json.dumps(raw))
    reloaded = StateManager(path=path)
    assert reloaded.get_position("TEST").scaled_pnl == 0.0


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    mgr = _manager(tmp_path)
    _open_position(mgr)
    assert not list(tmp_path.glob("*.tmp"))
    assert StateManager(path=mgr.path).get_position("TEST") is not None
