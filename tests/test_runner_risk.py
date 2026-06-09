"""Regression tests for the runner's discipline gates: they used to run on
state that was never updated (daily-loss limit and streak lockout could never
fire; open_positions only ever incremented)."""

import datetime as dt

import pytest

from runner.conditions import ConditionVector, classify
from runner.risk import RiskConfig, RiskEngine, RunnerState

NOW = dt.datetime(2026, 6, 9, 15, 0, tzinfo=dt.timezone.utc)


def _state(tmp_path, monkeypatch, equity=1000.0):
    import runner.risk as rr
    monkeypatch.setattr(rr, "STATE_DIR", tmp_path)
    return RunnerState.load("test_ep", equity)


def _cv():
    # A clean green-light candidate (mirrors MockSource AAAA)
    cv = ConditionVector(
        symbol="AAAA", asof="2026-06-09T14:00:00+00:00", price=4.10,
        float_shares=6e6, market_cap=None, avg_vol_20d=1.5e6, sector=None,
        rvol=8.0, gap_pct=20.0, premarket_vol=None, vol_today=8e6,
        vol_to_float=1.33, gap_atr=2.0, pct_change=36.7, vwap=3.9,
        dist_vwap_pct=5.1, vwap_slope=None, dist_pm_high_pct=-2.4,
        dist_pm_low_pct=20.6, extension_pct=5.1, has_news=True,
        catalyst_type="fda", spread_pct=0.5, halts_today=0, atr_pct=7.3,
        minutes_since_open=30.0, market_regime="risk_on",
    )
    return classify(cv)


@pytest.fixture
def engine():
    return RiskEngine(RiskConfig(use_classifier=False))


def test_green_light_candidate_is_taken(tmp_path, monkeypatch, engine):
    state = _state(tmp_path, monkeypatch)
    d = engine.evaluate(_cv(), state, now=NOW)
    assert d.action == "take", d.reason
    assert d.shares >= 1 and d.stop < d.entry


def test_daily_loss_limit_fires_after_broker_sync(tmp_path, monkeypatch, engine):
    state = _state(tmp_path, monkeypatch)
    state.sync_with_broker(1000.0, 0)            # anchors day_start_equity
    state.sync_with_broker(800.0, 0)             # -20% on the day
    d = engine.evaluate(_cv(), state, now=NOW)
    assert d.action == "skip"
    assert "daily loss limit" in d.reason


def test_sync_anchors_day_start_only_once_per_day(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch)
    state.sync_with_broker(1000.0, 0)
    state.sync_with_broker(1100.0, 1)
    assert state.day_start_equity == 1000.0
    assert state.daily_pnl == pytest.approx(100.0)
    assert state.open_positions == 1


def test_loss_streak_triggers_cooldown(tmp_path, monkeypatch, engine):
    cfg = engine.cfg
    state = _state(tmp_path, monkeypatch)
    for _ in range(cfg.max_consecutive_losses):
        state.register_outcome(-10.0, -1.0, cfg, now=NOW)
    assert state.locked_until is not None
    d = engine.evaluate(_cv(), state, now=NOW)
    assert d.action == "skip"
    assert "cooldown" in d.reason


def test_register_outcome_decrements_open_positions(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch)
    state.open_positions = 2
    state.register_outcome(5.0, 0.5, RiskConfig(), now=NOW)
    assert state.open_positions == 1


def test_max_concurrent_enforced_within_a_cycle(tmp_path, monkeypatch, engine):
    # Simulates the cycle loop: each take increments state before the next
    # candidate is evaluated, so one scan can't submit past the cap.
    state = _state(tmp_path, monkeypatch)
    taken = 0
    for _ in range(5):
        d = engine.evaluate(_cv(), state, now=NOW)
        if d.action == "take":
            taken += 1
            state.trades_today += 1
            state.open_positions += 1
    assert taken == engine.cfg.max_concurrent


def test_state_roundtrip_ignores_missing_new_fields(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch)
    state.save()
    reloaded = RunnerState.load("test_ep", 1000.0)
    assert reloaded.synced_date == ""
