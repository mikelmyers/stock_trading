"""Calibrate agent thresholds from simulation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import BASE_DIR

LEARNED_PARAMS_FILE = BASE_DIR / "learned_params.json"

DEFAULT_PARAMS = {
    "min_setup_score": 70,
    "min_composite_score": 55,
    "volume_multiplier": 1.5,
    "compression_ratio": 0.85,
    "min_ceiling_touches": 2,
    "min_probability_confidence": "LOW",
    "expectancy_threshold": 0.0,
    "trained_on_simulations": 0,
    "trained_win_rate": 0.0,
    "trained_expectancy": 0.0,
    "enabled_setups": ["breakout", "ma_pullback"],
    "setup_performance": {},
    "options_backtest": {},
}


def load_learned_params() -> dict:
    if LEARNED_PARAMS_FILE.exists():
        data = json.loads(LEARNED_PARAMS_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT_PARAMS, **data}
    return dict(DEFAULT_PARAMS)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_learned_params(params: dict) -> Path:
    LEARNED_PARAMS_FILE.write_text(
        json.dumps(_json_safe(params), indent=2), encoding="utf-8"
    )
    return LEARNED_PARAMS_FILE


# A setup needs at least this many REAL trades before win/expectancy stats
# mean anything; 5 was noise-mining.
MIN_REAL_TRADES_TO_ENABLE = 30


def _real_only(results: list[dict]) -> list[dict]:
    """Calibration evidence: real (non-bootstrap) trades at the base slippage
    level. Bootstrap resamples add zero information, and counting each setup
    once per slippage level triple-counted everything."""
    real = [r for r in results if r.get("bootstrap_id", 0) == 0]
    if not real:
        return []
    base_slip = min(r.get("slippage_pct", 0.0) for r in real)
    return [r for r in real if r.get("slippage_pct", 0.0) == base_slip]


def _expectancy(rows: list[dict]) -> tuple[float, float]:
    """(win_rate, expectancy in R) with explicit empty-side guards — the old
    ``np.mean([...]) or 1`` never engaged because NaN is truthy."""
    if not rows:
        return 0.0, 0.0
    win_r = [r["pnl_r"] for r in rows if r["won"]]
    loss_r = [abs(r["pnl_r"]) for r in rows if not r["won"]]
    wr = len(win_r) / len(rows)
    aw = float(np.mean(win_r)) if win_r else 0.0
    al = float(np.mean(loss_r)) if loss_r else 0.0
    return wr, (wr * aw) - ((1 - wr) * al)


def _setup_stats(real_results: list[dict], setup_type: str) -> dict:
    real = [r for r in real_results if r.get("setup_type") == setup_type]
    if not real:
        return {"real_count": 0, "count": 0, "win_rate": 0, "expectancy": 0, "enabled": False}
    wr, exp = _expectancy(real)
    return {
        "real_count": len(real),
        "count": len(real),
        "win_rate": round(wr * 100, 1),
        "expectancy": round(float(exp), 3),
        "enabled": len(real) >= MIN_REAL_TRADES_TO_ENABLE and exp > 0,
    }


def calibrate(results: list[dict]) -> dict:
    """
    Learn optimal thresholds and which setup types have positive edge.
    Disables losing patterns, enables winning ones.
    """
    params = load_learned_params()

    if not results:
        return params

    # All calibration statistics run on real, base-slippage trades only.
    # Bootstrap copies and slippage variants stay useful for stress reporting
    # (summarize_results) but are not evidence.
    real = _real_only(results)
    if not real:
        return params

    win_rate, expectancy = _expectancy(real)

    setup_types = sorted({r.get("setup_type", "unknown") for r in real})
    setup_performance = {
        st: _setup_stats(real, st) for st in setup_types
    }

    enabled = [st for st, perf in setup_performance.items() if perf["enabled"]]
    if not enabled:
        best = max(setup_performance.items(), key=lambda x: x[1]["expectancy"], default=(None, {}))
        if best[0] and best[1].get("real_count", 0) >= 1:
            enabled = [best[0]]

    best_score = 70
    best_exp = -999.0
    enabled_results = [r for r in real if r.get("setup_type") in enabled] if enabled else real

    for cutoff in range(50, 101, 10):
        subset = [r for r in enabled_results if r["setup_score"] >= cutoff]
        if len(subset) < 20:
            continue
        _, sub_exp = _expectancy(subset)
        if sub_exp > best_exp:
            best_exp = sub_exp
            best_score = cutoff

    by_ticker: dict[str, list] = {}
    for r in real:
        by_ticker.setdefault(r["ticker"], []).append(r["pnl_r"])

    strong_tickers = [
        t for t, pnls in by_ticker.items()
        if len(pnls) >= 20 and np.mean(pnls) > 0.2
    ]

    params.update({
        "min_setup_score": best_score,
        "min_composite_score": max(50, best_score - 15),
        "expectancy_threshold": round(max(0, best_exp), 3),
        "trained_on_simulations": len(results),
        "trained_real_trades": len(real),
        "trained_win_rate": round(win_rate * 100, 1),
        "trained_expectancy": round(float(expectancy), 3),
        "strong_tickers": strong_tickers[:20],
        "score_expectancy_map": _score_map(real),
        "setup_performance": setup_performance,
        "enabled_setups": enabled or list(setup_performance.keys()),
    })

    # Confidence tiers gate on REAL trade counts (synthetic resamples used to
    # be able to satisfy these thresholds regardless of evidence).
    if win_rate >= 0.55 and len(real) >= 500:
        params["min_probability_confidence"] = "MEDIUM"
    if win_rate >= 0.60 and len(real) >= 2000:
        params["min_probability_confidence"] = "HIGH"

    return params


def _score_map(results: list[dict]) -> dict:
    buckets: dict[str, list] = {}
    for r in results:
        key = str((r["setup_score"] // 10) * 10)
        buckets.setdefault(key, []).append(r["pnl_r"])
    return {
        k: round(float(np.mean(v)), 3)
        for k, v in sorted(buckets.items())
        if len(v) >= 5
    }