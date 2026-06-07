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


def _setup_stats(results: list[dict], setup_type: str) -> dict:
    real = [r for r in results if r.get("setup_type") == setup_type and r.get("bootstrap_id", 0) == 0]
    all_type = [r for r in results if r.get("setup_type") == setup_type]
    if not all_type:
        return {"real_count": 0, "count": 0, "win_rate": 0, "expectancy": 0, "enabled": False}

    wins = [r for r in all_type if r["won"]]
    wr = len(wins) / len(all_type)
    aw = np.mean([r["pnl_r"] for r in wins]) if wins else 0
    al = np.mean([abs(r["pnl_r"]) for r in all_type if not r["won"]]) or 1
    exp = (wr * aw) - ((1 - wr) * al)
    if not np.isfinite(exp):
        exp = float(aw) if wr >= 1.0 else 0.0

    return {
        "real_count": len(real),
        "count": len(all_type),
        "win_rate": round(wr * 100, 1),
        "expectancy": round(float(exp), 3),
        "enabled": len(real) >= 5 and exp > 0,
    }


def calibrate(results: list[dict]) -> dict:
    """
    Learn optimal thresholds and which setup types have positive edge.
    Disables losing patterns, enables winning ones.
    """
    params = load_learned_params()

    if not results:
        return params

    wins = [r for r in results if r["won"]]
    win_rate = len(wins) / len(results)
    avg_win = np.mean([r["pnl_r"] for r in wins]) if wins else 0
    avg_loss = np.mean([abs(r["pnl_r"]) for r in results if not r["won"]]) or 1
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    setup_types = sorted({r.get("setup_type", "unknown") for r in results})
    setup_performance = {
        st: _setup_stats(results, st) for st in setup_types
    }

    enabled = [st for st, perf in setup_performance.items() if perf["enabled"]]
    if not enabled:
        best = max(setup_performance.items(), key=lambda x: x[1]["expectancy"], default=(None, {}))
        if best[0] and best[1].get("real_count", 0) >= 1:
            enabled = [best[0]]

    best_score = 70
    best_exp = -999.0
    enabled_results = [r for r in results if r.get("setup_type") in enabled] if enabled else results

    for cutoff in range(50, 101, 10):
        subset = [r for r in enabled_results if r["setup_score"] >= cutoff]
        if len(subset) < 20:
            continue
        sub_wins = [r for r in subset if r["won"]]
        sub_wr = len(sub_wins) / len(subset)
        sub_aw = np.mean([r["pnl_r"] for r in sub_wins]) if sub_wins else 0
        sub_al = np.mean([abs(r["pnl_r"]) for r in subset if not r["won"]]) or 1
        sub_exp = (sub_wr * sub_aw) - ((1 - sub_wr) * sub_al)
        if sub_exp > best_exp:
            best_exp = sub_exp
            best_score = cutoff

    by_ticker: dict[str, list] = {}
    for r in results:
        if r.get("bootstrap_id", 0) == 0:
            by_ticker.setdefault(r["ticker"], []).append(r["pnl_r"])

    strong_tickers = [
        t for t, pnls in by_ticker.items()
        if len(pnls) >= 3 and np.mean(pnls) > 0.2
    ]

    params.update({
        "min_setup_score": best_score,
        "min_composite_score": max(50, best_score - 15),
        "expectancy_threshold": round(max(0, best_exp), 3),
        "trained_on_simulations": len(results),
        "trained_win_rate": round(win_rate * 100, 1),
        "trained_expectancy": round(float(expectancy), 3),
        "strong_tickers": strong_tickers[:20],
        "score_expectancy_map": _score_map(results),
        "setup_performance": setup_performance,
        "enabled_setups": enabled or list(setup_performance.keys()),
    })

    if win_rate >= 0.55 and len(results) >= 500:
        params["min_probability_confidence"] = "MEDIUM"
    if win_rate >= 0.60 and len(results) >= 2000:
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