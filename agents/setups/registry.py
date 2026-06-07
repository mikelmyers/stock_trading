"""Setup registry: add new patterns here to expand the agent."""

from __future__ import annotations

import pandas as pd

from agents.setups.bear_breakdown import analyze_bear_breakdown
from agents.setups.credit_put import analyze_credit_put
from agents.setups.breakout import analyze_breakout
from agents.setups.bull_flag import analyze_bull_flag
from agents.setups.double_bottom import analyze_double_bottom
from agents.setups.gap_fill import analyze_gap_fill
from agents.setups.ma_pullback import analyze_ma_pullback
from agents.setups.mean_reversion import analyze_mean_reversion
from agents.setups.rsi_divergence import analyze_rsi_divergence
from agents.setups.vwap_reclaim import analyze_vwap_reclaim
from training.calibrator import load_learned_params

SETUP_REGISTRY: dict[str, callable] = {
    "breakout": analyze_breakout,
    "ma_pullback": analyze_ma_pullback,
    "bull_flag": analyze_bull_flag,
    "double_bottom": analyze_double_bottom,
    "mean_reversion": analyze_mean_reversion,
    "gap_fill": analyze_gap_fill,
    "vwap_reclaim": analyze_vwap_reclaim,
    "rsi_divergence": analyze_rsi_divergence,
    "bear_breakdown": analyze_bear_breakdown,
    "credit_put": analyze_credit_put,
}

SETUP_CATEGORIES = {
    "momentum": ["breakout", "bull_flag", "vwap_reclaim"],
    "pullback": ["ma_pullback", "mean_reversion", "rsi_divergence"],
    "reversal": ["double_bottom", "gap_fill"],
    "bearish": ["bear_breakdown"],
    "options_style": ["credit_put"],
}


def get_enabled_setups() -> dict[str, callable]:
    learned = load_learned_params()
    enabled = learned.get("enabled_setups")
    if not enabled:
        return SETUP_REGISTRY
    return {k: v for k, v in SETUP_REGISTRY.items() if k in enabled}


def analyze_all_setups(df: pd.DataFrame, enabled_only: bool = True) -> list[dict]:
    registry = get_enabled_setups() if enabled_only else SETUP_REGISTRY
    results = []
    for name, analyzer in registry.items():
        try:
            setup = analyzer(df)
            setup["setup_type"] = name
            results.append(setup)
        except Exception:
            pass
    return results


def get_best_setup(df: pd.DataFrame, enabled_only: bool = True) -> dict:
    all_setups = analyze_all_setups(df, enabled_only=enabled_only)
    valid = [s for s in all_setups if s["is_valid_setup"]]
    if valid:
        return max(valid, key=lambda s: s["confidence_score"])
    if all_setups:
        return max(all_setups, key=lambda s: s["confidence_score"])
    return {
        "setup_type": "none",
        "setup_name": "None",
        "bias": "neutral",
        "is_valid_setup": False,
        "confidence_score": 0,
        "current_price": 0,
        "resistance_level": 0,
        "stop_loss": 0,
        "atr_14": 0,
        "details": "No setup detected",
    }