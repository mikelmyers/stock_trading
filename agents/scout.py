"""Scout agent: dispatches to all registered setup patterns."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.registry import analyze_all_setups, get_best_setup


def analyze_breakout_setup(df: pd.DataFrame) -> dict:
    """Backward-compatible: returns best setup across all patterns."""
    return get_best_setup(df)


def scan_all_setups(df: pd.DataFrame) -> list[dict]:
    """Return all setup evaluations for a ticker."""
    return analyze_all_setups(df)