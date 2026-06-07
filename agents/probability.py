"""Probability engine: historical breakout outcome simulation."""

import numpy as np
import pandas as pd

from agents.indicators import calculate_atr


def _detect_historical_breakouts(df: pd.DataFrame) -> list[int]:
    """Return indices where a breakout pattern occurred historically."""
    breakouts = []
    if len(df) < 30:
        return breakouts

    for i in range(25, len(df) - 5):
        window = df.iloc[: i + 1]
        recent_max = window["High"].iloc[-21:-1].max()
        avg_vol = window["Volume"].iloc[-21:-1].mean()
        close = window["Close"].iloc[-1]
        volume = window["Volume"].iloc[-1]

        atr_20 = calculate_atr(window, 20).iloc[-1]
        atr_5 = calculate_atr(window, 5).iloc[-1]
        compressed = atr_5 < atr_20 * 0.85

        if close > recent_max and volume > avg_vol * 1.5 and compressed:
            breakouts.append(i)

    return breakouts


def _outcome_after_breakout(
    df: pd.DataFrame, idx: int, hold_days: int, stop_pct: float = 0.02
) -> dict:
    entry = df["Close"].iloc[idx]
    resistance = df["High"].iloc[idx - 1]
    stop = resistance * (1 - stop_pct)
    risk = entry - stop
    if risk <= 0:
        risk = entry * 0.02

    forward = df.iloc[idx + 1 : idx + 1 + hold_days]
    if forward.empty:
        return {"won": False, "r_multiple": 0.0}

    max_high = forward["High"].max()
    min_low = forward["Low"].min()

    if min_low <= stop:
        return {"won": False, "r_multiple": -1.0}

    r_mult = (max_high - entry) / risk
    won = r_mult >= 1.0
    return {"won": won, "r_multiple": round(r_mult, 2)}


def estimate_breakout_probability(df: pd.DataFrame, hold_days: int = 10) -> dict:
    """
    Backtest similar breakout setups on this ticker's history.
    Returns conditional probability profile.
    """
    breakouts = _detect_historical_breakouts(df)

    if not breakouts:
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "avg_winner_r": 0.0,
            "avg_loser_r": 0.0,
            "expectancy": 0.0,
            "probability_score": 0,
            "confidence": "LOW",
            "summary": "Insufficient historical breakout samples.",
        }

    outcomes = [_outcome_after_breakout(df, idx, hold_days) for idx in breakouts]
    wins = [o for o in outcomes if o["won"]]
    losses = [o for o in outcomes if not o["won"]]

    win_rate = len(wins) / len(outcomes)
    avg_win_r = np.mean([o["r_multiple"] for o in wins]) if wins else 0.0
    avg_loss_r = np.mean([abs(o["r_multiple"]) for o in losses]) if losses else 1.0
    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)

    prob_score = min(100, int(win_rate * 60 + max(0, expectancy) * 20 + min(len(outcomes), 10) * 2))

    if len(outcomes) >= 8 and win_rate >= 0.55:
        confidence = "HIGH"
    elif len(outcomes) >= 4 and win_rate >= 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "sample_size": len(outcomes),
        "win_rate": round(win_rate * 100, 1),
        "avg_winner_r": round(float(avg_win_r), 2),
        "avg_loser_r": round(float(avg_loss_r), 2),
        "expectancy": round(float(expectancy), 2),
        "probability_score": prob_score,
        "confidence": confidence,
        "summary": (
            f"Historical breakouts: {len(outcomes)} samples, "
            f"{win_rate*100:.0f}% reached 1R within {hold_days} days."
        ),
    }