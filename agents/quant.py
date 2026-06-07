"""Quant agent: statistical metrics and signal quality."""

from __future__ import annotations

import numpy as np


def analyze_quant_signal(setup: dict, context: dict, probability: dict, regime: dict) -> dict:
    """Compute quantitative signal quality score."""
    factors = {
        "setup_score": setup.get("confidence_score", 0) / 100,
        "context_score": context.get("context_score", 0) / 100,
        "probability_score": probability.get("probability_score", 0) / 100,
        "regime_score": regime.get("regime_score", 50) / 100,
        "expectancy": _norm(probability.get("expectancy", 0), -1, 2),
        "sample_size": _norm(probability.get("sample_size", 0), 0, 20),
    }

    weights = {
        "setup_score": 0.30,
        "context_score": 0.20,
        "probability_score": 0.20,
        "regime_score": 0.15,
        "expectancy": 0.10,
        "sample_size": 0.05,
    }

    signal_quality = sum(factors[k] * weights[k] for k in weights) * 100
    edge = probability.get("expectancy", 0)

    return {
        "signal_quality": round(signal_quality, 1),
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "edge_estimate": round(edge, 3),
        "kelly_fraction": round(_kelly(probability) * 100, 1),
        "summary": (
            f"Signal quality {signal_quality:.0f}/100, "
            f"edge {edge:+.2f}R, Kelly { _kelly(probability)*100:.0f}%"
        ),
    }


def summarize_backtest(results: list[dict]) -> dict:
    """Quant metrics from training/backtest results."""
    if not results:
        return {}

    pnls = [r["pnl_r"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    sharpe = (np.mean(pnls) / np.std(pnls)) * (252 ** 0.5) if np.std(pnls) > 0 else 0

    return {
        "profit_factor": round(float(profit_factor), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_winner_r": round(float(avg_win), 2),
        "avg_loser_r": round(float(avg_loss), 2),
        "max_consecutive_losses": _max_consecutive_losses(pnls),
    }


def _kelly(probability: dict) -> float:
    wr = probability.get("win_rate", 0) / 100
    aw = probability.get("avg_winner_r", 1)
    al = probability.get("avg_loser_r", 1) or 1
    if al == 0 or aw == 0:
        return 0
    kelly = wr - ((1 - wr) / (aw / al))
    return max(0, min(0.25, kelly * 0.5))


def _norm(val: float, lo: float, hi: float) -> float:
    return max(0, min(1, (val - lo) / (hi - lo)))


def _max_consecutive_losses(pnls: list[float]) -> int:
    max_streak = streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak