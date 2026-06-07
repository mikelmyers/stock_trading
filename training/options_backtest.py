"""Options backtest: simulate defined-risk spreads on historical stock paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.options import _historical_volatility


def simulate_spread_on_path(
    entry_price: float,
    forward_df: pd.DataFrame,
    bias: str = "bullish",
    max_risk: float = 10.0,
    hold_days: int = 14,
) -> dict:
    """
    Model a vertical spread using stock path + HV (no historical chain needed).
    Spread width = 5% of price; max loss capped at max_risk.
    """
    if forward_df.empty:
        return {"pnl": 0, "won": False, "r_multiple": 0}

    width = entry_price * 0.05
    hv = 0.35
    forward = forward_df.iloc[:hold_days]

    if bias == "bullish":
        long_strike = entry_price
        short_strike = entry_price + width
        debit = width * 0.4
        contracts = max(1, int(max_risk / (debit * 100))) if debit else 1
        max_loss = debit * 100 * contracts

        for _, bar in forward.iterrows():
            if bar["Low"] <= entry_price * 0.95:
                return {"pnl": -max_loss, "won": False, "r_multiple": -1.0, "exit": "STOP"}
        final = float(forward["Close"].iloc[-1])
        intrinsic = max(0, final - long_strike) - max(0, final - short_strike)
        pnl = (intrinsic * 100 * contracts) - (debit * 100 * contracts)
    else:
        long_strike = entry_price
        short_strike = entry_price - width
        debit = width * 0.4
        contracts = max(1, int(max_risk / (debit * 100))) if debit else 1
        max_loss = debit * 100 * contracts

        for _, bar in forward.iterrows():
            if bar["High"] >= entry_price * 1.05:
                return {"pnl": -max_loss, "won": False, "r_multiple": -1.0, "exit": "STOP"}
        final = float(forward["Close"].iloc[-1])
        intrinsic = max(0, long_strike - final) - max(0, short_strike - final)
        pnl = (intrinsic * 100 * contracts) - (debit * 100 * contracts)

    return {
        "pnl": round(pnl, 2),
        "won": pnl > 0,
        "r_multiple": round(pnl / max_risk, 2),
        "exit": "EXPIRY",
    }


def backtest_options_from_equity_results(
    history: dict[str, pd.DataFrame],
    equity_results: list[dict],
    sample_size: int = 200,
) -> dict:
    """Run options spread sims on a sample of historical equity setups."""
    if not equity_results:
        return {"count": 0, "summary": "No equity setups to model"}

    real = [r for r in equity_results if r.get("bootstrap_id", 0) == 0]
    sample = real[:sample_size] if len(real) > sample_size else real
    outcomes = []

    for result in sample:
        ticker = result.get("ticker")
        if not ticker or ticker not in history:
            continue
        df = history[ticker]
        entry_date = result.get("entry_date")
        try:
            matches = df.index[df.index.astype(str).str[:10] == entry_date[:10]]
            if matches.empty:
                continue
            idx = df.index.get_loc(matches[0])
            forward = df.iloc[idx + 1 :]
            bias = "bearish" if result.get("setup_type") == "bear_breakdown" else "bullish"
            out = simulate_spread_on_path(
                result["entry_price"], forward, bias=bias,
            )
            out["ticker"] = ticker
            out["setup_type"] = result.get("setup_type")
            outcomes.append(out)
        except Exception:
            pass

    if not outcomes:
        return {"count": 0, "summary": "Could not match options to equity paths"}

    wins = [o for o in outcomes if o["won"]]
    wr = len(wins) / len(outcomes)
    avg_pnl = float(np.mean([o["pnl"] for o in outcomes]))
    avg_r = float(np.mean([o["r_multiple"] for o in outcomes]))

    return {
        "count": len(outcomes),
        "win_rate": round(wr * 100, 1),
        "avg_pnl": round(avg_pnl, 2),
        "avg_r": round(avg_r, 3),
        "summary": (
            f"Options spreads on {len(outcomes)} setups: "
            f"{wr*100:.0f}% win, {avg_r:+.2f}R avg"
        ),
    }