"""Teacher agent: post-trade review and feedback calibration."""

from agents.risk import get_max_risk_for_trust, get_risk_tier_label
from config import MIN_TRUST_SCORE_FOR_SCALE_UP
from state import AgentState, ClosedTrade


def generate_trade_review(trade: ClosedTrade, state: AgentState) -> dict:
    """Post-trade analysis for trust score calibration."""
    win_rate = state.wins / state.total_trades if state.total_trades else 0
    avg_r = (
        sum(t.r_multiple for t in state.trade_history) / len(state.trade_history)
        if state.trade_history
        else 0
    )

    lessons = []
    if trade.r_multiple >= 2.0:
        lessons.append("Textbook runner — setup fidelity validated.")
    elif trade.r_multiple > 0:
        lessons.append("Winner, but left money on table — review scale-out timing.")
    elif trade.exit_reason == "HARD_STOP":
        lessons.append("Stopped out — check if breakout was a fakeout (volume quality).")
    elif trade.exit_reason == "TIME_STOP":
        lessons.append("No follow-through — consolidation may have been too loose.")
    elif trade.exit_reason == "MANUAL":
        lessons.append("Manual exit — record whether plan was followed.")
    else:
        lessons.append("Review entry criteria; setup may not have matched archetype.")

    if trade.user_feedback:
        lessons.append(f"Your notes: {trade.user_feedback}")

    return {
        "ticker": trade.ticker,
        "result": "WIN" if trade.pnl > 0 else "LOSS",
        "r_multiple": trade.r_multiple,
        "pnl": trade.pnl,
        "exit_reason": trade.exit_reason,
        "trust_score": state.trust_score,
        "risk_tier": get_risk_tier_label(state.trust_score),
        "max_risk_allowed": get_max_risk_for_trust(state.trust_score),
        "win_rate": round(win_rate * 100, 1),
        "avg_r": round(avg_r, 2),
        "lessons": lessons,
        "scale_up_eligible": state.trust_score >= MIN_TRUST_SCORE_FOR_SCALE_UP,
    }


def format_feedback_prompt(ticker: str) -> str:
    return (
        f"Rate the {ticker} trade:\n"
        "  --fidelity 0.0-1.0  (how clean was the setup?)\n"
        "  --aligned yes/no    (did you follow the exit plan?)\n"
        "  --notes \"...\"       (optional notes)"
    )