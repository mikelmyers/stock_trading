"""Margin compliance: worst-case buying power check (FINRA-style)."""

from __future__ import annotations


def analyze_margin(trade_plan: dict, account_equity: float = 2000.0) -> dict:
    """
    Estimate intraday margin usage for a proposed trade.
    Uses ±10% underlying move stress test from original design.
    """
    if trade_plan.get("action") != "TRADE_PROPOSAL":
        return {"compliant": True, "summary": "No trade proposed"}

    entry = trade_plan["entry_price"]
    shares = trade_plan["shares"]
    direction = trade_plan.get("direction", "LONG")
    notional = entry * shares
    stress_pct = 0.10

    if direction == "SHORT":
        stress_loss = entry * stress_pct * shares
    else:
        stress_loss = entry * stress_pct * shares

    margin_req = notional * 0.5 + stress_loss
    buying_power = account_equity * 2.0
    usage_pct = (margin_req / buying_power) * 100 if buying_power else 100
    compliant = usage_pct <= 50

    return {
        "compliant": compliant,
        "margin_required": round(margin_req, 2),
        "buying_power": round(buying_power, 2),
        "usage_pct": round(usage_pct, 1),
        "stress_test": "±10% underlying move",
        "summary": (
            f"Margin usage {usage_pct:.0f}% of buying power "
            f"({'OK' if compliant else 'BLOCKED — reduce size'})"
        ),
    }