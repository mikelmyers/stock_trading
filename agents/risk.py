"""Risk agent: enforces max risk and position sizing."""

from config import MAX_RISK_PER_TRADE, RISK_TIERS


def get_max_risk_for_trust(trust_score: float) -> float:
    """Return max risk dollars based on agent trust score."""
    tier = RISK_TIERS[0]
    for t in RISK_TIERS:
        if trust_score >= t["min_trust"]:
            tier = t
    return tier["max_risk"]


def get_risk_tier_label(trust_score: float) -> str:
    return next(
        (t["label"] for t in reversed(RISK_TIERS) if trust_score >= t["min_trust"]),
        "Sandbox",
    )


def calculate_risk_and_sizing(
    setup: dict, max_risk: float = MAX_RISK_PER_TRADE
) -> dict:
    """Enforce max risk using a structural stop below broken resistance."""
    entry_price = setup["current_price"]
    bias = setup.get("bias", "bullish")
    stop_loss = setup.get("stop_loss") or round(setup["resistance_level"] * 0.98, 2)
    stop_loss = round(stop_loss, 2)

    if bias == "bearish":
        risk_per_share = stop_loss - entry_price
        target_1 = round(entry_price - risk_per_share, 2)
        target_2 = round(entry_price - (risk_per_share * 2), 2)
        direction = "SHORT"
    else:
        risk_per_share = entry_price - stop_loss
        target_1 = round(entry_price + risk_per_share, 2)
        target_2 = round(entry_price + (risk_per_share * 2), 2)
        direction = "LONG"

    if risk_per_share <= 0:
        return {"action": "SKIP", "reason": "Invalid risk-to-reward metrics"}

    shares_to_buy = round(max_risk / risk_per_share, 4)
    total_capital_required = round(shares_to_buy * entry_price, 2)
    actual_risk = round(shares_to_buy * risk_per_share, 2)

    return {
        "action": "TRADE_PROPOSAL",
        "direction": direction,
        "shares": shares_to_buy,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "risk_per_share": round(risk_per_share, 2),
        "total_capital_allocated": total_capital_required,
        "max_risk_exposure": max_risk,
        "actual_risk": actual_risk,
        "target_1": target_1,
        "target_2": target_2,
        "reward_risk_ratio": 2.0,
    }