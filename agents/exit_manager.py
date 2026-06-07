"""Exit engine: trailing stops, scale-out ladder, and time-based exits."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from agents.indicators import calculate_atr
from config import (
    MAX_HOLDING_DAYS,
    MIN_PROFIT_BY_DAY,
    SCALE_OUT_LEVELS,
    TIME_STOP_DAYS,
    TRAILING_STOP_ATR_MULT,
)
from state import ActivePosition


def build_exit_plan(setup: dict, trade_plan: dict) -> dict:
    """Generate the trade management sheet before entry."""
    entry = trade_plan["entry_price"]
    stop = trade_plan["stop_loss"]
    risk = trade_plan["risk_per_share"]
    atr = setup.get("atr_14", risk)

    scale_targets = []
    for level in SCALE_OUT_LEVELS:
        target_price = round(entry + (risk * level["r_multiple"]), 2)
        shares_at_level = round(trade_plan["shares"] * level["pct_to_sell"], 4)
        scale_targets.append({
            "label": level["label"],
            "price": target_price,
            "r_multiple": level["r_multiple"],
            "shares_to_sell": shares_at_level,
            "pct_position": level["pct_to_sell"],
        })

    initial_trail = round(entry - (atr * TRAILING_STOP_ATR_MULT), 2)

    return {
        "hard_stop": stop,
        "initial_trailing_stop": max(stop, initial_trail),
        "scale_out_ladder": scale_targets,
        "max_hold_days": MAX_HOLDING_DAYS,
        "time_stop_day": TIME_STOP_DAYS,
        "time_stop_rule": (
            f"If <25% of max profit realized by day {TIME_STOP_DAYS}, exit at market"
        ),
        "runner_rule": (
            f"After scale-outs, trail remainder at {TRAILING_STOP_ATR_MULT}x ATR "
            f"(${atr}) below high water mark"
        ),
    }


def evaluate_position(
    position: ActivePosition,
    df: pd.DataFrame,
) -> dict:
    """
    Daily evaluation of an open position.
    Returns action dict: HOLD, SCALE_OUT, TRAIL_UPDATE, or EXIT.
    """
    if len(df) < 15:
        return {"action": "HOLD", "reason": "Insufficient data"}

    current_close = df["Close"].iloc[-1]
    current_low = df["Low"].iloc[-1]
    atr = calculate_atr(df, window=14).iloc[-1]
    risk = position.risk_per_share

    days_held = _days_since(position.entry_date)
    unrealized_r = (current_close - position.entry_price) / risk if risk > 0 else 0
    max_theoretical_profit = risk * position.shares
    current_profit = (current_close - position.entry_price) * position.shares_remaining
    profit_pct_of_max = (
        current_profit / max_theoretical_profit if max_theoretical_profit > 0 else 0
    )

    # 1. Hard stop
    if current_low <= position.stop_loss:
        return {
            "action": "EXIT",
            "reason": "HARD_STOP",
            "exit_price": position.stop_loss,
            "message": f"Stop hit at ${position.stop_loss}",
        }

    # 2. Trailing stop (only active after first scale-out or 1R profit)
    new_hwm = max(position.high_water_mark, current_close)
    if position.scale_outs_hit or unrealized_r >= 1.0:
        new_trail = round(new_hwm - (atr * TRAILING_STOP_ATR_MULT), 2)
        new_trail = max(new_trail, position.stop_loss)
        if current_low <= position.trailing_stop and position.trailing_stop > position.stop_loss:
            return {
                "action": "EXIT",
                "reason": "TRAILING_STOP",
                "exit_price": position.trailing_stop,
                "message": f"Trailing stop hit at ${position.trailing_stop}",
            }
    else:
        new_trail = position.trailing_stop
        new_hwm = position.high_water_mark

    # 3. Scale-out ladder
    for level in SCALE_OUT_LEVELS:
        label = level["label"]
        if label in position.scale_outs_hit:
            continue
        target = position.entry_price + (risk * level["r_multiple"])
        if current_close >= target:
            shares_to_sell = round(position.shares * level["pct_to_sell"], 4)
            shares_to_sell = min(shares_to_sell, position.shares_remaining)
            return {
                "action": "SCALE_OUT",
                "reason": label,
                "exit_price": round(target, 2),
                "shares_to_sell": shares_to_sell,
                "shares_remaining_after": round(
                    position.shares_remaining - shares_to_sell, 4
                ),
                "message": (
                    f"Hit {label} at ${target:.2f} — sell {shares_to_sell} shares "
                    f"({level['pct_to_sell']*100:.0f}% of position)"
                ),
                "new_high_water_mark": new_hwm,
                "new_trailing_stop": max(new_trail, position.stop_loss),
            }

    # 4. Time-based exit
    if days_held >= MAX_HOLDING_DAYS:
        return {
            "action": "EXIT",
            "reason": "MAX_HOLD_TIME",
            "exit_price": current_close,
            "message": f"Max hold period ({MAX_HOLDING_DAYS} days) reached",
        }

    for day_threshold, min_profit in sorted(MIN_PROFIT_BY_DAY.items()):
        if days_held >= day_threshold and profit_pct_of_max < min_profit:
            return {
                "action": "EXIT",
                "reason": "TIME_STOP",
                "exit_price": current_close,
                "message": (
                    f"Day {day_threshold}: only {profit_pct_of_max*100:.0f}% of max profit "
                    f"(need {min_profit*100:.0f}%) — cut the dead weight"
                ),
            }

    # 5. Trail update for runner
    trail_update = None
    if new_hwm > position.high_water_mark or new_trail > position.trailing_stop:
        trail_update = {
            "high_water_mark": new_hwm,
            "trailing_stop": new_trail,
        }

    return {
        "action": "HOLD",
        "reason": "WITHIN_PLAN",
        "current_price": round(current_close, 2),
        "unrealized_r": round(unrealized_r, 2),
        "days_held": days_held,
        "profit_pct_of_max": round(profit_pct_of_max * 100, 1),
        "trail_update": trail_update,
        "message": (
            f"Holding {position.ticker}: {unrealized_r:+.1f}R, "
            f"day {days_held}, trail ${new_trail:.2f}"
        ),
    }


def _days_since(date_str: str) -> int:
    entry = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - entry).days