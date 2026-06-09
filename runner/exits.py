"""Intraday exit logic for runner trades — momentum-scalp rules, NOT swing.

Runners reverse fast and you never hold low-float overnight, so the exits are
tight and time-bounded: scale 1/3 at +1R and +2R, trail the runner hard, cut the
moment momentum breaks (close back below VWAP), −1R hard stop, and a non-negotiable
flat-by-close. Stateless replay over the intraday bars since entry (same robustness
trick as the swing manager); long-side (these setups are bullish momentum).
"""
from __future__ import annotations

SCALE_PCT = 0.33
TRAIL_R_MULT = 1.0          # trail 1R below the high-water mark (tighter than swing)
FLAT_CLOSE_MIN = 5          # exit everything within 5 min of the close


def plan_exit(entry: float, stop: float, bars: list[dict], minutes_to_close: float):
    """bars: intraday bars since entry, each {high, low, close, vwap}.
    Returns (exit_all, reason, remaining_fraction)."""
    risk = entry - stop
    if risk <= 0:
        return (True, "BAD_RISK", 0.0)
    t1, t2 = entry + risk, entry + 2 * risk
    scale1 = scale2 = armed = False
    frac, hwm, trail = 1.0, entry, stop
    for b in bars:
        high, low, close, vwap = (float(b["high"]), float(b["low"]),
                                  float(b["close"]), b.get("vwap"))
        if low <= stop:
            return (True, "HARD_STOP", 0.0)
        if close >= t1 and not scale1:
            scale1 = armed = True; frac -= SCALE_PCT; hwm = max(hwm, high)
        if close >= t2 and not scale2:
            scale2 = True; frac -= SCALE_PCT; hwm = max(hwm, high)
        if armed:
            hwm = max(hwm, high)
            trail = max(trail, hwm - TRAIL_R_MULT * risk)
            if low <= trail and trail > stop:
                return (True, "TRAIL", 0.0)
        if vwap is not None and close < vwap:        # momentum broke -> cut remainder
            return (True, "VWAP_LOSS", 0.0)
    if minutes_to_close <= FLAT_CLOSE_MIN:
        return (True, "FLAT_CLOSE", 0.0)
    return (False, "HOLD", round(frac, 2))
