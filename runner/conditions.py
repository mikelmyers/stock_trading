"""Condition-vector for low-float momentum runner setups.

The feature snapshot recorded for every scanned candidate — the data the
classifier learns from. Seed thresholds are PRIORS the classifier refines as it
accumulates its own outcomes; nothing here is a hard truth. Two gates:
  * is_candidate — LOOSE; defines the pool we log (incl. near-misses, so the
    classifier learns the full spectrum, not just winners).
  * green_light  — the A+ cluster prior ("push" candidates).
Plus blowup_flags — the conditions that historically turn a -1R into a disaster.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

# --- loose candidate gate (what enters the log) ---
CAND_PRICE_MIN, CAND_PRICE_MAX = 0.50, 20.0
CAND_RVOL_MIN = 2.0
CAND_GAP_MIN = 5.0           # percent
# --- A+ green-light prior ("push" cluster) ---
AP_FLOAT_MAX = 20_000_000
AP_RVOL_MIN = 5.0
AP_GAP_MIN = 15.0
# --- blow-up flag thresholds ---
EXTENDED_PCT = 50.0          # >50% above VWAP = parabolic / chase risk
WIDE_SPREAD_PCT = 1.0        # >1% spread = slippage eats you
LATE_MINUTES = 120           # gap-and-go degrades after first ~2h
REGULAR_SESSION_MIN = 390


@dataclass
class ConditionVector:
    symbol: str
    asof: str
    # identity / static
    price: Optional[float] = None
    float_shares: Optional[float] = None
    market_cap: Optional[float] = None
    avg_vol_20d: Optional[float] = None
    sector: Optional[str] = None
    # in-play / volume
    rvol: Optional[float] = None
    gap_pct: Optional[float] = None
    premarket_vol: Optional[float] = None
    vol_today: Optional[float] = None
    vol_to_float: Optional[float] = None
    gap_atr: Optional[float] = None
    # momentum / price action
    pct_change: Optional[float] = None
    vwap: Optional[float] = None
    dist_vwap_pct: Optional[float] = None
    vwap_slope: Optional[float] = None
    dist_pm_high_pct: Optional[float] = None
    dist_pm_low_pct: Optional[float] = None
    extension_pct: Optional[float] = None
    # catalyst
    has_news: bool = False
    catalyst_type: Optional[str] = None
    # tradability / risk
    spread_pct: Optional[float] = None
    halts_today: Optional[int] = None
    atr_pct: Optional[float] = None
    # context / timing
    minutes_since_open: Optional[float] = None
    market_regime: Optional[str] = None
    # derived (filled by classify)
    is_candidate: bool = False
    green_light: bool = False
    blowup_flags: str = ""

    def to_row(self) -> dict:
        return asdict(self)


# ---------- feature helpers (raw market data -> numbers) ----------
def vwap(bars) -> Optional[float]:
    """Volume-weighted average of typical price over intraday bars
    (each bar: dict/obj with high, low, close, volume)."""
    num = den = 0.0
    for b in bars:
        v = float(b["volume"]); tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3
        num += tp * v; den += v
    return num / den if den else None


def rvol(vol_today: float, avg_vol_20d: float, minutes_since_open: float) -> Optional[float]:
    """Today's volume vs the volume you'd EXPECT by this time of day."""
    if not avg_vol_20d or minutes_since_open <= 0:
        return None
    frac = min(minutes_since_open, REGULAR_SESSION_MIN) / REGULAR_SESSION_MIN
    expected = avg_vol_20d * frac
    return vol_today / expected if expected else None


def pct(a: float, b: float) -> Optional[float]:
    return (a - b) / b * 100 if b else None


def classify(cv: ConditionVector) -> ConditionVector:
    """Set is_candidate, green_light, and blowup_flags from the seed priors."""
    cv.is_candidate = bool(
        cv.price is not None and CAND_PRICE_MIN <= cv.price <= CAND_PRICE_MAX
        and (cv.rvol or 0) >= CAND_RVOL_MIN
        and (cv.gap_pct or 0) >= CAND_GAP_MIN
    )
    flags = []
    if cv.catalyst_type == "offering":
        flags.append("dilution")
    if (cv.extension_pct or 0) > EXTENDED_PCT:
        flags.append("extended")
    if (cv.spread_pct or 0) > WIDE_SPREAD_PCT:
        flags.append("wide_spread")
    if (cv.minutes_since_open or 0) > LATE_MINUTES:
        flags.append("late_day")
    if not cv.has_news:
        flags.append("no_catalyst")
    if (cv.halts_today or 0) >= 2:
        flags.append("multi_halt")
    cv.blowup_flags = ",".join(flags)

    cv.green_light = bool(
        cv.is_candidate
        and (cv.float_shares is None or cv.float_shares <= AP_FLOAT_MAX)
        and (cv.rvol or 0) >= AP_RVOL_MIN
        and (cv.gap_pct or 0) >= AP_GAP_MIN
        and cv.has_news
        and (cv.dist_vwap_pct or -1) > 0           # holding above VWAP
        and not flags                               # no blow-up flag present
    )
    return cv
