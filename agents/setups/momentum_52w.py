"""52-week-high momentum: buy proven leaders near new highs, not short-term pops.

Distinct from ``breakout`` (a short-term volatility squeeze, which backtests
negative on large caps) and from the mean-reversion-heavy rest of the book. This
is the slow, well-documented momentum anomaly (Jegadeesh-Titman 12-1 momentum;
George-Hwang 52-week-high): stocks already in a strong intermediate uptrend and
trading near their 52-week high tend to keep outperforming. Causal — every value
uses only data through the current bar.
"""

import pandas as pd

from agents.indicators import calculate_atr, calculate_rsi
from agents.setups.base import empty_setup


def analyze_momentum_52w(df: pd.DataFrame) -> dict:
    if len(df) < 252:
        return empty_setup("momentum_52w", "52-Week High Momentum")

    close = df["Close"]
    c = float(close.iloc[-1])
    high_252 = float(df["High"].iloc[-252:].max())
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    ret_126 = c / float(close.iloc[-126]) - 1.0          # ~6-month return
    atr14 = float(calculate_atr(df, 14).iloc[-1])
    rsi14 = float(calculate_rsi(df, 14).iloc[-1])
    ret_5 = c / float(close.iloc[-5]) - 1.0

    near_high = c >= 0.95 * high_252        # within 5% of the 52-week high
    strong_mom = ret_126 > 0.10             # real intermediate momentum
    uptrend = c > sma50 > sma200            # stacked moving averages
    not_blowoff = rsi14 < 85 and ret_5 < 0.15   # not a parabolic last-week pop

    valid = bool(near_high and strong_mom and uptrend and not_blowoff)

    # ATR stop, consistent with the rest of the book (risk = 2 ATR).
    stop = round(c - 2.0 * atr14, 2)

    # Confidence: blend proximity-to-high and momentum strength into 0-100.
    score = 0
    if near_high:
        score += int(min(40, 40 * (c / high_252)))     # closer to high = better
    if strong_mom:
        score += int(min(35, 35 * (ret_126 / 0.30)))   # capped at +30% 6mo
    if uptrend:
        score += 25

    out = empty_setup("momentum_52w", "52-Week High Momentum")
    out.update({
        "bias": "bullish",
        "is_valid_setup": valid,
        "confidence_score": int(min(100, score)),
        "current_price": round(c, 2),
        "resistance_level": round(high_252, 2),
        "stop_loss": stop,
        "atr_14": round(atr14, 2),
        "ret_126": round(ret_126, 4),
        "dist_to_high_pct": round((c / high_252 - 1.0) * 100, 2),
        "volume_ratio": 0,
        "details": "Leader near 52-week high with strong 6-month momentum",
    })
    return out
