"""Double Bottom: two lows at support → neckline break."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_double_bottom(df: pd.DataFrame) -> dict:
    if len(df) < 45:
        return empty_setup("double_bottom", "Double Bottom")

    window = df.iloc[-40:]
    lows = window["Low"]
    first_low_idx = lows.idxmin()
    first_low = lows.min()

    remaining = window.loc[first_low_idx:]
    if len(remaining) < 15:
        return empty_setup("double_bottom", "Double Bottom")

    second_low = remaining.iloc[5:]["Low"].min()
    tolerance = first_low * 0.02
    matched_lows = abs(second_low - first_low) <= tolerance

    neckline = window["High"].max()
    close = df["Close"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    neckline_break = close > neckline * 0.998
    vol_confirm = volume > avg_vol * 1.4

    score = 0
    if matched_lows:
        score += 35
    if neckline_break:
        score += 35
    if vol_confirm:
        score += 30

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(min(first_low, second_low) * 0.97, 2)

    return {
        "setup_type": "double_bottom",
        "setup_name": "Double Bottom",
        "bias": "bullish",
        "is_valid_setup": matched_lows and neckline_break and vol_confirm,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(neckline, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": "Two lows at support, neckline breakout on volume",
    }