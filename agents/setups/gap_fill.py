"""Gap Fill: gap down → price recovering to fill the gap."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_gap_fill(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return empty_setup("gap_fill", "Gap Fill")

    prev_close = df["Close"].iloc[-2]
    today_open = df["Open"].iloc[-1]
    close = df["Close"].iloc[-1]
    low = df["Low"].iloc[-1]

    gap_pct = (today_open - prev_close) / prev_close if prev_close else 0
    gap_down = gap_pct < -0.02
    gap_top = prev_close
    filling = close > today_open and close < gap_top
    partial_fill = (close - today_open) / (gap_top - today_open) if gap_top != today_open else 0

    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()
    vol_rising = volume > avg_vol * 1.1

    score = 0
    if gap_down:
        score += 30
    if filling and partial_fill > 0.4:
        score += 35
    if vol_rising:
        score += 20
    if close > df["Open"].iloc[-1]:
        score += 15

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(low * 0.99, 2)

    return {
        "setup_type": "gap_fill",
        "setup_name": "Gap Fill",
        "bias": "bullish",
        "is_valid_setup": gap_down and filling and partial_fill > 0.4 and vol_rising,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(gap_top, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": f"Gap-down fill {partial_fill*100:.0f}% toward ${gap_top:.2f}",
    }