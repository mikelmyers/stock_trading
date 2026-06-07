"""Bull Flag: sharp rally (flagpole) → tight consolidation → breakout."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_bull_flag(df: pd.DataFrame) -> dict:
    if len(df) < 35:
        return empty_setup("bull_flag", "Bull Flag")

    pole_start = df["Close"].iloc[-25]
    pole_end = df["Close"].iloc[-15]
    pole_move = (pole_end - pole_start) / pole_start if pole_start else 0

    flag = df.iloc[-10:]
    flag_range = (flag["High"].max() - flag["Low"].min()) / pole_end if pole_end else 1
    flag_vol_declining = flag["Volume"].iloc[-5:].mean() < flag["Volume"].iloc[-10:-5].mean()

    close = df["Close"].iloc[-1]
    flag_high = flag["High"].iloc[:-1].max()
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    strong_pole = pole_move > 0.08
    tight_flag = flag_range < 0.06
    breakout = close > flag_high
    vol_confirm = volume > avg_vol * 1.3

    score = 0
    if strong_pole:
        score += 30
    if tight_flag and flag_vol_declining:
        score += 30
    if breakout and vol_confirm:
        score += 40

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(flag["Low"].min() * 0.99, 2)

    return {
        "setup_type": "bull_flag",
        "setup_name": "Bull Flag",
        "bias": "bullish",
        "is_valid_setup": strong_pole and tight_flag and breakout and vol_confirm,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(flag_high, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": "Flagpole rally, tight flag consolidation, volume breakout",
    }