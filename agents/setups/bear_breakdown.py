"""Bear Breakdown: support break on volume in downtrend (short bias)."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_bear_breakdown(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return empty_setup("bear_breakdown", "Bear Breakdown")

    df = df.copy()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    support = df["Low"].iloc[-20:-1].min()
    close = df["Close"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    downtrend = close < sma50 if pd.notna(sma50) else False
    broke_support = close < support * 0.995
    vol_surge = volume > avg_vol * 1.5

    score = 0
    if downtrend:
        score += 30
    if broke_support:
        score += 40
    if vol_surge:
        score += 30

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(support * 1.02, 2)
    target = round(close - (stop - close), 2)

    return {
        "setup_type": "bear_breakdown",
        "setup_name": "Bear Breakdown",
        "bias": "bearish",
        "is_valid_setup": downtrend and broke_support and vol_surge,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(stop, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": f"Support break at ${support:.2f} in downtrend, short bias",
    }