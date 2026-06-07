"""RSI Bullish Divergence: price lower low, RSI higher low in uptrend."""

import pandas as pd

from agents.indicators import calculate_atr, calculate_rsi
from agents.setups.base import empty_setup


def analyze_rsi_divergence(df: pd.DataFrame) -> dict:
    if len(df) < 40:
        return empty_setup("rsi_divergence", "RSI Divergence")

    df = df.copy()
    df["RSI"] = calculate_rsi(df, 14)
    df["SMA_50"] = df["Close"].rolling(50).mean()

    recent = df.iloc[-25:]
    mid = len(recent) // 2

    first_low_idx = recent.iloc[:mid]["Low"].idxmin()
    second_low_idx = recent.iloc[mid:]["Low"].idxmin()

    price_ll = recent.loc[second_low_idx, "Low"] < recent.loc[first_low_idx, "Low"]
    rsi_hl = recent.loc[second_low_idx, "RSI"] > recent.loc[first_low_idx, "RSI"]

    close = df["Close"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]
    uptrend = close > sma50 if pd.notna(sma50) else True
    rsi_now = df["RSI"].iloc[-1]
    recovering = rsi_now > 35 and rsi_now > df["RSI"].iloc[-3]

    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    score = 0
    if price_ll and rsi_hl:
        score += 40
    if uptrend:
        score += 25
    if recovering:
        score += 20
    if volume > avg_vol:
        score += 15

    recent_high = df["High"].iloc[-15:].max()
    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(df["Low"].iloc[-5:].min() * 0.98, 2)

    return {
        "setup_type": "rsi_divergence",
        "setup_name": "RSI Divergence",
        "bias": "bullish",
        "is_valid_setup": price_ll and rsi_hl and uptrend and recovering,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(recent_high, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": f"Bullish RSI divergence, RSI now {rsi_now:.0f}",
    }