"""Mean Reversion: oversold RSI in an uptrend → bounce."""

import pandas as pd

from agents.indicators import calculate_atr, calculate_rsi
from agents.setups.base import empty_setup


def analyze_mean_reversion(df: pd.DataFrame) -> dict:
    if len(df) < 30:
        return empty_setup("mean_reversion", "Mean Reversion")

    df = df.copy()
    df["RSI"] = calculate_rsi(df, 14)
    df["SMA_50"] = df["Close"].rolling(50).mean()

    close = df["Close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]
    prev_rsi = df["RSI"].iloc[-3]

    uptrend = close > sma50 if pd.notna(sma50) else close > df["Close"].iloc[-20]
    oversold = rsi < 38
    recovering = rsi > prev_rsi
    bounce = close > df["Open"].iloc[-1]

    recent_high = df["High"].iloc[-15:].max()
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    score = 0
    if uptrend:
        score += 25
    if oversold:
        score += 25
    if recovering:
        score += 25
    if bounce:
        score += 25

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(df["Low"].iloc[-5:].min() * 0.98, 2)

    return {
        "setup_type": "mean_reversion",
        "setup_name": "Mean Reversion",
        "bias": "bullish",
        "is_valid_setup": uptrend and oversold and recovering and bounce,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(recent_high, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": f"RSI {rsi:.0f} oversold in uptrend with bounce",
    }