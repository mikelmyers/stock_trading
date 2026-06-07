"""Credit Put Spread: sell OTM put in uptrend (premium collection)."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_credit_put(df: pd.DataFrame) -> dict:
    """High-probability credit put in confirmed uptrend — options-style equity signal."""
    if len(df) < 55:
        return empty_setup("credit_put", "Credit Put")

    df = df.copy()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    close = df["Close"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]
    sma200 = df["SMA_200"].iloc[-1] if pd.notna(df["SMA_200"].iloc[-1]) else sma50

    uptrend = close > sma50 > sma200 if pd.notna(sma200) else close > sma50
    support = df["Low"].iloc[-20:].min()
    holding_support = close > support * 1.03
    low_vol = df["Volume"].iloc[-1] < df["Volume"].iloc[-20:-1].mean()

    score = 0
    if uptrend:
        score += 35
    if holding_support:
        score += 35
    if low_vol:
        score += 30

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(support * 0.97, 2)
    vol_avg = df["Volume"].iloc[-20:-1].mean()
    volume_ratio = round(df["Volume"].iloc[-1] / vol_avg, 2) if vol_avg > 0 else 0.0

    return {
        "setup_type": "credit_put",
        "setup_name": "Credit Put Zone",
        "bias": "bullish",
        "options_style": "sell_put_spread",
        "is_valid_setup": uptrend and holding_support and low_vol,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(close * 1.05, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": volume_ratio,
        "details": "Uptrend above support — favorable for credit put / cash-secured put",
    }