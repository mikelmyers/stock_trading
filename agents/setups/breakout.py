"""Core Breakout: squeeze → ceiling → volume ignition."""

import numpy as np
import pandas as pd

from agents.indicators import calculate_atr


def analyze_breakout(df: pd.DataFrame) -> dict:
    if len(df) < 25:
        return _empty("breakout")

    df = df.copy()
    df["ATR_20"] = calculate_atr(df, window=20)
    df["ATR_5"] = calculate_atr(df, window=5)
    is_compressed = df["ATR_5"].iloc[-1] < (df["ATR_20"].iloc[-1] * 0.85)

    recent_max = df["High"].iloc[-20:-1].max()
    tolerance = recent_max * 0.015
    ceiling_touches = (
        df.iloc[-20:-1][abs(df.iloc[-20:-1]["High"] - recent_max) <= tolerance].shape[0]
    )

    current_close = df["Close"].iloc[-1]
    current_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].iloc[-20:-1].mean()

    price_breakout = current_close > recent_max
    volume_surge = current_volume > (avg_volume * 1.5)

    score = 0
    if is_compressed:
        score += 30
    if ceiling_touches >= 2:
        score += 30
    if price_breakout and volume_surge:
        score += 40

    atr_14 = calculate_atr(df, window=14).iloc[-1]
    stop = round(recent_max * 0.98, 2)

    return {
        "setup_type": "breakout",
        "setup_name": "Core Breakout",
        "bias": "bullish",
        "is_valid_setup": price_breakout and volume_surge and is_compressed,
        "confidence_score": score,
        "current_price": round(current_close, 2),
        "resistance_level": round(recent_max, 2),
        "stop_loss": stop,
        "atr_14": round(atr_14, 2),
        "ceiling_touches": int(ceiling_touches),
        "volume_ratio": round(current_volume / avg_volume, 2) if avg_volume else 0,
        "details": "Squeeze under resistance with volume surge breakout",
    }


def _empty(setup_type: str) -> dict:
    return {
        "setup_type": setup_type,
        "setup_name": setup_type,
        "is_valid_setup": False,
        "confidence_score": 0,
        "current_price": 0,
        "resistance_level": 0,
        "stop_loss": 0,
        "atr_14": 0,
        "details": "",
    }