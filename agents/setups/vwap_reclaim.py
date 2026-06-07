"""VWAP Reclaim: price dips below VWAP in uptrend, reclaims on volume."""

import pandas as pd

from agents.indicators import calculate_atr
from agents.setups.base import empty_setup


def analyze_vwap_reclaim(df: pd.DataFrame) -> dict:
    if len(df) < 25:
        return empty_setup("vwap_reclaim", "VWAP Reclaim")

    df = df.copy()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    close = df["Close"].iloc[-1]
    low = df["Low"].iloc[-1]
    vwap = df["VWAP"].iloc[-1]
    sma20 = df["SMA_20"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    uptrend = close > sma20 if pd.notna(sma20) else True
    dipped_below = low < vwap * 0.995
    reclaimed = close > vwap
    vol_confirm = volume > avg_vol * 1.2

    recent_high = df["High"].iloc[-15:].max()
    score = 0
    if uptrend:
        score += 25
    if dipped_below:
        score += 25
    if reclaimed:
        score += 30
    if vol_confirm:
        score += 20

    atr = calculate_atr(df, 14).iloc[-1]
    stop = round(min(vwap * 0.97, df["Low"].iloc[-3:].min() * 0.99), 2)

    return {
        "setup_type": "vwap_reclaim",
        "setup_name": "VWAP Reclaim",
        "bias": "bullish",
        "is_valid_setup": uptrend and dipped_below and reclaimed and vol_confirm,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(recent_high, 2),
        "stop_loss": stop,
        "atr_14": round(atr, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": f"Reclaimed VWAP ${vwap:.2f} on volume after intraday dip",
    }