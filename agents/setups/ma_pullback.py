"""MA Pullback: strong trend → low-volume dip to 21 EMA → bounce."""

import pandas as pd

from agents.indicators import calculate_atr


def analyze_ma_pullback(df: pd.DataFrame) -> dict:
    if len(df) < 55:
        return _empty()

    df = df.copy()
    # Backtests pre-attach a full-history EMA_21 (causal, exact at each bar);
    # live scans pass a raw window, so compute it on the fly.
    if "EMA_21" not in df.columns:
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    close = df["Close"].iloc[-1]
    ema21 = df["EMA_21"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]
    sma50_prev = df["SMA_50"].iloc[-10]
    low = df["Low"].iloc[-1]
    volume = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:-1].mean()

    uptrend = close > sma50 and sma50 > sma50_prev
    touched_ema = low <= ema21 * 1.01 and close >= ema21 * 0.99
    low_volume = volume < avg_vol * 0.85
    bounce = close > df["Open"].iloc[-1]
    higher_lows = df["Low"].iloc[-5:].min() > df["Low"].iloc[-15:-5].min()

    recent_high = df["High"].iloc[-20:].max()
    stop = round(min(ema21 * 0.97, df["Low"].iloc[-3:].min() * 0.99), 2)
    atr_14 = calculate_atr(df, window=14).iloc[-1]

    score = 0
    if uptrend:
        score += 25
    if touched_ema:
        score += 25
    if low_volume:
        score += 20
    if bounce:
        score += 15
    if higher_lows:
        score += 15

    valid = uptrend and touched_ema and low_volume and bounce

    return {
        "setup_type": "ma_pullback",
        "setup_name": "MA Pullback",
        "bias": "bullish",
        "is_valid_setup": valid,
        "confidence_score": score,
        "current_price": round(close, 2),
        "resistance_level": round(recent_high, 2),
        "stop_loss": stop,
        "atr_14": round(atr_14, 2),
        "ema_21": round(ema21, 2),
        "sma_50": round(sma50, 2),
        "volume_ratio": round(volume / avg_vol, 2) if avg_vol else 0,
        "details": "Uptrend pullback to 21 EMA on low volume with bounce",
    }


def _empty() -> dict:
    return {
        "setup_type": "ma_pullback",
        "setup_name": "MA Pullback",
        "is_valid_setup": False,
        "confidence_score": 0,
        "current_price": 0,
        "resistance_level": 0,
        "stop_loss": 0,
        "atr_14": 0,
        "details": "",
    }