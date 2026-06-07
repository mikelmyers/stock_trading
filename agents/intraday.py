"""Intraday timing: refine swing entries using 1-hour data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from agents.indicators import calculate_rsi


def analyze_intraday(ticker: str, setup: dict) -> dict:
    """
    Check 1h chart for entry quality on a daily swing setup.
    Prevents buying extended moves at the top of the hour.
    """
    try:
        df = yf.download(
            ticker, period="5d", interval="1h",
            progress=False, auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
    except Exception:
        return _neutral("Intraday data unavailable")

    if len(df) < 10:
        return _neutral("Insufficient intraday bars")

    close = float(df["Close"].iloc[-1])
    vwap = float((df["Close"] * df["Volume"]).sum() / df["Volume"].sum())
    rsi_1h = float(calculate_rsi(df, 14).iloc[-1])
    bias = setup.get("bias", "bullish")

    if bias == "bullish":
        extended = close > vwap * 1.015 and rsi_1h > 68
        ideal = close <= vwap * 1.005 and rsi_1h < 55
        if ideal:
            timing, score = "ENTER", 90
            msg = "1h pullback to VWAP — good swing entry window"
        elif extended:
            timing, score = "WAIT", 30
            msg = f"1h extended (RSI {rsi_1h:.0f}) — wait for hourly pullback"
        else:
            timing, score = "OK", 65
            msg = f"1h neutral (RSI {rsi_1h:.0f}) — acceptable entry"
    else:
        extended = close < vwap * 0.985 and rsi_1h < 32
        ideal = close >= vwap * 0.995 and rsi_1h > 45
        if ideal:
            timing, score = "ENTER", 90
            msg = "1h bounce to VWAP — good short entry window"
        elif extended:
            timing, score = "WAIT", 30
            msg = f"1h oversold extension — wait for hourly bounce"
        else:
            timing, score = "OK", 65
            msg = f"1h neutral for short entry"

    return {
        "timing": timing,
        "timing_score": score,
        "rsi_1h": round(rsi_1h, 1),
        "price_vs_vwap_pct": round((close / vwap - 1) * 100, 2),
        "allow_entry": timing in ("ENTER", "OK"),
        "summary": msg,
    }


def _neutral(msg: str) -> dict:
    return {
        "timing": "OK",
        "timing_score": 50,
        "rsi_1h": 0,
        "price_vs_vwap_pct": 0,
        "allow_entry": True,
        "summary": msg,
    }