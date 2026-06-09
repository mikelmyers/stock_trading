"""Shared technical indicators."""

import numpy as np
import pandas as pd


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # numpy true-range avoids pandas concat + row-wise max, which dominated the
    # backtest hot loop. Result is identical to the prior concat/.max(axis=1):
    # np.nanmax mirrors pandas skipna max, so row 0 (no prior close) falls back
    # to High-Low exactly as before.
    high = df["High"].to_numpy(dtype="float64")
    low = df["Low"].to_numpy(dtype="float64")
    close = df["Close"].to_numpy(dtype="float64")
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    true_range = np.nanmax(
        np.stack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]),
        axis=0,
    )
    return pd.Series(true_range, index=df.index).rolling(window=window).mean()


def calculate_rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-weighted average of typical price over the trailing ``window`` bars.

    A daily-bar "VWAP" anchored at the start of whatever frame you happen to
    hold (the old cumsum version) means a decades-old average in training and a
    60-day average live — two unrelated numbers. A fixed trailing window is the
    same quantity everywhere and is causal at every bar.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = (typical * df["Volume"]).rolling(window).sum()
    return pv / df["Volume"].rolling(window).sum()