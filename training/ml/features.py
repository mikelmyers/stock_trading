"""Point-in-time feature engineering for ML models.

Every feature here is *causal*: its value at bar ``i`` depends only on data up to
and including bar ``i``. That is the single most important property for avoiding
look-ahead bias — the model must never see information that wouldn't have been
available at the moment a trade decision is made.

``compute_feature_frame`` returns a DataFrame aligned to the input index, so a
feature row for a decision at bar ``i`` is simply ``frame.iloc[i]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.indicators import calculate_atr, calculate_rsi

# Column order is stable so downstream models see consistent features.
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "atr_pct", "atr_squeeze", "realized_vol_20",
    "rsi_14", "rsi_slope_3",
    "dist_ema21_atr", "dist_sma50_pct", "dist_sma200_pct",
    "ema21_gt_sma50", "sma50_gt_sma200",
    "vol_ratio_20", "vol_trend_5_20",
    "dist_high_20_pct", "dist_high_252_pct", "dist_low_20_pct",
    "up_days_10",
]


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a causal feature DataFrame aligned to ``df.index``.

    NaNs appear in the warm-up region (e.g. before 200 bars exist for SMA_200).
    Gradient-boosted trees handle NaN natively, so we leave them in rather than
    forward-filling (which could leak) or dropping (which the caller may prefer
    to do explicitly).
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ema21 = close.ewm(span=21, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = calculate_atr(df, 14)
    atr5 = calculate_atr(df, 5)
    rsi14 = calculate_rsi(df, 14)
    vol20 = volume.rolling(20).mean()
    vol5 = volume.rolling(5).mean()
    daily_ret = close.pct_change()

    high20 = high.rolling(20).max()
    high252 = high.rolling(252).max()
    low20 = low.rolling(20).min()

    out = pd.DataFrame(index=df.index)
    # Momentum / returns over multiple horizons
    out["ret_1d"] = daily_ret
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)
    out["ret_60d"] = close.pct_change(60)
    # Volatility / compression
    out["atr_pct"] = atr14 / close
    out["atr_squeeze"] = atr5 / atr14          # < 1 means recent range compressed
    out["realized_vol_20"] = daily_ret.rolling(20).std()
    # Oscillators
    out["rsi_14"] = rsi14
    out["rsi_slope_3"] = rsi14 - rsi14.shift(3)
    # Location relative to trend
    out["dist_ema21_atr"] = (close - ema21) / atr14
    out["dist_sma50_pct"] = (close - sma50) / sma50
    out["dist_sma200_pct"] = (close - sma200) / sma200
    out["ema21_gt_sma50"] = (ema21 > sma50).astype("float64")
    out["sma50_gt_sma200"] = (sma50 > sma200).astype("float64")
    # Volume
    out["vol_ratio_20"] = volume / vol20
    out["vol_trend_5_20"] = vol5 / vol20
    # Structure / extremes
    out["dist_high_20_pct"] = (close - high20) / high20
    out["dist_high_252_pct"] = (close - high252) / high252
    out["dist_low_20_pct"] = (close - low20) / low20
    out["up_days_10"] = (daily_ret > 0).rolling(10).sum()

    return out[FEATURE_COLUMNS]
