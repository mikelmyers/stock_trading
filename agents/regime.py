"""Regime agent: classify market environment for trade filtering."""

import pandas as pd

from data import fetch_ticker_df


def analyze_regime() -> dict:
    """Classify broad market regime using SPY."""
    try:
        spy = fetch_ticker_df("SPY", period="1y")
        vix = fetch_ticker_df("^VIX", period="60d")
    except Exception:
        return _neutral("Data unavailable")

    close = spy["Close"].iloc[-1]
    sma200 = spy["Close"].rolling(200).mean().iloc[-1]
    ret_20 = (spy["Close"].iloc[-1] - spy["Close"].iloc[-21]) / spy["Close"].iloc[-21]
    vix_level = vix["Close"].iloc[-1] if not vix.empty else 20

    if close > sma200 and ret_20 > 0.02 and vix_level < 22:
        regime = "RISK_ON"
        score = 90
        summary = "Bull regime: SPY above 200MA, positive momentum, low fear."
    elif close < sma200 or vix_level > 28:
        regime = "RISK_OFF"
        score = 20
        summary = "Defensive regime: weak trend or elevated fear — reduce size."
    else:
        regime = "NEUTRAL"
        score = 55
        summary = "Mixed regime: selective setups only."

    return {
        "regime": regime,
        "regime_score": score,
        "spy_vs_200ma": round((close / sma200 - 1) * 100, 2) if sma200 else 0,
        "spy_20d_return": round(ret_20 * 100, 2),
        "vix": round(vix_level, 1),
        "summary": summary,
        "allow_new_trades": regime != "RISK_OFF",
    }


def _neutral(reason: str) -> dict:
    return {
        "regime": "NEUTRAL",
        "regime_score": 50,
        "spy_vs_200ma": 0,
        "spy_20d_return": 0,
        "vix": 0,
        "summary": reason,
        "allow_new_trades": True,
    }