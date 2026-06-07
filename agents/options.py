"""Options agent: defined-risk options research (no execution)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from agents.indicators import calculate_atr


def analyze_options(ticker: str, setup: dict, max_risk: float = 10.0) -> dict:
    """
    Suggest defined-risk options structures matching setup bias.
    Uses real option chains when available, HV fallback otherwise.
    """
    bias = setup.get("bias", "neutral")
    price = setup.get("current_price", 0)
    if not price:
        return _skip("No price")

    hv = _historical_volatility(ticker)
    chain = _fetch_chain(ticker, price)

    if bias == "bullish":
        primary = "bull_call_spread"
        alt = "long_call"
        structure = _bull_spread(chain, price, max_risk)
    elif bias == "bearish":
        primary = "bear_put_spread"
        alt = "long_put"
        structure = _bear_spread(chain, price, max_risk)
    else:
        primary = "iron_condor"
        alt = "calendar_spread"
        structure = _neutral_spread(chain, price, max_risk)

    iv_rank = _iv_rank_proxy(hv)

    return {
        "available": structure is not None,
        "primary_strategy": primary,
        "alt_strategy": alt,
        "historical_vol": round(hv * 100, 1),
        "iv_rank_proxy": iv_rank,
        "expected_move_pct": round(hv * 100 * 0.85, 1),
        "structure": structure or {},
        "max_risk": max_risk,
        "summary": (
            f"{primary.replace('_', ' ').title()} — "
            f"HV {hv*100:.0f}%, IV-rank-proxy {iv_rank}, "
            f"max risk ${max_risk}"
        ),
        "note": "Research only. Verify chain liquidity and spreads before trading.",
    }


def _historical_volatility(ticker: str, window: int = 20) -> float:
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        returns = df["Close"].pct_change().dropna()
        return float(returns.tail(window).std() * (252 ** 0.5))
    except Exception:
        return 0.35


def _iv_rank_proxy(hv: float) -> str:
    if hv > 0.55:
        return "HIGH"
    if hv > 0.35:
        return "MEDIUM"
    return "LOW"


def _fetch_chain(ticker: str, price: float) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        exp = next((e for e in expirations if _days_out(e) >= 21), expirations[0])
        chain = t.option_chain(exp)
        return {
            "expiration": exp,
            "calls": chain.calls,
            "puts": chain.puts,
            "price": price,
        }
    except Exception:
        return None


def _days_out(exp: str) -> int:
    from datetime import datetime
    exp_date = datetime.strptime(exp, "%Y-%m-%d")
    return (exp_date - datetime.now()).days


def _bull_spread(chain: dict | None, price: float, max_risk: float) -> dict | None:
    if not chain:
        width = round(price * 0.03, 0)
        return {
            "type": "bull_call_spread",
            "long_strike": round(price, 0),
            "short_strike": round(price + width, 0),
            "width": width,
            "est_max_loss": max_risk,
            "expiration": "30-45 DTE (estimate)",
        }
    calls = chain["calls"]
    price = chain["price"]
    otm = calls[calls["strike"] >= price * 0.98].head(8)
    if len(otm) < 2:
        return None
    long_row = otm.iloc[0]
    short_row = otm.iloc[min(3, len(otm) - 1)]
    width = short_row["strike"] - long_row["strike"]
    debit = max(0.05, (long_row.get("lastPrice", 0) - short_row.get("lastPrice", 0)))
    contracts = max(1, int(max_risk / (debit * 100))) if debit else 1
    return {
        "type": "bull_call_spread",
        "expiration": chain["expiration"],
        "long_strike": float(long_row["strike"]),
        "short_strike": float(short_row["strike"]),
        "est_debit": round(debit, 2),
        "contracts": contracts,
        "est_max_loss": round(min(max_risk, debit * 100 * contracts), 2),
    }


def _bear_spread(chain: dict | None, price: float, max_risk: float) -> dict | None:
    if not chain:
        width = round(price * 0.03, 0)
        return {
            "type": "bear_put_spread",
            "long_strike": round(price, 0),
            "short_strike": round(price - width, 0),
            "width": width,
            "est_max_loss": max_risk,
        }
    puts = chain["puts"]
    otm = puts[puts["strike"] <= price * 1.02].tail(8)
    if len(otm) < 2:
        return None
    long_row = otm.iloc[-1]
    short_row = otm.iloc[max(0, len(otm) - 4)]
    debit = max(0.05, (long_row.get("lastPrice", 0) - short_row.get("lastPrice", 0)))
    contracts = max(1, int(max_risk / (debit * 100))) if debit else 1
    return {
        "type": "bear_put_spread",
        "expiration": chain["expiration"],
        "long_strike": float(long_row["strike"]),
        "short_strike": float(short_row["strike"]),
        "est_debit": round(debit, 2),
        "contracts": contracts,
        "est_max_loss": round(min(max_risk, debit * 100 * contracts), 2),
    }


def _neutral_spread(chain: dict | None, price: float, max_risk: float) -> dict:
    return {
        "type": "iron_condor",
        "est_max_loss": max_risk,
        "wing_width": round(price * 0.05, 0),
        "note": "Sell OTM put spread + OTM call spread, 30-45 DTE",
    }


def _skip(reason: str) -> dict:
    return {
        "available": False,
        "summary": reason,
        "structure": {},
        "note": "",
    }