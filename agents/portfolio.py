"""Portfolio agent: correlation and concentration warnings."""

from __future__ import annotations

import yfinance as yf

from state import StateManager


def analyze_portfolio(ticker: str, state_mgr: StateManager | None = None) -> dict:
    state_mgr = state_mgr or StateManager()
    open_pos = state_mgr.get_open_positions()

    if not open_pos:
        return {
            "concentration_risk": "LOW",
            "correlated_positions": [],
            "allow_trade": True,
            "summary": "No open positions — portfolio clear",
        }

    tickers = [p.ticker for p in open_pos] + [ticker.upper()]
    warnings = []
    correlated = []

    if len(open_pos) >= 3:
        warnings.append(f"Already {len(open_pos)} open positions")

    try:
        data = yf.download(
            tickers, period="3mo", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        returns = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    s = data["Close"].pct_change().dropna()
                else:
                    s = data[t]["Close"].pct_change().dropna()
                returns[t] = s
            except Exception:
                pass

        new_ret = returns.get(ticker.upper())
        if new_ret is not None:
            for pos in open_pos:
                other = returns.get(pos.ticker)
                if other is not None and len(new_ret) > 10 and len(other) > 10:
                    corr = new_ret.tail(30).corr(other.tail(30))
                    if corr and corr > 0.75:
                        correlated.append(f"{pos.ticker} (r={corr:.2f})")
    except Exception:
        pass

    if correlated:
        warnings.append(f"High correlation with: {', '.join(correlated)}")

    risk = "HIGH" if len(warnings) >= 2 else ("MEDIUM" if warnings else "LOW")

    return {
        "concentration_risk": risk,
        "open_count": len(open_pos),
        "correlated_positions": correlated,
        "allow_trade": risk != "HIGH",
        "summary": "; ".join(warnings) if warnings else "Portfolio exposure acceptable",
    }