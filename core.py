"""Core business logic for the trading research agent."""

from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf

from agents.exit_manager import build_exit_plan, evaluate_position
from agents.orchestrator import build_trade_sheet
from agents.risk import get_max_risk_for_trust, get_risk_tier_label
from agents.teacher import generate_trade_review
from data import extract_ticker_df, fetch_multiple, fetch_ticker_df, get_all_watchlist_tickers
from reports import format_status, format_trade_sheet, save_trade_sheet
from state import ActivePosition, StateManager


def scan_market(state_mgr: StateManager | None = None) -> list[dict]:
    """Scan full watchlist for valid breakout setups."""
    state_mgr = state_mgr or StateManager()
    trust = state_mgr.state.trust_score
    ticker_map = get_all_watchlist_tickers()
    tickers = list(ticker_map.keys())

    data = fetch_multiple(tickers)
    alerts = []

    for ticker, cap_category in ticker_map.items():
        try:
            df = extract_ticker_df(data, ticker, len(tickers))
            if df.empty or len(df) < 25:
                continue

            sheet = build_trade_sheet(ticker, cap_category, trust, df=df)
            if (
                sheet["setup"]["is_valid_setup"]
                and sheet["recommendation"] in ("TRADE", "WATCH")
                and sheet["trade_plan"].get("action") == "TRADE_PROPOSAL"
            ):
                alerts.append(sheet)
        except Exception as e:
            print(f"  [!] Error processing {ticker}: {e}")

    alerts.sort(key=lambda s: s["composite_score"], reverse=True)
    return alerts


def analyze_ticker(ticker: str, state_mgr: StateManager | None = None) -> dict:
    """Full analysis on a single ticker."""
    state_mgr = state_mgr or StateManager()
    ticker_map = get_all_watchlist_tickers()
    cap = ticker_map.get(ticker.upper(), "Custom")
    return build_trade_sheet(ticker.upper(), cap, state_mgr.state.trust_score)


def track_ticker(
    ticker: str,
    state_mgr: StateManager | None = None,
    entry: float | None = None,
    stop: float | None = None,
    shares: float | None = None,
) -> ActivePosition:
    """
    Start tracking a position.
    Without manual params, uses current breakout analysis.
    """
    state_mgr = state_mgr or StateManager()
    ticker = ticker.upper()

    if state_mgr.has_open_position(ticker):
        raise ValueError(f"Already tracking {ticker}. Close it first.")

    sheet = analyze_ticker(ticker, state_mgr)
    setup = sheet["setup"]
    plan = sheet["trade_plan"]
    exit_plan = sheet.get("exit_plan", {})

    if entry is None:
        if plan.get("action") != "TRADE_PROPOSAL":
            raise ValueError(
                f"Cannot auto-track {ticker}: {plan.get('reason', 'no valid setup')}. "
                "Use --entry, --stop, --shares for manual tracking."
            )
        entry = plan["entry_price"]
        stop = plan["stop_loss"]
        shares = plan["shares"]
    else:
        if stop is None:
            stop = round(setup.get("resistance_level", entry) * 0.98, 2)
        if shares is None:
            max_risk = get_max_risk_for_trust(state_mgr.state.trust_score)
            risk_per_share = entry - stop
            if risk_per_share <= 0:
                raise ValueError("Stop must be below entry price.")
            shares = round(max_risk / risk_per_share, 4)

    trailing = exit_plan.get("initial_trailing_stop", stop)

    position = ActivePosition(
        ticker=ticker,
        cap_category=sheet["cap_category"],
        entry_price=entry,
        stop_loss=stop,
        shares=shares,
        shares_remaining=shares,
        entry_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        confidence_score=setup.get("confidence_score", 0),
        resistance_level=setup.get("resistance_level", entry),
        max_risk=round(shares * (entry - stop), 2),
        high_water_mark=entry,
        trailing_stop=trailing,
        composite_score=sheet.get("composite_score", 0),
    )
    state_mgr.open_position(position)
    return position


def monitor_positions(state_mgr: StateManager | None = None) -> list[dict]:
    """Daily check on all open positions. Returns list of actions taken."""
    state_mgr = state_mgr or StateManager()
    positions = state_mgr.get_open_positions()
    actions = []

    if not positions:
        return actions

    tickers = [p.ticker for p in positions]
    data = fetch_multiple(tickers)

    for pos in positions:
        try:
            df = extract_ticker_df(data, pos.ticker, len(tickers))
            result = evaluate_position(pos, df)

            if result["action"] == "EXIT":
                closed = state_mgr.close_position(
                    pos.ticker, result["exit_price"], result["reason"]
                )
                if closed:
                    review = generate_trade_review(closed, state_mgr.state)
                    actions.append({"type": "EXIT", "position": pos, "result": result, "review": review})

            elif result["action"] == "SCALE_OUT":
                event = state_mgr.partial_scale_out(
                    pos.ticker,
                    result["shares_to_sell"],
                    result["exit_price"],
                    result["reason"],
                )
                if result.get("new_high_water_mark"):
                    state_mgr.update_position_stops(
                        pos.ticker,
                        result["new_high_water_mark"],
                        result["new_trailing_stop"],
                    )
                actions.append({"type": "SCALE_OUT", "position": pos, "result": result, "event": event})

            elif result["action"] == "HOLD":
                if result.get("trail_update"):
                    state_mgr.update_position_stops(
                        pos.ticker,
                        result["trail_update"]["high_water_mark"],
                        result["trail_update"]["trailing_stop"],
                    )
                actions.append({"type": "HOLD", "position": pos, "result": result})

        except Exception as e:
            actions.append({"type": "ERROR", "position": pos, "error": str(e)})

    return actions


def close_position_manual(
    ticker: str,
    exit_price: float,
    reason: str = "MANUAL",
    state_mgr: StateManager | None = None,
) -> dict | None:
    state_mgr = state_mgr or StateManager()
    closed = state_mgr.close_position(ticker.upper(), exit_price, reason)
    if closed:
        return generate_trade_review(closed, state_mgr.state)
    return None


def submit_feedback(
    ticker: str,
    fidelity: float,
    execution_aligned: bool,
    notes: str = "",
    state_mgr: StateManager | None = None,
) -> bool:
    state_mgr = state_mgr or StateManager()
    return state_mgr.add_feedback_to_last_trade(
        ticker.upper(), fidelity, execution_aligned, notes
    )


def get_status(state_mgr: StateManager | None = None) -> str:
    state_mgr = state_mgr or StateManager()
    status = format_status(state_mgr.state)
    tier = get_risk_tier_label(state_mgr.state.trust_score)
    max_risk = get_max_risk_for_trust(state_mgr.state.trust_score)
    return status + f"\n  Risk Tier: {tier} (max ${max_risk}/trade)\n" + "=" * 64


def export_sheet(ticker: str, state_mgr: StateManager | None = None) -> tuple[dict, str]:
    sheet = analyze_ticker(ticker, state_mgr)
    path = save_trade_sheet(sheet)
    return sheet, str(path)