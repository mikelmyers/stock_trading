"""Orchestrator: combines all agents into a unified trade sheet."""

from __future__ import annotations

from agents.context import analyze_context
from agents.earnings import analyze_earnings
from agents.exit_manager import build_exit_plan
from agents.intraday import analyze_intraday
from agents.margin import analyze_margin
from agents.options import analyze_options
from agents.portfolio import analyze_portfolio
from agents.probability import estimate_breakout_probability
from agents.qa import analyze_qa
from agents.quant import analyze_quant_signal
from agents.regime import analyze_regime
from agents.risk import calculate_risk_and_sizing, get_max_risk_for_trust
from agents.sector import analyze_sector
from agents.setups.registry import analyze_all_setups, get_best_setup
from config import MIN_CONTEXT_SCORE_FOR_TRADE, MIN_QA_SCORE, MIN_REGIME_SCORE
from data import fetch_ticker_df
from training.calibrator import load_learned_params


def build_trade_sheet(
    ticker: str,
    cap_category: str = "Unknown",
    trust_score: float = 0.0,
    df=None,
    account_equity: float = 2000.0,
) -> dict:
    """Run full multi-agent pipeline and return actionable trade sheet."""
    if df is None:
        df = fetch_ticker_df(ticker)

    all_setups = analyze_all_setups(df, enabled_only=False)
    setup = get_best_setup(df, enabled_only=True)
    context = analyze_context(ticker, df)
    regime = analyze_regime()
    sector = analyze_sector(ticker, df)
    earnings = analyze_earnings(ticker)
    portfolio = analyze_portfolio(ticker)
    intraday = analyze_intraday(ticker, setup)
    probability = estimate_breakout_probability(df)
    max_risk = get_max_risk_for_trust(trust_score)

    if setup.get("current_price"):
        trade_plan = calculate_risk_and_sizing(setup, max_risk)
    else:
        trade_plan = {"action": "SKIP", "reason": "No price data"}

    margin = analyze_margin(trade_plan, account_equity)
    exit_plan = {}
    options_plan = {}
    if trade_plan.get("action") == "TRADE_PROPOSAL":
        exit_plan = build_exit_plan(setup, trade_plan)
        options_plan = analyze_options(ticker, setup, max_risk)

    quant = analyze_quant_signal(setup, context, probability, regime)
    learned = load_learned_params()
    composite = _composite_score(setup, context, probability, regime, quant, sector, intraday)

    sheet_partial = {
        "setup": setup,
        "context": context,
        "regime": regime,
        "probability": probability,
        "quant": quant,
        "earnings": earnings,
        "portfolio": portfolio,
        "intraday": intraday,
        "margin": margin,
    }
    qa = analyze_qa(ticker, df, sheet_partial)

    recommendation = _recommend(
        setup, composite, context, regime, quant, qa, probability,
        earnings, portfolio, intraday, margin, learned,
    )

    return {
        "ticker": ticker.upper(),
        "cap_category": cap_category,
        "recommendation": recommendation,
        "composite_score": composite,
        "setup": setup,
        "all_setups": all_setups,
        "context": context,
        "regime": regime,
        "sector": sector,
        "earnings": earnings,
        "portfolio": portfolio,
        "intraday": intraday,
        "margin": margin,
        "probability": probability,
        "quant": quant,
        "qa": qa,
        "options": options_plan,
        "trade_plan": trade_plan,
        "exit_plan": exit_plan,
        "max_risk_allowed": max_risk,
        "trust_score": trust_score,
        "learned_params": {
            "min_setup_score": learned.get("min_setup_score"),
            "enabled_setups": learned.get("enabled_setups", []),
            "options_backtest": learned.get("options_backtest", {}),
            "trained_simulations": learned.get("trained_on_simulations", 0),
            "trained_win_rate": learned.get("trained_win_rate", 0),
            "trained_expectancy": learned.get("trained_expectancy", 0),
        },
    }


def _recommend(
    setup: dict,
    composite: int,
    context: dict,
    regime: dict,
    quant: dict,
    qa: dict,
    probability: dict,
    earnings: dict,
    portfolio: dict,
    intraday: dict,
    margin: dict,
    learned: dict,
) -> str:
    min_score = learned.get("min_setup_score", 70)
    min_composite = learned.get("min_composite_score", 55)
    min_exp = learned.get("expectancy_threshold", 0.0)

    if qa["status"] == "FAIL":
        return "PASS"
    if regime.get("regime") == "RISK_OFF":
        return "PASS"
    if not earnings.get("allow_trade", True):
        return "PASS"
    if not portfolio.get("allow_trade", True):
        return "PASS"
    if not margin.get("compliant", True):
        return "PASS"
    if not setup["is_valid_setup"]:
        return "PASS"

    trade_ok = (
        setup["confidence_score"] >= min_score
        and composite >= min_composite
        and probability["expectancy"] >= min_exp
        and context.get("context_score", 0) >= MIN_CONTEXT_SCORE_FOR_TRADE
        and regime.get("regime_score", 0) >= MIN_REGIME_SCORE
        and quant.get("signal_quality", 0) >= 50
        and qa.get("qa_score", 0) >= MIN_QA_SCORE
        and intraday.get("allow_entry", True)
    )

    if trade_ok and intraday.get("timing") == "WAIT":
        return "WATCH"

    if trade_ok:
        return "TRADE"
    if setup["confidence_score"] >= min_score - 10 and composite >= min_composite - 10:
        return "WATCH"
    return "PASS"


def _composite_score(
    setup: dict, context: dict, probability: dict,
    regime: dict, quant: dict, sector: dict, intraday: dict,
) -> int:
    score = 0
    score += setup.get("confidence_score", 0) * 0.20
    score += context.get("context_score", 0) * 0.16
    score += probability.get("probability_score", 0) * 0.16
    score += regime.get("regime_score", 0) * 0.12
    score += quant.get("signal_quality", 0) * 0.16
    score += sector.get("sector_score", 0) * 0.10
    score += intraday.get("timing_score", 50) * 0.10
    return min(100, int(score))