"""End-to-end simulation of the multi-agent trading pipeline."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents.exit_manager import evaluate_position
from agents.orchestrator import build_trade_sheet
from agents.teacher import generate_trade_review
from config import STATE_FILE
from state import ActivePosition, StateManager


def make_breakout_df() -> pd.DataFrame:
    """Synthetic 60-day chart with a textbook breakout on the last bar."""
    np.random.seed(7)
    dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
    base = 42.0
    closes, highs, lows, volumes = [], [], [], []

    for i in range(60):
        if i < 40:
            c = base + np.random.uniform(-0.4, 0.4)
            v = 800_000
        elif i < 58:
            c = base + np.random.uniform(-0.2, 0.2)
            v = 700_000
        elif i == 58:
            c = base + 0.1
            v = 750_000
        else:
            c = base + 2.8
            v = 2_100_000
        closes.append(c)
        highs.append(c + 0.25)
        lows.append(c - 0.25)
        volumes.append(v)

    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


def make_forward_days(entry: float, stop: float, days: int = 12) -> list[pd.DataFrame]:
    """Simulate price path: dip, rally to 1R, 2R, then trail."""
    risk = entry - stop
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=60), periods=60 + days, freq="D"
    )
    base_df = make_breakout_df()
    path = []

    for d in range(days):
        day = d + 1
        if day <= 2:
            close = entry - 0.3
            low = stop + 0.05
        elif day == 3:
            close = entry + risk * 1.05
            low = entry
        elif day == 5:
            close = entry + risk * 2.1
            low = entry + risk * 0.5
        elif day <= 8:
            close = entry + risk * 2.5
            low = close - 0.8
        else:
            close = entry + risk * 1.8
            low = close - 1.5

        row = pd.DataFrame(
            {
                "Open": [close - 0.1],
                "High": [close + 0.3],
                "Low": [low],
                "Close": [close],
                "Volume": [1_200_000],
            },
            index=[dates[-(days - d)]],
        )
        path.append(row)

    frames = [base_df] + path
    return frames


def run_simulation() -> None:
    sim_state = Path(STATE_FILE).with_name("trade_state_sim.json")
    if sim_state.exists():
        sim_state.unlink()

    print("=" * 68)
    print("  MULTI-AGENT SIMULATION")
    print("=" * 68)

    # ── Phase 1: Orchestrator runs all worker agents ──
    print("\n[PHASE 1] Orchestrator dispatches worker agents...\n")
    df = make_breakout_df()
    sheet = build_trade_sheet("SIM", "Small Cap", trust_score=0.0, df=df)

    agents_report = [
        ("Scout", f"setup={sheet['setup']['confidence_score']}/100, "
                  f"valid={sheet['setup']['is_valid_setup']}"),
        ("Context", sheet["context"]["summary"]),
        ("Probability", sheet["probability"]["summary"]),
        ("Risk", f"max_risk=${sheet['max_risk_allowed']}, "
                 f"action={sheet['trade_plan'].get('action')}"),
        ("Exit Manager", f"targets={len(sheet['exit_plan'].get('scale_out_ladder', []))} levels"),
    ]
    for name, result in agents_report:
        print(f"  [{name:14s}] {result}")

    print(f"\n  [Orchestrator ] Composite={sheet['composite_score']}/100, "
          f"Recommendation={sheet['recommendation']}")

    plan = sheet["trade_plan"]
    if plan.get("action") != "TRADE_PROPOSAL":
        print("\nSimulation failed: no trade proposal generated.")
        return

    # ── Phase 2: User authorizes → State Tracker opens position ──
    print("\n[PHASE 2] User authorizes trade → State Tracker records position\n")
    state_mgr = StateManager(sim_state)
    exit_plan = sheet["exit_plan"]

    position = ActivePosition(
        ticker="SIM",
        cap_category="Small Cap",
        entry_price=plan["entry_price"],
        stop_loss=plan["stop_loss"],
        shares=plan["shares"],
        shares_remaining=plan["shares"],
        entry_date=(datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
        confidence_score=sheet["setup"]["confidence_score"],
        resistance_level=sheet["setup"]["resistance_level"],
        max_risk=plan["actual_risk"],
        high_water_mark=plan["entry_price"],
        trailing_stop=exit_plan["initial_trailing_stop"],
        composite_score=sheet["composite_score"],
    )
    state_mgr.open_position(position)
    print(f"  Tracked: {position.shares} shares @ ${position.entry_price}, "
          f"stop ${position.stop_loss}, risk ${position.max_risk}")

    # ── Phase 3: Daily monitor loop (Exit Manager) ──
    print("\n[PHASE 3] Daily monitor loop — Exit Manager evaluates each day\n")
    forward = make_forward_days(plan["entry_price"], plan["stop_loss"])

    for day_num, day_df in enumerate(forward, 1):
        full_df = pd.concat([make_breakout_df(), day_df]).drop_duplicates()
        pos = state_mgr.get_position("SIM")
        if not pos:
            break

        result = evaluate_position(pos, full_df)
        price = result.get("current_price", result.get("exit_price", "?"))
        print(f"  Day {day_num:2d} | ${price} | {result['action']:10s} | "
              f"{result.get('message', result.get('reason', ''))}")

        if result["action"] == "SCALE_OUT":
            state_mgr.partial_scale_out(
                "SIM", result["shares_to_sell"], result["exit_price"], result["reason"]
            )
            if result.get("new_high_water_mark"):
                state_mgr.update_position_stops(
                    "SIM", result["new_high_water_mark"], result["new_trailing_stop"]
                )
        elif result["action"] == "EXIT":
            closed = state_mgr.close_position("SIM", result["exit_price"], result["reason"])
            if closed:
                state_mgr.add_feedback_to_last_trade("SIM", 0.95, True, "Sim followed plan")
                review = generate_trade_review(closed, state_mgr.state)
                print(f"\n  [Teacher] {review['result']} | P&L ${review['pnl']:+.2f} "
                      f"({review['r_multiple']:+.1f}R) | Trust={review['trust_score']}% "
                      f"({review['risk_tier']})")
                for lesson in review["lessons"]:
                    print(f"            -> {lesson}")
            break
        elif result.get("trail_update"):
            state_mgr.update_position_stops(
                "SIM",
                result["trail_update"]["high_water_mark"],
                result["trail_update"]["trailing_stop"],
            )

    # ── Summary ──
    s = state_mgr.state
    print("\n" + "=" * 68)
    print("  SIMULATION RESULTS")
    print("=" * 68)
    print(f"  Trades closed:  {s.total_trades}")
    print(f"  Scale-outs:     {len(s.scale_out_log)}")
    print(f"  Total P&L:      ${s.total_pnl:+.2f}")
    print(f"  Trust score:    {s.trust_score}%")
    print(f"  State saved:    {sim_state}")
    print("=" * 68)

    sim_state.unlink()


if __name__ == "__main__":
    run_simulation()