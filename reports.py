"""Trade sheet formatting and report export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import REPORTS_DIR


def format_trade_sheet(sheet: dict) -> str:
    lines = [
        "=" * 64,
        f"  TRADE SHEET: {sheet['ticker']} ({sheet['cap_category']})",
        f"  Recommendation: {sheet['recommendation']} | Composite: {sheet['composite_score']}/100",
        "=" * 64,
        "",
        f"SETUP: {sheet['setup'].get('setup_name', 'Unknown')} ({sheet['setup'].get('setup_type', '')})",
        f"  Valid:          {'YES' if sheet['setup']['is_valid_setup'] else 'NO'}",
        f"  Setup score:    {sheet['setup']['confidence_score']}/100",
        f"  Price:          ${sheet['setup'].get('current_price', 'N/A')}",
        f"  Stop:           ${sheet['setup'].get('stop_loss', 'N/A')}",
        f"  Target/resist:  ${sheet['setup'].get('resistance_level', 'N/A')}",
        f"  Volume ratio:   {sheet['setup'].get('volume_ratio', 'N/A')}x",
        f"  Detail:         {sheet['setup'].get('details', '')}",
        "",
        "CONTEXT (Market Tailwind)",
        f"  {sheet['context']['summary']}",
        f"  vs SPY (20d):   {sheet['context']['relative_strength_vs_spy']:+.1f}%",
        f"  vs Sector:      {sheet['context']['relative_strength_vs_sector']:+.1f}%",
        f"  Context score:  {sheet['context']['context_score']}/100",
        "",
        "REGIME (Market Environment)",
        f"  {sheet.get('regime', {}).get('summary', 'N/A')}",
        f"  Regime:         {sheet.get('regime', {}).get('regime', 'N/A')}",
        f"  VIX:            {sheet.get('regime', {}).get('vix', 'N/A')}",
        f"  SPY 20d:        {sheet.get('regime', {}).get('spy_20d_return', 0):+.1f}%",
        "",
        "PROBABILITY (Historical Edge)",
        f"  {sheet['probability']['summary']}",
        f"  Win rate:       {sheet['probability']['win_rate']}%",
        f"  Expectancy:     {sheet['probability']['expectancy']}R",
        f"  Confidence:     {sheet['probability']['confidence']}",
        "",
        "SECTOR (Rotation)",
        f"  {sheet.get('sector', {}).get('summary', 'N/A')}",
        "",
        "EARNINGS (Catalyst Risk)",
        f"  {sheet.get('earnings', {}).get('summary', 'N/A')}",
        "",
        "PORTFOLIO (Exposure)",
        f"  {sheet.get('portfolio', {}).get('summary', 'N/A')}",
        "",
        "INTRADAY (Entry Timing)",
        f"  {sheet.get('intraday', {}).get('summary', 'N/A')}",
        f"  Timing:         {sheet.get('intraday', {}).get('timing', 'N/A')}",
        "",
        "MARGIN (Compliance)",
        f"  {sheet.get('margin', {}).get('summary', 'N/A')}",
        "",
        "QUANT (Signal Quality)",
        f"  {sheet.get('quant', {}).get('summary', 'N/A')}",
        "",
        "QA (Quality Checks)",
        f"  {sheet.get('qa', {}).get('summary', 'N/A')}",
        "",
    ]

    opts = sheet.get("options", {})
    if opts.get("available") or opts.get("structure"):
        lines += [
            "OPTIONS (Defined Risk)",
            f"  {opts.get('summary', '')}",
            f"  Strategy:       {opts.get('primary_strategy', 'N/A')}",
            f"  Alt:            {opts.get('alt_strategy', 'N/A')}",
            f"  {opts.get('note', '')}",
            "",
        ]

    valid_alts = [s for s in sheet.get("all_setups", []) if s.get("is_valid_setup")]
    if len(valid_alts) > 1:
        lines.append("OTHER VALID SETUPS:")
        for s in valid_alts:
            if s["setup_type"] != sheet["setup"].get("setup_type"):
                lines.append(f"  - {s['setup_name']}: {s['confidence_score']}/100")
        lines.append("")

    plan = sheet.get("trade_plan", {})
    if plan.get("action") == "TRADE_PROPOSAL":
        lines += [
            "EXECUTION PLAN (Risk Agent)",
            f"  Trust tier:     {sheet.get('trust_score', 0)}% -> max risk ${sheet['max_risk_allowed']}",
            f"  Entry:          ${plan['entry_price']}",
            f"  Stop loss:      ${plan['stop_loss']}",
            f"  Shares:         {plan['shares']}",
            f"  Capital needed: ${plan['total_capital_allocated']}",
            f"  Actual risk:    ${plan['actual_risk']}",
            f"  Target 1 (1R):  ${plan['target_1']}",
            f"  Target 2 (2R):  ${plan['target_2']}",
            "",
        ]

        exit_plan = sheet.get("exit_plan", {})
        if exit_plan:
            lines.append("EXIT PLAN (Trade Management Sheet)")
            lines.append(f"  Hard stop:      ${exit_plan['hard_stop']}")
            lines.append(f"  Initial trail:  ${exit_plan['initial_trailing_stop']}")
            for target in exit_plan.get("scale_out_ladder", []):
                lines.append(
                    f"  {target['label']}: sell {target['shares_to_sell']} @ "
                    f"${target['price']} ({target['pct_position']*100:.0f}%)"
                )
            lines.append(f"  {exit_plan['runner_rule']}")
            lines.append(f"  {exit_plan['time_stop_rule']}")
            lines.append(f"  Max hold:       {exit_plan['max_hold_days']} days")
    else:
        lines.append(f"NO TRADE: {plan.get('reason', 'Setup criteria not met')}")

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


def save_trade_sheet(sheet: dict) -> Path:
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ticker = sheet["ticker"]
    path = Path(REPORTS_DIR) / f"{ticker}_{ts}.json"
    path.write_text(json.dumps(sheet, indent=2), encoding="utf-8")

    txt_path = path.with_suffix(".txt")
    txt_path.write_text(format_trade_sheet(sheet), encoding="utf-8")
    return txt_path


def format_status(state) -> str:
    lines = [
        "=" * 64,
        "  AGENT STATUS",
        "=" * 64,
        f"  Trust Score:  {state.trust_score}%",
        f"  Total Trades: {state.total_trades}",
        f"  Win Rate:     {(state.wins/state.total_trades*100) if state.total_trades else 0:.1f}%",
        f"  Total P&L:    ${state.total_pnl:+.2f}",
        "",
    ]

    open_pos = [p for p in state.active_positions if p.status == "OPEN"]
    if open_pos:
        lines.append(f"  OPEN POSITIONS ({len(open_pos)}):")
        for p in open_pos:
            lines.append(
                f"    {p.ticker}: entry ${p.entry_price} | stop ${p.stop_loss} | "
                f"{p.shares_remaining}/{p.shares} shares | since {p.entry_date}"
            )
            if p.scale_outs_hit:
                lines.append(f"      Scale-outs hit: {', '.join(p.scale_outs_hit)}")
    else:
        lines.append("  No open positions.")

    if state.trade_history:
        lines.append("")
        lines.append("  RECENT CLOSED TRADES:")
        for t in state.trade_history[-5:]:
            lines.append(
                f"    {t.ticker}: ${t.pnl:+.2f} ({t.r_multiple:+.1f}R) "
                f"[{t.exit_reason}] {t.exit_date}"
            )

    lines.append("=" * 64)
    return "\n".join(lines)