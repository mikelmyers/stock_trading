"""Command-line interface for the trading research agent."""

from __future__ import annotations

import argparse
import sys

from agents.teacher import format_feedback_prompt
from core import (
    analyze_ticker,
    close_position_manual,
    export_sheet,
    get_status,
    monitor_positions,
    scan_market,
    submit_feedback,
    track_ticker,
)
from reports import format_trade_sheet
from state import StateManager


def cmd_scan(_args: argparse.Namespace) -> int:
    print("=" * 64)
    print("  MULTI-STYLE TRADING SCANNER")
    print("=" * 64)

    alerts = scan_market()
    if not alerts:
        print("\nNo valid breakouts detected.")
        return 0

    print(f"\nFound {len(alerts)} setup(s):\n")
    for sheet in alerts:
        print(format_trade_sheet(sheet))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    sheet = analyze_ticker(args.ticker)
    print(format_trade_sheet(sheet))

    if args.save:
        _, path = export_sheet(args.ticker)
        print(f"\nSaved to: {path}")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    try:
        pos = track_ticker(
            args.ticker,
            entry=args.entry,
            stop=args.stop,
            shares=args.shares,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Now tracking {pos.ticker}")
    print(f"  Entry:  ${pos.entry_price}")
    print(f"  Stop:   ${pos.stop_loss}")
    print(f"  Shares: {pos.shares}")
    print(f"  Risk:   ${pos.max_risk}")
    print(f"  Trail:  ${pos.trailing_stop}")
    print("\nRun `python cli.py monitor` daily to check exits.")
    return 0


def cmd_monitor(_args: argparse.Namespace) -> int:
    print("=" * 64)
    print("  POSITION MONITOR")
    print("=" * 64)

    actions = monitor_positions()
    if not actions:
        print("\nNo active positions to monitor.")
        return 0

    for action in actions:
        if action["type"] == "EXIT":
            r = action["result"]
            rev = action["review"]
            print(f"\n  [EXIT] {action['position'].ticker}: {r['message']}")
            print(f"         P&L: ${rev['pnl']:+.2f} ({rev['r_multiple']:+.1f}R)")
            print(f"         Trust: {rev['trust_score']}% ({rev['risk_tier']})")
            for lesson in rev["lessons"]:
                print(f"         -> {lesson}")
            print(f"\n{format_feedback_prompt(action['position'].ticker)}")

        elif action["type"] == "SCALE_OUT":
            r = action["result"]
            ev = action.get("event")
            pnl_str = f" | P&L: ${ev.pnl:+.2f}" if ev else ""
            print(f"\n  [SCALE OUT] {action['position'].ticker}: {r['message']}{pnl_str}")

        elif action["type"] == "HOLD":
            print(f"  [HOLD] {action['result']['message']}")

        elif action["type"] == "ERROR":
            print(f"  [!] {action['position'].ticker}: {action['error']}")

    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    print(get_status())
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    review = close_position_manual(args.ticker, args.price, args.reason)
    if not review:
        print(f"No open position found for {args.ticker.upper()}")
        return 1

    print(f"Closed {args.ticker.upper()} @ ${args.price}")
    print(f"  P&L: ${review['pnl']:+.2f} ({review['r_multiple']:+.1f}R)")
    print(f"  Trust: {review['trust_score']}% ({review['risk_tier']})")
    print(f"\n{format_feedback_prompt(args.ticker.upper())}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    aligned = args.aligned.lower() in ("yes", "y", "true", "1")
    ok = submit_feedback(args.ticker, args.fidelity, aligned, args.notes or "")
    if not ok:
        print(f"No closed trade found for {args.ticker.upper()}")
        return 1

    state = StateManager().state
    print(f"Feedback recorded for {args.ticker.upper()}")
    print(f"  Updated trust score: {state.trust_score}%")
    return 0


def cmd_history(_args: argparse.Namespace) -> int:
    state = StateManager().state
    if not state.trade_history:
        print("No trade history yet.")
        return 0

    print("=" * 64)
    print("  TRADE HISTORY")
    print("=" * 64)
    for t in state.trade_history:
        print(
            f"  {t.ticker:6s} {t.entry_date} -> {t.exit_date}  "
            f"${t.pnl:+7.2f} ({t.r_multiple:+.1f}R)  [{t.exit_reason}]"
        )
        if t.user_feedback:
            print(f"         Notes: {t.user_feedback}")

    if state.scale_out_log:
        print("\n  SCALE-OUT LOG:")
        for s in state.scale_out_log:
            print(f"    {s.ticker} {s.date}: sold {s.shares_sold} @ ${s.price} ({s.level}) ${s.pnl:+.2f}")

    print(f"\n  Total P&L: ${state.total_pnl:+.2f}")
    print("=" * 64)
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    sheet, path = export_sheet(args.ticker)
    print(format_trade_sheet(sheet))
    print(f"\nSaved to: {path}")
    return 0


def cmd_full(_args: argparse.Namespace) -> int:
    cmd_scan(_args)
    print()
    return cmd_monitor(_args)


def cmd_checklist(_args: argparse.Namespace) -> int:
    from training.checklist import print_status
    print(print_status())
    return 0


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """2k checkpoint training — use after each new module batch."""
    from training.checklist import CHECKPOINT_SIMULATIONS
    args.simulations = CHECKPOINT_SIMULATIONS
    args.no_bootstrap = False
    args.force = False
    print("  [CHECKPOINT] 2k run — hold full 10k until checklist complete\n")
    return cmd_train(args)


def cmd_download_data(args: argparse.Namespace) -> int:
    """Pre-download training universe OHLCV (resumable per-ticker cache)."""
    from config import TRAINING_USE_MAX_HISTORY, TRAINING_YEARS
    from training.history import download_history
    from training.universe import load_training_universe, universe_stats

    tickers = load_training_universe(refresh_sp500=args.refresh_universe)
    stats = universe_stats(tickers)
    period = "max" if TRAINING_USE_MAX_HISTORY or args.years >= 10 else f"{args.years}y"

    print("=" * 64)
    print("  DOWNLOAD TRAINING DATA")
    print("=" * 64)
    print(f"  Universe:   {stats['total']} tickers (S&P 500 file: {stats['sp500_cached']})")
    print(f"  Period:     {period}")
    print(f"  Refresh:    {args.refresh_cache or args.refresh_universe}")
    print()

    history = download_history(
        tickers, years=args.years, refresh=args.refresh_cache,
    )
    if history:
        sample = next(iter(history.values()))
        print()
        print(f"  Loaded {len(history)}/{len(tickers)} tickers")
        print(f"  Bars per ticker: ~{len(sample)} "
              f"({sample.index[0].date()} → {sample.index[-1].date()})")
    else:
        print("  No data downloaded — check network or retry with --refresh-cache")
        return 1
    print("=" * 64)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from training.backtester import run_training
    from training.checklist import (
        CHECKPOINT_SIMULATIONS, FINAL_SIMULATIONS,
        is_ready_for_final_train, print_status,
    )

    if args.simulations >= FINAL_SIMULATIONS and not args.force:
        if not is_ready_for_final_train():
            print("BLOCKED: Full 10k training requires all modules complete.")
            print(print_status())
            print("\nUse: python cli.py checkpoint   (2k after each batch)")
            print("Or:  python cli.py train -n 10000 --force  (override)")
            return 1

    label = "CHECKPOINT" if args.simulations <= CHECKPOINT_SIMULATIONS else "FULL"
    print("=" * 64)
    print(f"  TRAINING [{label}] — {args.simulations:,} simulations on REAL historical data")
    print("=" * 64)
    print()
    from training.universe import universe_stats

    ustats = universe_stats()
    print("  Data source: Yahoo Finance (real OHLCV, not synthetic)")
    print(f"  Universe:    {ustats['total']} tickers (S&P 500 + ETFs + extras)")
    print("  Method:      Walk history → simulate exits → bootstrap expand")
    print()

    summary = run_training(
        simulations=args.simulations,
        years=args.years,
        use_bootstrap=not args.no_bootstrap,
        workers=args.workers,
        refresh_cache=getattr(args, "refresh_cache", False),
    )

    print()
    print("=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Tickers loaded:   {summary.get('tickers_scanned', '?')}"
          f"/{summary.get('tickers_requested', '?')}")
    print(f"  History period:   {summary.get('history_period', summary.get('years'))}")
    print(f"  Total simulations:  {summary['count']:,}")
    print(f"  Real historical:  {summary.get('real_trades', 0):,}")
    print(f"  Bootstrapped:     {summary.get('bootstrapped', 0):,}")
    print(f"  Win rate:         {summary['win_rate']}%")
    print(f"  Expectancy:       {summary['expectancy']}R per trade")
    print(f"  Avg P&L:          ${summary['avg_pnl_dollars']} (at $10 risk)")
    print(f"  Max drawdown:     {summary.get('max_drawdown_r', 0)}R")
    qm = summary.get("quant_metrics", {})
    if qm:
        print(f"  Profit factor:    {qm.get('profit_factor', 'N/A')}")
        print(f"  Sharpe ratio:     {qm.get('sharpe_ratio', 'N/A')}")
    print()
    learned = summary.get("learned_params", {})
    print(f"  Learned min score: {learned.get('min_setup_score')}")
    print(f"  Enabled setups:    {', '.join(learned.get('enabled_setups', []))}")
    perf = learned.get("setup_performance", {})
    if perf:
        print()
        print("  PER-SETUP PERFORMANCE:")
        for name, stats in perf.items():
            flag = "ON " if stats.get("enabled") else "OFF"
            print(f"    [{flag}] {name:14s}  real={stats.get('real_count',0):4d}  "
                  f"win={stats.get('win_rate',0):5.1f}%  exp={stats.get('expectancy',0):+.3f}R")
    opts_bt = learned.get("options_backtest", {})
    if opts_bt.get("count"):
        print(f"  Options backtest:  {opts_bt.get('summary', '')}")
    print(f"  Saved params:      learned_params.json")
    print(f"  Full report:       {summary.get('result_file')}")
    print("=" * 64)
    return 0


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    from config import TRAINING_YEARS

    parser.add_argument(
        "--years", type=int, default=TRAINING_YEARS,
        help=f"Years of history; >=10 uses max available (default: {TRAINING_YEARS})",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Re-download OHLCV even if per-ticker cache exists",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading_agent",
        description="Swing trading research agent — scan, analyze, track, and manage exits.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="Scan watchlist for breakout setups")

    p_analyze = sub.add_parser("analyze", help="Full analysis on one ticker")
    p_analyze.add_argument("ticker", help="Stock symbol (e.g. NVDA)")
    p_analyze.add_argument("--save", action="store_true", help="Save trade sheet to reports/")

    p_track = sub.add_parser("track", help="Start tracking a position you entered")
    p_track.add_argument("ticker", help="Stock symbol")
    p_track.add_argument("--entry", type=float, help="Entry price (manual override)")
    p_track.add_argument("--stop", type=float, help="Stop loss price")
    p_track.add_argument("--shares", type=float, help="Number of shares")

    sub.add_parser("monitor", help="Check open positions for exit signals")
    sub.add_parser("status", help="Show trust score, open positions, and stats")
    sub.add_parser("full", help="Run scan + monitor in one pass")

    p_close = sub.add_parser("close", help="Manually close a tracked position")
    p_close.add_argument("ticker")
    p_close.add_argument("--price", type=float, required=True, help="Exit price")
    p_close.add_argument("--reason", default="MANUAL", help="Exit reason label")

    p_review = sub.add_parser("review", help="Rate a closed trade (feeds trust score)")
    p_review.add_argument("ticker")
    p_review.add_argument("--fidelity", type=float, required=True, help="Setup quality 0.0-1.0")
    p_review.add_argument("--aligned", required=True, help="Followed exit plan? yes/no")
    p_review.add_argument("--notes", default="", help="Optional notes")

    sub.add_parser("history", help="Show all closed trades and scale-outs")

    p_sheet = sub.add_parser("sheet", help="Generate and save a trade sheet")
    p_sheet.add_argument("ticker")

    p_train = sub.add_parser("train", help="Run mass simulations on real historical data")
    p_train.add_argument(
        "--simulations", "-n", type=int, default=10_000,
        help="Target simulation count (default: 10000)",
    )
    _add_training_args(p_train)
    p_train.add_argument(
        "--no-bootstrap", action="store_true",
        help="Only use real historical setups (no resampling)",
    )
    p_train.add_argument(
        "--force", action="store_true",
        help="Allow full 10k even if build checklist incomplete",
    )

    p_download = sub.add_parser(
        "download-data",
        help="Pre-download S&P 500 + extras OHLCV (resumable cache)",
    )
    _add_training_args(p_download)
    p_download.add_argument(
        "--refresh-universe", action="store_true",
        help="Re-fetch S&P 500 constituent list from GitHub",
    )

    sub.add_parser("checklist", help="Show build progress before final 10k")
    p_checkpoint = sub.add_parser(
        "checkpoint", help="Run 2k checkpoint training (default after each batch)",
    )
    _add_training_args(p_checkpoint)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "track": cmd_track,
        "monitor": cmd_monitor,
        "status": cmd_status,
        "close": cmd_close,
        "review": cmd_review,
        "history": cmd_history,
        "sheet": cmd_sheet,
        "full": cmd_full,
        "train": cmd_train,
        "download-data": cmd_download_data,
        "checklist": cmd_checklist,
        "checkpoint": cmd_checkpoint,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())