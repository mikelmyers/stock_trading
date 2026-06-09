"""Runner orchestrator — the intraday loop that ties the system together.

Each cycle: manage exits on open positions -> scan for A+ candidates -> size with
the discipline engine -> submit entries -> log everything. Flat-by-close is enforced
by the exit rules (plan_exit returns FLAT_CLOSE near the bell). DRY-RUN by default;
--mock runs fully offline (no broker calls) so the wiring is testable.

    python -m runner.run --mock --equity 500             # offline single cycle
    python -m runner.run --equity 500 --leverage 4        # live DRY-RUN, one cycle
    python -m runner.run --equity 500 --live --loop        # live paper, intraday loop
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from runner.datasource import AlpacaSource, MockSource
from runner.logger import RunnerLog
from runner.risk import RiskConfig, RiskEngine, RunnerState

ENTRIES = Path(__file__).resolve().parent / "data" / "runner_entries.csv"


def _record_entry(episode, d, now_iso):
    row = {"episode": episode, "symbol": d.symbol, "entry": d.entry, "stop": d.stop,
           "qty": d.shares, "entry_time": now_iso}
    pd.DataFrame([row]).to_csv(ENTRIES, mode="a", header=not ENTRIES.exists(), index=False)


def _entry_resolver(symbol):
    if not ENTRIES.exists():
        return None
    e = pd.read_csv(ENTRIES)
    e = e[e["symbol"] == symbol]
    if not len(e):
        return None
    r = e.iloc[-1]
    return (float(r["entry"]), float(r["stop"]), str(r["entry_time"]), float(r["qty"]))


def cycle(mock: bool, live: bool, equity: float, leverage: float, episode: str):
    src = MockSource() if mock else AlpacaSource()
    eng = RiskEngine(RiskConfig(max_leverage=leverage))
    state = RunnerState.load(episode, equity)
    now = dt.datetime.now(dt.timezone.utc)
    print(f"\n--- cycle {now.strftime('%H:%M:%S')}Z  ep={episode}  eq=${equity:,.0f}  "
          f"dayP&L=${state.daily_pnl:+.0f}  streak={state.consecutive_losses}L"
          + (f"  LOCKED" if state.locked_until and now.isoformat() < state.locked_until else "") + " ---")

    # 1) exits first — sell what the rules say to sell
    if mock:
        print("  (mock: skipping live exit management)")
    else:
        from runner import execution
        execution.manage_exits(live, _entry_resolver)

    # 2) scan + size
    cvs = src.scan()
    cands = [c for c in cvs if c.is_candidate]
    decisions, takes = {}, []
    for c in cands:
        d = eng.evaluate(c, state)
        decisions[c.symbol] = d.action
        if d.action == "take":
            takes.append(d)
    print(f"  scanned {len(cvs)} | candidates {len(cands)} | takes {len(takes)}")
    for c in cands:
        d = eng.evaluate(c, state)
        flag = "TAKE" if d.action == "take" else "skip"
        print(f"    {c.symbol:<6} {flag:<5} {d.shares:>4}sh @ {c.price:<7.2f} stop {d.stop or 0:<7.2f} {d.reason}")

    # 3) log the whole pool (the classifier's future training data)
    RunnerLog(episode).log_candidates(cands, decisions)

    # 4) submit entries
    if takes and not mock:
        from runner import execution
        execution.submit_entries(takes, live, episode)
        if live:
            for d in takes:
                _record_entry(episode, d, now.isoformat())
            state.trades_today += len(takes)
            state.open_positions += len(takes)
            state.save()
    elif takes:
        print(f"  (mock: would submit {[t.symbol for t in takes]})")
    return takes


def _et_now():
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)


def loop(live, equity, leverage, episode, interval_s):
    print(f"runner loop every {interval_s}s until ~15:58 ET (Ctrl-C to stop)")
    while True:
        et = _et_now()
        mins = et.hour * 60 + et.minute
        if mins < 9 * 60 + 30:
            print("pre-market — waiting"); time.sleep(interval_s); continue
        if mins >= 15 * 60 + 58:
            print("near close — final exit pass"); cycle(False, live, equity, leverage, episode); break
        cycle(False, live, equity, leverage, episode)
        time.sleep(interval_s)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="offline single cycle (no broker)")
    p.add_argument("--live", action="store_true", help="submit to Alpaca paper (default dry-run)")
    p.add_argument("--loop", action="store_true", help="run the intraday loop")
    p.add_argument("--equity", type=float, default=500.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--episode", default="ep001")
    p.add_argument("--interval", type=int, default=120, help="loop seconds between cycles")
    a = p.parse_args(argv)
    if a.loop:
        loop(a.live, a.equity, a.leverage, a.episode, a.interval)
    else:
        cycle(a.mock, a.live, a.equity, a.leverage, a.episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
