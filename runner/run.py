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

from runner import clock
from runner.datasource import AlpacaSource, MockSource
from runner.logger import RunnerLog
from runner.risk import RiskConfig, RiskEngine, RunnerState

ENTRIES = Path(__file__).resolve().parent / "data" / "runner_entries.csv"


def _broker_sync(state: RunnerState):
    """Sync open-position count from the broker for RUNNER trades only.

    The paper account may also hold swing-book positions; counting those would
    block the runner's max_concurrent gate. Virtual stake equity (e.g. $500 for
    ep500) stays in state — not overwritten by total broker equity.
    """
    import requests
    from training.alpaca_exec import _creds, _headers
    kid, sec, base = _creds()
    hdr = _headers(kid, sec)
    pos = requests.get(f"{base}/v2/positions", headers=hdr, timeout=20)
    pos.raise_for_status()
    broker_pos = {p["symbol"]: p for p in pos.json()}
    runner_syms: set[str] = set()
    if ENTRIES.exists():
        e = pd.read_csv(ENTRIES)
        if "episode" in e.columns:
            e = e[e["episode"] == state.episode_id]
        runner_syms = set(e["symbol"].astype(str))
    runner_open = sum(
        1 for s in runner_syms
        if s in broker_pos and float(broker_pos[s]["qty"]) > 0
    )
    state.open_positions = runner_open
    if state.synced_date != state.date:
        state.day_start_equity = state.equity
        state.synced_date = state.date
    state.daily_pnl = state.equity - state.day_start_equity
    state.save()


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


def _runner_symbols(episode: str) -> set[str]:
    """Symbols the runner episode has entered — excludes swing-book ETFs."""
    if not ENTRIES.exists():
        return set()
    e = pd.read_csv(ENTRIES)
    if "episode" in e.columns:
        e = e[e["episode"] == episode]
    return set(e["symbol"].astype(str))


def cycle(mock: bool, live: bool, equity: float, leverage: float, episode: str):
    src = MockSource() if mock else AlpacaSource()
    cfg = RiskConfig(max_leverage=leverage)
    eng = RiskEngine(cfg)
    state = RunnerState.load(episode, equity)
    if not mock:
        try:
            _broker_sync(state)
        except Exception as e:
            print(f"  broker sync FAILED ({e}) — gates running on local state")
    now = dt.datetime.now(dt.timezone.utc)
    print(f"\n--- cycle {now.strftime('%H:%M:%S')}Z  ep={episode}  eq=${state.equity:,.0f}  "
          f"dayP&L=${state.daily_pnl:+.0f}  streak={state.consecutive_losses}L"
          + (f"  LOCKED" if state.locked_until and now.isoformat() < state.locked_until else "") + " ---")

    # 1) exits first — sell what the rules say to sell
    if mock:
        print("  (mock: skipping live exit management)")
    else:
        from runner import execution
        execution.manage_exits(live, _entry_resolver, state=state, cfg=cfg,
                              symbols=_runner_symbols(episode))

    # 2) scan + size — evaluate ONCE per candidate, and count provisional takes
    #    against the caps so a single scan can't blow through max_concurrent /
    #    max_trades_per_day (every candidate used to see the same state).
    cvs = src.scan()
    cands = [c for c in cvs if c.is_candidate]
    decisions, takes = {}, []
    for c in cands:
        d = eng.evaluate(c, state)
        decisions[c.symbol] = d
        if d.action == "take":
            takes.append(d)
            state.trades_today += 1
            state.open_positions += 1
    print(f"  scanned {len(cvs)} | candidates {len(cands)} | takes {len(takes)}")
    for c in cands:
        d = decisions[c.symbol]
        flag = "TAKE" if d.action == "take" else "skip"
        print(f"    {c.symbol:<6} {flag:<5} {d.shares:>4}sh @ {c.price:<7.2f} stop {d.stop or 0:<7.2f} {d.reason}")

    # 3) log the whole scanned pool — incl. near-misses when candidates=0
    dec_map = {s: d.action for s, d in decisions.items()}
    for cv in cvs:
        dec_map.setdefault(cv.symbol, "pass")
    RunnerLog(episode).log_candidates(cvs, dec_map)

    # 4) submit entries — only broker-ACCEPTED orders update state/the ledger
    if takes and not mock:
        from runner import execution
        accepted = execution.submit_entries(takes, live, episode)
        if live:
            rejected = len(takes) - len(accepted)
            state.trades_today -= rejected
            state.open_positions -= rejected
            for d in accepted:
                _record_entry(episode, d, now.isoformat())
            state.save()
    elif takes:
        print(f"  (mock: would submit {[t.symbol for t in takes]})")
    return takes


def loop(live, equity, leverage, episode, interval_s):
    print(f"runner loop every {interval_s}s until ~15:58 ET (Ctrl-C to stop)")
    while True:
        et = clock.et_now()
        if not clock.is_weekday(et):
            print("weekend — market closed"); time.sleep(interval_s); continue
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
