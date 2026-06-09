"""Runner scanner CLI — generate + log the condition-vector for today's candidates.

    python -m runner.scan --mock                 # offline self-test (synthetic)
    python -m runner.scan --episode ep001         # live Alpaca scan, log candidates
The classifier learns from the accumulating log; for now this is the data-generation
engine. Decisions/sizing/discipline come in the next components.
"""
from __future__ import annotations

import argparse

from runner.datasource import AlpacaSource, MockSource
from runner.logger import RunnerLog
from runner.risk import RiskConfig, RiskEngine, RunnerState


def run(use_mock: bool, episode: str, log: bool, equity: float | None, leverage: float):
    src = MockSource() if use_mock else AlpacaSource()
    cvs = src.scan()
    cands = [c for c in cvs if c.is_candidate]
    green = [c for c in cands if c.green_light]

    print("=" * 74)
    print(f"  RUNNER SCAN  ({'MOCK' if use_mock else 'ALPACA'})  episode={episode}")
    print("=" * 74)
    print(f"  scanned {len(cvs)}  |  candidates {len(cands)}  |  green-light {len(green)}")

    decisions = {}
    if equity is not None:
        eng = RiskEngine(RiskConfig(max_leverage=leverage))
        state = RunnerState.load(episode, equity)
        print(f"  equity ${equity:,.0f}  lev {leverage}x  | day P&L ${state.daily_pnl:+.0f}  "
              f"streak {state.consecutive_losses}L  trades {state.trades_today}"
              + (f"  | LOCKED until {state.locked_until[:16]}" if state.locked_until else ""))
        print(f"\n  {'sym':<6}{'px':>7}{'gap%':>6}{'rvol':>6}{'float':>8}  {'action':<6}{'sh':>5}{'stop':>8}{'risk$':>7}  reason")
        print("  " + "-" * 76)
        for c in sorted(cands, key=lambda x: (not x.green_light, x.symbol)):
            d = eng.evaluate(c, state)
            decisions[c.symbol] = d.action
            flt = f"{c.float_shares/1e6:.0f}M" if c.float_shares else "?"
            print(f"  {c.symbol:<6}{(c.price or 0):>7.2f}{(c.gap_pct or 0):>6.0f}"
                  f"{(c.rvol or 0):>6.1f}{flt:>8}  {d.action:<6}{d.shares:>5}"
                  f"{(d.stop or 0):>8.2f}{d.risk_dollars:>7.0f}  {d.reason}")
    else:
        print(f"  {'sym':<6}{'px':>7}{'gap%':>7}{'rvol':>6}{'dVWAP%':>8}{'float':>10}  flags")
        print("  " + "-" * 64)
        for c in sorted(cands, key=lambda x: (not x.green_light, x.symbol)):
            flt = f"{c.float_shares/1e6:.1f}M" if c.float_shares else "?"
            tag = "GREEN-LIGHT" if c.green_light else (c.blowup_flags or "watch")
            print(f"  {c.symbol:<6}{(c.price or 0):>7.2f}{(c.gap_pct or 0):>7.1f}"
                  f"{(c.rvol or 0):>6.1f}{(c.dist_vwap_pct or 0):>8.1f}{flt:>10}  {tag}")

    if log:
        n = RunnerLog(episode).log_candidates(cands, decisions)
        print(f"\n  logged {n} candidates -> runner/data/runner_candidates.csv")
    else:
        print("\n  (not logged — pass --log to persist)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="use synthetic data (offline)")
    p.add_argument("--episode", default="ep001", help="episode/stake id")
    p.add_argument("--log", action="store_true", help="append candidates to the ledger")
    p.add_argument("--equity", type=float, default=None, help="run the risk engine at this equity (shows sized decisions)")
    p.add_argument("--leverage", type=float, default=1.0, help="buying-power multiple (1=cash, 4=intraday margin)")
    a = p.parse_args(argv)
    run(a.mock, a.episode, a.log, a.equity, a.leverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
