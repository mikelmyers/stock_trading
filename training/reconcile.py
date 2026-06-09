"""Reconcile live Alpaca paper fills against the logged signals.

Answers: of the signals we logged, which got filled, what's open, what closed,
and what REAL R did they make -- vs the blind expectation (~+0.07R/trade).
This is the forward track record accumulating. Credentials from env (see
alpaca_exec); run wherever Alpaca is reachable (not the Claude sandbox).

    python -m training.reconcile
"""
from __future__ import annotations

import collections

import pandas as pd

from training.alpaca_exec import LOG, _get

EXPECTED_R = 0.07  # blind walk-forward expectation


def _fifo_closed(fills):
    """Pair buy/sell FILL activities per symbol (FIFO) -> list of round-trips."""
    buys = collections.defaultdict(collections.deque)
    closed = []
    for f in sorted(fills, key=lambda x: x.get("transaction_time", "")):
        sym, side = f["symbol"], f["side"]
        px, qty = float(f["price"]), float(f["qty"])
        if side == "buy":
            buys[sym].append([px, qty])
        else:  # sell closes prior buys
            while qty > 1e-9 and buys[sym]:
                lot = buys[sym][0]
                take = min(qty, lot[1])
                closed.append({"ticker": sym, "entry": lot[0], "exit": px, "qty": take})
                lot[1] -= take; qty -= take
                if lot[1] <= 1e-9:
                    buys[sym].popleft()
    return closed


def reconcile():
    log = pd.read_csv(LOG)
    stop_of = log.sort_values("asof").groupby("ticker")["stop"].last().to_dict()
    p_of = log.sort_values("asof").groupby("ticker")["model_p"].last().to_dict()

    positions = _get("/v2/positions")
    fills = _get("/v2/account/activities", activity_types="FILL")
    closed = _fifo_closed(fills)

    print("=" * 68)
    print("  PAPER-TRADE RECONCILIATION  (logged signals vs real Alpaca fills)")
    print("=" * 68)
    print(f"  logged signals: {len(log)}  |  open positions: {len(positions)}  |  "
          f"closed round-trips: {len(closed)}")

    if positions:
        print("\n  OPEN POSITIONS (unrealized R vs logged stop):")
        print(f"  {'ticker':<7}{'qty':>6}{'entry':>10}{'last':>10}{'uR':>7}{'P(win)':>8}")
        for p in positions:
            tk = p["symbol"]; e = float(p["avg_entry_price"]); last = float(p["current_price"])
            stop = stop_of.get(tk)
            uR = (last - e) / (e - stop) if stop and (e - stop) else float("nan")
            print(f"  {tk:<7}{float(p['qty']):>6.0f}{e:>10.2f}{last:>10.2f}{uR:>7.2f}"
                  f"{p_of.get(tk, float('nan')):>8.2f}")

    if closed:
        rs = []
        print("\n  CLOSED ROUND-TRIPS (realized R vs logged stop):")
        print(f"  {'ticker':<7}{'entry':>10}{'exit':>10}{'R':>7}")
        for c in closed:
            stop = stop_of.get(c["ticker"])
            R = (c["exit"] - c["entry"]) / (c["entry"] - stop) if stop and (c["entry"] - stop) else float("nan")
            rs.append(R)
            print(f"  {c['ticker']:<7}{c['entry']:>10.2f}{c['exit']:>10.2f}{R:>7.2f}")
        rs = [r for r in rs if r == r]
        if rs:
            mean = sum(rs) / len(rs); win = sum(r > 0 for r in rs) / len(rs)
            print("\n  " + "-" * 50)
            print(f"  REALIZED: {len(rs)} trades  win {win*100:.0f}%  mean R {mean:+.4f}")
            print(f"  Expected (blind walk-forward): {EXPECTED_R:+.4f}R, ~57% win, Sharpe ~1.8")
            verdict = ("ON TRACK" if mean >= EXPECTED_R * 0.5 else "BELOW EXPECTATION")
            print(f"  -> {verdict}  ({'too few trades to judge' if len(rs) < 30 else 'meaningful sample'})")
    else:
        print("\n  No closed trades yet -- let the forward record accumulate.")


def main(argv=None):
    reconcile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
