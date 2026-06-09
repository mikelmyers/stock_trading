"""Reconcile live Alpaca paper fills against the logged signals.

Answers: of the signals we logged, which got filled, what's open, what closed,
and what REAL R did they make -- vs the blind expectation (~+0.07R/trade).

Matching is EXACT by Alpaca order id via executions.csv (written by alpaca_exec
on submit). For any fill without an executions row (e.g. trades submitted before
order-id logging existed), it falls back to matching by ticker against
paper_trades.csv. Credentials from env; run where Alpaca is reachable.

    python -m training.reconcile
"""
from __future__ import annotations

import pandas as pd

from training.alpaca_exec import EXEC_LOG, LOG, _get

EXPECTED_R = 0.07  # blind walk-forward expectation


def _stop_lookups():
    """order_id -> (stop, model_p) from executions.csv; ticker -> (stop, model_p)
    from paper_trades.csv as fallback for un-logged fills."""
    by_id = {}
    if EXEC_LOG.exists():
        e = pd.read_csv(EXEC_LOG)
        for r in e.itertuples():
            by_id[str(r.order_id)] = (float(r.stop), getattr(r, "model_p", float("nan")))
    by_ticker = {}
    if LOG.exists():
        log = pd.read_csv(LOG).sort_values("asof")
        s = log.groupby("ticker")["stop"].last()
        p = log.groupby("ticker")["model_p"].last()
        by_ticker = {tk: (float(s[tk]), float(p.get(tk, float("nan")))) for tk in s.index}
    return by_id, by_ticker


def reconcile():
    by_id, by_ticker = _stop_lookups()
    orders = _get("/v2/orders", status="all", nested="true", limit=500)
    last_px = {p["symbol"]: float(p["current_price"]) for p in _get("/v2/positions")}

    open_t, closed_t, unmatched = [], [], 0
    for o in orders:
        if o.get("side") != "buy" or not o.get("filled_avg_price"):
            continue                                   # not a filled entry
        tk, entry = o["symbol"], float(o["filled_avg_price"])
        meta = by_id.get(str(o["id"])) or by_ticker.get(tk)
        if meta is None:
            unmatched += 1
            continue
        stop, mp = meta
        denom = entry - stop
        leg = next((l for l in (o.get("legs") or []) if l.get("side") == "sell"), None)
        if leg and leg.get("status") == "filled" and leg.get("filled_avg_price"):
            exit_ = float(leg["filled_avg_price"])
            R = (exit_ - entry) / denom if denom else float("nan")
            closed_t.append((tk, entry, exit_, R, mp))
        else:
            last = last_px.get(tk)
            uR = (last - entry) / denom if (last and denom) else float("nan")
            open_t.append((tk, entry, last, uR, mp))

    print("=" * 66)
    print("  PAPER-TRADE RECONCILIATION  (logged signals vs real Alpaca fills)")
    print("=" * 66)
    src = "exact order-id match" if EXEC_LOG.exists() else "ticker fallback (no executions.csv yet)"
    print(f"  matching: {src}  |  open: {len(open_t)}  closed: {len(closed_t)}"
          + (f"  |  {unmatched} fills unmatched (manual orders?)" if unmatched else ""))

    if open_t:
        print("\n  OPEN POSITIONS (unrealized R vs logged stop):")
        print(f"  {'ticker':<7}{'entry':>10}{'last':>10}{'uR':>7}{'P(win)':>8}")
        for tk, e, last, uR, mp in open_t:
            print(f"  {tk:<7}{e:>10.2f}{(last or 0):>10.2f}{uR:>7.2f}{mp:>8.2f}")

    if closed_t:
        print("\n  CLOSED ROUND-TRIPS (realized R vs logged stop):")
        print(f"  {'ticker':<7}{'entry':>10}{'exit':>10}{'R':>7}")
        rs = []
        for tk, e, x, R, mp in closed_t:
            rs.append(R)
            print(f"  {tk:<7}{e:>10.2f}{x:>10.2f}{R:>7.2f}")
        rs = [r for r in rs if r == r]
        if rs:
            mean = sum(rs) / len(rs); win = sum(r > 0 for r in rs) / len(rs)
            print("\n  " + "-" * 50)
            print(f"  REALIZED: {len(rs)} trades  win {win*100:.0f}%  mean R {mean:+.4f}")
            print(f"  Expected (blind walk-forward): {EXPECTED_R:+.4f}R, ~57% win, Sharpe ~1.8")
            verdict = "ON TRACK" if mean >= EXPECTED_R * 0.5 else "BELOW EXPECTATION"
            print(f"  -> {verdict}  ({'too few trades to judge' if len(rs) < 30 else 'meaningful sample'})")
    else:
        print("\n  No closed trades yet -- let the forward record accumulate.")


def main(argv=None):
    reconcile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
