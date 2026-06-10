"""Reconcile live Alpaca paper fills against the logged signals.

Answers: of the signals we logged, which got filled, what's open, what closed,
what REAL R did they make vs the blind expectation (+0.076R re-stated), and —
critically — whether broker state has drifted from the logs (unmatched fills,
positions with no signal row). This is a FEEDBACK LOOP, not a report: run with
--strict in the daily flow and a non-zero exit blocks anything downstream.

Matching is EXACT by Alpaca order id via executions.csv (written by alpaca_exec
on submit). For any fill without an executions row (e.g. trades submitted before
order-id logging existed), it falls back to matching by ticker against
paper_trades.csv. Credentials from env; run where Alpaca is reachable.

    python -m training.reconcile             # report
    python -m training.reconcile --strict    # exit 1 on any drift
"""
from __future__ import annotations

import argparse

import pandas as pd

from training.alpaca_exec import EXEC_LOG, LOG, _get
from training.fill_quality import fetch_all_orders

EXPECTED_R = 0.076   # re-stated blind walk-forward expectation (PR #7)
MIN_MEANINGFUL_TRADES = 30


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


def gather(get_fn=None) -> dict:
    """Pull broker truth and join it to the signal logs. Returns
    {open, closed, unmatched_fills, orphan_positions} — orphans are broker
    positions with NO signal/exec row at all (manual trades or lost logs)."""
    get_fn = get_fn or _get
    by_id, by_ticker = _stop_lookups()
    orders = fetch_all_orders(get_fn)
    positions = get_fn("/v2/positions")
    last_px = {p["symbol"]: float(p["current_price"]) for p in positions}

    open_t, closed_t, unmatched = [], [], []
    matched_symbols = set()
    for o in orders:
        if o.get("side") != "buy" or not o.get("filled_avg_price"):
            continue                                   # not a filled entry
        tk, entry = o["symbol"], float(o["filled_avg_price"])
        meta = by_id.get(str(o["id"])) or by_ticker.get(tk)
        if meta is None:
            unmatched.append(tk)
            continue
        matched_symbols.add(tk)
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

    known = set(by_ticker) | matched_symbols
    orphans = sorted({p["symbol"] for p in positions} - known)
    return {"open": open_t, "closed": closed_t,
            "unmatched_fills": unmatched, "orphan_positions": orphans}


def reconcile(get_fn=None, strict: bool = False) -> int:
    g = gather(get_fn)
    open_t, closed_t = g["open"], g["closed"]
    unmatched, orphans = g["unmatched_fills"], g["orphan_positions"]

    print("=" * 66)
    print("  PAPER-TRADE RECONCILIATION  (logged signals vs real Alpaca fills)")
    print("=" * 66)
    src = "exact order-id match" if EXEC_LOG.exists() else "ticker fallback (no executions.csv yet)"
    print(f"  matching: {src}  |  open: {len(open_t)}  closed: {len(closed_t)}")
    drift = False
    if unmatched:
        drift = True
        print(f"  [DRIFT] {len(unmatched)} filled buys with NO signal/exec row: "
              f"{', '.join(sorted(set(unmatched))[:8])}")
    if orphans:
        drift = True
        print(f"  [DRIFT] broker positions with NO log entry (unmanaged!): "
              f"{', '.join(orphans[:8])}")

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
            print(f"  Expected (re-stated blind walk-forward): {EXPECTED_R:+.4f}R/trade")
            verdict = "ON TRACK" if mean >= EXPECTED_R * 0.5 else "BELOW EXPECTATION"
            judge = ("meaningful sample" if len(rs) >= MIN_MEANINGFUL_TRADES
                     else "too few trades to judge")
            print(f"  -> {verdict}  ({judge})")
    else:
        print("\n  No closed trades yet -- let the forward record accumulate.")

    if strict and drift:
        print("\n  STRICT MODE: drift detected -> exit 1")
        return 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Reconcile fills vs signal logs")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on unmatched fills / orphan positions")
    a = p.parse_args(argv)
    return reconcile(strict=a.strict)


if __name__ == "__main__":
    raise SystemExit(main())
