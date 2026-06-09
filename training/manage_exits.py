"""Live exit manager — sells per the VALIDATED backtester rules, automatically.

Run daily. For each open Alpaca position it replays `backtester.simulate_trade_forward`'s
exit logic on the daily bars since entry and, if an exit has triggered, closes the
position at market. This removes the after-the-fact / emotional sell decision and
makes live behavior match what produced the +0.07R edge.

Exits implemented (long side):
  * HARD_STOP  -1R  -> already live as the Alpaca bracket stop (reported here)
  * TRAILING_STOP   -> 2x ATR below the high-water mark, armed once +1R is tagged
  * TIME_STOP       -> from day 10, exit if total return < 0.25R (a laggard)
  * MAX_HOLD        -> force-close at 14 trading days
Scale-outs at +1R/+2R are a documented v2 refinement (they ladder HOW winners are
sold; trailing/max-hold already ensure winners ARE sold). Stateless replay = robust.

    python -m training.manage_exits           # DRY-RUN (no orders)
    python -m training.manage_exits --live      # close positions that should exit
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import requests

from agents.indicators import calculate_atr
from training.alpaca_exec import EXEC_LOG, LOG, _creds, _get, _headers
from training.backtester import MAX_HOLDING_DAYS

TRAIL_ATR_MULT = 2.0


def _resolve(ticker: str, pos: dict) -> dict | None:
    """entry, stop, entry_date, qty for an open position — executions.csv first,
    then Alpaca's filled buy order + paper_trades.csv stop (covers pre-log trades)."""
    qty = float(pos["qty"])
    if EXEC_LOG.exists():
        e = pd.read_csv(EXEC_LOG)
        e = e[e["ticker"] == ticker].sort_values("submitted_at")
        if len(e):
            r = e.iloc[-1]
            return {"entry": float(r["entry_signal"]), "stop": float(r["stop"]),
                    "entry_date": pd.to_datetime(r["submitted_at"]).date(), "qty": qty}
    # fallback: latest filled buy from Alpaca + stop from the signal log
    orders = _get("/v2/orders", status="all", limit=200)
    buys = [o for o in orders if o["symbol"] == ticker and o["side"] == "buy"
            and o.get("filled_at")]
    stop = None
    if LOG.exists():
        log = pd.read_csv(LOG).sort_values("asof")
        s = log[log["ticker"] == ticker]["stop"]
        stop = float(s.iloc[-1]) if len(s) else None
    if not buys or stop is None:
        return None
    b = sorted(buys, key=lambda o: o["filled_at"])[-1]
    return {"entry": float(b["filled_avg_price"]), "stop": stop,
            "entry_date": pd.to_datetime(b["filled_at"]).date(), "qty": qty}


def _replay(entry: float, stop: float, bars: pd.DataFrame, atr: pd.Series):
    """Replay the long-side exit logic over bars AFTER entry. Returns
    (should_exit, reason, day_offset)."""
    risk = entry - stop
    if risk <= 0:
        return (False, "BAD_RISK", 0)
    t1 = entry + risk
    armed = False
    extreme = entry
    trailing = stop
    for d in range(min(len(bars), MAX_HOLDING_DAYS)):
        day_offset = d + 1
        high, low, close = (float(bars.iloc[d]["High"]), float(bars.iloc[d]["Low"]),
                            float(bars.iloc[d]["Close"]))
        if low <= stop:
            return (True, "HARD_STOP", day_offset)        # Alpaca handles, reported
        if close >= t1:
            armed = True
            extreme = max(extreme, close)
        if armed:
            a = float(atr.loc[bars.index[d]]) if bars.index[d] in atr.index else risk
            extreme = max(extreme, close)
            trailing = max(trailing, extreme - a * TRAIL_ATR_MULT)
            if low <= trailing and trailing > stop:
                return (True, "TRAILING_STOP", day_offset)
        if day_offset >= 10:
            total_r = (close - entry) / risk                # laggard check
            if total_r < 0.25:
                return (True, "TIME_STOP", day_offset)
    if len(bars) >= MAX_HOLDING_DAYS:
        return (True, "MAX_HOLD", MAX_HOLDING_DAYS)
    return (False, "HOLD", len(bars))


def _bars_since(ticker: str, entry_date) -> tuple[pd.DataFrame, pd.Series]:
    import yfinance as yf
    start = pd.Timestamp(entry_date) - pd.Timedelta(days=40)
    df = yf.Ticker(ticker).history(start=start.date().isoformat(), auto_adjust=False)
    if df.empty:
        return df, pd.Series(dtype=float)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    atr = calculate_atr(df, 14)
    after = df[df.index > pd.Timestamp(entry_date).normalize()]
    return after, atr


def _close(base, hdr, symbol, qty, live):
    if not live:
        return "DRY-RUN"
    try:
        # cancel the lingering bracket stop, then market-sell the position
        for o in _get("/v2/orders", status="open", symbols=symbol):
            requests.delete(f"{base}/v2/orders/{o['id']}", headers=hdr, timeout=20)
        order = {"symbol": symbol, "qty": qty, "side": "sell", "type": "market",
                 "time_in_force": "day"}
        r = requests.post(f"{base}/v2/orders", headers=hdr, json=order, timeout=20)
        r.raise_for_status()
        return f"SOLD id={r.json()['id'][:8]}"
    except Exception as e:
        return f"ERROR {getattr(e,'response',None) and e.response.text or e}"[:40]


def manage(live: bool):
    kid, sec, base = _creds()
    hdr = _headers(kid, sec)
    positions = [p for p in _get("/v2/positions") if float(p["qty"]) > 0]
    print("=" * 64)
    print(f"  EXIT MANAGER  ({'LIVE' if live else 'DRY-RUN'})  — {len(positions)} open positions")
    print("=" * 64)
    print(f"  {'ticker':<7}{'days':>5}{'entry':>9}{'last':>9}{'R':>7}  decision")
    print("  " + "-" * 56)
    for p in positions:
        tk = p["symbol"]
        info = _resolve(tk, p)
        if not info:
            print(f"  {tk:<7}{'?':>5}  could not resolve entry/stop — skipped")
            continue
        bars, atr = _bars_since(tk, info["entry_date"])
        days = len(bars)
        last = float(p["current_price"])
        r_now = (last - info["entry"]) / (info["entry"] - info["stop"]) if info["entry"] != info["stop"] else float("nan")
        do_exit, reason, _ = _replay(info["entry"], info["stop"], bars, atr)
        if do_exit and reason == "HARD_STOP":
            note = "stop (Alpaca handles)"          # don't double-submit
        elif do_exit:
            note = f"EXIT [{reason}] -> {_close(base, hdr, tk, info['qty'], live)}"
        else:
            note = f"hold (day {days}/{MAX_HOLDING_DAYS})"
        print(f"  {tk:<7}{days:>5}{info['entry']:>9.2f}{last:>9.2f}{r_now:>7.2f}  {note}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="actually submit sells (default dry-run)")
    a = p.parse_args(argv)
    manage(a.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
