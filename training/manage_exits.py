"""Live exit manager — sells per the VALIDATED backtester rules, automatically.

Run daily. For each open Alpaca position it replays `backtester.simulate_trade_forward`'s
long-side exit logic on the daily bars since entry and brings the live position to the
target state: scale out 33% at +1R and 33% at +2R, trail the runner at 2x ATR (armed
once +1R is tagged), TIME_STOP laggards from day 10, MAX_HOLD at 14 days. The -1R hard
stop rests as the Alpaca bracket; after each scale-out the stop is re-placed for the
remaining shares. Stateless replay (original qty taken from the buy's filled_qty) =>
robust across days and matches the +0.07R numbers.

    python -m training.manage_exits           # DRY-RUN (no orders)
    python -m training.manage_exits --live      # bring positions to target state
"""
from __future__ import annotations

import argparse
import math

import pandas as pd
import requests

from agents.indicators import calculate_atr
from training.alpaca_exec import EXEC_LOG, LOG, _creds, _get, _headers
from training.backtester import MAX_HOLDING_DAYS

TRAIL_ATR_MULT = 2.0
SCALE_PCT = 0.33                 # fraction of ORIGINAL position sold at each target


def _resolve(ticker: str, pos: dict) -> dict | None:
    """entry, stop, entry_date, orig_qty (entry size), cur_qty (now)."""
    cur_qty = float(pos["qty"])
    if EXEC_LOG.exists():
        e = pd.read_csv(EXEC_LOG)
        e = e[e["ticker"] == ticker].sort_values("submitted_at")
        if len(e):
            r = e.iloc[-1]
            return {"entry": float(r["entry_signal"]), "stop": float(r["stop"]),
                    "entry_date": pd.to_datetime(r["submitted_at"]).date(),
                    "orig_qty": float(r["qty"]), "cur_qty": cur_qty}
    # fallback for pre-log positions: filled buy order + signal-log stop
    orders = _get("/v2/orders", status="all", limit=200)
    buys = [o for o in orders if o["symbol"] == ticker and o["side"] == "buy" and o.get("filled_at")]
    stop = None
    if LOG.exists():
        s = pd.read_csv(LOG).sort_values("asof")
        s = s[s["ticker"] == ticker]["stop"]
        stop = float(s.iloc[-1]) if len(s) else None
    if not buys or stop is None:
        return None
    b = sorted(buys, key=lambda o: o["filled_at"])[-1]
    return {"entry": float(b["filled_avg_price"]), "stop": stop,
            "entry_date": pd.to_datetime(b["filled_at"]).date(),
            "orig_qty": float(b["filled_qty"]), "cur_qty": cur_qty}


def _replay(entry: float, stop: float, bars: pd.DataFrame, atr: pd.Series):
    """Return (exit_all, reason, remaining_fraction). Long side."""
    risk = entry - stop
    if risk <= 0:
        return (False, "BAD_RISK", 1.0)
    t1, t2 = entry + risk, entry + 2 * risk
    scale1 = scale2 = armed = False
    frac, realized_r, extreme, trailing = 1.0, 0.0, entry, stop
    for d in range(min(len(bars), MAX_HOLDING_DAYS)):
        day = d + 1
        high, low, close = (float(bars.iloc[d]["High"]), float(bars.iloc[d]["Low"]),
                            float(bars.iloc[d]["Close"]))
        if low <= stop:
            return (True, "HARD_STOP", 0.0)
        if close >= t1 and not scale1:
            scale1, armed = True, True
            frac -= SCALE_PCT; realized_r += SCALE_PCT * 1.0; extreme = max(extreme, close)
        if close >= t2 and not scale2:
            scale2 = True
            frac -= SCALE_PCT; realized_r += SCALE_PCT * 2.0; extreme = max(extreme, close)
        if armed or close >= t1:
            armed = True
            a = float(atr.loc[bars.index[d]]) if bars.index[d] in atr.index else risk
            extreme = max(extreme, close)
            trailing = max(trailing, extreme - a * TRAIL_ATR_MULT)
            if low <= trailing and trailing > stop:
                return (True, "TRAILING_STOP", 0.0)
        if day >= 10:
            total_r = realized_r + (close - entry) / risk * frac
            if total_r < 0.25:
                return (True, "TIME_STOP", 0.0)
    if len(bars) >= MAX_HOLDING_DAYS:
        return (True, "MAX_HOLD", 0.0)
    return (False, "HOLD", round(frac, 2))


def _bars_since(ticker: str, entry_date):
    import yfinance as yf
    start = (pd.Timestamp(entry_date) - pd.Timedelta(days=40)).date().isoformat()
    df = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if df.empty:
        return df, pd.Series(dtype=float)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    atr = calculate_atr(df, 14)
    return df[df.index > pd.Timestamp(entry_date).normalize()], atr


def _cancel_open(base, hdr, symbol):
    for o in _get("/v2/orders", status="open", symbols=symbol):
        requests.delete(f"{base}/v2/orders/{o['id']}", headers=hdr, timeout=20)


def _order(base, hdr, body, live):
    if not live:
        return "DRY-RUN"
    try:
        r = requests.post(f"{base}/v2/orders", headers=hdr, json=body, timeout=20)
        r.raise_for_status()
        return f"ok id={r.json()['id'][:8]}"
    except Exception as e:
        return f"ERR {getattr(e,'response',None) and e.response.text or e}"[:36]


def manage(live: bool):
    kid, sec, base = _creds()
    hdr = _headers(kid, sec)
    positions = [p for p in _get("/v2/positions") if float(p["qty"]) > 0]
    print("=" * 70)
    print(f"  EXIT MANAGER  ({'LIVE' if live else 'DRY-RUN'})  — {len(positions)} open positions")
    print("=" * 70)
    print(f"  {'ticker':<7}{'day':>4}{'entry':>9}{'last':>9}{'R':>6}  decision")
    print("  " + "-" * 60)
    for p in positions:
        tk = p["symbol"]
        info = _resolve(tk, p)
        if not info:
            print(f"  {tk:<7}{'?':>4}  could not resolve entry/stop — skipped")
            continue
        bars, atr = _bars_since(tk, info["entry_date"])
        last = float(p["current_price"])
        risk = info["entry"] - info["stop"]
        r_now = (last - info["entry"]) / risk if risk else float("nan")
        exit_all, reason, frac = _replay(info["entry"], info["stop"], bars, atr)
        cur = info["cur_qty"]

        if exit_all and reason == "HARD_STOP":
            note = "stop resting at Alpaca (no action)"
        elif exit_all:
            _cancel_open(base, hdr, tk)
            st = _order(base, hdr, {"symbol": tk, "qty": cur, "side": "sell",
                                    "type": "market", "time_in_force": "day"}, live)
            note = f"EXIT ALL [{reason}] {cur:g}sh -> {st}"
        else:
            target = math.floor(info["orig_qty"] * frac)
            sell_qty = cur - target
            if sell_qty >= 1:
                lvl = "+1R" if frac > 0.5 else "+2R"
                _cancel_open(base, hdr, tk)
                st = _order(base, hdr, {"symbol": tk, "qty": int(sell_qty), "side": "sell",
                                        "type": "market", "time_in_force": "day"}, live)
                if target >= 1:   # re-place the hard stop on the runner
                    _order(base, hdr, {"symbol": tk, "qty": int(target), "side": "sell",
                                       "type": "stop", "stop_price": round(info["stop"], 2),
                                       "time_in_force": "gtc"}, live)
                note = f"SCALE {lvl}: sell {int(sell_qty)}sh, hold {target}sh -> {st}"
            else:
                note = f"hold {cur:g}sh (day {len(bars)}/{MAX_HOLDING_DAYS}, frac {frac})"
        print(f"  {tk:<7}{len(bars):>4}{info['entry']:>9.2f}{last:>9.2f}{r_now:>6.2f}  {note}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="bring positions to target (default dry-run)")
    a = p.parse_args(argv)
    manage(a.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
