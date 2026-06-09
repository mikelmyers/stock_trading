"""Alpaca paper-trade execution for the validated book.

Credentials come from the ENVIRONMENT only (never hardcode / commit):
    export APCA_API_KEY_ID=...        # your key id
    export APCA_API_SECRET_KEY=...    # your secret (NEVER paste in chat/commits)
    export APCA_API_BASE_URL=https://paper-api.alpaca.markets

Daily flow:  python -m training.live_signals --top 10        # writes paper_trades.csv
             python -m training.alpaca_exec --selftest        # verify creds/market
             python -m training.alpaca_exec --from-log         # DRY-RUN by default
             python -m training.alpaca_exec --from-log --live  # actually submit

Sizing is the validated 1%-risk rule: qty = floor(equity * risk% / (entry-stop)),
submitted as a bracket (market entry + stop-loss). Cannot run from the Claude
sandbox (Alpaca host not in its allowlist); run on a GH Action or your machine.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import requests

LOG = Path(__file__).resolve().parent / "ml" / "datasets" / "paper_trades.csv"


def _creds():
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    if not kid or not sec:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment "
                         "(never in code). Secret must NOT be pasted in chat or committed.")
    if "paper" not in base:
        raise SystemExit(f"Refusing to run against a non-paper endpoint: {base}")
    return kid, sec, base.rstrip("/")


def _headers(kid, sec):
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _get(path, **params):
    kid, sec, base = _creds()
    r = requests.get(f"{base}{path}", headers=_headers(kid, sec), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def selftest():
    acct = _get("/v2/account")
    clock = _get("/v2/clock")
    print(f"  Connected. account status={acct['status']}  equity=${float(acct['equity']):,.2f}  "
          f"buying_power=${float(acct['buying_power']):,.2f}")
    print(f"  Market open={clock['is_open']}  next_open={clock['next_open']}")
    return acct


def submit_book(signals: pd.DataFrame, risk_pct: float, live: bool):
    kid, sec, base = _creds()
    acct = _get("/v2/account")
    equity = float(acct["equity"])
    print(f"\n  Account equity ${equity:,.2f}  |  {'LIVE SUBMIT' if live else 'DRY-RUN (no orders sent)'}  "
          f"|  risk {risk_pct}%/trade")
    print(f"  {'ticker':<7}{'side':<6}{'qty':>6}{'entry':>10}{'stop':>10}{'$risk':>9}  status")
    print("  " + "-" * 60)
    for s in signals.itertuples():
        per_share = abs(s.entry - s.stop)
        if per_share <= 0:
            continue
        dollar_risk = equity * risk_pct / 100.0
        qty = int(math.floor(dollar_risk / per_share))
        if qty < 1:
            print(f"  {s.ticker:<7}{'buy':<6}{0:>6}{s.entry:>10.2f}{s.stop:>10.2f}"
                  f"{dollar_risk:>9.0f}  skip (stop too wide for 1 share)")
            continue
        order = {"symbol": s.ticker, "qty": qty, "side": "buy", "type": "market",
                 "time_in_force": "gtc", "order_class": "oto",
                 "stop_loss": {"stop_price": round(s.stop, 2)}}
        status = "DRY-RUN"
        if live:
            try:
                resp = requests.post(f"{base}/v2/orders", headers=_headers(kid, sec),
                                     json=order, timeout=20)
                resp.raise_for_status()
                status = f"submitted id={resp.json()['id'][:8]}"
            except Exception as e:
                status = f"ERROR {getattr(e,'response',None) and e.response.text or e}"[:40]
        print(f"  {s.ticker:<7}{'buy':<6}{qty:>6}{s.entry:>10.2f}{s.stop:>10.2f}"
              f"{qty*per_share:>9.0f}  {status}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true", help="verify creds + market clock")
    p.add_argument("--from-log", action="store_true", help="submit the latest-date signals from paper_trades.csv")
    p.add_argument("--live", action="store_true", help="actually submit (default is dry-run)")
    p.add_argument("--risk", type=float, default=1.0, help="%% equity risk per trade")
    a = p.parse_args(argv)

    if a.selftest:
        selftest()
        return 0
    if a.from_log:
        if not LOG.exists():
            raise SystemExit(f"No signal log at {LOG}; run live_signals.py first.")
        df = pd.read_csv(LOG)
        latest = df[df["asof"] == df["asof"].max()].copy()
        print(f"  {len(latest)} signals from {df['asof'].max()} (latest in log)")
        submit_book(latest, a.risk, a.live)
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
