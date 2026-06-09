"""Runner execution — submit sized entries and run the intraday exits on Alpaca paper.

Reuses the swing model's paper-only/credential guards. DRY-RUN by default. Long-side
momentum scalps: entry = market buy + resting -1R stop (bracket); exits driven by
`runner.exits.plan_exit` on intraday bars (scale +1R/+2R, hard trail, VWAP-loss cut,
flat-by-close). Runs locally (the Claude sandbox can't reach Alpaca)."""
from __future__ import annotations

import datetime as dt
import math

import requests

from training.alpaca_exec import _creds, _headers
from runner.exits import plan_exit
from runner.logger import RunnerLog
from runner.risk import Decision

DATA = "https://data.alpaca.markets"


def submit_entries(decisions: list[Decision], live: bool, episode: str = "ep"):
    kid, sec, base = _creds()
    hdr = _headers(kid, sec)
    print(f"  ENTRIES ({'LIVE' if live else 'DRY-RUN'}):")
    for d in decisions:
        if d.action != "take" or d.shares < 1:
            continue
        coid = f"run-{d.symbol}-{dt.date.today().isoformat()}"
        order = {"symbol": d.symbol, "qty": d.shares, "side": "buy", "type": "market",
                 "time_in_force": "day", "order_class": "oto", "client_order_id": coid,
                 "stop_loss": {"stop_price": round(d.stop, 2)}}
        status = "DRY-RUN"
        if live:
            try:
                r = requests.post(f"{base}/v2/orders", headers=hdr, json=order, timeout=20)
                r.raise_for_status()
                oid = r.json()["id"]; status = f"submitted {oid[:8]}"
                RunnerLog.log_outcome  # (outcome filled later by the exit/EOD pass)
            except Exception as e:
                status = f"ERR {getattr(e,'response',None) and e.response.text or e}"[:40]
        print(f"    {d.symbol:<6} buy {d.shares} @ mkt  stop {d.stop:.2f}  risk ${d.risk_dollars:.0f}  {status}")


def _bars_since(symbol, kid, sec, entry_time) -> tuple[list[dict], float]:
    """Intraday 1-min bars since entry (with VWAP) + minutes to the 16:00 ET close."""
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    j = requests.get(f"{DATA}/v2/stocks/{symbol}/bars", headers=h,
                     params={"timeframe": "1Min", "start": entry_time, "limit": 400}, timeout=20).json()
    bars = [dict(high=b["h"], low=b["l"], close=b["c"], vwap=b.get("vw")) for b in j.get("bars", [])]
    now_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
    mins_to_close = (16 * 60) - (now_et.hour * 60 + now_et.minute)
    return bars, mins_to_close


def manage_exits(live: bool, get_entry_stop):
    """For each open position, replay the intraday exit rules and submit sells.
    `get_entry_stop(symbol) -> (entry, stop, entry_time_iso, orig_qty)`."""
    kid, sec, base = _creds()
    hdr = _headers(kid, sec)
    positions = requests.get(f"{base}/v2/positions", headers=hdr, timeout=20).json()
    print(f"  EXITS ({'LIVE' if live else 'DRY-RUN'}):")
    for p in positions:
        sym, cur = p["symbol"], float(p["qty"])
        if cur <= 0:
            continue
        info = get_entry_stop(sym)
        if not info:
            print(f"    {sym:<6} could not resolve entry — skipped"); continue
        entry, stop, etime, orig = info
        bars, mtc = _bars_since(sym, kid, sec, etime)
        exit_all, reason, frac = plan_exit(entry, stop, bars, mtc)
        if exit_all:
            qty = cur; note = f"EXIT ALL [{reason}]"
        else:
            target = math.floor(orig * frac)
            qty = max(cur - target, 0); note = f"scale -> hold {target} [{reason}]" if qty else f"hold [{reason}]"
        if qty < 1:
            print(f"    {sym:<6} {note}"); continue
        status = "DRY-RUN"
        if live:
            try:
                for o in requests.get(f"{base}/v2/orders", headers=hdr,
                                      params={"status": "open", "symbols": sym}, timeout=20).json():
                    requests.delete(f"{base}/v2/orders/{o['id']}", headers=hdr, timeout=20)
                r = requests.post(f"{base}/v2/orders", headers=hdr, timeout=20,
                                  json={"symbol": sym, "qty": int(qty), "side": "sell",
                                        "type": "market", "time_in_force": "day"})
                r.raise_for_status(); status = f"sold {int(qty)} {r.json()['id'][:8]}"
            except Exception as e:
                status = f"ERR {getattr(e,'response',None) and e.response.text or e}"[:40]
        print(f"    {sym:<6} {note}  sell {int(qty)} -> {status}")
