"""Daily ops report — one page that says whether the system is healthy.

Composes account state, open positions, the realized forward record vs the
re-stated blind benchmark, fill quality, drift flags, and the experiment trial
count into a single text report: printed, and archived to
reports/daily/YYYY-MM-DD.txt so the forward record has a tamper-evident trail.

    python -m training.daily_report
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from config import REPORTS_DIR
from training import experiments
from training.alpaca_exec import _get
from training.fill_quality import LEDGER as FQ_LEDGER
from training.fill_quality import report as fq_report
from training.reconcile import EXPECTED_R, MIN_MEANINGFUL_TRADES, gather

OUT_DIR = Path(REPORTS_DIR) / "daily"
TRADES_PER_YR_BENCH = 67   # re-stated blind walk-forward (PR #7)


def _record_start() -> dt.date | None:
    """First order-submission date — the start of the forward record."""
    from training.alpaca_exec import EXEC_LOG
    if not Path(EXEC_LOG).exists():
        return None
    sub = pd.read_csv(EXEC_LOG)["submitted_at"]
    if sub.empty:
        return None
    return pd.to_datetime(sub, utc=True, format="mixed").min().date()


def build_report(get_fn=None, now: dt.datetime | None = None) -> str:
    get_fn = get_fn or _get
    now = now or dt.datetime.now(dt.timezone.utc)
    acct = get_fn("/v2/account")
    g = gather(get_fn)

    L: list[str] = []
    L.append("=" * 66)
    L.append(f"  DAILY OPS REPORT  {now.date().isoformat()}")
    L.append("=" * 66)
    L.append(f"  equity ${float(acct['equity']):,.2f}   "
             f"buying power ${float(acct['buying_power']):,.2f}   "
             f"status {acct.get('status', '?')}")

    drift = g["unmatched_fills"] or g["orphan_positions"]
    L.append(f"  health: {'DRIFT DETECTED — investigate before trading' if drift else 'logs and broker agree'}")
    if g["unmatched_fills"]:
        L.append(f"    unmatched fills: {', '.join(sorted(set(g['unmatched_fills']))[:8])}")
    if g["orphan_positions"]:
        L.append(f"    unmanaged positions: {', '.join(g['orphan_positions'][:8])}")

    L.append(f"\n  OPEN ({len(g['open'])}):")
    if g["open"]:
        for tk, e, last, uR, mp in g["open"]:
            L.append(f"    {tk:<7} entry {e:<9.2f} last {(last or 0):<9.2f} "
                     f"uR {uR:+.2f}  P(win) {mp:.2f}")
    else:
        L.append("    none")

    rs = [t[3] for t in g["closed"] if t[3] == t[3]]   # (tk, entry, exit, R, mp)
    L.append(f"\n  FORWARD RECORD: {len(rs)} closed trades")
    if rs:
        mean = sum(rs) / len(rs)
        win = sum(r > 0 for r in rs) / len(rs)
        line = f"    mean R {mean:+.4f}  win {win*100:.0f}%"
        start = _record_start()
        if start:
            days = max((now.date() - start).days, 1)
            line += f"  pace ~{len(rs) / days * 365.25:.0f} trades/yr"
        L.append(line)
        L.append(f"    benchmark: {EXPECTED_R:+.4f}R at ~{TRADES_PER_YR_BENCH}/yr "
                 f"({'meaningful sample' if len(rs) >= MIN_MEANINGFUL_TRADES else f'need {MIN_MEANINGFUL_TRADES}+ to judge'})")
    else:
        L.append("    no closed trades yet")

    if Path(FQ_LEDGER).exists():
        fq = pd.read_csv(FQ_LEDGER, dtype={"order_id": str})
        L.append("")
        L.append(fq_report(fq))

    L.append(f"\n  experiment ledger: {experiments.trial_count()} trials logged")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Daily ops report")
    p.add_argument("--no-archive", action="store_true", help="print only")
    a = p.parse_args(argv)
    text = build_report()
    print(text)
    if not a.no_archive:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{dt.date.today().isoformat()}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\n  archived -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
