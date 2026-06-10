"""Fill-quality ledger — measures the one number the backtest cannot.

The simulator fills at the signal close (legacy) or next open (realism mode);
the live path market-buys whenever the order reaches the exchange. The gap
between signal price and actual fill is YOUR cost, it converges in weeks (not
the 2.5 years the edge itself needs), and if it exceeds ~0.02R round-trip it
eats most of the +0.076R edge. Track it from trade one.

    python -m training.fill_quality            # update ledger + print report
    python -m training.fill_quality --report   # report from the local ledger only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from training.alpaca_exec import EXEC_LOG, _get

LEDGER = Path(__file__).resolve().parent / "ml" / "datasets" / "fill_quality.csv"
COST_ASSUMPTION_R = 0.019      # the validated book's all-in round-trip assumption


def fetch_all_orders(get_fn=None, status: str = "all", page_limit: int = 500):
    """Every order, paginated — /v2/orders silently truncates at limit=500,
    which is how a forward record quietly loses its history."""
    get_fn = get_fn or _get
    out: list[dict] = []
    until = None
    while True:
        params = {"status": status, "limit": page_limit,
                  "direction": "desc", "nested": "true"}
        if until:
            params["until"] = until
        batch = get_fn("/v2/orders", **params)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page_limit:
            break
        until = batch[-1]["submitted_at"]
    seen: set = set()
    dedup = []
    for o in out:
        if o["id"] not in seen:
            seen.add(o["id"])
            dedup.append(o)
    return dedup


def compute_fill_quality(executions: pd.DataFrame, orders: list[dict]) -> pd.DataFrame:
    """One row per filled entry: signal price vs actual fill, in % and in R."""
    by_id = {str(o["id"]): o for o in orders}
    rows = []
    for r in executions.itertuples():
        o = by_id.get(str(r.order_id))
        if not o or not o.get("filled_avg_price"):
            continue
        signal, stop = float(r.entry_signal), float(r.stop)
        fill = float(o["filled_avg_price"])
        risk = signal - stop
        if risk <= 0:
            continue
        rows.append({
            "order_id": str(r.order_id), "ticker": r.ticker,
            "submitted_at": r.submitted_at, "filled_at": o.get("filled_at", ""),
            "signal_price": signal, "fill_price": fill, "stop": stop,
            "qty": float(o.get("filled_qty") or r.qty),
            "slippage_pct": round((fill - signal) / signal * 100, 4),
            "slippage_r": round((fill - signal) / risk, 4),
        })
    return pd.DataFrame(rows, columns=[
        "order_id", "ticker", "submitted_at", "filled_at", "signal_price",
        "fill_price", "stop", "qty", "slippage_pct", "slippage_r"])


def update_ledger(new_rows: pd.DataFrame, ledger: Path | None = None) -> pd.DataFrame:
    """Idempotent append (deduped by order_id) so reruns never double-count."""
    ledger = Path(ledger) if ledger else LEDGER
    if ledger.exists():
        old = pd.read_csv(ledger, dtype={"order_id": str})
        merged = pd.concat([old, new_rows], ignore_index=True)
        merged = merged.drop_duplicates("order_id", keep="first")
    else:
        merged = new_rows
    ledger.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ledger, index=False)
    return merged


def summarize(fq: pd.DataFrame) -> dict:
    if fq.empty:
        return {"n": 0, "verdict": "no filled entries yet"}
    r = fq["slippage_r"]
    # entry slippage happens twice per round trip (entry + exit) plus spread;
    # 2x mean entry slip is the conservative round-trip estimate
    est_round_trip = 2 * float(r.mean())
    ok = est_round_trip <= COST_ASSUMPTION_R
    return {
        "n": int(len(fq)),
        "mean_slippage_r": round(float(r.mean()), 4),
        "median_slippage_r": round(float(r.median()), 4),
        "worst_slippage_r": round(float(r.max()), 4),
        "mean_slippage_pct": round(float(fq["slippage_pct"].mean()), 4),
        "est_round_trip_r": round(est_round_trip, 4),
        "cost_assumption_r": COST_ASSUMPTION_R,
        "verdict": ("WITHIN the modeled cost" if ok else
                    "EXCEEDS the modeled cost — the live edge is smaller than the backtest's"),
    }


def report(fq: pd.DataFrame) -> str:
    s = summarize(fq)
    lines = ["=" * 64, "  FILL QUALITY  (signal price vs actual fill)", "=" * 64]
    if not s["n"]:
        lines.append("  no filled entries yet — let the record accumulate")
        return "\n".join(lines)
    lines += [
        f"  fills: {s['n']}",
        f"  entry slippage: mean {s['mean_slippage_r']:+.4f}R "
        f"({s['mean_slippage_pct']:+.3f}%)  median {s['median_slippage_r']:+.4f}R  "
        f"worst {s['worst_slippage_r']:+.4f}R",
        f"  est. round-trip cost ~{s['est_round_trip_r']:+.4f}R "
        f"vs modeled {s['cost_assumption_r']:.3f}R -> {s['verdict']}",
    ]
    worst = fq.nlargest(min(5, len(fq)), "slippage_r")
    lines.append("  worst fills:")
    for t in worst.itertuples():
        lines.append(f"    {t.ticker:<7} signal {t.signal_price:<8.2f} "
                     f"fill {t.fill_price:<8.2f} slip {t.slippage_r:+.3f}R")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fill-quality ledger")
    p.add_argument("--report", action="store_true",
                   help="report from the local ledger only (no API calls)")
    a = p.parse_args(argv)
    if a.report:
        fq = pd.read_csv(LEDGER, dtype={"order_id": str}) if LEDGER.exists() else pd.DataFrame()
        print(report(fq if len(fq) else compute_fill_quality(pd.DataFrame(), [])))
        return 0
    if not EXEC_LOG.exists():
        print(f"no executions log at {EXEC_LOG}; submit some orders first")
        return 0
    ex = pd.read_csv(EXEC_LOG, dtype={"order_id": str})
    fq = compute_fill_quality(ex, fetch_all_orders())
    merged = update_ledger(fq)
    print(report(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
