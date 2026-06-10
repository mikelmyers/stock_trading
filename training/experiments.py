"""Experiment registry — the multiple-testing ledger.

Every selection scheme, feature set, or objective you try is a draw against
the same data; without a ledger the harness slowly turns into an overfitting
machine (the fundamentals thread was saved by exactly one disciplined
walk-forward tiebreaker). Log every experiment here, including the failures.

Records are append-only JSONL (training/experiments.jsonl, committed) and a
rendered markdown table (training/EXPERIMENTS.md) for humans.

    python -m training.experiments log --name "short-interest features" \
        --dataset training/ml/datasets/survivorship_free_v2.parquet \
        --config '{"features": "+si_ratio,si_dtc", "objective": "win"}' \
        --metric "top10_net_r=+0.0301" --baseline "top10_net_r=+0.0278" \
        --verdict adopt --notes "per-year delta positive 6/8"
    python -m training.experiments list
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
LEDGER = BASE / "experiments.jsonl"
RENDERED = BASE / "EXPERIMENTS.md"

VERDICTS = ("adopt", "reject", "inconclusive", "running")


def _dataset_hash(path: str | Path | None) -> str | None:
    """Cheap content identity: size + mtime-free sample hash of the file head.
    Full-file hashing of a multi-GB parquet is wasteful; 64KB head + size is
    enough to detect 'the dataset changed under me'."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return f"missing:{p.name}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(65536))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def log_experiment(name: str, config: dict | str, verdict: str,
                   dataset: str | None = None, metric: str = "",
                   baseline: str = "", notes: str = "",
                   ledger: Path | None = None) -> dict:
    ledger = Path(ledger) if ledger else LEDGER   # resolve at call time
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    cfg = config if isinstance(config, dict) else json.loads(config or "{}")
    rec = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "config": cfg,
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
        "dataset": dataset,
        "dataset_hash": _dataset_hash(dataset),
        "metric": metric,
        "baseline": baseline,
        "verdict": verdict,
        "notes": notes,
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    render(ledger)
    return rec


def load(ledger: Path | None = None) -> list[dict]:
    ledger = Path(ledger) if ledger else LEDGER
    if not ledger.exists():
        return []
    return [json.loads(line) for line in
            ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def trial_count(ledger: Path | None = None) -> int:
    """How many draws have been taken against the data — the number that the
    significance of any single 'winning' experiment must be discounted by."""
    return len(load(ledger))


def render(ledger: Path | None = None, out: Path | None = None) -> str:
    out = Path(out) if out else RENDERED   # resolve at call time
    rows = load(ledger)
    lines = [
        "# Experiment ledger",
        "",
        f"**{len(rows)} experiments logged.** Every one of these is a draw "
        "against the same data — discount any single win accordingly "
        "(deflated-Sharpe logic). Auto-generated from experiments.jsonl; "
        "do not edit by hand.",
        "",
        "| date | experiment | metric | baseline | verdict | notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ts'][:10]} | {r['name']} | {r.get('metric','')} "
            f"| {r.get('baseline','')} | **{r['verdict']}** | {r.get('notes','')} |")
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--name", required=True)
    lg.add_argument("--config", default="{}", help="JSON config of the experiment")
    lg.add_argument("--dataset", default=None)
    lg.add_argument("--metric", default="", help="headline result, e.g. top10_net_r=+0.03")
    lg.add_argument("--baseline", default="", help="what it's compared against")
    lg.add_argument("--verdict", required=True, choices=VERDICTS)
    lg.add_argument("--notes", default="")
    sub.add_parser("list")
    a = p.parse_args(argv)
    if a.cmd == "log":
        rec = log_experiment(a.name, a.config, a.verdict, dataset=a.dataset,
                             metric=a.metric, baseline=a.baseline, notes=a.notes)
        print(f"logged [{rec['config_hash']}] {rec['name']} -> {rec['verdict']}"
              f"  (trial #{trial_count()})")
    else:
        for r in load():
            print(f"{r['ts'][:10]}  {r['verdict']:<13} {r['name']}  {r.get('metric','')}")
        print(f"\n{trial_count()} total trials against the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
