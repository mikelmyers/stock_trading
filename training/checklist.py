"""Track which modules are built before the final 10k training run."""

from __future__ import annotations

import json
from pathlib import Path

from config import BASE_DIR

CHECKLIST_FILE = BASE_DIR / "training" / "build_checklist.json"

# Status: done | pending
BUILD_CHECKLIST = {
    "patterns": {
        "breakout": "done",
        "ma_pullback": "done",
        "bull_flag": "done",
        "double_bottom": "done",
        "mean_reversion": "done",
        "gap_fill": "done",
        "vwap_reclaim": "pending",
        "rsi_divergence": "pending",
        "bear_breakdown": "done",
        "credit_put": "done",
    },
    "agents": {
        "context": "done",
        "regime": "done",
        "probability": "done",
        "quant": "done",
        "qa": "done",
        "options_research": "done",
        "options_backtest": "done",
        "earnings_filter": "done",
        "sector_rotation": "done",
        "portfolio_correlation": "done",
        "intraday_timing": "done",
        "margin_compliance": "done",
    },
    "training": {
        "checkpoint_2k_protocol": "done",
        "final_10k_run": "pending",
    },
}

CHECKPOINT_SIMULATIONS = 2_000
FINAL_SIMULATIONS = 10_000


def load_checklist() -> dict:
    if CHECKLIST_FILE.exists():
        return json.loads(CHECKLIST_FILE.read_text(encoding="utf-8"))
    return BUILD_CHECKLIST


def save_checklist(data: dict) -> None:
    CHECKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKLIST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_done(category: str, item: str) -> None:
    data = load_checklist()
    if category in data and item in data[category]:
        data[category][item] = "done"
        save_checklist(data)


def pending_items(include_final: bool = False) -> list[str]:
    data = load_checklist()
    out = []
    for cat, items in data.items():
        for name, status in items.items():
            if status == "pending":
                if name == "final_10k_run" and not include_final:
                    continue
                out.append(f"{cat}/{name}")
    return out


def is_ready_for_final_train() -> bool:
    """All build modules done — ready for the one-time final 10k run."""
    return len(pending_items(include_final=False)) == 0


def print_status() -> str:
    data = load_checklist()
    lines = ["BUILD CHECKLIST", "=" * 40]
    done_count = pending_count = 0
    for cat, items in data.items():
        lines.append(f"\n{cat.upper()}:")
        for name, status in items.items():
            icon = "[x]" if status == "done" else "[ ]"
            lines.append(f"  {icon} {name}")
            if status == "done":
                done_count += 1
            else:
                pending_count += 1
    lines.append(f"\n{done_count} done, {pending_count} pending")
    build_pending = pending_items(include_final=False)
    if is_ready_for_final_train():
        lines.append("\n>>> ALL MODULES BUILT — ready for: python cli.py train -n 10000 <<<")
    elif build_pending:
        lines.append(f"\n>>> Still building: {', '.join(build_pending)} <<<")
    else:
        lines.append("\n>>> Run 2k checkpoints as you add modules <<<")
    return "\n".join(lines)