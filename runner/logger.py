"""Episode + condition logger — the learning loop's memory.

Records every scanned candidate's condition-vector + the decision (push/watch/pass),
tagged by episode (a stake), to runner_candidates.csv. Outcomes are filled in later
(the "did it run" potential label + realized R) keyed by symbol+date. This append-only
ledger IS the training set the classifier learns from — winners, near-misses, and the
small capped losses that teach the blow-up conditions.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from runner.conditions import ConditionVector

LOG_DIR = Path(__file__).resolve().parent / "data"
CANDIDATES = LOG_DIR / "runner_candidates.csv"
OUTCOMES = LOG_DIR / "runner_outcomes.csv"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class RunnerLog:
    def __init__(self, episode_id: str, candidates_path: Path = CANDIDATES):
        self.episode_id = episode_id
        self.path = Path(candidates_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_candidates(self, cvs: list[ConditionVector], decisions: dict[str, str] | None = None):
        """Append the whole scanned pool (not just the ones we trade)."""
        decisions = decisions or {}
        ts = _now()
        rows = []
        for cv in cvs:
            r = cv.to_row()
            r["episode_id"] = self.episode_id
            r["logged_at"] = ts
            r["decision"] = decisions.get(cv.symbol, "pass")
            rows.append(r)
        if rows:
            pd.DataFrame(rows).to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        return len(rows)

    @staticmethod
    def log_outcome(symbol: str, asof: str, max_favorable_pct: float,
                    realized_r: float | None = None, bucket: str | None = None,
                    exit_reason: str | None = None, path: Path = OUTCOMES):
        """Attach the result to a logged candidate. `max_favorable_pct` is the
        setup's POTENTIAL (did the stock run?) — the clean meta-label, separate
        from how we managed the trade. `bucket`: monster/good/scratch/loss."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        row = {"symbol": symbol, "asof": asof, "logged_at": _now(),
               "max_favorable_pct": max_favorable_pct, "realized_r": realized_r,
               "bucket": bucket, "exit_reason": exit_reason}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def load_training_frame(candidates_path: Path = CANDIDATES,
                        outcomes_path: Path = OUTCOMES) -> pd.DataFrame:
    """Join logged conditions to outcomes -> the classifier's training table."""
    cand = pd.read_csv(candidates_path)
    if Path(outcomes_path).exists():
        out = pd.read_csv(outcomes_path).drop(columns=["logged_at"], errors="ignore")
        cand = cand.merge(out, on=["symbol", "asof"], how="left")
    return cand
