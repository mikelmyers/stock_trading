"""QA agent: data quality, overfitting checks, trade validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from training.calibrator import load_learned_params


def analyze_qa(ticker: str, df: pd.DataFrame, sheet: dict) -> dict:
    """Run quality checks on data and analysis output."""
    issues = []
    warnings = []
    passed = []

    if df.empty:
        issues.append("No price data returned")
    elif len(df) < 50:
        warnings.append(f"Limited history: {len(df)} bars")
    else:
        passed.append(f"History OK: {len(df)} bars")

    if not df.empty:
        last_date = df.index[-1]
        if hasattr(last_date, "tzinfo"):
            age_days = (datetime.now(timezone.utc) - last_date.to_pydatetime().replace(tzinfo=timezone.utc)).days
        else:
            age_days = (datetime.now() - pd.Timestamp(last_date).to_pydatetime()).days
        if age_days > 3:
            warnings.append(f"Data may be stale ({age_days} days old)")
        else:
            passed.append("Data freshness OK")

    learned = load_learned_params()
    trained = learned.get("trained_on_simulations", 0)
    real = learned.get("trained_real_trades")
    # (the old hardcoded 0.46-estimate check was mathematically unreachable)
    if trained > 0 and real is not None and real < trained * 0.3:
        warnings.append(
            f"Training is {trained:,} sims but only {real:,} are real trades — "
            "bootstrap-heavy, watch overfitting"
        )

    setup = sheet.get("setup", {})
    if setup.get("is_valid_setup") and sheet.get("context", {}).get("context_score", 0) < 50:
        warnings.append("Valid setup but weak market context")

    if sheet.get("regime", {}).get("regime") == "RISK_OFF":
        warnings.append("RISK_OFF regime — new trades discouraged")

    score = max(0, 100 - len(issues) * 40 - len(warnings) * 15)

    return {
        "qa_score": score,
        "status": "FAIL" if issues else ("WARN" if warnings else "PASS"),
        "issues": issues,
        "warnings": warnings,
        "passed": passed,
        "summary": (
            f"QA {score}/100 — {len(issues)} issues, {len(warnings)} warnings"
        ),
    }