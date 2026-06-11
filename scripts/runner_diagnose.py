"""Print why each Alpaca gainer passes or fails runner gates."""
from __future__ import annotations

from runner.conditions import (
    AP_GAP_MIN,
    AP_RVOL_MIN,
    CAND_GAP_MIN,
    CAND_PRICE_MAX,
    CAND_PRICE_MIN,
    CAND_RVOL_MIN,
)
from runner.datasource import AlpacaSource


def fail_reasons(c) -> str:
    reasons = []
    if c.price is None:
        reasons.append("no_price")
    elif c.price < CAND_PRICE_MIN:
        reasons.append(f"price<{CAND_PRICE_MIN}")
    elif c.price > CAND_PRICE_MAX:
        reasons.append(f"price>{CAND_PRICE_MAX}")
    if (c.rvol or 0) < CAND_RVOL_MIN:
        reasons.append(f"rvol<{CAND_RVOL_MIN}")
    if (c.gap_pct or 0) < CAND_GAP_MIN:
        reasons.append(f"gap<{CAND_GAP_MIN}")
    if not reasons:
        if c.is_candidate and not c.green_light:
            return c.blowup_flags or "candidate-not-green"
        return "ok"
    return ",".join(reasons)


def main() -> int:
    cvs = AlpacaSource().scan()
    print(f"Scanned {len(cvs)} Alpaca gainers")
    print(
        f"Candidate gate: price ${CAND_PRICE_MIN}-${CAND_PRICE_MAX}, "
        f"rvol>={CAND_RVOL_MIN}, gap>={CAND_GAP_MIN}%"
    )
    print(
        f"Green-light adds: rvol>={AP_RVOL_MIN}, gap>={AP_GAP_MIN}%, "
        "news, above VWAP, float<=20M, no blowup flags"
    )
    print()
    hdr = f"{'sym':<8}{'px':>7}{'gap%':>7}{'rvol':>6}{'dVWAP':>7}{'sprd':>6}{'news':>5}  status"
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(cvs, key=lambda x: -(x.gap_pct or 0)):
        status = "GREEN" if c.green_light else ("CAND" if c.is_candidate else "pass")
        print(
            f"{c.symbol:<8}{(c.price or 0):>7.2f}{(c.gap_pct or 0):>7.1f}"
            f"{(c.rvol or 0):>6.1f}{(c.dist_vwap_pct or 0):>7.1f}"
            f"{(c.spread_pct or 0):>6.1f}{str(c.has_news):>5}  {status}: {fail_reasons(c)}"
        )
    n_cand = sum(1 for c in cvs if c.is_candidate)
    n_green = sum(1 for c in cvs if c.green_light)
    print()
    print(f"candidates={n_cand}  green_light={n_green}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())