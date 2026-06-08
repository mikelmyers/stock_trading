# Survivorship-bias stress test (Rung 1 — free)

Quantifies how exposed the calibrated edge is to survivorship bias, using free
point-in-time S&P 500 membership (`fja05680/sp500`, MIT) + the simulated trades
from this run. It does **not** correct the bias (that needs delisted price data —
Rung 2); it **bounds** it.

Run: `python -m training.survivorship.stress_test`

## Findings (2001–2026 window)

- Of **1,194** companies ever in the S&P 500, our price cache has only **493**.
  We are **missing 701** (689 of them names that *exited* the index).
- That's **35.9% of S&P 500 member-time** with no data — almost entirely the
  removed/delisted/acquired names. Our backtest effectively only traded survivors.
- Edge on the genuine S&P-member subset of our setups: **+0.028 R**.
- Implied "ghost" trades on the missing names (at the covered setup rate):
  **~406k**, vs ~724k covered.
- **Break-even: those missing-name trades need only average worse than
  −0.049 R to erase the +0.028 R edge.** Even 10% of them being catastrophic
  (−1R, rest neutral) flips the blended edge negative.

## Verdict

The measured edge sits **within the survivorship-bias margin**. The missing
names skew toward failures/distress (that's largely *why* they were removed), so
their true average is very likely negative — and the bar to wipe out the edge is
tiny. The +0.06R calibration result therefore **cannot be trusted as tradeable
on its own**; it may be a real small edge or an artifact.

Caveats both ways: many S&P exits are M&A (acquired at a premium → those trades
may be roughly neutral, which *softens* the conclusion), but genuine failures can
lose far more than −1R via gap-downs through the stop (which *worsens* it). Given
the −0.049R break-even, neither nuance rescues confidence.

**Implication:** to validate (or kill) this edge for real, Rung 2 is required —
actual delisted daily prices (Tiingo ~$10/mo or EODHD ~$20/mo) backfilled into
the cache so the failures are simulated, not assumed. The ML model inherits the
same caveat: it was trained to rank *survivors*.
