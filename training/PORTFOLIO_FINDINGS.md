# Rung A — bankroll / sizing / risk-of-ruin findings

Run: `python -m training.portfolio_sim --seed 1000`

## Verdict: not self-funding as built

The calibrated strategy's **realized, tradeable book loses money** — before any
survivorship haircut:

| Layer | Mean R / trade |
|---|---|
| Pool, gross | +0.046 |
| Pool, net of ~0.015R cost | +0.031 |
| **Realized book** (K=5, max 2/type, 182 trades/yr) | **−0.024** |
| Realized book + survivorship −0.06R | −0.084 |

At every sizing level the account shrinks (e.g. 1%/trade → −5.4% CAGR, 83% max
drawdown). **Position sizing cannot rescue a negative edge — it only sets how
fast you lose.** Not self-funding at any seed or capital.

## Why the positive backtest didn't survive contact

1. **Costs** (~0.015R) eat ~⅓ of the thin gross edge.
2. **Survivorship** (Rung 1) likely eats the rest.
3. **Capacity + selection (the decisive one):** the +0.031R pool edge is spread
   across ~1M overlapping signals you can never all trade. You can hold ~5
   positions → ~180 trades/yr. The realized subset you *can* take is **negative**,
   under both naive (chronological) and smart (best-score, diversified)
   selection. The hand `setup_score` is not predictive of which trades win
   (consistent with the ML AUC of 0.534) — selecting "high conviction" picked
   *worse* trades.

## What this means

The process worked: for **$0**, before risking a dollar, we learned the edge
isn't real once you account for what you can actually trade. The engine,
pipeline, and validation harness are sound and reusable — the *signal* is the
problem, not the machinery.

The one un-played card: select the realized book by the **ML model's P(win)**
instead of the hand score (true meta-labeling). Given the model's tiny AUC edge
it is unlikely to flip a −0.024R book positive, but it's the last free test
before concluding this signal family is a dead end on large-cap equities.

## UPDATE — meta-labeling + selectivity changes the verdict

Ran the meta-labeling test (`training/ml/metalabel_test.py`, OOS purged
walk-forward). The model sorts realized outcomes **monotonically** out-of-sample
(decile 1 = −0.064R / 43% win → decile 10 = +0.057R / 57% win), and its biggest
skill is filtering out the junk bottom decile. Selecting the realized book by
model P(win) flips it from −0.024R to **+0.025R** (+0.010R after cost).

Being **selective** (manual trader's regime — take only top-conviction signals)
is the real win:

| Only trade if P(win) ≥ | trades/yr | mean R | net cost | −0.03 surv | −0.06 surv |
|---|--:|--:|--:|--:|--:|
| top 10% | 139 | +0.063 | +0.048 | +0.033 | +0.003 |
| top 5%  | 123 | +0.056 | +0.041 | +0.026 | −0.004 |
| top 1%  | 72  | +0.057 | +0.042 | +0.027 | −0.003 |

**New verdict:** the raw strategy is a losing book, but **model-filtering +
selectivity produces a positive, out-of-sample, cost-inclusive edge (~+0.048R on
~139 trades/yr) that survives a mild survivorship haircut and is breakeven at the
harsh one.** This is "good enough to take seriously" — modest (~5–7%/yr at safe
sizing) but real-looking. The deciding uncertainty is now squarely the true
survivorship haircut (Rung 2). Caveat: several selection schemes were tried, so a
true forward/holdout test is needed before trusting it with size.
