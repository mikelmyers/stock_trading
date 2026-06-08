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
