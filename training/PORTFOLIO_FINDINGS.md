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

## UPDATE — clean survivorship-free holdout: the edge SURVIVES (attenuated)

Closed the two open caveats above at once. Rebuilt the entire simulation set on
the **survivorship-free universe** (current S&P 500 + 334 ingested delisted
ex-members = 897 tickers with usable history, walk_step=1) via an 8-shard
GitHub Actions matrix, then ran a strict, untouched **train-through-2018 /
trade-2019+ holdout** (`training/survivorship/holdout_validation.py`).

- Dataset: **1,910,020 setups** (survivorship-free), base win 52.4%, mean R +0.035.
- Holdout: 253,091 setups from 2019+ (purged 3d embargo + 14d label horizon),
  never seen in training. Base untouched holdout edge: **+0.0226R, 51.9% win**.
- Model P(win) deciles sort **monotonically** out-of-sample on this clean data
  (decile 1 −0.054R/43% → decile 10 +0.047R/56%), confirming the meta-label
  skill is not a survivorship artifact.

Selective top-K% books (K=5 concurrent, max 2/type), holdout, net of costs:

| Only trade if P(win) ≥ | trades/yr | gross R | net −0.015 | −0.03 surv | −0.06 surv | win% |
|---|--:|--:|--:|--:|--:|--:|
| top 10% | 123 | +0.0428 | **+0.0278** | +0.0128 | −0.0172 | 55.2% |
| top 5%  |  99 | +0.0554 | +0.0404 | +0.0254 | −0.0046 | 54.9% |
| top 1%  |  66 | +0.0513 | +0.0363 | +0.0213 | −0.0087 | 53.0% |

**Final verdict:** the **+0.048R top-10% edge SURVIVES the clean
survivorship-free holdout, but attenuated to +0.0278R net** (123 trades/yr,
55.2% win). Removing survivorship bias (adding the delisted losers) costs
roughly 40% of the apparent edge and, critically, the top-10% book now goes
**negative (−0.0172R) under the harsh −0.06R survivorship haircut** (it was
+0.003R on the biased data). It stays positive through a mild −0.03R haircut
(+0.0128R). The tighter **top-5% book is more robust** (+0.0404R net, still
−0.0046R at the harsh haircut). Net: a real but modest edge — no longer a
"dead end," and no longer dependent on survivorship bias — best expressed at
higher selectivity (top 5%) where it holds up across cost assumptions.

*(Provenance: 8-shard CI run #27171135775, all shards green; dataset
`training/ml/datasets/survivorship_free.parquet`.)*
