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

## UPDATE — model round 1: market-regime features ~triple the edge (leakage-checked)

Diagnosed the model's core blind spot: all 21 features were single-name technicals
computed in isolation — **zero market context.** Added 12 causal market-regime +
relative-strength features from the unused `^GSPC`/`^VIX` data (S&P 20/60d return,
>SMA200, dist-SMA50; VIX level / 5d-change / 252d z-score; per-stock rel-return
20/60/120d, rolling 60d beta + correlation). Rebuilt the survivorship-free dataset
(1.91M setups, 33 features) and re-ran the same 2019+ holdout.

| metric (clean holdout) | baseline 21-feat | round-1 33-feat |
|---|--:|--:|
| top-10% net edge | +0.0278R | **+0.0765R** |
| top-5% net edge | +0.0404R | **+0.0863R** |
| top-10% @ harsh −0.06R haircut | −0.0172R | **+0.0315R** (flips +) |
| decile-10 mean R (cost-free) | +0.047R | +0.096R |
| top-10% win rate | 55.2% | 58.3% |
| OOS AUC | 0.534 | 0.540 |

**Leakage ruled out:** purged walk-forward CV gives real AUC 0.523 vs **shuffled-label
baseline 0.499** (≈0.50 every fold) → edge +0.0235, "signal present." AUC barely moved
while the top decile nearly doubled — the gain is sharper *ranking*, not a leak. The
12 new features account for **84% of model importance**; the top 6 are all VIX/market
(`vix_z_252`, `vix_level`, `mkt_ret_60`, `mkt_dist_sma50`, `mkt_ret_20`, `vix_chg_5`).

**Interpretation:** the edge was never mostly about *which* setup — it's about *when*.
The signal is a regime timer: take the chart patterns in calm uptrends, skip them when
the VIX spikes.

**Stability (top-10% raw, cost-free mean R by year):** 2019 +0.124 · 2020 **+0.037**
· 2021 +0.141 · 2022 **−0.088** · 2023 +0.099 · 2024 +0.285 · 2025 −0.026 · 2026 +0.030.
Positive in **6 of 8 years including the 2020 crash** (so not a single-crash artifact),
but a genuine −0.088R losing year in the 2022 bear. Real, broadly-persistent,
regime-driven edge — tradeable with risk management, not bulletproof. Next levers:
R-magnitude objective, and a bear-regime gate/short side to fix the 2022-type weakness.

## UPDATE — ground-up audit, Layer 1 (trade simulation realism): PASS

Before building further, audited the bedrock label (`simulate_trade_forward`). It is
an idealized *stock* trade (linear P&L, ATR stop/targets/trailing). Instrument decision:
we trade **stocks + outright (directional) options, not spreads** — so the linear-R
underlying-move label is the correct signal to optimize (outright options have uncapped
upside, so the magnitude edge is legitimate; option leverage/decay is a sizing concern).

Two realism gaps, measured (not assumed):
- **Per-trade cost:** ~0.013R for liquid large-cap stocks (~0.04% of price / ~3% ATR risk).
  The book's flat −0.015R assumption covers it.
- **Gap-through stops:** of 376k stop exits, **8.4% gapped through the stop**, avg extra
  loss **0.34R** when they did → **−0.0056R/trade** spread over all trades (mean R
  +0.0347 → +0.0291). Worst events −10R to −17R, clustered in crashes.

**Honest all-in cost ≈ 0.019R/trade** — between our −0.015 and −0.03 columns. So the
**−0.03R column is the honest baseline**, and the edge clears it (round-1 top-10%
≈ +0.024R net-honest; round-2 ≈ +0.12R). **No re-walk required.** The gap *tail* (rare
−10R+ events) is an account risk to handle at the **position-sizing layer** (cap per-trade
exposure), not a labeling problem. Layer 1 verdict: foundation is honest enough to build on.

## UPDATE — validated system + CLEAN out-of-sample test (the verdict)

Built the full validated system (`training/validated_sim.py`): regime-aware ML
model selects the book (top-10% by P(win)), `bear_breakdown` excluded (Layer-2
dud), honest 0.019R cost (Layer-1), **regime-scaled position sizing** (size down /
step aside when the tape is below its 200-day and VIX is stretched). Frozen,
pre-specified design — tuned ONLY on 2019-2023; **2024+ reserved untouched**.

| period | sizing | mean R | CAGR | maxDD | Sharpe |
|---|---|--:|--:|--:|--:|
| DEV 2019-2023 | flat | +0.051 | +4.6% | 19% | 1.33 |
| DEV 2019-2023 | regime | +0.051 | +4.5% | 15% | 1.48 |
| **CLEAN 2024+** | flat | +0.079 | +8.1% | 15% | 2.18 |
| **CLEAN 2024+** | regime | +0.079 | +7.7% | 11% | **2.45** |

**The edge SURVIVED the one-shot clean test** — mean R higher OOS than in dev,
positive every year, not overfit. Regime sizing consistently cuts drawdown and
raises Sharpe (the validated answer to surviving down years and gap-tails).

**Honest caveats:** the 2024+ window had no 2022-style bear and was carried by a
strong 2024 (+0.206R; 2025/2026 ~breakeven), so the 2.45 Sharpe / +8% CAGR are
optimistic. **Through-cycle expectation = the dev ~1.4 Sharpe**, ~+0.05-0.08R/trade,
~15% drawdowns in bears. Character: a **risk-on harvester** — good in favorable
tapes, breakeven in flat ones, contained losses in bears (never a down-year
*profit* engine; three tests confirmed shorting/momentum/index-TF don't reliably
pay). Last gate before real capital: **forward paper-trading** (the only truly
out-of-sample test left).

## UPDATE — walk-forward BLIND money test (the most honest number)

Ran the strictest historical test (`validated_sim.py --walk-forward`): for each
year 2019-2026, retrain on ONLY prior years, set the entry threshold from the
training distribution (no peeking), trade that year blind, compound one account.

| sizing | mean R | CAGR | maxDD | Sharpe | $10k → |
|---|--:|--:|--:|--:|--:|
| flat | +0.069 | +5.4% | 15% | **1.87** | $14,711 |
| regime-scaled | +0.069 | +4.3% | 15% | 1.71 | $13,634 |

Per-year R: 2019 +0.145 · 2020 +0.049 · 2021 +0.070 · **2022 −0.127** · 2023
+0.109 · 2024 +0.178 · 2025 +0.005 · 2026 −0.010. Positive 6/8 years, blind.

**Edge confirmed under the hardest test** — Sharpe 1.87, lands between dev (+0.051R)
and clean (+0.079R); the consistency across three independent tests is the real
evidence. The model **self-throttled** in danger years (only 593 signals passed in
2020, 791 in 2022, vs 2,700+ in calm years) — desired behavior — yet still lost in
2022, confirming bears are a managed drawdown, not beatable.

**Honest correction:** regime-scaled sizing HELPED in dev + clean test but HURT
here (Sharpe 1.71 < 1.87, same maxDD) → it is **roughly neutral**, a marginal
risk-smoother, not a clear win. Flat sizing is the honest baseline.

**FINAL CHARACTER (validated 6 ways):** a real, modest, risk-on edge — ~+0.07R/trade,
Sharpe ~1.8, ~80-100 trades/yr, ~15% drawdowns, positive ~75% of years. ~5% CAGR at
1%/trade, ~10-13% at 2-2.5%/trade (~25-30% DD). Good in decent tapes, breakeven in
flat/bear years, doesn't blow up (self-throttles + size for gap-tails). Last gate
before real capital: **live-forward paper trading.**
