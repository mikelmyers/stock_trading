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

## UPDATE — deployment toolchain + GO-LIVE PROTOCOL (closing section)

Two tools now operationalize everything above:
- `training/validated_sim.py` — validated system, clean OOS test, blind
  walk-forward money test, and sizing/risk-of-ruin study.
- `training/live_signals.py` — forward paper-trade harness: trains the deployment
  model on all history, scans currently-listed names (delisted/stale skipped),
  scores with the validated pipeline, prints today's regime-aware book, and
  appends dated signals to `paper_trades.csv` (an ungameable forward record).
  `--lookback N` backfills recent signals to seed the log.

**Sizing — the rule that prevents another blow-up:** single-bet Kelly computes to
~20%, but that is a TRAP — Kelly assumes independent sequential bets; you hold K=5
CORRELATED positions that lose together, which destroys the math. Anchor on the
*realized* drawdown of the blind book instead:

| risk/trade | CAGR | drawdown | verdict |
|---|--:|--:|---|
| **1%** | ~5% | ~15% | START HERE (0% ruin) |
| 2% | ~10% | ~30% | aggressive but survivable |
| 3%+ | 14%+ | 41%+ | courts ruin — don't |

Size so a simultaneous K-position overnight gap is a loss you can shrug off.

**GO-LIVE PROTOCOL (the disciplined path — this is what the $18k lesson buys):**
1. Run `live_signals.py` daily on FRESH data for 2-3 months → real forward record.
2. Score it vs the blind expectation (~+0.07R/trade, Sharpe ~1.8). Proceed only
   if it holds; a forward miss means stop, not size up.
3. Go live at **1% risk/trade** (not Kelly). Scale only after live confirmation.
4. Open items before scaling size: honest options P&L for the credit_put trades
   (the linear-R label flatters capped option payoffs), and a retrain cadence.

**Known caveats carried forward (eyes open):** edge is regime-dependent (loses in
grinding bears like 2022 — survive via sizing, don't try to beat it); shorting /
down-year profit is a proven loser here (3 tests); current live signals skew to
index-ETF credit_put (concentrated single bet + the weakest-modeled instrument).

**Bottom line:** a real, modest, honestly-sized, forward-testable long/risk-on
strategy — Sharpe ~1.8 through-cycle, ~5% CAGR at 1%/trade — with the tooling to
keep it honest. Not a money-printer; a survivable edge that won't blow up.

## UPDATE — cross-sectional ranking (breadth thesis): FAILS, and tells us why

Tested the highest-leverage "make it bigger" idea (`training/cross_sectional.py`):
rank the whole universe monthly, hold top-N, instead of waiting for ~9 discrete
setups. Grinold's law (IR ~= skill x sqrt(breadth)) says breadth is the lever.

| book | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| long top-50 | 35.7% | 1.04 | -45% |
| long top-50 + regime | 27.3% | 1.03 | -37% |
| equal-weight market | 19.2% | 0.98 | -27% |

**Information Coefficient = -0.0043 (≈ zero) → NO cross-sectional ranking skill.**
The flashy 35% CAGR is a mirage: (1) both top-50 AND bottom-50 beat the market →
the model just tilts to high-volatility names, not a directional rank; (2) the
"market" benchmark itself shows 19% CAGR (survivorship + equal-weight premium of
this universe); (3) Sharpe 1.04 is BELOW the event-driven 1.8 with 3x the drawdown.

**Why it fails (the valuable insight):** the model is 84% market-regime/VIX
features, which are IDENTICAL across all stocks on a given day → cross-sectionally
flat → zero ranking power. **Our edge is market TIMING ("when"), not stock
SELECTION ("which").** This validates the event-driven + regime-timing design as
the correct expression of a timing edge, and means cross-sectional ranking would
require an orthogonal CROSS-SECTIONAL alpha source -- fundamental/factor data
(value, quality, earnings) we don't have. Breadth lever is gated on DATA, not
modeling. Recommendation stands: ship the validated event-driven system.

## UPDATE — real fundamentals (SEC-XBRL) cross-sectional: weak signal, concept confirmed

Pulled free point-in-time fundamentals (SEC companyfacts XBRL via a public GCS
bucket; 989k rows, 12.3k companies, monthly snapshots 2013+, ticker-keyed) and
built classic value/quality/growth factors joined to our prices
(`training/cross_sectional_fund.py`).

| book | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|
| long top-50 (fundamentals) | 24.8% | 0.98 | -43% |
| + regime overlay | 17.9% | 0.92 | -42% |
| equal-weight market | 16.5% | 0.91 | -27% |

**IC flipped −0.004 (price-only, no skill) → +0.0093 (fundamentals)** — concept
confirmed: fundamentals carry cross-sectional signal price-only data lacks, and the
model leans on the right factors (sales/price, book/price, cash-flow/price,
earnings/price, size, asset-growth). BUT the edge is **weak** (IC < 0.02 bar),
standalone Sharpe ~1.0 barely beats market and is far below the event-driven 1.8,
with a 43% drawdown. We're rediscovering the most crowded factors (value/size) with
no neutralization; the stronger factors (estimate revisions, earnings surprise) need
analyst data this set lacks.

**Verdict:** not worth a standalone Model B. Higher-value use of the same data:
fold these fundamental factors in as EXTRA FEATURES on the proven event-driven
timing model (does "cheap + quality" sharpen the +0.077R "when" edge?) — reuses the
validated framework, point-in-time join on ticker+date. That's the next cheap test.

## UPDATE — fundamentals as FEATURES on the timing model: marginal (+0.0034R), below the bar

Point-in-time joined the SEC-XBRL fundamentals + price onto the event-driven setups
(682k setups 2013+, 78% fundamental coverage) and ran baseline vs +fundamentals on
the same 2019+ holdout (`training/augment_fundamentals.py`).

| model | top-10% net R | win% | OOS-AUC |
|---|--:|--:|--:|
| baseline (33 feats) | +0.0666 | 58.5% | 0.5174 |
| + 12 fundamental factors | +0.0700 | 59.0% | 0.5185 |

**Delta +0.0034R — NEUTRAL, below the noise floor for a single holdout** (AUC barely
moved). Factors used are sensible (rev_growth, asset_growth, sales/price, size,
cash-flow/price). Direction positive, consistent with the weak cross-sectional
signal, but too small to justify a production fundamentals-join dependency + 78%
coverage gap. **Verdict: don't add to the live system on one marginal holdout.** A
walk-forward (per-year delta) is the tiebreaker between "tiny real edge" and "noise";
absent that, skip it. The validated timing system stands as the deliverable.

## UPDATE — walk-forward tiebreaker: fundamentals features are NOISE (definitive)

Per-year delta (baseline vs +fundamentals, top-10% net R, retrained each year):
2019 +0.011 · 2020 +0.031 · 2021 -0.021 · 2022 -0.032 · 2023 +0.008 · 2024 -0.015
· 2025 +0.006 · 2026 -0.012.  **Mean -0.0029R, positive in 4/8 years → NOISE, drop.**

The +0.0034R single-holdout bump was a favorable-draw illusion; over a proper
walk-forward fundamentals net slightly NEGATIVE (and hurt in 2022/2021/2024). The
walk-forward did its job: avoided adding a fragile production fundamentals
dependency for a phantom edge.

**"Make it bigger" thread CLOSED with data — all accessible levers ruled out:**
(1) cross-sectional price-only = no skill; (2) cross-sectional + fundamentals =
weak (IC +0.009, ~market); (3) fundamentals as features = noise. The free SEC
fundamental axis (classic value/quality/growth) adds no tradeable edge; the
stronger factors (estimate revisions, earnings surprise) need paid, crowded data.
**Our edge is the regime TIMING; the validated event-driven system is the
deliverable.** Forward paper-trading remains the only un-run gate.

## UPDATE — measurement audit: HEADLINE NUMBERS ABOVE ARE STALE, re-run required

A code audit found two measurement bugs that inflate every realized-book number
quoted above (now fixed in `training/simutil.py` + the four book sims, with
regression tests under `tests/`):

1. **Sharpe was annualized with sqrt(252) over per-trade-event returns**
   (~80-100 events/yr, not 252), inflating it ~sqrt(252/trades_per_yr) ≈
   1.6-1.9x. The "Sharpe 1.87 blind / 2.45 clean / benchmark ~1.8" figures are
   wrong; true through-cycle Sharpe is plausibly **~1.0-1.2**. Now computed on
   a business-daily-resampled equity curve.
2. **Close dates added `days_held` (TRADING days) as CALENDAR days**, freeing
   K-slot capacity ~30% early — trades/yr and compounded CAGR are overstated
   in every capacity-limited book above.

Also fixed with label impact (smaller): the backtester's trailing stop was
raised with the current bar's close and triggered on the same bar's low
(intrabar lookahead, now prior-bar level); purge gaps treated the 14-trading-day
horizon as 14 calendar days; short entries received favorable slippage; the
legacy calibrator counted bootstrap resamples and slippage variants as evidence.

**RE-STATED RESULTS (2026-06-10):** Re-ran `validated_sim` on the existing
`survivorship_free_v2.parquet` dataset (1.69M rows, 52.5% base win, market
features present) using the fixed measurement code from PR #6. All 47 regression
tests pass. Full console output saved under `training/output/`.

These re-stated figures are the **new go-live benchmark** for the forward
paper-trading gate. Compare against the stale headline numbers above (flat
sizing = honest baseline per the blind walk-forward section).

**Caveat — labels not re-walked:** the underlying trade labels in
`survivorship_free_v2.parquet` still predate the trailing-stop intrabar
lookahead fix (PR #6). A future full survivorship re-walk would relabel every
setup and may shift mean R modestly; that compute was not available for this
re-statement. The Sharpe / trades-yr / CAGR corrections below are from the
measurement fixes only (daily Sharpe, `np.busday_offset` close dates, training-
distribution P-floor).

### Walk-forward blind money test (2019–2026, flat sizing)

| metric | stale | re-stated |
|---|---:|---:|
| mean R / trade | +0.069 | **+0.076** |
| CAGR (1%/trade) | +5.4% | **+4.9%** |
| max drawdown | 15% | **19%** |
| Sharpe | 1.87 | **0.84** |
| trades / yr | ~80–100 | **67** |
| $10k → | $14,711 | **$14,195** |

per-year mean R (stale → re-stated):
2019 +0.145→**+0.168** · 2020 +0.049→**+0.196** · 2021 +0.070→**+0.138** ·
2022 −0.127→**−0.232** · 2023 +0.109→**+0.119** · 2024 +0.178→**+0.177** ·
2025 +0.005→**+0.012** · 2026 −0.010→**−0.056**

Sharpe and trades/yr dropped as expected; mean R barely moved. Regime-scaled
sizing: Sharpe 1.71→**0.87**, CAGR +4.3%→**+4.2%**, maxDD 15%→**16%**.

### DEV period (2019–2023, flat sizing)

| metric | stale | re-stated |
|---|---:|---:|
| mean R / trade | +0.051 | **+0.140** |
| CAGR (1%/trade) | +4.6% | **+9.3%** |
| max drawdown | 19% | **12%** |
| Sharpe | 1.33 | **1.43** |
| trades / yr | (not logged) | **66** |

per-year mean R (re-stated):
2019 **+0.168** · 2020 **+0.269** · 2021 **+0.175** · 2022 **−0.182** ·
2023 **+0.222**

Dev mean R rose because the P-floor now comes from the training distribution
(not the eval window's own quantile), concentrating higher-conviction trades.
Sharpe stayed in the ~1.0–1.5 band. Regime-scaled: Sharpe 1.48→**1.41**,
CAGR +4.5%→**+7.6%**, maxDD 15%→**8%**.

### CLEAN one-shot test (2024+, flat sizing)

| metric | stale | re-stated |
|---|---:|---:|
| mean R / trade | +0.079 | **+0.102** |
| CAGR (1%/trade) | +8.1% | **+6.3%** |
| max drawdown | 15% | **9%** |
| Sharpe | 2.18 | **1.12** |
| trades / yr | (not logged) | **63** |

per-year mean R (stale → re-stated):
2024 +0.206→**+0.160** · 2025 ~0→**+0.035** · 2026 ~0→**+0.187**

The headline 2.45 Sharpe / +8% CAGR were inflated ~2×; re-stated Sharpe **1.12**
is the honest clean-test number. Regime-scaled: Sharpe 2.45→**1.14**, CAGR
+7.7%→**+5.4%**, maxDD 11%→**8%**.

### Updated go-live benchmark (forward paper-trading gate)

Score live signals against the **walk-forward blind book** (the hardest test):

> ~**+0.076R**/trade · Sharpe ~**0.8–0.9** · ~**67 trades/yr** · ~**5% CAGR**
> at 1%/trade · ~**19%** max drawdown · positive ~75% of years.

Proceed to live capital only if forward paper-trading holds these re-stated
expectations — not the stale ~1.8 Sharpe / ~80–100 trades-yr figures quoted
earlier in this file.
