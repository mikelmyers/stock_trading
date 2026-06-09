"""Cross-sectional ranking prototype -- testing the breadth thesis.

Instead of waiting for ~9 discrete setups (~100 trades/yr), score the model's
expected forward return for EVERY stock on a monthly rebalance, hold the top-N,
refresh. Grinold: IR ~= skill * sqrt(breadth) -- breadth is the lever. This tests
whether the model has cross-sectional ranking skill (IC) and whether a top-N book
beats the event-driven Sharpe ~1.8, using the data we already have.

    python -m training.cross_sectional
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from training.history import download_history
from training.ml.features import FEATURE_COLUMNS, compute_feature_frame, compute_market_frame
from training.universe import load_training_universe

REBAL = 21          # trading days between rebalances (~monthly), = label horizon
TOPN = 50           # long-only book size
COST = 0.0015       # per-rebalance turnover cost (~15 bps)


def build_panel():
    hist = download_history(load_training_universe())
    market = compute_market_frame(hist.get("^GSPC"), hist.get("^VIX"))
    rebal_dates = market.index[260::REBAL]            # global monthly calendar
    reg = market.reindex(rebal_dates)[["mkt_above_sma200", "vix_z_252"]]

    rows = []
    for tk, df in hist.items():
        if tk in ("^GSPC", "^VIX") or len(df) < 300:
            continue
        feats = compute_feature_frame(df, market=market)
        close = df["Close"]
        fwd = close.shift(-REBAL) / close - 1.0       # forward 21d return
        sub = feats.reindex(rebal_dates)
        f = fwd.reindex(rebal_dates)
        for d in rebal_dates:
            v = f.get(d)
            if v is None or not np.isfinite(v):
                continue
            r = sub.loc[d, FEATURE_COLUMNS]
            if r.isna().all():
                continue
            rows.append((d, tk, v, *r.values))
    panel = pd.DataFrame(rows, columns=["date", "ticker", "fwd", *FEATURE_COLUMNS])
    panel["year"] = panel["date"].dt.year
    return panel, reg


def run():
    import lightgbm as lgb
    panel, reg = build_panel()
    print("=" * 66)
    print("  CROSS-SECTIONAL RANKING PROTOTYPE (breadth thesis)")
    print("=" * 66)
    print(f"  panel: {len(panel):,} stock-months, {panel['ticker'].nunique()} names, "
          f"{panel['year'].min()}-{panel['year'].max()}, rebal={REBAL}d, top-{TOPN}")

    years = sorted(y for y in panel["year"].unique() if y >= 2019)
    ics, top_rets, bot_rets, reg_rets, mkt_rets, dates = [], [], [], [], [], []
    for Y in years:
        tr = panel[panel["year"] < Y]
        te = panel[panel["year"] == Y]
        if len(tr) < 50_000 or te.empty:
            continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                              n_jobs=-1, verbosity=-1)
        m.fit(tr[FEATURE_COLUMNS].to_numpy("float64"), tr["fwd"].to_numpy("float64"))
        te = te.copy()
        te["pred"] = m.predict(te[FEATURE_COLUMNS].to_numpy("float64"))
        for d, g in te.groupby("date"):
            if len(g) < TOPN * 2:
                continue
            g = g.sort_values("pred", ascending=False)
            ics.append(g["pred"].corr(g["fwd"], method="spearman"))
            top = g.head(TOPN)["fwd"].mean() - COST
            bot = g.tail(TOPN)["fwd"].mean()
            top_rets.append(top); bot_rets.append(bot); mkt_rets.append(g["fwd"].mean())
            # regime overlay: size down the book in risk-off rebalances
            rr = reg.loc[d] if d in reg.index else None
            scale = 1.0
            if rr is not None:
                if rr["mkt_above_sma200"] < 0.5:
                    scale *= 0.4
                if rr["vix_z_252"] > 1.0:
                    scale *= 0.6
            reg_rets.append(top * scale)
            dates.append(d)

    def stats(rets):
        r = pd.Series(rets, index=pd.DatetimeIndex(dates)).dropna()
        per_yr = 252 / REBAL
        cum = (1 + r).prod()
        cagr = cum ** (per_yr / len(r)) - 1
        sharpe = r.mean() / r.std() * np.sqrt(per_yr) if r.std() else 0
        eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
        return cagr, sharpe, dd

    ic = np.nanmean(ics)
    print(f"\n  Information Coefficient (rank corr pred vs actual): {ic:+.4f}  "
          f"({'skill present' if ic > 0.02 else 'WEAK/none'})")
    print(f"  Decile spread per month: top-{TOPN} {np.mean(top_rets)*100:+.2f}%  vs  "
          f"bottom-{TOPN} {np.mean(bot_rets)*100:+.2f}%  (market {np.mean(mkt_rets)*100:+.2f}%)")
    print(f"\n  {'book':<22}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
    print("  " + "-" * 44)
    for name, rets in [(f"long top-{TOPN}", top_rets),
                       (f"long top-{TOPN} + regime", reg_rets),
                       ("equal-weight market", mkt_rets)]:
        cg, sh, dd = stats(rets)
        print(f"  {name:<22}{cg*100:>7.1f}%{sh:>8.2f}{dd*100:>7.0f}%")
    print(f"\n  Benchmark to beat: event-driven system Sharpe ~1.8 (but only ~100 trades/yr).")
    print(f"  Breadth here: ~{len(top_rets)} rebalances x {TOPN} names = "
          f"{len(top_rets)*TOPN:,} positions taken.")


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
