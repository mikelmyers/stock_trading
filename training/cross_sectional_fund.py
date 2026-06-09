"""Cross-sectional ranking with REAL point-in-time fundamentals (Model B test).

Uses the SEC-XBRL fundamentals (monthly point-in-time snapshots, 2013+, keyed by
ticker) joined to our price data to build classic cross-sectional alpha factors --
value (E/P, B/P, S/P, CF/P), quality (gross profitability, ROE, margin), leverage,
accruals, growth -- then asks the question price-only features couldn't: does the
model have cross-sectional ranking skill (IC) when fed cross-sectionally-VARYING
inputs?

    SNAP=/tmp/snapshots.csv.gz python -m training.cross_sectional_fund
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from training.history import download_history
from training.ml.features import compute_market_frame
from training.universe import load_training_universe

SNAP = os.environ.get("SNAP", "/tmp/snapshots.csv.gz")
HOLD_M = 1          # months held per rebalance
TOPN = 50
COST = 0.0015
FACTORS = ["ep", "bp", "sp", "cfp", "gp_assets", "roe", "margin", "lev",
           "accruals", "rev_growth", "asset_growth", "size"]


def build_panel():
    f = pd.read_csv(SNAP, parse_dates=["snapshot"], dtype={"cik": "Int64"})
    f = f.dropna(subset=["ticker"])
    univ = set(load_training_universe())
    f = f[f["ticker"].isin(univ)].sort_values(["ticker", "snapshot"])
    hist = download_history(load_training_universe())
    market = compute_market_frame(hist.get("^GSPC"), hist.get("^VIX"))
    reg = market[["mkt_above_sma200", "vix_z_252"]]

    rows = []
    for tk, g in f.groupby("ticker"):
        if tk not in hist or len(hist[tk]) < 300:
            continue
        px = hist[tk]["Close"].sort_index()
        g = g.copy()
        # YoY growth on the monthly snapshot series (12 snapshots ~ 1 year)
        g["rev_growth"] = g["Revenue_ttm"] / g["Revenue_ttm"].shift(12) - 1
        g["asset_growth"] = g["Assets"] / g["Assets"].shift(12) - 1
        for r in g.itertuples():
            d = r.snapshot
            ipos = px.index.searchsorted(d, side="right") - 1
            fpos = px.index.searchsorted(d + pd.DateOffset(months=HOLD_M), side="right") - 1
            if ipos < 0 or fpos <= ipos:
                continue
            entry, exit_ = px.iloc[ipos], px.iloc[fpos]
            mcap = entry * r.Shares                      # $ x billion-shares = $B
            if not np.isfinite(mcap) or mcap < 0.3:      # drop sub-$300M
                continue
            fwd = exit_ / entry - 1.0
            if not np.isfinite(fwd):
                continue
            rows.append({
                "snapshot": d, "ticker": tk, "fwd": fwd,
                "ep": r.Earnings_ttm / mcap, "bp": r.Equity / mcap,
                "sp": r.Revenue_ttm / mcap, "cfp": r.NetCashOperating_ttm / mcap,
                "gp_assets": r.GrossProfit_ttm / r.Assets if r.Assets else np.nan,
                "roe": r.Earnings_ttm / r.Equity if r.Equity else np.nan,
                "margin": r.GrossProfit_ttm / r.Revenue_ttm if r.Revenue_ttm else np.nan,
                "lev": r.Liabilities / r.Assets if r.Assets else np.nan,
                "accruals": (r.Earnings_ttm - r.NetCashOperating_ttm) / r.Assets if r.Assets else np.nan,
                "rev_growth": r.rev_growth, "asset_growth": r.asset_growth,
                "size": np.log(mcap),
            })
    panel = pd.DataFrame(rows)
    panel["year"] = panel["snapshot"].dt.year
    return panel, reg


def run():
    import lightgbm as lgb
    panel, reg = build_panel()
    print("=" * 66)
    print("  CROSS-SECTIONAL with REAL FUNDAMENTALS (Model B test)")
    print("=" * 66)
    print(f"  panel: {len(panel):,} stock-months, {panel['ticker'].nunique()} names, "
          f"{panel['year'].min()}-{panel['year'].max()}, top-{TOPN}")

    years = sorted(y for y in panel["year"].unique() if y >= 2018)
    ics, top_rets, bot_rets, reg_rets, mkt_rets, dates = [], [], [], [], [], []
    for Y in years:
        tr = panel[panel["year"] < Y]
        te = panel[panel["year"] == Y].copy()
        if len(tr) < 20_000 or te.empty:
            continue
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, min_child_samples=100,
                              n_jobs=-1, verbosity=-1)
        m.fit(tr[FACTORS].to_numpy("float64"), tr["fwd"].to_numpy("float64"))
        te["pred"] = m.predict(te[FACTORS].to_numpy("float64"))
        for d, gg in te.groupby("snapshot"):
            if len(gg) < TOPN * 2:
                continue
            gg = gg.sort_values("pred", ascending=False)
            ics.append(gg["pred"].corr(gg["fwd"], method="spearman"))
            top = gg.head(TOPN)["fwd"].mean() - COST
            top_rets.append(top); bot_rets.append(gg.tail(TOPN)["fwd"].mean())
            mkt_rets.append(gg["fwd"].mean()); dates.append(d)
            rr = reg.asof(d)
            scale = 1.0
            if rr is not None and np.isfinite(rr.get("mkt_above_sma200", np.nan)):
                if rr["mkt_above_sma200"] < 0.5:
                    scale *= 0.4
                if rr["vix_z_252"] > 1.0:
                    scale *= 0.6
            reg_rets.append(top * scale)

    def stats(rets):
        r = pd.Series(rets).dropna(); py = 12 / HOLD_M
        cagr = (1 + r).prod() ** (py / len(r)) - 1
        sharpe = r.mean() / r.std() * np.sqrt(py) if r.std() else 0
        eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
        return cagr, sharpe, dd

    ic = np.nanmean(ics)
    print(f"\n  Information Coefficient: {ic:+.4f}  "
          f"({'SKILL PRESENT' if ic > 0.02 else 'weak/none'})")
    print(f"  Monthly: top-{TOPN} {np.mean(top_rets)*100:+.2f}%  bottom-{TOPN} "
          f"{np.mean(bot_rets)*100:+.2f}%  market {np.mean(mkt_rets)*100:+.2f}%")
    print(f"\n  {'book':<22}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
    print("  " + "-" * 44)
    for name, rets in [(f"long top-{TOPN}", top_rets),
                       (f"long top-{TOPN} + regime", reg_rets),
                       ("equal-weight market", mkt_rets)]:
        cg, sh, dd = stats(rets)
        print(f"  {name:<22}{cg*100:>7.1f}%{sh:>8.2f}{dd*100:>7.0f}%")
    # which factors does it lean on?
    imp = pd.Series(m.feature_importances_, index=FACTORS).sort_values(ascending=False)
    print(f"\n  top factors: " + ", ".join(f"{k}({v})" for k, v in imp.head(6).items()))
    print(f"  price-only IC was -0.004 (no skill) -- compare above.")


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
