"""Live-signal harness: emit TODAY's recommended trades and log them for a
forward (blind) paper-trade track record.

Trains the deployment model on all labeled history, then scans the latest bar of
every ticker for valid setups, scores them with the exact validated pipeline
(regime-aware model, bear_breakdown excluded, blind P-floor), and prints the
ranked book a real trader would take today. Appends each signal, date-stamped,
to ``training/ml/datasets/paper_trades.csv`` so you can score them as the future
arrives -- the only test that can't be gamed.

Run daily (with fresh data) once a live feed is wired:
    python -m training.live_signals --top 10 --risk 0.5
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from agents.setups.registry import SETUP_REGISTRY
from training.history import download_history
from training.ml.features import FEATURE_COLUMNS, compute_feature_frame, compute_market_frame
from training.universe import load_training_universe

DATASET = Path(__file__).resolve().parent / "ml" / "datasets" / "survivorship_free_v2.parquet"
LOG = DATASET.parent / "paper_trades.csv"
DEAD = {"bear_breakdown"}
# setup_code must match training encoding (alphabetical over non-dead setups).
SETUP_CODE = {s: i for i, s in enumerate(sorted(set(SETUP_REGISTRY) - DEAD))}


def _deployment_model():
    import lightgbm as lgb
    df = pd.read_parquet(DATASET)
    df = df[~df["setup_type"].isin(DEAD)].copy()
    df["setup_code"] = df["setup_type"].map(SETUP_CODE)
    fc = FEATURE_COLUMNS + ["setup_score", "setup_code"]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                           n_jobs=-1, verbosity=-1)
    m.fit(df[fc].to_numpy("float64"), df["y_win"].to_numpy("int64"))
    floor = float(np.quantile(m.predict_proba(df[fc].to_numpy("float64"))[:, 1], 0.90))
    return m, fc, floor


def scan(top: int, risk_pct: float, lookback: int = 1):
    hist = download_history(load_training_universe())
    market = compute_market_frame(hist.get("^GSPC"), hist.get("^VIX"))
    model, fc, floor = _deployment_model()

    # regime context (latest market bar)
    mk_last = market.iloc[-1]
    regime_ok = mk_last["mkt_above_sma200"] > 0.5 and mk_last["vix_z_252"] < 1.5
    asof = max(df.index[-1] for df in hist.values() if len(df))

    cands = []
    stale = 0
    for tk, df in hist.items():
        if tk in ("^GSPC", "^VIX") or len(df) < 260:
            continue
        # LIVE means currently listed: skip names whose last bar isn't recent
        # (the survivorship-free universe carries 334 delisted names whose last
        # bar is their delisting day -- great for backtests, untradeable today).
        if (asof - df.index[-1]).days > 7:
            stale += 1
            continue
        feats = compute_feature_frame(df, market=market)
        for back in range(lookback):                 # scan last `lookback` bars
            i = len(df) - 1 - back
            if i < 260:
                break
            frow = feats.iloc[i]
            if frow[FEATURE_COLUMNS].isna().all():
                continue
            win = df.iloc[: i + 1]
            for stype, analyzer in SETUP_REGISTRY.items():
                if stype in DEAD:
                    continue
                try:
                    s = analyzer(win)
                except Exception:
                    continue
                if not s.get("is_valid_setup"):
                    continue
                x = np.array([[*(frow[c] for c in FEATURE_COLUMNS),
                               s.get("confidence_score", 0), SETUP_CODE[stype]]], dtype="float64")
                p = float(model.predict_proba(x)[:, 1][0])
                if p < floor:
                    continue
                entry = s["current_price"]; stop = s["stop_loss"]
                cands.append({
                    "asof": str(df.index[i].date()), "ticker": tk, "setup": stype,
                    "bias": s.get("bias", "bullish"), "entry": round(entry, 2),
                    "stop": round(stop, 2), "atr14": s.get("atr_14", 0),
                    "risk_pct_of_price": round(abs(entry - stop) / entry * 100, 1) if entry else 0,
                    "model_p": round(p, 3), "suggested_risk_pct": risk_pct,
                })

    cands.sort(key=lambda r: (r["asof"], r["model_p"]), reverse=True)
    book = cands[:top]

    print("=" * 74)
    print(f"  LIVE SIGNALS as of {str(asof.date())}   (model P-floor {floor:.3f})")
    print("=" * 74)
    print(f"  Regime: S&P>200d={'Y' if mk_last['mkt_above_sma200']>0.5 else 'N'}  "
          f"VIX z-score={mk_last['vix_z_252']:+.2f}  -> {'RISK-ON (full size)' if regime_ok else 'CAUTION (size down / stand aside)'}")
    print(f"  ({stale} delisted/stale names skipped -- not currently tradeable)")
    print(f"  {len(cands)} setups cleared the model; showing top {len(book)}:\n")
    print(f"  {'ticker':<7}{'setup':<15}{'bias':<9}{'entry':>9}{'stop':>9}{'risk%':>7}{'P(win)':>8}")
    print("  " + "-" * 62)
    for r in book:
        print(f"  {r['ticker']:<7}{r['setup']:<15}{r['bias']:<9}{r['entry']:>9.2f}"
              f"{r['stop']:>9.2f}{r['risk_pct_of_price']:>6.1f}%{r['model_p']:>8.3f}")
    if not regime_ok:
        print("\n  ** Regime is risk-off: the system would trade SMALL or not at all today. **")

    if book:
        new = pd.DataFrame(book)
        new["logged_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        hdr = not LOG.exists()
        new.to_csv(LOG, mode="a", header=hdr, index=False)
        print(f"\n  Logged {len(book)} signals -> {LOG} (track these forward to build a blind record).")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--risk", type=float, default=0.5, help="suggested %% equity risk/trade")
    p.add_argument("--lookback", type=int, default=1,
                   help="scan the last N bars per name (1=today; >1 backfills recent signals)")
    a = p.parse_args(argv)
    scan(a.top, a.risk, a.lookback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
