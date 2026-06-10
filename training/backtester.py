"""Mass simulation engine on real historical market data."""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents.indicators import calculate_atr, calculate_rolling_vwap
from agents.setups.registry import SETUP_REGISTRY
from config import (
    BASE_DIR,
    CHECKPOINT_CHUNK_SIZE,
    CHECKPOINT_EARLY_STOP_SETUPS,
    CHECKPOINT_MAX_SETUPS_PER_TICKER,
    CHECKPOINT_WALK_STEP,
    FULL_EARLY_STOP_SETUPS,
    FULL_MAX_SETUPS_PER_TICKER,
    FULL_WALK_STEP,
    MAX_HOLDING_DAYS,
    TRAINING_YEARS,
)
from training.history import download_history
from training.universe import load_training_universe

TRAINING_DIR = BASE_DIR / "training"
RESULTS_DIR = TRAINING_DIR / "results"

TRAINING_UNIVERSE = load_training_universe()


@dataclass
class TrainingProfile:
    """Controls thoroughness vs speed for checkpoint vs final runs."""
    name: str
    walk_step: int = 1
    max_setups_per_ticker: int | None = None
    early_stop_real_setups: int | None = None
    chunk_size: int = 24
    slippage_levels: list[float] | None = None


def _profile_for_simulations(simulations: int) -> TrainingProfile:
    is_checkpoint = simulations <= 2_500
    if is_checkpoint:
        return TrainingProfile(
            name="checkpoint",
            walk_step=CHECKPOINT_WALK_STEP,
            max_setups_per_ticker=CHECKPOINT_MAX_SETUPS_PER_TICKER,
            early_stop_real_setups=CHECKPOINT_EARLY_STOP_SETUPS,
            chunk_size=CHECKPOINT_CHUNK_SIZE,
            slippage_levels=[0.0],
        )
    return TrainingProfile(
        name="full",
        walk_step=FULL_WALK_STEP,
        max_setups_per_ticker=FULL_MAX_SETUPS_PER_TICKER,
        early_stop_real_setups=FULL_EARLY_STOP_SETUPS,
        chunk_size=CHECKPOINT_CHUNK_SIZE,
        slippage_levels=[0.0, 0.1, 0.2],
    )


def _cap_setups(
    setups: list[tuple[int, dict]],
    max_per_ticker: int,
    ticker: str,
) -> list[tuple[int, dict]]:
    """Stratified sample so each pattern type is represented."""
    if len(setups) <= max_per_ticker:
        return setups

    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    by_type: dict[str, list[tuple[int, dict]]] = {}
    for item in setups:
        by_type.setdefault(item[1]["setup_type"], []).append(item)

    n_types = len(by_type)
    per_type = max(1, max_per_ticker // n_types)
    capped: list[tuple[int, dict]] = []

    for items in by_type.values():
        if len(items) <= per_type:
            capped.extend(items)
        else:
            picks = rng.choice(len(items), size=per_type, replace=False)
            capped.extend(items[i] for i in picks)

    if len(capped) > max_per_ticker:
        picks = rng.choice(len(capped), size=max_per_ticker, replace=False)
        capped = [capped[i] for i in picks]

    return capped


@dataclass
class SimResult:
    ticker: str
    entry_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    pnl_r: float
    pnl_dollars: float
    exit_reason: str
    days_held: int
    setup_score: int
    setup_type: str
    won: bool
    bootstrap_id: int = 0
    slippage_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Trailing window passed to each analyzer per bar. Must exceed every analyzer's
# raw price lookback (deepest is double_bottom's 40 bars) and the largest length
# guard (55). Window-start-sensitive indicators (EMA_21, VWAP, SMA_200) are
# precomputed full-history below, so the window only needs to cover raw lookbacks.
BACKTEST_LOOKBACK = 128


def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach full-history causal indicators once per ticker.

    Each is causal, so its value at bar ``i`` is identical whether computed over
    the whole series or over the prefix ``df.iloc[:i+1]``. Pre-attaching them lets
    the per-bar walk read them from a small trailing window instead of recomputing
    over a growing prefix — turning O(n^2) setup detection into O(n) with
    bit-for-bit identical results.
    """
    out = df.copy()
    out["EMA_21"] = out["Close"].ewm(span=21, adjust=False).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    out["VWAP"] = calculate_rolling_vwap(out, 20)
    return out


def find_historical_setups(
    df: pd.DataFrame, min_forward: int = 15, step: int = 1,
    lookback: int = BACKTEST_LOOKBACK,
) -> list[tuple[int, dict]]:
    """Walk history and return (bar_index, setup) for every valid pattern."""
    found = []
    ind = _precompute_indicators(df)
    n = len(ind)
    for i in range(40, n - min_forward, step):
        # Bounded trailing window: identical results to df.iloc[:i+1] because
        # analyzers only read precomputed indicators + raw lookbacks <= 40 bars,
        # and length guards (<=55) behave the same once the window reaches 256.
        start = max(0, i - lookback + 1)
        window = ind.iloc[start : i + 1]
        for _setup_type, analyzer in SETUP_REGISTRY.items():
            setup = analyzer(window)
            if setup["is_valid_setup"]:
                found.append((i, setup))
    return found


def simulate_trade_forward(
    df: pd.DataFrame,
    entry_idx: int,
    setup: dict,
    max_risk: float = 10.0,
    slippage_pct: float = 0.0,
    atr14: pd.Series | None = None,
) -> SimResult | None:
    """Simulate one trade forward using real subsequent bars."""
    if not setup.get("is_valid_setup"):
        return None

    bearish = setup.get("bias") == "bearish"
    # adverse slippage: longs fill higher, shorts fill LOWER (the old +slip on
    # shorts improved their entries)
    slip = slippage_pct / 100
    entry = setup["current_price"] * ((1 - slip) if bearish else (1 + slip))
    stop = round(setup.get("stop_loss") or setup["resistance_level"] * 0.98, 2)
    risk_per_share = (stop - entry) if bearish else (entry - stop)
    if risk_per_share <= 0:
        return None

    shares = max_risk / risk_per_share
    risk = risk_per_share
    target_1 = entry - risk if bearish else entry + risk
    target_2 = entry - risk * 2 if bearish else entry + risk * 2
    scale_1_done = scale_2_done = False
    shares_remaining = shares
    extreme = entry
    trailing_stop = stop
    total_pnl = 0.0

    entry_date = str(df.index[entry_idx].date()) if hasattr(df.index[entry_idx], "date") else str(entry_idx)

    for day_offset in range(1, MAX_HOLDING_DAYS + 1):
        bar_idx = entry_idx + day_offset
        if bar_idx >= len(df):
            break

        bar = df.iloc[bar_idx]
        close = float(bar["Close"])
        low = float(bar["Low"])
        high = float(bar["High"])

        def _pnl(exit_px: float, remaining: float) -> float:
            if bearish:
                return (entry - exit_px) * remaining + total_pnl
            return (exit_px - entry) * remaining + total_pnl

        stopped = (high >= stop) if bearish else (low <= stop)
        if stopped:
            pnl = _pnl(stop, shares_remaining)
            return SimResult(
                ticker="", entry_date=entry_date, entry_price=round(entry, 2),
                exit_price=stop, stop_loss=stop, pnl_r=round(pnl / max_risk, 2),
                pnl_dollars=round(pnl, 2), exit_reason="HARD_STOP",
                days_held=day_offset, setup_score=setup["confidence_score"],
                setup_type=setup["setup_type"], won=pnl > 0, slippage_pct=slippage_pct,
            )

        hit_t1 = (close <= target_1) if bearish else (close >= target_1)
        if hit_t1 and not scale_1_done:
            sold = shares * 0.33
            total_pnl += ((entry - target_1) if bearish else (target_1 - entry)) * sold
            shares_remaining -= sold
            scale_1_done = True
            extreme = min(extreme, close) if bearish else max(extreme, close)

        hit_t2 = (close <= target_2) if bearish else (close >= target_2)
        if hit_t2 and not scale_2_done:
            sold = shares * 0.33
            total_pnl += ((entry - target_2) if bearish else (target_2 - entry)) * sold
            shares_remaining -= sold
            scale_2_done = True
            extreme = min(extreme, close) if bearish else max(extreme, close)

        if scale_1_done or hit_t1:
            # calculate_atr is causal, so a precomputed full-series ATR read at
            # bar_idx equals recomputing on the prefix — same value, far cheaper.
            if atr14 is not None:
                atr = float(atr14.iloc[bar_idx])
            else:
                atr = float(calculate_atr(df.iloc[: bar_idx + 1], 14).iloc[-1])
            # The level a live stop order rests at TODAY was set from data
            # through yesterday's close — test the bar against the prior level
            # first, then ratchet with today's close/ATR for tomorrow.
            # (Raising the trail with today's close and triggering it on the
            # same bar's low was intrabar lookahead.)
            if bearish:
                trail_hit = high >= trailing_stop and trailing_stop < stop
                if not trail_hit:
                    extreme = min(extreme, close)
                    trailing_stop = min(trailing_stop, extreme + atr * 2.0)
            else:
                trail_hit = low <= trailing_stop and trailing_stop > stop
                if not trail_hit:
                    extreme = max(extreme, close)
                    trailing_stop = max(trailing_stop, extreme - atr * 2.0)
            if trail_hit:
                pnl = _pnl(trailing_stop, shares_remaining)
                return SimResult(
                    ticker="", entry_date=entry_date, entry_price=round(entry, 2),
                    exit_price=round(trailing_stop, 2), stop_loss=stop,
                    pnl_r=round(pnl / max_risk, 2), pnl_dollars=round(pnl, 2),
                    exit_reason="TRAILING_STOP", days_held=day_offset,
                    setup_score=setup["confidence_score"],
                    setup_type=setup["setup_type"], won=pnl > 0,
                    slippage_pct=slippage_pct,
                )

        if day_offset >= 10:
            max_profit = risk * shares
            current = _pnl(close, shares_remaining)
            if current < max_profit * 0.25:
                return SimResult(
                    ticker="", entry_date=entry_date, entry_price=round(entry, 2),
                    exit_price=round(close, 2), stop_loss=stop,
                    pnl_r=round(current / max_risk, 2), pnl_dollars=round(current, 2),
                    exit_reason="TIME_STOP", days_held=day_offset,
                    setup_score=setup["confidence_score"],
                    setup_type=setup["setup_type"], won=current > 0,
                    slippage_pct=slippage_pct,
                )

    final_close = float(df.iloc[min(entry_idx + MAX_HOLDING_DAYS, len(df) - 1)]["Close"])
    pnl = _pnl(final_close, shares_remaining)
    return SimResult(
        ticker="", entry_date=entry_date, entry_price=round(entry, 2),
        exit_price=round(final_close, 2), stop_loss=stop,
        pnl_r=round(pnl / max_risk, 2), pnl_dollars=round(pnl, 2),
        exit_reason="MAX_HOLD", days_held=MAX_HOLDING_DAYS,
        setup_score=setup["confidence_score"],
        setup_type=setup["setup_type"], won=pnl > 0,
        slippage_pct=slippage_pct,
    )


def _process_ticker(args: tuple) -> tuple[str, list[dict], int]:
    ticker, df_dict, slippage_levels, walk_step, max_setups = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = pd.DataFrame(df_dict)
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    raw_setups = find_historical_setups(df, step=walk_step)
    if max_setups:
        setups = _cap_setups(raw_setups, max_setups, ticker)
    else:
        setups = raw_setups

    atr14 = calculate_atr(df, 14)
    results = []
    for idx, setup in setups:
        for slip in slippage_levels:
            sim = simulate_trade_forward(df, idx, setup, slippage_pct=slip, atr14=atr14)
            if sim:
                sim.ticker = ticker
                results.append(sim.to_dict())
    return ticker, results, len(raw_setups)


def bootstrap_expand(base_results: list[dict], target: int, seed: int = 42) -> list[dict]:
    """Resample real trade outcomes with noise to reach target simulation count."""
    if not base_results:
        return []
    if len(base_results) >= target:
        return base_results[:target]

    rng = np.random.default_rng(seed)
    expanded = list(base_results)
    id_counter = 0

    while len(expanded) < target:
        sample = base_results[rng.integers(0, len(base_results))]
        noise_r = rng.normal(0, 0.15)
        noise_dollars = noise_r * 10.0
        slip = round(float(rng.uniform(0.05, 0.35)), 3)

        copy = dict(sample)
        copy["bootstrap_id"] = id_counter
        copy["pnl_r"] = round(sample["pnl_r"] + noise_r, 2)
        copy["pnl_dollars"] = round(sample["pnl_dollars"] + noise_dollars, 2)
        copy["won"] = bool(copy["pnl_r"] > 0)
        copy["slippage_pct"] = slip
        expanded.append(copy)
        id_counter += 1

    return expanded[:target]


def summarize_results(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}

    pnls = [r["pnl_r"] for r in results]
    wins = [r for r in results if r["won"]]
    losses = [r for r in results if not r["won"]]

    win_rate = len(wins) / len(results)
    avg_win = np.mean([r["pnl_r"] for r in wins]) if wins else 0
    avg_loss = np.mean([abs(r["pnl_r"]) for r in losses]) if losses else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    by_score = {}
    for r in results:
        bucket = (r["setup_score"] // 10) * 10
        by_score.setdefault(bucket, []).append(r["pnl_r"])

    score_edge = {
        str(k): round(float(np.mean(v)), 3)
        for k, v in sorted(by_score.items())
    }

    by_ticker: dict[str, list] = {}
    for r in results:
        by_ticker.setdefault(r["ticker"], []).append(r["pnl_r"])

    return {
        "count": len(results),
        "real_trades": len([r for r in results if r.get("bootstrap_id", 0) == 0]),
        "bootstrapped": len([r for r in results if r.get("bootstrap_id", 0) > 0]),
        "win_rate": round(win_rate * 100, 1),
        "avg_winner_r": round(float(avg_win), 2),
        "avg_loser_r": round(float(avg_loss), 2),
        "expectancy": round(float(expectancy), 3),
        "total_pnl_dollars": round(sum(r["pnl_dollars"] for r in results), 2),
        "avg_pnl_dollars": round(float(np.mean([r["pnl_dollars"] for r in results])), 2),
        "max_drawdown_r": round(float(min(np.minimum.accumulate(np.cumsum(pnls)))), 2) if pnls else 0,
        "score_edge": score_edge,
        "best_tickers": sorted(
            [(t, round(float(np.mean(p)), 3)) for t, p in by_ticker.items() if len(p) >= 3],
            key=lambda x: x[1], reverse=True,
        )[:10],
        "exit_reasons": _count_field(results, "exit_reason"),
        "by_setup_type": _summarize_by_setup(results),
    }


def _summarize_by_setup(results: list[dict]) -> dict:
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.get("setup_type", "unknown"), []).append(r)

    out = {}
    for setup_type, trades in by_type.items():
        real = [t for t in trades if t.get("bootstrap_id", 0) == 0]
        wins = [t for t in trades if t["won"]]
        wr = len(wins) / len(trades) if trades else 0
        avg_pnl = float(np.mean([t["pnl_r"] for t in trades])) if trades else 0
        out[setup_type] = {
            "count": len(trades),
            "real_count": len(real),
            "win_rate": round(wr * 100, 1),
            "expectancy": round(avg_pnl, 3),
        }
    return out


def _count_field(results: list[dict], field: str) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r[field]] = counts.get(r[field], 0) + 1
    return counts


def run_training(
    simulations: int = 10_000,
    years: int | None = None,
    slippage_levels: list[float] | None = None,
    use_bootstrap: bool = True,
    workers: int = 4,
    tickers: list[str] | None = None,
    refresh_cache: bool = False,
) -> dict:
    """
    Run mass simulations on real historical data.

    Phase 1: Walk real price history, find actual breakout setups
    Phase 2: Simulate each with real forward bars + exit rules
    Phase 3: Bootstrap resample to reach target count (10k/50k/100k)
    Phase 4: Calibrate agent parameters from results
    """
    from training.calibrator import calibrate, save_learned_params

    years = TRAINING_YEARS if years is None else years
    profile = _profile_for_simulations(simulations)
    slippage_levels = slippage_levels or profile.slippage_levels or [0.0, 0.1, 0.2]
    ticker_list = tickers or load_training_universe()

    period_label = "max" if years >= 10 else f"{years}y"
    print(f"Downloading real data for {len(ticker_list)} tickers (period={period_label})...")
    history = download_history(ticker_list, years=years, refresh=refresh_cache)
    print(f"  Loaded {len(history)} tickers with sufficient history.")
    if history:
        sample = next(iter(history.values()))
        print(f"  Sample bars per ticker: ~{len(sample)} "
              f"({sample.index[0].date()} → {sample.index[-1].date()})")

    setup_names = ", ".join(SETUP_REGISTRY.keys())
    print(f"Phase 1-2: Walking history [{profile.name}] walk_step={profile.walk_step} "
          f"cap={profile.max_setups_per_ticker or 'none'} "
          f"early_stop={profile.early_stop_real_setups or 'none'}")
    print(f"  Patterns: {setup_names}")
    base_results: list[dict] = []
    tickers_done = 0
    phase_start = time.time()

    items = list(history.items())
    rng = np.random.default_rng(42)
    rng.shuffle(items)

    def _run_ticker(ticker: str, df: pd.DataFrame) -> tuple[str, list[dict], int]:
        raw = find_historical_setups(df, step=profile.walk_step)
        setups = (
            _cap_setups(raw, profile.max_setups_per_ticker, ticker)
            if profile.max_setups_per_ticker
            else raw
        )
        atr14 = calculate_atr(df, 14)
        results = []
        for idx, setup in setups:
            for slip in slippage_levels:
                sim = simulate_trade_forward(df, idx, setup, slippage_pct=slip, atr14=atr14)
                if sim:
                    sim.ticker = ticker
                    results.append(sim.to_dict())
        return ticker, results, len(raw)

    def _log_ticker(ticker: str, added: int, raw: int, elapsed: float) -> None:
        cap_note = ""
        if profile.max_setups_per_ticker and raw > added:
            cap_note = f" (capped from {raw:,})"
        print(
            f"  [{tickers_done}/{len(items)}] {ticker}: "
            f"+{added:,} sims{cap_note} — {len(base_results):,} total ({elapsed:.0f}s)"
        )

    if workers > 1 and len(history) > 4:
        for chunk_start in range(0, len(items), profile.chunk_size):
            chunk = items[chunk_start : chunk_start + profile.chunk_size]
            tasks = []
            for ticker, df in chunk:
                df_reset = df.reset_index()
                date_col = df_reset.columns[0]
                df_reset = df_reset.rename(columns={date_col: "date"})
                tasks.append((
                    ticker, df_reset.to_dict(), slippage_levels,
                    profile.walk_step, profile.max_setups_per_ticker,
                ))

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process_ticker, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    tickers_done += 1
                    t0 = time.time()
                    try:
                        ticker, batch, raw = future.result()
                        base_results.extend(batch)
                        _log_ticker(ticker, len(batch), raw, time.time() - phase_start)
                    except Exception as e:
                        print(f"  [!] {futures[future]}: {e}")

            if (
                profile.early_stop_real_setups
                and len(base_results) >= profile.early_stop_real_setups
            ):
                print(
                    f"  Early stop: {len(base_results):,} real setups "
                    f"(target {profile.early_stop_real_setups:,})"
                )
                break
    else:
        for ticker, df in items:
            tickers_done += 1
            ticker_name, batch, raw = _run_ticker(ticker, df)
            base_results.extend(batch)
            _log_ticker(ticker_name, len(batch), raw, time.time() - phase_start)
            if (
                profile.early_stop_real_setups
                and len(base_results) >= profile.early_stop_real_setups
            ):
                print(
                    f"  Early stop: {len(base_results):,} real setups "
                    f"(target {profile.early_stop_real_setups:,})"
                )
                break

    print(f"  Found {len(base_results)} real historical simulations.")

    if use_bootstrap and len(base_results) < simulations:
        print(f"Phase 3: Bootstrapping to {simulations:,} simulations...")
        all_results = bootstrap_expand(base_results, simulations)
    else:
        # The simulation target is a floor (how many to bootstrap up to), not a
        # cap. When real history already meets it, keep every real setup across
        # all tickers — discarding real data to hit an exact count would bias
        # calibration toward whichever tickers finished first and shrink the
        # per-pattern sample sizes the trust thresholds rely on.
        print(f"Phase 3: Real setups ({len(base_results):,}) >= target "
              f"({simulations:,}) — keeping all real, no bootstrap.")
        all_results = base_results

    from agents.quant import summarize_backtest

    summary = summarize_results(all_results)
    summary["quant_metrics"] = summarize_backtest(all_results)
    summary["data_source"] = "yfinance_real_ohlcv"
    summary["years"] = years
    summary["history_period"] = period_label
    summary["training_profile"] = profile.name
    summary["walk_step"] = profile.walk_step
    summary["max_setups_per_ticker"] = profile.max_setups_per_ticker
    summary["tickers_requested"] = len(ticker_list)
    summary["tickers_scanned"] = tickers_done
    summary["phase12_seconds"] = round(time.time() - phase_start, 1)
    summary["target_simulations"] = simulations

    print("Phase 4: Options spread backtest on equity paths...")
    from training.options_backtest import backtest_options_from_equity_results
    options_bt = backtest_options_from_equity_results(history, base_results)
    summary["options_backtest"] = options_bt
    print(f"  {options_bt.get('summary', 'No options results')}")

    print("Phase 5: Calibrating agent parameters...")
    learned = calibrate(all_results)
    learned["options_backtest"] = options_bt
    save_learned_params(learned)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"training_{simulations}_{ts}.json"
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    result_path.write_text(
        json.dumps(_json_safe({
            "summary": summary,
            "learned_params": learned,
            "sample_trades": all_results[:50],
        }), indent=2),
        encoding="utf-8",
    )

    summary["learned_params"] = learned
    summary["result_file"] = str(result_path)
    return summary