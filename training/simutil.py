"""Shared simulation utilities: trading-day arithmetic and honest Sharpe.

Two bugs these helpers exist to kill (and keep dead):

* ``days_held`` from the backtester counts TRADING days. Adding it to the entry
  date as calendar days closes positions ~30% early, freeing capacity slots
  that don't exist and inflating trades/yr and compounded CAGR in every
  realized-book simulation.
* Annualizing per-trade-event returns with sqrt(252) inflates Sharpe by
  ~sqrt(252 / trades-per-year). The equity curve must be resampled to daily
  frequency (flat days count) before annualizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def trading_close_dates(dates, days_held) -> pd.DatetimeIndex:
    """Entry date + N TRADING days, as calendar timestamps.

    Weekends are skipped via ``np.busday_offset``; market holidays are not
    (a ~9-day/yr conservative approximation — close dates can only be later
    in reality, never earlier, so capacity is never overstated... and the
    error is <4% of the window vs ~40% for the calendar-day bug).
    """
    d = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]")
    held = np.asarray(days_held, dtype="int64")
    return pd.DatetimeIndex(np.busday_offset(d, held, roll="forward"))


def trading_to_calendar_days(trading_days: int) -> int:
    """Conservative calendar-day cover for a trading-day horizon (7/5 + holiday pad)."""
    return int(np.ceil(trading_days * 7 / 5)) + 2


def event_curve_sharpe(curve: pd.Series) -> float:
    """Annualized Sharpe from an event-based equity curve.

    ``curve`` is equity indexed by (possibly irregular) dates. Resamples to
    business-daily with forward-fill so flat days are counted, then annualizes
    with sqrt(252). Returns 0.0 when undefined.
    """
    if len(curve) < 2:
        return 0.0
    s = curve.copy()
    s.index = pd.to_datetime(s.index)
    daily = s.sort_index().resample("B").last().ffill()
    rets = daily.pct_change().dropna()
    sd = rets.std()
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))
