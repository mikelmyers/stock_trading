"""Sample-uniqueness weights for overlapping trade events.

With walk_step=1, a persistent pattern fires on many consecutive bars of the
same ticker; those rows share most of their forward window, so their labels
are nearly copies. Training on them unweighted lets the model farm easy
duplicates and makes 1.9M rows masquerade as 1.9M observations.

Weight_i = average over event i's holding window of 1/(number of concurrent
events of the same ticker covering that day) — López de Prado, *Advances in
Financial Machine Learning*, ch. 4 (average uniqueness). A lone event gets
1.0; thirty stacked near-duplicates get ~1/30 each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_weights(start_ord: np.ndarray, days_held: np.ndarray) -> np.ndarray:
    """Average uniqueness for one ticker's events, given business-day ordinals."""
    end_ord = start_ord + np.maximum(days_held, 1)        # inclusive window end
    lo = int(start_ord.min())
    hi = int(end_ord.max())
    span = hi - lo + 1
    # concurrency per day via difference array
    diff = np.zeros(span + 1, dtype="int64")
    np.add.at(diff, start_ord - lo, 1)
    np.add.at(diff, end_ord - lo + 1, -1)
    conc = np.cumsum(diff[:-1])
    # prefix sums of 1/concurrency -> window means in O(1) per event
    inv = np.zeros(span, dtype="float64")
    np.divide(1.0, conc, out=inv, where=conc > 0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    s, e = start_ord - lo, end_ord - lo
    return (csum[e + 1] - csum[s]) / (e - s + 1)


def uniqueness_weights(dates, days_held, tickers=None) -> np.ndarray:
    """Per-row average-uniqueness weights in (0, 1].

    Parameters
    ----------
    dates : entry dates (anything pd.to_datetime accepts).
    days_held : holding period in TRADING days.
    tickers : optional per-row symbols; overlap is only counted within the
        same ticker (events on different names are independent). When None,
        all rows are treated as one group.
    """
    d = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    ords = np.array(
        np.busday_count(np.datetime64("2000-01-03"), d.values.astype("datetime64[D]")),
        dtype="int64")
    held = np.asarray(days_held, dtype="int64")
    out = np.empty(len(d), dtype="float64")
    if tickers is None:
        out[:] = _group_weights(ords, held)
        return out
    tick = pd.Series(list(tickers)).reset_index(drop=True)
    for _, idx in tick.groupby(tick).groups.items():
        ix = np.asarray(idx)
        out[ix] = _group_weights(ords[ix], held[ix])
    return out
