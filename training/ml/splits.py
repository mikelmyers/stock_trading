"""Purged, embargoed walk-forward cross-validation.

Financial samples violate the IID assumption ordinary k-fold relies on:

  * Two setups a few days apart share an overlapping forward window, so their
    labels are correlated. If one lands in train and the other in test, the
    model effectively trains on test-period information (leakage).
  * The future cannot inform the past — splits must be strictly time-ordered.

This module produces expanding-window folds where the training set ends a gap of
``label_horizon`` trading days (converted to calendar days) plus ``embargo``
calendar days *before* each test block begins. That
gap purges any training label whose forward window could overlap the test
period. See López de Prado, *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.simutil import trading_to_calendar_days


def purged_walkforward_splits(
    dates,
    n_splits: int = 5,
    label_horizon: int = 14,
    embargo: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) integer-index pairs over time-ordered ``dates``.

    Parameters
    ----------
    dates : array-like of date-like values (one per sample, any order).
    n_splits : number of forward test blocks.
    label_horizon : max TRADING days a label looks forward (== MAX_HOLDING_DAYS);
        converted to a calendar-day gap internally.
    embargo : extra calendar days dropped before each test block.
    """
    dates = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    values = dates.values
    unique_sorted = np.unique(values)

    # First chunk is the initial (always-train) history; the rest are test blocks.
    chunks = np.array_split(unique_sorted, n_splits + 1)
    gap = np.timedelta64(trading_to_calendar_days(label_horizon) + embargo, "D")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        block = chunks[k]
        if len(block) == 0:
            continue
        test_start, test_end = block[0], block[-1]
        train_mask = values < (test_start - gap)
        test_mask = (values >= test_start) & (values <= test_end)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, test_idx))
    return splits
