"""Purge-gap regression test: the gap before each test block must cover the
label horizon in CALENDAR days (14 trading days ~ 20 calendar days), plus the
embargo."""

import numpy as np
import pandas as pd

from training.ml.splits import purged_walkforward_splits


def test_purge_gap_covers_trading_day_horizon():
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    splits = purged_walkforward_splits(dates, n_splits=4, label_horizon=14, embargo=3)
    assert splits
    values = dates.values
    for train_idx, test_idx in splits:
        last_train = values[train_idx].max()
        first_test = values[test_idx].min()
        gap_days = (first_test - last_train) / np.timedelta64(1, "D")
        # 14 trading days -> >=20 calendar days, +3 embargo
        assert gap_days >= 20 + 3

def test_splits_are_time_ordered_and_expanding():
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    splits = purged_walkforward_splits(dates, n_splits=4)
    prev_train = 0
    for train_idx, test_idx in splits:
        assert dates.values[train_idx].max() < dates.values[test_idx].min()
        assert len(train_idx) >= prev_train  # expanding window
        prev_train = len(train_idx)
