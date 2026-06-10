"""Regression test for extract_ticker_df: with group_by="ticker" layout the
symbol sits on column level 0; the old code dropped level 1 (the field names)
whenever exactly one ticker was requested, breaking df["Close"] downstream."""

import numpy as np
import pandas as pd

from data import extract_ticker_df

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _frame(tickers):
    idx = pd.date_range("2026-01-02", periods=5, freq="B")
    cols = pd.MultiIndex.from_product([tickers, FIELDS])
    return pd.DataFrame(np.ones((5, len(cols))), index=idx, columns=cols)


def test_single_ticker_group_by_ticker_layout():
    df = extract_ticker_df(_frame(["NVDA"]), "NVDA", num_tickers=1)
    assert list(df.columns) == FIELDS
    assert df["Close"].iloc[-1] == 1.0


def test_multi_ticker_layout():
    df = extract_ticker_df(_frame(["NVDA", "AMD"]), "AMD", num_tickers=2)
    assert list(df.columns) == FIELDS


def test_field_major_layout_falls_back():
    idx = pd.date_range("2026-01-02", periods=5, freq="B")
    cols = pd.MultiIndex.from_product([FIELDS, ["NVDA"]])
    data = pd.DataFrame(np.ones((5, len(cols))), index=idx, columns=cols)
    df = extract_ticker_df(data, "NVDA", num_tickers=1)
    assert list(df.columns) == FIELDS


def test_flat_columns_passthrough():
    idx = pd.date_range("2026-01-02", periods=5, freq="B")
    data = pd.DataFrame(np.ones((5, 5)), index=idx, columns=FIELDS)
    df = extract_ticker_df(data, "NVDA", num_tickers=1)
    assert list(df.columns) == FIELDS
