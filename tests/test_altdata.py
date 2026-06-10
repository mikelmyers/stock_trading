"""Alt-data tests. The ones that matter most are the POINT-IN-TIME guarantees:
a setup row must never see a short-interest record before FINRA published it,
or an insider filing before it hit EDGAR."""

import numpy as np
import pandas as pd
import pytest

from training.altdata.insider import attach_insider, normalize as norm_insider
from training.altdata.short_interest import (
    attach_short_interest,
    normalize as norm_si,
)


def _si_raw():
    return pd.DataFrame({
        "symbolCode": ["AAA", "AAA", "BBB"],
        "settlementDate": ["2026-01-15", "2026-01-30", "2026-01-15"],
        "currentShortPositionQuantity": [1_000_000, 1_500_000, 200_000],
        "previousShortPositionQuantity": [800_000, 1_000_000, 250_000],
        "averageDailyVolumeQuantity": [100_000, 100_000, 50_000],
        "daysToCoverQuantity": [10.0, 15.0, 4.0],
    })


class TestShortInterest:
    def test_normalize_maps_aliases(self):
        si = norm_si(_si_raw())
        assert {"symbol", "settlement_date", "public_date",
                "short_interest", "days_to_cover", "si_chg"} <= set(si.columns)
        assert si.iloc[0]["si_chg"] == pytest.approx(0.25)   # 1.0M vs 0.8M

    def test_publication_lag_applied(self):
        si = norm_si(_si_raw())
        # 2026-01-15 (Thu) + 9 business days = 2026-01-28
        assert si.iloc[0]["public_date"] == pd.Timestamp("2026-01-28")

    def test_unrecognized_file_raises(self):
        with pytest.raises(ValueError, match="missing"):
            norm_si(pd.DataFrame({"foo": [1]}))

    def test_attach_is_point_in_time(self):
        si = norm_si(_si_raw())
        ev = pd.DataFrame({
            "ticker": ["AAA", "AAA"],
            # settled 01-15 but published 01-28: the 01-20 setup must NOT see it
            "date": ["2026-01-20", "2026-02-02"],
        })
        joined = attach_short_interest(ev, si)
        before = joined[joined["date"] == "2026-01-20"].iloc[0]
        after = joined[joined["date"] == "2026-02-02"].iloc[0]
        assert pd.isna(before["si_dtc"])          # not yet public
        assert after["si_dtc"] == pytest.approx(10.0)


def _insider_raw():
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3"],
        "FILING_DATE": ["2026-01-10", "2026-01-10", "2026-02-20"],
        "ISSUERTRADINGSYMBOL": ["AAA", "AAA", "AAA"],
    })
    trans = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3"],
        "TRANS_CODE": ["P", "P", "S"],
        "TRANS_SHARES": [1000, 500, 2000],
        "TRANS_PRICEPERSHARE": [10.0, 10.0, 12.0],
    })
    return sub, trans


class TestInsider:
    def test_normalize_aggregates_buys_and_sells(self):
        ins = norm_insider(*_insider_raw())
        jan = ins[ins["public_date"] == "2026-01-11"].iloc[0]   # filing +1d
        assert jan["net_buy_usd"] == pytest.approx(15_000.0)
        assert jan["buyers"] == 2                                # cluster of 2
        feb = ins[ins["public_date"] == "2026-02-21"].iloc[0]
        assert feb["net_buy_usd"] == pytest.approx(-24_000.0)
        assert feb["sell_events"] == 1

    def test_attach_point_in_time_and_window(self):
        ins = norm_insider(*_insider_raw())
        ev = pd.DataFrame({
            "ticker": ["AAA"] * 3,
            "date": ["2026-01-10",   # filing day itself: not yet public
                     "2026-01-15",   # sees the Jan cluster buy
                     "2026-06-01"],  # >90d later: window expired -> 0
        })
        joined = attach_insider(ev, ins).set_index("date")
        assert joined.loc[pd.Timestamp("2026-01-10"), "ins_netbuy_90d"] == 0.0
        assert joined.loc[pd.Timestamp("2026-01-15"), "ins_netbuy_90d"] == pytest.approx(15_000.0)
        assert joined.loc[pd.Timestamp("2026-01-15"), "ins_buyers_90d"] == 2
        assert joined.loc[pd.Timestamp("2026-06-01"), "ins_netbuy_90d"] == 0.0

    def test_missing_columns_raise(self):
        sub, trans = _insider_raw()
        with pytest.raises(ValueError, match="SUBMISSION"):
            norm_insider(sub.drop(columns=["FILING_DATE"]), trans)


def test_augment_alt_verdict_logic():
    from training.augment_alt import verdict_of
    assert verdict_of(0.01, 7, 8) == "adopt"
    assert verdict_of(0.001, 4, 8) == "reject"
    assert verdict_of(-0.01, 2, 8) == "reject"
    assert verdict_of(0.005, 5, 8) == "inconclusive"
