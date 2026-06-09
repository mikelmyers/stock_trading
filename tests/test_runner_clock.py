"""The old runner computed 'ET' as UTC-4 year-round; in winter (EST=UTC-5)
every time gate ran an hour early. These tests pin real zoneinfo behavior."""

import datetime as dt

from runner.clock import ET, et_now, is_weekday, minutes_since_open, minutes_to_close


def test_winter_offset_is_utc_minus_5():
    # 2026-01-15 14:30 UTC == 09:30 EST. The old UTC-4 math said 10:30.
    winter_utc = dt.datetime(2026, 1, 15, 14, 30, tzinfo=dt.timezone.utc)
    et = winter_utc.astimezone(ET)
    assert (et.hour, et.minute) == (9, 30)
    assert minutes_since_open(et) == 0


def test_summer_offset_is_utc_minus_4():
    summer_utc = dt.datetime(2026, 6, 9, 13, 30, tzinfo=dt.timezone.utc)
    et = summer_utc.astimezone(ET)
    assert (et.hour, et.minute) == (9, 30)


def test_minutes_to_close():
    et = dt.datetime(2026, 6, 9, 15, 58, tzinfo=ET)
    assert minutes_to_close(et) == 2


def test_weekend_detection():
    assert not is_weekday(dt.datetime(2026, 6, 13, 12, 0, tzinfo=ET))  # Saturday
    assert is_weekday(dt.datetime(2026, 6, 9, 12, 0, tzinfo=ET))       # Tuesday


def test_et_now_is_tz_aware():
    assert et_now().tzinfo is not None
