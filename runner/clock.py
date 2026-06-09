"""Market clock — real Eastern time via zoneinfo.

The old code hardcoded ET as UTC-4 in three places, which runs one hour ahead
of the actual market all winter (EST = UTC-5): the loop would start "at 9:30"
while the market was still pre-open, and flat-by-close fired at 14:58 ET.
Market holidays are still not modeled; gate on the broker clock for those.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN_MIN = 9 * 60 + 30
CLOSE_MIN = 16 * 60


def et_now() -> dt.datetime:
    return dt.datetime.now(ET)


def minutes_since_open(now: dt.datetime | None = None) -> float:
    now = now or et_now()
    return max(now.hour * 60 + now.minute - OPEN_MIN, 0)


def minutes_to_close(now: dt.datetime | None = None) -> float:
    now = now or et_now()
    return CLOSE_MIN - (now.hour * 60 + now.minute)


def is_weekday(now: dt.datetime | None = None) -> bool:
    return (now or et_now()).weekday() < 5
