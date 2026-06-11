"""Data sources for the runner scanner — agnostic core + Alpaca adapter + mock.

A DataSource yields ConditionVectors for the day's candidates. The base class owns
the raw->features computation (so every adapter produces identical condition-vectors);
adapters only implement `movers()` (which symbols are in play) and `snapshot()` (raw
fields for one symbol). MockSource lets the whole pipeline run/test offline (the
sandbox can't reach Alpaca). Float comes from a pluggable provider (SEC proxy / Webull).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from runner import conditions as C


def _avg_vol_20d(daily_bars: list, prev_daily: dict) -> Optional[float]:
    """20-day average volume from prior daily bars.

    New/low-history runners often return only today's bar from Alpaca; the old
    ``sum(prior)/max(len(prior),1)`` path produced 0 and killed rvol for every
    gainer. Fall back to yesterday's bar when no prior history exists.
    """
    prior = daily_bars[:-1] if daily_bars else []
    if prior:
        return sum(b["v"] for b in prior) / len(prior)
    if prev_daily.get("v"):
        return float(prev_daily["v"])
    return None


def _intraday_vol(bars: list) -> Optional[float]:
    if not bars:
        return None
    return float(sum(b["v"] for b in bars))


def _is_warrant_or_unit(symbol: str) -> bool:
    """Skip warrant/unit tickers from the movers screener (not runner setups)."""
    sym = symbol.upper()
    if "." in sym:
        return True
    # Screener units: SPKLW, DAICW, etc. (5+ chars ending in W)
    return len(sym) >= 5 and sym.endswith("W")


def _build_cv(symbol: str, raw: dict, minutes_since_open: float, regime: str | None):
    bars = raw.get("bars") or []
    price, prev_close, day_open = raw.get("price"), raw.get("prev_close"), raw.get("day_open")
    vw = C.vwap(bars) if bars else raw.get("vwap")
    vol_today, avg20, flt, atr = (raw.get("vol_today"), raw.get("avg_vol_20d"),
                                  raw.get("float_shares"), raw.get("atr"))
    bid, ask = raw.get("bid"), raw.get("ask")
    cv = C.ConditionVector(
        symbol=symbol, asof=raw.get("asof", ""),
        price=price, float_shares=flt, market_cap=raw.get("market_cap"),
        avg_vol_20d=avg20, sector=raw.get("sector"),
        rvol=C.rvol(vol_today, avg20, minutes_since_open) if (vol_today and avg20) else raw.get("rvol"),
        gap_pct=C.pct(day_open, prev_close) if (day_open and prev_close) else raw.get("gap_pct"),
        premarket_vol=raw.get("premarket_vol"), vol_today=vol_today,
        vol_to_float=(vol_today / flt) if (vol_today and flt) else None,
        gap_atr=(abs(day_open - prev_close) / atr) if (day_open and prev_close and atr) else None,
        pct_change=C.pct(price, prev_close) if (price and prev_close) else None,
        vwap=vw, dist_vwap_pct=C.pct(price, vw) if (price and vw) else None,
        vwap_slope=raw.get("vwap_slope"),
        dist_pm_high_pct=C.pct(price, raw["pm_high"]) if (price and raw.get("pm_high")) else None,
        dist_pm_low_pct=C.pct(price, raw["pm_low"]) if (price and raw.get("pm_low")) else None,
        extension_pct=C.pct(price, vw) if (price and vw) else None,
        has_news=bool(raw.get("news")), catalyst_type=raw.get("catalyst_type"),
        spread_pct=((ask - bid) / price * 100) if (ask and bid and price) else raw.get("spread_pct"),
        halts_today=raw.get("halts_today"),
        atr_pct=(atr / price * 100) if (atr and price) else None,
        minutes_since_open=minutes_since_open, market_regime=regime,
    )
    return C.classify(cv)


class DataSource:
    """Subclass implements movers() and snapshot(symbol)."""
    def movers(self) -> list[str]:
        raise NotImplementedError

    def snapshot(self, symbol: str) -> dict:
        raise NotImplementedError

    def minutes_since_open(self) -> float:
        raise NotImplementedError

    def regime(self) -> Optional[str]:
        return None

    def scan(self) -> list[C.ConditionVector]:
        mso, reg = self.minutes_since_open(), self.regime()
        out = []
        errors = 0
        for sym in self.movers():
            try:
                out.append(_build_cv(sym, self.snapshot(sym), mso, reg))
            except Exception as e:
                # never silent: an auth/rate-limit failure must not look like
                # "scanned 0 candidates" while the loop runs blind
                errors += 1
                print(f"    scan error {sym}: {e!r}"[:120])
        if errors:
            print(f"    ({errors} symbols failed to scan)")
        return out


class MockSource(DataSource):
    """Synthetic runners for offline testing — covers A+, dilution, extended, near-miss."""
    _DATA = {
        # symbol: raw snapshot (a clean A+ green-light)
        "AAAA": dict(asof="2026-06-09T14:00:00Z", price=4.10, prev_close=3.00, day_open=3.60,
                     vol_today=8_000_000, avg_vol_20d=1_500_000, float_shares=6_000_000,
                     atr=0.30, bid=4.09, ask=4.11, news=["FDA approval"], catalyst_type="fda",
                     halts_today=0, pm_high=4.20, pm_low=3.40,
                     bars=[dict(high=3.6, low=3.5, close=3.55, volume=2e6),
                           dict(high=4.2, low=3.9, close=4.10, volume=6e6)]),
        # dilution blow-up flag (offering)
        "BBBB": dict(asof="2026-06-09T14:00:00Z", price=2.50, prev_close=2.00, day_open=2.40,
                     vol_today=5_000_000, avg_vol_20d=900_000, float_shares=9_000_000,
                     atr=0.25, bid=2.49, ask=2.52, news=["$20M registered offering"],
                     catalyst_type="offering", halts_today=1, pm_high=2.60, pm_low=2.30,
                     bars=[dict(high=2.4, low=2.3, close=2.35, volume=2e6),
                           dict(high=2.6, low=2.4, close=2.50, volume=3e6)]),
        # near-miss: low RVOL, no news
        "CCCC": dict(asof="2026-06-09T14:00:00Z", price=6.0, prev_close=5.7, day_open=5.8,
                     vol_today=600_000, avg_vol_20d=1_000_000, float_shares=40_000_000,
                     atr=0.2, bid=5.98, ask=6.02, news=[], catalyst_type=None, halts_today=0,
                     pm_high=6.1, pm_low=5.75,
                     bars=[dict(high=5.8, low=5.7, close=5.75, volume=3e5),
                           dict(high=6.1, low=5.9, close=6.0, volume=3e5)]),
    }

    def movers(self):
        return list(self._DATA)

    def snapshot(self, symbol):
        return self._DATA[symbol]

    def minutes_since_open(self):
        return 30.0

    def regime(self):
        return "risk_on"


class AlpacaSource(DataSource):
    """Live adapter. Needs APCA_API_KEY_ID/SECRET (paid SIP feed recommended).
    float_provider(symbol)->shares is pluggable (SEC proxy / Webull). Untested in
    the Claude sandbox (network blocked) — runs locally."""
    DATA = "https://data.alpaca.markets"

    def __init__(self, float_provider: Optional[Callable[[str], Optional[float]]] = None,
                 top_n: int = 25):
        import requests  # noqa
        self._requests = requests
        self.kid = os.environ["APCA_API_KEY_ID"]
        self.sec = os.environ["APCA_API_SECRET_KEY"]
        self.float_provider = float_provider or (lambda s: None)
        self.top_n = top_n

    def _get(self, url, **params):
        h = {"APCA-API-KEY-ID": self.kid, "APCA-API-SECRET-KEY": self.sec}
        r = self._requests.get(url, headers=h, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def movers(self):
        j = self._get(f"{self.DATA}/v1beta1/screener/stocks/movers", top=self.top_n)
        out = []
        for m in j.get("gainers", []):
            sym = m["symbol"]
            if _is_warrant_or_unit(sym):
                continue
            out.append(sym)
        return out

    def snapshot(self, symbol):
        import datetime as dt
        snap = self._get(f"{self.DATA}/v2/stocks/{symbol}/snapshot")
        day, prev = snap.get("dailyBar", {}), snap.get("prevDailyBar", {})
        q = snap.get("latestQuote", {})
        trade = snap.get("latestTrade", {})
        daily = self._get(f"{self.DATA}/v2/stocks/{symbol}/bars",
                          timeframe="1Day", limit=21).get("bars", [])
        avg20 = _avg_vol_20d(daily, prev)
        mins = self._get(f"{self.DATA}/v2/stocks/{symbol}/bars",
                         timeframe="1Min", limit=390).get("bars", [])
        news = self._get(f"{self.DATA}/v1beta1/news", symbols=symbol, limit=5).get("news") or []
        vol_today = day.get("v")
        intraday = _intraday_vol(mins)
        if intraday and intraday > (vol_today or 0):
            vol_today = intraday
        # asof = the SCAN time. dailyBar.t is the bar's 4am open timestamp,
        # constant all day — using it made the labeler measure MFE over
        # midnight-2am and collapsed every intraday scan to one row.
        # price = last trade, not the (possibly minutes-stale) daily-bar close.
        return dict(
            asof=dt.datetime.now(dt.timezone.utc).isoformat(),
            price=trade.get("p") or day.get("c"), prev_close=prev.get("c"),
            day_open=day.get("o"), vol_today=vol_today, avg_vol_20d=avg20,
            float_shares=self.float_provider(symbol), bid=q.get("bp"), ask=q.get("ap"),
            news=news, catalyst_type=_catalyst_type(news), halts_today=None,
            pm_high=None, pm_low=None,
            bars=[dict(high=b["h"], low=b["l"], close=b["c"], volume=b["v"]) for b in mins],
        )

    def minutes_since_open(self):
        from runner import clock
        return clock.minutes_since_open()

    def regime(self):
        return None


def _catalyst_type(news) -> Optional[str]:
    """Crude headline -> catalyst category. Refine later."""
    text = " ".join((n.get("headline", "") for n in news)).lower()
    if any(k in text for k in ("offering", "registered direct", "dilut", "warrant")):
        return "offering"
    if any(k in text for k in ("fda", "phase", "trial")):
        return "fda"
    if any(k in text for k in ("earnings", "beats", "revenue", "guidance")):
        return "earnings"
    if any(k in text for k in ("partnership", "contract", "award", "deal")):
        return "partnership"
    return "news" if news else None
