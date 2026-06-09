"""Discipline + ruin-aware sizing engine — the psychology layer.

Rule-based (no learning needed), so it protects the account from trade one while
the classifier is still dumb. Turns "blow up to learn" into "lose small to learn":
every trade is sized so a normal losing streak is survivable, every loss is capped
at the -1R stop, and the rules block the behaviors that actually kill accounts
(revenge after losses, trading through a bad day, chasing blow-up conditions).

Decision flow per candidate (first failing gate wins):
  daily-loss-limit -> cooldown-lockout -> max-trades -> max-concurrent
  -> not-green-light -> blow-up-flag veto -> size (conviction x survival x liquidity).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from runner.conditions import ConditionVector

STATE_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class RiskConfig:
    # sizing (fraction of equity risked = the -1R amount)
    base_risk_frac: float = 0.08        # green-light "push"
    watch_risk_frac: float = 0.0        # non-green-light: default don't trade
    max_risk_frac: float = 0.10         # hard cap regardless of conviction
    # survival: cap risk so N straight losses keep drawdown <= max_streak_dd
    survival_streak: int = 7
    max_streak_dd: float = 0.50
    # stop placement (the -1R)
    default_stop_pct: float = 0.07
    min_stop_pct: float = 0.03
    max_stop_pct: float = 0.12
    # discipline gates
    daily_loss_limit_frac: float = 0.15  # stop for the day at -15%
    max_consecutive_losses: int = 3      # then cooldown
    cooldown_minutes: int = 60
    max_trades_per_day: int = 10
    max_concurrent: int = 2
    max_pct_of_dollar_vol: float = 0.01  # position notional <= 1% of today's $ volume
    max_leverage: float = 1.0            # notional <= equity x this (1.0 = cash, no margin)
    require_green_light: bool = True
    flat_by_close: bool = True
    # classifier gate (active only once a model is trained; dormant otherwise)
    use_classifier: bool = True
    min_p_monster: float = 0.50
    max_p_loss: float = 0.50

    def survival_cap(self) -> float:
        """Max risk fraction so survival_streak losses keep dd <= max_streak_dd."""
        return 1.0 - (1.0 - self.max_streak_dd) ** (1.0 / max(self.survival_streak, 1))

    def effective_max_risk(self) -> float:
        return min(self.max_risk_frac, self.survival_cap())


@dataclass
class Decision:
    symbol: str
    action: str                 # "take" | "skip"
    reason: str
    shares: int = 0
    entry: Optional[float] = None
    stop: Optional[float] = None
    risk_dollars: float = 0.0
    risk_frac: float = 0.0
    notional: float = 0.0
    flat_by_close: bool = True


@dataclass
class RunnerState:
    episode_id: str
    date: str = ""
    day_start_equity: float = 0.0
    equity: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    open_positions: int = 0
    locked_until: Optional[str] = None       # iso ts; trading blocked until then
    synced_date: str = ""                    # last date broker equity anchored the day

    @classmethod
    def load(cls, episode_id: str, equity: float) -> "RunnerState":
        p = STATE_DIR / f"state_{episode_id}.json"
        today = dt.date.today().isoformat()
        if p.exists():
            s = cls(**json.loads(p.read_text()))
            s.equity = equity
            if s.date != today:                  # new day -> reset daily counters
                s.date, s.day_start_equity, s.daily_pnl = today, equity, 0.0
                s.trades_today, s.consecutive_losses, s.locked_until = 0, 0, None
            return s
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return cls(episode_id=episode_id, date=today, day_start_equity=equity, equity=equity)

    def save(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATE_DIR / f"state_{self.episode_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt state

    def sync_with_broker(self, broker_equity: float, open_position_count: int):
        """Replace gate inputs with broker truth each cycle.

        The local counters drift (entries that never filled, stops that filled
        at the broker, manual trades), and equity as a CLI constant means the
        daily-loss limit can never fire. daily_pnl derived from broker equity
        includes unrealized P&L — the conservative measure for a circuit breaker.
        """
        self.equity = broker_equity
        self.open_positions = open_position_count
        if self.synced_date != self.date:    # first sync of the day anchors it
            self.day_start_equity = broker_equity
            self.synced_date = self.date
        self.daily_pnl = broker_equity - self.day_start_equity
        self.save()

    def register_outcome(self, pnl_dollars: float, r_multiple: float, cfg: RiskConfig, now=None):
        """Update streak/pnl/locks after a trade closes."""
        now = now or dt.datetime.now(dt.timezone.utc)
        self.equity += pnl_dollars
        self.daily_pnl += pnl_dollars
        self.open_positions = max(self.open_positions - 1, 0)
        if r_multiple > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= cfg.max_consecutive_losses:
                self.locked_until = (now + dt.timedelta(minutes=cfg.cooldown_minutes)).isoformat()
        if self.daily_pnl <= -cfg.daily_loss_limit_frac * self.day_start_equity:
            self.locked_until = _end_of_day(now)        # done for the day
        self.save()


def _end_of_day(now) -> str:
    return now.replace(hour=23, minute=59, second=0, microsecond=0).isoformat()


class RiskEngine:
    def __init__(self, cfg: RiskConfig | None = None, scorer=None):
        self.cfg = cfg or RiskConfig()
        self._scorer = scorer            # injectable; defaults to classifier.score (lazy)

    def _score(self, cv):
        """Classifier P(monster)/P(loss), or None when no model is trained yet."""
        if not self.cfg.use_classifier:
            return None
        if self._scorer is None:
            from runner.classifier import score as _s
            self._scorer = _s
        return self._scorer(cv)

    def _stop(self, cv: ConditionVector) -> Optional[float]:
        """-1R stop: VWAP if it's below price, else a default %; clamped to [min,max]%."""
        entry = cv.price
        if not entry:
            return None
        if cv.vwap and cv.vwap < entry:
            dist = (entry - cv.vwap) / entry
        else:
            dist = self.cfg.default_stop_pct
        dist = min(max(dist, self.cfg.min_stop_pct), self.cfg.max_stop_pct)
        return round(entry * (1 - dist), 4)

    def evaluate(self, cv: ConditionVector, state: RunnerState, now=None) -> Decision:
        now = now or dt.datetime.now(dt.timezone.utc)
        c = self.cfg
        d = Decision(symbol=cv.symbol, action="skip", reason="", entry=cv.price,
                     flat_by_close=c.flat_by_close)

        if state.daily_pnl <= -c.daily_loss_limit_frac * state.day_start_equity:
            d.reason = "daily loss limit hit — done for the day"; return d
        if state.locked_until and now.isoformat() < state.locked_until:
            d.reason = f"cooldown/lockout until {state.locked_until[:16]}"; return d
        if state.trades_today >= c.max_trades_per_day:
            d.reason = "max trades/day reached"; return d
        if state.open_positions >= c.max_concurrent:
            d.reason = "max concurrent positions"; return d
        if c.require_green_light and not cv.green_light:
            d.reason = f"not green-light ({cv.blowup_flags or 'watch'})"; return d
        if cv.blowup_flags:
            d.reason = f"blow-up flag veto [{cv.blowup_flags}]"; return d
        sc = self._score(cv)                    # None until the classifier is trained
        if sc is not None and (sc["p_monster"] < c.min_p_monster or sc["p_loss"] > c.max_p_loss):
            d.reason = f"classifier veto (pm {sc['p_monster']:.2f} / pl {sc['p_loss']:.2f})"; return d

        stop = self._stop(cv)
        if not stop or stop >= cv.price:
            d.reason = "no valid stop"; return d
        risk_frac = min(c.base_risk_frac if cv.green_light else c.watch_risk_frac,
                        c.effective_max_risk())
        if risk_frac <= 0:
            d.reason = "risk fraction zero"; return d
        per_share = cv.price - stop
        shares = int((state.equity * risk_frac) / per_share)
        capped_by = "risk"
        # buying-power cap: notional can't exceed equity x leverage (small cash acct)
        bp_sh = int(state.equity * c.max_leverage / cv.price)
        if bp_sh < shares:
            shares, capped_by = bp_sh, "buying_power"
        # liquidity cap: don't hold more than max_pct_of_dollar_vol of today's $ volume
        if cv.vol_today:
            liq_sh = int(c.max_pct_of_dollar_vol * cv.vol_today)
            if liq_sh < shares:
                shares, capped_by = liq_sh, "liquidity"
        if shares < 1:
            d.reason = "size < 1 share (stop too wide / BP / liquidity cap)"; return d

        d.action, d.reason = "take", f"green-light push (capped by {capped_by})"
        d.shares, d.stop = shares, stop
        d.risk_dollars = round(shares * per_share, 2)
        d.risk_frac = round(d.risk_dollars / state.equity, 4)
        d.notional = round(shares * cv.price, 2)
        return d
