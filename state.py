"""Persistent state for active positions, history, and trust metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import STATE_FILE


@dataclass
class ScaleOutEvent:
    ticker: str
    date: str
    shares_sold: float
    price: float
    level: str
    pnl: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivePosition:
    ticker: str
    cap_category: str
    entry_price: float
    stop_loss: float
    shares: float
    shares_remaining: float
    entry_date: str
    confidence_score: int
    resistance_level: float
    max_risk: float
    high_water_mark: float
    trailing_stop: float
    composite_score: int = 0
    scale_outs_hit: list[str] = field(default_factory=list)
    status: str = "OPEN"

    @property
    def risk_per_share(self) -> float:
        return self.entry_price - self.stop_loss

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedTrade:
    ticker: str
    entry_price: float
    exit_price: float
    shares: float
    entry_date: str
    exit_date: str
    exit_reason: str
    pnl: float
    r_multiple: float
    confidence_score: int
    setup_fidelity: float
    execution_aligned: bool
    user_feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    trust_score: float = 0.0
    active_positions: list[ActivePosition] = field(default_factory=list)
    trade_history: list[ClosedTrade] = field(default_factory=list)
    scale_out_log: list[ScaleOutEvent] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust_score": self.trust_score,
            "active_positions": [p.to_dict() for p in self.active_positions],
            "trade_history": [t.to_dict() for t in self.trade_history],
            "scale_out_log": [s.to_dict() for s in self.scale_out_log],
            "total_trades": self.total_trades,
            "wins": self.wins,
            "total_pnl": self.total_pnl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        positions = [ActivePosition(**p) for p in data.get("active_positions", [])]
        history = [ClosedTrade(**t) for t in data.get("trade_history", [])]
        scale_log = [ScaleOutEvent(**s) for s in data.get("scale_out_log", [])]
        return cls(
            trust_score=data.get("trust_score", 0.0),
            active_positions=positions,
            trade_history=history,
            scale_out_log=scale_log,
            total_trades=data.get("total_trades", 0),
            wins=data.get("wins", 0),
            total_pnl=data.get("total_pnl", 0.0),
        )


class StateManager:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or STATE_FILE)
        self.state = self._load()

    def _load(self) -> AgentState:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return AgentState.from_dict(json.load(f))
        return AgentState()

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def has_open_position(self, ticker: str) -> bool:
        return any(
            p.ticker == ticker.upper() and p.status == "OPEN"
            for p in self.state.active_positions
        )

    def open_position(self, position: ActivePosition) -> None:
        if self.has_open_position(position.ticker):
            raise ValueError(f"Already tracking open position for {position.ticker}")
        self.state.active_positions.append(position)
        self.save()

    def close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str,
        execution_aligned: bool = True,
        setup_fidelity: float = 1.0,
        user_feedback: str = "",
    ) -> ClosedTrade | None:
        ticker = ticker.upper()
        for i, pos in enumerate(self.state.active_positions):
            if pos.ticker == ticker and pos.status == "OPEN":
                pnl = (exit_price - pos.entry_price) * pos.shares_remaining
                r_mult = (
                    (exit_price - pos.entry_price) / pos.risk_per_share
                    if pos.risk_per_share > 0
                    else 0.0
                )
                closed = ClosedTrade(
                    ticker=ticker,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    shares=pos.shares_remaining,
                    entry_date=pos.entry_date,
                    exit_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    exit_reason=exit_reason,
                    pnl=round(pnl, 2),
                    r_multiple=round(r_mult, 2),
                    confidence_score=pos.confidence_score,
                    setup_fidelity=setup_fidelity,
                    execution_aligned=execution_aligned,
                    user_feedback=user_feedback,
                )
                self.state.active_positions.pop(i)
                self.state.trade_history.append(closed)
                self.state.total_trades += 1
                self.state.total_pnl = round(self.state.total_pnl + closed.pnl, 2)
                if closed.pnl > 0:
                    self.state.wins += 1
                self._update_trust_score()
                self.save()
                return closed
        return None

    def partial_scale_out(
        self, ticker: str, shares_sold: float, price: float, level_label: str
    ) -> ScaleOutEvent | None:
        ticker = ticker.upper()
        for pos in self.state.active_positions:
            if pos.ticker == ticker and pos.status == "OPEN":
                pnl = (price - pos.entry_price) * shares_sold
                event = ScaleOutEvent(
                    ticker=ticker,
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    shares_sold=shares_sold,
                    price=price,
                    level=level_label,
                    pnl=round(pnl, 2),
                )
                self.state.scale_out_log.append(event)
                self.state.total_pnl = round(self.state.total_pnl + event.pnl, 2)
                pos.shares_remaining = round(pos.shares_remaining - shares_sold, 4)
                pos.scale_outs_hit.append(level_label)
                if pos.shares_remaining <= 0:
                    pos.status = "CLOSED"
                    self.state.active_positions = [
                        p for p in self.state.active_positions
                        if not (p.ticker == ticker and p.status == "CLOSED")
                    ]
                self.save()
                return event
        return None

    def update_position_stops(
        self, ticker: str, high_water_mark: float, trailing_stop: float
    ) -> None:
        ticker = ticker.upper()
        for pos in self.state.active_positions:
            if pos.ticker == ticker and pos.status == "OPEN":
                pos.high_water_mark = high_water_mark
                pos.trailing_stop = trailing_stop
                self.save()
                return

    def add_feedback_to_last_trade(
        self, ticker: str, fidelity: float, execution_aligned: bool, notes: str = ""
    ) -> bool:
        ticker = ticker.upper()
        for trade in reversed(self.state.trade_history):
            if trade.ticker == ticker:
                trade.setup_fidelity = fidelity
                trade.execution_aligned = execution_aligned
                trade.user_feedback = notes
                self._update_trust_score()
                self.save()
                return True
        return False

    def get_open_positions(self) -> list[ActivePosition]:
        return [p for p in self.state.active_positions if p.status == "OPEN"]

    def get_position(self, ticker: str) -> ActivePosition | None:
        ticker = ticker.upper()
        for pos in self.state.active_positions:
            if pos.ticker == ticker and pos.status == "OPEN":
                return pos
        return None

    def _update_trust_score(self) -> None:
        if not self.state.trade_history:
            return

        win_rate = self.state.wins / self.state.total_trades if self.state.total_trades else 0
        avg_r = sum(t.r_multiple for t in self.state.trade_history) / len(self.state.trade_history)
        avg_fidelity = sum(t.setup_fidelity for t in self.state.trade_history) / len(
            self.state.trade_history
        )
        aligned_pct = sum(1 for t in self.state.trade_history if t.execution_aligned) / len(
            self.state.trade_history
        )

        fidelity_pts = avg_fidelity * 30
        execution_pts = aligned_pct * 30
        expectancy_pts = min(40, max(0, (win_rate * 20) + (avg_r * 10)))

        self.state.trust_score = round(
            min(100, fidelity_pts + execution_pts + expectancy_pts), 1
        )