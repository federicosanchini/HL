from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Position:
    id: int
    perp: str
    side: int  # +1 long, -1 short
    qty: float  # absolute coin size, > 0
    entry_price: float
    entry_time: pd.Timestamp
    expiry_bars: int  # bars until expiry close (HODL)
    leverage: float
    initial_margin: float  # isolated bucket; absorbs funding
    notional: float  # qty * entry_price
    cumulative_funding: float = 0.0  # signed: + received, - paid
    cumulative_fees: float = 0.0  # always >= 0
    realized_pnl: float = 0.0  # set on close; excludes fees
    mark_price: float = 0.0
    bar_age: int = 0  # full bars elapsed since entry
    closed: bool = False
    close_reason: Optional[str] = (
        None  # "expiry" | "liquidation" | "stoploss" | "force"
    )
    close_price: Optional[float] = (
        None  # forced exit price (e.g. liq); None = use bar open
    )

    def update_funding(self, rate: float, oracle_px: float) -> None:
        """Apply hourly funding payment to this position.

        HL convention: longs pay positive funding, shorts receive it.
        delta = -side * qty * oracle_px * rate  (signed cashflow into margin)
        """
        if not (math.isfinite(rate) and math.isfinite(oracle_px)):
            return
        delta = -self.side * self.qty * oracle_px * rate
        self.cumulative_funding += delta
        self.initial_margin += delta

    def mark(self, px: float) -> None:
        """Update mark_price; advance bar_age."""
        if math.isfinite(px) and px > 0:
            self.mark_price = px
        self.bar_age += 1

    @property
    def unrealized_pnl(self) -> float:
        if self.mark_price <= 0:
            return 0.0
        return self.side * self.qty * (self.mark_price - self.entry_price)

    @property
    def notional_at_mark(self) -> float:
        return self.qty * (self.mark_price if self.mark_price > 0 else self.entry_price)

    def liquidation_price(self) -> float:
        """Isolated-margin liq price.

        equity_at_p = initial_margin + side*qty*(p - entry)
        mm_required(p) = mm_frac * qty * p
        liq when equity == mm_required:
            initial_margin + side*qty*(p - entry) = mm_frac*qty*p
            p*(side*qty - mm_frac*qty) = side*qty*entry - initial_margin
            p = (side*qty*entry - initial_margin) / (qty*(side - mm_frac))
        """
        if self.qty <= 0 or self.leverage <= 0:
            return float("nan")
        mm_frac = 1.0 / (2.0 * self.leverage)
        denom = self.qty * (self.side - mm_frac)
        if abs(denom) < 1e-12:
            return float("nan")
        num = self.side * self.qty * self.entry_price - self.initial_margin
        liq = num / denom
        return liq if liq > 0 else float("nan")


@dataclass
class NewOrder:
    """Strategy → simulator: open this position at next available fill."""

    perp: str
    side: int
    notional: float
    leverage: float
    expiry_bars: int


# Legacy stub kept for back-compat; real strategies live in strategies.py.
class HODL(Position):
    """Deprecated stub. See strategies.HODL_10 / strategies.HODL_30."""

    pass
