from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd


class CloseReason(Enum):
    EXPIRY = "expiry"
    LIQUIDATION = "liquidation"
    STOPLOSS = "stoploss"
    FORCE = "force"
    NET = "net"


class ExpiryMode(Enum):
    RESET_LATEST = "reset_latest"
    KEEP_EARLIEST = "keep_earliest"
    PROPORTIONAL = "proportional"


@dataclass
class Position:
    id: int
    perp: str
    side: int          # +1 long, -1 short
    qty: float         # absolute coin size, > 0
    entry_price: float
    entry_time: pd.Timestamp
    expiry_bars: int   # total bars from open to expiry (reference only)
    abs_expiry_bar: int  # absolute bar index at expiry; updated on merge per expiry_mode
    leverage: float
    initial_margin: float   # isolated bucket; absorbs funding and partial-close reductions
    notional: float         # qty * entry_price; updated on merge/partial-close
    expiry_mode: ExpiryMode = ExpiryMode.RESET_LATEST
    tranches: List[Tuple[float, int]] = field(default_factory=list)  # [(qty, abs_expiry_bar)] proportional only
    mm_rate: float = 0.05              # maintenance margin rate = 1/(2*max_asset_leverage)
    cumulative_funding: float = 0.0    # signed: + received, - paid
    cumulative_fees: float = 0.0       # always >= 0; includes open fee + all close fees
    realized_pnl: float = 0.0          # accumulates via += in both partial and full closes
    mark_price: float = 0.0
    bar_age: int = 0                   # full bars elapsed since open (or last reset on merge)
    closed: bool = False
    close_reason: Optional[CloseReason] = None
    close_price: Optional[float] = None  # override fill price (liq); None = bar open
    close_qty: Optional[float] = None    # None = full close; set for proportional partial closes

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
        """Isolated-margin liq price via HL exact formula.

        Derived by solving: (initial_margin + unrealized_pnl) = qty * liq_px * mm_rate
        Closed form (mark-independent):
            liq_px = (entry_px * qty - initial_margin) / (qty * (1 - mm_rate))  [long]
            liq_px = (entry_px * qty + initial_margin) / (qty * (1 + mm_rate))  [short]
        Equivalent to: price - side * margin_available / qty / (1 - mm_rate * side)
        where margin_available = (initial_margin + unrealized_pnl) - qty * price * mm_rate.
        """
        if self.qty <= 0 or self.mark_price <= 0:
            return float("nan")
        l = self.mm_rate
        price = self.mark_price
        equity = self.initial_margin + self.unrealized_pnl
        maintenance_required = self.qty * price * l
        margin_available = equity - maintenance_required
        denom = 1.0 - l * self.side
        if abs(denom) < 1e-12:
            return float("nan")
        liq = price - self.side * margin_available / self.qty / denom
        return liq if liq > 0 else float("nan")


@dataclass
class NewOrder:
    """Strategy → simulator: open this position at next available fill."""

    perp: str
    side: int
    notional: float
    leverage: float
    expiry_bars: int
    limit_price: Optional[float] = None  # None = market; set = limit (checked vs bar high/low)
