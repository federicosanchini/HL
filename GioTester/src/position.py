from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class CloseReason(Enum):
    EXPIRY = "expiry"
    LIQUIDATION = "liquidation"
    STOPLOSS = "stoploss"
    FORCE = "force"
    NET = "net"
    TRIGGER = "trigger"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    GTC = "gtc"
    IOC = "ioc"
    ALO = "alo"
    TRIGGER = "trigger"
    TWAP = "twap"
    SCALE = "scale"


SUPPORTED_ORDER_TYPES = {OrderType.MARKET.value}
SUPPORTED_MARGIN_MODES = {"cross", "isolated"}
TRIGGER_DIRECTIONS = {"stop", "tp"}


def normalize_margin_mode(value: object) -> str:
    mode = str(value or "cross").lower()
    if mode not in SUPPORTED_MARGIN_MODES:
        allowed = ", ".join(sorted(SUPPORTED_MARGIN_MODES))
        raise ValueError(f"margin_mode must be one of: {allowed}")
    return mode


@dataclass(frozen=True)
class OrderCommand:
    """Strategy command accepted by the simulator.

    The first implementation admits only market-like commands because hourly
    OHLC data cannot faithfully represent Hyperliquid's resting order book.
    Unsupported order types are rejected explicitly by execution code.
    """

    asset: str
    side: int  # +1 buy, -1 sell
    order_type: str = OrderType.MARKET.value
    notional: Optional[float] = None
    size: Optional[float] = None
    leverage: float = 1.0
    reduce_only: bool = False
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    client_id: Optional[str] = None
    trigger_px: Optional[float] = None
    trigger_direction: Optional[str] = None

    @property
    def perp(self) -> str:
        return self.asset

    def normalized_type(self) -> str:
        return str(self.order_type).lower()

    def validate_basic(self) -> None:
        if not self.asset:
            raise ValueError("order asset is required")
        if self.side not in (-1, 1):
            raise ValueError("order side must be +1 buy or -1 sell")
        if self.notional is None and self.size is None:
            raise ValueError("order requires either notional or size")
        if self.notional is not None and self.notional <= 0:
            raise ValueError("order notional must be positive")
        if self.size is not None and self.size <= 0:
            raise ValueError("order size must be positive")
        if self.leverage <= 0:
            raise ValueError("order leverage must be positive")
        if self.spread_bps < 0 or self.slippage_bps < 0:
            raise ValueError("spread_bps and slippage_bps must be non-negative")

    def validate_trigger(self) -> None:
        """Validate a TRIGGER order per spec R2.

        Side-opposes-position is checked at placement time (Task 3), not here.
        """
        if self.normalized_type() != OrderType.TRIGGER.value:
            raise ValueError("order_type must be trigger")
        if not self.reduce_only:
            raise ValueError("trigger order must be reduce_only")
        if self.notional is not None:
            raise ValueError("trigger order requires explicit size, not notional")
        if self.size is None:
            raise ValueError("trigger order requires explicit size")
        if self.size <= 0:
            raise ValueError("trigger order size must be positive")
        if self.trigger_px is None or not math.isfinite(self.trigger_px) or self.trigger_px <= 0:
            raise ValueError("trigger_px must be a finite positive price")
        if self.trigger_direction not in TRIGGER_DIRECTIONS:
            raise ValueError('trigger_direction must be "stop" or "tp"')


@dataclass
class Position:
    """One signed Hyperliquid perp position."""

    asset: str
    size: float  # signed coin size; positive long, negative short
    entry_price: float
    entry_time: pd.Timestamp
    leverage: float
    mm_rate: float
    # Active only in isolated margin mode. Cross-margin logic must not read this.
    isolated_margin: float = 0.0
    cumulative_funding: float = 0.0
    cumulative_fees: float = 0.0
    realized_pnl: float = 0.0
    mark_price: float = 0.0

    @property
    def perp(self) -> str:
        return self.asset

    @property
    def side(self) -> int:
        if self.size > 0:
            return 1
        if self.size < 0:
            return -1
        return 0

    @property
    def qty(self) -> float:
        return abs(self.size)

    @property
    def notional_at_mark(self) -> float:
        px = self.mark_price if self.mark_price > 0 else self.entry_price
        return abs(self.size) * px

    @property
    def signed_invested_notional(self) -> float:
        """Signed USD cost basis. Positive long, negative short."""
        return self.size * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.mark_price <= 0:
            return 0.0
        return self.size * (self.mark_price - self.entry_price)

    @property
    def isolated_equity(self) -> float:
        """Equity constrained to this isolated position."""
        return (
            self.isolated_margin
            + self.unrealized_pnl
            + self.cumulative_funding
            - self.cumulative_fees
        )

    @property
    def maintenance_margin(self) -> float:
        return max(self.notional_at_mark * self.mm_rate, 0.0)

    def mark(self, px: float) -> None:
        if not (math.isfinite(px) and px > 0):
            raise ValueError(f"invalid mark price for {self.asset}: {px}")
        self.mark_price = px

    def apply_funding(self, rate: float, oracle_px: float) -> float:
        """Apply hourly funding using Hyperliquid's oracle notional convention."""
        if not (math.isfinite(rate) and math.isfinite(oracle_px) and oracle_px > 0):
            raise ValueError(f"invalid funding inputs for {self.asset}")
        delta = -self.size * oracle_px * rate
        self.cumulative_funding += delta
        return delta


@dataclass(frozen=True)
class ExecutionEvent:
    timestamp: str
    event_type: str
    asset: str
    side: int
    notional: float
    fill_price: float
    fee: float
    realized_pnl: float = 0.0
    reason: Optional[str] = None


@dataclass(frozen=True)
class OrderRejectedEvent:
    timestamp: str
    asset: str
    side: int
    order_type: str
    reason: str


@dataclass(frozen=True)
class FundingEvent:
    timestamp: str
    asset: str
    funding_rate: float
    oracle_px: float
    payment: float


@dataclass(frozen=True)
class LiquidationEvent:
    timestamp: str
    asset: str
    account_equity: float
    maintenance_margin: float
    fill_price: float
    realized_pnl: float
    fee: float
