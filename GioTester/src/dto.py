from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import pandas as pd

from .position import Position


@dataclass(frozen=True)
class MarketAssetView:
    asset: str
    trade_px: float
    mark_px: float
    oracle_px: float
    funding_rate: float
    high_px: float
    low_px: float


@dataclass(frozen=True)
class PositionView:
    asset: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    notional_at_mark: float
    signed_invested_notional: float
    cumulative_funding: float
    cumulative_fees: float

    @classmethod
    def from_position(cls, pos: Position) -> "PositionView":
        return cls(
            asset=pos.asset,
            size=pos.size,
            entry_price=pos.entry_price,
            mark_price=pos.mark_price,
            unrealized_pnl=pos.unrealized_pnl,
            notional_at_mark=pos.notional_at_mark,
            signed_invested_notional=pos.signed_invested_notional,
            cumulative_funding=pos.cumulative_funding,
            cumulative_fees=pos.cumulative_fees,
        )


@dataclass(frozen=True)
class AccountView:
    cash: float
    equity: float
    initial_margin_required: float
    maintenance_margin_required: float
    available_balance: float


@dataclass(frozen=True)
class StrategyState:
    timestamp: pd.Timestamp
    bar_index: int
    total_bars: int
    market: Mapping[str, MarketAssetView]
    positions: Mapping[str, PositionView]
    account: AccountView
    current_ranks_row: Optional[Mapping[str, Tuple[float, float]]]
    is_release_bar: bool


def readonly_mapping(d):
    return MappingProxyType(dict(d))
