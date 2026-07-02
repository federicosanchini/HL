"""Mutable bar-level simulation state shared by engine internals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dto import AccountView, MarketAssetView, PositionView, StrategyState, readonly_mapping
from .position import (
    ExecutionEvent,
    FundingEvent,
    LiquidationEvent,
    OrderRejectedEvent,
    Position,
    normalize_margin_mode,
)


class _PerpRowView:
    """Lazy mapping[perp -> value] over a dict-of-ndarray, masked by finiteness."""

    __slots__ = ("_d", "_i", "_finite_idx")

    def __init__(
        self, d: Dict[str, np.ndarray], i: int, finite_idx: Optional[int] = None
    ) -> None:
        self._d = d
        self._i = i
        self._finite_idx = finite_idx

    def get(self, perp: str, default=None):
        a = self._d.get(perp)
        if a is None:
            return default
        v = a[self._i]
        if self._finite_idx is None:
            return float(v) if math.isfinite(v) else default
        return v if math.isfinite(v[self._finite_idx]) else default

    def __contains__(self, perp: str) -> bool:
        return self.get(perp) is not None

    def items(self) -> Iterator[Tuple[str, object]]:
        i, fi = self._i, self._finite_idx
        for p, a in self._d.items():
            v = a[i]
            if fi is None:
                if math.isfinite(v):
                    yield p, float(v)
            else:
                if math.isfinite(v[fi]):
                    yield p, v


@dataclass
class MarketState:
    """Single hourly bar snapshot.

    trade_px is OHLC open, mark_px is OHLC close proxy, and oracle_px is the
    required Hyperliquid oracle input used for funding.
    """

    timestamp: pd.Timestamp
    bar_index: int
    total_bars: int
    ohlc_row: _PerpRowView
    funding_row: _PerpRowView
    oracle: _PerpRowView
    current_ranks_row: Optional[Dict[str, Tuple[float, float]]]
    mm_rate_cache: Dict[str, float] = field(default_factory=dict)
    is_release_bar: bool = False

    def trade_px(self, asset: str) -> float:
        row = self.ohlc_row.get(asset)
        if row is None or not math.isfinite(row[0]) or row[0] <= 0:
            raise ValueError(f"missing trade price for {asset} at {self.timestamp}")
        return float(row[0])

    def mark_px(self, asset: str) -> float:
        row = self.ohlc_row.get(asset)
        if row is None:
            raise ValueError(f"missing mark price for {asset} at {self.timestamp}")
        close_px = row[3]
        if math.isfinite(close_px) and close_px > 0:
            return float(close_px)
        # Fallback for incomplete OHLC bars: reuse trade price proxy (open).
        open_px = row[0]
        if math.isfinite(open_px) and open_px > 0:
            return float(open_px)
        raise ValueError(f"missing mark price for {asset} at {self.timestamp}")

    def oracle_px(self, asset: str) -> float:
        px = self.oracle.get(asset)
        if px is None or not math.isfinite(px) or px <= 0:
            raise ValueError(f"missing oracle price for {asset} at {self.timestamp}")
        return float(px)

    def funding_rate(self, asset: str) -> float:
        rate = self.funding_row.get(asset)
        # Missing funding observations are treated as zero funding for that hour.
        if rate is None or not math.isfinite(rate):
            return 0.0
        return float(rate)

    def bar_range(self, asset: str) -> Tuple[float, float]:
        """Effective (high, low) for `asset`'s completed bar, with R3's NaN fallback.

        If high (resp. low) is non-finite, substitute max(open, close_fallback)
        (resp. min(open, close_fallback)), where close_fallback mirrors mark_px's
        open-fallback: close if finite and positive, else open.
        """
        row = self.ohlc_row.get(asset)
        if row is None:
            raise ValueError(f"missing bar range for {asset} at {self.timestamp}")
        open_px = float(row[0])
        high_px = float(row[1])
        low_px = float(row[2])
        close_px = float(row[3])
        close_fallback = close_px if math.isfinite(close_px) and close_px > 0 else open_px
        eff_high = high_px if math.isfinite(high_px) else max(open_px, close_fallback)
        eff_low = low_px if math.isfinite(low_px) else min(open_px, close_fallback)
        return eff_high, eff_low

    def asset_view(self, asset: str) -> MarketAssetView:
        high_px, low_px = self.bar_range(asset)
        return MarketAssetView(
            asset=asset,
            trade_px=self.trade_px(asset),
            mark_px=self.mark_px(asset),
            oracle_px=self.oracle_px(asset),
            funding_rate=self.funding_rate(asset),
            high_px=high_px,
            low_px=low_px,
        )


@dataclass
class StateBucket:
    """Mutable account state shared by engine internals."""

    market_state: MarketState
    cash: float = 0.0
    positions_by_asset: Dict[str, Position] = field(default_factory=dict)
    stats_data_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    execution_events: List[ExecutionEvent] = field(default_factory=list)
    rejected_orders: List[OrderRejectedEvent] = field(default_factory=list)
    funding_events: List[FundingEvent] = field(default_factory=list)
    liquidation_events: List[LiquidationEvent] = field(default_factory=list)
    margin_mode: str = "cross"

    def __post_init__(self) -> None:
        self.margin_mode = normalize_margin_mode(self.margin_mode)

    def get_position(self, asset: str) -> Optional[Position]:
        return self.positions_by_asset.get(asset)

    def account_equity(self) -> float:
        if self.margin_mode == "isolated":
            return self.cash + sum(
                max(pos.isolated_equity, 0.0) for pos in self.positions_by_asset.values()
            )
        return self.cash + sum(pos.unrealized_pnl for pos in self.positions_by_asset.values())

    def initial_margin_required(self) -> float:
        if self.margin_mode == "isolated":
            return sum(pos.isolated_margin for pos in self.positions_by_asset.values())
        total = 0.0
        for pos in self.positions_by_asset.values():
            if pos.leverage <= 0:
                raise RuntimeError(f"invalid leverage for {pos.asset}: {pos.leverage}")
            total += pos.notional_at_mark / pos.leverage
        return total

    def maintenance_margin_required(self) -> float:
        return sum(pos.maintenance_margin for pos in self.positions_by_asset.values())

    def available_balance(self) -> float:
        if self.margin_mode == "isolated":
            return self.cash
        return self.account_equity() - self.initial_margin_required()

    def account_view(self) -> AccountView:
        return AccountView(
            cash=self.cash,
            equity=self.account_equity(),
            initial_margin_required=self.initial_margin_required(),
            maintenance_margin_required=self.maintenance_margin_required(),
            available_balance=self.available_balance(),
        )

    def strategy_state(self) -> StrategyState:
        market = {}
        for asset in self.market_state.ohlc_row._d:
            try:
                market[asset] = self.market_state.asset_view(asset)
            except ValueError:
                continue
        positions = {
            asset: PositionView.from_position(pos)
            for asset, pos in self.positions_by_asset.items()
        }
        ranks = self.market_state.current_ranks_row
        return StrategyState(
            timestamp=self.market_state.timestamp,
            bar_index=self.market_state.bar_index,
            total_bars=self.market_state.total_bars,
            market=readonly_mapping(market),
            positions=readonly_mapping(positions),
            account=self.account_view(),
            current_ranks_row=readonly_mapping(ranks) if ranks is not None else None,
            is_release_bar=self.market_state.is_release_bar,
        )

    def perp_stats(self, asset: str) -> Dict[str, float]:
        d = self.stats_data_bucket.get(asset)
        if d is None:
            d = {
                "locked_realized": 0.0,
                "locked_realized_long": 0.0,
                "locked_realized_short": 0.0,
                "locked_liq_long": 0.0,
                "locked_liq_short": 0.0,
                "locked_funding": 0.0,
                "locked_fees": 0.0,
                "n_opened": 0,
                "n_closed": 0,
                "n_liquidated": 0,
            }
            self.stats_data_bucket[asset] = d
        return d
