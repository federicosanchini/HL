# src/genome/library.py
from __future__ import annotations

import math
from typing import Dict, List, Set

from src import OrderCommand, OrderType

from .registry import register


@register("universe_filter", "all_tradable")
class AllTradableFilter:
    """Every asset that has a market view this bar is eligible."""

    def eligible(self, state) -> Set[str]:
        return set(state.market.keys())


@register("signal", "rank")
class RankSignal:
    """Score = the prediction at `field_index` from the release-bar ranks row."""

    def __init__(self, *, field_index: int = 0) -> None:
        self.field_index = int(field_index)

    def score(self, state, universe: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not state.current_ranks_row:
            return out
        for asset, preds in state.current_ranks_row.items():
            if asset not in universe:
                continue
            sig = preds[self.field_index]
            if math.isfinite(sig):
                out[asset] = float(sig)
        return out


@register("entry_timing", "release_bar")
class ReleaseBarTiming:
    """Enter only on a signal-release bar."""

    def should_enter(self, state) -> bool:
        return bool(state.is_release_bar and state.current_ranks_row)


@register("sizing", "fixed_notional")
class FixedNotionalSizing:
    """Open a fixed USD notional per selected asset; longs first, then shorts."""

    def __init__(
        self,
        *,
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)

    def orders_for(self, state, longs: List[str], shorts: List[str]) -> List[OrderCommand]:
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset in shorts:
                _open(asset, -1, self.notional_short)
        return orders
