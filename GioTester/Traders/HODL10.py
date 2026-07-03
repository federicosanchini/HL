from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Single-file HODL10 strategy loaded from GioTester/traders."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        blackout_days_end: int = 50,
        bars_per_day: int = 24,
        **_,
    ) -> None:
        self.name = "HODL10"
        self.expiry_days = 10
        self.signal_horizon_days = 10
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = self.expiry_days * self.bars_per_day
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.blackout_bars_end = int(blackout_days_end) * self.bars_per_day
        self._entry_bar_by_asset: Dict[str, int] = {}

    def _entry_gate(self, state) -> bool:
        return (state.total_bars - state.bar_index) >= self.blackout_bars_end

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _expiry_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live_assets = set(state.positions)
        for asset in list(self._entry_bar_by_asset):
            if asset not in live_assets:
                self._entry_bar_by_asset.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar_by_asset.setdefault(asset, state.bar_index)
            if state.bar_index - entry_bar < self.expiry_bars:
                continue
            if abs(pos.size) * pos.mark_price < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._expiry_orders(state)
        if not state.is_release_bar or not state.current_ranks_row:
            return orders
        if not self._entry_gate(state):
            return orders

        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return orders

        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                self._entry_bar_by_asset[asset] = state.bar_index
                orders.append(
                    OrderCommand(
                        asset=asset,
                        side=1,
                        order_type=OrderType.MARKET.value,
                        notional=self.notional_long,
                        leverage=self.leverage,
                    )
                )
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                self._entry_bar_by_asset[asset] = state.bar_index
                orders.append(
                    OrderCommand(
                        asset=asset,
                        side=-1,
                        order_type=OrderType.MARKET.value,
                        notional=self.notional_short,
                        leverage=self.leverage,
                    )
                )
        return orders
