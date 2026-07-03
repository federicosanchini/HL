from __future__ import annotations

from typing import List

from src import OrderCommand, OrderType


class Trader:
    """Equally weighted buy-and-hold portfolio benchmark.

    At bar 0, converts total equity into an equally weighted long position
    across all tradeable assets. No rebalancing, no exits.
    """

    def __init__(
        self,
        *,
        leverage: float = 1.0,
        min_notional_usd: float = 10.0,
        **_,
    ) -> None:
        self.name = "EqualWeight"
        self.leverage = float(leverage)
        self.min_notional_usd = float(min_notional_usd)
        self._entered = False

    def run(self, state) -> List[OrderCommand]:
        if self._entered:
            return []

        assets = sorted(state.market.keys())
        if not assets:
            return []

        equity = float(state.account.equity)
        notional_per_asset = equity / len(assets)

        if notional_per_asset < self.min_notional_usd:
            return []

        self._entered = True
        return [
            OrderCommand(
                asset=asset,
                side=1,
                order_type=OrderType.MARKET.value,
                notional=notional_per_asset,
                leverage=self.leverage,
            )
            for asset in assets
        ]
