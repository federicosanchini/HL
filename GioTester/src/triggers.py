"""Trigger (stop/tp) order placement and lifecycle — Phase E engine v2.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R2
(trigger orders: placement, replace, auto-cancel). Fire/fill (R3/R4/R5) land
in a later task.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .position import OrderCommand, OrderRejectedEvent
from .state import MarketState, RestingTrigger, StateBucket


def _reject(order: OrderCommand, ms: MarketState, reason: str) -> OrderRejectedEvent:
    return OrderRejectedEvent(
        timestamp=ms.timestamp.isoformat(),
        asset=order.asset,
        side=order.side,
        order_type=order.normalized_type(),
        reason=reason,
    )


def place_triggers(
    trigger_orders: Iterable[OrderCommand],
    ms: MarketState,
    bucket: StateBucket,
    min_notional_usd: float,
) -> None:
    """Place/replace resting triggers per R2.

    Orders are grouped by asset in emission order. For each asset with >=1
    emitted trigger (valid or not), the asset's resting set is cleared first;
    each trigger is then validated individually in emission order — invalid
    ones are rejected loudly (`OrderRejectedEvent`), valid ones rest. Assets
    absent from `trigger_orders` are untouched (market orders never reach
    this function, so they cannot disturb resting triggers).
    """
    by_asset: Dict[str, List[OrderCommand]] = {}
    for order in trigger_orders:
        by_asset.setdefault(order.asset, []).append(order)

    for asset, orders in by_asset.items():
        bucket.resting_triggers[asset] = []
        for order in orders:
            try:
                order.validate_trigger()
            except ValueError as exc:
                bucket.rejected_orders.append(_reject(order, ms, str(exc)))
                continue

            existing = bucket.get_position(asset)
            if existing is None or existing.size == 0:
                bucket.rejected_orders.append(
                    _reject(order, ms, "trigger requires an open opposing position")
                )
                continue
            if existing.side == order.side:
                bucket.rejected_orders.append(
                    _reject(order, ms, "trigger requires an open opposing position")
                )
                continue

            try:
                close_px = ms.mark_px(asset)
            except ValueError as exc:
                bucket.rejected_orders.append(_reject(order, ms, str(exc)))
                continue

            notional = abs(order.size) * close_px
            if notional < min_notional_usd:
                bucket.rejected_orders.append(
                    _reject(order, ms, "trigger below minimum notional at placement")
                )
                continue

            bucket.resting_triggers[asset].append(
                RestingTrigger(order=order, placed_bar=ms.bar_index)
            )


def auto_cancel_triggers(bucket: StateBucket, asset: str) -> None:
    """Drop all resting triggers for `asset` (position reached flat or flipped)."""
    bucket.resting_triggers.pop(asset, None)
