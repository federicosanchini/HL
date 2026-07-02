"""Trigger (stop/tp) order placement and lifecycle — Phase E engine v2.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R2
(trigger orders: placement, replace, auto-cancel). Fire/fill (R3/R4/R5) land
in a later task.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from .execution import _apply_delta
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


def trigger_fill_px(
    order: OrderCommand,
    open_px: float,
    eff_high: float,
    eff_low: float,
) -> Optional[float]:
    """Pure R3 fire/fill matrix. Returns None if the trigger does not fire this bar.

    Rows (side, trigger_direction) -> fires iff / gap-through (fill at open) iff:
      sell, stop: low<=px  / open<=px
      sell, tp:   high>=px / open>=px
      buy,  stop: high>=px / open>=px
      buy,  tp:   low<=px  / open<=px
    Gap-through fills at open (worse for the trader); otherwise fills at trigger_px.
    """
    px = order.trigger_px
    direction = order.trigger_direction
    if order.side == -1 and direction == "stop":
        if eff_low <= px:
            return open_px if open_px <= px else px
        return None
    if order.side == -1 and direction == "tp":
        if eff_high >= px:
            return open_px if open_px >= px else px
        return None
    if order.side == 1 and direction == "stop":
        if eff_high >= px:
            return open_px if open_px >= px else px
        return None
    if order.side == 1 and direction == "tp":
        if eff_low <= px:
            return open_px if open_px <= px else px
        return None
    return None


def _selection_key(
    item: Tuple[float, RestingTrigger, int], closing_long: bool
) -> Tuple[float, bool, str, int]:
    """R4 tie key: (fill price worst-first, client_id is None, client_id or "", placement_index)."""
    fill_px, trig, placement_index = item
    # Worst-for-trader is lowest price closing a long, highest closing a short.
    # Negate for shorts so ascending sort still puts the worst price first.
    price_component = fill_px if closing_long else -fill_px
    client_id = trig.order.client_id
    return (price_component, client_id is None, client_id or "", placement_index)


def fill_triggers(
    bucket: StateBucket,
    ms: MarketState,
    fee_bps: float,
    suppress: FrozenSet[str],
) -> None:
    """Evaluate and fill resting triggers per R3 (fire/fill matrix) + R4 (worst-fill
    selection on same-bar multi-fire) + size clamp.

    Assets processed in sorted order; assets in `suppress` (R5 liquidation
    precedence) are skipped entirely — their triggers stay resting untouched.
    Triggers placed at the current bar (`placed_bar >= ms.bar_index`) are not
    yet active (R2) and are skipped for firing but remain resting.
    """
    for asset in sorted(bucket.resting_triggers):
        if asset in suppress:
            continue
        triggers = bucket.resting_triggers.get(asset, [])
        if not triggers:
            continue

        position = bucket.get_position(asset)
        if position is None or position.size == 0:
            auto_cancel_triggers(bucket, asset)
            continue

        open_px = ms.trade_px(asset)
        eff_high, eff_low = ms.bar_range(asset)

        fired: List[Tuple[float, RestingTrigger, int]] = []
        not_fired: List[RestingTrigger] = []
        for idx, trig in enumerate(triggers):
            if trig.placed_bar >= ms.bar_index:
                not_fired.append(trig)
                continue
            fill_px = trigger_fill_px(trig.order, open_px, eff_high, eff_low)
            if fill_px is None:
                not_fired.append(trig)
            else:
                fired.append((fill_px, trig, idx))

        if not fired:
            continue

        closing_long = position.side == 1
        fill_px, selected, _ = min(fired, key=lambda item: _selection_key(item, closing_long))

        # Non-fired triggers stay resting; both the selected and the other
        # fired-but-not-selected triggers are removed (R4: "cancel the others").
        bucket.resting_triggers[asset] = not_fired

        clamped_size = min(selected.order.size, abs(position.size))
        synthetic_order = dataclasses.replace(selected.order, size=clamped_size)
        delta = synthetic_order.side * clamped_size
        notional = abs(delta) * fill_px
        fee = notional * fee_bps / 1e4

        _apply_delta(
            position,
            synthetic_order,
            delta,
            fill_px,
            fill_px,
            fee,
            bucket,
            ms,
            reason="trigger",
        )

        if asset not in bucket.positions_by_asset:
            auto_cancel_triggers(bucket, asset)
