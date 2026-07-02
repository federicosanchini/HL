"""Market execution for hourly OHLC replay."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import pandas as pd

from .data_loader import max_leverage_for
from .position import (
    CloseReason,
    ExecutionEvent,
    FundingEvent,
    LiquidationEvent,
    OrderCommand,
    OrderRejectedEvent,
    Position,
    SUPPORTED_ORDER_TYPES,
)
from .state import MarketState, StateBucket


EPS = 1e-10


def _reject(order: OrderCommand, ms: MarketState, reason: str) -> OrderRejectedEvent:
    return OrderRejectedEvent(
        timestamp=ms.timestamp.isoformat(),
        asset=order.asset,
        side=order.side,
        order_type=order.normalized_type(),
        reason=reason,
    )


def _fill_price(order: OrderCommand, trade_px: float) -> float:
    if not (math.isfinite(trade_px) and trade_px > 0):
        raise ValueError(f"invalid trade price: {trade_px}")
    half_spread = order.spread_bps / 20000.0
    slippage = order.slippage_bps / 10000.0
    adj = half_spread + slippage
    return trade_px * (1.0 + adj) if order.side == 1 else trade_px * (1.0 - adj)


def _order_qty(order: OrderCommand, fill_px: float) -> float:
    if order.size is not None:
        return float(order.size)
    assert order.notional is not None
    return float(order.notional) / fill_px


def _signed_delta(order: OrderCommand, fill_px: float) -> float:
    return order.side * _order_qty(order, fill_px)


def _hypothetical_position(
    existing: Optional[Position],
    order: OrderCommand,
    delta: float,
    fill_px: float,
    mark_px: float,
    mm_rate: float,
    ts: pd.Timestamp,
) -> Optional[Position]:
    if existing is None:
        return Position(
            asset=order.asset,
            size=delta,
            entry_price=fill_px,
            entry_time=ts,
            leverage=order.leverage,
            mm_rate=mm_rate,
            mark_price=mark_px,
        )

    new_size = existing.size + delta
    if abs(new_size) <= EPS:
        return None
    if existing.size * delta > 0:
        old_qty = abs(existing.size)
        add_qty = abs(delta)
        entry = (old_qty * existing.entry_price + add_qty * fill_px) / (old_qty + add_qty)
        leverage = order.leverage
    elif existing.size * new_size > 0:
        entry = existing.entry_price
        leverage = existing.leverage
    else:
        entry = fill_px
        leverage = order.leverage
    return Position(
        asset=existing.asset,
        size=new_size,
        entry_price=entry,
        entry_time=existing.entry_time if existing.size * new_size > 0 else ts,
        leverage=leverage,
        mm_rate=existing.mm_rate,
        mark_price=mark_px,
    )


def _order_applies_leverage(existing: Optional[Position], delta: float) -> bool:
    if existing is None:
        return True
    new_size = existing.size + delta
    if abs(new_size) <= EPS:
        return False
    return existing.size * delta > 0 or existing.size * new_size < 0


def _validate_leverage(order: OrderCommand, existing: Optional[Position], delta: float) -> Optional[str]:
    if not _order_applies_leverage(existing, delta):
        return None
    max_leverage = max_leverage_for(order.asset)
    if order.leverage > max_leverage + EPS:
        return f"leverage {order.leverage:g} exceeds max {max_leverage:g} for {order.asset}"
    return None


def _initial_margin_after(
    bucket: StateBucket,
    order: OrderCommand,
    hypothetical: Optional[Position],
) -> float:
    total = 0.0
    for asset, pos in bucket.positions_by_asset.items():
        if asset == order.asset:
            if hypothetical is not None:
                total += hypothetical.notional_at_mark / hypothetical.leverage
        else:
            total += pos.notional_at_mark / pos.leverage
    if order.asset not in bucket.positions_by_asset and hypothetical is not None:
        total += hypothetical.notional_at_mark / hypothetical.leverage
    return total


def _isolated_margin_to_add(
    existing: Optional[Position],
    order: OrderCommand,
    delta: float,
    fill_px: float,
) -> float:
    if existing is None:
        return abs(delta) * fill_px / order.leverage

    new_size = existing.size + delta
    if abs(new_size) <= EPS:
        return 0.0
    if existing.size * delta > 0:
        return abs(delta) * fill_px / order.leverage
    if existing.size * new_size < 0:
        return abs(new_size) * fill_px / order.leverage
    return 0.0


def _assert_reduce_only(order: OrderCommand, existing: Optional[Position], delta: float) -> Optional[str]:
    if not order.reduce_only:
        return None
    if existing is None or existing.size == 0:
        return "reduce_only order cannot open a position"
    if existing.size * delta >= 0:
        return "reduce_only order must oppose the current position"
    if abs(delta) > abs(existing.size) + EPS:
        return "reduce_only order cannot flip the position"
    return None


def execute_order(
    order: OrderCommand,
    ms: MarketState,
    bucket: StateBucket,
    fee_bps: float,
    min_notional_usd: float,
    mm_rate: float,
) -> None:
    """Execute or reject a single strategy command."""
    try:
        order.validate_basic()
    except ValueError as exc:
        bucket.rejected_orders.append(_reject(order, ms, str(exc)))
        return

    if order.normalized_type() not in SUPPORTED_ORDER_TYPES:
        bucket.rejected_orders.append(
            _reject(order, ms, "order type requires unsupported order-book semantics")
        )
        return

    try:
        trade_px = ms.trade_px(order.asset)
        mark_px = ms.mark_px(order.asset)
        fill_px = _fill_price(order, trade_px)
    except ValueError as exc:
        bucket.rejected_orders.append(_reject(order, ms, str(exc)))
        return

    if fill_px <= 0 or not math.isfinite(fill_px):
        bucket.rejected_orders.append(_reject(order, ms, "non-positive fill price"))
        return

    delta = _signed_delta(order, fill_px)
    notional = abs(delta) * fill_px
    if notional < min_notional_usd:
        bucket.rejected_orders.append(_reject(order, ms, "order below minimum notional"))
        return

    existing = bucket.get_position(order.asset)
    reduce_err = _assert_reduce_only(order, existing, delta)
    if reduce_err is not None:
        bucket.rejected_orders.append(_reject(order, ms, reduce_err))
        return
    leverage_err = _validate_leverage(order, existing, delta)
    if leverage_err is not None:
        bucket.rejected_orders.append(_reject(order, ms, leverage_err))
        return

    fee = notional * fee_bps / 1e4
    hypothetical = _hypothetical_position(existing, order, delta, fill_px, mark_px, mm_rate, ms.timestamp)
    if bucket.margin_mode == "cross":
        equity_after_fee = bucket.account_equity() - fee
        if equity_after_fee < _initial_margin_after(bucket, order, hypothetical) - EPS:
            bucket.rejected_orders.append(_reject(order, ms, "insufficient cross margin"))
            return
    else:
        margin_to_add = _isolated_margin_to_add(existing, order, delta, fill_px)
        if bucket.cash < margin_to_add - EPS:
            bucket.rejected_orders.append(_reject(order, ms, "insufficient isolated margin"))
            return

    if existing is None:
        _open_position(order, delta, fill_px, mark_px, fee, bucket, ms)
    else:
        _apply_delta(existing, order, delta, fill_px, mark_px, fee, bucket, ms)


def _open_position(
    order: OrderCommand,
    delta: float,
    fill_px: float,
    mark_px: float,
    fee: float,
    bucket: StateBucket,
    ms: MarketState,
) -> None:
    notional = abs(delta) * fill_px
    isolated_margin = notional / order.leverage if bucket.margin_mode == "isolated" else 0.0
    pos = Position(
        asset=order.asset,
        size=delta,
        entry_price=fill_px,
        entry_time=ms.timestamp,
        leverage=order.leverage,
        mm_rate=ms.mm_rate_cache.get(order.asset, 0.05),
        isolated_margin=isolated_margin,
        cumulative_fees=fee,
        mark_price=mark_px,
    )
    if bucket.margin_mode == "isolated":
        bucket.cash -= isolated_margin
    else:
        bucket.cash -= fee
    bucket.positions_by_asset[order.asset] = pos
    stats = bucket.perp_stats(order.asset)
    stats["locked_fees"] += fee
    stats["n_opened"] += 1
    bucket.execution_events.append(
        ExecutionEvent(
            timestamp=ms.timestamp.isoformat(),
            event_type="position_opened",
            asset=order.asset,
            side=order.side,
            notional=notional,
            fill_price=fill_px,
            fee=fee,
        )
    )


def _apply_delta(
    pos: Position,
    order: OrderCommand,
    delta: float,
    fill_px: float,
    mark_px: float,
    fee: float,
    bucket: StateBucket,
    ms: MarketState,
    reason: Optional[str] = None,
) -> None:
    if bucket.margin_mode == "isolated":
        _apply_delta_isolated(pos, order, delta, fill_px, mark_px, fee, bucket, ms, reason=reason)
        return

    old_size = pos.size
    old_side = pos.side
    event_type = "position_increased"
    realized = 0.0
    bucket.cash -= fee

    if old_size * delta > 0:
        old_qty = abs(old_size)
        add_qty = abs(delta)
        pos.entry_price = (old_qty * pos.entry_price + add_qty * fill_px) / (old_qty + add_qty)
        pos.size = old_size + delta
        pos.leverage = order.leverage
    else:
        close_qty = min(abs(delta), abs(old_size))
        realized = old_side * close_qty * (fill_px - pos.entry_price)
        bucket.cash += realized
        pos.realized_pnl += realized
        remaining = old_size + delta

        if abs(remaining) <= EPS:
            event_type = "position_closed"
            bucket.positions_by_asset.pop(pos.asset, None)
        elif old_size * remaining > 0:
            event_type = "position_reduced"
            pos.size = remaining
            pos.mark(mark_px)
        else:
            event_type = "position_flipped"
            pos.size = remaining
            pos.entry_price = fill_px
            pos.entry_time = ms.timestamp
            pos.cumulative_funding = 0.0
            pos.cumulative_fees = 0.0
            pos.leverage = order.leverage
            pos.mark(mark_px)

        stats = bucket.perp_stats(pos.asset)
        stats["locked_realized"] += realized
        if old_side == 1:
            stats["locked_realized_long"] += realized
        else:
            stats["locked_realized_short"] += realized
        if event_type == "position_closed":
            stats["n_closed"] += 1
        elif event_type == "position_flipped":
            stats["n_closed"] += 1
            stats["n_opened"] += 1

    if pos.asset in bucket.positions_by_asset:
        pos.cumulative_fees += fee
        pos.mark(mark_px)

    stats = bucket.perp_stats(order.asset)
    stats["locked_fees"] += fee
    bucket.execution_events.append(
        ExecutionEvent(
            timestamp=ms.timestamp.isoformat(),
            event_type=event_type,
            asset=order.asset,
            side=order.side,
            notional=abs(delta) * fill_px,
            fill_price=fill_px,
            fee=fee,
            realized_pnl=realized,
            reason=reason,
        )
    )


def _apply_delta_isolated(
    pos: Position,
    order: OrderCommand,
    delta: float,
    fill_px: float,
    mark_px: float,
    fee: float,
    bucket: StateBucket,
    ms: MarketState,
    reason: Optional[str] = None,
) -> None:
    old_size = pos.size
    old_side = pos.side
    event_type = "position_increased"
    realized = 0.0

    if old_size * delta > 0:
        old_qty = abs(old_size)
        add_qty = abs(delta)
        add_margin = add_qty * fill_px / order.leverage
        pos.entry_price = (old_qty * pos.entry_price + add_qty * fill_px) / (old_qty + add_qty)
        pos.size = old_size + delta
        pos.leverage = order.leverage
        pos.isolated_margin += add_margin
        pos.cumulative_fees += fee
        pos.mark(mark_px)
        bucket.cash -= add_margin
    else:
        close_qty = min(abs(delta), abs(old_size))
        close_ratio = close_qty / abs(old_size)
        close_fee = fee * close_qty / abs(delta)
        realized = old_side * close_qty * (fill_px - pos.entry_price)
        remaining = old_size + delta

        released_margin = pos.isolated_margin * close_ratio
        released_funding = pos.cumulative_funding * close_ratio
        released_prior_fees = pos.cumulative_fees * close_ratio
        bucket.cash += (
            released_margin
            + released_funding
            - released_prior_fees
            + realized
            - close_fee
        )
        pos.realized_pnl += realized

        if abs(remaining) <= EPS:
            event_type = "position_closed"
            bucket.positions_by_asset.pop(pos.asset, None)
        elif old_size * remaining > 0:
            event_type = "position_reduced"
            keep_ratio = 1.0 - close_ratio
            pos.size = remaining
            pos.isolated_margin *= keep_ratio
            pos.cumulative_funding *= keep_ratio
            pos.cumulative_fees *= keep_ratio
            pos.mark(mark_px)
        else:
            event_type = "position_flipped"
            open_qty = abs(remaining)
            open_fee = fee - close_fee
            open_margin = open_qty * fill_px / order.leverage
            bucket.cash -= open_margin
            pos.size = remaining
            pos.entry_price = fill_px
            pos.entry_time = ms.timestamp
            pos.leverage = order.leverage
            pos.isolated_margin = open_margin
            pos.cumulative_funding = 0.0
            pos.cumulative_fees = open_fee
            pos.realized_pnl = 0.0
            pos.mark(mark_px)

        stats = bucket.perp_stats(pos.asset)
        stats["locked_realized"] += realized
        if old_side == 1:
            stats["locked_realized_long"] += realized
        else:
            stats["locked_realized_short"] += realized
        if event_type == "position_closed":
            stats["n_closed"] += 1
        elif event_type == "position_flipped":
            stats["n_closed"] += 1
            stats["n_opened"] += 1

    stats = bucket.perp_stats(order.asset)
    stats["locked_fees"] += fee
    bucket.execution_events.append(
        ExecutionEvent(
            timestamp=ms.timestamp.isoformat(),
            event_type=event_type,
            asset=order.asset,
            side=order.side,
            notional=abs(delta) * fill_px,
            fill_price=fill_px,
            fee=fee,
            realized_pnl=realized,
            reason=reason,
        )
    )


def mark_positions(bucket: StateBucket, ms: MarketState) -> None:
    for pos in bucket.positions_by_asset.values():
        pos.mark(ms.mark_px(pos.asset))


def apply_funding(bucket: StateBucket, ms: MarketState) -> None:
    for pos in bucket.positions_by_asset.values():
        rate = ms.funding_rate(pos.asset)
        oracle_px = ms.oracle_px(pos.asset)
        payment = pos.apply_funding(rate, oracle_px)
        if bucket.margin_mode == "cross":
            bucket.cash += payment
        bucket.perp_stats(pos.asset)["locked_funding"] += payment
        bucket.funding_events.append(
            FundingEvent(
                timestamp=ms.timestamp.isoformat(),
                asset=pos.asset,
                funding_rate=rate,
                oracle_px=oracle_px,
                payment=payment,
            )
        )


def close_position(
    pos: Position,
    fill_px: float,
    fee_bps: float,
    bucket: StateBucket,
    ms: MarketState,
    reason: CloseReason,
) -> ExecutionEvent:
    side = pos.side
    qty = abs(pos.size)
    notional = qty * fill_px
    fee = notional * fee_bps / 1e4
    realized = side * qty * (fill_px - pos.entry_price)
    if bucket.margin_mode == "isolated":
        cash_delta = pos.isolated_margin + pos.cumulative_funding - pos.cumulative_fees + realized - fee
        if reason == CloseReason.LIQUIDATION:
            cash_delta = max(cash_delta, 0.0)
        bucket.cash += cash_delta
    else:
        bucket.cash += realized - fee

    stats = bucket.perp_stats(pos.asset)
    stats["locked_realized"] += realized
    stats["locked_fees"] += fee
    if reason == CloseReason.LIQUIDATION:
        stats["n_liquidated"] += 1
        if side == 1:
            stats["locked_liq_long"] += realized
        else:
            stats["locked_liq_short"] += realized
    else:
        stats["n_closed"] += 1
        if side == 1:
            stats["locked_realized_long"] += realized
        else:
            stats["locked_realized_short"] += realized

    bucket.positions_by_asset.pop(pos.asset, None)
    event = ExecutionEvent(
        timestamp=ms.timestamp.isoformat(),
        event_type="position_liquidated" if reason == CloseReason.LIQUIDATION else "position_closed",
        asset=pos.asset,
        side=-side,
        notional=notional,
        fill_price=fill_px,
        fee=fee,
        realized_pnl=realized,
        reason=reason.value,
    )
    bucket.execution_events.append(event)
    return event


def close_all(
    bucket: StateBucket,
    ms: MarketState,
    fee_bps: float,
    reason: CloseReason,
) -> List[ExecutionEvent]:
    events: List[ExecutionEvent] = []
    for pos in list(bucket.positions_by_asset.values()):
        events.append(close_position(pos, ms.trade_px(pos.asset), fee_bps, bucket, ms, reason))
    return events


def liquidate_if_needed(bucket: StateBucket, ms: MarketState, fee_bps: float) -> None:
    if bucket.margin_mode == "isolated":
        for asset in sorted(bucket.positions_by_asset):
            pos = bucket.positions_by_asset.get(asset)
            if pos is None:
                continue
            isolated_equity = pos.isolated_equity
            maintenance = pos.maintenance_margin
            if isolated_equity >= maintenance - EPS:
                continue
            fill_px = ms.mark_px(pos.asset)
            event = close_position(pos, fill_px, fee_bps, bucket, ms, CloseReason.LIQUIDATION)
            bucket.liquidation_events.append(
                LiquidationEvent(
                    timestamp=ms.timestamp.isoformat(),
                    asset=event.asset,
                    account_equity=isolated_equity,
                    maintenance_margin=maintenance,
                    fill_price=fill_px,
                    realized_pnl=event.realized_pnl,
                    fee=event.fee,
                )
            )
        return

    equity = bucket.account_equity()
    maintenance = bucket.maintenance_margin_required()
    if not bucket.positions_by_asset or equity >= maintenance - EPS:
        return

    for asset in sorted(bucket.positions_by_asset):
        pos = bucket.positions_by_asset.get(asset)
        if pos is None:
            continue
        fill_px = ms.mark_px(pos.asset)
        event = close_position(pos, fill_px, fee_bps, bucket, ms, CloseReason.LIQUIDATION)
        bucket.liquidation_events.append(
            LiquidationEvent(
                timestamp=ms.timestamp.isoformat(),
                asset=event.asset,
                account_equity=equity,
                maintenance_margin=maintenance,
                fill_price=fill_px,
                realized_pnl=event.realized_pnl,
                fee=event.fee,
            )
        )


def execute_orders(
    orders: Iterable[OrderCommand],
    ms: MarketState,
    bucket: StateBucket,
    fee_bps: float,
    min_notional_usd: float,
) -> None:
    for order in orders:
        execute_order(
            order=order,
            ms=ms,
            bucket=bucket,
            fee_bps=fee_bps,
            min_notional_usd=min_notional_usd,
            mm_rate=ms.mm_rate_cache.get(order.asset, 0.05),
        )
