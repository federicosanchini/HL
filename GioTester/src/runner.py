"""Backtest entry-point: orchestrates deterministic hourly replay."""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Protocol

import numpy as np
from tqdm import tqdm

from .config import BacktestConfig
from .data_loader import MarketData, mm_rate_for
from .data_prep import SimData, prepare_sim_data
from .execution import (
    apply_funding,
    close_all,
    close_position,
    execute_order,
    liquidate_if_needed,
    mark_positions,
)
from .metrics import print_metrics_table, series_metrics
from .position import (
    CloseReason,
    OrderCommand,
    OrderRejectedEvent,
    OrderType,
    normalize_margin_mode,
)
from .result import SimResult
from .state import MarketState, PendingOrder, StateBucket, _PerpRowView
from .strategy_loader import load_trader
from .triggers import (
    execute_deferred_triggers,
    fill_triggers,
    place_triggers,
    precedence_scan,
    sweep_stale_triggers,
)


class Strategy(Protocol):
    name: str

    def run(self, state):
        ...


def _strategy_margin_mode(strategy: Strategy) -> str:
    return normalize_margin_mode(getattr(strategy, "margin_mode", "cross"))


def _market_state_for(sd: SimData, i: int, mm_cache: Dict[str, float]) -> MarketState:
    ts = sd.timeline[i]
    is_release = False
    ranks_row = None
    if ts.hour == 0:
        day = ts.normalize()
        ranks_row = sd.ranks_by_release.get(day)
        is_release = ranks_row is not None
    return MarketState(
        timestamp=ts,
        bar_index=i,
        total_bars=len(sd.timeline),
        ohlc_row=_PerpRowView(sd.ohlc, i, finite_idx=0),
        funding_row=_PerpRowView(sd.funding, i),
        oracle=_PerpRowView(sd.oracle, i),
        current_ranks_row=ranks_row,
        mm_rate_cache=mm_cache,
        is_release_bar=is_release,
    )


def _strategy_orders(strategy: Strategy, bucket: StateBucket):
    state = bucket.strategy_state()
    if not hasattr(strategy, "run"):
        raise TypeError("strategy must expose run(state)")
    out = strategy.run(state)
    if out is None:
        return []
    orders = list(out)
    for order in orders:
        if not isinstance(order, OrderCommand):
            raise TypeError("strategy.run(state) must return OrderCommand objects")
    return orders


def _has_priceable_ohlc(sd: SimData, asset: str, i: int) -> bool:
    row = sd.ohlc.get(asset)
    if row is None:
        return False
    open_px = row[i][0]
    return bool(math.isfinite(open_px) and open_px > 0)


def _force_close_positions_before_price_gap(
    bucket: StateBucket,
    sd: SimData,
    i: int,
    fee_bps: float,
) -> None:
    next_i = i + 1
    if next_i >= sd.n_bars:
        return

    ms = bucket.market_state
    for pos in list(bucket.positions_by_asset.values()):
        if _has_priceable_ohlc(sd, pos.asset, next_i):
            continue
        close_position(pos, ms.trade_px(pos.asset), fee_bps, bucket, ms, CloseReason.FORCE)


def _reject_pending_event(pending: PendingOrder, reason: str) -> OrderRejectedEvent:
    order = pending.order
    return OrderRejectedEvent(
        timestamp=pending.queued_ts.isoformat(),
        asset=order.asset,
        side=order.side,
        order_type=order.normalized_type(),
        reason=reason,
    )


def _fill_pending_orders(
    bucket: StateBucket, ms: MarketState, fee_bps: float, min_notional_usd: float
) -> None:
    """R1 fill pass (start of bar i+1): re-mark every open position at open[i+1],
    then fill queued market orders FIFO with the mark pinned to open[i+1] — never
    close. Rejections carry the order's original queued timestamp (anti-look-ahead).
    """
    for pos in bucket.positions_by_asset.values():
        pos.mark(ms.trade_px(pos.asset))
    for pending in bucket.pending_orders:
        order = pending.order
        open_px = ms.trade_px(order.asset)
        n_before = len(bucket.rejected_orders)
        execute_order(
            order=order,
            ms=ms,
            bucket=bucket,
            fee_bps=fee_bps,
            min_notional_usd=min_notional_usd,
            mm_rate=ms.mm_rate_cache.get(order.asset, 0.05),
            mark_px_override=open_px,
        )
        stamp = pending.queued_ts.isoformat()
        for k in range(n_before, len(bucket.rejected_orders)):
            bucket.rejected_orders[k] = dataclasses.replace(
                bucket.rejected_orders[k], timestamp=stamp
            )
    bucket.pending_orders = []


def _partition_strategy_output(
    orders: List[OrderCommand],
    ms: MarketState,
    bucket: StateBucket,
    i: int,
    min_notional_usd: float,
) -> None:
    """R1/R2 partition of strategy output: TRIGGER -> placement path (R2);
    everything else -> pending market queue (R1), stamped with this bar."""
    trigger_orders: List[OrderCommand] = []
    for order in orders:
        if order.normalized_type() == OrderType.TRIGGER.value:
            trigger_orders.append(order)
        else:
            bucket.pending_orders.append(
                PendingOrder(order=order, queued_ts=ms.timestamp, queued_bar=i)
            )
    if trigger_orders:
        place_triggers(trigger_orders, ms, bucket, min_notional_usd)


def _reject_unpriceable_pendings(bucket: StateBucket, sd: SimData, i: int) -> None:
    """R1 single rejection site for unpriceable fill bars: drop each pending whose
    fill bar (i+1) has no priceable open, rejecting it with its queued timestamp."""
    next_i = i + 1
    surviving: List[PendingOrder] = []
    for pending in bucket.pending_orders:
        if next_i < sd.n_bars and _has_priceable_ohlc(sd, pending.order.asset, next_i):
            surviving.append(pending)
        else:
            bucket.rejected_orders.append(
                _reject_pending_event(pending, "pending order unpriceable at fill bar")
            )
    bucket.pending_orders = surviving


def run_backtest(
    strategy: Strategy,
    market: MarketData,
    ranks_path: str,
    bt_cfg: BacktestConfig,
    verbose: bool = True,
    detailed_output: bool = False,
) -> SimResult:
    """Thin wrapper: prepare SimData, then run the v2 pipeline (R6)."""
    sd = prepare_sim_data(market, ranks_path)
    return run_backtest_prepared(
        strategy, sd, bt_cfg, verbose=verbose, detailed_output=detailed_output
    )


def run_backtest_prepared(
    strategy: Strategy,
    sd: SimData,
    bt_cfg: BacktestConfig,
    verbose: bool = False,
    detailed_output: bool = False,
) -> SimResult:
    """Engine v2 bar pipeline (spec R6) over a prepared `SimData`.

    Normal bar i: re-mark + FIFO pending fills at open[i] (R1) -> liquidation
    precedence scan (R5) -> trigger fills on the bar range (R3/R4) -> mark at
    close[i] -> close-mark liquidation + deferred suppressed triggers (R5) ->
    strategy.run + partition (queue markets / place triggers) -> funding ->
    liquidation -> gap scan (force-close + unpriceable-pending reject) ->
    equity snapshot. Stale-trigger sweeps run after every position-removing step.
    Last bar: reject all pendings ("backtest end") -> triggers on bar N-1's range
    -> close_all(FORCE) -> snapshot (no strategy, funding, or gap scan).
    """
    n_bars = sd.n_bars
    mm_cache: Dict[str, float] = {p: mm_rate_for(p) for p in sd.perps}
    margin_mode = _strategy_margin_mode(strategy)

    bucket = StateBucket(
        market_state=_market_state_for(sd, 0, mm_cache),
        cash=bt_cfg.initial_equity,
        margin_mode=margin_mode,
    )
    for p in sd.perps:
        bucket.perp_stats(p)

    n_perps = len(sd.perps)
    perp_idx = sd.perp_to_idx
    eq_matrix = np.zeros((n_perps, n_bars))
    pos_matrix = np.zeros((n_perps, n_bars))
    pos_qty_matrix = np.zeros((n_perps, n_bars))
    total_eq = np.zeros(n_bars)

    fee_bps = bt_cfg.taker_fee_bps
    min_notional = bt_cfg.min_notional_usd
    last_idx = n_bars - 1
    strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
    iterator = tqdm(range(n_bars), desc=strategy_name) if verbose else range(n_bars)

    for i in iterator:
        ms = _market_state_for(sd, i, mm_cache)
        bucket.market_state = ms

        if i == last_idx:
            # (last bar) reject all queued markets: no fill bar exists.
            for pending in bucket.pending_orders:
                bucket.rejected_orders.append(
                    _reject_pending_event(pending, "backtest end")
                )
            bucket.pending_orders = []
            # Evaluate resting triggers on bar N-1's range before the final close.
            suppress = precedence_scan(bucket, ms)
            fill_triggers(bucket, ms, fee_bps, suppress)
            sweep_stale_triggers(bucket)
            execute_deferred_triggers(bucket, ms, fee_bps, suppress)
            sweep_stale_triggers(bucket)
            close_all(bucket, ms, fee_bps, CloseReason.FORCE)
            sweep_stale_triggers(bucket)
        else:
            # (a) re-mark at open[i] + FIFO pending market fills pinned to open.
            _fill_pending_orders(bucket, ms, fee_bps, min_notional)
            sweep_stale_triggers(bucket)
            # (b) liquidation precedence scan (marks each candidate at its extreme).
            suppress = precedence_scan(bucket, ms)
            # (c) trigger fills on the bar range (R3 fire/fill + R4 worst-fill).
            fill_triggers(bucket, ms, fee_bps, suppress)
            sweep_stale_triggers(bucket)
            # (d) mark all positions at close[i].
            mark_positions(bucket, ms)
            # (e) close-mark liquidation, then sweep orphaned triggers.
            liquidate_if_needed(bucket, ms, fee_bps)
            sweep_stale_triggers(bucket)
            # (f) deferred suppressed triggers on assets that survived liquidation.
            execute_deferred_triggers(bucket, ms, fee_bps, suppress)
            sweep_stale_triggers(bucket)
            # (g) strategy decision -> partition into pending markets / triggers.
            orders = _strategy_orders(strategy, bucket)
            _partition_strategy_output(orders, ms, bucket, i, min_notional)
            # (h) funding after all trigger fills (never before its info set).
            apply_funding(bucket, ms)
            # (i) liquidation on the post-funding book.
            liquidate_if_needed(bucket, ms, fee_bps)
            sweep_stale_triggers(bucket)
            # (j) gap scan: force-close + reject/drop unpriceable pendings (R1).
            _force_close_positions_before_price_gap(bucket, sd, i, fee_bps)
            sweep_stale_triggers(bucket)
            _reject_unpriceable_pendings(bucket, sd, i)

        contrib = np.zeros(n_perps)
        for p, s in bucket.stats_data_bucket.items():
            contrib[perp_idx[p]] = (
                s["locked_realized"] + s["locked_funding"] - s["locked_fees"]
            )

        for pos in bucket.positions_by_asset.values():
            j = perp_idx[pos.asset]
            contrib[j] += pos.unrealized_pnl
            pos_matrix[j, i] = pos.signed_invested_notional
            pos_qty_matrix[j, i] = pos.size

        eq_matrix[:, i] = contrib
        total_eq[i] = bucket.account_equity()

    per_perp_eq = {p: eq_matrix[perp_idx[p]] for p in sd.perps}
    per_perp_position = {p: pos_matrix[perp_idx[p]] for p in sd.perps}
    per_perp_position_qty = {p: pos_qty_matrix[perp_idx[p]] for p in sd.perps}

    metrics_total = series_metrics(total_eq, bt_cfg.initial_equity, bt_cfg.annualization)
    metrics_per_perp: Dict[str, Dict[str, float]] = {}
    for p, series in per_perp_eq.items():
        if series[-1] == 0.0 and not np.any(series != 0.0):
            continue
        eq_p = bt_cfg.initial_equity + series
        metrics_per_perp[p] = series_metrics(eq_p, bt_cfg.initial_equity, bt_cfg.annualization)

    n_opened = sum(s["n_opened"] for s in bucket.stats_data_bucket.values())
    n_closed = sum(s["n_closed"] for s in bucket.stats_data_bucket.values())
    n_liq = sum(s["n_liquidated"] for s in bucket.stats_data_bucket.values())
    long_pnl = sum(s["locked_realized_long"] for s in bucket.stats_data_bucket.values())
    short_pnl = sum(s["locked_realized_short"] for s in bucket.stats_data_bucket.values())
    liq_long_pnl = sum(s["locked_liq_long"] for s in bucket.stats_data_bucket.values())
    liq_short_pnl = sum(s["locked_liq_short"] for s in bucket.stats_data_bucket.values())
    funding_pnl = sum(s["locked_funding"] for s in bucket.stats_data_bucket.values())
    total_fees = sum(s["locked_fees"] for s in bucket.stats_data_bucket.values())

    if verbose and detailed_output:
        print_metrics_table(strategy_name, metrics_per_perp, metrics_total)

    if not math.isfinite(bucket.cash):
        raise RuntimeError(f"cash went non-finite: {bucket.cash}")

    return SimResult(
        strategy_name=strategy_name,
        margin_mode=margin_mode,
        timeline=sd.timeline,
        total_equity=total_eq,
        per_perp_equity=per_perp_eq,
        per_perp_position=per_perp_position,
        per_perp_position_qty=per_perp_position_qty,
        liquidation_events=bucket.liquidation_events,
        execution_events=bucket.execution_events,
        rejected_orders=bucket.rejected_orders,
        funding_events=bucket.funding_events,
        metrics_total=metrics_total,
        metrics_per_perp=metrics_per_perp,
        n_opened=int(n_opened),
        n_closed=int(n_closed),
        n_liquidated=int(n_liq),
        long_pnl=float(long_pnl),
        short_pnl=float(short_pnl),
        liq_long_pnl=float(liq_long_pnl),
        liq_short_pnl=float(liq_short_pnl),
        funding_pnl=float(funding_pnl),
        total_fees=float(total_fees),
    )


def run_backtest_from_trader_file(
    trader_path: str,
    market: MarketData,
    ranks_path: str,
    bt_cfg: BacktestConfig,
    verbose: bool = True,
    detailed_output: bool = False,
    *trader_args,
    **trader_kwargs,
) -> SimResult:
    trader = load_trader(trader_path, *trader_args, **trader_kwargs)
    return run_backtest(trader, market, ranks_path, bt_cfg, verbose, detailed_output)
