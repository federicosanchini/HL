# src/genome/library.py
from __future__ import annotations

import math
from typing import Dict, List, Set

from src import OrderCommand, OrderType

from .registry import register


@register("universe_filter", "all_tradable")
class AllTradableFilter:
    """Every asset that has a market view this bar is eligible."""

    def __init__(self, **_) -> None:
        pass

    def eligible(self, state) -> Set[str]:
        return set(state.market.keys())


@register("signal", "rank")
class RankSignal:
    """Score = the prediction at `field_index` from the release-bar ranks row."""

    def __init__(self, *, field_index: int = 0, **_) -> None:
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

    def __init__(self, **_) -> None:
        pass

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


# --- append to src/genome/library.py ---
from .genes import EntryLedger


def _market_reduce_order(asset: str, size: float, leverage: float) -> OrderCommand:
    """Full reduce-only market close, used for expiry (R7)."""
    return OrderCommand(
        asset=asset,
        side=-1 if size > 0 else 1,
        order_type=OrderType.MARKET.value,
        size=abs(size),
        leverage=leverage,
        reduce_only=True,
    )


def _trigger_order(
    asset: str,
    size: float,
    trigger_px: float,
    trigger_direction: str,
    leverage: float,
    client_id: str,
) -> OrderCommand:
    """Reduce-only trigger placement (R2). `size` is the SIGNED position size;
    the trigger side is the opposite of the position (closes it)."""
    return OrderCommand(
        asset=asset,
        side=-1 if size > 0 else 1,
        order_type=OrderType.TRIGGER.value,
        size=abs(size),
        leverage=leverage,
        reduce_only=True,
        trigger_px=trigger_px,
        trigger_direction=trigger_direction,
        client_id=client_id,
    )


@register("exit_rule", "bracket")
class BracketExit:
    """Symmetric-or-asymmetric stop/take-profit bracket with a time expiry.

    R7 contract: emits the FULL desired trigger set every bar a position exists
    (the engine's per-asset replace rule, R2, makes re-emission idempotent).
    tp_pct=inf -> stop-only; sl_pct=inf -> take-profit-only. Levels are anchored
    to `ledger` (refreshed from PositionView.entry_price by the adapter, R7).

    Expiry: once `bar_index - entry_bar >= expiry_bars`, this gene emits ONLY a
    reduce-only MARKET close for that bar -- no trigger orders. R2's replace rule
    only clears an asset's resting triggers when >=1 trigger order is emitted for
    that asset THIS bar; emitting zero triggers on the expiry bar does NOT cancel
    previously-resting SL/TP. That is accepted-and-documented v2 behavior: the
    engine auto-cancels resting triggers the instant the position actually closes
    (any path -- fill, liquidation, force-close), and in the worst case the
    resting SL fires intrabar *before* the queued market exit fills next open,
    which is still a protective, deterministic outcome (never a naked position).
    """

    def __init__(
        self,
        *,
        sl_pct: float = 0.05,
        tp_pct: float = 0.10,
        expiry_bars: int = 240,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self.expiry_bars = int(expiry_bars)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)

    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        for asset, pos in state.positions.items():
            rec = ledger.get(asset)
            mv = state.market.get(asset)
            if rec is None or rec.price <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue

            entry = rec.price
            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            if expired:
                orders.append(_market_reduce_order(asset, pos.size, self.leverage))
                continue

            long = pos.size > 0
            if math.isfinite(self.sl_pct):
                sl_px = entry * (1.0 - self.sl_pct) if long else entry * (1.0 + self.sl_pct)
                orders.append(
                    _trigger_order(asset, pos.size, sl_px, "stop", self.leverage, f"{asset}:sl")
                )
            if math.isfinite(self.tp_pct):
                tp_px = entry * (1.0 + self.tp_pct) if long else entry * (1.0 - self.tp_pct)
                orders.append(
                    _trigger_order(asset, pos.size, tp_px, "tp", self.leverage, f"{asset}:tp")
                )
        return orders


@register("exit_rule", "trailing")
class TrailingExit:
    """Trailing stop that ratchets with the best favorable excursion seen.

    R7 contract: peak/trough is tracked from `mv.high_px` (long) / `mv.low_px`
    (short) -- not mark_px -- so an intrabar wick is captured even if the bar
    closes off the extreme. Monotonic per asset, seeded from the ledger's entry
    price on first observation. Re-emits a single stop trigger at
    `peak*(1-trail_pct)` (long) / `trough*(1+trail_pct)` (short) every bar (R2
    replace makes re-emission idempotent). Expiry follows the same
    market-only-close pattern as BracketExit (see its docstring for the
    trigger-coexistence rationale).
    """

    def __init__(
        self,
        *,
        trail_pct: float = 0.05,
        expiry_bars: int = 240,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.trail_pct = float(trail_pct)
        self.expiry_bars = int(expiry_bars)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)
        self._peak_px: Dict[str, float] = {}

    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._peak_px):
            if asset not in live:
                del self._peak_px[asset]

        for asset, pos in state.positions.items():
            rec = ledger.get(asset)
            mv = state.market.get(asset)
            if rec is None or rec.price <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue

            long = pos.size > 0
            if asset not in self._peak_px:
                self._peak_px[asset] = rec.price  # seed from ledger entry on first observation
            if long:
                self._peak_px[asset] = max(self._peak_px[asset], mv.high_px)
            else:
                self._peak_px[asset] = min(self._peak_px[asset], mv.low_px)
            peak = self._peak_px[asset]

            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue

            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            if expired:
                orders.append(_market_reduce_order(asset, pos.size, self.leverage))
                continue

            stop_px = peak * (1.0 - self.trail_pct) if long else peak * (1.0 + self.trail_pct)
            orders.append(
                _trigger_order(asset, pos.size, stop_px, "stop", self.leverage, f"{asset}:sl")
            )
        return orders
