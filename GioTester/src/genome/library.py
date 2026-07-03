# src/genome/library.py
"""Gene library (Phase T taxonomy v1).

Composition constraint (Task T2): `entry_timing`/`delay` fires on non-release
bars (k bars after the most recent release), where `state.current_ranks_row`
is None. Plain `signal`/`rank` and `signal`/`rank_30d` read that row directly
and return {} off-release, which would starve `delay`-timed entries. `delay`
must therefore be composed with a *_cached rank signal (`rank_cached`,
`rank_30d_cached`) or a non-rank signal (`momentum`, `funding_carry`). The
Phase 1 sweep generator must respect this pairing when enumerating genomes.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Set

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


@register("signal", "rank_30d")
class Rank30dSignal(RankSignal):
    """RankSignal locked to field_index=1 (30-day horizon prediction column).

    Distinct kind from `rank` for mechanism accounting (macro-plan §2) even
    though the implementation is a thin subclass.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.pop("field_index", None)
        super().__init__(field_index=1, **kwargs)


class _CachedRankSignal:
    """RankSignal variant that caches the last non-None `current_ranks_row`.

    observe() latches the most recently seen release row so score() keeps
    returning values on non-release bars (the row itself is None there).
    Requires the adapter's observe hook to be driven every bar. See module
    docstring for the `delay`-timing pairing constraint.
    """

    def __init__(self, *, field_index: int = 0, **_) -> None:
        self.field_index = int(field_index)
        self._cached_row: Optional[Dict[str, tuple]] = None

    def observe(self, state) -> None:
        if state.current_ranks_row is not None:
            self._cached_row = state.current_ranks_row

    def score(self, state, universe: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        row = self._cached_row
        if not row:
            return out
        for asset, preds in row.items():
            if asset not in universe:
                continue
            sig = preds[self.field_index]
            if math.isfinite(sig):
                out[asset] = float(sig)
        return out


@register("signal", "rank_cached")
class RankCachedSignal(_CachedRankSignal):
    """Cached variant of `rank` (field_index=0 by default); see _CachedRankSignal."""


@register("signal", "rank_30d_cached")
class Rank30dCachedSignal(_CachedRankSignal):
    """Cached variant of `rank_30d`, locked to field_index=1."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("field_index", None)
        super().__init__(field_index=1, **kwargs)


@register("signal", "momentum")
class MomentumSignal:
    """Trailing price-momentum score: px_now / px_then - 1 over `lookback_bars`.

    Stateful: observe() must be driven every bar (adapter observe hook) to
    accumulate a per-asset `mark_px` deque of length `lookback_bars + 1`.
    score() only emits a value for assets with a FULL window that are also in
    `universe`; assets with insufficient history are skipped entirely (never
    zero-filled). Deque insertion order is per-asset chronological and
    dict-iteration order across assets does not affect the output (the result
    is filtered to `universe`; downstream selection sorts by score).
    """

    def __init__(self, *, lookback_bars: int = 72, **_) -> None:
        self.lookback_bars = int(lookback_bars)
        self._history: Dict[str, Deque[float]] = {}

    def observe(self, state) -> None:
        for asset, mv in state.market.items():
            px = mv.mark_px
            if not math.isfinite(px):
                continue
            dq = self._history.get(asset)
            if dq is None:
                dq = deque(maxlen=self.lookback_bars + 1)
                self._history[asset] = dq
            dq.append(px)

    def score(self, state, universe: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for asset in state.market:
            if asset not in universe:
                continue
            dq = self._history.get(asset)
            if dq is None or len(dq) < self.lookback_bars + 1:
                continue
            px_then, px_now = dq[0], dq[-1]
            if not (math.isfinite(px_then) and px_then != 0.0 and math.isfinite(px_now)):
                continue
            out[asset] = px_now / px_then - 1.0
        return out


@register("signal", "funding_carry")
class FundingCarrySignal:
    """Score = -funding_rate for universe assets with a finite funding rate.

    Stateless: shorts expensive-to-hold longs (positive funding), longs
    negative-funding assets (gets paid to hold).
    """

    def __init__(self, **_) -> None:
        pass

    def score(self, state, universe: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for asset, mv in state.market.items():
            if asset not in universe:
                continue
            rate = mv.funding_rate
            if math.isfinite(rate):
                out[asset] = -float(rate)
        return out


@register("entry_timing", "release_bar")
class ReleaseBarTiming:
    """Enter only on a signal-release bar."""

    def __init__(self, **_) -> None:
        pass

    def should_enter(self, state) -> bool:
        return bool(state.is_release_bar and state.current_ranks_row)


@register("entry_timing", "delay")
class DelayTiming:
    """Enter exactly `bars_after_release` bars after the most recent release bar.

    Stateful: observe() must be driven every bar (adapter observe hook) to
    latch release-bar indices even on bars where should_enter() itself isn't
    otherwise consulted. Re-arms on every subsequent release bar (the most
    recent one wins). Fires on exactly one bar per release cycle -- one bar
    earlier or later returns False. Composes only with *_cached rank signals
    or non-rank signals (`momentum`, `funding_carry`); see module docstring.
    """

    def __init__(self, *, bars_after_release: int = 24, **_) -> None:
        self.bars_after_release = int(bars_after_release)
        self._last_release_bar: Optional[int] = None

    def observe(self, state) -> None:
        if state.is_release_bar and state.current_ranks_row:
            self._last_release_bar = state.bar_index

    def should_enter(self, state) -> bool:
        if self._last_release_bar is None:
            return False
        return (state.bar_index - self._last_release_bar) == self.bars_after_release


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


@register("sizing", "percent_of_equity")
class PercentOfEquitySizing:
    """Open `pct` of current account equity as notional per selected asset.

    Notional is recomputed every call from `state.account.equity` (not a fixed
    USD amount) -- longs first, then shorts, same order convention as
    FixedNotionalSizing. Per-asset notional below `min_notional_usd` is
    skipped (that asset only; equity-derived notional is identical across
    assets in a single call so this either skips all selections or none,
    absent an equity change mid-call, which cannot happen since sizing is
    computed once per bar).
    """

    def __init__(
        self,
        *,
        pct: float = 0.005,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
        **_,
    ) -> None:
        self.pct = float(pct)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)

    def orders_for(self, state, longs: List[str], shorts: List[str]) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        notional = self.pct * state.account.equity
        if notional < self.min_notional_usd:
            return orders

        def _open(asset: str, side: int) -> None:
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

        for asset in longs:
            _open(asset, 1)
        for asset in shorts:
            _open(asset, -1)
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


@register("exit_rule", "time_only")
class TimeOnlyExit:
    """Pure time-based exit: no triggers ever, market close at expiry.

    R7 contract compliance is trivial here since the FULL desired trigger set
    for this gene is always empty -- there is nothing to re-emit while the
    position is held. At `bar_index - entry_bar >= expiry_bars`, emits a
    single reduce-only MARKET full-size close (same helper/pattern as
    BracketExit/TrailingExit expiry). min-notional-gated like the v1 exits.
    """

    def __init__(
        self,
        *,
        expiry_bars: int = 240,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
        **_,
    ) -> None:
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

            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            if expired:
                orders.append(_market_reduce_order(asset, pos.size, self.leverage))
        return orders
