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


# --- append to src/genome/library.py ---
from .genes import EntryLedger


def _reduce_order(asset: str, size: float, leverage: float) -> OrderCommand:
    return OrderCommand(
        asset=asset,
        side=-1 if size > 0 else 1,
        order_type=OrderType.MARKET.value,
        size=abs(size),
        leverage=leverage,
        reduce_only=True,
    )


def _signed_ret(size: float, mark_px: float, entry_px: float) -> float:
    side = 1.0 if size > 0 else -1.0
    return side * (mark_px / entry_px - 1.0)


@register("exit_rule", "bracket")
class BracketExit:
    """Symmetric-or-asymmetric stop/take-profit bracket with a time expiry.

    tp_pct=inf -> stop-only; sl_pct=inf -> take-profit-only.
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
            ret = _signed_ret(pos.size, mark_px, rec.price)
            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            hit = ret <= -self.sl_pct or ret >= self.tp_pct
            if not (hit or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(_reduce_order(asset, pos.size, self.leverage))
        return orders


@register("exit_rule", "trailing")
class TrailingExit:
    """Exit when price retraces `trail_pct` from the best favorable price seen."""

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
        self._peak_ret: Dict[str, float] = {}

    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._peak_ret):
            if asset not in live:
                del self._peak_ret[asset]
        for asset, pos in state.positions.items():
            rec = ledger.get(asset)
            mv = state.market.get(asset)
            if rec is None or rec.price <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = _signed_ret(pos.size, mark_px, rec.price)
            peak = max(self._peak_ret.get(asset, ret), ret)
            self._peak_ret[asset] = peak
            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            retraced = (peak - ret) >= self.trail_pct
            if not (retraced or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(_reduce_order(asset, pos.size, self.leverage))
        return orders
