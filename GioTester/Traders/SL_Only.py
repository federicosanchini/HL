from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry; hard stop-loss only — winners ride to expiry."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        sl_pct: float = 0.05,
        **_,
    ) -> None:
        self.name = "StopOnly"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.sl_pct = float(sl_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}

    def _should_exit(self, ret: float) -> bool:
        return ret <= -self.sl_pct

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

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

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (self._should_exit(ret) or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
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

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
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
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
