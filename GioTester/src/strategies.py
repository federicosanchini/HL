from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .position import CloseReason, ExpiryMode, NewOrder, Position  # noqa: E402
from .state import StateBucket  # noqa: E402


class Strategy:
    """Base. Subclasses override _signal_for_perp + _entry_gate as needed."""

    name: str = "Strategy"

    def __init__(
        self,
        expiry_days: int,
        n: int = 3,
        leverage: float = 1.0,
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        taker_fee_bps: float = 4.5,
        min_notional_usd: float = 10.0,
        blackout_days_end: int = 50,
        bars_per_day: int = 24,
        expiry_mode: ExpiryMode = ExpiryMode.RESET_LATEST,
    ) -> None:
        self.expiry_days = int(expiry_days)
        self.expiry_bars = int(expiry_days) * int(bars_per_day)
        self.n = int(n)
        self.leverage = float(leverage)
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.taker_fee_bps = float(taker_fee_bps)
        self.min_notional_usd = float(min_notional_usd)
        self.blackout_bars_end = int(blackout_days_end) * int(bars_per_day)
        self.expiry_mode = expiry_mode
        self._min_notional_warned: set = set()

    # ——— main hook called by simulator each bar ———
    def update_positions(
        self, bucket: StateBucket
    ) -> Tuple[List[Position], List[NewOrder]]:
        closed = self.close_positions(bucket)
        opened = self.open_positions(bucket)
        self.save_stats(bucket, closed, opened)
        return closed, opened

    # ——— close ———
    def close_positions(
        self, bucket: StateBucket, verbose: bool = False
    ) -> List[Position]:
        ms = bucket.market_state
        bar_idx = ms.bar_index
        out: List[Position] = []
        for pos in list(bucket.current_positions.values()):
            # 1) liquidation (always active) — checked vs latest mark
            liq_px = pos.liquidation_price()
            if math.isfinite(liq_px) and pos.mark_price > 0:
                crossed = (pos.side == 1 and pos.mark_price <= liq_px) or (
                    pos.side == -1 and pos.mark_price >= liq_px
                )
                if crossed:
                    pos.close_reason = CloseReason.LIQUIDATION
                    pos.close_price = liq_px
                    pos.close_qty = None  # always full
                    out.append(pos)
                    continue
            # 2) stoploss placeholder
            if self._stoploss_hit(pos, bucket):
                pos.close_reason = CloseReason.STOPLOSS
                pos.close_price = None
                pos.close_qty = None
                out.append(pos)
                continue
            # 3) expiry — behaviour depends on expiry_mode set on position
            if pos.expiry_mode == ExpiryMode.PROPORTIONAL:
                expired = [(q, e) for q, e in pos.tranches if bar_idx >= e]
                if not expired:
                    continue
                remaining = [(q, e) for q, e in pos.tranches if bar_idx < e]
                close_qty = sum(q for q, _ in expired)
                mark = pos.mark_price if pos.mark_price > 0 else pos.entry_price
                remaining_notional = (pos.qty - close_qty) * mark
                if (
                    remaining_notional < self.min_notional_usd
                    or close_qty >= pos.qty - 1e-10
                ):
                    # force full close — residual too small or last tranche
                    pos.tranches = []
                    pos.close_reason = CloseReason.EXPIRY
                    pos.close_price = None
                    pos.close_qty = None
                else:
                    pos.tranches = remaining
                    pos.close_reason = CloseReason.EXPIRY
                    pos.close_price = None
                    pos.close_qty = close_qty
                out.append(pos)
            else:
                if bar_idx >= pos.abs_expiry_bar:
                    residual = abs(pos.qty) * (
                        pos.mark_price if pos.mark_price > 0 else pos.entry_price
                    )
                    if residual < self.min_notional_usd:
                        if pos.id not in self._min_notional_warned:
                            if verbose:
                                print(
                                    f"[keep] pos#{pos.id} {pos.perp}: residual ${residual:.2f} < "
                                    f"${self.min_notional_usd:.2f} min_notional, holding"
                                )
                            self._min_notional_warned.add(pos.id)
                        continue
                    pos.close_reason = CloseReason.EXPIRY
                    pos.close_price = None
                    pos.close_qty = None
                    out.append(pos)
        return out

    def _stoploss_hit(self, pos: Position, bucket: StateBucket) -> bool:
        return False

    # ——— open ———
    def open_positions(self, bucket: StateBucket) -> List[NewOrder]:
        return []

    def save_stats(
        self, bucket: StateBucket, closed: List[Position], opened: List[NewOrder]
    ) -> None:
        for pos in closed:
            s = bucket.perp_stats(pos.perp)
            if pos.close_qty is None:  # full close only
                s["n_closed"] += 1
            if pos.close_reason == CloseReason.LIQUIDATION:
                s["n_liquidated"] += 1
        for o in opened:
            s = bucket.perp_stats(o.perp)
            s["n_opened"] += 1

    # ——— gates ———
    def _entry_gate(self, bucket: StateBucket) -> bool:
        ms = bucket.market_state
        remaining = ms.total_bars - ms.bar_index
        if remaining < self.blackout_bars_end:
            return False
        return True


class HODL(Strategy):
    """Long top-N, short bottom-N at each ranks release_date; hold expiry_bars."""

    name = "HODL"

    def open_positions(self, bucket: StateBucket) -> List[NewOrder]:
        ms = bucket.market_state
        if not ms.is_release_bar:
            return []
        if ms.current_ranks_row is None or len(ms.current_ranks_row) == 0:
            return []
        if not self._entry_gate(bucket):
            return []

        sig_idx = 0 if self.expiry_days == 10 else 1  # 0 -> pred_10d, 1 -> pred_30d
        cands: List[Tuple[str, float]] = []
        for perp, preds in ms.current_ranks_row.items():
            if perp not in ms.ohlc_row:
                continue
            sig = preds[sig_idx]
            if sig is None or not math.isfinite(sig):
                continue
            cands.append((perp, float(sig)))
        if len(cands) < 2:
            return []

        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]

        orders: List[NewOrder] = []
        for perp, _ in longs:
            if self.notional_long < self.min_notional_usd:
                continue
            orders.append(
                NewOrder(
                    perp=perp,
                    side=+1,
                    notional=self.notional_long,
                    leverage=self.leverage,
                    expiry_bars=self.expiry_bars,
                )
            )
        for perp, _ in shorts:
            if self.notional_short < self.min_notional_usd:
                continue
            orders.append(
                NewOrder(
                    perp=perp,
                    side=-1,
                    notional=self.notional_short,
                    leverage=self.leverage,
                    expiry_bars=self.expiry_bars,
                )
            )
        return orders


class HODL10(HODL):
    name = "HODL10"

    def __init__(self, **kw) -> None:
        kw.setdefault("expiry_days", 10)
        super().__init__(**kw)


class HODL30(HODL):
    name = "HODL30"

    def __init__(self, **kw) -> None:
        kw.setdefault("expiry_days", 30)
        super().__init__(**kw)


# ——— expiry_mode variants (factory-generated) ———


def _make_hodl_variant(days: int, mode: ExpiryMode, name_suffix: str) -> type:
    """Factory: create HODL variant with fixed expiry_days and expiry_mode."""

    class _HODLVariant(HODL):
        name = f"HODL{days}_{name_suffix}"

        def __init__(self, **kw) -> None:
            kw.setdefault("expiry_days", days)
            kw.setdefault("expiry_mode", mode)
            super().__init__(**kw)

    return _HODLVariant


HODL10_reset = _make_hodl_variant(10, ExpiryMode.KEEP_EARLIEST, "reset")
HODL30_reset = _make_hodl_variant(30, ExpiryMode.KEEP_EARLIEST, "reset")
HODL10_exp = _make_hodl_variant(10, ExpiryMode.PROPORTIONAL, "exp")
HODL30_exp = _make_hodl_variant(30, ExpiryMode.PROPORTIONAL, "exp")


# ——— existing compound strategies ———


class gap_HODL10(HODL10):
    """HODL10 with limit entries: longs GAP bps below bar open, shorts GAP bps above."""

    name = "gap_HODL10"

    def __init__(self, gap_bps: float = 50.0, **kw) -> None:
        super().__init__(**kw)
        self.gap_bps = float(gap_bps)

    def open_positions(self, bucket: StateBucket) -> List[NewOrder]:
        orders = super().open_positions(bucket)
        ms = bucket.market_state
        for order in orders:
            row = ms.ohlc_row.get(order.perp)
            if row is None:
                continue
            ref = float(row[0])
            if order.side == 1:
                order.limit_price = ref * (1.0 - self.gap_bps / 1e4)
            else:
                order.limit_price = ref * (1.0 + self.gap_bps / 1e4)
        return orders


class HODL_combined(HODL):
    """Long perps that rank top-N in BOTH pred_10d and pred_30d; short bottom-N in both."""

    name = "HODL_combined"

    def __init__(self, n_long: int = 3, n_short: int = 3, **kw) -> None:
        kw.setdefault("expiry_days", 15)
        kw.setdefault("n", max(n_long, n_short))
        super().__init__(**kw)
        self.n_long = int(n_long)
        self.n_short = int(n_short)

    def open_positions(self, bucket: StateBucket) -> List[NewOrder]:
        ms = bucket.market_state
        if not ms.is_release_bar:
            return []
        if not ms.current_ranks_row:
            return []
        if not self._entry_gate(bucket):
            return []

        cands: List[Tuple[str, float, float]] = []
        for perp, preds in ms.current_ranks_row.items():
            if perp not in ms.ohlc_row:
                continue
            p10, p30 = preds
            if not (math.isfinite(p10) and math.isfinite(p30)):
                continue
            cands.append((perp, p10, p30))

        if len(cands) < 2:
            return []

        by_10 = sorted(cands, key=lambda t: (t[1], t[0]))
        by_30 = sorted(cands, key=lambda t: (t[2], t[0]))

        top_10 = {p for p, _, _ in by_10[-self.n_long :]}
        top_30 = {p for p, _, _ in by_30[-self.n_long :]}
        long_perps = sorted(top_10 & top_30)

        bot_10 = {p for p, _, _ in by_10[: self.n_short]}
        bot_30 = {p for p, _, _ in by_30[: self.n_short]}
        short_perps = sorted((bot_10 & bot_30) - set(long_perps))

        orders: List[NewOrder] = []
        for perp in long_perps:
            if self.notional_long >= self.min_notional_usd:
                orders.append(
                    NewOrder(
                        perp=perp,
                        side=+1,
                        notional=self.notional_long,
                        leverage=self.leverage,
                        expiry_bars=self.expiry_bars,
                    )
                )
        for perp in short_perps:
            if self.notional_short >= self.min_notional_usd:
                orders.append(
                    NewOrder(
                        perp=perp,
                        side=-1,
                        notional=self.notional_short,
                        leverage=self.leverage,
                        expiry_bars=self.expiry_bars,
                    )
                )
        return orders
