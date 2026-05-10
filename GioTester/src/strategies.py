from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .position import NewOrder, Position  # noqa: E402
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
        self._min_notional_warned: set = (
            set()
        )  # pos ids warned once; avoids per-bar spam

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
        out: List[Position] = []
        for pos in list(bucket.current_positions.values()):
            # 1) liquidation (always active) — checked vs latest mark
            liq_px = pos.liquidation_price()
            if math.isfinite(liq_px) and pos.mark_price > 0:
                crossed = (pos.side == 1 and pos.mark_price <= liq_px) or (
                    pos.side == -1 and pos.mark_price >= liq_px
                )
                if crossed:
                    pos.close_reason = "liquidation"
                    pos.close_price = liq_px
                    out.append(pos)
                    continue
            # 2) stoploss placeholder
            if self._stoploss_hit(pos, bucket):
                pos.close_reason = "stoploss"
                pos.close_price = None
                out.append(pos)
                continue
            # 3) expiry
            if pos.bar_age >= pos.expiry_bars:
                # min-notional gate on close (force keep if residual < threshold)
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
                pos.close_reason = "expiry"
                pos.close_price = None
                out.append(pos)
        return out

    def _stoploss_hit(self, pos: Position, bucket: StateBucket) -> bool:
        return False  # placeholder — no SL/TP for HODL

    # ——— open ———
    def open_positions(self, bucket: StateBucket) -> List[NewOrder]:
        return []

    def save_stats(
        self, bucket: StateBucket, closed: List[Position], opened: List[NewOrder]
    ) -> None:
        for pos in closed:
            s = bucket.perp_stats(pos.perp)
            s["n_closed"] += 1
            if pos.close_reason == "liquidation":
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
        # Filter: only perps that have OHLC at this bar
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

        cands.sort(key=lambda t: (t[1], t[0]))  # ascending by signal, deterministic
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        # avoid overlap when universe small
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


class HODL_10(HODL):
    name = "HODL_10"

    def __init__(self, **kw) -> None:
        kw.setdefault("expiry_days", 10)
        super().__init__(**kw)


class HODL_30(HODL):
    name = "HODL_30"

    def __init__(self, **kw) -> None:
        kw.setdefault("expiry_days", 30)
        super().__init__(**kw)


class HODL_combined(HODL):
    """Long perps that rank top-N in BOTH pred_10d and pred_30d; short bottom-N in both.
    n_long / n_short control the intersection pool size per signal independently.
    """

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

        top_10 = {p for p, _, _ in by_10[-self.n_long:]}
        top_30 = {p for p, _, _ in by_30[-self.n_long:]}
        long_perps = sorted(top_10 & top_30)

        bot_10 = {p for p, _, _ in by_10[: self.n_short]}
        bot_30 = {p for p, _, _ in by_30[: self.n_short]}
        short_perps = sorted((bot_10 & bot_30) - set(long_perps))

        orders: List[NewOrder] = []
        for perp in long_perps:
            if self.notional_long >= self.min_notional_usd:
                orders.append(
                    NewOrder(perp=perp, side=+1, notional=self.notional_long,
                             leverage=self.leverage, expiry_bars=self.expiry_bars)
                )
        for perp in short_perps:
            if self.notional_short >= self.min_notional_usd:
                orders.append(
                    NewOrder(perp=perp, side=-1, notional=self.notional_short,
                             leverage=self.leverage, expiry_bars=self.expiry_bars)
                )
        return orders
