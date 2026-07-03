from __future__ import annotations

import math
from typing import Dict, List

from src import OrderCommand, OrderType
from src.genome import Genome, GeneSpec, build_trader
from strategy_harness import load, make_state, market_view, position_view


def _genome_bracket():
    return Genome(
        name="BracketSLTP",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional", {"notional_long": 10.0, "notional_short": 10.0,
                                           "min_notional_usd": 10.0, "leverage": 1.0}),
        exit_rule=GeneSpec("bracket", {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240,
                                       "min_notional_usd": 10.0, "leverage": 1.0}),
        n_long=2,
        n_short=2,
    )


def _key(orders):
    return sorted(
        (o.asset, o.side, o.order_type, o.notional, o.size, o.reduce_only,
         o.trigger_px, o.trigger_direction)
        for o in orders
    )


def _release_state(bar_index=0):
    ranks = {
        "AAA": (0.90, 0.0), "BBB": (0.80, 0.0), "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0), "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_entry_orders_match_legacy():
    legacy = load("SLTP_Bracket.py")
    genome = build_trader(_genome_bracket())
    state = _release_state()
    assert _key(genome.run(state)) == _key(legacy.run(state))


# --- v2-native exit equivalence -------------------------------------------
# R8(2): the legacy Traders/*.py files still emit market reduce-only exits
# (v1 contract). Genome exit genes now emit TRIGGER placements (R7/v2). These
# are no longer directly comparable, so exit equivalence is instead proven
# against a test-local v2-native comparator that independently derives the
# same SL/TP trigger set from its own entry tracking.


class _V2BracketComparator:
    """Hand-written trigger-emitting trader, independent of src/genome, used
    only to prove the genome's trigger stream is correct -- not to duplicate
    its implementation."""

    def __init__(self, *, sl_pct=0.05, tp_pct=0.10, leverage=1.0):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.leverage = leverage
        self._entry_px: Dict[str, float] = {}

    def _trigger(self, asset, size, trigger_px, direction, client_id):
        return OrderCommand(
            asset=asset,
            side=-1 if size > 0 else 1,
            order_type=OrderType.TRIGGER.value,
            size=abs(size),
            leverage=self.leverage,
            reduce_only=True,
            trigger_px=trigger_px,
            trigger_direction=direction,
            client_id=client_id,
        )

    def run(self, state) -> List[OrderCommand]:
        live = set(state.positions)
        for asset in list(self._entry_px):
            if asset not in live:
                del self._entry_px[asset]

        orders: List[OrderCommand] = []
        for asset, pos in state.positions.items():
            entry = self._entry_px.get(asset)
            if entry is None:
                continue
            long = pos.size > 0
            sl_px = entry * (1 - self.sl_pct) if long else entry * (1 + self.sl_pct)
            tp_px = entry * (1 + self.tp_pct) if long else entry * (1 - self.tp_pct)
            orders.append(self._trigger(asset, pos.size, sl_px, "stop", f"{asset}:sl"))
            orders.append(self._trigger(asset, pos.size, tp_px, "tp", f"{asset}:tp"))

        if state.is_release_bar and state.current_ranks_row:
            ranks = state.current_ranks_row
            cands = sorted(((a, p[0]) for a, p in ranks.items() if math.isfinite(p[0])),
                            key=lambda t: (t[1], t[0]))
            shorts = [a for a, _ in cands[:2]]
            longs = [a for a, _ in cands[-2:]]
            longs = [a for a in longs if a not in shorts]
            for asset in longs:
                self._entry_px[asset] = state.market[asset].mark_px
                orders.append(OrderCommand(asset=asset, side=1, order_type=OrderType.MARKET.value,
                                           notional=10.0, leverage=self.leverage))
            for asset in shorts:
                self._entry_px[asset] = state.market[asset].mark_px
                orders.append(OrderCommand(asset=asset, side=-1, order_type=OrderType.MARKET.value,
                                           notional=10.0, leverage=self.leverage))
        return orders


def test_exit_orders_match_v2_native_comparator_over_sequence():
    comparator = _V2BracketComparator(sl_pct=0.05, tp_pct=0.10)
    genome = build_trader(_genome_bracket())

    # bar 0: both open entries at px 100
    s0 = _release_state(bar_index=0)
    assert _key(genome.run(s0)) == _key(comparator.run(s0))

    # bar 1: positions held (opened at bar 0); genome must re-emit the same
    # anchored trigger set the comparator independently derives.
    # size 0.15 keeps every position's notional (0.15 * ~90-112) above the
    # 10.0 min-notional gate both implementations apply.
    marks = {"AAA": 112.0, "BBB": 108.0, "EEE": 100.0, "CCC": 88.0, "DDD": 92.0}
    market = {a: market_view(a, px) for a, px in marks.items()}
    positions = {
        "AAA": position_view("AAA", size=0.15, entry_price=100.0, mark_price=112.0),
        "BBB": position_view("BBB", size=0.15, entry_price=100.0, mark_price=108.0),
        "CCC": position_view("CCC", size=-0.15, entry_price=100.0, mark_price=88.0),
        "DDD": position_view("DDD", size=-0.15, entry_price=100.0, mark_price=92.0),
    }
    s1 = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    assert _key(genome.run(s1)) == _key(comparator.run(s1))


def test_exit_orders_diverge_from_legacy_by_design():
    """Divergence canary (R8(3)): replaces the removed
    test_exit_orders_match_legacy_over_sequence. Same held-position sequence,
    but legacy emits v1 market reduce-only exits while genome emits v2
    TRIGGER placements -- streams differ by design, each with the expected
    order type. Legacy Traders/*.py keep running unmodified under v2 with
    expected metric shifts (documented in the phase report)."""
    legacy = load("SLTP_Bracket.py")
    genome = build_trader(_genome_bracket())

    s0 = _release_state(bar_index=0)
    legacy.run(s0)
    genome.run(s0)

    marks = {"AAA": 112.0, "BBB": 112.0, "EEE": 100.0, "CCC": 88.0, "DDD": 88.0}
    market = {a: market_view(a, px) for a, px in marks.items()}
    positions = {
        "AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0),
        "BBB": position_view("BBB", size=0.1, entry_price=100.0, mark_price=112.0),
        "CCC": position_view("CCC", size=-0.1, entry_price=100.0, mark_price=88.0),
        "DDD": position_view("DDD", size=-0.1, entry_price=100.0, mark_price=88.0),
    }
    s1 = make_state(bar_index=1, market=market, positions=positions, is_release=False)

    legacy_orders = legacy.run(s1)
    genome_orders = genome.run(s1)

    assert _key(legacy_orders) != _key(genome_orders)
    # legacy: v1 market reduce-only exits (AAA/BBB hit tp -> exit; CCC/DDD hit sl -> exit)
    assert legacy_orders and all(o.order_type == "market" and o.reduce_only for o in legacy_orders)
    # genome: v2 trigger placements (full SL+TP set re-emitted every held bar)
    assert genome_orders and all(o.order_type == "trigger" and o.reduce_only for o in genome_orders)
