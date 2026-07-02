# tests/test_genome_equivalence.py
from __future__ import annotations

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
        (o.asset, o.side, o.order_type, o.notional, o.size, o.reduce_only) for o in orders
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


def test_exit_orders_match_legacy_over_sequence():
    legacy = load("SLTP_Bracket.py")
    genome = build_trader(_genome_bracket())

    # bar 0: both open entries at px 100
    s0 = _release_state(bar_index=0)
    assert _key(genome.run(s0)) == _key(legacy.run(s0))

    # bar 1: AAA & BBB (longs) +12% -> tp; CCC & DDD (shorts) -12% price move
    marks = {"AAA": 112.0, "BBB": 112.0, "EEE": 100.0, "CCC": 88.0, "DDD": 88.0}
    market = {a: market_view(a, px) for a, px in marks.items()}
    positions = {
        "AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0),
        "BBB": position_view("BBB", size=0.1, entry_price=100.0, mark_price=112.0),
        "CCC": position_view("CCC", size=-0.1, entry_price=100.0, mark_price=88.0),
        "DDD": position_view("DDD", size=-0.1, entry_price=100.0, mark_price=88.0),
    }
    s1 = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    assert _key(genome.run(s1)) == _key(legacy.run(s1))
