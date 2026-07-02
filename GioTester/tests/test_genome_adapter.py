# tests/test_genome_adapter.py
from __future__ import annotations

import src.genome.library  # noqa: F401
from src.genome.adapter import GeneSpec, Genome, build_trader, select_longs_shorts
from strategy_harness import make_state, market_view, position_view


def _bracket_genome():
    return Genome(
        name="GenomeBracket",
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


def _release_state(bar_index=0):
    ranks = {
        "AAA": (0.90, 0.0), "BBB": (0.80, 0.0), "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0), "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_select_top_bottom_n2():
    scores = {"AAA": 0.90, "BBB": 0.80, "EEE": 0.40, "CCC": 0.20, "DDD": 0.05}
    longs, shorts = select_longs_shorts(scores, 2, 2, held=set())
    assert set(longs) == {"AAA", "BBB"}
    assert set(shorts) == {"CCC", "DDD"}


def test_select_excludes_held():
    scores = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.20, "DDD": 0.05}
    longs, shorts = select_longs_shorts(scores, 2, 2, held={"AAA"})
    assert "AAA" not in longs


def test_select_needs_two_candidates():
    assert select_longs_shorts({"AAA": 0.5}, 2, 2, set()) == ([], [])


def test_run_emits_entries_on_release():
    trader = build_trader(_bracket_genome())
    orders = trader.run(_release_state())
    longs = {o.asset for o in orders if o.side == 1}
    shorts = {o.asset for o in orders if o.side == -1}
    assert longs == {"AAA", "BBB"}
    assert shorts == {"CCC", "DDD"}
    assert trader.margin_mode == "cross"
    assert trader.name == "GenomeBracket"


def test_run_no_entries_off_release():
    trader = build_trader(_bracket_genome())
    non_release = make_state(market={}, ranks=None, is_release=False)
    assert trader.run(non_release) == []


def test_run_exits_tracked_entry_on_tp():
    trader = build_trader(_bracket_genome())
    trader.run(_release_state())  # opens AAA long @100 (ledger records entry)
    # next bar: AAA up 12% -> bracket take-profit exit
    market = {"AAA": market_view("AAA", 112.0)}
    positions = {"AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0)}
    state = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    orders = trader.run(state)
    assert any(o.asset == "AAA" and o.reduce_only for o in orders)
