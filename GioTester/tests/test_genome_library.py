# tests/test_genome_library.py
from __future__ import annotations

import src.genome.library  # noqa: F401  (registers genes on import)
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view


def _release_state():
    ranks = {
        "AAA": (0.90, 0.0),
        "BBB": (0.80, 0.0),
        "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0),
        "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(market=market, ranks=ranks, is_release=True)


def test_all_tradable_filter_returns_market_keys():
    gene = build_gene("universe_filter", "all_tradable", {})
    state = _release_state()
    assert gene.eligible(state) == set(state.market)


def test_rank_signal_scores_finite_in_universe():
    gene = build_gene("signal", "rank", {})
    state = _release_state()
    universe = set(state.market)
    scores = gene.score(state, universe)
    assert scores["AAA"] == 0.90
    assert set(scores) == universe


def test_rank_signal_excludes_out_of_universe():
    gene = build_gene("signal", "rank", {})
    state = _release_state()
    scores = gene.score(state, {"AAA", "BBB"})
    assert set(scores) == {"AAA", "BBB"}


def test_release_bar_timing():
    gene = build_gene("entry_timing", "release_bar", {})
    assert gene.should_enter(_release_state()) is True
    non_release = make_state(market={}, ranks=None, is_release=False)
    assert gene.should_enter(non_release) is False


def test_fixed_notional_sizing_longs_then_shorts():
    gene = build_gene(
        "sizing",
        "fixed_notional",
        {"notional_long": 10.0, "notional_short": 10.0, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA", "BBB"], ["CCC", "DDD"])
    assert [o.asset for o in orders] == ["AAA", "BBB", "CCC", "DDD"]
    assert [o.side for o in orders] == [1, 1, -1, -1]
    assert all(o.order_type == "market" and not o.reduce_only for o in orders)
    assert all(o.notional == 10.0 for o in orders)


def test_fixed_notional_gates_below_min():
    gene = build_gene(
        "sizing",
        "fixed_notional",
        {"notional_long": 5.0, "notional_short": 10.0, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA"], ["CCC"])
    # long gated out (5 < 10); only the short survives
    assert [o.asset for o in orders] == ["CCC"]
