# tests/test_genome_adapter.py
from __future__ import annotations

import pytest

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
    # next bar: AAA held; bracket re-emits its full SL+TP trigger set anchored
    # to the realized entry_price (100.0), regardless of the current mark.
    market = {"AAA": market_view("AAA", 112.0)}
    positions = {"AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0)}
    state = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    orders = trader.run(state)

    aaa_orders = [o for o in orders if o.asset == "AAA"]
    assert len(aaa_orders) == 2
    by_client = {o.client_id: o for o in aaa_orders}
    sl, tp = by_client["AAA:sl"], by_client["AAA:tp"]

    assert sl.order_type == "trigger" and sl.trigger_direction == "stop"
    assert sl.side == -1 and sl.reduce_only and sl.size == 0.1
    assert sl.trigger_px == pytest.approx(95.0)  # entry*(1-sl_pct)

    assert tp.order_type == "trigger" and tp.trigger_direction == "tp"
    assert tp.side == -1 and tp.reduce_only and tp.size == 0.1
    assert tp.trigger_px == pytest.approx(110.0)  # entry*(1+tp_pct)


def test_run_entry_decision_bar_emits_no_triggers():
    # On the bar a position is opened, the position does not exist yet in
    # state.positions (it fills next open, R1); the exit gene has nothing to
    # anchor to, so only entry (market) orders are emitted -- no TRIGGER orders.
    trader = build_trader(_bracket_genome())
    orders = trader.run(_release_state())
    assert orders  # entries were emitted
    assert all(o.order_type != "trigger" for o in orders)
    assert all(o.trigger_px is None and o.trigger_direction is None for o in orders)


# --- genome-level knob hoist (leverage / min_notional_usd) ------------------


def test_genome_leverage_propagates_to_entry_and_trigger_orders():
    genome = _bracket_genome()
    genome = Genome(**{**genome.__dict__, "leverage": 2.0})
    trader = build_trader(genome)

    entry_orders = trader.run(_release_state())
    assert entry_orders
    assert all(o.leverage == 2.0 for o in entry_orders)

    market = {"AAA": market_view("AAA", 112.0)}
    positions = {"AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0)}
    state = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    exit_orders = trader.run(state)
    aaa_orders = [o for o in exit_orders if o.asset == "AAA"]
    assert len(aaa_orders) == 2
    assert all(o.leverage == 2.0 for o in aaa_orders)


def test_genome_min_notional_usd_gates_sizing():
    genome = _bracket_genome()
    genome = Genome(**{**genome.__dict__, "min_notional_usd": 25.0})
    trader = build_trader(genome)

    # sizing gene's notional_long/notional_short (10.0 each) are now below the
    # genome-hoisted min_notional_usd (25.0) -> no entry orders are emitted.
    orders = trader.run(_release_state())
    assert orders == []


def test_genome_defaults_preserve_existing_behavior():
    # leverage=1.0, min_notional_usd=10.0 defaults must reproduce the
    # pre-hoist behavior exactly (existing tests above assert this already;
    # this pins the default values themselves).
    genome = Genome(
        name="Defaults",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional"),
        exit_rule=GeneSpec("bracket"),
    )
    assert genome.leverage == 1.0
    assert genome.min_notional_usd == 10.0
    trader = build_trader(genome)
    orders = trader.run(_release_state())
    assert orders
    assert all(o.leverage == 1.0 for o in orders)
