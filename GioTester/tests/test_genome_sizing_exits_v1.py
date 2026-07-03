# tests/test_genome_sizing_exits_v1.py
"""Task T3: percent_of_equity sizing + time_only exit."""
from __future__ import annotations

import src.genome.library  # noqa: F401  (registers genes on import)
from src.genome.genes import EntryLedger
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view, position_view


def _release_state():
    # make_state()'s default AccountView has equity=2000.0 (strategy_harness.py).
    market = {a: market_view(a, 100.0) for a in ("AAA", "BBB", "CCC", "DDD")}
    return make_state(market=market, ranks=None, is_release=True)


def _held_state(mark, *, size=1.0, bar_index=0, entry=100.0):
    market = {"BTC": market_view("BTC", mark)}
    positions = {"BTC": position_view("BTC", size=size, entry_price=entry, mark_price=mark)}
    return make_state(bar_index=bar_index, market=market, positions=positions)


def _ledger(entry_bar=0, entry_px=100.0):
    led = EntryLedger()
    led.record("BTC", entry_bar, entry_px)
    return led


# --- PercentOfEquitySizing --------------------------------------------------

def test_percent_of_equity_notional_matches_pct_times_equity():
    gene = build_gene(
        "sizing", "percent_of_equity",
        {"pct": 0.01, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA"], [])
    assert len(orders) == 1
    assert orders[0].notional == 20.0  # 0.01 * 2000


def test_percent_of_equity_skips_below_min_notional():
    gene = build_gene(
        "sizing", "percent_of_equity",
        {"pct": 0.0001, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()  # notional = 0.2 < 10 -> skip all
    orders = gene.orders_for(state, ["AAA"], ["BBB"])
    assert orders == []


def test_percent_of_equity_longs_then_shorts():
    gene = build_gene(
        "sizing", "percent_of_equity",
        {"pct": 0.01, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA", "BBB"], ["CCC", "DDD"])
    assert [o.asset for o in orders] == ["AAA", "BBB", "CCC", "DDD"]
    assert [o.side for o in orders] == [1, 1, -1, -1]
    assert all(o.order_type == "market" and not o.reduce_only for o in orders)
    assert all(o.notional == 20.0 for o in orders)
    assert all(o.leverage == 1.0 for o in orders)


# --- TimeOnlyExit ------------------------------------------------------------

def test_time_only_emits_nothing_pre_expiry():
    gene = build_gene(
        "exit_rule", "time_only",
        {"expiry_bars": 5, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    led = _ledger(entry_bar=0, entry_px=100.0)
    orders = gene.exits(_held_state(mark=101.0, bar_index=4), led)
    assert orders == []


def test_time_only_emits_market_reduce_only_full_size_at_expiry():
    gene = build_gene(
        "exit_rule", "time_only",
        {"expiry_bars": 5, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    led = _ledger(entry_bar=0, entry_px=100.0)
    orders = gene.exits(_held_state(mark=101.0, bar_index=5, size=2.0), led)
    assert len(orders) == 1
    o = orders[0]
    assert o.order_type == "market"
    assert o.reduce_only is True
    assert o.size == 2.0
    assert o.side == -1  # closes a long
    assert o.trigger_px is None and o.trigger_direction is None


def test_time_only_never_emits_trigger_orders_across_multibar_sequence():
    gene = build_gene(
        "exit_rule", "time_only",
        {"expiry_bars": 3, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    led = _ledger(entry_bar=0, entry_px=100.0)
    all_orders = []
    for bar in range(0, 8):
        state = _held_state(mark=100.0 + bar, bar_index=bar)
        all_orders.extend(gene.exits(state, led))
    assert all(o.order_type != "trigger" for o in all_orders)
    # exactly one market exit emitted (at bar 3, and remains held bars after
    # since this simplified harness doesn't actually close the position --
    # the gene keeps emitting the same reduce-only close each bar past
    # expiry, all still market, never trigger).
    assert all(o.order_type == "market" for o in all_orders)
    assert len(all_orders) >= 1


def test_time_only_skips_below_min_notional_at_expiry():
    gene = build_gene(
        "exit_rule", "time_only",
        {"expiry_bars": 5, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    led = _ledger(entry_bar=0, entry_px=100.0)
    # size 0.05 * mark 100 = 5.0 < 10 -> emit nothing even past expiry
    orders = gene.exits(_held_state(mark=100.0, bar_index=5, size=0.05), led)
    assert orders == []
