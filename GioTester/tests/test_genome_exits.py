from __future__ import annotations

import math

import pytest

import src.genome.library  # noqa: F401
from src.genome.genes import EntryLedger
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view, position_view


def _held_state(mark, *, size=1.0, bar_index=0, entry=100.0, high_px=None, low_px=None):
    market = {"BTC": market_view("BTC", mark, high_px=high_px, low_px=low_px)}
    positions = {"BTC": position_view("BTC", size=size, entry_price=entry, mark_price=mark)}
    return make_state(bar_index=bar_index, market=market, positions=positions)


def _ledger(entry_bar=0, entry_px=100.0):
    led = EntryLedger()
    led.record("BTC", entry_bar, entry_px)
    return led


def _by_client(orders):
    return {o.client_id: o for o in orders}


# --- BracketExit ---------------------------------------------------------

def test_bracket_long_emits_sl_and_tp():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=100.0), _ledger())
    assert len(orders) == 2
    byc = _by_client(orders)
    sl, tp = byc["BTC:sl"], byc["BTC:tp"]

    assert sl.asset == "BTC" and sl.side == -1 and sl.reduce_only and sl.size == 1.0
    assert sl.order_type == "trigger" and sl.trigger_direction == "stop"
    assert sl.trigger_px == pytest.approx(95.0)

    assert tp.asset == "BTC" and tp.side == -1 and tp.reduce_only and tp.size == 1.0
    assert tp.order_type == "trigger" and tp.trigger_direction == "tp"
    assert tp.trigger_px == pytest.approx(110.0)


def test_bracket_short_emits_mirrored_sl_and_tp():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=100.0, size=-1.0), _ledger())
    byc = _by_client(orders)
    sl, tp = byc["BTC:sl"], byc["BTC:tp"]

    assert sl.side == 1 and sl.trigger_direction == "stop"
    assert sl.trigger_px == pytest.approx(105.0)  # entry*(1+sl_pct)

    assert tp.side == 1 and tp.trigger_direction == "tp"
    assert tp.trigger_px == pytest.approx(90.0)  # entry*(1-tp_pct)


def test_bracket_tp_inf_emits_sl_only():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": math.inf, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=100.0), _ledger())
    assert len(orders) == 1
    assert orders[0].client_id == "BTC:sl" and orders[0].trigger_direction == "stop"


def test_bracket_sl_inf_emits_tp_only():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": math.inf, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=100.0), _ledger())
    assert len(orders) == 1
    assert orders[0].client_id == "BTC:tp" and orders[0].trigger_direction == "tp"


def test_bracket_reemits_identical_set_every_bar():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    key = lambda orders: sorted(
        (o.asset, o.side, o.trigger_px, o.trigger_direction, o.reduce_only, o.size) for o in orders
    )
    first = gene.exits(_held_state(mark=100.0, bar_index=1), _ledger())
    second = gene.exits(_held_state(mark=103.0, bar_index=2), _ledger())  # unrelated price move
    assert key(first) == key(second)


def test_bracket_expiry_emits_market_only_no_triggers():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 10, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=101.0, bar_index=10), _ledger(entry_bar=0))
    # Only the reduce-only market close is emitted; resting triggers (if any)
    # are left to the engine's auto-cancel-on-close or a protective intrabar
    # fire -- see BracketExit docstring for the R2-replace-rule rationale.
    assert len(orders) == 1
    o = orders[0]
    assert o.order_type == "market" and o.reduce_only and o.size == 1.0
    assert o.trigger_px is None and o.trigger_direction is None


def test_bracket_skips_below_min_notional():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    # size 0.05 * mark 100 = 5.0 < 10 -> emit nothing at all
    assert gene.exits(_held_state(mark=100.0, size=0.05), _ledger()) == []


# --- TrailingExit ----------------------------------------------------------

def test_trailing_ratchets_from_high_not_mark():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 240, "min_notional_usd": 10.0})
    led = _ledger(entry_px=100.0)
    # bar 1: mark closes at 105 but wicks up to 120 intrabar -> peak should track 120
    state = _held_state(mark=105.0, bar_index=1, high_px=120.0, low_px=100.0)
    orders = gene.exits(state, led)
    assert len(orders) == 1
    assert orders[0].trigger_px == pytest.approx(120.0 * 0.95)


def test_trailing_peak_monotonic_lower_high_later_unchanged():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 240, "min_notional_usd": 10.0})
    led = _ledger(entry_px=100.0)
    gene.exits(_held_state(mark=105.0, bar_index=1, high_px=120.0, low_px=100.0), led)
    orders = gene.exits(_held_state(mark=105.0, bar_index=2, high_px=110.0, low_px=100.0), led)
    assert orders[0].trigger_px == pytest.approx(120.0 * 0.95)  # unchanged, peak stays 120


def test_trailing_expiry_emits_market_order():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 5, "min_notional_usd": 10.0})
    led = _ledger(entry_bar=0, entry_px=100.0)
    orders = gene.exits(_held_state(mark=101.0, bar_index=5, high_px=101.0, low_px=101.0), led)
    assert len(orders) == 1
    o = orders[0]
    assert o.order_type == "market" and o.reduce_only
    assert o.trigger_px is None


def test_trailing_skips_below_min_notional():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 240, "min_notional_usd": 10.0})
    assert gene.exits(_held_state(mark=100.0, size=0.05, high_px=100.0, low_px=100.0), _ledger()) == []
