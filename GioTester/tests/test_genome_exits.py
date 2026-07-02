from __future__ import annotations

import math

import src.genome.library  # noqa: F401
from src.genome.genes import EntryLedger
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view, position_view


def _held_state(mark, *, size=1.0, bar_index=0, entry=100.0):
    market = {"BTC": market_view("BTC", mark)}
    positions = {"BTC": position_view("BTC", size=size, entry_price=entry, mark_price=mark)}
    return make_state(bar_index=bar_index, market=market, positions=positions)


def _ledger(entry_bar=0, entry_px=100.0):
    led = EntryLedger()
    led.record("BTC", entry_bar, entry_px)
    return led


def test_bracket_take_profit_fires():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=111.0), _ledger())  # +11% long
    assert len(orders) == 1
    o = orders[0]
    assert o.asset == "BTC" and o.side == -1 and o.reduce_only and o.size == 1.0


def test_bracket_stop_loss_fires():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=94.0), _ledger())  # -6% long
    assert len(orders) == 1 and orders[0].side == -1


def test_bracket_holds_inside_band():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    assert gene.exits(_held_state(mark=103.0), _ledger()) == []  # +3%, no expiry


def test_bracket_expiry_fires_even_inside_band():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 10, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=101.0, bar_index=10), _ledger(entry_bar=0))
    assert len(orders) == 1


def test_bracket_stop_only_ignores_upside():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": math.inf, "expiry_bars": 240, "min_notional_usd": 10.0})
    assert gene.exits(_held_state(mark=130.0), _ledger()) == []  # +30% never hits tp=inf
    assert len(gene.exits(_held_state(mark=94.0), _ledger())) == 1


def test_bracket_skips_below_min_notional():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    # size 0.05 * mark 111 = 5.55 < 10 -> skip despite tp hit
    assert gene.exits(_held_state(mark=111.0, size=0.05), _ledger()) == []


def test_trailing_exits_on_retrace():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 240, "min_notional_usd": 10.0})
    led = _ledger()
    # ride up to 120 (peak), then retrace to 113 (>5% off peak) -> exit
    assert gene.exits(_held_state(mark=110.0), led) == []
    assert gene.exits(_held_state(mark=120.0), led) == []
    orders = gene.exits(_held_state(mark=113.0), led)
    assert len(orders) == 1 and orders[0].reduce_only
