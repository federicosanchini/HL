# tests/test_evolution_canonical.py
"""Task P1: canonical genome/window hashing.

Plan: docs/superpowers/plans/2026-07-03-phase1-batch-eval.md, Task P1 +
Global constraints (canonical JSON, forbidden builtin hash(), fingerprint
def).
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

import src.genome.library  # noqa: F401 -- side effect: registers reference genes
from src.evolution.canonical import (
    canonical_genome_dict,
    canonical_json,
    data_window_hash,
    fingerprint,
    fingerprint_hash,
    genome_hash,
)
from src.genome.adapter import GeneSpec, Genome, build_trader
from src.genome.genes import EntryLedger
from src.runner import ENGINE_SEMANTICS_VERSION
from engine_harness import build_sim_data
from strategy_harness import make_state, market_view, position_view

ROOT = Path(__file__).resolve().parent.parent


def _bracket_genome(*, name="G", sl_pct=0.05, tp_pct=0.10, n_long=2, n_short=2) -> Genome:
    return Genome(
        name=name,
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional"),
        exit_rule=GeneSpec("bracket", {"sl_pct": sl_pct, "tp_pct": tp_pct}),
        n_long=n_long,
        n_short=n_short,
    )


# --- genome_hash: process-independence, name-freedom ------------------------

_SUBPROCESS_CODE = """
import sys
sys.path.insert(0, {root!r})
from src.genome.adapter import GeneSpec, Genome
from src.evolution.canonical import genome_hash

g = Genome(
    name="fixed-subprocess-genome",
    universe_filter=GeneSpec("all_tradable"),
    signal=GeneSpec("rank"),
    entry_timing=GeneSpec("release_bar"),
    sizing=GeneSpec("fixed_notional"),
    exit_rule=GeneSpec("bracket", {{"sl_pct": 0.05, "tp_pct": 0.10}}),
    n_long=2,
    n_short=2,
)
print(genome_hash(g))
"""


def test_genome_hash_stable_across_process_restarts():
    # Same genome (modulo display name), computed in-process vs. in a fresh
    # `python -c` subprocess. If genome_hash secretly depended on builtin
    # hash() (per-process salted via PYTHONHASHSEED) these would differ.
    expected = genome_hash(_bracket_genome(name="in-process"))

    code = _SUBPROCESS_CODE.format(root=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    got = result.stdout.strip()
    assert got == expected
    assert len(got) == 64  # sha256 hex digest


def test_genome_hash_ignores_display_name():
    a = _bracket_genome(name="Alpha")
    b = _bracket_genome(name="Bravo")
    assert genome_hash(a) == genome_hash(b)


def test_genome_hash_differs_on_structural_change():
    a = _bracket_genome(n_long=2, n_short=2)
    b = _bracket_genome(n_long=3, n_short=2)
    assert genome_hash(a) != genome_hash(b)


# --- canonical_json: allow_nan guard ----------------------------------------

def test_canonical_json_raises_on_inf():
    with pytest.raises(ValueError):
        canonical_json({"x": math.inf})


def test_canonical_json_raises_on_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_canonical_json_is_sorted_and_compact():
    out = canonical_json({"b": 1, "a": 2})
    assert out == '{"a":2,"b":1}'  # sorted keys, no incidental whitespace


# --- disabled bracket leg: None <-> inf round-trip --------------------------

def test_canonical_genome_dict_serializes_disabled_leg_as_none():
    genome = _bracket_genome(sl_pct=None, tp_pct=0.10)
    d = canonical_genome_dict(genome)
    assert d["exit_rule"]["params"]["sl_pct"] is None
    assert d["exit_rule"]["params"]["tp_pct"] == 0.10
    canonical_json(d)  # must not raise (no raw inf reached json.dumps)


def test_genome_hash_identical_for_none_and_inf_disabled_leg():
    # A genome built with sl_pct=math.inf (BracketExit's internal spelling)
    # must hash identically to one built with sl_pct=None (the sweep's JSON-
    # legal spelling) -- both mean "leg disabled".
    g_none = _bracket_genome(sl_pct=None, tp_pct=0.10)
    g_inf = _bracket_genome(sl_pct=math.inf, tp_pct=0.10)
    assert genome_hash(g_none) == genome_hash(g_inf)


def test_disabled_bracket_leg_round_trips_none_to_inf_and_emits_only_tp():
    genome = _bracket_genome(sl_pct=None, tp_pct=0.10)
    trader = build_trader(genome)

    assert math.isinf(trader._exit.sl_pct)
    assert trader._exit.tp_pct == 0.10

    ledger = EntryLedger()
    ledger.record("BTC", 0, 100.0)
    state = make_state(
        market={"BTC": market_view("BTC", 105.0)},
        positions={"BTC": position_view("BTC", size=1.0, entry_price=100.0, mark_price=105.0)},
    )
    orders = trader._exit.exits(state, ledger)
    assert len(orders) == 1
    assert orders[0].trigger_direction == "tp"
    assert orders[0].order_type == "trigger"


# --- data_window_hash --------------------------------------------------------

def test_data_window_hash_differs_for_different_length_windows():
    sd_a = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 3})
    sd_b = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 5})
    assert data_window_hash(sd_a) != data_window_hash(sd_b)


def test_data_window_hash_stable_for_identical_window():
    sd_a = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 4})
    sd_b = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 4})
    assert data_window_hash(sd_a) == data_window_hash(sd_b)


# --- fingerprint / fingerprint_hash ------------------------------------------

def test_fingerprint_fields_and_hash_determinism():
    sd = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 4})
    fp = fingerprint(sd)
    assert fp == {
        "engine_semantics_version": ENGINE_SEMANTICS_VERSION,
        "data_window_hash": data_window_hash(sd),
        "scorer_version": "s1",
        "gene_library_version": "t1",
    }
    assert fingerprint_hash(fp) == fingerprint_hash(fingerprint(sd))


def test_fingerprint_hash_differs_across_data_windows():
    sd_a = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 3})
    sd_b = build_sim_data({"BTC": [[100.0, 101.0, 99.0, 100.0]] * 6})
    assert fingerprint_hash(fingerprint(sd_a)) != fingerprint_hash(fingerprint(sd_b))
