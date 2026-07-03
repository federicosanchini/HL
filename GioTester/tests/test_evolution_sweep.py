# tests/test_evolution_sweep.py
"""Task P1: sweep generator + validate_genome composition guard.

Plan: docs/superpowers/plans/2026-07-03-phase1-batch-eval.md, Task P1.
"""
from __future__ import annotations

import logging

import pytest

import src.genome.library  # noqa: F401 -- side effect: registers reference genes
from src.evolution.canonical import genome_hash
from src.evolution.sweep import (
    MAX_VARIANTS_PER_KIND,
    N_LONG_SHORT_GRID,
    VARIANT_TABLES,
    generate_seed_genomes,
)
from src.genome.adapter import GeneSpec, Genome, build_trader, validate_genome


# --- validate_genome ----------------------------------------------------------

def _genome(signal_kind: str, timing_kind: str) -> Genome:
    return Genome(
        name="t",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec(signal_kind),
        entry_timing=GeneSpec(timing_kind),
        sizing=GeneSpec("fixed_notional"),
        exit_rule=GeneSpec("bracket"),
    )


@pytest.mark.parametrize("signal_kind", ["rank", "rank_30d"])
def test_validate_genome_raises_on_delay_with_noncached_rank(signal_kind):
    with pytest.raises(ValueError):
        validate_genome(_genome(signal_kind, "delay"))


@pytest.mark.parametrize(
    "signal_kind", ["rank_cached", "rank_30d_cached", "momentum", "funding_carry"]
)
def test_validate_genome_passes_on_delay_with_cached_or_nonrank_signal(signal_kind):
    validate_genome(_genome(signal_kind, "delay"))  # must not raise


def test_validate_genome_passes_on_release_bar_with_any_signal():
    validate_genome(_genome("rank", "release_bar"))  # must not raise


def test_build_trader_raises_on_invalid_composition():
    with pytest.raises(ValueError):
        build_trader(_genome("rank", "delay"))


# --- VARIANT_TABLES budget ----------------------------------------------------

def test_variant_tables_budget_at_most_three_per_kind():
    assert VARIANT_TABLES  # non-empty
    for kind, variants in VARIANT_TABLES.items():
        assert 1 <= len(variants) <= MAX_VARIANTS_PER_KIND, kind


def test_bracket_variant_table_hits_the_budget_ceiling():
    assert len(VARIANT_TABLES["bracket"]) == MAX_VARIANTS_PER_KIND == 3


# --- generate_seed_genomes -----------------------------------------------------

@pytest.fixture(scope="module")
def seed_genomes():
    return generate_seed_genomes()


def test_generate_seed_genomes_count_positive(seed_genomes):
    assert len(seed_genomes) > 0


def test_generate_seed_genomes_logs_count(caplog):
    with caplog.at_level(logging.INFO, logger="src.evolution.sweep"):
        genomes = generate_seed_genomes()
    assert any("generate_seed_genomes" in r.message for r in caplog.records)
    assert any(str(len(genomes)) in r.message for r in caplog.records)


def test_one_sided_variants_present(seed_genomes):
    assert any(g.n_long == 0 and g.n_short > 0 for g in seed_genomes)
    assert any(g.n_short == 0 and g.n_long > 0 for g in seed_genomes)
    # every knob-grid n/k pair should show up on at least one genome
    seen_nk = {(g.n_long, g.n_short) for g in seed_genomes}
    assert seen_nk == set(N_LONG_SHORT_GRID)


def test_all_generated_genomes_pass_validate(seed_genomes):
    for g in seed_genomes:
        validate_genome(g)  # must not raise for any emitted genome


def test_no_delay_rank_combo_emitted(seed_genomes):
    for g in seed_genomes:
        if g.entry_timing.kind == "delay":
            assert g.signal.kind not in ("rank", "rank_30d")


def test_generate_seed_genomes_deterministic_order_across_calls():
    a = generate_seed_genomes()
    b = generate_seed_genomes()
    hashes_a = [genome_hash(g) for g in a]
    hashes_b = [genome_hash(g) for g in b]
    assert hashes_a == hashes_b
    assert hashes_a == sorted(hashes_a)  # sorted by genome_hash, as documented


def test_generate_seed_genomes_no_duplicate_hashes(seed_genomes):
    hashes = [genome_hash(g) for g in seed_genomes]
    assert len(hashes) == len(set(hashes))


def test_generate_seed_genomes_all_build_traders_cleanly(seed_genomes):
    # Spot-check: every emitted genome must be buildable (validate_genome is
    # re-run inside build_trader too -- this proves the two call sites agree).
    for g in seed_genomes[::200]:  # sample across the sorted sweep, not all 3500+
        build_trader(g)
