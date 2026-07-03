# src/evolution/sweep.py
"""Phase 1 seed-sweep generator (Task P1).

`generate_seed_genomes()` enumerates the cartesian product of gene kinds
(taxonomy v1, `src/genome/library.py`) x per-kind param variants x knob grids,
drops structurally-invalid compositions via `validate_genome` (skip + log,
never silently emit a zero-trade genome), deduplicates by `genome_hash`, and
returns the result sorted by genome_hash -- fully deterministic across
process restarts (no builtin `hash()`; see `canonical.py`).
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Tuple

from src.genome.adapter import GeneSpec, Genome, validate_genome

from .canonical import genome_hash

logger = logging.getLogger(__name__)

# --- per-kind param variant tables ------------------------------------------
# Deliberately small and hard-coded (<= MAX_VARIANTS_PER_KIND each -- asserted
# below and covered by test_evolution_sweep.py). A kind absent from this table
# has no tunable params worth sweeping in Phase 1: it gets a single
# empty-params variant, i.e. its gene's own __init__ defaults apply.
MAX_VARIANTS_PER_KIND = 3

VARIANT_TABLES: Dict[str, List[dict]] = {
    "bracket": [
        {"sl_pct": 0.05, "tp_pct": 0.10},
        {"sl_pct": 0.03, "tp_pct": None},  # disabled TP leg
        {"sl_pct": None, "tp_pct": 0.10},  # disabled SL leg
    ],
    "trailing": [{"trail_pct": 0.05}, {"trail_pct": 0.10}],
    "time_only": [{"expiry_bars": 240}, {"expiry_bars": 720}],
    "momentum": [{"lookback_bars": 72}, {"lookback_bars": 168}],
    "delay": [{"bars_after_release": 24}, {"bars_after_release": 72}],
}

for _kind, _variants in VARIANT_TABLES.items():
    assert len(_variants) <= MAX_VARIANTS_PER_KIND, (
        f"{_kind!r}: {len(_variants)} variants exceeds budget of {MAX_VARIANTS_PER_KIND}"
    )
del _kind, _variants

# --- gene kinds per slot (taxonomy v1, src/genome/library.py) --------------
UNIVERSE_KINDS = ["all_tradable"]
SIGNAL_KINDS = [
    "rank", "rank_30d", "rank_cached", "rank_30d_cached", "momentum", "funding_carry",
]
TIMING_KINDS = ["release_bar", "delay"]
SIZING_KINDS = ["fixed_notional", "percent_of_equity"]
EXIT_KINDS = ["bracket", "trailing", "time_only"]

# --- knob grids ---------------------------------------------------------------
# Two-sided books at n_long==n_short in {1,2,3}, plus one-sided books at k=2.
N_LONG_SHORT_GRID: List[Tuple[int, int]] = [(1, 1), (2, 2), (3, 3), (0, 2), (2, 0)]
LEVERAGE_GRID = [1.0, 2.0, 3.0]
MARGIN_MODE_GRID = ["cross"]


def _variants_for(kind: str) -> List[dict]:
    return VARIANT_TABLES.get(kind, [{}])


def _name_for(
    sig_kind: str, timing_kind: str, sizing_kind: str, exit_kind: str,
    n_long: int, n_short: int, leverage: float,
) -> str:
    """Display label only -- excluded from genome_hash (name-free hashing)."""
    return (
        f"seed:{sig_kind}|{timing_kind}|{sizing_kind}|{exit_kind}"
        f"|n{n_long}-{n_short}|lev{leverage:g}"
    )


def generate_seed_genomes() -> List[Genome]:
    """Enumerate the Phase-1 seed sweep, deterministically ordered by genome_hash."""
    seen: Dict[str, Genome] = {}
    n_evaluated = 0
    n_skipped_invalid = 0

    combos = itertools.product(
        UNIVERSE_KINDS,
        SIGNAL_KINDS,
        TIMING_KINDS,
        SIZING_KINDS,
        EXIT_KINDS,
        N_LONG_SHORT_GRID,
        LEVERAGE_GRID,
        MARGIN_MODE_GRID,
    )
    for uf_kind, sig_kind, timing_kind, sizing_kind, exit_kind, nk, leverage, margin_mode in combos:
        n_long, n_short = nk
        for sig_params in _variants_for(sig_kind):
            for timing_params in _variants_for(timing_kind):
                for exit_params in _variants_for(exit_kind):
                    n_evaluated += 1
                    genome = Genome(
                        name=_name_for(
                            sig_kind, timing_kind, sizing_kind, exit_kind,
                            n_long, n_short, leverage,
                        ),
                        universe_filter=GeneSpec(uf_kind, {}),
                        signal=GeneSpec(sig_kind, dict(sig_params)),
                        entry_timing=GeneSpec(timing_kind, dict(timing_params)),
                        sizing=GeneSpec(sizing_kind, {}),
                        exit_rule=GeneSpec(exit_kind, dict(exit_params)),
                        n_long=n_long,
                        n_short=n_short,
                        margin_mode=margin_mode,
                        leverage=leverage,
                    )
                    try:
                        validate_genome(genome)
                    except ValueError:
                        n_skipped_invalid += 1
                        continue
                    seen.setdefault(genome_hash(genome), genome)

    ordered = [seen[h] for h in sorted(seen)]
    n_duplicates = n_evaluated - n_skipped_invalid - len(ordered)
    logger.info(
        "generate_seed_genomes: %d genomes (%d combos evaluated, "
        "%d skipped invalid, %d duplicate hashes)",
        len(ordered), n_evaluated, n_skipped_invalid, n_duplicates,
    )
    return ordered
