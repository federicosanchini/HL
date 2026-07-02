"""Compositional-genome layer.

A strategy = {universe_filter, signal, entry_timing, sizing, exit_rule}.
build_trader(genome) returns a standard Trader the existing engine runs unchanged.
"""

from . import library  # noqa: F401  — side effect: registers reference genes
from .adapter import Genome, GeneSpec, ComposedTrader, build_trader, select_longs_shorts
from .genes import EntryLedger
from .registry import SLOTS, build_gene, register

__all__ = [
    "Genome",
    "GeneSpec",
    "ComposedTrader",
    "build_trader",
    "select_longs_shorts",
    "EntryLedger",
    "SLOTS",
    "build_gene",
    "register",
]
