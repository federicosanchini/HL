# src/evolution/canonical.py
"""Deterministic genome/window hashing (Phase 1 Task P1).

Global constraint (plan `docs/superpowers/plans/2026-07-03-phase1-batch-eval.md`):
builtin `hash()` is FORBIDDEN anywhere under `src/evolution/` -- CPython salts
str/bytes hashing per-process (`PYTHONHASHSEED`) unless explicitly pinned, so
it cannot back a durable digest key across process restarts. `hashlib.sha256`
is used throughout instead.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict

from src import ENGINE_SEMANTICS_VERSION
from src.data_prep import SimData
from src.genome.adapter import GeneSpec, Genome

SCORER_VERSION = "s1"
GENE_LIBRARY_VERSION = "t1"


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sort params by key; map +/-inf floats to `None`.

    A disabled bracket leg (Task P1) may reach here either as `None` (the
    sweep's spelling) or as `math.inf` (BracketExit's internal spelling, e.g.
    if a genome was built by hand). Both mean "leg disabled" and must hash
    identically -- and `canonical_json`'s `allow_nan=False` guard would raise
    on a raw `inf` reaching `json.dumps`, so it is normalized away here.
    """
    out: Dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float) and math.isinf(value):
            value = None
        out[key] = value
    return out


def _slot_dict(spec: GeneSpec) -> Dict[str, Any]:
    return {"kind": spec.kind, "params": _normalize_params(dict(spec.params))}


def canonical_genome_dict(genome: Genome) -> dict:
    """Name-free structural view of a genome.

    `genome.name` is a display label, not identity, so it is deliberately
    excluded: two genomes that differ only in `.name` hash identically.
    """
    return {
        "universe_filter": _slot_dict(genome.universe_filter),
        "signal": _slot_dict(genome.signal),
        "entry_timing": _slot_dict(genome.entry_timing),
        "sizing": _slot_dict(genome.sizing),
        "exit_rule": _slot_dict(genome.exit_rule),
        "n_long": genome.n_long,
        "n_short": genome.n_short,
        "margin_mode": genome.margin_mode,
        "leverage": genome.leverage,
    }


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    `allow_nan=False` makes an accidental `inf`/`nan` reaching this function
    raise loudly (`ValueError`) instead of silently emitting the non-RFC
    `Infinity`/`NaN` tokens `json` would otherwise produce.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def genome_hash(genome: Genome) -> str:
    payload = canonical_json(canonical_genome_dict(genome))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def data_window_hash(sd: SimData) -> str:
    """Identity of an evaluated data window.

    Computed from `(timeline[0], timeline[-1], n_bars, sorted(perps))` so
    in-sample and holdout slices of the SAME underlying data get DISTINCT
    hashes -- and therefore distinct fingerprints -- preventing any cross-fold
    digest collision downstream (Task P3/P4).
    """
    payload = [
        sd.timeline[0].isoformat(),
        sd.timeline[-1].isoformat(),
        sd.n_bars,
        sorted(sd.perps),
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def fingerprint(sd: SimData) -> dict:
    return {
        "engine_semantics_version": ENGINE_SEMANTICS_VERSION,
        "data_window_hash": data_window_hash(sd),
        "scorer_version": SCORER_VERSION,
        "gene_library_version": GENE_LIBRARY_VERSION,
    }


def fingerprint_hash(fp: dict) -> str:
    return hashlib.sha256(canonical_json(fp).encode("utf-8")).hexdigest()
