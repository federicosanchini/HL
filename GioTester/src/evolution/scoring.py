# src/evolution/scoring.py
"""Statistically-sensitive scorer (Phase 1 Task P2).

Plan: `docs/superpowers/plans/2026-07-03-phase1-batch-eval.md`, Task P2.
Objective grounding: `docs/superpowers/specs/2026-07-02-strategy-evolution-daemon-design.md`
S2 ("Objective Function"): normalized PnL (return on peak capital at risk,
not raw USD), lower-CI-bound ranking via a fixed-seed block bootstrap
(common random numbers across genomes for apples-to-apples CI overlap),
an additive/sign-independent liquidation penalty, and a complexity penalty
counted as total free params across the genome's five `GeneSpec`s.

Pure functions over `SimResult` (see `src/result.py`) -- no engine state, no
I/O, no wall-clock, no builtin `hash()` (per `src/evolution/canonical.py`'s
global constraint; this module derives its bootstrap seed from
`data_window_hash` via `hashlib.sha256` only).

SIGN-BUG NOTE (do not reintroduce): the liquidation penalty is a flat
ADDITIVE subtraction (`base -= LIQ_PENALTY`), never a multiplicative
scaling of the signed score. Multiplying a negative `lower_ci_norm_pnl` by
a penalty factor > 1 would make a liquidated genome's score *increase*
(less negative -> "better"), inverting the intent. Additive subtraction is
sign-independent: it always pushes the score down by a fixed amount
regardless of whether the base score is positive or negative.
"""
from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.genome.adapter import Genome
    from src.result import SimResult

SCORER_VERSION = "s1"

LIQ_PENALTY = 1.0
COMPLEXITY_COEF = 0.001
HEADROOM_FLOOR = 0.1
HEADROOM_MAX_PENALTY = 0.05

_HEADROOM_EPS = 1e-9


def gross_notional_series(result: "SimResult") -> np.ndarray:
    """Per-bar sum over perps of |per_perp_position| (signed entry-notional).

    Returns a zero-length array if `result` has no per-perp position series.
    """
    if not result.per_perp_position:
        return np.zeros(0, dtype=float)
    arrays = [np.abs(np.asarray(arr, dtype=float)) for arr in result.per_perp_position.values()]
    return np.sum(np.stack(arrays, axis=0), axis=0)


def peak_gross(result: "SimResult") -> float:
    """Max of `gross_notional_series(result)`; 0.0 if empty or all-zero."""
    series = gross_notional_series(result)
    if series.size == 0:
        return 0.0
    return float(series.max())


def normalized_pnl(result: "SimResult", initial_equity: float) -> float:
    """(final_equity - initial_equity) / max(peak_gross, initial_equity).

    Zero-trade genomes (peak_gross == 0.0) fall back to `initial_equity` as
    the denominator -- and since a zero-trade run's final equity equals
    `initial_equity` (no fills => no PnL), this yields an exact 0.0, never a
    NaN from a zero denominator.
    """
    denom = max(peak_gross(result), initial_equity)
    if denom <= 0:
        return 0.0
    final_equity = float(result.total_equity[-1]) if result.total_equity.size > 0 else initial_equity
    return (final_equity - initial_equity) / denom


def _circular_block_resample(
    diffs: np.ndarray, starts: np.ndarray, block_bars: int, n: int
) -> np.ndarray:
    """One circular block-bootstrap resample of `diffs`, truncated to `n` values.

    `starts` are block start indices into `diffs` (length N = diffs.size).
    Each block takes `block_bars` consecutive elements from `diffs`, wrapping
    around circularly (index modulo N) past the end of the series. Blocks are
    concatenated in `starts` order and truncated to exactly `n` elements --
    the original series length -- so every resample sums the same number of
    per-bar diffs as the real series, regardless of how `block_bars` divides
    `n`. Exposed (not fully private) so tests can verify the truncation and
    wrap-around behavior directly without reaching into `bootstrap_pnl_ci`'s
    RNG loop.
    """
    N = diffs.shape[0]
    idx = (starts[:, None] + np.arange(block_bars)[None, :]) % N
    return diffs[idx].reshape(-1)[:n]


def bootstrap_pnl_ci(
    total_equity: np.ndarray,
    data_window_hash: str,
    initial_equity: float,
    peak_gross_denom: float,
    block_bars: int = 336,
    n_boot: int = 500,
) -> Tuple[float, float]:
    """Circular block-bootstrap 90% CI on normalized total PnL.

    Seed is derived ONLY from `data_window_hash` (common random numbers
    across genomes evaluated on the same data window -- makes CI overlap
    checks apples-to-apples: two genomes' bootstrap draws use the identical
    resampling pattern, so any CI separation reflects the genomes, not RNG
    noise). Resamples the per-bar equity *diffs* via circular blocks of
    `block_bars` consecutive diffs (wrap-around), truncated to exactly the
    original N diffs before summing, then normalizes by
    `max(peak_gross_denom, initial_equity)` -- the same denominator
    convention as `normalized_pnl`.

    Returns `(0.0, 0.0)` (documented sentinel, not a raised error) when there
    is not enough data for even one full block: `N < block_bars` or `N == 0`.
    """
    diffs = np.diff(np.asarray(total_equity, dtype=float))
    N = diffs.shape[0]
    if N == 0 or N < block_bars:
        return (0.0, 0.0)

    denom = max(peak_gross_denom, initial_equity)
    if denom <= 0:
        return (0.0, 0.0)

    seed = int.from_bytes(hashlib.sha256(data_window_hash.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.Generator(np.random.PCG64(seed))

    n_blocks = math.ceil(N / block_bars)
    resampled_norm_pnl = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, N, size=n_blocks)
        resample = _circular_block_resample(diffs, starts, block_bars, N)
        resampled_norm_pnl[b] = float(resample.sum()) / denom

    lo, hi = np.percentile(resampled_norm_pnl, [5, 95])
    return float(lo), float(hi)


def margin_headroom(result: "SimResult") -> float:
    """Min over bars of (equity - maintenance) / equity, using EXACT per-bar maintenance.

    `per_perp_maintenance` (Task P1) is close-marked exact maintenance margin,
    not an entry-notional approximation -- so this cannot be optimistic about
    a losing short (whose maintenance grows as the mark moves against it).
    Bars where `equity <= EPS` are skipped (division-by-~zero guard); if no
    bar is valid the documented sentinel is `0.0` (never inf/nan).
    """
    equity = np.asarray(result.total_equity, dtype=float)
    if equity.size == 0:
        return 0.0

    maint_total = np.zeros(equity.shape[0], dtype=float)
    for arr in result.per_perp_maintenance.values():
        maint_total += np.asarray(arr, dtype=float)

    valid = equity > _HEADROOM_EPS
    if not np.any(valid):
        return 0.0

    headroom = (equity[valid] - maint_total[valid]) / equity[valid]
    return float(headroom.min())


def complexity(genome: "Genome") -> int:
    """Total count of non-None free params across the 5 GeneSpecs' params dicts."""
    specs = (
        genome.universe_filter,
        genome.signal,
        genome.entry_timing,
        genome.sizing,
        genome.exit_rule,
    )
    return sum(1 for spec in specs for value in spec.params.values() if value is not None)


def score_v1(
    lower_ci_norm_pnl: float,
    n_liquidated: int,
    complexity: int,
    headroom: float,
) -> float:
    """Composite score (`scorer_version` s1).

    base = lower-CI normalized PnL, then:
      - liquidation penalty: flat ADDITIVE `-LIQ_PENALTY` if `n_liquidated>0`
        (sign-independent -- see module docstring; NEVER a multiplicative
        scale of the signed base score).
      - complexity penalty: `-COMPLEXITY_COEF * complexity`.
      - headroom penalty: linear ramp from 0 (at `headroom >= HEADROOM_FLOOR`)
        to `-HEADROOM_MAX_PENALTY` (at `headroom <= 0`).
    """
    base = lower_ci_norm_pnl
    if n_liquidated > 0:
        base -= LIQ_PENALTY
    base -= COMPLEXITY_COEF * complexity
    if headroom < HEADROOM_FLOOR:
        base -= HEADROOM_MAX_PENALTY * (HEADROOM_FLOOR - max(headroom, 0.0)) / HEADROOM_FLOOR
    return base
