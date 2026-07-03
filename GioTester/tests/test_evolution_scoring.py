# tests/test_evolution_scoring.py
"""Task P2: statistically-sensitive scorer.

Plan: docs/superpowers/plans/2026-07-03-phase1-batch-eval.md, Task P2.
Objective grounding: docs/superpowers/specs/2026-07-02-strategy-evolution-daemon-design.md S2.

Hand-built `SimResult` fixtures throughout -- no engine run needed, these are
pure functions over the result's arrays.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evolution.scoring import (  # noqa: E402
    COMPLEXITY_COEF,
    HEADROOM_FLOOR,
    HEADROOM_MAX_PENALTY,
    LIQ_PENALTY,
    SCORER_VERSION,
    _circular_block_resample,
    bootstrap_pnl_ci,
    complexity,
    gross_notional_series,
    margin_headroom,
    normalized_pnl,
    peak_gross,
    score_v1,
)
from src.genome.adapter import GeneSpec, Genome  # noqa: E402
from src.result import SimResult  # noqa: E402


def _make_result(
    *,
    total_equity,
    per_perp_position=None,
    per_perp_maintenance=None,
    n_liquidated=0,
) -> SimResult:
    total_equity = np.asarray(total_equity, dtype=float)
    n_bars = total_equity.shape[0]
    timeline = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n_bars, freq="h", tz="UTC"))
    return SimResult(
        strategy_name="test",
        timeline=timeline,
        total_equity=total_equity,
        per_perp_equity={},
        per_perp_position=per_perp_position or {},
        per_perp_position_qty={},
        liquidation_events=[],
        execution_events=[],
        rejected_orders=[],
        funding_events=[],
        metrics_total={"PnL": 0.0, "DD": 0.0, "Sharpe": 0.0, "Sortino": 0.0},
        metrics_per_perp={},
        n_opened=0,
        n_closed=0,
        n_liquidated=n_liquidated,
        per_perp_maintenance=per_perp_maintenance or {},
    )


# --- constants sanity ---------------------------------------------------------

def test_scorer_version_is_s1():
    assert SCORER_VERSION == "s1"


# --- gross_notional_series / peak_gross ---------------------------------------

def test_peak_gross_sums_across_perps_and_takes_max_over_bars():
    result = _make_result(
        total_equity=[1000.0, 1050.0, 1100.0],
        per_perp_position={
            "BTC": np.array([200.0, -300.0, 100.0]),
            "ETH": np.array([50.0, 100.0, 400.0]),
        },
    )
    # per-bar gross = |BTC| + |ETH|: [250, 400, 500]
    np.testing.assert_allclose(gross_notional_series(result), [250.0, 400.0, 500.0])
    assert peak_gross(result) == 500.0


def test_peak_gross_zero_when_no_positions():
    result = _make_result(total_equity=[1000.0, 1000.0])
    assert gross_notional_series(result).size == 0
    assert peak_gross(result) == 0.0


# --- normalized_pnl -------------------------------------------------------------

def test_normalized_pnl_known_values():
    result = _make_result(
        total_equity=[1000.0, 1050.0, 1100.0],
        per_perp_position={"BTC": np.array([500.0, 500.0, 500.0])},
    )
    # peak_gross=500, denom=max(500,1000)=1000; (1100-1000)/1000
    assert normalized_pnl(result, initial_equity=1000.0) == pytest.approx(0.1)


def test_normalized_pnl_zero_trade_is_exact_zero_not_nan():
    result = _make_result(total_equity=[1000.0, 1000.0, 1000.0])
    value = normalized_pnl(result, initial_equity=1000.0)
    assert value == 0.0
    assert not np.isnan(value)


def test_normalized_pnl_double_zero_denom_guarded():
    # peak_gross=0 AND initial_equity=0 -- denom would be 0; must not raise/NaN.
    result = _make_result(total_equity=[0.0, 0.0])
    value = normalized_pnl(result, initial_equity=0.0)
    assert value == 0.0
    assert not np.isnan(value)


# --- _circular_block_resample: exact truncation to N --------------------------

def test_circular_block_resample_wraps_and_truncates_to_n():
    diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # N=5
    starts = np.array([3, 0])
    resample = _circular_block_resample(diffs, starts, block_bars=3, n=5)
    assert resample.shape[0] == 5
    # block@3 wraps: idx (3,4,0) -> [4,5,1]; block@0: idx (0,1,2) -> [1,2,3]
    # concatenated [4,5,1,1,2,3], truncated to 5 -> [4,5,1,1,2]
    np.testing.assert_allclose(resample, [4.0, 5.0, 1.0, 1.0, 2.0])


# --- bootstrap_pnl_ci -----------------------------------------------------------

def _bootstrap_equity() -> np.ndarray:
    diffs_pattern = [5.0, -3.0, 2.0, -1.0, 4.0] * 5  # N=25 diffs
    return np.concatenate(([1000.0], 1000.0 + np.cumsum(diffs_pattern)))


def test_bootstrap_pnl_ci_deterministic_across_calls_and_lo_le_hi():
    equity = _bootstrap_equity()
    lo_a, hi_a = bootstrap_pnl_ci(equity, "deadbeef", 1000.0, 500.0, block_bars=5, n_boot=50)
    lo_b, hi_b = bootstrap_pnl_ci(equity, "deadbeef", 1000.0, 500.0, block_bars=5, n_boot=50)
    assert (lo_a, hi_a) == (lo_b, hi_b)
    assert lo_a <= hi_a


_BOOTSTRAP_SUBPROCESS_CODE = """
import sys
sys.path.insert(0, {root!r})
import numpy as np
from src.evolution.scoring import bootstrap_pnl_ci

diffs_pattern = [5.0, -3.0, 2.0, -1.0, 4.0] * 5
equity = np.concatenate(([1000.0], 1000.0 + np.cumsum(diffs_pattern)))
lo, hi = bootstrap_pnl_ci(equity, "deadbeef", 1000.0, 500.0, block_bars=5, n_boot=50)
print(repr((lo, hi)))
"""


def test_bootstrap_pnl_ci_deterministic_across_process_restarts():
    # Seed is derived ONLY from data_window_hash -- proves the window-seed
    # (common random numbers) is stable across a fresh interpreter, not
    # accidentally salted by anything process-local.
    equity = _bootstrap_equity()
    expected = bootstrap_pnl_ci(equity, "deadbeef", 1000.0, 500.0, block_bars=5, n_boot=50)

    code = _BOOTSTRAP_SUBPROCESS_CODE.format(root=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    got = ast.literal_eval(result.stdout.strip())
    assert got == expected


def test_bootstrap_pnl_ci_short_series_returns_zero_zero():
    result_a = bootstrap_pnl_ci(
        np.array([1000.0]), "hash", 1000.0, 500.0, block_bars=5, n_boot=10,
    )
    assert result_a == (0.0, 0.0)  # N == 0

    short_equity = np.array([1000.0, 1005.0, 1002.0])  # N=2 diffs < block_bars=5
    result_b = bootstrap_pnl_ci(short_equity, "hash", 1000.0, 500.0, block_bars=5, n_boot=10)
    assert result_b == (0.0, 0.0)


# --- margin_headroom -------------------------------------------------------------

def test_margin_headroom_losing_short_shrinks_correctly():
    # A short losing as price rises: maintenance grows bar over bar while
    # equity falls -- headroom should shrink accordingly, using the EXACT
    # per-bar maintenance (no entry-notional optimism).
    result = _make_result(
        total_equity=[1000.0, 800.0, 0.0],
        per_perp_maintenance={"BTC": np.array([100.0, 300.0, 500.0])},
    )
    # bar0: (1000-100)/1000 = 0.9
    # bar1: (800-300)/800 = 0.625
    # bar2: equity<=EPS -> guarded/skipped, never divides by ~0
    value = margin_headroom(result)
    assert value == pytest.approx(0.625)
    assert np.isfinite(value)


def test_margin_headroom_all_bars_guarded_returns_zero():
    result = _make_result(
        total_equity=[0.0, 0.0],
        per_perp_maintenance={"BTC": np.array([10.0, 20.0])},
    )
    assert margin_headroom(result) == 0.0


def test_margin_headroom_no_positions_is_full_headroom():
    result = _make_result(total_equity=[1000.0, 1000.0])
    assert margin_headroom(result) == 1.0


# --- complexity -------------------------------------------------------------------

def _genome_with_params() -> Genome:
    return Genome(
        name="t",
        universe_filter=GeneSpec("all_tradable", {}),
        signal=GeneSpec("rank", {"field_index": 0}),
        entry_timing=GeneSpec("release_bar", {}),
        sizing=GeneSpec("fixed_notional", {"notional_long": 10.0}),
        exit_rule=GeneSpec(
            "bracket", {"sl_pct": 0.05, "tp_pct": None, "expiry_bars": 240}
        ),
    )


def test_complexity_counts_non_none_params_across_all_five_specs():
    # signal: field_index (1) + sizing: notional_long (1)
    # + exit_rule: sl_pct, expiry_bars (2; tp_pct=None excluded) = 4
    assert complexity(_genome_with_params()) == 4


def test_complexity_zero_for_empty_params():
    genome = Genome(
        name="t",
        universe_filter=GeneSpec("all_tradable", {}),
        signal=GeneSpec("rank", {}),
        entry_timing=GeneSpec("release_bar", {}),
        sizing=GeneSpec("fixed_notional", {}),
        exit_rule=GeneSpec("time_only", {}),
    )
    assert complexity(genome) == 0


# --- score_v1: CRITICAL sign-bug regression --------------------------------------

def test_score_v1_liquidation_penalty_additive_with_negative_base():
    # THE sign-bug regression test: liquidation penalty must be additive and
    # sign-independent. A multiplicative penalty on a NEGATIVE base score
    # would make the liquidated genome's score LESS negative (i.e. rank
    # ABOVE the clean genome) -- exactly backwards. Additive subtraction
    # always pushes the liquidated genome strictly below the clean one.
    base = -0.2
    liquidated = score_v1(base, n_liquidated=1, complexity=5, headroom=0.5)
    clean = score_v1(base, n_liquidated=0, complexity=5, headroom=0.5)
    assert liquidated < clean
    assert liquidated == pytest.approx(clean - LIQ_PENALTY)


def test_score_v1_liquidation_penalty_additive_with_positive_base():
    base = 0.3
    liquidated = score_v1(base, n_liquidated=2, complexity=0, headroom=0.5)
    clean = score_v1(base, n_liquidated=0, complexity=0, headroom=0.5)
    assert liquidated < clean
    assert liquidated == pytest.approx(clean - LIQ_PENALTY)


def test_score_v1_complexity_penalty():
    assert score_v1(0.0, n_liquidated=0, complexity=10, headroom=1.0) == pytest.approx(
        -COMPLEXITY_COEF * 10
    )
    assert score_v1(0.0, n_liquidated=0, complexity=0, headroom=1.0) == 0.0


def test_score_v1_headroom_penalty_linear_ramp():
    # at/above the floor: no penalty
    assert score_v1(0.0, 0, 0, headroom=HEADROOM_FLOOR) == 0.0
    assert score_v1(0.0, 0, 0, headroom=1.0) == 0.0
    # halfway to zero from the floor: half the max penalty
    half = score_v1(0.0, 0, 0, headroom=HEADROOM_FLOOR / 2)
    assert half == pytest.approx(-HEADROOM_MAX_PENALTY / 2)
    # at/below zero: full max penalty, clamped (no larger penalty for very negative headroom)
    at_zero = score_v1(0.0, 0, 0, headroom=0.0)
    very_negative = score_v1(0.0, 0, 0, headroom=-5.0)
    assert at_zero == pytest.approx(-HEADROOM_MAX_PENALTY)
    assert very_negative == pytest.approx(-HEADROOM_MAX_PENALTY)


def test_score_v1_combines_all_penalties():
    base = -0.2
    score = score_v1(base, n_liquidated=1, complexity=10, headroom=0.0)
    expected = base - LIQ_PENALTY - COMPLEXITY_COEF * 10 - HEADROOM_MAX_PENALTY
    assert score == pytest.approx(expected)
