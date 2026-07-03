"""M1 closure: genome-built trader driven through the real engine v2 pipeline.

Plan: docs/superpowers/plans/2026-07-03-phaseT-taxonomy-v1.md, Task T1.
Gene contract: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md R7.

Builds a bracket genome (rank signal, release_bar timing, fixed_notional sizing,
bracket exit) via `src.genome.build_trader` and drives it through
`run_backtest_prepared` on synthetic `SimData` (engine_harness). Verifies the
whole pipeline end-to-end: release-bar entry decision -> next-open fill (R1) ->
one-bar-gap trigger placement/activation (R2) -> stop-loss fire/fill (R3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.genome.library  # noqa: E402,F401
from src.config import BacktestConfig  # noqa: E402
from src.genome.adapter import GeneSpec, Genome, build_trader  # noqa: E402
from src.runner import run_backtest_prepared  # noqa: E402
from tests.engine_harness import build_sim_data  # noqa: E402


def _bracket_genome(**overrides) -> Genome:
    kwargs = dict(
        name="GenomeBracketIntegration",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec(
            "fixed_notional",
            {"notional_long": 10.0, "notional_short": 10.0, "min_notional_usd": 10.0, "leverage": 1.0},
        ),
        exit_rule=GeneSpec(
            "bracket",
            {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0, "leverage": 1.0},
        ),
        n_long=1,
        n_short=0,
    )
    kwargs.update(overrides)
    return Genome(**kwargs)


def _exec_at(result, ts: str):
    return [e for e in result.execution_events if e.timestamp == ts]


def test_bracket_genome_trigger_fires_one_bar_after_placement():
    # bar0 (midnight UTC, release bar) -> entry queues; bar1 -> fills open @100,
    # gene places SL/TP triggers (active bar2, one-bar gap R2); bar2 low breaches
    # the 5% stop (95.0) -> trigger fill at bar2, fill_price == trigger_px.
    sd = build_sim_data(
        {
            "AAA": [
                [100, 100.5, 99.5, 100.2],  # bar0: decision bar
                [100, 101, 99, 100],        # bar1: entry fills @ open == 100
                [100, 101, 95, 99],         # bar2 (last bar): low breaches SL @95
            ],
        },
        start="2025-01-01",  # periods start at midnight UTC -> bar0.hour == 0
    )
    bar0_day = sd.timeline[0].normalize()
    assert sd.timeline[0].hour == 0
    sd.ranks_by_release = {bar0_day: {"AAA": (0.9, 0.0)}}

    trader = build_trader(_bracket_genome())
    cfg = BacktestConfig(initial_equity=2000.0)
    result = run_backtest_prepared(trader, sd, cfg, verbose=False)

    def _iso(k: int) -> str:
        return result.timeline[k].isoformat()

    # No trigger fill at bar1: the gene only places triggers on bar1 (once the
    # position exists); R2 activation requires a bar of separation.
    bar1_triggers = [e for e in _exec_at(result, _iso(1)) if e.reason == "trigger"]
    assert bar1_triggers == []

    bar2_triggers = [e for e in _exec_at(result, _iso(2)) if e.reason == "trigger"]
    assert len(bar2_triggers) == 1
    ev = bar2_triggers[0]
    assert ev.asset == "AAA"
    assert ev.fill_price == pytest.approx(95.0)
