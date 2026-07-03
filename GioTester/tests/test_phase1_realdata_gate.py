"""Phase 1 Task P0 -- real-data seam gate.

Plan: docs/superpowers/plans/2026-07-03-phase1-batch-eval.md, Task P0.

Macro-plan originally named a "legacy SLTP_Bracket vs genome allclose" gate;
that is stale now that Phase E deliberately diverged genome exits (v2 TRIGGER
placements) from legacy exits (v1 market reduce-only) -- see
test_real_data_smoke.py and test_genome_equivalence.py's divergence canary.
Equity curves are EXPECTED to differ between the two paths. This module
replaces that stale gate with the honest end-to-end seam checks:

1. the genome-built trader completes a real-data backtest cleanly through the
   real v2 pipeline (finite equity, stamped engine version, at least one fill).
2. the whole pipeline is deterministic on real data (same genome, same data,
   twice -> identical equity curve AND identical execution-event stream). This
   is the load-bearing gate: everything downstream (Phase 1 scoring/digests)
   assumes deterministic replay.
3. entry logic itself is unchanged by v2 (only exits diverge by design): the
   genome and the legacy `SLTP_Bracket.py` open the same (asset, side) set on
   the first signal release.

Skips outright if `../data/*.csv` is absent. Marked `slow`: loads and replays
real OHLC/funding/oracle CSVs. Run `python -m pytest -m "not slow"` to skip
explicitly even when data is present.

Runtime note: the raw CSVs span ~2025-06-05..2026-05-09 and the default
`DataConfig` window (start_date + holding_days, stop_at_last_signal_date)
would clip to the *entire* remaining ~7-month tail of that range (holding_days
pushes the exclusive end past the data's actual max, so it's a no-op clip).
Five backtests run across the three tests below; over the full ~7-month/~160
perp window that risks minutes of wall-clock time per run. To stay well under
budget this module explicitly narrows the window to one month
(2025-10-10..2025-11-09, ~720 bars) via `DataConfig.end_date`, which still
covers 30 daily signal releases -- ample for the determinism and
entry-parity checks below.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT.parent / "data"
REQUIRED = [
    "perps_prices_1h_ohlc.csv",
    "all_perps_hourly_funding.csv",
    "oracle_price.csv",
    "ranks_crowdcent.csv",
]
DATA_PRESENT = DATA_DIR.is_dir() and all((DATA_DIR / f).exists() for f in REQUIRED)

from src import BacktestConfig, DataConfig, DataLoader, load_trader  # noqa: E402
from src.data_prep import SimData, prepare_sim_data  # noqa: E402
from src.genome import GeneSpec, Genome, build_trader  # noqa: E402
from src.runner import run_backtest_prepared  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DATA_PRESENT, reason="../data/*.csv not present"),
]

_DATA_CFG = DataConfig(
    ranks_path=DATA_DIR / "ranks_crowdcent.csv",
    ohlc_path=DATA_DIR / "perps_prices_1h_ohlc.csv",
    funding_path=DATA_DIR / "all_perps_hourly_funding.csv",
    oracle_path=DATA_DIR / "oracle_price.csv",
    rank_source="crowdcent",
    n_long=2,
    n_short=2,
    start_date="2025-10-10",
    end_date="2025-11-09",  # narrowed window -- see module docstring runtime note
)

_BT_CFG = BacktestConfig(initial_equity=2000.0, n=2, leverage=1.0, taker_fee_bps=4.5)


@pytest.fixture(scope="module")
def sd() -> SimData:
    """Load real CSVs once and share the prepared SimData across all tests.

    SimData is read-only from run_backtest_prepared's perspective (dense
    ndarrays indexed by bar, never mutated in place), so reusing one instance
    across independent backtest runs -- including the two same-genome runs in
    the determinism test -- does not leak state between runs.
    """
    loader = DataLoader(_DATA_CFG)
    pred_result = loader.load_predictions()
    market = loader.load_market_data(pred_result.metadata)
    return prepare_sim_data(market, str(_DATA_CFG.ranks_path))


def _bracket_genome() -> Genome:
    return Genome(
        name="BracketSLTP",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec(
            "fixed_notional",
            {
                "notional_long": 10.0,
                "notional_short": 10.0,
                "min_notional_usd": 10.0,
                "leverage": 1.0,
            },
        ),
        exit_rule=GeneSpec(
            "bracket",
            {
                "sl_pct": 0.05,
                "tp_pct": 0.10,
                "expiry_bars": 240,
                "min_notional_usd": 10.0,
                "leverage": 1.0,
            },
        ),
        n_long=2,
        n_short=2,
    )


def _load_legacy_bracket():
    return load_trader(
        str(ROOT / "Traders" / "SLTP_Bracket.py"),
        n=2,
        leverage=1.0,
        notional_long=10.0,
        notional_short=10.0,
        min_notional_usd=_BT_CFG.min_notional_usd,
        taker_fee_bps=_BT_CFG.taker_fee_bps,
        bars_per_day=_BT_CFG.bars_per_day,
        blackout_days_end=_BT_CFG.blackout_days_end,
    )


def _event_tuples(events) -> List[Tuple]:
    return [
        (e.timestamp, e.event_type, e.asset, e.side, e.fill_price, e.reason)
        for e in events
    ]


def test_bracket_genome_completes_on_real_data(sd: SimData) -> None:
    trader = build_trader(_bracket_genome())
    result = run_backtest_prepared(trader, sd, _BT_CFG, verbose=False)

    assert result is not None
    assert math.isfinite(result.total_equity[-1])
    assert result.engine_semantics_version == 2
    assert len(result.execution_events) >= 1


def test_real_data_determinism(sd: SimData) -> None:
    # Fresh trader + fresh EntryLedger state each run -- genes are stateful
    # (delay/momentum caches, EntryLedger), so this actually exercises replay
    # determinism rather than reusing warm state from a prior run.
    trader_a = build_trader(_bracket_genome())
    result_a = run_backtest_prepared(trader_a, sd, _BT_CFG, verbose=False)

    trader_b = build_trader(_bracket_genome())
    result_b = run_backtest_prepared(trader_b, sd, _BT_CFG, verbose=False)

    assert np.array_equal(result_a.total_equity, result_b.total_equity)
    assert _event_tuples(result_a.execution_events) == _event_tuples(
        result_b.execution_events
    )


def test_entry_parity_with_legacy_first_release(sd: SimData) -> None:
    """Entry logic is unchanged by v2 (surviving equivalence); exits diverge
    by design (genome emits TRIGGER placements, legacy emits market reduce-only
    closes -- see module docstring / test_real_data_smoke.py). This test only
    compares ENTRY events, and only near the first release, so exit divergence
    later in the run cannot affect it.
    """
    genome_trader = build_trader(_bracket_genome())
    genome_result = run_backtest_prepared(genome_trader, sd, _BT_CFG, verbose=False)

    legacy_trader = _load_legacy_bracket()
    legacy_result = run_backtest_prepared(legacy_trader, sd, _BT_CFG, verbose=False)

    # `sd.ranks_by_release` is keyed off the WHOLE ranks CSV (2025-06-05 on),
    # independent of the clipped market timeline -- most of those release days
    # predate `sd.timeline[0]` and can never fire (no bar exists for them).
    # Restrict to release days that actually fall inside the backtest window.
    window_start, window_end_ts = sd.timeline[0], sd.timeline[-1]
    release_days = sorted(
        d for d in sd.ranks_by_release if window_start <= d <= window_end_ts
    )
    assert len(release_days) >= 2, "need >=2 in-window release days for the widen fallback"

    def _opened_asset_sides(events, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        out: Dict[Tuple[str, int], pd.Timestamp] = {}
        for e in events:
            if e.event_type != "position_opened":
                continue
            ts = pd.Timestamp(e.timestamp)
            if start_ts <= ts < end_ts:
                out.setdefault((e.asset, e.side), ts)
        return set(out.keys())

    # Entries fill one bar after the release-bar decision (R1 pipeline), so a
    # 48-bar window comfortably covers the first release's fills.
    window_end = release_days[0] + pd.Timedelta(hours=48)
    genome_entries = _opened_asset_sides(
        genome_result.execution_events, release_days[0], window_end
    )
    legacy_entries = _opened_asset_sides(
        legacy_result.execution_events, release_days[0], window_end
    )

    if not genome_entries and not legacy_entries:
        # First release didn't yield >=2 rankable candidates on either path
        # (select_longs_shorts' two-sided floor / legacy's len(cands)<2 guard)
        # -- widen to the first 2 releases per the task's documented fallback.
        window_end = release_days[1] + pd.Timedelta(hours=48)
        genome_entries = _opened_asset_sides(
            genome_result.execution_events, release_days[0], window_end
        )
        legacy_entries = _opened_asset_sides(
            legacy_result.execution_events, release_days[0], window_end
        )

    assert genome_entries, "expected at least one entry near the first release(s)"
    assert genome_entries == legacy_entries
