"""Real-data smoke gate (Task 8 / spec R8).

Not an equivalence test: legacy Traders emit v1 market reduce-only exits while
genome exit genes emit v2 TRIGGER placements (R7) -- their P&L curves are
expected to diverge under v2 (see test_genome_equivalence.py's divergence
canary). This test only proves both paths complete cleanly through the real
v2 pipeline on real data: finite cash, matching bar counts, and the
engine_semantics_version stamp.

Skips outright if `../data/*.csv` is absent (CI-less / data-less machines).
Marked `slow`: loads and replays the full real OHLC/funding/oracle CSVs,
which can take real wall-clock time. Run `python -m pytest -m "not slow"` to
skip it explicitly even when data is present.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

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

from src import BacktestConfig, DataConfig, DataLoader, load_trader, run_backtest  # noqa: E402
from src.genome import Genome, GeneSpec, build_trader  # noqa: E402


def _genome_bracket() -> Genome:
    return Genome(
        name="BracketSLTP",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional", {"notional_long": 10.0, "notional_short": 10.0,
                                           "min_notional_usd": 10.0, "leverage": 1.0}),
        exit_rule=GeneSpec("bracket", {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240,
                                       "min_notional_usd": 10.0, "leverage": 1.0}),
        n_long=2,
        n_short=2,
    )


@pytest.mark.slow
@pytest.mark.skipif(not DATA_PRESENT, reason="../data/*.csv not present")
def test_genome_and_legacy_complete_on_real_data():
    cfg = DataConfig(
        ranks_path=DATA_DIR / "ranks_crowdcent.csv",
        ohlc_path=DATA_DIR / "perps_prices_1h_ohlc.csv",
        funding_path=DATA_DIR / "all_perps_hourly_funding.csv",
        oracle_path=DATA_DIR / "oracle_price.csv",
        rank_source="crowdcent",
        n_long=2,
        n_short=2,
    )
    loader = DataLoader(cfg)
    pred_result = loader.load_predictions()
    market = loader.load_market_data(pred_result.metadata)

    bt_cfg = BacktestConfig(initial_equity=2000.0, n=2, leverage=1.0, taker_fee_bps=4.5)

    legacy = load_trader(
        str(ROOT / "Traders" / "SLTP_Bracket.py"),
        n=2,
        leverage=1.0,
        notional_long=10.0,
        notional_short=10.0,
        min_notional_usd=bt_cfg.min_notional_usd,
        taker_fee_bps=bt_cfg.taker_fee_bps,
        bars_per_day=bt_cfg.bars_per_day,
        blackout_days_end=bt_cfg.blackout_days_end,
    )
    genome_trader = build_trader(_genome_bracket())

    legacy_result = run_backtest(
        legacy, market, str(cfg.ranks_path), bt_cfg, verbose=False
    )
    genome_result = run_backtest(
        genome_trader, market, str(cfg.ranks_path), bt_cfg, verbose=False
    )

    assert math.isfinite(legacy_result.total_equity[-1])
    assert math.isfinite(genome_result.total_equity[-1])
    assert legacy_result.engine_semantics_version == 2
    assert genome_result.engine_semantics_version == 2
    assert len(legacy_result.timeline) == len(genome_result.timeline)
