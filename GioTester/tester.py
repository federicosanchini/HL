"""
Backtester entry-point. Runs HODL_10 then gap_HODL10, prints metric tables.

Confidence: HIGH (orchestration only)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import BacktestConfig, DataConfig  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.simulator import run_backtest, log_results  # noqa: E402
from src.strategies import HODL_10, HODL_30, HODL_combined, gap_HODL10  # noqa: E402

# ——— constants ———
INITIAL_EQUITY = 2000.0
N = 2
N_long = 2
N_short = 2
EXPIRY_DAYS = 20
LEVERAGE = 1.0
NOTIONAL_LONG = 11.0
NOTIONAL_SHORT = 11.0
FEE_BPS = 4.5
GAP_BPS = 50.0

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    cfg = DataConfig(
        ranks_path=DATA_DIR / "ranks_crowdcent.csv",
        ohlc_path=DATA_DIR / "perps_prices_1h_ohlc.csv",
        funding_path=DATA_DIR / "all_perps_hourly_funding.csv",
        oracle_path=DATA_DIR / "oracle_price.csv",
        rank_source="crowdcent",
        n_long=N,
        n_short=N,
    )

    loader = DataLoader(cfg)
    pred_result = loader.load_predictions()
    market = loader.load_market_data(pred_result.metadata)

    bt_cfg = BacktestConfig(
        initial_equity=INITIAL_EQUITY,
        n=N,
        leverage=LEVERAGE,
        taker_fee_bps=FEE_BPS,
    )

    _shared = dict(
        n=N,
        leverage=LEVERAGE,
        notional_long=NOTIONAL_LONG,
        notional_short=NOTIONAL_SHORT,
        taker_fee_bps=FEE_BPS,
        min_notional_usd=bt_cfg.min_notional_usd,
        blackout_days_end=bt_cfg.blackout_days_end,
        bars_per_day=bt_cfg.bars_per_day,
    )

    strategies = [
        HODL_10(**_shared),
        gap_HODL10(gap_bps=GAP_BPS, **_shared),
    ]

    # ——— commented-out alternatives ———
    # HODL_30(**_shared),
    # HODL_combined(
    #     expiry_days=EXPIRY_DAYS, n_long=N_long, n_short=N_short,
    #     **{k: v for k, v in _shared.items() if k != "n"},
    # ),

    for strat in strategies:
        result = run_backtest(strat, market, str(cfg.ranks_path), bt_cfg, verbose=True)
        out = RESULTS_DIR / f"{strat.name}.json"
        log_results(result, str(out))
        print(f"  → saved {out}")


if __name__ == "__main__":
    main()
