"""
Backtester entry-point. Runs HODL_10 then HODL_30, prints metric tables.

Confidence: HIGH (orchestration only)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import BacktestConfig, DataConfig  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.simulator import run_backtest  # noqa: E402
from src.strategies import HODL_10, HODL_30  # noqa: E402

# ——— constants ———
INITIAL_EQUITY = 2000.0
N = 2
LEVERAGE = 1.0
NOTIONAL = 11.0
FEE_BPS = 4.5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    cfg = DataConfig(
        ranks_path=DATA_DIR / "ranks_crowdcent.csv",
        ohlc_path=DATA_DIR / "perps_prices_1h_ohlc.csv",
        funding_path=DATA_DIR / "all_perps_hourly_funding.csv",
        oracle_path=DATA_DIR / "oracle_price.csv",
        rank_source="crowdcent",
        n_long=N,
        n_short=N,
        notional_per_trade=NOTIONAL,
    )

    loader = DataLoader(cfg)
    pred_result = loader.load_predictions()
    market = loader.load_market_data(pred_result.metadata)

    bt_cfg = BacktestConfig(
        initial_equity=INITIAL_EQUITY,
        n=N,
        leverage=LEVERAGE,
        notional_per_trade=NOTIONAL,
        taker_fee_bps=FEE_BPS,
    )

    strategies = [
        HODL_10(
            n=N,
            leverage=LEVERAGE,
            notional_per_trade=NOTIONAL,
            taker_fee_bps=FEE_BPS,
            min_notional_usd=bt_cfg.min_notional_usd,
            blackout_days_end=bt_cfg.blackout_days_end,
            bars_per_day=bt_cfg.bars_per_day,
        ),
        HODL_30(
            n=N,
            leverage=LEVERAGE,
            notional_per_trade=NOTIONAL,
            taker_fee_bps=FEE_BPS,
            min_notional_usd=bt_cfg.min_notional_usd,
            blackout_days_end=bt_cfg.blackout_days_end,
            bars_per_day=bt_cfg.bars_per_day,
        ),
    ]

    for strat in strategies:
        run_backtest(strat, market, str(cfg.ranks_path), bt_cfg, verbose=True)


if __name__ == "__main__":
    main()
