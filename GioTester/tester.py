"""
Backtester entry-point. Runs HODL10_reset, HODL30_reset, gap_HODL10, prints metric tables.

Confidence: HIGH (orchestration only)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.config import BacktestConfig, DataConfig  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.simulator import run_backtest, log_results  # noqa: E402
import src.strategies as strat

# ——— constants ———
INITIAL_EQUITY = 2000.0
N = 3
LEVERAGE = 1.0
NOTIONAL_LONG = 10.0
NOTIONAL_SHORT = 10.0
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
        strat.HODL10(**_shared),
        strat.HODL30(**_shared),
        strat.HODL10_reset(**_shared),
        strat.HODL30_reset(**_shared),
        strat.HODL10_exp(**_shared),
        strat.HODL30_exp(**_shared),
    ]

    for strategy in strategies:
        result = run_backtest(
            strategy, market, str(cfg.ranks_path), bt_cfg, verbose=True
        )
        out = RESULTS_DIR / f"{strategy.name}.json"
        log_results(result, str(out))
        print(f"  → saved {out}")
        net_pnl = (
            result.long_pnl + result.short_pnl + result.funding_pnl - result.total_fees
        )
        print(
            f"  PnL breakdown:"
            f"  long={result.long_pnl:+.4f}"
            f"  short={result.short_pnl:+.4f}"
            f"  funding={result.funding_pnl:+.4f}"
            f"  fees_paid={result.total_fees:.4f}"
            f"  net={net_pnl:+.4f}"
        )


if __name__ == "__main__":
    main()
