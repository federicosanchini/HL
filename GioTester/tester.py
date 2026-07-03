"""Backtester entry-point. Loads every standalone strategy in traders/."""

from __future__ import annotations

from pathlib import Path

from src import (
    BacktestConfig,
    DataConfig,
    DataLoader,
    log_results,
    print_result_summary,
    run_backtest_from_trader_file,
)

# ——— constants ———
INITIAL_EQUITY = 2000.0
N = 2
LEVERAGE = 1.0
NOTIONAL_LONG = 10.0
NOTIONAL_SHORT = 10.0
FEE_BPS = 4.5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRADERS_DIR = Path(__file__).resolve().parent / "traders"


def discover_trader_files() -> list[Path]:
    files = sorted(
        p for p in TRADERS_DIR.glob("*.py")
        if p.is_file() and not p.name.startswith("_")
    )
    if not files:
        raise FileNotFoundError(f"no strategy files found in {TRADERS_DIR}")
    return files


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

    shared = dict(
        n=N,
        leverage=LEVERAGE,
        notional_long=NOTIONAL_LONG,
        notional_short=NOTIONAL_SHORT,
        taker_fee_bps=FEE_BPS,
        min_notional_usd=bt_cfg.min_notional_usd,
        blackout_days_end=bt_cfg.blackout_days_end,
        bars_per_day=bt_cfg.bars_per_day,
    )

    for trader_path in discover_trader_files():
        result = run_backtest_from_trader_file(
            str(trader_path),
            market,
            str(cfg.ranks_path),
            bt_cfg,
            verbose=True,
            **shared,
        )
        out = RESULTS_DIR / f"{trader_path.stem}.json"
        log_results(result, str(out))
        print_result_summary(result, str(out))


if __name__ == "__main__":
    main()
