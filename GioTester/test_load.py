"""
Quick pipeline smoke test. Run from GioTester/:
    python main.py
"""

from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd

from src.config import DataConfig
from src.data_loader import DataLoader, TICKER_SET

DATA = Path(__file__).parent.parent / "data"

CONFIGS = {
    "crowdcent": DataConfig(
        ranks_path=DATA / "ranks_crowdcent.csv",
        ohlc_path=DATA / "perps_prices_1h_ohlc.csv",
        funding_path=DATA / "all_perps_hourly_funding.csv",
        oracle_path=DATA / "oracle_price.csv",
        rank_source="crowdcent",
    ),
    "numerai": DataConfig(
        ranks_path=DATA / "ranks_numerai.csv",
        ohlc_path=DATA / "perps_prices_1h_ohlc.csv",
        funding_path=DATA / "all_perps_hourly_funding.csv",
        oracle_path=DATA / "oracle_price.csv",
        rank_source="numerai",
    ),
}


def _print_list(label: str, items: List[str], limit: int = 20) -> None:
    if not items:
        print(f"  {label}: none")
        return
    shown = items[:limit]
    tail = f"  ... +{len(items) - limit} more" if len(items) > limit else ""
    print(f"  {label} ({len(items)}): {', '.join(shown)}{tail}")


def check_missing(
    name: str,
    pred_result,
    market,
    signals: pd.DataFrame,
    schedule: pd.DataFrame,
    loader: DataLoader,
) -> None:
    m = pred_result.metadata
    print(f"\n--- missing / gaps: {name} ---")

    # 1. Raw symbols that couldn't be mapped to any HL ticker.
    _print_list(
        "raw symbols unmatched to HL universe",
        m.get("unmatched_raw_symbols_sample", []),
    )

    # 2. HL tickers present in universe but absent from predictions entirely.
    _print_list(
        "HL tickers with no predictions",
        m.get("hl_symbols_missing_from_predictions_sample", []),
    )

    # 3. Prediction dates skipped (too few rows for min_prediction_rows_per_day).
    skipped_dates = sorted(
        str(d.date()) if hasattr(d, "date") else str(d)
        for d, count in loader.daily_signal_counts.items()
        if count == 0
    )
    _print_list("signal dates skipped (too few prediction rows)", skipped_dates)

    # 4. Symbols selected in signals but missing from OHLC.
    ohlc_syms = set(market.ohlc["perp"].unique())
    signal_syms = set(signals["symbol"].unique()) if not signals.empty else set()
    missing_ohlc = sorted(signal_syms - ohlc_syms)
    _print_list("signal symbols missing from OHLC", missing_ohlc)

    # 5. Symbols selected in signals but missing from funding.
    funding_syms = set(market.funding["perp"].unique())
    missing_funding = sorted(signal_syms - funding_syms)
    _print_list("signal symbols missing from funding", missing_funding)

    # 6. Symbols selected in signals but missing from oracle.
    oracle_syms = set(market.oracle["perp"].unique())
    missing_oracle = sorted(signal_syms - oracle_syms)
    _print_list("signal symbols missing from oracle", missing_oracle)

    # 7. Skipped signals by reason (from build_trade_schedule).
    if loader.skipped_signals:
        reason_counts = Counter(s["reason"] for s in loader.skipped_signals)
        for reason, count in sorted(reason_counts.items()):
            affected = sorted(
                {s["symbol"] for s in loader.skipped_signals if s["reason"] == reason}
            )
            _print_list(f"skipped [{reason}] x{count}", affected)
    else:
        print("  skipped signals: none")

    # 8. Signal dates that produced signals but zero scheduled trades.
    scheduled_dates = (
        set(schedule["signal_date"].unique()) if not schedule.empty else set()
    )
    signal_dates = set(signals["signal_date"].unique()) if not signals.empty else set()
    dates_no_trades = sorted(
        str(d.date()) if hasattr(d, "date") else str(d)
        for d in signal_dates - scheduled_dates
    )
    _print_list("signal dates with signals but no scheduled trades", dates_no_trades)


def run_pipeline(name: str, cfg: DataConfig) -> None:
    print(f"\n{'='*60}")
    print(f"  Source: {name}")
    print(f"{'='*60}")

    loader = DataLoader(cfg)
    pred_result, market, signals, schedule = loader.load_all()

    m = pred_result.metadata
    print(
        f"[predictions]  rows={m['rows_final']}  "
        f"dates={m['signal_date_min']} → {m['signal_date_max']}  "
        f"pred_col={m['prediction_col_used']}"
    )

    print(
        f"[ohlc]         rows={len(market.ohlc)}  "
        f"symbols={market.ohlc['perp'].nunique()}  "
        f"time={market.ohlc['time'].min()} → {market.ohlc['time'].max()}"
    )

    print(f"[funding]      rows={len(market.funding)}")
    print(f"[oracle]       rows={len(market.oracle)}")

    print(
        f"[signals]      rows={len(signals)}  "
        f"dates={signals['signal_date'].nunique()}  "
        f"longs={(signals['side'] == 1).sum()}  "
        f"shorts={(signals['side'] == -1).sum()}"
    )

    print(
        f"[schedule]     rows={len(schedule)}  "
        f"skipped={len(loader.skipped_signals)}"
    )

    check_missing(name, pred_result, market, signals, schedule, loader)

    assert len(pred_result.ranks) > 0, "ranks empty"
    assert len(market.ohlc) > 0, "ohlc empty"
    assert len(market.funding) > 0, "funding empty"
    assert len(market.oracle) > 0, "oracle empty"
    assert len(signals) > 0, "signals empty"
    assert len(schedule) > 0, "schedule empty"

    print("\n  OK")


if __name__ == "__main__":
    results = {}
    for name, cfg in CONFIGS.items():
        run_pipeline(name, cfg)
        results[name] = DataLoader(cfg).load_predictions()

    print("\nAll pipelines passed.")
