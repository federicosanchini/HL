"""
Quick data-loader smoke test. Run from GioTester/:
    python test_load.py
"""

from pathlib import Path
from typing import List

from src.config import DataConfig
from src.data_loader import DataLoader

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


def run_pipeline(name: str, cfg: DataConfig) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Source: {name}")
    print(f"{'=' * 60}")

    loader = DataLoader(cfg)
    pred_result = loader.load_predictions()
    market = loader.load_market_data(pred_result.metadata)
    signals = loader.create_daily_signals(pred_result.ranks)
    m = pred_result.metadata

    print(
        f"[predictions] rows={m['rows_final']} "
        f"dates={m['signal_date_min']} -> {m['signal_date_max']} "
        f"pred_col={m['prediction_col_used']}"
    )
    print(
        f"[ohlc]        rows={len(market.ohlc)} "
        f"symbols={market.ohlc['perp'].nunique()} "
        f"time={market.ohlc['time'].min()} -> {market.ohlc['time'].max()}"
    )
    print(f"[funding]     rows={len(market.funding)}")
    print(f"[oracle]      rows={len(market.oracle)}")
    print(
        f"[signals]     rows={len(signals)} "
        f"dates={signals['signal_date'].nunique() if not signals.empty else 0}"
    )

    _print_list(
        "raw symbols unmatched to HL universe",
        m.get("unmatched_raw_symbols_sample", []),
    )
    skipped_dates = sorted(
        str(d.date()) if hasattr(d, "date") else str(d)
        for d, count in loader.daily_signal_counts.items()
        if count == 0
    )
    _print_list("signal dates skipped", skipped_dates)

    assert len(pred_result.ranks) > 0, "ranks empty"
    assert len(market.ohlc) > 0, "ohlc empty"
    assert len(market.funding) > 0, "funding empty"
    assert len(market.oracle) > 0, "oracle empty"
    print("\n  OK")


if __name__ == "__main__":
    for source_name, source_cfg in CONFIGS.items():
        run_pipeline(source_name, source_cfg)
    print("\nAll loader checks passed.")
