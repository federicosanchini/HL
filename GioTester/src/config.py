from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    # --- Input paths ---
    ranks_path: Path
    ohlc_path: Path
    funding_path: Path
    oracle_path: Path

    # --- Prediction parsing ---
    rank_source: str = "auto"  # "auto" | "numerai" | "crowdcent"
    prediction_col: Optional[str] = None
    symbol_col: Optional[str] = None
    date_col: Optional[str] = None
    min_prediction_rows_per_day: int = 2

    # --- Date window ---
    # Market data is clipped to [start_date, end_exclusive).
    # end_exclusive = last signal date + holding_days when stop_at_last_signal_date=True.
    start_date: Optional[str] = "2025-10-10"
    end_date: Optional[str] = None
    holding_days: Optional[float] = 30.0
    stop_at_last_signal_date: bool = True

    # --- Signal selection ---
    n_long: int = 3
    n_short: int = 3

    # --- Trade scheduling ---
    entry_hour_utc: int = 0
    max_entry_delay_hours: int = 23
    execution_price_col: str = "open"
    notional_per_trade: float = 10.0


@dataclass
class BacktestConfig:
    """Runtime knobs for the hourly simulator."""
    initial_equity: float = 2000.0
    n: int = 3                       # top-N long, bottom-N short
    leverage: float = 1.0
    notional_per_trade: float = 10.0 # USD per leg at entry
    taker_fee_bps: float = 4.5       # 0.045%
    liquidation_fee_frac: float = 0.01   # 1% of remaining margin → insurance
    min_notional_usd: float = 10.0
    blackout_days_end: int = 50      # block entries in last N days
    bars_per_day: int = 24
    annualization: int = 8760
