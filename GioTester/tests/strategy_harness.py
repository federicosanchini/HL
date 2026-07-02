"""Test helpers: build frozen StrategyState fixtures and load trader files."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_trader  # noqa: E402
from src.dto import (  # noqa: E402
    AccountView,
    MarketAssetView,
    PositionView,
    StrategyState,
)

TRADERS = ROOT / "Traders"

SHARED_KWARGS = dict(
    n=2,
    leverage=1.0,
    notional_long=10.0,
    notional_short=10.0,
    taker_fee_bps=4.5,
    min_notional_usd=10.0,
    blackout_days_end=0,
    bars_per_day=24,
)


def load(filename: str, **overrides):
    kwargs = dict(SHARED_KWARGS)
    kwargs.update(overrides)
    return load_trader(str(TRADERS / filename), **kwargs)


def market_view(asset: str, mark_px: float, *, trade_px=None, funding_rate=0.0,
                 high_px=None, low_px=None):
    px = float(mark_px)
    return MarketAssetView(
        asset=asset,
        trade_px=float(px if trade_px is None else trade_px),
        mark_px=px,
        oracle_px=px,
        funding_rate=float(funding_rate),
        high_px=float(px if high_px is None else high_px),
        low_px=float(px if low_px is None else low_px),
    )


def position_view(asset: str, size: float, entry_price: float, mark_price: float):
    size = float(size)
    entry_price = float(entry_price)
    mark_price = float(mark_price)
    return PositionView(
        asset=asset,
        size=size,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=size * (mark_price - entry_price),
        notional_at_mark=abs(size) * mark_price,
        signed_invested_notional=size * entry_price,
        cumulative_funding=0.0,
        cumulative_fees=0.0,
    )


def make_state(*, bar_index=0, total_bars=100000, market=None, positions=None,
               ranks=None, is_release=False, timestamp=None):
    return StrategyState(
        timestamp=timestamp or pd.Timestamp("2025-10-10", tz="UTC"),
        bar_index=bar_index,
        total_bars=total_bars,
        market=market or {},
        positions=positions or {},
        account=AccountView(
            cash=2000.0,
            equity=2000.0,
            initial_margin_required=0.0,
            maintenance_margin_required=0.0,
            available_balance=2000.0,
        ),
        current_ranks_row=ranks,
        is_release_bar=is_release,
    )
