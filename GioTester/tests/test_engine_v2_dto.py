"""Task 1 (Phase E engine v2): MarketAssetView high/low + MarketState.bar_range NaN fallback.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R3's NaN
high/low fallback paragraph and R7 (strategy surface).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state import MarketState, _PerpRowView  # noqa: E402

from strategy_harness import market_view  # noqa: E402


def _make_ms(ohlc: dict) -> MarketState:
    """Build a single-bar MarketState from {asset: [open, high, low, close]}."""
    d = {asset: np.array([row], dtype=float) for asset, row in ohlc.items()}
    ohlc_row = _PerpRowView(d, 0, finite_idx=0)
    funding_row = _PerpRowView({}, 0)
    oracle_d = {asset: np.array([row[0]], dtype=float) for asset, row in ohlc.items()}
    oracle = _PerpRowView(oracle_d, 0)
    return MarketState(
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        bar_index=0,
        total_bars=10,
        ohlc_row=ohlc_row,
        funding_row=funding_row,
        oracle=oracle,
        current_ranks_row=None,
    )


def test_market_asset_view_carries_explicit_high_low():
    view = market_view("BTC", 100.0, high_px=110.0, low_px=95.0)
    assert view.high_px == 110.0
    assert view.low_px == 95.0
    assert view.mark_px == 100.0


def test_harness_market_view_defaults_high_low_to_mark():
    view = market_view("BTC", 100.0)
    assert view.high_px == 100.0
    assert view.low_px == 100.0


def test_bar_range_finite_ohlc_returns_high_low():
    ms = _make_ms({"BTC": [100.0, 110.0, 95.0, 105.0]})
    eff_high, eff_low = ms.bar_range("BTC")
    assert eff_high == 110.0
    assert eff_low == 95.0


def test_bar_range_nan_high_falls_back_to_max_open_close():
    ms = _make_ms({"BTC": [100.0, math.nan, 95.0, 105.0]})
    eff_high, eff_low = ms.bar_range("BTC")
    assert eff_high == max(100.0, 105.0)
    assert eff_low == 95.0


def test_bar_range_nan_low_falls_back_to_min_open_close():
    ms = _make_ms({"BTC": [100.0, 110.0, math.nan, 105.0]})
    eff_high, eff_low = ms.bar_range("BTC")
    assert eff_high == 110.0
    assert eff_low == min(100.0, 105.0)


def test_bar_range_nan_close_and_nan_high_falls_back_to_open():
    ms = _make_ms({"BTC": [100.0, math.nan, math.nan, math.nan]})
    eff_high, eff_low = ms.bar_range("BTC")
    assert eff_high == 100.0
    assert eff_low == 100.0


def test_asset_view_populates_high_low_from_bar_range():
    ms = _make_ms({"BTC": [100.0, 110.0, 95.0, 105.0]})
    view = ms.asset_view("BTC")
    assert view.high_px == 110.0
    assert view.low_px == 95.0
