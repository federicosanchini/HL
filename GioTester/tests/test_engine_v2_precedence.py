"""Task 5 (Phase E engine v2): liquidation precedence scan + deferred trigger execution.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R5
(conservative breach scan marking one position at its adverse extreme and all
others at open; deferred execution of suppressed-but-fired triggers on assets
that survived the close-mark liquidation pass) + R3's NaN fallback note.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution import EPS  # noqa: E402
from src.position import OrderCommand, OrderType, Position  # noqa: E402
from src.state import MarketState, RestingTrigger, StateBucket, _PerpRowView  # noqa: E402
from src.triggers import (  # noqa: E402
    auto_cancel_triggers,
    execute_deferred_triggers,
    precedence_scan,
)


FEE_BPS = 10.0


def _make_ms(ohlc: dict, bar_index: int = 1) -> MarketState:
    """Build a single-bar MarketState from {asset: [open, high, low, close]}."""
    d = {asset: np.array([row], dtype=float) for asset, row in ohlc.items()}
    ohlc_row = _PerpRowView(d, 0, finite_idx=0)
    funding_row = _PerpRowView({}, 0)
    oracle_d = {asset: np.array([row[0]], dtype=float) for asset, row in ohlc.items()}
    oracle = _PerpRowView(oracle_d, 0)
    return MarketState(
        timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        bar_index=bar_index,
        total_bars=10,
        ohlc_row=ohlc_row,
        funding_row=funding_row,
        oracle=oracle,
        current_ranks_row=None,
    )


def _bucket(ms: MarketState, margin_mode: str = "cross", cash: float = 1000.0) -> StateBucket:
    return StateBucket(market_state=ms, cash=cash, margin_mode=margin_mode)


def _long_position(asset: str = "BTC", size: float = 1.0, entry_price: float = 100.0,
                    mm_rate: float = 0.05, isolated_margin: float = 0.0) -> Position:
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2025-01-01", tz="UTC"),
        leverage=1.0,
        mm_rate=mm_rate,
        isolated_margin=isolated_margin,
        mark_price=entry_price,
    )


def _trigger(asset="BTC", side=-1, size=0.5, trigger_px=90.0, direction="stop", **overrides):
    kwargs = dict(
        asset=asset,
        side=side,
        order_type=OrderType.TRIGGER.value,
        size=size,
        reduce_only=True,
        trigger_px=trigger_px,
        trigger_direction=direction,
    )
    kwargs.update(overrides)
    return OrderCommand(**kwargs)


def _rest(bucket: StateBucket, asset: str, orders, placed_bar: int = 0) -> None:
    bucket.resting_triggers[asset] = [
        RestingTrigger(order=o, placed_bar=placed_bar) for o in orders
    ]


# --- cross-mode breach at extreme: long position, low breaches maintenance ---

def test_cross_breach_at_extreme_long_low_suppresses():
    ms = _make_ms({"BTC": [100.0, 105.0, 50.0, 100.0]})
    bucket = _bucket(ms, cash=10.0)
    bucket.positions_by_asset["BTC"] = _long_position()
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0)], placed_bar=0)

    result = precedence_scan(bucket, ms)

    assert result == frozenset({"BTC"})


# --- no breach -> empty set ---

def test_cross_no_breach_returns_empty_set():
    ms = _make_ms({"BTC": [100.0, 105.0, 95.0, 100.0]})
    bucket = _bucket(ms, cash=1000.0)
    bucket.positions_by_asset["BTC"] = _long_position()
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0)], placed_bar=0)

    result = precedence_scan(bucket, ms)

    assert result == frozenset()


# --- other positions marked at open in the scan, not close ---

def test_cross_scan_marks_other_positions_at_open_not_close():
    # BTC (candidate, has resting trigger): extreme low=50 alone would breach.
    # ETH (other position, no resting trigger): open=100 (no upnl), but
    # close=1000 would (wrongly) inject a huge unrealized gain that rescues
    # the breach if the scan mistakenly used close instead of open.
    ms = _make_ms({
        "BTC": [100.0, 105.0, 50.0, 100.0],
        "ETH": [100.0, 100.0, 100.0, 1000.0],
    })
    bucket = _bucket(ms, cash=10.0)
    bucket.positions_by_asset["BTC"] = _long_position(asset="BTC")
    bucket.positions_by_asset["ETH"] = _long_position(asset="ETH")
    _rest(bucket, "BTC", [_trigger(asset="BTC", trigger_px=90.0)], placed_bar=0)

    result = precedence_scan(bucket, ms)

    # equity(open) = 10 + (50-100) + (100-100) = -40; maintenance = 2.5 + 5 = 7.5 -> breach.
    # equity(close, wrong) = 10 + (50-100) + (1000-100) = 860; maintenance = 2.5+50=52.5 -> no breach.
    assert result == frozenset({"BTC"})


# --- isolated variant breach ---

def test_isolated_breach_at_extreme_suppresses():
    ms = _make_ms({"BTC": [100.0, 105.0, 50.0, 100.0]})
    bucket = _bucket(ms, margin_mode="isolated", cash=1000.0)
    bucket.positions_by_asset["BTC"] = _long_position(isolated_margin=20.0)
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0)], placed_bar=0)

    result = precedence_scan(bucket, ms)

    # isolated_equity = 20 + (50-100) = -30; maintenance = 50*0.05 = 2.5 -> breach.
    assert result == frozenset({"BTC"})


def test_isolated_no_breach_returns_empty_set():
    ms = _make_ms({"BTC": [100.0, 105.0, 95.0, 100.0]})
    bucket = _bucket(ms, margin_mode="isolated", cash=1000.0)
    bucket.positions_by_asset["BTC"] = _long_position(isolated_margin=20.0)
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0)], placed_bar=0)

    result = precedence_scan(bucket, ms)

    assert result == frozenset()


# --- scan is pure: no mutation of bucket/positions/marks ---

def test_scan_is_pure_no_mutation():
    ms = _make_ms({"BTC": [100.0, 105.0, 50.0, 100.0]})
    bucket = _bucket(ms, cash=10.0)
    pos = _long_position()
    bucket.positions_by_asset["BTC"] = pos
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0)], placed_bar=0)

    cash_before = bucket.cash
    size_before = pos.size
    entry_before = pos.entry_price
    mark_before = pos.mark_price
    triggers_before = list(bucket.resting_triggers["BTC"])

    precedence_scan(bucket, ms)

    assert bucket.cash == cash_before
    assert pos.size == size_before
    assert pos.entry_price == entry_before
    assert pos.mark_price == mark_before
    assert bucket.resting_triggers["BTC"] == triggers_before
    assert "BTC" in bucket.positions_by_asset


# --- determinism: identical set on repeated calls ---

def test_scan_deterministic_across_repeated_calls():
    ms = _make_ms({
        "BTC": [100.0, 105.0, 50.0, 100.0],
        "ETH": [100.0, 100.0, 100.0, 1000.0],
    })
    bucket = _bucket(ms, cash=10.0)
    bucket.positions_by_asset["BTC"] = _long_position(asset="BTC")
    bucket.positions_by_asset["ETH"] = _long_position(asset="ETH")
    _rest(bucket, "BTC", [_trigger(asset="BTC", trigger_px=90.0)], placed_bar=0)

    first = precedence_scan(bucket, ms)
    second = precedence_scan(bucket, ms)

    assert first == second == frozenset({"BTC"})


# --- deferred execution: surviving suppressed asset fires at R3 price ---

def test_deferred_execution_fires_surviving_suppressed_asset():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, cash=1000.0)
    bucket.positions_by_asset["BTC"] = _long_position(size=1.0)
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0, size=1.0)], placed_bar=0)

    execute_deferred_triggers(bucket, ms, FEE_BPS, suppressed=frozenset({"BTC"}))

    assert len(bucket.execution_events) == 1
    event = bucket.execution_events[0]
    assert event.fill_price == 90.0
    assert event.reason == "trigger"
    assert "BTC" not in bucket.positions_by_asset  # fully closed (size matched)


def test_deferred_execution_skips_liquidated_asset():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, cash=1000.0)
    bucket.positions_by_asset["BTC"] = _long_position(size=1.0)
    _rest(bucket, "BTC", [_trigger(trigger_px=90.0, size=1.0)], placed_bar=0)

    # Simulate liquidate_if_needed having fired: position removed, triggers
    # auto-cancelled, exactly as the runner would do before calling this.
    del bucket.positions_by_asset["BTC"]
    auto_cancel_triggers(bucket, "BTC")

    execute_deferred_triggers(bucket, ms, FEE_BPS, suppressed=frozenset({"BTC"}))

    assert bucket.execution_events == []
    assert "BTC" not in bucket.resting_triggers
