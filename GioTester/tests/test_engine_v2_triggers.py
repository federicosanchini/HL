"""Task 4 (Phase E engine v2): trigger fire/fill core.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rules R3
(fire/fill matrix, size clamp, fees, no min-notional on fills, NaN fallback)
and R4 (worst-fill selection + exact tie key).
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

from src.execution import execute_order  # noqa: E402
from src.position import OrderCommand, OrderType, Position  # noqa: E402
from src.state import MarketState, RestingTrigger, StateBucket, _PerpRowView  # noqa: E402
from src.triggers import auto_cancel_triggers, fill_triggers, trigger_fill_px  # noqa: E402


FEE_BPS = 10.0  # 10 bps taker


def _make_ms(ohlc: dict, bar_index: int = 0) -> MarketState:
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


def _bucket(ms: MarketState, position: Position = None, margin_mode: str = "cross") -> StateBucket:
    bucket = StateBucket(market_state=ms, cash=1000.0, margin_mode=margin_mode)
    if position is not None:
        bucket.positions_by_asset[position.asset] = position
    return bucket


def _long_position(asset: str = "BTC", size: float = 1.0, entry_price: float = 100.0,
                    leverage: float = 1.0) -> Position:
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2025-01-01", tz="UTC"),
        leverage=leverage,
        mm_rate=0.05,
        mark_price=entry_price,
    )


def _short_position(asset: str = "BTC", size: float = -1.0, entry_price: float = 100.0,
                     leverage: float = 1.0) -> Position:
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2025-01-01", tz="UTC"),
        leverage=leverage,
        mm_rate=0.05,
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


# --- trigger_fill_px: 4 matrix rows, fire at trigger_px (no gap-through) ---

def test_matrix_sell_stop_fires_at_trigger_px():
    order = _trigger(side=-1, direction="stop", trigger_px=90.0)
    # open=100, low touches 85 (through 90) but open itself doesn't breach -> fill at px
    assert trigger_fill_px(order, open_px=100.0, eff_high=105.0, eff_low=85.0) == 90.0


def test_matrix_sell_tp_fires_at_trigger_px():
    order = _trigger(side=-1, direction="tp", trigger_px=110.0)
    assert trigger_fill_px(order, open_px=100.0, eff_high=115.0, eff_low=95.0) == 110.0


def test_matrix_buy_stop_fires_at_trigger_px():
    order = _trigger(side=1, direction="stop", trigger_px=110.0)
    assert trigger_fill_px(order, open_px=100.0, eff_high=115.0, eff_low=95.0) == 110.0


def test_matrix_buy_tp_fires_at_trigger_px():
    order = _trigger(side=1, direction="tp", trigger_px=90.0)
    assert trigger_fill_px(order, open_px=100.0, eff_high=105.0, eff_low=85.0) == 90.0


# --- gap-through: fill at open, one per direction ---

def test_gap_through_sell_stop_fills_at_open():
    order = _trigger(side=-1, direction="stop", trigger_px=90.0)
    # open already below trigger_px -> gap through, fill at open
    assert trigger_fill_px(order, open_px=85.0, eff_high=95.0, eff_low=80.0) == 85.0


def test_gap_through_sell_tp_fills_at_open():
    order = _trigger(side=-1, direction="tp", trigger_px=110.0)
    assert trigger_fill_px(order, open_px=115.0, eff_high=120.0, eff_low=112.0) == 115.0


def test_gap_through_buy_stop_fills_at_open():
    order = _trigger(side=1, direction="stop", trigger_px=110.0)
    assert trigger_fill_px(order, open_px=115.0, eff_high=120.0, eff_low=112.0) == 115.0


def test_gap_through_buy_tp_fills_at_open():
    order = _trigger(side=1, direction="tp", trigger_px=90.0)
    assert trigger_fill_px(order, open_px=85.0, eff_high=95.0, eff_low=80.0) == 85.0


# --- not fired: range doesn't reach ---

def test_not_fired_stays_resting():
    ms = _make_ms({"BTC": [100.0, 105.0, 95.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, _long_position())
    order = _trigger(side=-1, direction="stop", trigger_px=80.0)  # low=95, never touches 80
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.resting_triggers["BTC"][0].order is order
    assert bucket.execution_events == []


# --- activation delay: placed_bar == bar_index not evaluated ---

def test_activation_delay_placed_same_bar_not_evaluated():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=3)
    bucket = _bucket(ms, _long_position())
    order = _trigger(side=-1, direction="stop", trigger_px=90.0)  # would fire (low=80<=90)
    _rest(bucket, "BTC", [order], placed_bar=3)  # placed this bar -> not active
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.execution_events == []


# --- size clamp ---

def test_size_clamp_fills_min_of_trigger_and_position_size():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    position = _long_position(size=0.4)
    bucket = _bucket(ms, position)
    order = _trigger(side=-1, direction="stop", size=1.0, trigger_px=90.0)
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.execution_events) == 1
    event = bucket.execution_events[0]
    assert event.notional == pytest.approx(0.4 * 90.0)
    assert "BTC" not in bucket.positions_by_asset  # fully closed (clamped size == position size)


# --- position absent: auto-cancelled, no fill event ---

def test_position_absent_auto_cancelled_no_event():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, position=None)
    order = _trigger(side=-1, direction="stop", trigger_px=90.0)
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert "BTC" not in bucket.resting_triggers
    assert bucket.execution_events == []


# --- R4: bracket hits both SL and TP same bar -> only SL fills (worst for trader) ---

def test_r4_bracket_only_sl_fills_long():
    ms = _make_ms({"BTC": [100.0, 120.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, _long_position())
    sl = _trigger(side=-1, direction="stop", trigger_px=90.0, client_id="sl")   # fires, fill 90
    tp = _trigger(side=-1, direction="tp", trigger_px=110.0, client_id="tp")    # fires, fill 110
    _rest(bucket, "BTC", [sl, tp], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.execution_events) == 1
    assert bucket.execution_events[0].fill_price == 90.0  # lowest = worst for long-close
    assert bucket.resting_triggers.get("BTC", []) == []  # tp removed, position fully closed


# --- R4 tie: same fill px, client_id "a" beats None ---

def test_r4_tie_client_id_beats_none():
    # Same fill px for both triggers; distinct sizes let the remaining position
    # size reveal which trigger the selection actually filled.
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, _long_position(size=1.0))
    t_none = _trigger(side=-1, direction="stop", size=0.3, trigger_px=90.0, client_id=None)
    t_a = _trigger(side=-1, direction="stop", size=0.7, trigger_px=90.0, client_id="a")
    _rest(bucket, "BTC", [t_none, t_a], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.execution_events) == 1
    assert bucket.execution_events[0].notional == pytest.approx(0.7 * 90.0)  # "a" (size 0.7) filled
    assert bucket.positions_by_asset["BTC"].size == pytest.approx(0.3)  # 1.0 - 0.7
    assert bucket.resting_triggers["BTC"] == []  # both fired that bar: "a" filled, none cancelled


def test_r4_tie_selection_key_prefers_client_id_over_none_directly():
    from src.triggers import _selection_key
    from src.state import RestingTrigger as RT

    t_none = _trigger(side=-1, direction="stop", trigger_px=90.0, client_id=None)
    t_a = _trigger(side=-1, direction="stop", trigger_px=90.0, client_id="a")
    item_none = (90.0, RT(order=t_none, placed_bar=0), 0)
    item_a = (90.0, RT(order=t_a, placed_bar=0), 1)
    winner = min([item_none, item_a], key=lambda it: _selection_key(it, closing_long=True))
    assert winner[1].order.client_id == "a"


# --- suppress set: skip asset entirely, triggers stay resting ---

def test_suppress_skips_asset_entirely():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, _long_position())
    order = _trigger(side=-1, direction="stop", trigger_px=90.0)  # would fire
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset({"BTC"}))
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.execution_events == []


# --- NaN high fallback: fire decision via max(open, close) ---

def test_nan_high_fallback_used_for_fire_decision():
    ms = _make_ms({"BTC": [100.0, float("nan"), 95.0, 108.0]}, bar_index=1)
    # eff_high = max(open=100, close=108) = 108
    bucket = _bucket(ms, _short_position())
    order = _trigger(side=1, direction="stop", trigger_px=105.0)  # buy-close-short stop: high>=px
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.execution_events) == 1
    assert bucket.execution_events[0].fill_price == 105.0  # eff_high=108 >= 105, no gap (open=100<105)


# --- fee + reason ---

def test_fee_debited_and_reason_is_trigger():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    bucket = _bucket(ms, _long_position(size=1.0))
    order = _trigger(side=-1, direction="stop", size=1.0, trigger_px=90.0)
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    event = bucket.execution_events[0]
    expected_fee = 1.0 * 90.0 * FEE_BPS / 1e4
    assert event.fee == pytest.approx(expected_fee)
    assert event.reason == "trigger"


# --- cross margin variant: compare against execute_order oracle ---

def test_cross_margin_fill_matches_execute_order_oracle():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)

    # Oracle: a plain reduce-only market sell at the same fill price via execute_order.
    oracle_ms = _make_ms({"BTC": [90.0, 90.0, 90.0, 90.0]}, bar_index=1)  # trade_px == fill_px
    oracle_bucket = _bucket(oracle_ms, _long_position(size=1.0), margin_mode="cross")
    oracle_order = OrderCommand(asset="BTC", side=-1, size=1.0, reduce_only=True)
    execute_order(oracle_order, oracle_ms, oracle_bucket, fee_bps=FEE_BPS, min_notional_usd=0.0, mm_rate=0.05)
    oracle_event = oracle_bucket.execution_events[0]

    bucket = _bucket(ms, _long_position(size=1.0), margin_mode="cross")
    trig = _trigger(side=-1, direction="stop", size=1.0, trigger_px=90.0)
    _rest(bucket, "BTC", [trig], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    event = bucket.execution_events[0]

    assert event.fill_price == oracle_event.fill_price
    assert event.fee == pytest.approx(oracle_event.fee)
    assert event.realized_pnl == pytest.approx(oracle_event.realized_pnl)
    assert bucket.cash == pytest.approx(oracle_bucket.cash)
    assert event.event_type == oracle_event.event_type


# --- isolated margin variant: compare against execute_order oracle ---

def test_isolated_margin_fill_matches_execute_order_oracle():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)

    oracle_ms = _make_ms({"BTC": [90.0, 90.0, 90.0, 90.0]}, bar_index=1)
    oracle_bucket = _bucket(oracle_ms, _long_position(size=1.0), margin_mode="isolated")
    oracle_bucket.positions_by_asset["BTC"].isolated_margin = 100.0
    oracle_order = OrderCommand(asset="BTC", side=-1, size=1.0, reduce_only=True)
    execute_order(oracle_order, oracle_ms, oracle_bucket, fee_bps=FEE_BPS, min_notional_usd=0.0, mm_rate=0.05)
    oracle_event = oracle_bucket.execution_events[0]

    bucket = _bucket(ms, _long_position(size=1.0), margin_mode="isolated")
    bucket.positions_by_asset["BTC"].isolated_margin = 100.0
    trig = _trigger(side=-1, direction="stop", size=1.0, trigger_px=90.0)
    _rest(bucket, "BTC", [trig], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    event = bucket.execution_events[0]

    assert event.fill_price == oracle_event.fill_price
    assert event.fee == pytest.approx(oracle_event.fee)
    assert event.realized_pnl == pytest.approx(oracle_event.realized_pnl)
    assert bucket.cash == pytest.approx(oracle_bucket.cash)
    assert event.event_type == oracle_event.event_type


# --- no min-notional check on fills (R3): tiny clamped size still fills ---

def test_no_min_notional_check_on_fill():
    ms = _make_ms({"BTC": [100.0, 105.0, 80.0, 102.0]}, bar_index=1)
    position = _long_position(size=0.001)  # tiny remaining position
    bucket = _bucket(ms, position)
    order = _trigger(side=-1, direction="stop", size=1.0, trigger_px=90.0)  # clamps to 0.001
    _rest(bucket, "BTC", [order], placed_bar=0)
    fill_triggers(bucket, ms, FEE_BPS, suppress=frozenset())
    assert len(bucket.execution_events) == 1
    assert bucket.execution_events[0].notional == pytest.approx(0.001 * 90.0)
