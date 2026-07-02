"""Task 3 (Phase E engine v2): StateBucket pending/resting stores + trigger placement path.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R1
(PendingOrder shape) and R2 (placement/replace/auto-cancel/gates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.position import OrderCommand, OrderType, Position  # noqa: E402
from src.state import MarketState, PendingOrder, RestingTrigger, StateBucket, _PerpRowView  # noqa: E402
from src.triggers import auto_cancel_triggers, place_triggers  # noqa: E402


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


def _bucket(ms: MarketState, position: Position = None) -> StateBucket:
    bucket = StateBucket(market_state=ms, cash=1000.0)
    if position is not None:
        bucket.positions_by_asset[position.asset] = position
    return bucket


def _long_position(asset: str = "BTC", size: float = 1.0, entry_price: float = 100.0) -> Position:
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2025-01-01", tz="UTC"),
        leverage=1.0,
        mm_rate=0.05,
        mark_price=entry_price,
    )


def _short_position(asset: str = "BTC", size: float = -1.0, entry_price: float = 100.0) -> Position:
    return Position(
        asset=asset,
        size=size,
        entry_price=entry_price,
        entry_time=pd.Timestamp("2025-01-01", tz="UTC"),
        leverage=1.0,
        mm_rate=0.05,
        mark_price=entry_price,
    )


def _sl_trigger(asset="BTC", side=-1, size=0.5, trigger_px=90.0, direction="stop", **overrides):
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


# --- PendingOrder / RestingTrigger dataclass shape (R1) ---

def test_pending_order_shape():
    order = OrderCommand(asset="BTC", side=1, size=1.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    p = PendingOrder(order=order, queued_ts=ts, queued_bar=3)
    assert p.order is order
    assert p.queued_ts == ts
    assert p.queued_bar == 3


def test_resting_trigger_shape():
    order = _sl_trigger()
    r = RestingTrigger(order=order, placed_bar=5)
    assert r.order is order
    assert r.placed_bar == 5


def test_statebucket_defaults_empty_stores():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = StateBucket(market_state=ms)
    assert bucket.pending_orders == []
    assert bucket.resting_triggers == {}


# --- Replace rule ---

def test_replace_clears_old_set_when_next_bar_places_one():
    ms0 = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]}, bar_index=0)
    bucket = _bucket(ms0, _long_position())
    t1 = _sl_trigger(trigger_px=90.0)
    t2 = _sl_trigger(trigger_px=80.0, size=0.3)
    place_triggers([t1, t2], ms0, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 2

    ms1 = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]}, bar_index=1)
    bucket.market_state = ms1
    t3 = _sl_trigger(trigger_px=85.0, size=0.2)
    place_triggers([t3], ms1, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.resting_triggers["BTC"][0].order is t3
    assert bucket.resting_triggers["BTC"][0].placed_bar == 1


def test_all_invalid_new_set_leaves_empty_resting_and_rejections():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    # same-side invalid trigger (buy while long -> doesn't oppose)
    bad1 = _sl_trigger(side=1, trigger_px=90.0)
    bad2 = _sl_trigger(side=1, trigger_px=80.0)
    place_triggers([bad1, bad2], ms, bucket, min_notional_usd=1.0)
    assert bucket.resting_triggers["BTC"] == []
    assert len(bucket.rejected_orders) == 2
    for rej in bucket.rejected_orders:
        assert rej.reason == "trigger requires an open opposing position"


def test_market_order_asset_not_in_trigger_orders_leaves_triggers_intact():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    t1 = _sl_trigger(trigger_px=90.0)
    place_triggers([t1], ms, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 1

    # Calling place_triggers with an empty list (as happens when a bar's
    # emitted orders contain a market order for BTC but no trigger for BTC)
    # must not touch BTC's resting set.
    place_triggers([], ms, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.resting_triggers["BTC"][0].order is t1


# --- Placement min-notional gate ---

def test_placement_min_notional_gate_rejects_below_threshold():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position(size=1.0))
    # close[i] = 105.0; size 0.05 -> notional 5.25 < min_notional_usd=10
    order = _sl_trigger(size=0.05, trigger_px=90.0)
    place_triggers([order], ms, bucket, min_notional_usd=10.0)
    assert bucket.resting_triggers["BTC"] == []
    assert len(bucket.rejected_orders) == 1
    assert bucket.rejected_orders[0].reason == "trigger below minimum notional at placement"


def test_placement_min_notional_gate_accepts_at_or_above_threshold():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position(size=1.0))
    # close[i] = 105.0; size 0.5 -> notional 52.5 >= min_notional_usd=10
    order = _sl_trigger(size=0.5, trigger_px=90.0)
    place_triggers([order], ms, bucket, min_notional_usd=10.0)
    assert len(bucket.resting_triggers["BTC"]) == 1
    assert bucket.rejected_orders == []


# --- Side-opposes-position validation ---

def test_side_opposes_rejects_same_side_as_long_position():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    order = _sl_trigger(side=1, trigger_px=90.0)  # buy while long: does not oppose
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert bucket.resting_triggers["BTC"] == []
    assert bucket.rejected_orders[0].reason == "trigger requires an open opposing position"


def test_side_opposes_accepts_sell_against_long_position():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    order = _sl_trigger(side=-1, trigger_px=90.0)  # sell closes long: opposes
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 1


def test_side_opposes_accepts_buy_against_short_position():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _short_position())
    order = _sl_trigger(side=1, trigger_px=110.0)  # buy closes short: opposes
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert len(bucket.resting_triggers["BTC"]) == 1


def test_side_opposes_rejects_when_no_position():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, position=None)
    order = _sl_trigger(side=-1, trigger_px=90.0)
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert bucket.resting_triggers["BTC"] == []
    assert bucket.rejected_orders[0].reason == "trigger requires an open opposing position"


# --- validate_trigger() failures surface as loud rejections too ---

def test_invalid_trigger_shape_rejected_loudly():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    order = _sl_trigger(reduce_only=False)  # violates validate_trigger()
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert bucket.resting_triggers["BTC"] == []
    assert len(bucket.rejected_orders) == 1
    assert "reduce_only" in bucket.rejected_orders[0].reason


# --- auto_cancel_triggers ---

def test_auto_cancel_drops_resting_set():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms, _long_position())
    order = _sl_trigger(trigger_px=90.0)
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert "BTC" in bucket.resting_triggers

    auto_cancel_triggers(bucket, "BTC")
    assert "BTC" not in bucket.resting_triggers


def test_auto_cancel_noop_when_asset_absent():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]})
    bucket = _bucket(ms)
    auto_cancel_triggers(bucket, "ETH")  # must not raise
    assert bucket.resting_triggers == {}


# --- placed_bar recorded correctly ---

def test_placed_bar_recorded_as_current_bar_index():
    ms = _make_ms({"BTC": [100.0, 110.0, 90.0, 105.0]}, bar_index=7)
    bucket = _bucket(ms, _long_position())
    order = _sl_trigger(trigger_px=90.0)
    place_triggers([order], ms, bucket, min_notional_usd=1.0)
    assert bucket.resting_triggers["BTC"][0].placed_bar == 7
