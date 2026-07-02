"""Task 2 (Phase E engine v2): OrderCommand trigger fields + partition rule.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rule R2
(trigger orders) + R3's CloseReason note.

Scope: `OrderCommand.trigger_px` / `.trigger_direction`, `CloseReason.TRIGGER`,
`OrderCommand.validate_trigger()`, and the partition guarantee that a TRIGGER
order reaching `execute_order` is still rejected (SUPPORTED_ORDER_TYPES stays
market-only; side-opposes-position is Task 3's job, not validated here).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.position import CloseReason, OrderCommand, OrderType, SUPPORTED_ORDER_TYPES  # noqa: E402


def _valid_trigger(**overrides) -> OrderCommand:
    kwargs = dict(
        asset="BTC",
        side=-1,
        order_type=OrderType.TRIGGER.value,
        size=0.5,
        reduce_only=True,
        trigger_px=100.0,
        trigger_direction="stop",
    )
    kwargs.update(overrides)
    return OrderCommand(**kwargs)


def test_close_reason_trigger_member_exists():
    assert CloseReason.TRIGGER.value == "trigger"


def test_valid_trigger_passes_validate_trigger():
    order = _valid_trigger()
    order.validate_trigger()  # must not raise


def test_valid_trigger_tp_direction_passes():
    order = _valid_trigger(trigger_direction="tp")
    order.validate_trigger()


def test_trigger_requires_order_type_trigger():
    order = _valid_trigger(order_type=OrderType.MARKET.value)
    with pytest.raises(ValueError, match="order_type"):
        order.validate_trigger()


def test_trigger_requires_reduce_only():
    order = _valid_trigger(reduce_only=False)
    with pytest.raises(ValueError, match="reduce_only"):
        order.validate_trigger()


def test_trigger_rejects_notional_instead_of_size():
    order = _valid_trigger(size=None, notional=100.0)
    with pytest.raises(ValueError, match="size"):
        order.validate_trigger()


def test_trigger_rejects_missing_size():
    order = _valid_trigger(size=None)
    with pytest.raises(ValueError, match="size"):
        order.validate_trigger()


def test_trigger_rejects_zero_size():
    order = _valid_trigger(size=0.0)
    with pytest.raises(ValueError, match="size"):
        order.validate_trigger()


def test_trigger_rejects_negative_size():
    order = _valid_trigger(size=-0.5)
    with pytest.raises(ValueError, match="size"):
        order.validate_trigger()


def test_trigger_rejects_none_trigger_px():
    order = _valid_trigger(trigger_px=None)
    with pytest.raises(ValueError, match="trigger_px"):
        order.validate_trigger()


def test_trigger_rejects_nan_trigger_px():
    order = _valid_trigger(trigger_px=math.nan)
    with pytest.raises(ValueError, match="trigger_px"):
        order.validate_trigger()


def test_trigger_rejects_inf_trigger_px():
    order = _valid_trigger(trigger_px=math.inf)
    with pytest.raises(ValueError, match="trigger_px"):
        order.validate_trigger()


def test_trigger_rejects_zero_trigger_px():
    order = _valid_trigger(trigger_px=0.0)
    with pytest.raises(ValueError, match="trigger_px"):
        order.validate_trigger()


def test_trigger_rejects_negative_trigger_px():
    order = _valid_trigger(trigger_px=-10.0)
    with pytest.raises(ValueError, match="trigger_px"):
        order.validate_trigger()


def test_trigger_rejects_bad_direction():
    order = _valid_trigger(trigger_direction="sl")
    with pytest.raises(ValueError, match="trigger_direction"):
        order.validate_trigger()


def test_trigger_rejects_none_direction():
    order = _valid_trigger(trigger_direction=None)
    with pytest.raises(ValueError, match="trigger_direction"):
        order.validate_trigger()


def test_plain_market_order_validate_basic_unaffected():
    order = OrderCommand(asset="BTC", side=1, order_type=OrderType.MARKET.value, size=1.0)
    order.validate_basic()  # must not raise; trigger fields default to None


def test_supported_order_types_stays_market_only():
    assert SUPPORTED_ORDER_TYPES == {OrderType.MARKET.value}


def test_trigger_order_type_not_in_supported_order_types():
    assert OrderType.TRIGGER.value not in SUPPORTED_ORDER_TYPES
