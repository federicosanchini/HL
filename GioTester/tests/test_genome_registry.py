from __future__ import annotations

import pytest

from src.genome.registry import SLOTS, build_gene, register


def test_register_and_build():
    @register("signal", "dummy_test")
    class Dummy:
        def __init__(self, *, k=1):
            self.k = k

    obj = build_gene("signal", "dummy_test", {"k": 7})
    assert obj.k == 7


def test_slots_are_the_five_expected():
    assert SLOTS == (
        "universe_filter",
        "signal",
        "entry_timing",
        "sizing",
        "exit_rule",
    )


def test_unknown_slot_raises():
    with pytest.raises(KeyError):
        register("not_a_slot", "x")


def test_duplicate_kind_raises():
    @register("sizing", "dup_test")
    class A:
        pass

    with pytest.raises(KeyError):
        @register("sizing", "dup_test")
        class B:
            pass


def test_build_unregistered_raises():
    with pytest.raises(KeyError):
        build_gene("exit_rule", "does_not_exist", {})
