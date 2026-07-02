# tests/test_genome_ledger.py
from __future__ import annotations

from src.genome.genes import EntryLedger


def test_record_and_get():
    led = EntryLedger()
    led.record("BTC", bar_index=5, price=100.0)
    rec = led.get("BTC")
    assert rec is not None
    assert rec.bar_index == 5
    assert rec.price == 100.0
    assert "BTC" in led
    assert led.get("ETH") is None


def test_prune_drops_absent_assets():
    led = EntryLedger()
    led.record("BTC", 1, 100.0)
    led.record("ETH", 1, 200.0)
    led.prune({"BTC"})
    assert "BTC" in led
    assert "ETH" not in led


def test_record_overwrites():
    led = EntryLedger()
    led.record("BTC", 1, 100.0)
    led.record("BTC", 9, 150.0)
    rec = led.get("BTC")
    assert rec.bar_index == 9 and rec.price == 150.0
