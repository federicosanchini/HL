# src/genome/genes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Set, runtime_checkable

from src import OrderCommand


@dataclass(frozen=True)
class EntryRecord:
    bar_index: int
    price: float


@dataclass
class EntryLedger:
    """Central record of when/at-what-price each open position was entered.

    The adapter maintains one ledger per backtest so exit genes stay stateless
    with respect to entry tracking.
    """

    _by_asset: Dict[str, EntryRecord] = field(default_factory=dict)

    def record(self, asset: str, bar_index: int, price: float) -> None:
        self._by_asset[asset] = EntryRecord(bar_index=int(bar_index), price=float(price))

    def get(self, asset: str) -> EntryRecord | None:
        return self._by_asset.get(asset)

    def prune(self, live_assets: Set[str]) -> None:
        for asset in list(self._by_asset):
            if asset not in live_assets:
                del self._by_asset[asset]

    def __contains__(self, asset: object) -> bool:
        return asset in self._by_asset


@runtime_checkable
class UniverseFilter(Protocol):
    def eligible(self, state) -> Set[str]: ...


@runtime_checkable
class Signal(Protocol):
    def score(self, state, universe: Set[str]) -> Dict[str, float]: ...


@runtime_checkable
class EntryTiming(Protocol):
    def should_enter(self, state) -> bool: ...


@runtime_checkable
class Sizing(Protocol):
    def orders_for(self, state, longs: List[str], shorts: List[str]) -> List[OrderCommand]: ...


@runtime_checkable
class ExitRule(Protocol):
    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]: ...
