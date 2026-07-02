from __future__ import annotations

from typing import Callable, Dict, Type

SLOTS = (
    "universe_filter",
    "signal",
    "entry_timing",
    "sizing",
    "exit_rule",
)

_REGISTRY: Dict[str, Dict[str, Type]] = {slot: {} for slot in SLOTS}


def register(slot: str, kind: str) -> Callable[[Type], Type]:
    if slot not in _REGISTRY:
        raise KeyError(f"unknown slot: {slot!r}; expected one of {SLOTS}")

    def _decorator(cls: Type) -> Type:
        if kind in _REGISTRY[slot]:
            raise KeyError(f"duplicate gene kind {kind!r} in slot {slot!r}")
        _REGISTRY[slot][kind] = cls
        return cls

    return _decorator


def build_gene(slot: str, kind: str, params: dict) -> object:
    try:
        cls = _REGISTRY[slot][kind]
    except KeyError as exc:
        raise KeyError(f"no gene registered for slot {slot!r} kind {kind!r}") from exc
    return cls(**params)
