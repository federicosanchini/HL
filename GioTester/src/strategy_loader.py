from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(path: Path) -> ModuleType:
    module_name = f"giotester_external_trader_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import trader module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_trader(path: str | Path, *args, **kwargs) -> Any:
    """Load a standalone strategy file and instantiate its Trader class."""
    trader_path = Path(path)
    if trader_path.suffix.lower() != ".py":
        raise ValueError("external strategy file must be a .py file")
    if not trader_path.exists():
        raise FileNotFoundError(f"strategy file not found: {trader_path}")

    module = _load_module(trader_path)
    trader_cls = getattr(module, "Trader", None)
    if trader_cls is None:
        raise ValueError(f"{trader_path} must define class Trader")
    trader = trader_cls(*args, **kwargs)
    run = getattr(trader, "run", None)
    if run is None or not callable(run):
        raise ValueError("Trader must expose callable run(state)")
    return trader
