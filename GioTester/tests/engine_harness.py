"""Synthetic SimData + scripted trader harness for engine v2 pipeline tests.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md (R6). Builds a
minimal `SimData` (hand-set OHLC/funding/oracle) directly, without touching the
CSV loaders, and drives it through `run_backtest_prepared` with a `ScriptedTrader`
that emits pre-baked `OrderCommand`s keyed by bar index.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BacktestConfig  # noqa: E402
from src.data_prep import SimData  # noqa: E402
from src.position import OrderCommand, OrderType  # noqa: E402
from src.result import SimResult  # noqa: E402
from src.runner import run_backtest_prepared  # noqa: E402


def build_sim_data(
    ohlc: Dict[str, Sequence[Sequence[float]]],
    *,
    funding: Optional[Dict[str, Sequence[float]]] = None,
    oracle: Optional[Dict[str, Sequence[float]]] = None,
    start: str = "2025-01-01",
) -> SimData:
    """Build a `SimData` from per-asset OHLC rows ``[[open, high, low, close], ...]``.

    Every asset must share the same bar count. Funding defaults to 0.0 per bar;
    oracle defaults to each bar's open price. Missing values may be ``np.nan`` to
    model unpriceable/gap bars.
    """
    perps = sorted(ohlc)
    if not perps:
        raise ValueError("at least one asset required")
    n_bars = len(next(iter(ohlc.values())))
    for asset, rows in ohlc.items():
        if len(rows) != n_bars:
            raise ValueError(f"{asset}: expected {n_bars} bars, got {len(rows)}")

    timeline = pd.DatetimeIndex(
        pd.date_range(start=start, periods=n_bars, freq="h", tz="UTC")
    )
    perp_to_idx = {p: i for i, p in enumerate(perps)}
    ohlc_d: Dict[str, np.ndarray] = {p: np.asarray(ohlc[p], dtype=float) for p in perps}

    funding = funding or {}
    oracle = oracle or {}
    funding_d: Dict[str, np.ndarray] = {}
    oracle_d: Dict[str, np.ndarray] = {}
    for p in perps:
        funding_d[p] = (
            np.asarray(funding[p], dtype=float)
            if p in funding
            else np.zeros(n_bars, dtype=float)
        )
        oracle_d[p] = (
            np.asarray(oracle[p], dtype=float)
            if p in oracle
            else ohlc_d[p][:, 0].copy()
        )

    return SimData(
        timeline=timeline,
        perps=perps,
        perp_to_idx=perp_to_idx,
        ohlc=ohlc_d,
        funding=funding_d,
        oracle=oracle_d,
        ranks_by_release={},
    )


def market_order(asset: str, side: int, *, notional=None, size=None, leverage=1.0,
                 reduce_only=False) -> OrderCommand:
    return OrderCommand(
        asset=asset,
        side=side,
        order_type=OrderType.MARKET.value,
        notional=notional,
        size=size,
        leverage=leverage,
        reduce_only=reduce_only,
    )


def trigger_order(asset: str, side: int, *, size: float, trigger_px: float,
                  direction: str) -> OrderCommand:
    return OrderCommand(
        asset=asset,
        side=side,
        order_type=OrderType.TRIGGER.value,
        size=size,
        reduce_only=True,
        trigger_px=trigger_px,
        trigger_direction=direction,
    )


class ScriptedTrader:
    """Emits pre-scripted `OrderCommand`s keyed by bar index."""

    def __init__(
        self,
        script: Dict[int, List[OrderCommand]],
        *,
        name: str = "Scripted",
        margin_mode: str = "cross",
    ) -> None:
        self._script = script
        self.name = name
        self.margin_mode = margin_mode

    def run(self, state) -> List[OrderCommand]:
        return list(self._script.get(state.bar_index, []))


def run_scripted(
    sim_data: SimData,
    script: Dict[int, List[OrderCommand]],
    *,
    bt_cfg: Optional[BacktestConfig] = None,
    margin_mode: str = "cross",
    name: str = "Scripted",
) -> SimResult:
    trader = ScriptedTrader(script, name=name, margin_mode=margin_mode)
    cfg = bt_cfg or BacktestConfig()
    return run_backtest_prepared(trader, sim_data, cfg, verbose=False)
