"""SimResult dataclass + JSON serialization + terminal summary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .position import ExecutionEvent, FundingEvent, LiquidationEvent, OrderRejectedEvent


@dataclass
class SimResult:
    strategy_name: str
    timeline: pd.DatetimeIndex
    total_equity: np.ndarray
    per_perp_equity: Dict[str, np.ndarray]
    per_perp_position: Dict[str, np.ndarray]
    per_perp_position_qty: Dict[str, np.ndarray]
    liquidation_events: List[LiquidationEvent]
    execution_events: List[ExecutionEvent]
    rejected_orders: List[OrderRejectedEvent]
    funding_events: List[FundingEvent]
    metrics_total: Dict[str, float]
    metrics_per_perp: Dict[str, Dict[str, float]]
    n_opened: int
    n_closed: int
    n_liquidated: int
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    liq_long_pnl: float = 0.0
    liq_short_pnl: float = 0.0
    funding_pnl: float = 0.0
    total_fees: float = 0.0
    margin_mode: str = "cross"
    engine_semantics_version: int = 2
    # Per-perp, per-bar maintenance margin (Task P1). Additive/internal only --
    # not serialized by log_results (schema v5 is unchanged); consumed by
    # src/evolution/scoring.py's exact margin-headroom calc (Task P2).
    per_perp_maintenance: Dict[str, np.ndarray] = field(default_factory=dict)


def log_results(result: SimResult, path: str) -> None:
    """Serialize SimResult to JSON for GioVisualizer.

    Schema v5 changes per_perp_position to signed invested USD per asset per
    bar. Schema v4 added margin_mode. Schema v3 added per_perp_position as
    signed token quantity; that legacy quantity is now preserved explicitly in
    per_perp_position_qty.
    """
    traded = set(result.metrics_per_perp.keys())
    payload = {
        "schema_version": 5,
        "strategy": result.strategy_name,
        "margin_mode": result.margin_mode,
        "engine_semantics_version": result.engine_semantics_version,
        "timeline": [ts.isoformat() for ts in result.timeline],
        "total_equity": result.total_equity.tolist(),
        "per_perp_equity": {
            p: arr.tolist() for p, arr in result.per_perp_equity.items() if p in traded
        },
        "per_perp_position": {
            p: arr.tolist() for p, arr in result.per_perp_position.items() if p in traded
        },
        "per_perp_position_qty": {
            p: arr.tolist()
            for p, arr in result.per_perp_position_qty.items()
            if p in traded
        },
        "execution_events": [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "asset": e.asset,
                "side": e.side,
                "notional": e.notional,
                "fill_price": e.fill_price,
                "fee": e.fee,
                "realized_pnl": e.realized_pnl,
                "reason": e.reason,
            }
            for e in result.execution_events
        ],
        "funding_events": [
            {
                "timestamp": e.timestamp,
                "asset": e.asset,
                "funding_rate": e.funding_rate,
                "oracle_px": e.oracle_px,
                "payment": e.payment,
            }
            for e in result.funding_events
        ],
        "rejected_orders": [
            {
                "timestamp": e.timestamp,
                "asset": e.asset,
                "side": e.side,
                "order_type": e.order_type,
                "reason": e.reason,
            }
            for e in result.rejected_orders
        ],
        "liquidation_events": [
            {
                "timestamp": e.timestamp,
                "asset": e.asset,
                "net_cash_loss": e.realized_pnl - e.fee,
                "account_equity": e.account_equity,
                "maintenance_margin": e.maintenance_margin,
                "fill_price": e.fill_price,
                "realized_pnl": e.realized_pnl,
                "fee": e.fee,
            }
            for e in result.liquidation_events
        ],
        "metrics_total": result.metrics_total,
        "metrics_per_perp": result.metrics_per_perp,
        "n_opened": result.n_opened,
        "n_closed": result.n_closed,
        "n_liquidated": result.n_liquidated,
        "long_pnl": result.long_pnl,
        "short_pnl": result.short_pnl,
        "liq_long_pnl": result.liq_long_pnl,
        "liq_short_pnl": result.liq_short_pnl,
        "funding_pnl": result.funding_pnl,
        "total_fees": result.total_fees,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def print_result_summary(result: SimResult, saved_path: Optional[str] = None) -> None:
    SEP = "─" * 56

    net_pnl = (
        result.long_pnl
        + result.short_pnl
        + result.liq_long_pnl
        + result.liq_short_pnl
        + result.funding_pnl
        - result.total_fees
    )
    positions_total = result.long_pnl + result.short_pnl
    liq_total = result.liq_long_pnl + result.liq_short_pnl

    title = f" {result.strategy_name} "
    side = "─" * 20
    print()
    print(f"  | {side} |{title}| {side} |")
    print()

    print("  Trades")
    print(f"  {SEP}")
    print(f"  {'Opened':<14}{result.n_opened:>8}")
    print(f"  {'Closed':<14}{result.n_closed:>8}")
    print(f"  {'Liquidated':<14}{result.n_liquidated:>8}")
    print()

    print("  PnL Breakdown")
    print(f"  {SEP}")

    label_w = 16
    col_w = 13
    print(f"  {'':<{label_w}}" f"{'LONG':>{col_w}}{'SHORT':>{col_w}}{'TOTAL':>{col_w}}")
    print(
        f"  {'Positions':<{label_w}}"
        f"{result.long_pnl:+{col_w},.2f}"
        f"{result.short_pnl:+{col_w},.2f}"
        f"{positions_total:+{col_w},.2f}"
    )
    print(
        f"  {'Liquidations':<{label_w}}"
        f"{result.liq_long_pnl:+{col_w},.2f}"
        f"{result.liq_short_pnl:+{col_w},.2f}"
        f"{liq_total:+{col_w},.2f}"
    )
    print()

    total_col_start = label_w + col_w * 2
    print(f"  {'Funding':<{total_col_start}}{result.funding_pnl:+{col_w},.2f}")
    print(f"  {'Fees':<{total_col_start}}{-result.total_fees:+{col_w},.2f}")
    print()
    print(f"  {SEP}")
    print(f"  {'NET PnL':<{total_col_start}}{net_pnl:+{col_w},.2f}")
    print(f"  {SEP}")

    if saved_path:
        print()
        print("  Saved →")
        print(f"  {saved_path}")
        print(f"  {SEP}")
