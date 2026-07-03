"""Series-level performance metrics. Pure functions, no engine state."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def max_drawdown(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(dd.min())


def sharpe(returns: np.ndarray, ann: int) -> float:
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sd <= 0:
        return 0.0
    return mu / sd * math.sqrt(ann)


def sortino(returns: np.ndarray, ann: int) -> float:
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    dd = (
        float(np.std(downside, ddof=1))
        if downside.size > 1
        else float(np.abs(downside[0]))
    )
    if dd <= 0:
        return 0.0
    return mu / dd * math.sqrt(ann)


def series_metrics(eq: np.ndarray, initial: float, ann: int) -> Dict[str, float]:
    pnl = float(eq[-1] - eq[0]) if len(eq) > 0 else 0.0
    base = initial if initial > 0 else 1.0
    rets = np.diff(eq) / base
    return {
        "PnL": pnl,
        "DD": max_drawdown(eq),
        "Sharpe": sharpe(rets, ann),
        "Sortino": sortino(rets, ann),
    }


def print_metrics_table(
    name: str,
    metrics_per_perp: Dict[str, Dict[str, float]],
    metrics_total: Dict[str, float],
) -> None:
    print(f"\n=== {name} ===")
    cols = ("PnL", "DD", "Sharpe", "Sortino")
    rows = sorted(metrics_per_perp.items())
    perp_w = max(6, max((len(p) for p, _ in rows), default=6))
    fmt_h = f"{{:<{perp_w}}}  " + "  ".join(f"{{:>10}}" for _ in cols)
    fmt_r = f"{{:<{perp_w}}}  " + "  ".join(f"{{:>10.4f}}" for _ in cols)
    print(fmt_h.format("PERP", *cols))
    print("-" * (perp_w + 2 + 12 * len(cols)))
    for p, m in rows:
        print(fmt_r.format(p, m["PnL"], m["DD"], m["Sharpe"], m["Sortino"]))
    print("-" * (perp_w + 2 + 12 * len(cols)))
    print(fmt_r.format(
        "TOTAL",
        metrics_total["PnL"], metrics_total["DD"],
        metrics_total["Sharpe"], metrics_total["Sortino"],
    ))
