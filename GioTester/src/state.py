"""
Per-bar market snapshot and simulator state container.

Confidence: HIGH
Uncertainty: none material; container types only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .position import Position  # noqa: E402


@dataclass
class MarketState:
    """Single-bar snapshot. Rebuilt every iteration; no history retained."""
    timestamp: pd.Timestamp
    bar_index: int
    total_bars: int
    ohlc_row: Dict[str, np.ndarray]            # perp -> [o,h,l,c]
    funding_row: Dict[str, float]              # perp -> funding rate (hourly)
    oracle: Dict[str, float]                   # perp -> oracle px
    current_ranks_row: Optional[Dict[str, Tuple[float, float]]]  # perp -> (pred_10d, pred_30d) or None
    is_release_bar: bool = False               # True iff timestamp == 00:00 UTC && a release_date present


@dataclass
class StateBucket:
    """Mutable simulator state passed to strategies each bar."""
    market_state: MarketState
    current_positions: Dict[int, Position] = field(default_factory=dict)
    stats_data_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cash: float = 0.0

    def perp_stats(self, perp: str) -> Dict[str, float]:
        d = self.stats_data_bucket.get(perp)
        if d is None:
            d = {
                "locked_realized": 0.0,   # sum realized_pnl of closed positions on this perp
                "locked_funding": 0.0,    # sum cumulative_funding of closed positions
                "locked_fees": 0.0,       # sum cumulative_fees of closed positions
                "n_opened": 0,
                "n_closed": 0,
                "n_liquidated": 0,
            }
            self.stats_data_bucket[perp] = d
        return d
