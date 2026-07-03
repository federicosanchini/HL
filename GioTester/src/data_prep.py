"""Build dense per-perp arrays + ranks lookup from raw MarketData."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_loader import DataLoader, MarketData


@dataclass
class SimData:
    """Frozen, simulator-ready market snapshot keyed by bar index."""

    timeline: pd.DatetimeIndex
    perps: List[str]
    perp_to_idx: Dict[str, int]
    ohlc: Dict[str, np.ndarray]  # perp -> (n_bars, 4)  NaN where missing
    funding: Dict[str, np.ndarray]  # perp -> (n_bars,)
    oracle: Dict[str, np.ndarray]  # perp -> (n_bars,)
    ranks_by_release: Dict[pd.Timestamp, Dict[str, Tuple[float, float]]]

    @property
    def n_bars(self) -> int:
        return len(self.timeline)


def _scatter(
    df: pd.DataFrame,
    val_cols,
    perps: List[str],
    n_bars: int,
    ts_to_idx: Dict[pd.Timestamp, int],
    label: str,
):
    """Densify long-format (perp, time, vals) into perp -> ndarray."""
    is_matrix = isinstance(val_cols, (list, tuple))
    width = len(val_cols) if is_matrix else 1
    shape = (n_bars, width) if is_matrix else (n_bars,)
    out: Dict[str, np.ndarray] = {p: np.full(shape, np.nan) for p in perps}
    perp_set = set(perps)
    for perp, g in tqdm(
        df.groupby("perp", sort=False),
        desc=label,
        total=len(perps),
        leave=False,
    ):
        if perp not in perp_set:
            continue
        idx = g["time"].map(ts_to_idx).fillna(-1).astype(np.int64).to_numpy()
        valid = idx >= 0
        if is_matrix:
            sub = g[list(val_cols)].to_numpy()
            out[perp][idx[valid]] = sub[valid]
        else:
            out[perp][idx[valid]] = g[val_cols].to_numpy()[valid]
    return out


def _load_ranks(
    ranks_path: str, perp_set: set
) -> Dict[pd.Timestamp, Dict[str, Tuple[float, float]]]:
    raw = pd.read_csv(ranks_path)
    raw["release_date"] = pd.to_datetime(
        raw["release_date"], utc=True, errors="coerce"
    ).dt.floor("D")
    raw = raw.dropna(subset=["release_date", "id"])
    raw["perp"] = raw["id"].map(DataLoader._normalize_symbol)
    raw = raw.dropna(subset=["perp"])
    raw = raw[raw["perp"].isin(perp_set)]
    for col in ("pred_10d", "pred_30d"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            raw[col] = np.nan

    out: Dict[pd.Timestamp, Dict[str, Tuple[float, float]]] = {}
    for date_ts, g in raw.groupby("release_date", sort=False):
        d: Dict[str, Tuple[float, float]] = {}
        for perp, p10, p30 in zip(
            g["perp"].values, g["pred_10d"].values, g["pred_30d"].values
        ):
            d[str(perp)] = (
                float(p10) if math.isfinite(p10) else float("nan"),
                float(p30) if math.isfinite(p30) else float("nan"),
            )
        out[pd.Timestamp(date_ts)] = d
    return out


def prepare_sim_data(market: MarketData, ranks_path: str) -> SimData:
    timeline = pd.DatetimeIndex(sorted(pd.unique(market.ohlc["time"])))
    n_bars = len(timeline)
    if n_bars == 0:
        raise ValueError("Empty timeline; no OHLC data after clipping.")

    ts_to_idx = {ts: i for i, ts in enumerate(timeline)}
    perps = sorted(market.ohlc["perp"].unique())
    perp_to_idx = {p: i for i, p in enumerate(perps)}

    ohlc_d = _scatter(
        market.ohlc,
        ("open", "high", "low", "close"),
        perps,
        n_bars,
        ts_to_idx,
        "OHLC arrays",
    )
    fund_d = _scatter(
        market.funding, "fundingRate", perps, n_bars, ts_to_idx, "funding arrays"
    )
    orac_d = _scatter(
        market.oracle, "oraclePx", perps, n_bars, ts_to_idx, "oracle arrays"
    )

    ranks = _load_ranks(ranks_path, set(perps))

    return SimData(
        timeline=timeline,
        perps=perps,
        perp_to_idx=perp_to_idx,
        ohlc=ohlc_d,
        funding=fund_d,
        oracle=orac_d,
        ranks_by_release=ranks,
    )
