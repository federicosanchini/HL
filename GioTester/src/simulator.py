from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from .config import BacktestConfig  # noqa: E402
from .data_loader import DataLoader, MarketData, mm_rate_for  # noqa: E402
from .position import CloseReason, ExpiryMode, NewOrder, Position  # noqa: E402
from .state import MarketState, StateBucket  # noqa: E402
from .strategies import Strategy  # noqa: E402


# ——— prepared data structure ———
@dataclass
class SimData:
    timeline: pd.DatetimeIndex  # tz-aware UTC, sorted unique
    ts_to_idx: Dict[pd.Timestamp, int]
    perps: List[str]
    ohlc: Dict[str, np.ndarray]  # perp -> (n_bars, 4) [o,h,l,c]; NaN if missing
    funding: Dict[str, np.ndarray]  # perp -> (n_bars,)
    oracle: Dict[str, np.ndarray]  # perp -> (n_bars,)
    ranks_by_release: Dict[pd.Timestamp, Dict[str, Tuple[float, float]]]


@dataclass
class SimResult:
    strategy_name: str
    timeline: pd.DatetimeIndex
    total_equity: np.ndarray  # equity curve (initial_cash + cum contribution)
    per_perp_equity: Dict[str, np.ndarray]  # per-perp cum contribution series
    metrics_total: Dict[str, float]
    metrics_per_perp: Dict[str, Dict[str, float]]
    n_opened: int
    n_closed: int
    n_liquidated: int
    long_pnl: float = 0.0    # realized PnL from long positions
    short_pnl: float = 0.0   # realized PnL from short positions
    funding_pnl: float = 0.0 # net funding received (negative = paid)
    total_fees: float = 0.0  # total fees paid (always >= 0)


# ————————————————————————————————————————————————————————————————————————— #
# Data prep
# ————————————————————————————————————————————————————————————————————————— #


def _prepare_sim_data(
    market: MarketData, ranks_path: str, start_clip: Optional[pd.Timestamp] = None
) -> SimData:
    """Build dense per-perp arrays aligned to a common hourly timeline + ranks lookup."""
    timeline = pd.DatetimeIndex(sorted(pd.unique(market.ohlc["time"])))
    n_bars = len(timeline)
    ts_to_idx: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(timeline)}

    perps = sorted(market.ohlc["perp"].unique())

    def _idx_map(times: pd.Series) -> np.ndarray:
        return times.map(ts_to_idx).fillna(-1).astype(np.int64).to_numpy()

    n_perps = len(perps)

    # OHLC dense
    ohlc_d: Dict[str, np.ndarray] = {p: np.full((n_bars, 4), np.nan) for p in perps}
    for perp, g in tqdm(market.ohlc.groupby("perp", sort=False),
                        desc="OHLC arrays", total=n_perps, leave=False):
        idx = _idx_map(g["time"])
        valid = idx >= 0
        sub = g[["open", "high", "low", "close"]].to_numpy()
        ohlc_d[perp][idx[valid]] = sub[valid]

    fund_d: Dict[str, np.ndarray] = {p: np.full(n_bars, np.nan) for p in perps}
    for perp, g in tqdm(market.funding.groupby("perp", sort=False),
                        desc="funding arrays", total=n_perps, leave=False):
        if perp not in fund_d:
            continue
        idx = _idx_map(g["time"])
        valid = idx >= 0
        fund_d[perp][idx[valid]] = g["fundingRate"].to_numpy()[valid]

    orac_d: Dict[str, np.ndarray] = {p: np.full(n_bars, np.nan) for p in perps}
    for perp, g in tqdm(market.oracle.groupby("perp", sort=False),
                        desc="oracle arrays", total=n_perps, leave=False):
        if perp not in orac_d:
            continue
        idx = _idx_map(g["time"])
        valid = idx >= 0
        orac_d[perp][idx[valid]] = g["oraclePx"].to_numpy()[valid]

    # Ranks: read raw to keep both pred_10d and pred_30d
    ranks_raw = pd.read_csv(ranks_path)
    ranks_raw["release_date"] = pd.to_datetime(
        ranks_raw["release_date"], utc=True, errors="coerce"
    ).dt.floor("D")
    ranks_raw = ranks_raw.dropna(subset=["release_date", "id"])
    ranks_raw["perp"] = ranks_raw["id"].map(
        DataLoader._normalize_symbol
    )  # staticmethod
    ranks_raw = ranks_raw.dropna(subset=["perp"])
    ranks_raw = ranks_raw[ranks_raw["perp"].isin(set(perps))]
    for col in ("pred_10d", "pred_30d"):
        if col in ranks_raw.columns:
            ranks_raw[col] = pd.to_numeric(ranks_raw[col], errors="coerce")
        else:
            ranks_raw[col] = np.nan

    ranks_lookup: Dict[pd.Timestamp, Dict[str, Tuple[float, float]]] = {}
    for date_ts, g in ranks_raw.groupby("release_date", sort=False):
        d: Dict[str, Tuple[float, float]] = {}
        for perp, p10, p30 in zip(
            g["perp"].values, g["pred_10d"].values, g["pred_30d"].values
        ):
            d[str(perp)] = (
                float(p10) if math.isfinite(p10) else float("nan"),
                float(p30) if math.isfinite(p30) else float("nan"),
            )
        ranks_lookup[pd.Timestamp(date_ts)] = d

    return SimData(
        timeline=timeline,
        ts_to_idx=ts_to_idx,
        perps=perps,
        ohlc=ohlc_d,
        funding=fund_d,
        oracle=orac_d,
        ranks_by_release=ranks_lookup,
    )


# ————————————————————————————————————————————————————————————————————————— #
# Bar helpers
# ————————————————————————————————————————————————————————————————————————— #


def _build_market_state(sd: SimData, i: int, n_bars: int) -> MarketState:
    ts = sd.timeline[i]
    ohlc_row: Dict[str, np.ndarray] = {}
    funding_row: Dict[str, float] = {}
    oracle_row: Dict[str, float] = {}
    for perp in sd.perps:
        row = sd.ohlc[perp][i]
        if math.isfinite(row[0]):
            ohlc_row[perp] = row
        f = sd.funding[perp][i]
        if math.isfinite(f):
            funding_row[perp] = float(f)
        o = sd.oracle[perp][i]
        if math.isfinite(o):
            oracle_row[perp] = float(o)

    is_release = False
    ranks_row: Optional[Dict[str, Tuple[float, float]]] = None
    if ts.hour == 0:
        day = ts.normalize()
        if day in sd.ranks_by_release:
            is_release = True
            ranks_row = sd.ranks_by_release[day]

    mm_rate_cache = {perp: mm_rate_for(perp) for perp in sd.perps}

    return MarketState(
        timestamp=ts,
        bar_index=i,
        total_bars=n_bars,
        ohlc_row=ohlc_row,
        funding_row=funding_row,
        oracle=oracle_row,
        current_ranks_row=ranks_row,
        mm_rate_cache=mm_rate_cache,
        is_release_bar=is_release,
    )


def _execute_close(
    pos: Position,
    fill_px: float,
    fee_bps: float,
    bucket: StateBucket,
) -> None:
    """Close pos at fill_px; settle margin + fees; update locked stats."""
    notional_close = abs(pos.qty) * fill_px
    fee = notional_close * fee_bps / 1e4
    pnl = pos.side * pos.qty * (fill_px - pos.entry_price)
    pos.realized_pnl += pnl  # accumulate: may include prior partial-close PnL
    pos.cumulative_fees += fee
    margin_after = pos.initial_margin + pnl - fee
    if pos.close_reason == CloseReason.LIQUIDATION:
        # HL: maintenance margin forfeited to HLP vault; trader receives nothing back.
        margin_after = 0.0
    else:
        margin_after = max(margin_after, 0.0)
    bucket.cash += margin_after
    pos.closed = True
    pos.mark_price = fill_px
    s = bucket.perp_stats(pos.perp)
    s["locked_realized"] += pos.realized_pnl
    if pos.side == 1:
        s["locked_realized_long"] += pos.realized_pnl
    else:
        s["locked_realized_short"] += pos.realized_pnl
    s["locked_funding"] += pos.cumulative_funding
    s["locked_fees"] += pos.cumulative_fees
    if pos.close_reason == CloseReason.FORCE:
        s["n_closed"] += 1


def _execute_open(
    order: NewOrder,
    fill_px: float,
    fee_bps: float,
    ts: pd.Timestamp,
    next_id: int,
    bucket: StateBucket,
    ohlc_row: Optional[np.ndarray] = None,
    mm_rate: float = 0.05,
    abs_expiry_bar: int = 0,
    expiry_mode: ExpiryMode = ExpiryMode.RESET_LATEST,
) -> Optional[Position]:
    if not (math.isfinite(fill_px) and fill_px > 0):
        return None
    # limit order: check if bar's high/low reaches the limit
    if order.limit_price is not None:
        lp = order.limit_price
        if ohlc_row is None or not (math.isfinite(lp) and lp > 0):
            return None
        if order.side == 1 and ohlc_row[2] > lp:   # long: low must reach limit
            return None
        if order.side == -1 and ohlc_row[1] < lp:  # short: high must reach limit
            return None
        fill_px = lp
    notional = order.notional
    qty = notional / fill_px
    margin = notional / order.leverage
    fee = notional * fee_bps / 1e4
    if bucket.cash < margin + fee:
        return None
    bucket.cash -= margin + fee
    pos = Position(
        id=next_id,
        perp=order.perp,
        side=int(order.side),
        qty=qty,
        entry_price=fill_px,
        entry_time=ts,
        expiry_bars=order.expiry_bars,
        abs_expiry_bar=abs_expiry_bar,
        leverage=order.leverage,
        initial_margin=margin,
        notional=notional,
        expiry_mode=expiry_mode,
        tranches=[(qty, abs_expiry_bar)] if expiry_mode == ExpiryMode.PROPORTIONAL else [],
        mm_rate=mm_rate,
        cumulative_fees=fee,
        mark_price=fill_px,
    )
    bucket.current_positions[next_id] = pos
    bucket.register_position(pos)
    return pos


def _execute_partial_close(
    pos: Position,
    close_qty: float,
    fill_px: float,
    fee_bps: float,
    bucket: StateBucket,
) -> None:
    """Reduce pos by close_qty at fill_px. Position stays in bucket.current_positions."""
    close_frac = close_qty / pos.qty
    margin_released = pos.initial_margin * close_frac
    fee = close_qty * fill_px * fee_bps / 1e4
    pnl = pos.side * close_qty * (fill_px - pos.entry_price)
    pos.realized_pnl += pnl
    pos.cumulative_fees += fee
    pos.qty -= close_qty
    pos.initial_margin -= margin_released
    pos.notional = pos.qty * pos.entry_price
    pos.close_qty = None
    margin_return = max(margin_released + pnl - fee, 0.0)
    bucket.cash += margin_return


def _merge_into(
    existing: Position,
    order: NewOrder,
    fill_px: float,
    fee_bps: float,
    ts: pd.Timestamp,
    bucket: StateBucket,
    abs_expiry_bar: int,
) -> None:
    """VWAP-merge a same-side order into an existing position (HL netting behaviour)."""
    add_qty = order.notional / fill_px
    add_margin = order.notional / order.leverage
    fee = order.notional * fee_bps / 1e4
    if bucket.cash < add_margin + fee:
        return
    bucket.cash -= add_margin + fee
    existing.cumulative_fees += fee
    total_qty = existing.qty + add_qty
    existing.entry_price = (
        existing.qty * existing.entry_price + add_qty * fill_px
    ) / total_qty
    existing.qty = total_qty
    existing.notional = total_qty * existing.entry_price
    existing.initial_margin += add_margin
    if existing.expiry_mode == ExpiryMode.RESET_LATEST:
        existing.abs_expiry_bar = abs_expiry_bar
        existing.bar_age = 0
    elif existing.expiry_mode == ExpiryMode.PROPORTIONAL:
        existing.tranches.append((add_qty, abs_expiry_bar))
    # "keep_earliest": abs_expiry_bar unchanged


def _net_against(
    opposing: Position,
    order: NewOrder,
    fill_px: float,
    fee_bps: float,
    ts: pd.Timestamp,
    next_id: int,
    bucket: StateBucket,
    abs_expiry_bar: int,
    strategy: Strategy,
    mm_rate: float,
) -> Optional[Position]:
    """Net new order against opposing-side position. Returns new Position if remainder."""
    order_qty = order.notional / fill_px
    if order_qty >= opposing.qty - 1e-10:
        # Close opposing fully; open remainder if above min notional
        opposing.close_reason = CloseReason.NET
        _execute_close(opposing, fill_px, fee_bps, bucket)
        bucket.unregister_position(opposing.perp, opposing.side)
        bucket.current_positions.pop(opposing.id, None)
        remainder_qty = order_qty - opposing.qty
        remainder_notional = remainder_qty * fill_px
        if remainder_notional >= strategy.min_notional_usd:
            rem_order = NewOrder(
                perp=order.perp,
                side=order.side,
                notional=remainder_notional,
                leverage=order.leverage,
                expiry_bars=order.expiry_bars,
            )
            return _execute_open(
                rem_order, fill_px, fee_bps, ts, next_id, bucket,
                mm_rate=mm_rate, abs_expiry_bar=abs_expiry_bar,
                expiry_mode=strategy.expiry_mode,
            )
    else:
        # Partially reduce opposing
        _execute_partial_close(opposing, order_qty, fill_px, fee_bps, bucket)
    return None


# ————————————————————————————————————————————————————————————————————————— #
# Metrics
# ————————————————————————————————————————————————————————————————————————— #


def _max_drawdown(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(dd.min())


def _sharpe(returns: np.ndarray, ann: int) -> float:
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sd <= 0:
        return 0.0
    return mu / sd * math.sqrt(ann)


def _sortino(returns: np.ndarray, ann: int) -> float:
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


def _series_metrics(eq: np.ndarray, initial: float, ann: int) -> Dict[str, float]:
    pnl = float(eq[-1] - eq[0]) if len(eq) > 0 else 0.0
    base = initial if initial > 0 else 1.0
    rets = np.diff(eq) / base
    return {
        "PnL": pnl,
        "DD": _max_drawdown(eq),
        "Sharpe": _sharpe(rets, ann),
        "Sortino": _sortino(rets, ann),
    }


def _print_table(
    name: str,
    metrics_per_perp: Dict[str, Dict[str, float]],
    metrics_total: Dict[str, float],
) -> None:
    print(f"\n=== {name} ===")
    cols = ("PnL", "DD", "Sharpe", "Sortino")
    rows = [(p, m) for p, m in sorted(metrics_per_perp.items())]
    widths = {"perp": max(6, max((len(p) for p, _ in rows), default=6))}
    fmt_h = f"{{:<{widths['perp']}}}  " + "  ".join(f"{{:>10}}" for _ in cols)
    fmt_r = f"{{:<{widths['perp']}}}  " + "  ".join(f"{{:>10.4f}}" for _ in cols)
    print(fmt_h.format("PERP", *cols))
    print("-" * (widths["perp"] + 2 + (12 * len(cols))))
    for p, m in rows:
        print(fmt_r.format(p, m["PnL"], m["DD"], m["Sharpe"], m["Sortino"]))
    print("-" * (widths["perp"] + 2 + (12 * len(cols))))
    print(
        fmt_r.format(
            "TOTAL",
            metrics_total["PnL"],
            metrics_total["DD"],
            metrics_total["Sharpe"],
            metrics_total["Sortino"],
        )
    )


# ————————————————————————————————————————————————————————————————————————— #
# Run
# ————————————————————————————————————————————————————————————————————————— #


def run_backtest(
    strategy: Strategy,
    market: MarketData,
    ranks_path: str,
    bt_cfg: BacktestConfig,
    verbose: bool = True,
) -> SimResult:
    sd = _prepare_sim_data(market, ranks_path)
    n_bars = len(sd.timeline)
    if n_bars == 0:
        raise ValueError("Empty timeline; no OHLC data after clipping.")

    bucket = StateBucket(
        market_state=_build_market_state(sd, 0, n_bars),
        cash=bt_cfg.initial_equity,
    )
    # Pre-init perp stats
    for p in sd.perps:
        bucket.perp_stats(p)

    # per-perp cumulative contribution series
    per_perp_eq: Dict[str, np.ndarray] = {p: np.zeros(n_bars) for p in sd.perps}
    total_eq = np.zeros(n_bars)

    next_id = 1
    fee_bps = bt_cfg.taker_fee_bps

    iterator = tqdm(range(n_bars), desc=strategy.name) if verbose else range(n_bars)
    last_idx = n_bars - 1

    for i in iterator:
        ms = _build_market_state(sd, i, n_bars)
        bucket.market_state = ms

        # End-of-data: force close everyone at this last bar's open
        if i == last_idx:
            for pos in list(bucket.current_positions.values()):
                pos.close_reason = CloseReason.FORCE
                pos.close_price = None
            # synthetic flush — drain via execute_close at bar open
            for pos in list(bucket.current_positions.values()):
                row = ms.ohlc_row.get(pos.perp)
                if row is not None and math.isfinite(row[0]):
                    px = float(row[0])
                else:
                    px = pos.mark_price if pos.mark_price > 0 else pos.entry_price
                _execute_close(pos, px, fee_bps, bucket)
                bucket.unregister_position(pos.perp, pos.side)
                bucket.current_positions.pop(pos.id, None)
        else:
            closed_list, new_orders = strategy.update_positions(bucket)

            # Closes at this bar's open (forced fills override)
            for pos in closed_list:
                if pos.close_reason == CloseReason.LIQUIDATION and pos.close_price is not None:
                    fill = pos.close_price
                else:
                    o = ms.ohlc_row.get(pos.perp)
                    fill = (
                        float(o[0])
                        if o is not None and math.isfinite(o[0])
                        else pos.entry_price
                    )
                if pos.close_qty is not None:
                    # Proportional partial close — position stays in bucket
                    _execute_partial_close(pos, pos.close_qty, fill, fee_bps, bucket)
                else:
                    _execute_close(pos, fill, fee_bps, bucket)
                    bucket.unregister_position(pos.perp, pos.side)
                    del bucket.current_positions[pos.id]

            # Opens — with HL-style netting: same-side merges, opposing-side nets
            for order in new_orders:
                row = ms.ohlc_row.get(order.perp)
                if row is None or not math.isfinite(row[0]):
                    continue
                if order.notional < strategy.min_notional_usd:
                    continue
                fill_px = float(row[0])
                abs_exp = i + order.expiry_bars
                existing = bucket.get_position(order.perp, order.side)
                opposing = bucket.get_position(order.perp, -order.side)
                if existing is not None:
                    _merge_into(existing, order, fill_px, fee_bps, ms.timestamp, bucket, abs_exp)
                elif opposing is not None:
                    new_pos = _net_against(
                        opposing, order, fill_px, fee_bps, ms.timestamp,
                        next_id, bucket, abs_exp, strategy, ms.mm_rate_cache.get(order.perp, 0.05),
                    )
                    if new_pos is not None:
                        next_id += 1
                else:
                    placed = _execute_open(
                        order, fill_px, fee_bps, ms.timestamp, next_id, bucket,
                        ohlc_row=row, mm_rate=ms.mm_rate_cache.get(order.perp, 0.05),
                        abs_expiry_bar=abs_exp, expiry_mode=strategy.expiry_mode,
                    )
                    if placed is not None:
                        next_id += 1

            # Mark + funding for open positions
            for pos in bucket.current_positions.values():
                row = ms.ohlc_row.get(pos.perp)
                close_px = (
                    float(row[3])
                    if row is not None and math.isfinite(row[3])
                    else pos.mark_price
                )
                pos.mark(close_px)
                rate = ms.funding_row.get(pos.perp, float("nan"))
                ora = ms.oracle.get(pos.perp)  # None if missing; no mark fallback
                if math.isfinite(rate) and ora is not None and math.isfinite(ora):
                    pos.update_funding(rate, ora)

        # Snapshot per-perp contribution at this bar
        contrib_per_perp: Dict[str, float] = {}
        for p in sd.perps:
            s = bucket.stats_data_bucket[p]
            base = s["locked_realized"] + s["locked_funding"] - s["locked_fees"]
            contrib_per_perp[p] = base
        for pos in bucket.current_positions.values():
            contrib_per_perp[pos.perp] = contrib_per_perp.get(pos.perp, 0.0) + (
                pos.realized_pnl + pos.unrealized_pnl + pos.cumulative_funding - pos.cumulative_fees
            )

        for p in sd.perps:
            per_perp_eq[p][i] = contrib_per_perp.get(p, 0.0)

        total_eq[i] = bt_cfg.initial_equity + sum(contrib_per_perp.values())

    # Metrics
    metrics_total = _series_metrics(
        total_eq, bt_cfg.initial_equity, bt_cfg.annualization
    )
    metrics_per_perp: Dict[str, Dict[str, float]] = {}
    for p, series in per_perp_eq.items():
        if series[-1] == 0.0 and not np.any(series != 0.0):
            continue  # skip perps that never traded
        eq_p = bt_cfg.initial_equity + series  # quasi-equity for DD basis
        metrics_per_perp[p] = _series_metrics(
            eq_p, bt_cfg.initial_equity, bt_cfg.annualization
        )

    n_opened = sum(s["n_opened"] for s in bucket.stats_data_bucket.values())
    n_closed = sum(s["n_closed"] for s in bucket.stats_data_bucket.values())
    n_liq = sum(s["n_liquidated"] for s in bucket.stats_data_bucket.values())

    long_pnl = sum(s["locked_realized_long"] for s in bucket.stats_data_bucket.values())
    short_pnl = sum(s["locked_realized_short"] for s in bucket.stats_data_bucket.values())
    funding_pnl = sum(s["locked_funding"] for s in bucket.stats_data_bucket.values())
    total_fees = sum(s["locked_fees"] for s in bucket.stats_data_bucket.values())

    if verbose:
        _print_table(strategy.name, metrics_per_perp, metrics_total)
        print(f"  trades opened={n_opened}  closed={n_closed}  liquidated={n_liq}")

    return SimResult(
        strategy_name=strategy.name,
        timeline=sd.timeline,
        total_equity=total_eq,
        per_perp_equity=per_perp_eq,
        metrics_total=metrics_total,
        metrics_per_perp=metrics_per_perp,
        n_opened=int(n_opened),
        n_closed=int(n_closed),
        n_liquidated=int(n_liq),
        long_pnl=float(long_pnl),
        short_pnl=float(short_pnl),
        funding_pnl=float(funding_pnl),
        total_fees=float(total_fees),
    )


def log_results(result: SimResult, path: str) -> None:
    """Serialize SimResult to JSON for GioVisualizer. Only includes perps that traded."""
    import json
    from pathlib import Path

    traded = set(result.metrics_per_perp.keys())
    payload = {
        "strategy": result.strategy_name,
        "timeline": [ts.isoformat() for ts in result.timeline],
        "total_equity": result.total_equity.tolist(),
        "per_perp_equity": {
            p: arr.tolist()
            for p, arr in result.per_perp_equity.items()
            if p in traded
        },
        "metrics_total": result.metrics_total,
        "metrics_per_perp": result.metrics_per_perp,
        "n_opened": result.n_opened,
        "n_closed": result.n_closed,
        "n_liquidated": result.n_liquidated,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
