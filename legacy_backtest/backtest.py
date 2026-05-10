#!/usr/bin/env python3
"""
Hyperliquid daily long/short portfolio backtester.

The strategy:
  1. Loads daily predictions/ranks from Numerai or Crowdcent, e.g.:
        ranks_numerai.csv:   id,symbol,date,prediction
        ranks_crowdcent.csv: id,pred_10d,pred_30d,release_date
  2. Normalizes symbol names, then keeps only symbols in the Hyperliquid tradable universe below.
  3. Each day, goes long the N highest prediction symbols and short the N lowest.
  4. If an order is in the same direction as an existing position, it is merged
     into that position by updating the weighted-average entry price.
  5. If an order is opposite an existing position, it reduces, closes, or flips it.
  6. Schedules a time-based exit for each newly opened daily-signal quantity
     after --holding-days days, unless that exposure was already closed or liquidated.
  7. Applies hourly funding.
  8. Tracks hourly account equity, margin/equity approximations, long/short PnL,
     fees, liquidations, win rates, CAGR, Sharpe, max drawdown, and trade counts.

Important modelling assumptions:
  - This is a backtest approximation, not a replica of Hyperliquid's matching engine.
  - Hyperliquid liquidations and unrealized PnL use mark price, but the provided
    perps_prices_1h_ohlc.csv has OHLC prices. The script uses close as the hourly
    mark proxy and high/low as intrahour liquidation proxies.
  - Funding is applied hourly using:
        funding_pnl = -side * abs(position_qty) * oracle_price * funding_rate
    so a positive funding rate costs longs and pays shorts.
  - Isolated margin is used by default. Each position has its own collateral bucket.
    Funding is added to/subtracted from that position margin, not from cross-account equity.
  - Isolated liquidation price is recomputed hourly from:
        isolated_margin + unrealized_pnl = maintenance_margin
  - Maintenance margin is approximated as:
        sum(abs(qty) * mark_price * maintenance_margin_rate(symbol, notional))
    using a partial table from Hyperliquid docs plus a configurable default.
  - The default N is 3 long + 3 short because you mentioned "days with 6 trades".
  - By default, positions are additive: each new daily signal creates an order of
    fixed notional. Existing same-side positions are merged instead of closed.
    A timed-exit schedule still tracks each daily-signal quantity separately, so
    the quantity added by a signal is closed after --holding-days days by default.
  - User defaults in this file: isolated margin, 1x leverage, $10 notional per
    order, $2,000 starting capital, and a 2025-06-05 UTC backtest start.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Exact Hyperliquid tradable universe supplied by the user.
TICKERS = [
    'ATOM','REQ','CRV','MAVIA','SAGA','NEAR','MORPHO','MANTA','MOVE','XAI',
    'ETC','DOGE','SOPH','CELO','MAV','POPCAT','SCR','COMP','GMT','SOL','IMX',
    'JUP','RUNE','LAUNCHCOIN','UMA','TRB','USTC','AIXBT','IOTA','VIRTUAL',
    'ALGO','GMX','ANIME','BCH','BIO','BSV','NXPC','MOODENG','TNSR','HBAR',
    'SNX','ZEREBRO','HYPER','SAND','BERA','PURR','GAS','LDO','ONDO','DYDX',
    'FTT','TON','EIGEN','LTC','BLAST','AI16Z','OMNI','AAVE','OGN','SUI',
    'MEME','FXS','NEIROETH','NIL','CFX','ME','XRP','TIA','BNB','NOT','IP',
    'OM','TAO','OP','CAKE','AVAX','kPEPE','GALA','MNT','BOME','SUPER','SEI',
    'VINE','KAS','BABY','STX','S','FARTCOIN','STG','RENDER','ENA','LINK',
    'ARB','ARK','BIGTIME','BTC','ETH','RSR','kDOGS','BRETT','BANANA','XLM',
    'INJ','ENS','AR','DOT','SPX','ETHFI','PAXG','kLUNC','GOAT','kSHIB','FIL',
    'MEW','STRK','TRX','ZK','KAITO','PENGU','kBONK','VVV','ORDI','INIT','APT',
    'REZ','LAYER','ZEN','SUSHI','kFLOKI','ADA','kNEIRO','PEOPLE','ZORA',
    'PENDLE','APE','HYPE','FET','CHILLGUY','MELANIA','GRIFFAIN','PNUT','DOOD',
    'WIF','ACE','ZETA','TRUMP','NEO','JTO','YGG','ZRO','PROMPT','WLD','W',
    'MERL','BLUR','UNI','DYM','MINA','MKR','POLYX','POL','IO','TURBO','PYTH',
    'USUAL','GRASS','ALT','HMSTR','WCT','SYRUP','RESOLV','PROVE','YZY','WLFI',
    'TST','PUMP','LINEA','SKY','ASTER','0G','STBL','AVNT','XPL','ZEC','ICP'
]
TICKER_SET = set(TICKERS)

# Case-insensitive canonical map plus aliases for common predictor naming differences.
# Hyperliquid uses a lowercase "k" prefix for some perps such as kPEPE.
CANONICAL_TICKER_BY_UPPER = {sym.upper(): sym for sym in TICKERS}
SYMBOL_ALIASES = {
    # common non-Hyperliquid names -> Hyperliquid perp names
    "PEPE": "kPEPE",
    "1000PEPE": "kPEPE",
    "BONK": "kBONK",
    "1000BONK": "kBONK",
    "SHIB": "kSHIB",
    "1000SHIB": "kSHIB",
    "LUNC": "kLUNC",
    "1000LUNC": "kLUNC",
    "FLOKI": "kFLOKI",
    "1000FLOKI": "kFLOKI",
    "DOGS": "kDOGS",
    "1000DOGS": "kDOGS",
    "NEIRO": "kNEIRO",
    "1000NEIRO": "kNEIRO",
}


# Partial mainnet max-leverage table from Hyperliquid docs.
# For assets not listed here, --default-max-leverage is used.
MAX_LEVERAGE_BY_SYMBOL = {
    "BTC": 40,
    "ETH": 25,
    "SOL": 20,
    "XRP": 20,
}
for _sym in [
    "DOGE", "kPEPE", "SUI", "WLD", "TRUMP", "LTC", "ENA", "POPCAT", "WIF",
    "AAVE", "kBONK", "LINK", "CRV", "AVAX", "ADA", "UNI", "NEAR", "TIA",
    "APT", "BCH", "HYPE", "FARTCOIN",
    "OP", "ARB", "LDO", "TON", "MKR", "ONDO", "JUP", "INJ", "kSHIB", "SEI",
    "TRX", "BNB", "DOT",
]:
    MAX_LEVERAGE_BY_SYMBOL[_sym] = 10


@dataclass
class Position:
    symbol: str
    qty: float                  # signed base-asset quantity: + long, - short
    entry_price: float
    opened_at: pd.Timestamp
    initial_margin: float      # in isolated mode, this is current isolated margin after funding
    cumulative_funding: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_realized_pnl: float = 0.0
    opening_orders: int = 1
    last_prediction: Optional[float] = None
    last_signal_date: Optional[str] = None

    @property
    def side(self) -> int:
        return 1 if self.qty > 0 else -1

    @property
    def abs_qty(self) -> float:
        return abs(self.qty)

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.qty * (mark_price - self.entry_price)

    def notional(self, mark_price: float) -> float:
        return abs(self.qty) * mark_price


@dataclass
class BacktestConfig:
    ranks_path: Path
    ohlc_path: Path
    funding_path: Path
    oracle_path: Path
    output_dir: Path
    rank_source: str = "auto"
    prediction_col: Optional[str] = None
    symbol_col: Optional[str] = None
    date_col: Optional[str] = None
    n_long: int = 3
    n_short: int = 3
    initial_equity: float = 2000.0
    notional_per_trade: float = 10.0
    leverage: float = 1.0
    margin_mode: str = "isolated"
    start_date: Optional[str] = "2025-10-10"
    end_date: Optional[str] = None
    holding_days: Optional[float] = 30.0
    stop_at_last_signal_date: bool = True
    default_max_leverage: float = 5.0
    fee_rate: float = 0.00045
    liquidation_fee_rate: float = 0.0
    entry_hour_utc: int = 0
    max_entry_delay_hours: int = 23
    execution_price_col: str = "open"
    close_at_end: bool = True
    min_prediction_rows_per_day: int = 2
    force_exact_universe: bool = True


class HyperliquidBacktester:
    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self.positions: Dict[str, Position] = {}

        # In isolated mode, cash is free USDC not locked inside positions.
        # In cross mode, cash is the collateral account after realized PnL, fees, and funding.
        self.cash = float(cfg.initial_equity)

        self.total_realized_pnl = 0.0
        self.total_funding_pnl = 0.0
        self.total_fees = 0.0
        self.long_realized_pnl = 0.0
        self.short_realized_pnl = 0.0
        self.long_unrealized_pnl = 0.0
        self.short_unrealized_pnl = 0.0

        self.last_mark: Dict[str, float] = {}
        self.trade_records: List[dict] = []
        self.close_records: List[dict] = []
        self.hourly_records: List[dict] = []
        self.position_records: List[dict] = []
        self.skipped_signal_records: List[dict] = []
        self.timed_exit_schedule: List[dict] = []
        self.timed_exit_records: List[dict] = []
        self.daily_signal_counts: Dict[str, int] = {}
        self.prediction_metadata: Dict[str, object] = {}

    # -----------------------------
    # Data loading
    # -----------------------------

    @staticmethod
    def normalize_symbol(symbol: object) -> Optional[str]:
        """Map raw predictor symbols/ids to Hyperliquid canonical tickers."""
        if pd.isna(symbol):
            return None
        raw = str(symbol).strip()
        if not raw:
            return None

        # Remove frequent exchange quote suffixes without touching symbols like 0G.
        cleaned = raw.replace("/", "-").split(":")[0].strip()
        for suffix in ["-USDC", "-USD", "-USDT", "USDC", "USDT", "USD"]:
            if cleaned.upper().endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)]
                break

        key = cleaned.upper()
        if key in SYMBOL_ALIASES:
            return SYMBOL_ALIASES[key]
        return CANONICAL_TICKER_BY_UPPER.get(key)

    @staticmethod
    def _first_existing_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    @staticmethod
    def _parse_time(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("h")

    @staticmethod
    def _find_col(columns: Iterable[str], candidates: List[str], label: str) -> str:
        lower_map = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        raise ValueError(f"Could not find {label} column. Tried {candidates}; found {list(columns)}")

    def load_predictions(self) -> pd.DataFrame:
        if not self.cfg.ranks_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.cfg.ranks_path}")

        ranks_raw = pd.read_csv(self.cfg.ranks_path)
        if ranks_raw.empty:
            raise ValueError(f"Predictions file is empty: {self.cfg.ranks_path}")

        columns = list(ranks_raw.columns)
        rank_source = (self.cfg.rank_source or "auto").lower()
        if rank_source not in {"auto", "numerai", "crowdcent"}:
            raise ValueError("rank_source must be one of: auto, numerai, crowdcent")

        if rank_source == "auto":
            lower_cols = {c.lower() for c in columns}
            if {"pred_10d", "release_date"}.issubset(lower_cols):
                inferred_source = "crowdcent"
            elif {"symbol", "date", "prediction"}.issubset(lower_cols):
                inferred_source = "numerai"
            else:
                inferred_source = "generic"
        else:
            inferred_source = rank_source

        if self.cfg.symbol_col:
            symbol_col = self._find_col(columns, [self.cfg.symbol_col], "symbol")
        elif inferred_source == "crowdcent":
            # Crowdcent stores the tradable symbol in the id column.
            symbol_col = self._find_col(columns, ["symbol", "ticker", "perp", "asset", "coin", "id"], "symbol")
        else:
            # Numerai has both id and symbol; id is numeric, so prefer symbol.
            symbol_col = self._find_col(columns, ["symbol", "ticker", "perp", "asset", "coin", "id"], "symbol")

        if self.cfg.date_col:
            date_col = self._find_col(columns, [self.cfg.date_col], "date")
        else:
            date_col = self._find_col(
                columns,
                ["date", "release_date", "time", "timestamp", "datetime"],
                "date",
            )

        if self.cfg.prediction_col:
            pred_col = self._find_col(columns, [self.cfg.prediction_col], "prediction")
        elif inferred_source == "crowdcent":
            # Default Crowdcent horizon. Override with --prediction-col pred_30d.
            pred_col = self._find_col(
                columns,
                ["pred_10d", "prediction", "pred", "score", "rank", "signal", "pred_30d"],
                "prediction",
            )
        else:
            pred_col = self._find_col(
                columns,
                ["prediction", "pred", "score", "rank", "signal", "target", "pred_10d", "pred_30d"],
                "prediction",
            )

        ranks = pd.DataFrame({
            "source_id": ranks_raw["id"] if "id" in ranks_raw.columns else pd.NA,
            "raw_symbol": ranks_raw[symbol_col],
            "date": ranks_raw[date_col],
            "prediction": ranks_raw[pred_col],
        })
        ranks["symbol"] = ranks["raw_symbol"].map(self.normalize_symbol)
        ranks["date"] = pd.to_datetime(ranks["date"], utc=True, errors="coerce").dt.date.astype("string")
        ranks["prediction"] = pd.to_numeric(ranks["prediction"], errors="coerce")

        rows_before_cleaning = len(ranks)
        invalid_symbol_rows = int(ranks["symbol"].isna().sum())
        ranks = ranks.dropna(subset=["symbol", "date", "prediction"])

        rows_before_universe_filter = len(ranks)
        ranks = ranks[ranks["symbol"].isin(TICKER_SET)].copy()
        rows_after_universe_filter = len(ranks)

        if ranks.empty:
            raw_sample = sorted(ranks_raw[symbol_col].astype(str).str.strip().dropna().unique().tolist())[:50]
            raise ValueError(
                "No predictions matched the Hyperliquid universe after symbol normalization. "
                f"source={inferred_source}, symbol_col={symbol_col}, date_col={date_col}, "
                f"prediction_col={pred_col}, raw_symbol_sample={raw_sample}"
            )

        if self.cfg.start_date:
            start_day = pd.Timestamp(self.cfg.start_date, tz="UTC").date().isoformat()
            ranks = ranks[ranks["date"] >= start_day].copy()
            if ranks.empty:
                raise ValueError(f"No predictions remain on or after start_date={self.cfg.start_date}.")

        ranks = (
            ranks.sort_values(["date", "symbol", "prediction"])
            .drop_duplicates(["date", "symbol"], keep="last")
            .reset_index(drop=True)
        )

        raw_unique = set(ranks_raw[symbol_col].astype(str).str.strip().dropna().unique())
        normalized_unique = set(filter(None, (self.normalize_symbol(x) for x in raw_unique)))
        unmatched_raw_symbols = sorted(
            x for x in raw_unique
            if self.normalize_symbol(x) is None
        )
        universe_missing_after_normalization = sorted(TICKER_SET - normalized_unique)
        signal_date_min = str(ranks["date"].min()) if not ranks.empty else None
        signal_date_max = str(ranks["date"].max()) if not ranks.empty else None

        self.prediction_metadata = {
            "ranks_path": str(self.cfg.ranks_path),
            "rank_source_requested": self.cfg.rank_source,
            "rank_source_inferred": inferred_source,
            "symbol_col_used": symbol_col,
            "date_col_used": date_col,
            "prediction_col_used": pred_col,
            "rows_before_cleaning": int(rows_before_cleaning),
            "rows_with_unmatched_or_invalid_symbol": invalid_symbol_rows,
            "rows_before_universe_filter": int(rows_before_universe_filter),
            "rows_after_universe_filter_before_start_date": int(rows_after_universe_filter),
            "rows_final": int(len(ranks)),
            "signal_date_min_after_filters": signal_date_min,
            "signal_date_max_after_filters": signal_date_max,
            "unique_raw_symbols": int(len(raw_unique)),
            "unique_normalized_hl_symbols": int(len(normalized_unique)),
            "unmatched_raw_symbols_sample": unmatched_raw_symbols[:100],
            "hyperliquid_symbols_missing_from_predictions_sample": universe_missing_after_normalization[:100],
        }
        return ranks

    def effective_market_end_exclusive(self) -> Optional[pd.Timestamp]:
        """Return exclusive UTC timestamp used to cap hourly market data.

        By default the backtest stops at the end of the last available signal date.
        This prevents the equity curve from drifting after the prediction file ends
        simply because OHLC/funding data continues. Use --continue-after-last-signal
        to restore the previous behavior. If --end-date is supplied, it takes
        precedence and is interpreted as inclusive through that UTC calendar day.
        """
        end_day = None
        if self.cfg.end_date:
            end_day = pd.Timestamp(self.cfg.end_date, tz="UTC").date()
        elif self.cfg.stop_at_last_signal_date:
            max_signal = self.prediction_metadata.get("signal_date_max_after_filters")
            if max_signal:
                end_day = pd.Timestamp(str(max_signal), tz="UTC").date()
                # When timed exits are enabled, keep market data long enough for
                # positions opened on the final signal date to reach their exit.
                if self.cfg.holding_days is not None:
                    if self.cfg.holding_days <= 0:
                        raise ValueError("holding_days must be positive, or disable it with --no-time-exit.")
                    end_day = (
                        pd.Timestamp(end_day, tz="UTC")
                        + pd.Timedelta(days=float(self.cfg.holding_days))
                    ).date()

        if end_day is None:
            return None
        return pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1)

    def load_market_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for p in [self.cfg.ohlc_path, self.cfg.funding_path, self.cfg.oracle_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required market file not found: {p}")

        ohlc = pd.read_csv(self.cfg.ohlc_path)
        funding = pd.read_csv(self.cfg.funding_path)
        oracle = pd.read_csv(self.cfg.oracle_path)

        for name, df, cols in [
            ("OHLC", ohlc, ["perp", "time", "open", "high", "low", "close"]),
            ("funding", funding, ["perp", "time", "fundingRate"]),
            ("oracle", oracle, ["perp", "time", "oraclePx"]),
        ]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} file is missing columns: {missing}")

        for df in [ohlc, funding, oracle]:
            df["perp"] = df["perp"].astype(str).str.strip()
            df["time"] = self._parse_time(df["time"])
            df.dropna(subset=["perp", "time"], inplace=True)
            df = df[df["perp"].isin(TICKER_SET)]

        for col in ["open", "high", "low", "close", "volume", "num_trades"]:
            if col in ohlc.columns:
                ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")
        funding["fundingRate"] = pd.to_numeric(funding["fundingRate"], errors="coerce")
        oracle["oraclePx"] = pd.to_numeric(oracle["oraclePx"], errors="coerce")

        ohlc = ohlc[ohlc["perp"].isin(TICKER_SET)].dropna(subset=["open", "high", "low", "close"])
        funding = funding[funding["perp"].isin(TICKER_SET)].dropna(subset=["fundingRate"])
        oracle = oracle[oracle["perp"].isin(TICKER_SET)].dropna(subset=["oraclePx"])

        ohlc = ohlc.sort_values(["time", "perp"]).drop_duplicates(["time", "perp"], keep="last")
        funding = funding.sort_values(["time", "perp"]).drop_duplicates(["time", "perp"], keep="last")
        oracle = oracle.sort_values(["time", "perp"]).drop_duplicates(["time", "perp"], keep="last")

        if self.cfg.start_date:
            start_ts = pd.Timestamp(self.cfg.start_date, tz="UTC")
            ohlc = ohlc[ohlc["time"] >= start_ts].copy()
            funding = funding[funding["time"] >= start_ts].copy()
            oracle = oracle[oracle["time"] >= start_ts].copy()
            if ohlc.empty:
                raise ValueError(f"No OHLC rows remain on or after start_date={self.cfg.start_date}.")

        end_exclusive = self.effective_market_end_exclusive()
        if end_exclusive is not None:
            ohlc = ohlc[ohlc["time"] < end_exclusive].copy()
            funding = funding[funding["time"] < end_exclusive].copy()
            oracle = oracle[oracle["time"] < end_exclusive].copy()
            if ohlc.empty:
                raise ValueError(
                    "No OHLC rows remain after applying the effective end date. "
                    f"end_exclusive={end_exclusive}, end_date={self.cfg.end_date}, "
                    f"stop_at_last_signal_date={self.cfg.stop_at_last_signal_date}."
                )

        self.prediction_metadata["configured_end_date"] = self.cfg.end_date
        self.prediction_metadata["stop_at_last_signal_date"] = bool(self.cfg.stop_at_last_signal_date)
        self.prediction_metadata["effective_market_end_exclusive"] = str(end_exclusive) if end_exclusive is not None else None
        self.prediction_metadata["market_time_min_after_filters"] = str(ohlc["time"].min()) if not ohlc.empty else None
        self.prediction_metadata["market_time_max_after_filters"] = str(ohlc["time"].max()) if not ohlc.empty else None

        return ohlc, funding, oracle

    # -----------------------------
    # Signal preparation
    # -----------------------------

    def create_daily_signals(self, ranks: pd.DataFrame) -> pd.DataFrame:
        """Create one daily long/short signal table.

        Important zero-trade behavior:
        if n_long == 0 and n_short == 0, this returns an empty signal table and
        records 0 expected signals for each prediction date. The backtest then
        produces a flat account-equity curve instead of silently using old
        parser defaults.
        """
        if self.cfg.n_long < 0 or self.cfg.n_short < 0:
            raise ValueError("n_long and n_short must be >= 0.")

        cols = ["signal_date", "symbol", "prediction", "side", "daily_rank"]
        rows = []

        for date_str, g in ranks.groupby("date", sort=True):
            g = g.sort_values("prediction", ascending=False).reset_index(drop=True)
            if len(g) < self.cfg.min_prediction_rows_per_day:
                self.daily_signal_counts[str(date_str)] = 0
                continue

            parts = []
            if self.cfg.n_long > 0:
                longs = g.head(self.cfg.n_long).copy()
                longs["side"] = 1
                parts.append(longs)
            if self.cfg.n_short > 0:
                shorts = g.tail(self.cfg.n_short).copy().sort_values("prediction", ascending=True)
                shorts["side"] = -1
                parts.append(shorts)

            if not parts:
                self.daily_signal_counts[str(date_str)] = 0
                continue

            day = pd.concat(parts, ignore_index=True)

            # Avoid self-conflict when the universe is smaller than n_long + n_short.
            # If a symbol appears in both long and short legs, keep the higher-ranked
            # occurrence produced first above.
            day = day.drop_duplicates(["symbol"], keep="first")
            day["signal_date"] = date_str
            day["daily_rank"] = np.arange(1, len(day) + 1)
            self.daily_signal_counts[str(date_str)] = int(len(day))
            if not day.empty:
                rows.append(day[cols])

        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.concat(rows, ignore_index=True)

    def build_trade_schedule(self, daily_signals: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
        schedule_cols = ["time", "signal_date", "symbol", "side", "prediction", "price", "notional"]
        if daily_signals.empty:
            self.skipped_signal_records = []
            return pd.DataFrame(columns=schedule_cols)

        ohlc_by_symbol = {sym: g.sort_values("time").copy() for sym, g in ohlc.groupby("perp")}
        scheduled = []
        skipped = []

        price_col = self.cfg.execution_price_col
        if price_col not in ohlc.columns:
            raise ValueError(f"execution_price_col={price_col!r} not found in OHLC file.")

        for row in daily_signals.itertuples(index=False):
            date_ts = pd.Timestamp(str(row.signal_date), tz="UTC")
            start = date_ts + pd.Timedelta(hours=self.cfg.entry_hour_utc)
            end = start + pd.Timedelta(hours=self.cfg.max_entry_delay_hours)

            g = ohlc_by_symbol.get(row.symbol)
            if g is None or g.empty:
                skipped.append({
                    "signal_date": row.signal_date,
                    "symbol": row.symbol,
                    "side": row.side,
                    "prediction": row.prediction,
                    "reason": "symbol_missing_from_ohlc",
                })
                continue

            candidates = g[(g["time"] >= start) & (g["time"] <= end)]
            if candidates.empty:
                skipped.append({
                    "signal_date": row.signal_date,
                    "symbol": row.symbol,
                    "side": row.side,
                    "prediction": row.prediction,
                    "reason": "no_price_inside_entry_window",
                })
                continue

            exec_row = candidates.iloc[0]
            price = float(exec_row[price_col])
            if not np.isfinite(price) or price <= 0:
                skipped.append({
                    "signal_date": row.signal_date,
                    "symbol": row.symbol,
                    "side": row.side,
                    "prediction": row.prediction,
                    "reason": "invalid_execution_price",
                })
                continue

            scheduled.append({
                "time": exec_row["time"],
                "signal_date": row.signal_date,
                "symbol": row.symbol,
                "side": int(row.side),
                "prediction": float(row.prediction),
                "price": price,
                "notional": float(self.cfg.notional_per_trade),
            })

        self.skipped_signal_records = skipped

        if not scheduled:
            if len(daily_signals) == 0:
                return pd.DataFrame(columns=schedule_cols)
            raise ValueError(
                "No trades could be scheduled. Check date overlap between predictions and OHLC data."
            )
        schedule = pd.DataFrame(scheduled).sort_values(["time", "symbol", "side"]).reset_index(drop=True)
        return schedule[schedule_cols]

    # -----------------------------
    # Margin and liquidation helpers
    # -----------------------------

    def max_leverage_for_symbol(self, symbol: str) -> float:
        return float(MAX_LEVERAGE_BY_SYMBOL.get(symbol, self.cfg.default_max_leverage))

    def maintenance_margin_rate(self, symbol: str, notional: float) -> float:
        # Approximation: current docs define maintenance margin rate as half of
        # the initial margin rate at maximum leverage for the tier.
        max_lev = self.max_leverage_for_symbol(symbol)
        return 1.0 / (2.0 * max_lev)

    def position_maintenance_margin(self, pos: Position, mark_price: float) -> float:
        notional = pos.notional(mark_price)
        return notional * self.maintenance_margin_rate(pos.symbol, notional)

    def total_maintenance_margin(self) -> float:
        mm = 0.0
        for sym, pos in self.positions.items():
            mark = self.last_mark.get(sym, pos.entry_price)
            mm += self.position_maintenance_margin(pos, mark)
        return mm

    def total_unrealized_pnl(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            mark = self.last_mark.get(sym, pos.entry_price)
            total += pos.unrealized_pnl(mark)
        return total

    def account_equity(self) -> float:
        if self.cfg.margin_mode == "isolated":
            isolated_equity = 0.0
            for sym, pos in self.positions.items():
                mark = self.last_mark.get(sym, pos.entry_price)
                isolated_equity += pos.initial_margin + pos.unrealized_pnl(mark)
            return self.cash + isolated_equity
        return self.cash + self.total_unrealized_pnl()

    def approximate_liq_price(self, symbol: str, mark_price: float) -> Optional[float]:
        pos = self.positions.get(symbol)
        if pos is None or pos.abs_qty <= 0:
            return None

        notional = pos.notional(mark_price)
        mmr = self.maintenance_margin_rate(symbol, notional)
        side = pos.side

        if self.cfg.margin_mode == "isolated":
            # Liquidation condition for one isolated position:
            #   isolated_margin + side * qty_abs * (price - entry) = qty_abs * price * mmr
            # Solving for price gives:
            #   long:  (entry - margin / qty_abs) / (1 - mmr)
            #   short: (entry + margin / qty_abs) / (1 + mmr)
            margin_per_unit = pos.initial_margin / pos.abs_qty
            denom = 1.0 - side * mmr
            if denom <= 0:
                return None
            liq = (pos.entry_price - side * margin_per_unit) / denom
        else:
            account_value = self.account_equity()
            maintenance_required = self.total_maintenance_margin()
            margin_available = account_value - maintenance_required
            denom = 1.0 - mmr * side
            if denom <= 0:
                return None
            liq = mark_price - side * margin_available / pos.abs_qty / denom

        if not np.isfinite(liq):
            return None
        if liq <= 0:
            # At 1x isolated long margin, liquidation is usually at/near zero.
            return None
        return float(liq)

    # -----------------------------
    # Trading and accounting
    # -----------------------------

    def _required_initial_margin(self, notional: float) -> float:
        return abs(notional) / self.cfg.leverage

    def _has_cash_for(self, required_cash: float) -> bool:
        return self.cash + 1e-12 >= required_cash

    def schedule_timed_exit(
        self,
        opened_at: pd.Timestamp,
        symbol: str,
        side: int,
        qty_abs: float,
        prediction: Optional[float] = None,
        signal_date: Optional[str] = None,
    ) -> None:
        """Schedule a reduce-only close for a newly added daily-signal quantity."""
        if self.cfg.holding_days is None:
            return
        if self.cfg.holding_days <= 0:
            raise ValueError("holding_days must be positive, or disable it with --no-time-exit.")
        if qty_abs <= 0 or side not in (-1, 1):
            return
        close_time = opened_at + pd.Timedelta(days=float(self.cfg.holding_days))
        self.timed_exit_schedule.append({
            "opened_at": opened_at,
            "close_time": close_time,
            "symbol": symbol,
            "side_to_close": int(side),
            "qty_to_close": float(qty_abs),
            "prediction": prediction,
            "signal_date": signal_date,
            "status": "scheduled",
        })

    def process_timed_exits(self, time: pd.Timestamp, bars_now: pd.DataFrame) -> None:
        """Close scheduled daily-signal quantities after the configured holding period.

        This is reduce-only: if the matching exposure was already closed, reduced,
        flipped, or liquidated, the scheduled exit is skipped or only closes the
        remaining available quantity.
        """
        if self.cfg.holding_days is None or not self.timed_exit_schedule:
            return

        bar_map = {row.perp: row for row in bars_now.itertuples(index=False)} if not bars_now.empty else {}
        reason = f"time_exit_{float(self.cfg.holding_days):g}d"

        for exit_order in self.timed_exit_schedule:
            if exit_order.get("status") != "scheduled":
                continue
            close_time = pd.Timestamp(exit_order["close_time"])
            if time < close_time:
                continue

            symbol = exit_order["symbol"]
            pos = self.positions.get(symbol)
            if pos is None:
                exit_order["status"] = "skipped_already_closed"
                exit_order["processed_time"] = time
                self.timed_exit_records.append(dict(exit_order))
                continue
            if pos.side != int(exit_order["side_to_close"]):
                exit_order["status"] = "skipped_position_flipped"
                exit_order["processed_time"] = time
                self.timed_exit_records.append(dict(exit_order))
                continue

            qty_to_close = min(float(exit_order["qty_to_close"]), pos.abs_qty)
            if qty_to_close <= 1e-14:
                exit_order["status"] = "skipped_no_remaining_qty"
                exit_order["processed_time"] = time
                self.timed_exit_records.append(dict(exit_order))
                continue

            price = None
            row = bar_map.get(symbol)
            if row is not None and self.cfg.execution_price_col in bars_now.columns:
                candidate = getattr(row, self.cfg.execution_price_col)
                if np.isfinite(candidate) and candidate > 0:
                    price = float(candidate)
            if price is None:
                price = float(self.last_mark.get(symbol, pos.entry_price))
            if not np.isfinite(price) or price <= 0:
                exit_order["status"] = "skipped_invalid_exit_price"
                exit_order["processed_time"] = time
                self.timed_exit_records.append(dict(exit_order))
                continue

            self.execute_order(
                time=time,
                symbol=symbol,
                side=-pos.side,
                notional=qty_to_close * price,
                price=price,
                prediction=exit_order.get("prediction"),
                signal_date=exit_order.get("signal_date"),
                reason=reason,
            )
            exit_order["status"] = "executed"
            exit_order["processed_time"] = time
            exit_order["exit_price"] = price
            exit_order["qty_closed_requested"] = qty_to_close
            self.timed_exit_records.append(dict(exit_order))

    def execute_order(
        self,
        time: pd.Timestamp,
        symbol: str,
        side: int,
        notional: float,
        price: float,
        prediction: Optional[float] = None,
        signal_date: Optional[str] = None,
        reason: str = "signal",
    ) -> None:
        if not np.isfinite(price) or price <= 0:
            return
        if side not in (-1, 1):
            raise ValueError(f"Invalid side {side}; expected +1 or -1.")
        if self.cfg.leverage <= 0:
            raise ValueError("leverage must be positive.")
        if self.cfg.margin_mode not in {"isolated", "cross"}:
            raise ValueError("margin_mode must be 'isolated' or 'cross'.")

        max_lev = self.max_leverage_for_symbol(symbol)
        if self.cfg.leverage > max_lev:
            self.trade_records.append({
                "time": time,
                "symbol": symbol,
                "side": side,
                "price": price,
                "notional": notional,
                "qty": 0.0,
                "fee": 0.0,
                "realized_pnl": 0.0,
                "reason": reason,
                "action": "skipped_leverage_above_asset_max",
                "prediction": prediction,
                "signal_date": signal_date,
                "max_leverage": max_lev,
                "margin_mode": self.cfg.margin_mode,
            })
            return

        order_abs_qty = abs(notional) / price
        qty = side * order_abs_qty
        fee = abs(notional) * self.cfg.fee_rate
        old_pos = self.positions.get(symbol)
        realized = 0.0
        action = "open"

        # New position or same-side merge: requires fresh isolated collateral.
        if old_pos is None or old_pos.side == side:
            margin = self._required_initial_margin(notional)
            required_cash = fee + (margin if self.cfg.margin_mode == "isolated" else 0.0)
            if not self._has_cash_for(required_cash):
                self.trade_records.append({
                    "time": time,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "notional": abs(notional),
                    "qty": 0.0,
                    "fee": 0.0,
                    "realized_pnl": 0.0,
                    "reason": reason,
                    "action": "skipped_insufficient_free_collateral",
                    "prediction": prediction,
                    "signal_date": signal_date,
                    "max_leverage": max_lev,
                    "margin_mode": self.cfg.margin_mode,
                })
                return

            self.cash -= fee
            if self.cfg.margin_mode == "isolated":
                self.cash -= margin
            self.total_fees += fee

            if old_pos is None:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    entry_price=price,
                    opened_at=time,
                    initial_margin=margin,
                    cumulative_fees=fee,
                    opening_orders=1,
                    last_prediction=prediction,
                    last_signal_date=signal_date,
                )
                action = "open"
            else:
                old_abs = old_pos.abs_qty
                new_abs = old_abs + order_abs_qty
                old_pos.entry_price = (old_abs * old_pos.entry_price + order_abs_qty * price) / new_abs
                old_pos.qty += qty
                old_pos.initial_margin += margin
                old_pos.cumulative_fees += fee
                old_pos.opening_orders += 1
                old_pos.last_prediction = prediction
                old_pos.last_signal_date = signal_date
                action = "merge_same_side"

        else:
            # Opposite-side order: reduce, close, or flip. For isolated margin,
            # the proportional isolated collateral is released when size is closed.
            old_side = old_pos.side
            old_abs = old_pos.abs_qty
            close_abs = min(old_abs, order_abs_qty)
            close_fraction = close_abs / old_abs
            realized = close_abs * old_side * (price - old_pos.entry_price)
            released_margin = old_pos.initial_margin * close_fraction

            residual_signed = old_pos.qty + qty
            flips = abs(residual_signed) > 1e-14 and np.sign(residual_signed) != np.sign(old_pos.qty)
            residual_abs = abs(residual_signed) if flips else 0.0
            residual_notional = residual_abs * price
            residual_margin = self._required_initial_margin(residual_notional) if flips else 0.0

            available_after_close = self.cash
            if self.cfg.margin_mode == "isolated":
                available_after_close += released_margin + realized
            else:
                available_after_close += realized
            required_after_close = fee + (residual_margin if self.cfg.margin_mode == "isolated" else 0.0)

            if available_after_close + 1e-12 < required_after_close:
                self.trade_records.append({
                    "time": time,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "notional": abs(notional),
                    "qty": 0.0,
                    "fee": 0.0,
                    "realized_pnl": 0.0,
                    "reason": reason,
                    "action": "skipped_insufficient_free_collateral_after_close",
                    "prediction": prediction,
                    "signal_date": signal_date,
                    "max_leverage": max_lev,
                    "margin_mode": self.cfg.margin_mode,
                })
                return

            # Apply close/reduce cash flows.
            if self.cfg.margin_mode == "isolated":
                self.cash += released_margin + realized
            else:
                self.cash += realized
            self.cash -= fee
            if flips and self.cfg.margin_mode == "isolated":
                self.cash -= residual_margin

            self.total_fees += fee
            self.total_realized_pnl += realized
            old_pos.cumulative_realized_pnl += realized
            old_pos.cumulative_fees += fee

            if old_side == 1:
                self.long_realized_pnl += realized
            else:
                self.short_realized_pnl += realized

            allocated_close_fee = fee * (close_abs / order_abs_qty) if order_abs_qty > 0 else fee
            self.close_records.append({
                "time": time,
                "symbol": symbol,
                "side_closed": old_side,
                "qty_closed": close_abs,
                "entry_price": old_pos.entry_price,
                "exit_price": price,
                "realized_pnl": realized,
                "fee": allocated_close_fee,
                "net_pnl_after_allocated_fee": realized - allocated_close_fee,
                "reason": reason if reason != "daily_prediction" else "reduce_or_flip_by_signal",
                "opened_at": old_pos.opened_at,
                "holding_hours": (time - old_pos.opened_at) / pd.Timedelta(hours=1),
                "margin_mode": self.cfg.margin_mode,
            })

            old_pos.initial_margin -= released_margin
            old_pos.last_prediction = prediction
            old_pos.last_signal_date = signal_date

            if abs(residual_signed) < 1e-14:
                del self.positions[symbol]
                action = "close_opposite_side"
            elif np.sign(residual_signed) == np.sign(old_pos.qty):
                old_pos.qty = residual_signed
                action = "reduce_opposite_side"
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=residual_signed,
                    entry_price=price,
                    opened_at=time,
                    initial_margin=residual_margin,
                    cumulative_fees=fee - allocated_close_fee,
                    opening_orders=1,
                    last_prediction=prediction,
                    last_signal_date=signal_date,
                )
                action = "flip_opposite_side"

        # Only daily prediction orders create timed exits. Timed-exit orders are
        # reduce-only closes and must not recursively schedule another exit.
        if reason == "daily_prediction" and self.cfg.holding_days is not None:
            if action in {"open", "merge_same_side"}:
                self.schedule_timed_exit(
                    opened_at=time,
                    symbol=symbol,
                    side=side,
                    qty_abs=order_abs_qty,
                    prediction=prediction,
                    signal_date=signal_date,
                )
            elif action == "flip_opposite_side":
                self.schedule_timed_exit(
                    opened_at=time,
                    symbol=symbol,
                    side=side,
                    qty_abs=residual_abs,
                    prediction=prediction,
                    signal_date=signal_date,
                )

        self.trade_records.append({
            "time": time,
            "symbol": symbol,
            "side": side,
            "price": price,
            "notional": abs(notional),
            "qty": qty,
            "fee": fee,
            "realized_pnl": realized,
            "reason": reason,
            "action": action,
            "prediction": prediction,
            "signal_date": signal_date,
            "max_leverage": max_lev,
            "margin_mode": self.cfg.margin_mode,
        })

    def close_position(
        self,
        time: pd.Timestamp,
        symbol: str,
        price: float,
        reason: str,
        fee_rate: Optional[float] = None,
    ) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return

        fee_rate = self.cfg.fee_rate if fee_rate is None else fee_rate
        realized = pos.unrealized_pnl(price)
        notional = pos.notional(price)
        fee = notional * fee_rate

        if self.cfg.margin_mode == "isolated":
            # Release the remaining isolated margin plus realized PnL back to free USDC.
            self.cash += pos.initial_margin + realized - fee
        else:
            self.cash += realized - fee

        self.total_realized_pnl += realized
        self.total_fees += fee
        pos.cumulative_fees += fee
        pos.cumulative_realized_pnl += realized

        if pos.side == 1:
            self.long_realized_pnl += realized
        else:
            self.short_realized_pnl += realized

        self.close_records.append({
            "time": time,
            "symbol": symbol,
            "side_closed": pos.side,
            "qty_closed": pos.abs_qty,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "realized_pnl": realized,
            "fee": fee,
            "net_pnl_after_allocated_fee": realized - fee,
            "reason": reason,
            "opened_at": pos.opened_at,
            "holding_hours": (time - pos.opened_at) / pd.Timedelta(hours=1),
            "released_isolated_margin": pos.initial_margin if self.cfg.margin_mode == "isolated" else 0.0,
            "margin_mode": self.cfg.margin_mode,
        })

        self.trade_records.append({
            "time": time,
            "symbol": symbol,
            "side": -pos.side,
            "price": price,
            "notional": notional,
            "qty": -pos.qty,
            "fee": fee,
            "realized_pnl": realized,
            "reason": reason,
            "action": "forced_close" if reason == "liquidation" else "close",
            "prediction": pos.last_prediction,
            "signal_date": pos.last_signal_date,
            "max_leverage": self.max_leverage_for_symbol(symbol),
            "margin_mode": self.cfg.margin_mode,
        })

        del self.positions[symbol]

    def apply_funding(self, time: pd.Timestamp, funding_now: pd.DataFrame, oracle_now: pd.DataFrame) -> None:
        if funding_now.empty or not self.positions:
            return

        funding_map = dict(zip(funding_now["perp"], funding_now["fundingRate"]))
        oracle_map = dict(zip(oracle_now["perp"], oracle_now["oraclePx"])) if not oracle_now.empty else {}

        funding_liquidations: List[Tuple[str, float]] = []

        for sym, pos in list(self.positions.items()):
            if sym not in funding_map:
                continue
            rate = funding_map[sym]
            oracle_px = oracle_map.get(sym, self.last_mark.get(sym, pos.entry_price))
            if not np.isfinite(rate) or not np.isfinite(oracle_px) or oracle_px <= 0:
                continue

            # Positive funding rate: longs pay, shorts receive.
            funding_pnl = -pos.side * pos.abs_qty * oracle_px * float(rate)
            if self.cfg.margin_mode == "isolated":
                pos.initial_margin += funding_pnl
                # If funding has drained the isolated margin to zero or below,
                # the position must be liquidated. Collect for processing after
                # the loop to avoid mutating self.positions mid-iteration.
                if pos.initial_margin <= 0:
                    funding_liquidations.append((sym, oracle_px))
            else:
                self.cash += funding_pnl
            self.total_funding_pnl += funding_pnl
            pos.cumulative_funding += funding_pnl

        # Trigger funding-induced liquidations at the oracle price used for funding.
        for sym, liq_px in funding_liquidations:
            if sym in self.positions:  # may already be gone if listed twice
                self.close_position(
                    time=time,
                    symbol=sym,
                    price=liq_px,
                    reason="liquidation_funding",
                    fee_rate=self.cfg.liquidation_fee_rate,
                )

    def update_last_marks(self, bars_now: pd.DataFrame) -> None:
        for row in bars_now.itertuples(index=False):
            if np.isfinite(row.close) and row.close > 0:
                self.last_mark[row.perp] = float(row.close)

    def check_liquidations(self, time: pd.Timestamp, bars_now: pd.DataFrame) -> None:
        if bars_now.empty or not self.positions:
            return

        bar_map = {row.perp: row for row in bars_now.itertuples(index=False)}
        # Iterate repeatedly. In cross mode, closing one position changes margin_available
        # for the others; in isolated mode, this simply catches multiple independent hits.
        liquidated_any = True
        passes = 0
        while liquidated_any and passes < 10:
            liquidated_any = False
            passes += 1
            for sym in list(self.positions.keys()):
                if sym not in bar_map:
                    continue
                pos = self.positions[sym]
                row = bar_map[sym]
                mark = self.last_mark.get(sym, pos.entry_price)
                liq = self.approximate_liq_price(sym, mark)
                if liq is None:
                    continue

                hit = False
                # Use intrahour low/high as a proxy for mark-price liquidation risk.
                if pos.side == 1 and np.isfinite(row.low) and row.low <= liq:
                    hit = True
                elif pos.side == -1 and np.isfinite(row.high) and row.high >= liq:
                    hit = True

                if hit:
                    self.close_position(
                        time=time,
                        symbol=sym,
                        price=liq,
                        reason="liquidation",
                        fee_rate=self.cfg.liquidation_fee_rate,
                    )
                    liquidated_any = True
                    break

    # -----------------------------
    # Valuation and outputs
    # -----------------------------

    def current_exposure(self) -> Tuple[float, float, float, float]:
        long_exp = 0.0
        short_exp = 0.0
        for sym, pos in self.positions.items():
            mark = self.last_mark.get(sym, pos.entry_price)
            notion = pos.notional(mark)
            if pos.side == 1:
                long_exp += notion
            else:
                short_exp += notion
        gross = long_exp + short_exp
        net = long_exp - short_exp
        return long_exp, short_exp, gross, net

    def record_hour(self, time: pd.Timestamp) -> None:
        unreal_long = 0.0
        unreal_short = 0.0
        posted_margin = 0.0
        open_funding = 0.0
        open_fees = 0.0

        for sym, pos in self.positions.items():
            mark = self.last_mark.get(sym, pos.entry_price)
            upnl = pos.unrealized_pnl(mark)
            liq_price = self.approximate_liq_price(sym, mark)
            position_equity = pos.initial_margin + upnl if self.cfg.margin_mode == "isolated" else np.nan
            maintenance = self.position_maintenance_margin(pos, mark)
            posted_margin += pos.initial_margin
            open_funding += pos.cumulative_funding
            open_fees += pos.cumulative_fees
            self.position_records.append({
                "time": time,
                "symbol": sym,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "mark_price": mark,
                "notional": pos.notional(mark),
                "isolated_margin": pos.initial_margin if self.cfg.margin_mode == "isolated" else np.nan,
                "position_equity": position_equity,
                "maintenance_margin": maintenance,
                "unrealized_pnl": upnl,
                "cumulative_funding": pos.cumulative_funding,
                "cumulative_fees": pos.cumulative_fees,
                "approx_liquidation_price": liq_price,
                "opening_orders": pos.opening_orders,
                "last_signal_date": pos.last_signal_date,
                "last_prediction": pos.last_prediction,
            })
            if pos.side == 1:
                unreal_long += upnl
            else:
                unreal_short += upnl

        unreal_total = unreal_long + unreal_short
        equity = self.account_equity()
        long_exp, short_exp, gross_exp, net_exp = self.current_exposure()
        maintenance_margin = self.total_maintenance_margin()
        margin_ratio = maintenance_margin / equity if equity > 0 else np.inf

        # User-requested margin-like view. In isolated mode, this is the sum of
        # current isolated margins after funding plus unrealized PnL. In cross mode,
        # it keeps the earlier approximation.
        if self.cfg.margin_mode == "isolated":
            margin_portfolio_value = posted_margin + unreal_total
        else:
            margin_portfolio_value = posted_margin + unreal_total + self.total_funding_pnl - self.total_fees

        self.hourly_records.append({
            "time": time,
            "cash_collateral": self.cash,
            "account_equity": equity,
            "portfolio_margin_value": margin_portfolio_value,
            "posted_initial_margin_open": posted_margin,
            "maintenance_margin_required": maintenance_margin,
            "margin_ratio": margin_ratio,
            "unrealized_pnl_total": unreal_total,
            "unrealized_pnl_long": unreal_long,
            "unrealized_pnl_short": unreal_short,
            "realized_pnl_total": self.total_realized_pnl,
            "realized_pnl_long": self.long_realized_pnl,
            "realized_pnl_short": self.short_realized_pnl,
            "funding_pnl_total": self.total_funding_pnl,
            "fees_total": self.total_fees,
            "n_open_positions": len(self.positions),
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "gross_exposure": gross_exp,
            "net_exposure": net_exp,
            "gross_leverage_on_equity": gross_exp / equity if equity > 0 else np.inf,
        })

    def run(self) -> dict:
        ranks = self.load_predictions()
        ohlc, funding, oracle = self.load_market_data()
        signals = self.create_daily_signals(ranks)
        schedule = self.build_trade_schedule(signals, ohlc)

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        ranks.to_csv(self.cfg.output_dir / "filtered_predictions.csv", index=False)
        signals.to_csv(self.cfg.output_dir / "daily_signals.csv", index=False)
        schedule.to_csv(self.cfg.output_dir / "scheduled_trades.csv", index=False)
        with open(self.cfg.output_dir / "prediction_input_report.json", "w") as f:
            json.dump(make_json_safe(self.prediction_metadata), f, indent=2, default=str)
        if self.skipped_signal_records:
            pd.DataFrame(self.skipped_signal_records).to_csv(
                self.cfg.output_dir / "skipped_signals.csv", index=False
            )

        # Pre-group for fast hourly loop.
        ohlc_by_time = {t: g.copy() for t, g in ohlc.groupby("time", sort=True)}
        funding_by_time = {t: g.copy() for t, g in funding.groupby("time", sort=True)}
        oracle_by_time = {t: g.copy() for t, g in oracle.groupby("time", sort=True)}
        trades_by_time = {t: g.copy() for t, g in schedule.groupby("time", sort=True)}

        all_times = sorted(set(ohlc["time"]).union(set(schedule["time"])))
        if not all_times:
            raise ValueError("No hourly timestamps available for the backtest.")

        for time in all_times:
            bars_now = ohlc_by_time.get(time, pd.DataFrame(columns=ohlc.columns))
            self.update_last_marks(bars_now)

            # Close any daily-signal quantities whose holding period expires at
            # this timestamp before opening fresh positions for the same hour.
            self.process_timed_exits(time, bars_now)

            # Execute scheduled daily trades at the selected hourly candle open.
            trades_now = trades_by_time.get(time)
            if trades_now is not None and not trades_now.empty:
                for tr in trades_now.itertuples(index=False):
                    self.execute_order(
                        time=time,
                        symbol=tr.symbol,
                        side=int(tr.side),
                        notional=float(tr.notional),
                        price=float(tr.price),
                        prediction=float(tr.prediction),
                        signal_date=str(tr.signal_date),
                        reason="daily_prediction",
                    )

            # Funding is hourly.
            self.apply_funding(
                time=time,
                funding_now=funding_by_time.get(time, pd.DataFrame(columns=funding.columns)),
                oracle_now=oracle_by_time.get(time, pd.DataFrame(columns=oracle.columns)),
            )

            # Intrahour liquidation check using OHLC high/low as a mark proxy.
            self.check_liquidations(time, bars_now)

            # Record after trades, funding, and liquidations.
            self.record_hour(time)

            if self.account_equity() <= 0:
                warnings.warn(f"Account equity depleted at {time}. Stopping backtest.")
                break

        if self.cfg.close_at_end and self.positions:
            final_time = all_times[-1]
            for sym in list(self.positions.keys()):
                final_price = self.last_mark.get(sym, self.positions[sym].entry_price)
                self.close_position(final_time, sym, final_price, reason="end_of_backtest")
            self.record_hour(final_time)

        metrics = self.compute_metrics()

        pd.DataFrame(self.trade_records).to_csv(self.cfg.output_dir / "trades.csv", index=False)
        pd.DataFrame(self.close_records).to_csv(self.cfg.output_dir / "closed_positions.csv", index=False)
        self.liquidation_report().to_csv(self.cfg.output_dir / "liquidations.csv", index=False)
        pd.DataFrame(self.timed_exit_schedule).to_csv(self.cfg.output_dir / "timed_exits.csv", index=False)
        pd.DataFrame(self.hourly_records).to_csv(self.cfg.output_dir / "hourly_equity.csv", index=False)
        pd.DataFrame(self.position_records).to_csv(self.cfg.output_dir / "hourly_positions.csv", index=False)

        open_positions_df = pd.DataFrame([asdict(p) for p in self.positions.values()])
        open_positions_df.to_csv(self.cfg.output_dir / "open_positions_final.csv", index=False)

        daily_summary = self.daily_summary()
        daily_summary.to_csv(self.cfg.output_dir / "daily_summary.csv", index=False)

        with open(self.cfg.output_dir / "config.json", "w") as f:
            json.dump(make_json_safe(asdict(self.cfg)), f, indent=2, default=str)

        with open(self.cfg.output_dir / "metrics.json", "w") as f:
            json.dump(round_floats(make_json_safe(metrics), decimals=2), f, indent=2, default=str)

        self.try_plot_equity()
        return metrics

    def daily_summary(self) -> pd.DataFrame:
        if not self.hourly_records:
            return pd.DataFrame()
        h = pd.DataFrame(self.hourly_records)
        h["date"] = pd.to_datetime(h["time"], utc=True).dt.date.astype(str)
        daily = (
            h.sort_values("time")
            .groupby("date", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        daily["daily_return"] = daily["account_equity"].pct_change()
        daily["long_pnl"] = daily["realized_pnl_long"] + daily["unrealized_pnl_long"]
        daily["short_pnl"] = daily["realized_pnl_short"] + daily["unrealized_pnl_short"]
        daily["total_pnl"] = daily["account_equity"] - float(self.cfg.initial_equity)
        return daily

    def liquidation_report(self) -> pd.DataFrame:
        """Return a simple liquidation table with day, token, and USDC amounts.

        `amount` is the net realized PnL of the liquidated position after the
        allocated liquidation fee. It will usually be negative.
        `notional_liquidated` is the position size at the liquidation price.
        """
        columns = [
            "day",
            "time",
            "token",
            "side",
            "amount",
            "notional_liquidated",
            "qty_closed",
            "entry_price",
            "liquidation_price",
            "fee",
            "opened_at",
            "holding_hours",
        ]
        if not self.close_records:
            return pd.DataFrame(columns=columns)

        closed = pd.DataFrame(self.close_records)
        if closed.empty or "reason" not in closed.columns:
            return pd.DataFrame(columns=columns)

        liquidations = closed[closed["reason"].eq("liquidation")].copy()
        if liquidations.empty:
            return pd.DataFrame(columns=columns)

        liquidations["time"] = pd.to_datetime(liquidations["time"], utc=True)
        liquidations["day"] = liquidations["time"].dt.date.astype(str)
        liquidations["token"] = liquidations["symbol"]
        liquidations["side"] = np.where(liquidations["side_closed"].eq(1), "long", "short")
        liquidations["liquidation_price"] = liquidations["exit_price"]
        liquidations["notional_liquidated"] = (
            liquidations["qty_closed"].astype(float) * liquidations["liquidation_price"].astype(float)
        )
        liquidations["amount"] = liquidations["net_pnl_after_allocated_fee"].astype(float)

        return liquidations[columns].sort_values(["time", "token"]).reset_index(drop=True)

    def compute_metrics(self) -> dict:
        h = pd.DataFrame(self.hourly_records)
        trades = pd.DataFrame(self.trade_records)
        closed = pd.DataFrame(self.close_records)

        if h.empty:
            return {}

        h["time"] = pd.to_datetime(h["time"], utc=True)
        h = h.sort_values("time")
        start_equity = float(self.cfg.initial_equity)
        end_equity = float(h["account_equity"].iloc[-1])
        total_return = end_equity / start_equity - 1.0 if start_equity > 0 else np.nan

        elapsed_days = (h["time"].iloc[-1] - h["time"].iloc[0]) / pd.Timedelta(days=1)
        cagr = (end_equity / start_equity) ** (365.25 / elapsed_days) - 1.0 if elapsed_days > 0 and end_equity > 0 else np.nan

        eq = h["account_equity"].astype(float)
        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        max_drawdown = float(drawdown.min())

        hourly_returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        sharpe_hourly = np.nan
        sortino_hourly = np.nan
        calmar = np.nan
        if len(hourly_returns) > 2:
            std_pos = hourly_returns.std(ddof=1)
            if std_pos > 0:
                sharpe_hourly = float(hourly_returns.mean() / std_pos * math.sqrt(24 * 365.25))
            downside = hourly_returns[hourly_returns < 0]
            if len(downside) > 1:
                downside_std = float(np.sqrt((downside ** 2).mean()))
                if downside_std > 0:
                    annualized_mean = float(hourly_returns.mean()) * 24 * 365.25
                    annualized_down = downside_std * math.sqrt(24 * 365.25)
                    sortino_hourly = annualized_mean / annualized_down
        if not np.isnan(cagr) and max_drawdown < 0:
            calmar = cagr / abs(max_drawdown)

        # Daily trade coverage.
        expected_daily_trades = int(self.cfg.n_long + self.cfg.n_short)
        if not trades.empty:
            action_str = trades.get("action", pd.Series("", index=trades.index)).astype(str)
            sched = trades[trades["reason"].eq("daily_prediction") & ~action_str.str.startswith("skipped")]
        else:
            sched = pd.DataFrame()
        if not sched.empty and "signal_date" in sched.columns:
            executed_by_day = sched.groupby("signal_date").size()
        else:
            executed_by_day = pd.Series(dtype=int)

        all_signal_days = pd.Series(self.daily_signal_counts, dtype=int)
        if expected_daily_trades == 0:
            days_expected_or_more = int(len(all_signal_days))
            days_less_than_expected = 0
            days_with_signals_but_no_trades = 0
        else:
            days_expected_or_more = int((executed_by_day >= expected_daily_trades).sum())
            days_less_than_expected = int((executed_by_day < expected_daily_trades).sum())
            days_with_signals_but_no_trades = int(len(set(all_signal_days.index) - set(executed_by_day.index)))

        # Closed-position win rates.
        if not closed.empty:
            pnl = closed["net_pnl_after_allocated_fee"].astype(float)
            winners = pnl > 0
            gross_profit = float(pnl[pnl > 0].sum())
            gross_loss = float(pnl[pnl < 0].sum())
            profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
            win_rate = float(winners.mean())
            long_closed = closed[closed["side_closed"].eq(1)]
            short_closed = closed[closed["side_closed"].eq(-1)]
            long_win_rate = (
                float((long_closed["net_pnl_after_allocated_fee"].astype(float) > 0).mean())
                if not long_closed.empty else np.nan
            )
            short_win_rate = (
                float((short_closed["net_pnl_after_allocated_fee"].astype(float) > 0).mean())
                if not short_closed.empty else np.nan
            )
            avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
            avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
            n_liquidations = int(closed["reason"].isin(["liquidation", "liquidation_funding"]).sum())
            n_liquidations_price = int((closed["reason"] == "liquidation").sum())
            n_liquidations_funding = int((closed["reason"] == "liquidation_funding").sum())
            # Average holding duration by side (hours)
            if "holding_hours" in closed.columns:
                avg_hold_long  = float(long_closed["holding_hours"].mean())  if not long_closed.empty  else np.nan
                avg_hold_short = float(short_closed["holding_hours"].mean()) if not short_closed.empty else np.nan
            else:
                avg_hold_long = avg_hold_short = np.nan
            # Expectancy per trade
            expectancy = float(win_rate * avg_win + (1 - win_rate) * avg_loss) if not np.isnan(win_rate) else np.nan
        else:
            profit_factor = np.nan
            win_rate = np.nan
            long_win_rate = np.nan
            short_win_rate = np.nan
            avg_win = np.nan
            avg_loss = np.nan
            n_liquidations = 0
            n_liquidations_price = 0
            n_liquidations_funding = 0
            avg_hold_long = np.nan
            avg_hold_short = np.nan
            expectancy = np.nan

        metrics = {
            "initial_equity": start_equity,
            "margin_mode": self.cfg.margin_mode,
            "configured_start_date": self.cfg.start_date,
            "configured_end_date": self.cfg.end_date,
            "holding_days": self.cfg.holding_days,
            "num_timed_exit_schedule_rows": int(len(self.timed_exit_schedule)),
            "num_timed_exit_executed": int(sum(1 for x in self.timed_exit_records if x.get("status") == "executed")),
            "stop_at_last_signal_date": bool(self.cfg.stop_at_last_signal_date),
            "effective_market_end_exclusive": self.prediction_metadata.get("effective_market_end_exclusive"),
            "signal_date_max_after_filters": self.prediction_metadata.get("signal_date_max_after_filters"),
            "rank_source_inferred": self.prediction_metadata.get("rank_source_inferred"),
            "prediction_col_used": self.prediction_metadata.get("prediction_col_used"),
            "symbol_col_used": self.prediction_metadata.get("symbol_col_used"),
            "date_col_used": self.prediction_metadata.get("date_col_used"),
            "leverage": float(self.cfg.leverage),
            "notional_per_trade": float(self.cfg.notional_per_trade),
            "final_equity": end_equity,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "hourly_sharpe_annualized": sharpe_hourly,
            "total_realized_pnl": float(self.total_realized_pnl),
            "long_realized_pnl": float(self.long_realized_pnl),
            "short_realized_pnl": float(self.short_realized_pnl),
            "total_funding_pnl": float(self.total_funding_pnl),
            "total_fees": float(self.total_fees),
            "num_order_events": int(len(trades)),
            "num_daily_prediction_orders": int(len(sched)),
            "num_closed_position_events": int(len(closed)),
            "num_liquidations": n_liquidations,
            "closed_position_win_rate": win_rate,
            "closed_position_long_win_rate": long_win_rate,
            "closed_position_short_win_rate": short_win_rate,
            "profit_factor_closed_positions": profit_factor,
            "avg_closed_winner": avg_win,
            "avg_closed_loser": avg_loss,
            "expected_daily_trades": expected_daily_trades,
            "days_with_expected_trades_or_more": days_expected_or_more,
            "days_with_less_than_expected_trades": days_less_than_expected,
            "days_with_signals_but_no_executed_trades": days_with_signals_but_no_trades,
            "num_skipped_signals": int(len(self.skipped_signal_records)),
            "backtest_start": str(h["time"].iloc[0]),
            "backtest_end": str(h["time"].iloc[-1]),
            "elapsed_days": float(elapsed_days),
            "max_gross_leverage_on_equity": float(h["gross_leverage_on_equity"].replace([np.inf, -np.inf], np.nan).max()),
            # --- risk-adjusted return metrics ---
            "hourly_sortino_annualized": sortino_hourly,
            "calmar_ratio": calmar,
            # --- liquidation breakdown ---
            "num_liquidations_price_triggered": n_liquidations_price,
            "num_liquidations_funding_triggered": n_liquidations_funding,
            # --- trade quality ---
            "expectancy_per_trade_usdc": expectancy,
            "avg_holding_hours_long": avg_hold_long,
            "avg_holding_hours_short": avg_hold_short,
        }
        return metrics

    def try_plot_equity(self) -> None:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            h = pd.DataFrame(self.hourly_records)
            if h.empty:
                return
            h["time"] = pd.to_datetime(h["time"], utc=True)
            h = h.sort_values("time").reset_index(drop=True)
            h["long_pnl"] = h["realized_pnl_long"] + h["unrealized_pnl_long"]
            h["short_pnl"] = h["realized_pnl_short"] + h["unrealized_pnl_short"]
            h["total_pnl"] = h["account_equity"] - float(self.cfg.initial_equity)

            eq = h["account_equity"].astype(float)
            running_max = eq.cummax()
            drawdown_pct = (eq / running_max - 1.0) * 100.0

            # Liquidation markers used in the equity and PnL charts.
            # A small red point is placed directly on the corresponding curve.
            liquidation_markers = pd.DataFrame(columns=["time", "label"])
            if self.close_records:
                closed_for_markers = pd.DataFrame(self.close_records)
                required_cols = {"time", "symbol", "reason"}
                if required_cols.issubset(closed_for_markers.columns):
                    liq = closed_for_markers[
                        closed_for_markers["reason"].astype(str).str.startswith("liquidation")
                    ].copy()
                    if not liq.empty:
                        liq["time"] = pd.to_datetime(liq["time"], utc=True)
                        liq["token_label"] = liq["symbol"].astype(str)
                        if "reason" in liq.columns:
                            liq.loc[liq["reason"].eq("liquidation_funding"), "token_label"] += " (funding)"
                        if "side_closed" in liq.columns:
                            side_label = liq["side_closed"].map({1: "long", -1: "short"}).fillna("")
                            liq["token_label"] = liq["token_label"] + np.where(
                                side_label.ne(""), " " + side_label.astype(str), ""
                            )
                        liquidation_markers = (
                            liq.sort_values(["time", "token_label"])
                            .groupby("time", as_index=False)["token_label"]
                            .agg(lambda s: ", ".join(s))
                            .rename(columns={"token_label": "label"})
                        )

            def add_liquidation_markers(ax, y_values: pd.Series) -> None:
                """Draw small red liquidation markers directly on a plotted curve."""
                if liquidation_markers.empty:
                    return
                curve_points = pd.DataFrame({
                    "time": pd.to_datetime(h["time"], utc=True),
                    "y": pd.to_numeric(y_values, errors="coerce"),
                })
                marker_points = (
                    liquidation_markers[["time"]]
                    .merge(curve_points, on="time", how="left")
                    .dropna(subset=["y"])
                    .drop_duplicates(subset=["time"])
                )
                if marker_points.empty:
                    return
                ax.scatter(
                    marker_points["time"],
                    marker_points["y"],
                    color="#D32F2F",
                    s=80,
                    marker="X",
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=6,
                    label="Liquidation",
                )

            # ------------------------------------------------------------------
            # 1) Account equity only
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(h["time"], eq, color="#2196F3", linewidth=1.5, label="Account equity")
            ax.axhline(float(self.cfg.initial_equity), color="gray", linewidth=1,
                       linestyle="--", alpha=0.6, label="Initial equity")
            add_liquidation_markers(ax, eq)
            ax.set_title("Account Equity", fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("USDC")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.legend()
            ax.grid(True, alpha=0.20)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(self.cfg.output_dir / "equity_curve.png", dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------------
            # 2) Portfolio margin value (separate image)
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(h["time"], h["portfolio_margin_value"], color="#FF9800", linewidth=1.5,
                    label="Portfolio margin value")
            ax.set_title("Portfolio Margin Value", fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("USDC")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.legend()
            ax.grid(True, alpha=0.20)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(self.cfg.output_dir / "portfolio_margin.png", dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------------
            # 3) Long / short cumulative PnL
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(h["time"], h["long_pnl"],  color="#4CAF50", linewidth=1.4, label="Long PnL")
            ax.plot(h["time"], h["short_pnl"], color="#F44336", linewidth=1.4, label="Short PnL")
            ax.plot(h["time"], h["total_pnl"], color="#9C27B0", linewidth=1.8,
                    label="Total PnL", alpha=0.85)
            ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
            add_liquidation_markers(ax, h["total_pnl"])
            ax.set_title("Long / Short Cumulative PnL", fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("USDC")
            ax.legend()
            ax.grid(True, alpha=0.20)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(self.cfg.output_dir / "pnl_curve.png", dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------------
            # 4) Per-trade return distribution — long vs short
            # ------------------------------------------------------------------
            if self.close_records:
                closed = pd.DataFrame(self.close_records)
                closed["net_pnl"] = closed["net_pnl_after_allocated_fee"].astype(float)
                long_ret  = closed.loc[closed["side_closed"].eq(1),  "net_pnl"].dropna()
                short_ret = closed.loc[closed["side_closed"].eq(-1), "net_pnl"].dropna()

                fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
                for ax_sub, series, color, title in [
                    (axes[0], long_ret,  "#4CAF50", "Long trades"),
                    (axes[1], short_ret, "#F44336", "Short trades"),
                ]:
                    if series.empty:
                        ax_sub.set_title(f"{title} — no data")
                        continue
                    n_bins = min(40, max(10, len(series) // 5))
                    ax_sub.hist(series, bins=n_bins, color=color, edgecolor="white",
                                linewidth=0.5, alpha=0.85)
                    ax_sub.axvline(0, color="black", linewidth=1.2, linestyle="--")
                    ax_sub.axvline(float(series.mean()), color="navy", linewidth=1.2,
                                   linestyle="-", label=f"Mean: {series.mean():.4f}")
                    ax_sub.set_title(f"Return distribution — {title}", fontsize=11, fontweight="bold")
                    ax_sub.set_xlabel("Net PnL per trade (USDC)")
                    ax_sub.set_ylabel("Count")
                    ax_sub.legend(fontsize=9)
                    ax_sub.grid(True, alpha=0.20)

                    wr = (series > 0).mean() * 100
                    ax_sub.text(0.97, 0.95, f"Win rate: {wr:.1f}%\nn={len(series)}",
                                transform=ax_sub.transAxes, fontsize=9,
                                verticalalignment="top", horizontalalignment="right",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

                fig.suptitle("Per-Trade Return Distribution", fontsize=13, fontweight="bold")
                fig.tight_layout()
                fig.savefig(self.cfg.output_dir / "returns_distribution.png", dpi=150)
                plt.close(fig)

            # ------------------------------------------------------------------
            # 5) Drawdown over time
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(h["time"], drawdown_pct, 0, color="#F44336", alpha=0.4,
                            label="Drawdown")
            ax.plot(h["time"], drawdown_pct, color="#F44336", linewidth=0.8)
            ax.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("Drawdown (%)")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.legend()
            ax.grid(True, alpha=0.20)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(self.cfg.output_dir / "drawdown.png", dpi=150)
            plt.close(fig)

            # ------------------------------------------------------------------
            # 6) Cumulative funding PnL
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(h["time"], h["funding_pnl_total"], color="#607D8B", linewidth=1.5,
                    label="Cumulative funding PnL")
            ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
            ax.fill_between(h["time"], h["funding_pnl_total"], 0,
                            where=h["funding_pnl_total"] < 0, color="#F44336", alpha=0.15)
            ax.fill_between(h["time"], h["funding_pnl_total"], 0,
                            where=h["funding_pnl_total"] >= 0, color="#4CAF50", alpha=0.15)
            ax.set_title("Cumulative Funding PnL", fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("USDC")
            ax.legend()
            ax.grid(True, alpha=0.20)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(self.cfg.output_dir / "funding_pnl.png", dpi=150)
            plt.close(fig)

        except Exception as exc:
            warnings.warn(f"Could not create plots: {exc}")


def make_json_safe(obj):
    """Convert NaN/Infinity numpy values to None for strict JSON outputs."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def round_floats(obj, decimals: int = 2):
    """Recursively round all float values in a JSON-safe structure to `decimals` places."""
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, decimals) for v in obj]
    if isinstance(obj, float):
        return round(obj, decimals) if obj is not None else None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest Hyperliquid daily long/short predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--rank-source", type=str, default="auto", choices=["auto", "numerai", "crowdcent"],
                        help="Prediction/rank file schema. Use crowdcent for ranks_crowdcent.csv, numerai for ranks_numerai.csv, or auto to infer from columns.")
    parser.add_argument("--ranks-path", type=Path, default=None)
    parser.add_argument("--prediction-col", type=str, default=None,
                        help="Prediction column to rank on. Examples: prediction, pred_10d, pred_30d.")
    parser.add_argument("--symbol-col", type=str, default=None,
                        help="Raw symbol column. Numerai usually uses symbol; Crowdcent usually uses id.")
    parser.add_argument("--date-col", type=str, default=None,
                        help="Signal date column. Numerai usually uses date; Crowdcent usually uses release_date.")
    parser.add_argument("--ohlc-path", type=Path, default=None)
    parser.add_argument("--funding-path", type=Path, default=None)
    parser.add_argument("--oracle-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results_hyperliquid_backtest"))

    # Strategy/capital defaults are pulled from BacktestConfig.
    # This means that if you edit the dataclass defaults above, the command-line
    # defaults change too. Explicit CLI flags still take precedence.
    parser.add_argument("--n-long", type=int, default=BacktestConfig.__dataclass_fields__["n_long"].default)
    parser.add_argument("--n-short", type=int, default=BacktestConfig.__dataclass_fields__["n_short"].default)
    parser.add_argument("--initial-equity", type=float, default=BacktestConfig.__dataclass_fields__["initial_equity"].default)
    parser.add_argument("--notional-per-trade", type=float, default=BacktestConfig.__dataclass_fields__["notional_per_trade"].default)
    parser.add_argument("--leverage", type=float, default=BacktestConfig.__dataclass_fields__["leverage"].default)
    parser.add_argument("--margin-mode", type=str, default=BacktestConfig.__dataclass_fields__["margin_mode"].default, choices=["isolated", "cross"])
    parser.add_argument("--start-date", type=str, default=BacktestConfig.__dataclass_fields__["start_date"].default)
    parser.add_argument("--end-date", type=str, default=None,
                        help="Optional inclusive UTC end date, e.g. 2025-12-31. If omitted, the default is to stop at the last signal date plus the holding period when timed exits are enabled.")
    parser.add_argument("--holding-days", type=float, default=BacktestConfig.__dataclass_fields__["holding_days"].default,
                        help="Close each newly opened daily-signal quantity after this many days. Use 30 for pred_30d and 10 for pred_10d.")
    parser.add_argument("--no-time-exit", action="store_true",
                        help="Disable holding-period exits and only close by opposite signal, liquidation, or end-of-backtest.")
    parser.add_argument("--continue-after-last-signal", action="store_true",
                        help="Use all available market data after the final signal date. This restores the previous behavior.")
    parser.add_argument("--default-max-leverage", type=float, default=5.0)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--liquidation-fee-rate", type=float, default=0.0)

    parser.add_argument("--entry-hour-utc", type=int, default=0)
    parser.add_argument("--max-entry-delay-hours", type=int, default=23)
    parser.add_argument("--execution-price-col", type=str, default="open", choices=["open", "high", "low", "close"])
    parser.add_argument("--no-close-at-end", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if args.ranks_path is not None:
        ranks_path = args.ranks_path
    elif args.rank_source == "crowdcent":
        ranks_path = data_dir / "ranks_crowdcent.csv"
    else:
        ranks_path = data_dir / "ranks_numerai.csv"

    holding_days = None if args.no_time_exit else args.holding_days
    if holding_days is not None and holding_days <= 0:
        raise ValueError("--holding-days must be positive, or use --no-time-exit to disable timed exits.")

    cfg = BacktestConfig(
        ranks_path=ranks_path,
        ohlc_path=args.ohlc_path or data_dir / "perps_prices_1h_ohlc.csv",
        funding_path=args.funding_path or data_dir / "all_perps_hourly_funding.csv",
        oracle_path=args.oracle_path or data_dir / "oracle_price.csv",
        output_dir=args.output_dir,
        rank_source=args.rank_source,
        prediction_col=args.prediction_col,
        symbol_col=args.symbol_col,
        date_col=args.date_col,
        n_long=args.n_long,
        n_short=args.n_short,
        initial_equity=args.initial_equity,
        notional_per_trade=args.notional_per_trade,
        leverage=args.leverage,
        margin_mode=args.margin_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        holding_days=holding_days,
        stop_at_last_signal_date=not args.continue_after_last_signal,
        default_max_leverage=args.default_max_leverage,
        fee_rate=args.fee_rate,
        liquidation_fee_rate=args.liquidation_fee_rate,
        entry_hour_utc=args.entry_hour_utc,
        max_entry_delay_hours=args.max_entry_delay_hours,
        execution_price_col=args.execution_price_col,
        close_at_end=not args.no_close_at_end,
    )

    bt = HyperliquidBacktester(cfg)
    metrics = bt.run()

    print("\nBacktest complete.")
    print(f"Output directory: {cfg.output_dir.resolve()}")
    print(json.dumps(make_json_safe(metrics), indent=2, default=str))


if __name__ == "__main__":
    main()
