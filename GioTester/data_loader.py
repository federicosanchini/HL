from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DataConfig

# ————————————————————————————————————————————————————————————————————————— #
# Hyperliquid universe
# ————————————————————————————————————————————————————————————————————————— #

TICKERS = [
    "ATOM",
    "REQ",
    "CRV",
    "MAVIA",
    "SAGA",
    "NEAR",
    "MORPHO",
    "MANTA",
    "MOVE",
    "XAI",
    "ETC",
    "DOGE",
    "SOPH",
    "CELO",
    "MAV",
    "POPCAT",
    "SCR",
    "COMP",
    "GMT",
    "SOL",
    "IMX",
    "JUP",
    "RUNE",
    "LAUNCHCOIN",
    "UMA",
    "TRB",
    "USTC",
    "AIXBT",
    "IOTA",
    "VIRTUAL",
    "ALGO",
    "GMX",
    "ANIME",
    "BCH",
    "BIO",
    "BSV",
    "NXPC",
    "MOODENG",
    "TNSR",
    "HBAR",
    "SNX",
    "ZEREBRO",
    "HYPER",
    "SAND",
    "BERA",
    "PURR",
    "GAS",
    "LDO",
    "ONDO",
    "DYDX",
    "FTT",
    "TON",
    "EIGEN",
    "LTC",
    "BLAST",
    "AI16Z",
    "OMNI",
    "AAVE",
    "OGN",
    "SUI",
    "MEME",
    "FXS",
    "NEIROETH",
    "NIL",
    "CFX",
    "ME",
    "XRP",
    "TIA",
    "BNB",
    "NOT",
    "IP",
    "OM",
    "TAO",
    "OP",
    "CAKE",
    "AVAX",
    "kPEPE",
    "GALA",
    "MNT",
    "BOME",
    "SUPER",
    "SEI",
    "VINE",
    "KAS",
    "BABY",
    "STX",
    "S",
    "FARTCOIN",
    "STG",
    "RENDER",
    "ENA",
    "LINK",
    "ARB",
    "ARK",
    "BIGTIME",
    "BTC",
    "ETH",
    "RSR",
    "kDOGS",
    "BRETT",
    "BANANA",
    "XLM",
    "INJ",
    "ENS",
    "AR",
    "DOT",
    "SPX",
    "ETHFI",
    "PAXG",
    "kLUNC",
    "GOAT",
    "kSHIB",
    "FIL",
    "MEW",
    "STRK",
    "TRX",
    "ZK",
    "KAITO",
    "PENGU",
    "kBONK",
    "VVV",
    "ORDI",
    "INIT",
    "APT",
    "REZ",
    "LAYER",
    "ZEN",
    "SUSHI",
    "kFLOKI",
    "ADA",
    "kNEIRO",
    "PEOPLE",
    "ZORA",
    "PENDLE",
    "APE",
    "HYPE",
    "FET",
    "CHILLGUY",
    "MELANIA",
    "GRIFFAIN",
    "PNUT",
    "DOOD",
    "WIF",
    "ACE",
    "ZETA",
    "TRUMP",
    "NEO",
    "JTO",
    "YGG",
    "ZRO",
    "PROMPT",
    "WLD",
    "W",
    "MERL",
    "BLUR",
    "UNI",
    "DYM",
    "MINA",
    "MKR",
    "POLYX",
    "POL",
    "IO",
    "TURBO",
    "PYTH",
    "USUAL",
    "GRASS",
    "ALT",
    "HMSTR",
    "WCT",
    "SYRUP",
    "RESOLV",
    "PROVE",
    "YZY",
    "WLFI",
    "TST",
    "PUMP",
    "LINEA",
    "SKY",
    "ASTER",
    "0G",
    "STBL",
    "AVNT",
    "XPL",
    "ZEC",
    "ICP",
]
TICKER_SET = set(TICKERS)
CANONICAL_TICKER_BY_UPPER = {sym.upper(): sym for sym in TICKERS}
SYMBOL_ALIASES = {
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


# ————————————————————————————————————————————————————————————————————————— #
# Result dataclasses
# ————————————————————————————————————————————————————————————————————————— #


@dataclass
class PredictionLoadResult:
    ranks: pd.DataFrame
    metadata: dict


@dataclass
class MarketData:
    ohlc: pd.DataFrame
    funding: pd.DataFrame
    oracle: pd.DataFrame


# ————————————————————————————————————————————————————————————————————————— #
# DataLoader
# ————————————————————————————————————————————————————————————————————————— #


class DataLoader:
    def __init__(self, cfg: DataConfig) -> None:
        """Initialise loader with the given data configuration.

        Args:
            cfg: DataConfig instance specifying input paths, date window,
                 signal selection, and trade scheduling parameters.
        """
        self.cfg = cfg
        # Populated by create_daily_signals; read by backtester for trade-coverage metrics.
        self.daily_signal_counts: Dict[pd.Timestamp, int] = {}
        # Populated by build_trade_schedule; read by backtester for skip diagnostics.
        self.skipped_signals: List[dict] = []

    # ———————————————————————————————————————————————————————————————— #
    # Private helpers
    # ———————————————————————————————————————————————————————————————— #

    @staticmethod
    def _normalize_symbol(symbol: object) -> Optional[str]:
        """Map a raw predictor symbol to its canonical Hyperliquid ticker.

        Strips exchange suffixes (e.g. -USDT), applies SYMBOL_ALIASES
        (e.g. PEPE -> kPEPE), then looks up CANONICAL_TICKER_BY_UPPER.

        Args:
            symbol: Raw symbol value from the predictions CSV (any type).

        Returns:
            Canonical HL ticker string, or None if unrecognised.
        """
        if pd.isna(symbol):
            return None
        raw = str(symbol).strip()
        if not raw:
            return None
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
    def _find_col(columns: Iterable[str], candidates: List[str], label: str) -> str:
        """Return the first matching column name from a priority candidate list.

        Matching is case-insensitive; the original casing of the matched column
        is returned so it can be used directly for DataFrame indexing.

        Args:
            columns:    Iterable of column names present in the DataFrame.
            candidates: Ordered list of preferred column name patterns.
            label:      Human-readable description used in the error message.

        Returns:
            The first matched column name (original casing).

        Raises:
            ValueError: If none of the candidates match any column.
        """
        lower_map = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        raise ValueError(
            f"Could not find {label} column. Tried {candidates}; found {list(columns)}"
        )

    @staticmethod
    def _first_existing_col(
        columns: Iterable[str], candidates: List[str]
    ) -> Optional[str]:
        """Return the first matching column name, or None if no match found.

        Non-raising variant of _find_col for optional column detection.

        Args:
            columns:    Iterable of column names present in the DataFrame.
            candidates: Ordered list of preferred column name patterns.

        Returns:
            First matched column name (original casing), or None.
        """
        lower_map = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    # ———————————————————————————————————————————————————————————————— #
    # load_predictions
    # ———————————————————————————————————————————————————————————————— #

    def load_predictions(self) -> PredictionLoadResult:
        """Load, normalise, and filter the predictions CSV.

        Auto-detects source format (crowdcent / numerai) to resolve column
        names, maps raw symbols to canonical HL tickers, coerces types, drops
        invalid rows, clips to start_date, and deduplicates (date, symbol) pairs
        by keeping the highest prediction. Populates a diagnostic metadata dict.

        Returns:
            PredictionLoadResult with:
                ranks:    DataFrame[source_id, raw_symbol, date, prediction, symbol]
                          date is pd.Timestamp at midnight UTC.
                metadata: dict of parsing stats and column-resolution details.

        Raises:
            FileNotFoundError: ranks_path does not exist.
            ValueError:        File empty, no symbols matched HL universe,
                               or no rows remain after start_date filter.
        """
        if not self.cfg.ranks_path.exists():
            raise FileNotFoundError(
                f"Predictions file not found: {self.cfg.ranks_path}"
            )

        ranks_raw = pd.read_csv(self.cfg.ranks_path)
        if ranks_raw.empty:
            raise ValueError(f"Predictions file is empty: {self.cfg.ranks_path}")

        columns = list(ranks_raw.columns)
        lower_cols = {c.lower() for c in columns}

        rank_source = (self.cfg.rank_source or "auto").lower()
        if rank_source not in {"auto", "numerai", "crowdcent"}:
            raise ValueError("rank_source must be one of: auto, numerai, crowdcent")

        # Source inference only determines pred_col candidate ordering.
        # crowdcent detected by presence of pred_10d; all other column resolution is source-independent.
        if rank_source == "auto":
            inferred_source = "crowdcent" if "pred_10d" in lower_cols else "numerai"
        else:
            inferred_source = rank_source

        symbol_col = self._find_col(
            columns,
            (
                [self.cfg.symbol_col]
                if self.cfg.symbol_col
                else ["symbol", "ticker", "perp", "asset", "coin", "id"]
            ),
            "symbol",
        )
        date_col = self._find_col(
            columns,
            (
                [self.cfg.date_col]
                if self.cfg.date_col
                else ["date", "release_date", "time", "timestamp", "datetime"]
            ),
            "date",
        )

        if self.cfg.prediction_col:
            pred_col = self._find_col(columns, [self.cfg.prediction_col], "prediction")
        elif inferred_source == "crowdcent":
            pred_col = self._find_col(
                columns,
                [
                    "pred_10d",
                    "prediction",
                    "pred",
                    "score",
                    "rank",
                    "signal",
                    "pred_30d",
                ],
                "prediction",
            )
        else:
            pred_col = self._find_col(
                columns,
                [
                    "prediction",
                    "pred",
                    "score",
                    "rank",
                    "signal",
                    "target",
                    "pred_10d",
                    "pred_30d",
                ],
                "prediction",
            )

        ranks = pd.DataFrame(
            {
                "source_id": ranks_raw["id"] if "id" in ranks_raw.columns else pd.NA,
                "raw_symbol": ranks_raw[symbol_col],
                "date": ranks_raw[date_col],
                "prediction": ranks_raw[pred_col],
            }
        )

        ranks["symbol"] = ranks["raw_symbol"].map(self._normalize_symbol)
        # Unified date type: midnight UTC Timestamp, consistent with market data time index.
        ranks["date"] = pd.to_datetime(
            ranks["date"], utc=True, errors="coerce"
        ).dt.floor("D")
        ranks["prediction"] = pd.to_numeric(ranks["prediction"], errors="coerce")

        rows_before_cleaning = len(ranks)
        invalid_symbol_rows = int(ranks["symbol"].isna().sum())
        # _normalize_symbol returns only valid HL tickers or None; no TICKER_SET filter needed.
        ranks = ranks.dropna(subset=["symbol", "date", "prediction"])

        if ranks.empty:
            raw_sample = (
                ranks_raw[symbol_col].astype(str).str.strip().dropna().unique().tolist()
            )
            raise ValueError(
                "No predictions matched the Hyperliquid universe after symbol normalization. "
                f"source={inferred_source}, symbol_col={symbol_col}, "
                f"prediction_col={pred_col}, raw_symbol_sample={sorted(raw_sample)[:50]}"
            )

        if self.cfg.start_date:
            start_ts = pd.Timestamp(self.cfg.start_date, tz="UTC").floor("D")
            ranks = ranks[ranks["date"] >= start_ts].copy()
            if ranks.empty:
                raise ValueError(
                    f"No predictions remain on or after start_date={self.cfg.start_date}."
                )

        ranks = (
            ranks.sort_values(["date", "symbol", "prediction"])
            .drop_duplicates(["date", "symbol"], keep="last")
            .reset_index(drop=True)
        )

        raw_unique = set(
            ranks_raw[symbol_col].astype(str).str.strip().dropna().unique()
        )
        normalized_unique = set(
            filter(None, (self._normalize_symbol(x) for x in raw_unique))
        )

        metadata: dict = {
            "ranks_path": str(self.cfg.ranks_path),
            "rank_source_requested": self.cfg.rank_source,
            "rank_source_inferred": inferred_source,
            "symbol_col_used": symbol_col,
            "date_col_used": date_col,
            "prediction_col_used": pred_col,
            "rows_before_cleaning": int(rows_before_cleaning),
            "rows_with_invalid_symbol": invalid_symbol_rows,
            "rows_final": int(len(ranks)),
            "signal_date_min": (
                str(ranks["date"].min().date()) if not ranks.empty else None
            ),
            "signal_date_max": (
                str(ranks["date"].max().date()) if not ranks.empty else None
            ),
            "unique_raw_symbols": int(len(raw_unique)),
            "unique_normalized_hl_symbols": int(len(normalized_unique)),
            "unmatched_raw_symbols_sample": sorted(
                x for x in raw_unique if self._normalize_symbol(x) is None
            )[:100],
            "hl_symbols_missing_from_predictions_sample": sorted(
                TICKER_SET - normalized_unique
            )[:100],
        }
        return PredictionLoadResult(ranks=ranks, metadata=metadata)

    # ———————————————————————————————————————————————————————————————— #
    # _effective_market_end_exclusive
    # ———————————————————————————————————————————————————————————————— #

    def _effective_market_end_exclusive(self, metadata: dict) -> Optional[pd.Timestamp]:
        """Compute the exclusive upper bound for market data timestamps.

        Priority: cfg.end_date > last signal date + holding_days > None (no clip).
        The returned timestamp is midnight UTC of (end_day + 1 day), so standard
        half-open interval filtering [start, end_exclusive) includes the full end day.

        Args:
            metadata: dict returned by load_predictions; reads signal_date_max.

        Returns:
            Exclusive end pd.Timestamp (UTC midnight), or None for no upper clip.

        Raises:
            ValueError: holding_days is set but not positive.
        """
        if self.cfg.end_date:
            end_day = pd.Timestamp(self.cfg.end_date, tz="UTC").floor("D")
        elif self.cfg.stop_at_last_signal_date:
            max_signal = metadata.get("signal_date_max")
            if not max_signal:
                return None
            end_day = pd.Timestamp(str(max_signal), tz="UTC").floor("D")
            if self.cfg.holding_days is not None:
                if self.cfg.holding_days <= 0:
                    raise ValueError(
                        "holding_days must be positive, or set to None to disable."
                    )
                end_day = (
                    end_day + pd.Timedelta(days=float(self.cfg.holding_days))
                ).floor("D")
        else:
            return None
        return end_day + pd.Timedelta(days=1)

    # ———————————————————————————————————————————————————————————————— #
    # load_market_data
    # ———————————————————————————————————————————————————————————————— #

    def load_market_data(self, metadata: dict) -> MarketData:
        """Load and clean OHLC, funding, and oracle price CSVs.

        For each file: strips perp whitespace, parses time to UTC hourly
        Timestamps, coerces numeric columns, filters to TICKER_SET, drops rows
        with NaN key values, deduplicates (time, perp), and clips to the
        [start_date, effective_end_exclusive) window.

        Args:
            metadata: dict returned by load_predictions; used to derive the
                      effective market end timestamp via signal_date_max.

        Returns:
            MarketData with ohlc, funding, oracle DataFrames, each indexed by
            (time, perp) with no duplicates and within the configured time window.

        Raises:
            FileNotFoundError: Any of the three market files does not exist.
            ValueError:        Required columns missing, no OHLC rows remain
                               after start_date or end filter.
        """
        for p in [self.cfg.ohlc_path, self.cfg.funding_path, self.cfg.oracle_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required market file not found: {p}")

        ohlc = pd.read_csv(self.cfg.ohlc_path)
        funding = pd.read_csv(self.cfg.funding_path)
        oracle = pd.read_csv(self.cfg.oracle_path)

        required_cols = {
            "OHLC": (ohlc, ["perp", "time", "open", "high", "low", "close"]),
            "funding": (funding, ["perp", "time", "fundingRate"]),
            "oracle": (oracle, ["perp", "time", "oraclePx"]),
        }
        for name, (df, cols) in required_cols.items():
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} file missing columns: {missing}")

        # Strip, parse, drop invalids. TICKER_SET filter applied below after numeric coercion.
        for df in [ohlc, funding, oracle]:
            df["perp"] = df["perp"].astype(str).str.strip()
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor(
                "h"
            )

            df.dropna(subset=["perp", "time"], inplace=True)

        for col in ["open", "high", "low", "close"]:
            if col in ohlc.columns:
                ohlc[col] = pd.to_numeric(ohlc[col], errors="coerce")
        funding["fundingRate"] = pd.to_numeric(funding["fundingRate"], errors="coerce")
        oracle["oraclePx"] = pd.to_numeric(oracle["oraclePx"], errors="coerce")

        ohlc = ohlc[ohlc["perp"].isin(TICKER_SET)].dropna(
            subset=["open", "high", "low", "close"]
        )
        funding = funding[funding["perp"].isin(TICKER_SET)].dropna(
            subset=["fundingRate"]
        )
        oracle = oracle[oracle["perp"].isin(TICKER_SET)].dropna(subset=["oraclePx"])

        ohlc = ohlc.sort_values(["time", "perp"]).drop_duplicates(
            ["time", "perp"], keep="last"
        )
        funding = funding.sort_values(["time", "perp"]).drop_duplicates(
            ["time", "perp"], keep="last"
        )
        oracle = oracle.sort_values(["time", "perp"]).drop_duplicates(
            ["time", "perp"], keep="last"
        )

        if self.cfg.start_date:
            start_ts = pd.Timestamp(self.cfg.start_date, tz="UTC")
            ohlc = ohlc[ohlc["time"] >= start_ts].copy()
            funding = funding[funding["time"] >= start_ts].copy()
            oracle = oracle[oracle["time"] >= start_ts].copy()
            if ohlc.empty:
                raise ValueError(
                    f"No OHLC rows remain on or after start_date={self.cfg.start_date}."
                )

        end_exclusive = self._effective_market_end_exclusive(metadata)
        if end_exclusive is not None:
            ohlc = ohlc[ohlc["time"] < end_exclusive].copy()
            funding = funding[funding["time"] < end_exclusive].copy()
            oracle = oracle[oracle["time"] < end_exclusive].copy()
            if ohlc.empty:
                raise ValueError(
                    f"No OHLC rows remain after end filter. end_exclusive={end_exclusive}."
                )

        return MarketData(ohlc=ohlc, funding=funding, oracle=oracle)

    # ———————————————————————————————————————————————————————————————— #
    # create_daily_signals
    # ———————————————————————————————————————————————————————————————— #

    def create_daily_signals(self, ranks: pd.DataFrame) -> pd.DataFrame:
        """Build the daily long/short signal table from filtered predictions.

        For each date, sorts predictions descending, takes the top n_long as
        longs (side=+1) and bottom n_short as shorts (side=-1). Drops duplicate
        symbols within a day (long leg takes priority). Skips dates with fewer
        rows than min_prediction_rows_per_day. Populates self.daily_signal_counts.

        Args:
            ranks: DataFrame returned by load_predictions (PredictionLoadResult.ranks).
                   Must contain columns: date (Timestamp), symbol, prediction.

        Returns:
            DataFrame[signal_date, symbol, prediction, side, daily_rank].
            Empty DataFrame if no valid signal dates exist.

        Raises:
            ValueError: n_long or n_short is negative.
        """
        if self.cfg.n_long < 0 or self.cfg.n_short < 0:
            raise ValueError("n_long and n_short must be >= 0.")

        cols = ["signal_date", "symbol", "prediction", "side", "daily_rank"]
        rows: List[pd.DataFrame] = []
        self.daily_signal_counts = {}

        for date_ts, g in ranks.groupby("date", sort=True):
            g = g.sort_values("prediction", ascending=False).reset_index(drop=True)
            if len(g) < self.cfg.min_prediction_rows_per_day:
                self.daily_signal_counts[date_ts] = 0
                continue

            parts = []
            if self.cfg.n_long > 0:
                longs = g.head(self.cfg.n_long).copy()
                longs["side"] = 1
                parts.append(longs)
            if self.cfg.n_short > 0:
                shorts = (
                    g.tail(self.cfg.n_short)
                    .copy()
                    .sort_values("prediction", ascending=True)
                )
                shorts["side"] = -1
                parts.append(shorts)

            if not parts:
                self.daily_signal_counts[date_ts] = 0
                continue

            day = pd.concat(parts, ignore_index=True)
            # Guard against same symbol ranking in both legs when universe < n_long + n_short.
            day = day.drop_duplicates(["symbol"], keep="first")
            day["signal_date"] = date_ts
            day["daily_rank"] = np.arange(1, len(day) + 1)
            self.daily_signal_counts[date_ts] = int(len(day))
            rows.append(day[cols])

        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.concat(rows, ignore_index=True)

    # ———————————————————————————————————————————————————————————————— #
    # build_trade_schedule
    # ———————————————————————————————————————————————————————————————— #

    def build_trade_schedule(
        self, signals: pd.DataFrame, ohlc: pd.DataFrame
    ) -> pd.DataFrame:
        """Resolve each daily signal to a concrete execution candle and price.

        For each signal row, searches the OHLC data for the first valid bar
        within the entry window [signal_date + entry_hour_utc,
        signal_date + entry_hour_utc + max_entry_delay_hours]. Signals with no
        matching candle or invalid price are recorded in self.skipped_signals.
        Populates self.skipped_signals.

        Args:
            signals: DataFrame returned by create_daily_signals.
                     Must contain: signal_date (Timestamp), symbol, side, prediction.
            ohlc:    OHLC DataFrame from MarketData; must contain the column
                     named by cfg.execution_price_col.

        Returns:
            DataFrame[time, signal_date, symbol, side, prediction, price, notional]
            sorted by (time, symbol, side).

        Raises:
            ValueError: execution_price_col not found in ohlc, or no trades
                        could be scheduled after filtering.
        """
        schedule_cols = [
            "time",
            "signal_date",
            "symbol",
            "side",
            "prediction",
            "price",
            "notional",
        ]
        self.skipped_signals = []

        if signals.empty:
            return pd.DataFrame(columns=schedule_cols)

        price_col = self.cfg.execution_price_col
        if price_col not in ohlc.columns:
            raise ValueError(f"execution_price_col={price_col!r} not in OHLC columns.")

        ohlc_by_symbol = {sym: g.sort_values("time") for sym, g in ohlc.groupby("perp")}
        scheduled: List[dict] = []

        for row in signals.itertuples(index=False):
            start = row.signal_date + pd.Timedelta(hours=self.cfg.entry_hour_utc)
            end = start + pd.Timedelta(hours=self.cfg.max_entry_delay_hours)

            g = ohlc_by_symbol.get(row.symbol)
            if g is None or g.empty:
                self.skipped_signals.append(
                    {
                        "signal_date": row.signal_date,
                        "symbol": row.symbol,
                        "side": row.side,
                        "prediction": row.prediction,
                        "reason": "symbol_missing_from_ohlc",
                    }
                )
                continue

            candidates = g[(g["time"] >= start) & (g["time"] <= end)]
            if candidates.empty:
                self.skipped_signals.append(
                    {
                        "signal_date": row.signal_date,
                        "symbol": row.symbol,
                        "side": row.side,
                        "prediction": row.prediction,
                        "reason": "no_price_inside_entry_window",
                    }
                )
                continue

            price = float(candidates.iloc[0][price_col])
            if not np.isfinite(price) or price <= 0:
                self.skipped_signals.append(
                    {
                        "signal_date": row.signal_date,
                        "symbol": row.symbol,
                        "side": row.side,
                        "prediction": row.prediction,
                        "reason": "invalid_execution_price",
                    }
                )
                continue

            scheduled.append(
                {
                    "time": candidates.iloc[0]["time"],
                    "signal_date": row.signal_date,
                    "symbol": row.symbol,
                    "side": int(row.side),
                    "prediction": float(row.prediction),
                    "price": price,
                    "notional": float(self.cfg.notional_per_trade),
                }
            )

        if not scheduled:
            raise ValueError(
                "No trades could be scheduled. "
                "Check date overlap between predictions and OHLC data."
            )

        schedule = (
            pd.DataFrame(scheduled)
            .sort_values(["time", "symbol", "side"])
            .reset_index(drop=True)
        )
        return schedule[schedule_cols]

    # ———————————————————————————————————————————————————————————————— #
    # load_all — convenience chain
    # ———————————————————————————————————————————————————————————————— #

    def load_all(
        self,
    ) -> Tuple[PredictionLoadResult, MarketData, pd.DataFrame, pd.DataFrame]:
        """Run the full data loading pipeline in sequence.

        Chains load_predictions -> load_market_data -> create_daily_signals
        -> build_trade_schedule, passing outputs forward as inputs.

        Returns:
            Tuple of:
                [0] PredictionLoadResult  (ranks DataFrame + metadata dict)
                [1] MarketData            (ohlc, funding, oracle DataFrames)
                [2] signals DataFrame     (daily long/short signal table)
                [3] schedule DataFrame    (trade schedule with execution prices)
        """
        pred_result = self.load_predictions()
        market = self.load_market_data(pred_result.metadata)
        signals = self.create_daily_signals(pred_result.ranks)
        schedule = self.build_trade_schedule(signals, market.ohlc)
        return pred_result, market, signals, schedule
