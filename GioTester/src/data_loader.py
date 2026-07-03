from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import DataConfig

# ————————————————————————————————————————————————————————————————————————— #
# Hyperliquid universe
# ————————————————————————————————————————————————————————————————————————— #

TICKERS = [
    "0G",
    "AAVE",
    "ACE",
    "ADA",
    "ALGO",
    "ALT",
    "ANIME",
    "APE",
    "APT",
    "AR",
    "ARB",
    "ARK",
    "ASTER",
    "ATOM",
    "AVAX",
    "AVNT",
    "AIXBT",
    "BABY",
    "BANANA",
    "BERA",
    "BIGTIME",
    "BLAST",
    "BLUR",
    "BOME",
    "BTC",
    "BCH",
    "BIO",
    "BRETT",
    "BSV",
    "CAKE",
    "CELO",
    "CFX",
    "CHILLGUY",
    "COMP",
    "CRV",
    "DOGE",
    "DOOD",
    "DOT",
    "DYDX",
    "DYM",
    "EIGEN",
    "ENA",
    "ENS",
    "ETC",
    "ETH",
    "ETHFI",
    "FARTCOIN",
    "FET",
    "FIL",
    "FTT",
    "FXS",
    "GALA",
    "GAS",
    "GMT",
    "GOAT",
    "GRASS",
    "GRIFFAIN",
    "GMX",
    "HBAR",
    "HMSTR",
    "HYPE",
    "HYPER",
    "ICP",
    "IOTA",
    "INIT",
    "INJ",
    "IO",
    "IP",
    "IMX",
    "JTO",
    "JUP",
    "KAITO",
    "KAS",
    "kBONK",
    "kDOGS",
    "kFLOKI",
    "kLUNC",
    "kNEIRO",
    "kPEPE",
    "kSHIB",
    "LAUNCHCOIN",
    "LAYER",
    "LDO",
    "LINEA",
    "LINK",
    "LTC",
    "MANTA",
    "MAV",
    "MAVIA",
    "ME",
    "MELANIA",
    "MEME",
    "MERL",
    "MEW",
    "MINA",
    "MKR",
    "MNT",
    "MOODENG",
    "MORPHO",
    "MOVE",
    "NEAR",
    "NEO",
    "NEIROETH",
    "NIL",
    "NOT",
    "NXPC",
    "ONDO",
    "OMNI",
    "OM",
    "OP",
    "OGN",
    "ORDI",
    "PAXG",
    "PEOPLE",
    "PENDLE",
    "PENGU",
    "PNUT",
    "POL",
    "POLYX",
    "POPCAT",
    "PROMPT",
    "PROVE",
    "PUMP",
    "PURR",
    "PYTH",
    "REQ",
    "REZ",
    "RENDER",
    "RESOLV",
    "RSR",
    "RUNE",
    "S",
    "SAGA",
    "SAND",
    "SCR",
    "SEI",
    "SKY",
    "SOPH",
    "SOL",
    "SNX",
    "STBL",
    "STRK",
    "STG",
    "STX",
    "SUSHI",
    "SUI",
    "SUPER",
    "SPX",
    "SYRUP",
    "TAO",
    "TIA",
    "TNSR",
    "TON",
    "TRUMP",
    "TRB",
    "TRX",
    "TST",
    "TURBO",
    "USUAL",
    "USTC",
    "UMA",
    "UNI",
    "VIRTUAL",
    "VINE",
    "VVV",
    "W",
    "WLFI",
    "WCT",
    "WLD",
    "WIF",
    "XAI",
    "XLM",
    "XPL",
    "XRP",
    "YGG",
    "YZY",
    "ZEC",
    "ZEN",
    "ZK",
    "ZETA",
    "ZEREBRO",
    "ZRO",
    "ZORA",
]
TICKER_SET = set(TICKERS)
CANONICAL_TICKER_BY_UPPER = {sym.upper(): sym for sym in TICKERS}

# Tier-1/small-notional Hyperliquid risk settings used by the simulator.
# Maintenance margin rate = 1 / (2 * max_leverage_tier1).
# Default 10x max leverage covers mid/small-cap alts not listed below.
DEFAULT_MAX_LEVERAGE = 10.0
MAX_LEVERAGE_BY_PERP: Dict[str, float] = {
    "BTC": 40.0,
    "ETH": 25.0,
    "SOL": 20.0,
    "XRP": 20.0,
}
MM_RATE_BY_PERP: Dict[str, float] = {
    perp: 1.0 / (2.0 * max_leverage)
    for perp, max_leverage in MAX_LEVERAGE_BY_PERP.items()
}


def max_leverage_for(perp: str) -> float:
    """Return Tier-1 max leverage for perp. Defaults to 10x."""
    return MAX_LEVERAGE_BY_PERP.get(perp, DEFAULT_MAX_LEVERAGE)


def mm_rate_for(perp: str) -> float:
    """Return maintenance margin rate for perp. Defaults to 0.05 (10x tier)."""
    return MM_RATE_BY_PERP.get(perp, 1.0 / (2.0 * DEFAULT_MAX_LEVERAGE))


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
                [x for x in raw_unique if self._normalize_symbol(x) is None]
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
        # fundingRate at timestamp T = rate settled AT hour T (HL fundingHistory API
        # convention). Simulator applies it to positions open during bar T. One-bar shift
        # would occur only if T meant "upcoming hour" — HL API confirms it means "paid now".
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
