from __future__ import annotations

from typing import List, Set

import numpy as np
import pandas as pd

from data_loader import PredictionLoadResult


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman r via manual ranking — avoids scipy dependency."""
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def compare_sources(
    cc: PredictionLoadResult,
    nm: PredictionLoadResult,
    n_long: int = 3,
    n_short: int = 3,
) -> None:
    """Print full comparison report: coverage, rank agreement, signal independence.

    Args:
        cc:      PredictionLoadResult for the crowdcent source.
        nm:      PredictionLoadResult for the numerai source.
        n_long:  Top-N longs used in signal independence check.
        n_short: Bottom-N shorts used in signal independence check.
    """
    cr = cc.ranks
    nr = nm.ranks

    # ------------------------------------------------------------------ #
    # A — Coverage
    # ------------------------------------------------------------------ #

    cc_syms  = set(cr["symbol"].unique())
    nm_syms  = set(nr["symbol"].unique())
    only_cc_syms = sorted(cc_syms - nm_syms)
    only_nm_syms = sorted(nm_syms - cc_syms)

    cc_dates = set(cr["date"].unique())
    nm_dates = set(nr["date"].unique())
    both_dates   = sorted(cc_dates & nm_dates)
    only_cc_dates = sorted(cc_dates - nm_dates)
    only_nm_dates = sorted(nm_dates - cc_dates)

    cc_pairs = set(zip(cr["date"], cr["symbol"]))
    nm_pairs = set(zip(nr["date"], nr["symbol"]))

    print("\n" + "=" * 62)
    print("  SOURCE COMPARISON: crowdcent (cc) vs numerai (nm)")
    print("=" * 62)

    print("\n[A] COVERAGE")

    print(f"  symbols    cc:{len(cc_syms)}  nm:{len(nm_syms)}  "
          f"shared:{len(cc_syms & nm_syms)}  "
          f"only_cc:{len(only_cc_syms)}  only_nm:{len(only_nm_syms)}")

    print(f"  dates      cc:[{cr['date'].min().date()} → {cr['date'].max().date()}]"
          f" ({len(cc_dates)} days)  "
          f"nm:[{nr['date'].min().date()} → {nr['date'].max().date()}]"
          f" ({len(nm_dates)} days)  "
          f"shared:{len(both_dates)}  only_cc:{len(only_cc_dates)}  only_nm:{len(only_nm_dates)}")

    print(f"  pairs      cc:{len(cc_pairs)}  nm:{len(nm_pairs)}  "
          f"shared:{len(cc_pairs & nm_pairs)}  "
          f"only_cc:{len(cc_pairs - nm_pairs)}  only_nm:{len(nm_pairs - cc_pairs)}")

    def _show(label: str, items: List[str], limit: int = 20) -> None:
        if not items:
            return
        shown = items[:limit]
        tail  = f"  +{len(items) - limit} more" if len(items) > limit else ""
        print(f"  {label}: {', '.join(shown)}{tail}")

    _show("only_cc symbols", only_cc_syms)
    _show("only_nm symbols", only_nm_syms)
    _show("only_cc dates",   [str(d.date()) for d in only_cc_dates])
    _show("only_nm dates",   [str(d.date()) for d in only_nm_dates])

    # ------------------------------------------------------------------ #
    # B — Rank agreement on overlapping (date, symbol) pairs
    # ------------------------------------------------------------------ #

    print("\n[B] RANK AGREEMENT")

    merged = cr[["date", "symbol", "prediction"]].merge(
        nr[["date", "symbol", "prediction"]],
        on=["date", "symbol"],
        suffixes=("_cc", "_nm"),
    )

    if merged.empty:
        print("  No overlapping (date, symbol) pairs — skipping.")
    else:
        print(f"  Overlapping pairs: {len(merged)}")

        overall_r = _spearman(merged["prediction_cc"], merged["prediction_nm"])
        print(f"  Overall Spearman r = {overall_r:.4f}")

        per_date: pd.Series = (
            merged.groupby("date")[["prediction_cc", "prediction_nm"]]
            .apply(
                lambda g: _spearman(g["prediction_cc"], g["prediction_nm"])
                if len(g) >= 2 else np.nan
            )
            .dropna()
        )

        if not per_date.empty:
            print(f"  Per-date Spearman  mean:{per_date.mean():.4f}  "
                  f"std:{per_date.std():.4f}  "
                  f"min:{per_date.min():.4f}  "
                  f"max:{per_date.max():.4f}  "
                  f"(n={len(per_date)} dates)")
        else:
            print("  Per-date Spearman: no dates with >= 2 overlapping pairs.")

    # ------------------------------------------------------------------ #
    # C — Signal independence
    # ------------------------------------------------------------------ #

    print(f"\n[C] SIGNAL INDEPENDENCE  (n_long={n_long}, n_short={n_short})")

    if not both_dates:
        print("  No shared dates — skipping.")
        return

    cc_by_date = {d: g for d, g in cr.groupby("date")}
    nm_by_date = {d: g for d, g in nr.groupby("date")}

    jac_long:  List[float] = []
    jac_short: List[float] = []

    for date in both_dates:
        cc_day = cc_by_date[date].sort_values("prediction", ascending=False)
        nm_day = nm_by_date[date].sort_values("prediction", ascending=False)

        if n_long > 0:
            jac_long.append(_jaccard(
                set(cc_day.head(n_long)["symbol"]),
                set(nm_day.head(n_long)["symbol"]),
            ))
        if n_short > 0:
            jac_short.append(_jaccard(
                set(cc_day.tail(n_short)["symbol"]),
                set(nm_day.tail(n_short)["symbol"]),
            ))

    n_shared = len(both_dates)

    def _jaccard_report(label: str, vals: List[float]) -> None:
        if not vals:
            return
        arr = np.array(vals)
        full = int((arr == 1.0).sum())
        zero = int((arr == 0.0).sum())
        print(f"  {label}  mean:{arr.mean():.4f}  std:{arr.std():.4f}  "
              f"full_agree:{full}/{n_shared}  zero_agree:{zero}/{n_shared}")

    _jaccard_report("Long  Jaccard", jac_long)
    _jaccard_report("Short Jaccard", jac_short)
