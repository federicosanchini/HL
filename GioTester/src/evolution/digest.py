# src/evolution/digest.py
"""Digest-maker + resumable store (Phase 1 Task P3).

Plan: `docs/superpowers/plans/2026-07-03-phase1-batch-eval.md`, Task P3 +
Global constraints (canonical JSON, store key =
`(genome_hash|baseline_id, fingerprint_hash, record_type)`,
`record_type in {insample, holdout, baseline}`).

A "digest" is one flat, canonical-JSON-serializable dict summarizing a single
genome/baseline evaluation against a single data window (fingerprint). Digests
are the unit of resumable work: `DigestStore` is an append-only JSONL file,
keyed so a crashed/interrupted batch run can restart and skip everything
already scored, without re-running the (expensive, non-deterministic-wall-
time) simulation.

Holdout warm-up (`score_from_bar`): a holdout `SimData` window (Task P4)
includes a 720-bar embargo prefix so stateful genes (momentum, cached ranks,
delay timing, ...) enter the true holdout period already warm. Scoring must
NOT credit/blame the genome for that unscored prefix. `score_from_bar` marks
where the scored window begins; every field derived from the per-bar arrays
(`total_equity`, `per_perp_position`, `per_perp_maintenance`, `timeline`) or
from event timestamps is computed on the slice `[score_from_bar:]` /
timestamps `>= timeline[score_from_bar]`. `score_from_bar=0` (in-sample) is a
no-op slice -- the full series is scored, as before.

Two field groups are deliberately NOT windowed by `score_from_bar` (documented
here, not silently dropped):
  - `sortino` / `max_dd`: read straight off `result.metrics_total`, which the
    engine computes once over the full run. Re-deriving a windowed Sortino
    would require reimplementing `metrics.series_metrics`'s formulas here --
    out of scope for a digest-serialization module; P4/report-tooling can
    thread a windowed `metrics_total` through if this bites in practice.
  - `long_pnl` / `short_pnl` / `liq_long_pnl` / `liq_short_pnl`: these are
    `SimResult`'s own cumulative attribution scalars (engine-maintained
    single source of truth for side-of-book bucketing). Recomputing a
    windowed version here would mean duplicating the engine's realized-pnl
    bucketing rules (which side an event belongs to, liquidation vs. not) in
    a second place -- a correctness hazard project conventions (CLAUDE.md)
    explicitly warn against. Passed through unsliced.
  Everything else that touches `execution_events` (`n_liquidated`,
  `trade_count`, `pnl_concentration`) only needs an event's already-stamped
  `event_type`/`timestamp`/`realized_pnl` fields -- no bucketing logic to
  duplicate -- so those ARE windowed by timestamp.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.evolution import scoring
from src.evolution.canonical import (
    canonical_genome_dict,
    canonical_json,
    fingerprint_hash,
    genome_hash,
)
from src.genome.adapter import Genome

if TYPE_CHECKING:
    from src.result import SimResult

DigestKey = Tuple[Optional[str], str, str]


# --- small numeric/structural helpers ----------------------------------------

def _safe_float(value: object) -> Optional[float]:
    """Cast to float; non-finite (`nan`/`inf`) becomes `None`.

    `canonical_json` uses `allow_nan=False` -- an un-guarded `nan`/`inf`
    reaching it raises. Every numeric digest field is routed through this so
    a degenerate metric is stored as an explicit `null`, never a crash.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _kind_tuple(genome: Genome) -> List[str]:
    return [
        genome.universe_filter.kind,
        genome.signal.kind,
        genome.entry_timing.kind,
        genome.sizing.kind,
        genome.exit_rule.kind,
    ]


def _sliced_result(result: "SimResult", score_from_bar: int) -> "SimResult":
    """Shallow-sliced view of `result`'s bar-indexed series from `score_from_bar`.

    Only the fields `scoring.py`'s pure functions and this module's
    per-bar helpers read (`total_equity`, `per_perp_position`,
    `per_perp_maintenance`, `timeline`) are sliced; everything else
    (`execution_events`, cumulative attribution scalars, `metrics_total`, ...)
    is shared by reference -- nothing here mutates it.
    """
    if score_from_bar <= 0:
        return result
    return dataclasses.replace(
        result,
        timeline=result.timeline[score_from_bar:],
        total_equity=np.asarray(result.total_equity, dtype=float)[score_from_bar:],
        per_perp_position={
            perp: np.asarray(arr, dtype=float)[score_from_bar:]
            for perp, arr in result.per_perp_position.items()
        },
        per_perp_maintenance={
            perp: np.asarray(arr, dtype=float)[score_from_bar:]
            for perp, arr in result.per_perp_maintenance.items()
        },
    )


def _cutoff_timestamp(timeline: pd.DatetimeIndex, score_from_bar: int):
    """First timestamp INCLUDED in the scored window, or `None` if empty."""
    if len(timeline) == 0:
        return None
    if score_from_bar <= 0:
        return timeline[0]
    if score_from_bar >= len(timeline):
        return None
    return timeline[score_from_bar]


def _windowed_execution_events(events: Sequence, cutoff) -> list:
    if cutoff is None:
        return []
    return [e for e in events if pd.Timestamp(e.timestamp) >= cutoff]


def _pnl_concentration(events: Sequence) -> float:
    """Top-5 |realized_pnl| share of total |realized_pnl| over `events`.

    Includes liquidation-close events (no type filtering) per plan wording.
    Div0-guarded: no realized pnl at all -> `0.0` (documented sentinel,
    matching `scoring.py`'s convention), never NaN.
    """
    magnitudes = sorted((abs(e.realized_pnl) for e in events), reverse=True)
    total = sum(magnitudes)
    if total <= 0:
        return 0.0
    return sum(magnitudes[:5]) / total


def _per_month_pnl_signs(timeline: pd.DatetimeIndex, total_equity: np.ndarray) -> List[int]:
    """+1/-1/0 net-equity-change sign per calendar month, in month order."""
    equity = np.asarray(total_equity, dtype=float)
    if equity.size < 2 or len(timeline) < 2:
        return []
    diffs = np.diff(equity)
    monthly: Dict[Tuple[int, int], float] = {}
    for i, d in enumerate(diffs):
        ts = timeline[i + 1]
        key = (ts.year, ts.month)
        monthly[key] = monthly.get(key, 0.0) + float(d)
    signs: List[int] = []
    for key in sorted(monthly):
        v = monthly[key]
        signs.append(1 if v > 0 else (-1 if v < 0 else 0))
    return signs


def _when_earned(timeline: pd.DatetimeIndex, total_equity: np.ndarray) -> Dict[str, Optional[str]]:
    """Timestamps where cumulative positive-equity-gain first crosses 25/50/75%.

    Only positive per-bar diffs count toward the cumulative-gain curve (a
    drawdown never un-crosses a quartile once reached). No positive gain at
    all (flat or net-losing series) -> all `None`, never a spurious index.
    """
    empty = {"p25": None, "p50": None, "p75": None}
    equity = np.asarray(total_equity, dtype=float)
    if equity.size < 2 or len(timeline) < 2:
        return dict(empty)
    diffs = np.diff(equity)
    gains = np.clip(diffs, 0.0, None)
    total_gain = float(gains.sum())
    if total_gain <= 0:
        return dict(empty)
    cum = np.cumsum(gains)
    out: Dict[str, Optional[str]] = {}
    for label, frac in (("p25", 0.25), ("p50", 0.5), ("p75", 0.75)):
        idx = int(np.searchsorted(cum, frac * total_gain, side="left"))
        idx = min(idx, diffs.shape[0] - 1)
        out[label] = timeline[idx + 1].isoformat()
    return out


def _shared_metrics(
    result: "SimResult",
    initial_equity: float,
    fingerprint: dict,
    score_from_bar: int,
    complexity_value: int,
) -> dict:
    """Metric fields common to genome and baseline digests.

    See module docstring for exactly which fields are windowed by
    `score_from_bar` and which are intentionally passed through unsliced.
    """
    scored = _sliced_result(result, score_from_bar)

    gp = scoring.peak_gross(scored)
    norm_pnl = scoring.normalized_pnl(scored, initial_equity)
    ci_lo, ci_hi = scoring.bootstrap_pnl_ci(
        scored.total_equity, fingerprint["data_window_hash"], initial_equity, gp,
    )
    headroom = scoring.margin_headroom(scored)

    cutoff = _cutoff_timestamp(result.timeline, score_from_bar)
    windowed_events = _windowed_execution_events(result.execution_events, cutoff)
    n_liquidated = sum(1 for e in windowed_events if e.event_type == "position_liquidated")
    n_opened = sum(
        1 for e in windowed_events if e.event_type in ("position_opened", "position_flipped")
    )
    n_closed = sum(
        1 for e in windowed_events if e.event_type in ("position_closed", "position_flipped")
    )

    score = scoring.score_v1(ci_lo, n_liquidated, complexity_value, headroom)

    return {
        "normalized_pnl": _safe_float(norm_pnl),
        "ci_lo": _safe_float(ci_lo),
        "ci_hi": _safe_float(ci_hi),
        "score": _safe_float(score),
        "sortino": _safe_float(result.metrics_total["Sortino"]),
        "max_dd": _safe_float(result.metrics_total["DD"]),
        "margin_headroom": _safe_float(headroom),
        "n_liquidated": n_liquidated,
        "trade_count": n_opened + n_closed,
        "pnl_concentration": _safe_float(_pnl_concentration(windowed_events)),
        "long_pnl": _safe_float(result.long_pnl),
        "short_pnl": _safe_float(result.short_pnl),
        "liq_long_pnl": _safe_float(result.liq_long_pnl),
        "liq_short_pnl": _safe_float(result.liq_short_pnl),
        "per_month_pnl_signs": _per_month_pnl_signs(scored.timeline, scored.total_equity),
        "when_earned": _when_earned(scored.timeline, scored.total_equity),
        # Joined later by report tooling against the matching baseline digest
        # (same fingerprint_hash, record_type) -- always null out of this module.
        "benchmark_normalized_pnl": None,
        "benchmark_score": None,
    }


# --- public API ----------------------------------------------------------------

def make_digest(
    result: "SimResult",
    genome: Genome,
    initial_equity: float,
    fingerprint: dict,
    record_type: str,
    score_from_bar: int = 0,
) -> dict:
    """Build a canonical-JSON-serializable digest for one genome evaluation.

    `record_type` must be `"insample"` or `"holdout"` (baseline digests come
    from `make_baseline_digest`, never this function -- a genome is never a
    baseline). `score_from_bar` selects the scored window; see module
    docstring.
    """
    if record_type not in ("insample", "holdout"):
        raise ValueError(
            f"make_digest record_type must be 'insample' or 'holdout', got {record_type!r}"
        )
    if score_from_bar < 0:
        raise ValueError(f"score_from_bar must be >= 0, got {score_from_bar}")

    metrics = _shared_metrics(
        result,
        initial_equity,
        fingerprint,
        score_from_bar,
        complexity_value=scoring.complexity(genome),
    )
    return {
        "genome_hash": genome_hash(genome),
        "baseline_id": None,
        "kind_tuple": _kind_tuple(genome),
        "genome": canonical_genome_dict(genome),
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "record_type": record_type,
        "status": "ok",
        "scorer_version": scoring.SCORER_VERSION,
        "score_from_bar": score_from_bar,
        **metrics,
    }


def make_baseline_digest(
    result: "SimResult",
    baseline_id: str,
    initial_equity: float,
    fingerprint: dict,
) -> dict:
    """Build a digest for a non-genome baseline/placebo run (e.g. `"baseline:EWP"`).

    Same metric fields as `make_digest`, but genome identity fields are null
    and the store key resolves on `baseline_id` instead of `genome_hash`.
    Complexity is defined as 0 (baselines are outside the genome search
    space, not five-gene compositions).
    """
    metrics = _shared_metrics(
        result, initial_equity, fingerprint, score_from_bar=0, complexity_value=0,
    )
    return {
        "genome_hash": None,
        "baseline_id": baseline_id,
        "kind_tuple": None,
        "genome": None,
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "record_type": "baseline",
        "status": "ok",
        "scorer_version": scoring.SCORER_VERSION,
        "score_from_bar": 0,
        **metrics,
    }


def error_digest(
    genome_or_id: Union[Genome, str],
    fingerprint: dict,
    record_type: str,
    reason: str,
) -> dict:
    """Minimal, key-able error record: a failed eval must still occupy its key.

    This is what makes the store's status-aware skip work: an `"error"`
    record does NOT block re-evaluation (only `"ok"` does), so a failed
    genome/window pair is automatically retried on the next batch run without
    special-casing -- but the failure is still durably logged in the
    meantime.
    """
    if isinstance(genome_or_id, Genome):
        gh: Optional[str] = genome_hash(genome_or_id)
        bid: Optional[str] = None
    else:
        gh = None
        bid = str(genome_or_id)
    return {
        "genome_hash": gh,
        "baseline_id": bid,
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "record_type": record_type,
        "status": "error",
        "reason": str(reason),
        "scorer_version": scoring.SCORER_VERSION,
    }


class DigestStore:
    """Append-only JSONL digest store, resumable across crashed/killed batch runs.

    Key = `(genome_hash or baseline_id, fingerprint_hash, record_type)`.
    **Single-writer assumption**: intended for one process appending (if a
    future parallel-worker batch runner lands, worker results must funnel
    through one appender -- concurrent appenders would race the
    repair-before-append truncation and the read-then-append skip check).
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    @staticmethod
    def key(digest: dict) -> DigestKey:
        ident = digest.get("genome_hash") or digest.get("baseline_id")
        return (ident, digest["fingerprint_hash"], digest["record_type"])

    def _repair(self) -> None:
        """Drop a trailing crash artifact: a last line with no terminating `\\n`.

        Idempotent -- safe (and required) to call before every append. A
        clean file (empty, or already ending in `\\n`) is left untouched.
        """
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        truncated = data[: last_newline + 1] if last_newline != -1 else b""
        self.path.write_bytes(truncated)

    def append(self, digest: dict) -> bool:
        """Append `digest` unless an `"ok"` record already exists for its key.

        Returns `True` if written, `False` if skipped (status-aware skip: an
        `"error"` record for the same key does NOT block re-appending -- only
        `"ok"` does). Repairs a crash-truncated tail first.
        """
        self._repair()
        existing = self.load()
        k = self.key(digest)
        prior = existing.get(k)
        if prior is not None and prior.get("status") == "ok":
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json(digest))
            f.write("\n")
        return True

    def load(self) -> Dict[DigestKey, dict]:
        """Parse the store into a `{key: digest}` map.

        Tolerates a trailing partial line (no terminating newline -- an
        in-flight/crashed write) by dropping it, without raising. On a
        duplicate key, an `"ok"` record always supersedes an `"error"`
        record, regardless of which was written first.
        """
        if not self.path.exists():
            return {}
        data = self.path.read_bytes()
        if not data:
            return {}
        text = data.decode("utf-8")
        lines = text.split("\n")
        # Trailing element after the final split is either "" (file ended
        # with \n -- a clean boundary) or an incomplete/partial line (file did
        # NOT end with \n -- an in-flight write) -- either way, drop it.
        lines = lines[:-1]

        result: Dict[DigestKey, dict] = {}
        for line in lines:
            if not line:
                continue
            obj = json.loads(line)
            k = self.key(obj)
            prior = result.get(k)
            if prior is not None and prior.get("status") == "ok" and obj.get("status") != "ok":
                continue  # ok already resolved for this key; never let an error override it
            result[k] = obj
        return result

    def has_ok(self, key: DigestKey) -> bool:
        entry = self.load().get(key)
        return entry is not None and entry.get("status") == "ok"
