# tests/test_evolution_digest.py
"""Task P3: digest-maker + resumable DigestStore.

Plan: docs/superpowers/plans/2026-07-03-phase1-batch-eval.md, Task P3 +
Global constraints (canonical JSON, store key
`(genome_hash|baseline_id, fingerprint_hash, record_type)`).

Hand-built `SimResult` fixtures throughout, same style as
`test_evolution_scoring.py` -- these are pure functions/classes over a
result's arrays and events, no engine run needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evolution import scoring  # noqa: E402
from src.evolution.canonical import canonical_json, fingerprint_hash, genome_hash  # noqa: E402
from src.evolution.digest import (  # noqa: E402
    DigestStore,
    error_digest,
    make_baseline_digest,
    make_digest,
)
from src.genome.adapter import GeneSpec, Genome  # noqa: E402
from src.position import ExecutionEvent  # noqa: E402
from src.result import SimResult  # noqa: E402


# --- fixtures -----------------------------------------------------------------

def _make_result(
    *,
    total_equity,
    per_perp_position=None,
    per_perp_maintenance=None,
    execution_events=None,
    n_liquidated=0,
    n_opened=0,
    n_closed=0,
    long_pnl=0.0,
    short_pnl=0.0,
    liq_long_pnl=0.0,
    liq_short_pnl=0.0,
    metrics_total=None,
) -> SimResult:
    total_equity = np.asarray(total_equity, dtype=float)
    n_bars = total_equity.shape[0]
    timeline = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=n_bars, freq="h", tz="UTC"))
    return SimResult(
        strategy_name="test",
        timeline=timeline,
        total_equity=total_equity,
        per_perp_equity={},
        per_perp_position=per_perp_position or {},
        per_perp_position_qty={},
        liquidation_events=[],
        execution_events=execution_events or [],
        rejected_orders=[],
        funding_events=[],
        metrics_total=metrics_total or {"PnL": 0.0, "DD": 0.0, "Sharpe": 0.0, "Sortino": 0.0},
        metrics_per_perp={},
        n_opened=n_opened,
        n_closed=n_closed,
        n_liquidated=n_liquidated,
        long_pnl=long_pnl,
        short_pnl=short_pnl,
        liq_long_pnl=liq_long_pnl,
        liq_short_pnl=liq_short_pnl,
        per_perp_maintenance=per_perp_maintenance or {},
    )


def _genome(**overrides) -> Genome:
    defaults = dict(
        name="G",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional"),
        exit_rule=GeneSpec("bracket", {"sl_pct": 0.05, "tp_pct": 0.10}),
        n_long=2,
        n_short=2,
    )
    defaults.update(overrides)
    return Genome(**defaults)


def _fp(sd_hash: str = "deadbeef") -> dict:
    return {
        "engine_semantics_version": 2,
        "data_window_hash": sd_hash,
        "scorer_version": "s1",
        "gene_library_version": "t1",
    }


# --- make_digest ----------------------------------------------------------------

def test_make_digest_has_all_fields_and_roundtrips_and_score_matches():
    equity = [1000.0, 1010.0, 1020.0, 1015.0, 1030.0]
    result = _make_result(
        total_equity=equity,
        per_perp_position={"BTC": np.array([500.0, 500.0, 500.0, 500.0, 500.0])},
        n_opened=1,
        n_closed=1,
    )
    genome = _genome()
    fp = _fp()
    digest = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")

    # canonical-JSON round trip: no inf/nan reaches json.dumps, and parsing
    # back yields the identical structure.
    encoded = canonical_json(digest)
    decoded = json.loads(encoded)
    assert decoded == digest

    expected_fields = {
        "genome_hash", "baseline_id", "kind_tuple", "genome", "fingerprint",
        "fingerprint_hash", "record_type", "status", "scorer_version",
        "score_from_bar", "normalized_pnl", "ci_lo", "ci_hi", "score",
        "sortino", "max_dd", "margin_headroom", "n_liquidated", "trade_count",
        "pnl_concentration", "long_pnl", "short_pnl", "liq_long_pnl",
        "liq_short_pnl", "per_month_pnl_signs", "when_earned",
        "benchmark_normalized_pnl", "benchmark_score",
    }
    assert expected_fields <= set(digest.keys())

    assert digest["genome_hash"] == genome_hash(genome)
    assert digest["kind_tuple"] == ["all_tradable", "rank", "release_bar", "fixed_notional", "bracket"]
    assert digest["record_type"] == "insample"
    assert digest["status"] == "ok"
    assert digest["scorer_version"] == scoring.SCORER_VERSION
    assert digest["fingerprint_hash"] == fingerprint_hash(fp)
    assert digest["benchmark_normalized_pnl"] is None
    assert digest["benchmark_score"] is None

    # score matches scoring.score_v1 fed the same lower_ci/n_liquidated/complexity/headroom
    gp = scoring.peak_gross(result)
    ci_lo, _ = scoring.bootstrap_pnl_ci(result.total_equity, fp["data_window_hash"], 1000.0, gp)
    headroom = scoring.margin_headroom(result)
    comp = scoring.complexity(genome)
    expected_score = scoring.score_v1(ci_lo, digest["n_liquidated"], comp, headroom)
    assert digest["score"] == pytest.approx(expected_score)
    assert digest["ci_lo"] == pytest.approx(ci_lo)


def test_make_digest_rejects_bad_record_type():
    result = _make_result(total_equity=[1000.0, 1000.0])
    with pytest.raises(ValueError):
        make_digest(result, _genome(), initial_equity=1000.0, fingerprint=_fp(), record_type="baseline")


# --- make_baseline_digest --------------------------------------------------------

def test_make_baseline_digest_has_null_genome_fields_and_baseline_key():
    result = _make_result(total_equity=[1000.0, 1005.0, 1010.0])
    fp = _fp()
    digest = make_baseline_digest(result, "baseline:EWP", initial_equity=1000.0, fingerprint=fp)

    assert digest["genome_hash"] is None
    assert digest["kind_tuple"] is None
    assert digest["genome"] is None
    assert digest["baseline_id"] == "baseline:EWP"
    assert digest["record_type"] == "baseline"
    assert digest["status"] == "ok"

    k = DigestStore.key(digest)
    assert k == ("baseline:EWP", fingerprint_hash(fp), "baseline")

    canonical_json(digest)  # must not raise


# --- DigestStore: roundtrip, status-aware skip -----------------------------------

def test_store_status_aware_skip(tmp_path):
    store = DigestStore(tmp_path / "digests.jsonl")
    genome = _genome()
    fp = _fp()
    result = _make_result(total_equity=[1000.0, 1000.0])

    err = error_digest(genome, fp, "insample", reason="boom")
    k = DigestStore.key(err)

    assert store.append(err) is True
    assert store.has_ok(k) is False  # an error record does not satisfy has_ok

    # same key is re-appendable after an error (status-aware skip: error does not block)
    ok = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    assert DigestStore.key(ok) == k
    assert store.append(ok) is True
    assert store.has_ok(k) is True

    # once an ok record exists, further appends for the same key are skipped
    assert store.append(ok) is False
    assert store.append(err) is False

    loaded = store.load()
    assert loaded[k]["status"] == "ok"


def test_load_ok_supersedes_error_regardless_of_file_order(tmp_path):
    path = tmp_path / "digests.jsonl"
    genome = _genome()
    fp = _fp()
    result = _make_result(total_equity=[1000.0, 1000.0])
    err = error_digest(genome, fp, "insample", reason="boom")
    ok = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    k = DigestStore.key(ok)
    assert DigestStore.key(err) == k

    # error line, then ok line
    path.write_text(canonical_json(err) + "\n" + canonical_json(ok) + "\n", encoding="utf-8")
    assert DigestStore(path).load()[k]["status"] == "ok"

    # ok line, then error line -- still resolves to ok
    path.write_text(canonical_json(ok) + "\n" + canonical_json(err) + "\n", encoding="utf-8")
    assert DigestStore(path).load()[k]["status"] == "ok"


# --- repair-before-append / trailing partial line --------------------------------

def test_repair_before_append_drops_partial_crash_tail(tmp_path):
    path = tmp_path / "digests.jsonl"
    fp = _fp()
    result = _make_result(total_equity=[1000.0, 1000.0])

    genome_a = _genome(name="A")
    genome_b = _genome(name="B", exit_rule=GeneSpec("bracket", {"sl_pct": 0.03, "tp_pct": None}))
    genome_c = _genome(name="C", n_long=3)

    complete = make_digest(result, genome_a, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    crashed = make_digest(result, genome_b, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    truncated_line = canonical_json(crashed)[:-10]  # chop the tail -- invalid JSON, no newline

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(complete) + "\n")
        f.write(truncated_line)  # simulates a crash mid-write: no trailing newline

    store = DigestStore(path)
    new_digest = make_digest(result, genome_c, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    assert store.append(new_digest) is True

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 2  # partial crash artifact dropped; complete + new survive
    for line in lines:
        json.loads(line)  # every remaining line must parse cleanly

    loaded = store.load()
    assert DigestStore.key(complete) in loaded
    assert DigestStore.key(new_digest) in loaded
    assert DigestStore.key(crashed) not in loaded  # the partial write was never recovered


def test_load_tolerates_trailing_partial_line_without_raising(tmp_path):
    path = tmp_path / "digests.jsonl"
    genome = _genome()
    fp = _fp()
    result = _make_result(total_equity=[1000.0, 1000.0])
    complete = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(complete) + "\n")
        f.write('{"genome_hash":"trunc')  # partial JSON, no trailing newline

    loaded = DigestStore(path).load()  # must not raise

    assert DigestStore.key(complete) in loaded
    assert len(loaded) == 1


# --- holdout score_from_bar: sliced series, distinct digest ----------------------

def test_holdout_score_from_bar_scores_sliced_series_differently():
    # Bar 0 alone carries heavy maintenance (headroom crashes below the s1
    # floor); every later bar is comfortably margined. An in-sample digest
    # (score_from_bar=0) sees that bar-0 dip and eats the headroom penalty; a
    # holdout digest scored from bar 3 onward never sees it.
    equity = [1000.0, 1200.0, 1400.0, 1300.0, 1100.0, 900.0, 700.0]
    n = len(equity)
    per_perp_position = {"BTC": np.full(n, 500.0)}
    per_perp_maintenance = {"BTC": np.array([950.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])}
    result = _make_result(
        total_equity=equity,
        per_perp_position=per_perp_position,
        per_perp_maintenance=per_perp_maintenance,
    )
    genome = _genome()
    fp = _fp()

    full_digest = make_digest(
        result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample", score_from_bar=0,
    )
    holdout_digest = make_digest(
        result, genome, initial_equity=1000.0, fingerprint=fp, record_type="holdout", score_from_bar=3,
    )

    assert full_digest["margin_headroom"] == pytest.approx(0.05)
    assert holdout_digest["margin_headroom"] == pytest.approx((700.0 - 100.0) / 700.0)
    assert full_digest["score"] != holdout_digest["score"]
    assert holdout_digest["score_from_bar"] == 3
    assert full_digest["score_from_bar"] == 0
    assert holdout_digest["record_type"] == "holdout"

    # per_month_pnl_signs/when_earned are derived from the sliced series too --
    # fewer bars are visible to the holdout digest.
    assert len(holdout_digest["per_month_pnl_signs"]) <= len(full_digest["per_month_pnl_signs"])


# --- inf/nan guard ----------------------------------------------------------------

def test_inf_nan_metrics_guarded_to_none_and_canonical_json_does_not_raise():
    equity = [1000.0, 1010.0, float("nan")]
    result = _make_result(
        total_equity=equity,
        per_perp_position={"BTC": np.array([500.0, 500.0, 500.0])},
        metrics_total={"PnL": 0.0, "DD": 0.0, "Sharpe": 0.0, "Sortino": float("nan")},
    )
    genome = _genome()
    fp = _fp()
    digest = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")

    assert digest["normalized_pnl"] is None
    assert digest["sortino"] is None

    encoded = canonical_json(digest)  # allow_nan=False -- must not raise
    decoded = json.loads(encoded)
    assert decoded["normalized_pnl"] is None
    assert decoded["sortino"] is None


# --- n_liquidated / trade_count windowing by score_from_bar -----------------------

def test_windowed_execution_event_counts_and_pnl_concentration():
    equity = [1000.0] * 6
    timeline_hours = 6
    events = [
        ExecutionEvent(
            timestamp=pd.Timestamp("2025-01-01T00:00:00Z").isoformat(),
            event_type="position_opened", asset="BTC", side=1,
            notional=100.0, fill_price=10.0, fee=0.0, realized_pnl=0.0,
        ),
        # this liquidation happens BEFORE bar index 3's timestamp -- excluded
        # from a score_from_bar=3 window.
        ExecutionEvent(
            timestamp=pd.Timestamp("2025-01-01T01:00:00Z").isoformat(),
            event_type="position_liquidated", asset="BTC", side=-1,
            notional=50.0, fill_price=9.0, fee=1.0, realized_pnl=-20.0,
        ),
        # this close happens at/after bar index 3's timestamp -- included.
        ExecutionEvent(
            timestamp=pd.Timestamp("2025-01-01T03:00:00Z").isoformat(),
            event_type="position_closed", asset="ETH", side=-1,
            notional=200.0, fill_price=5.0, fee=1.0, realized_pnl=30.0,
        ),
    ]
    result = _make_result(total_equity=equity, execution_events=events)
    genome = _genome()
    fp = _fp()

    full = make_digest(result, genome, initial_equity=1000.0, fingerprint=fp, record_type="insample")
    holdout = make_digest(
        result, genome, initial_equity=1000.0, fingerprint=fp, record_type="holdout", score_from_bar=3,
    )
    assert timeline_hours == 6  # sanity: fixture has 6 hourly bars, index 3 is a real cutoff

    assert full["n_liquidated"] == 1
    assert full["trade_count"] == 2  # 1 opened + 1 closed (liquidation excluded, per engine convention)

    assert holdout["n_liquidated"] == 0  # the liquidation is before the holdout cutoff
    assert holdout["trade_count"] == 1  # only the 03:00 position_closed is at/after the cutoff


# --- error_digest -----------------------------------------------------------------

def test_error_digest_is_minimal_and_keyable_for_genome_and_baseline():
    fp = _fp()
    genome = _genome()

    err_genome = error_digest(genome, fp, "insample", reason="RuntimeError: boom")
    assert err_genome["status"] == "error"
    assert err_genome["reason"] == "RuntimeError: boom"
    assert err_genome["genome_hash"] == genome_hash(genome)
    assert err_genome["baseline_id"] is None
    assert DigestStore.key(err_genome) == (genome_hash(genome), fingerprint_hash(fp), "insample")
    canonical_json(err_genome)

    err_baseline = error_digest("baseline:EWP", fp, "baseline", reason="data missing")
    assert err_baseline["genome_hash"] is None
    assert err_baseline["baseline_id"] == "baseline:EWP"
    assert DigestStore.key(err_baseline) == ("baseline:EWP", fingerprint_hash(fp), "baseline")
    canonical_json(err_baseline)
