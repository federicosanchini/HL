"""Task 6 (Phase E engine v2): runner pipeline integration tests.

Spec: docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md, rules R1
(decide-close/fill-next-open, anti-look-ahead pinning, unpriceable + backtest-end
rejection sites) and R6 (bar pipeline order). Drives synthetic `SimData` through
`run_backtest_prepared` via `engine_harness`.

Uses a made-up asset ("SYN") so maintenance/leverage tables fall back to defaults
(mm_rate 0.05, max_leverage 10) and hand-computed numbers hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BacktestConfig  # noqa: E402
from tests.engine_harness import (  # noqa: E402
    build_sim_data,
    market_order,
    run_scripted,
    trigger_order,
)

NAN = float("nan")


def _iso(result, k: int) -> str:
    return result.timeline[k].isoformat()


def _exec_at(result, ts: str):
    return [e for e in result.execution_events if e.timestamp == ts]


# --- (i) order queued bar i fills at open[i+1] ------------------------------

def test_market_order_fills_at_next_open():
    sd = build_sim_data({
        "SYN": [[100, 100, 100, 100], [110, 110, 110, 110], [120, 120, 120, 120]],
    })
    cfg = BacktestConfig(initial_equity=2000.0)
    result = run_scripted(sd, {0: [market_order("SYN", 1, notional=100.0)]}, bt_cfg=cfg)

    opens = [e for e in result.execution_events if e.event_type == "position_opened"]
    assert len(opens) == 1
    ev = opens[0]
    assert ev.fill_price == 110.0          # open[1], not open[0]/close[1]
    assert ev.timestamp == _iso(result, 1)
    assert ev.fee == pytest.approx(100.0 * cfg.taker_fee_bps / 1e4)
    # No execution event on the decision bar (bar 0).
    assert _exec_at(result, _iso(result, 0)) == []


# --- (ii) ANTI-LOOK-AHEAD: fill/accept independent of close[i+1] -------------

def test_margin_decision_independent_of_next_close():
    # Sized so the open[1]-pinned margin check sits right at the boundary and
    # accepts. A look-ahead engine using close[1] would reject dataset B
    # (close 1000 inflates the hypothetical notional past available margin).
    cfg = BacktestConfig(initial_equity=100.05, taker_fee_bps=4.5, min_notional_usd=10.0)
    script = {0: [market_order("SYN", 1, notional=100.0)]}

    sd_a = build_sim_data({
        "SYN": [[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]],
    })
    sd_b = build_sim_data({
        "SYN": [[100, 100, 100, 100], [100, 1000, 100, 1000], [100, 100, 100, 100]],
    })
    res_a = run_scripted(sd_a, script, bt_cfg=cfg)
    res_b = run_scripted(sd_b, script, bt_cfg=cfg)

    assert res_a.rejected_orders == res_b.rejected_orders == []
    assert res_a.execution_events == res_b.execution_events
    opens = [e for e in res_a.execution_events if e.event_type == "position_opened"]
    assert len(opens) == 1 and opens[0].fill_price == 100.0


# --- (iii) unpriceable pending -> reject at bar-i gap scan -------------------

def test_unpriceable_fill_bar_rejected_with_queued_ts():
    sd = build_sim_data({
        "SYN": [[100, 100, 100, 100], [NAN, NAN, NAN, NAN], [100, 100, 100, 100]],
    })
    result = run_scripted(sd, {0: [market_order("SYN", 1, notional=100.0)]})

    assert result.rejected_orders, "expected a rejection"
    rej = result.rejected_orders[0]
    assert rej.asset == "SYN"
    assert rej.reason == "pending order unpriceable at fill bar"
    assert rej.timestamp == _iso(result, 0)          # queued_ts, not fill bar
    assert result.execution_events == []             # never filled


# --- (iv) backtest-end rejection --------------------------------------------

def test_pending_at_last_bar_rejected_backtest_end():
    # Order queued at bar 1 (=N-2) would fill at bar 2 (last) -> rejected.
    sd = build_sim_data({
        "SYN": [[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]],
    })
    result = run_scripted(sd, {1: [market_order("SYN", 1, notional=100.0)]})

    reasons = [r for r in result.rejected_orders if r.reason == "backtest end"]
    assert len(reasons) == 1
    assert reasons[0].asset == "SYN"
    assert reasons[0].timestamp == _iso(result, 1)   # queued_ts
    assert result.execution_events == []


# --- (v) one-bar protection gap ---------------------------------------------

def test_one_bar_protection_gap_then_fires():
    # Entry queued bar0 fills open bar1; trigger placed bar1 (active bar2).
    # bar1 low (80) breaches stop 90 but the trigger is not yet active -> no fill.
    # bar2 low (85) breaches -> fires.
    sd = build_sim_data({
        "SYN": [
            [100, 100, 100, 100],
            [100, 105, 80, 100],
            [100, 105, 85, 100],
            [100, 100, 100, 100],
        ],
    })
    script = {
        0: [market_order("SYN", 1, notional=100.0)],
        1: [trigger_order("SYN", -1, size=1.0, trigger_px=90.0, direction="stop")],
    }
    result = run_scripted(sd, script)

    # No trigger fill on bar 1.
    assert all(e.reason != "trigger" for e in _exec_at(result, _iso(result, 1)))
    # Trigger fires on bar 2 at the stop price (open 100 not <= 90 -> fill at 90).
    bar2 = [e for e in _exec_at(result, _iso(result, 2)) if e.reason == "trigger"]
    assert len(bar2) == 1
    assert bar2[0].fill_price == 90.0
    assert bar2[0].event_type == "position_closed"


# --- (vi) funding runs after triggers ---------------------------------------

def test_no_funding_for_asset_closed_by_trigger_that_bar():
    sd = build_sim_data(
        {
            "SYN": [
                [100, 100, 100, 100],
                [100, 105, 95, 100],
                [100, 105, 85, 100],
                [100, 100, 100, 100],
            ],
        },
        funding={"SYN": [0.001, 0.001, 0.001, 0.001]},
    )
    script = {
        0: [market_order("SYN", 1, notional=100.0)],
        1: [trigger_order("SYN", -1, size=1.0, trigger_px=90.0, direction="stop")],
    }
    result = run_scripted(sd, script)

    # Trigger closes SYN on bar 2 before funding runs -> no funding event bar 2.
    bar2_funding = [
        f for f in result.funding_events
        if f.asset == "SYN" and f.timestamp == _iso(result, 2)
    ]
    assert bar2_funding == []
    # Sanity: the trigger did fire on bar 2.
    assert any(e.reason == "trigger" for e in _exec_at(result, _iso(result, 2)))


# --- (vii) last-bar trigger fires before close_all (reason "trigger") --------

def test_last_bar_trigger_fires_not_force():
    sd = build_sim_data({
        "SYN": [
            [100, 100, 100, 100],
            [100, 105, 95, 100],
            [100, 105, 85, 100],   # last bar, low breaches stop 90
        ],
    })
    script = {
        0: [market_order("SYN", 1, notional=100.0)],
        1: [trigger_order("SYN", -1, size=1.0, trigger_px=90.0, direction="stop")],
    }
    result = run_scripted(sd, script)

    last = _exec_at(result, _iso(result, 2))
    closes = [e for e in last if e.event_type == "position_closed"]
    assert len(closes) == 1
    assert closes[0].reason == "trigger"          # not "force"
    assert closes[0].fill_price == 90.0
    assert all(e.reason != "force" for e in last)


# --- (viii) determinism ------------------------------------------------------

def test_determinism_two_runs_byte_identical():
    sd_kwargs = dict(
        ohlc={
            "SYN": [
                [100, 100, 100, 100],
                [100, 105, 80, 100],
                [100, 105, 85, 100],
                [100, 100, 100, 100],
            ],
        },
    )
    script = {
        0: [market_order("SYN", 1, notional=100.0)],
        1: [trigger_order("SYN", -1, size=1.0, trigger_px=90.0, direction="stop")],
    }
    res1 = run_scripted(build_sim_data(**sd_kwargs), script)
    res2 = run_scripted(build_sim_data(**sd_kwargs), script)

    assert np.array_equal(res1.total_equity, res2.total_equity)
    assert res1.execution_events == res2.execution_events
    assert res1.rejected_orders == res2.rejected_orders
    assert res1.funding_events == res2.funding_events
    assert res1.liquidation_events == res2.liquidation_events


# --- (ix) liquidated asset's triggers auto-cancelled ------------------------

def test_liquidation_cancels_resting_triggers():
    # SYN long at 10x liquidates on the bar-2 crash. Its resting stop was
    # suppressed by the precedence scan (R5), so liquidation strictly wins and
    # the trigger is swept — it must never produce a "trigger" execution event.
    sd = build_sim_data({
        "SYN": [
            [100, 100, 100, 100],
            [100, 105, 95, 100],
            [95, 95, 40, 45],      # crash: extreme low 40 breaches, close 45 liquidates
            [45, 45, 45, 45],
        ],
    })
    cfg = BacktestConfig(initial_equity=100.0, taker_fee_bps=4.5, min_notional_usd=10.0)
    script = {
        0: [market_order("SYN", 1, notional=900.0, leverage=10.0)],
        1: [trigger_order("SYN", -1, size=9.0, trigger_px=90.0, direction="stop")],
    }
    result = run_scripted(sd, script, bt_cfg=cfg)

    assert any(e.asset == "SYN" for e in result.liquidation_events)
    assert all(e.reason != "trigger" for e in result.execution_events)
    assert result.n_liquidated >= 1
