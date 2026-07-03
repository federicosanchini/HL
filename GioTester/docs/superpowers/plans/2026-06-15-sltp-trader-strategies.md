# SL/TP Trader Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five standalone stop-loss / take-profit strategy files to `GioTester/Traders/`, each sharing a HODL-style rank-based entry and differing only in exit discipline.

**Architecture:** Each file is a self-contained `Trader` class (no shared import between trader files — standalone is the project UX). All five share an identical skeleton: rank universe by `pred_10d` on release bars, long top-`n` / short bottom-`n` with fixed notional, skip names already held, evaluate a signed return every bar against a strategy-tracked entry price, and force-close at a 10-day expiry backstop. They differ only in the exit predicate (bracket / SL-only / TP-only / asymmetric / trailing). No engine code changes.

**Tech Stack:** Python 3.12, `src` public API (`OrderCommand`, `OrderType`, `load_trader`), pytest for unit tests, pandas for test timestamps.

---

## Background facts (verified against the codebase)

- `state.current_ranks_row[asset]` is `(pred_10d, pred_30d)`; index 0 is the 10-day signal (`HODL10` uses the same). Present only on `state.is_release_bar`.
- The engine (`runner.run_backtest`) marks positions **before** calling `strategy.run(state)`, so `state.market[a].mark_px` and `PositionView.mark_price` are current-bar.
- Orders returned by `run()` are filled within the same bar at `trade_px`.
- `execution.execute_one` rejects **any** order whose notional `< min_notional_usd` (`execution.py:203`), reduce-only included. So an exit must be skipped when `abs(size) * mark_px < min_notional_usd` (mirrors `HODL10._expiry_orders`); residual dust is cleared by the engine's `close_all` at the final bar.
- `load_trader(path, **kwargs)` instantiates `Trader(**kwargs)` (`strategy_loader.py:31`).
- `tester.py` injects shared kwargs: `n` (=2), `leverage`, `notional_long`, `notional_short`, `taker_fee_bps`, `min_notional_usd`, `blackout_days_end`, `bars_per_day`. `margin_mode` is **not** injected — default it in `__init__` like `HODL10`.

## Shared exit/entry semantics (reference — every file embeds its own copy)

- **Signed return:** `ret = side * (mark_px / entry_price - 1)`, `side = +1` long / `-1` short. `ret > 0` favorable for both sides.
- **Entry:** on `is_release_bar` with ranks, rank `(asset, pred_10d)` for finite-signal assets in `state.market`; sort ascending by `(signal, asset)`; `shorts = first n`, `longs = last n`; drop shorts that overlap longs; skip any asset already in `state.positions`; record `_entry_bar[asset]` and `_entry_px[asset] = mark_px`.
- **Exit:** every bar, for each held position with a recorded `_entry_px`, compute `ret`; close (reduce-only, `size=abs(pos.size)`) when the strategy's exit predicate fires **or** `bar_index - entry_bar >= expiry_bars` (`expiry_bars = 10 * bars_per_day`); skip the close if `abs(size)*mark_px < min_notional_usd`.
- **Pruning:** at the top of the exit pass, drop tracking-dict entries for assets no longer in `state.positions`.

## File Structure

- Create: `GioTester/Traders/SLTP_Bracket.py` — `BracketSLTP` (SL −5% + TP +10%)
- Create: `GioTester/Traders/SL_Only.py` — `StopOnly` (SL −5%, no TP)
- Create: `GioTester/Traders/TP_Only.py` — `TargetOnly` (TP +10%, no SL)
- Create: `GioTester/Traders/SLTP_Asym.py` — `AsymRR` (SL −4% + TP +12%)
- Create: `GioTester/Traders/Trailing.py` — `TrailStop` (trailing 5% from peak `ret`)
- Create: `GioTester/tests/strategy_harness.py` — builds `StrategyState` fixtures + loads traders
- Create: `GioTester/tests/test_sltp_strategies.py` — unit tests for all five
- Modify: none (no engine changes)

---

## Task 0: Test dependency + harness

**Files:**
- Create: `GioTester/tests/strategy_harness.py`

- [ ] **Step 1: Ensure pytest is installed**

Run: `python -m pytest --version`
Expected: prints a version (e.g. `pytest 8.x`). If it errors with "No module named pytest", run `python -m pip install pytest` and re-run.

- [ ] **Step 2: Write the harness**

Create `GioTester/tests/strategy_harness.py`:

```python
"""Test helpers: build frozen StrategyState fixtures and load trader files."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_trader  # noqa: E402
from src.dto import (  # noqa: E402
    AccountView,
    MarketAssetView,
    PositionView,
    StrategyState,
)

TRADERS = ROOT / "Traders"

SHARED_KWARGS = dict(
    n=2,
    leverage=1.0,
    notional_long=10.0,
    notional_short=10.0,
    taker_fee_bps=4.5,
    min_notional_usd=10.0,
    blackout_days_end=0,
    bars_per_day=24,
)


def load(filename: str, **overrides):
    kwargs = dict(SHARED_KWARGS)
    kwargs.update(overrides)
    return load_trader(str(TRADERS / filename), **kwargs)


def market_view(asset: str, mark_px: float, *, trade_px=None, funding_rate=0.0):
    px = float(mark_px)
    return MarketAssetView(
        asset=asset,
        trade_px=float(px if trade_px is None else trade_px),
        mark_px=px,
        oracle_px=px,
        funding_rate=float(funding_rate),
    )


def position_view(asset: str, size: float, entry_price: float, mark_price: float):
    size = float(size)
    entry_price = float(entry_price)
    mark_price = float(mark_price)
    return PositionView(
        asset=asset,
        size=size,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=size * (mark_price - entry_price),
        notional_at_mark=abs(size) * mark_price,
        signed_invested_notional=size * entry_price,
        cumulative_funding=0.0,
        cumulative_fees=0.0,
    )


def make_state(*, bar_index=0, total_bars=100000, market=None, positions=None,
               ranks=None, is_release=False, timestamp=None):
    return StrategyState(
        timestamp=timestamp or pd.Timestamp("2025-10-10", tz="UTC"),
        bar_index=bar_index,
        total_bars=total_bars,
        market=market or {},
        positions=positions or {},
        account=AccountView(
            cash=2000.0,
            equity=2000.0,
            initial_margin_required=0.0,
            maintenance_margin_required=0.0,
            available_balance=2000.0,
        ),
        current_ranks_row=ranks,
        is_release_bar=is_release,
    )
```

- [ ] **Step 3: Sanity-check the harness imports**

Run: `python -c "import sys; sys.path.insert(0,'tests'); import strategy_harness; print('ok')"`
Expected: prints `ok` (confirms `src` and `src.dto` import cleanly).

- [ ] **Step 4: Commit**

```bash
git add GioTester/tests/strategy_harness.py
git commit -m "test: add strategy test harness for SL/TP traders"
```

---

## Task 1: BracketSLTP (SL + TP) + shared-entry tests

**Files:**
- Create: `GioTester/Traders/SLTP_Bracket.py`
- Test: `GioTester/tests/test_sltp_strategies.py`

- [ ] **Step 1: Write failing tests (entry behavior + bracket exits)**

Create `GioTester/tests/test_sltp_strategies.py`:

```python
from __future__ import annotations

import math

from strategy_harness import load, make_state, market_view, position_view


def _release_state(bar_index=0):
    # Five ranked assets; ascending signal order DDD < CCC < EEE < BBB < AAA.
    ranks = {
        "AAA": (0.90, 0.0),
        "BBB": (0.80, 0.0),
        "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0),
        "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_entry_longs_top_shorts_bottom_n2():
    t = load("SLTP_Bracket.py")
    orders = t.run(_release_state())
    longs = {o.asset for o in orders if o.side == 1}
    shorts = {o.asset for o in orders if o.side == -1}
    assert longs == {"AAA", "BBB"}
    assert shorts == {"CCC", "DDD"}
    assert all(o.order_type == "market" and not o.reduce_only for o in orders)
    assert all(o.notional == 10.0 for o in orders)


def test_entry_skips_already_held():
    t = load("SLTP_Bracket.py")
    # AAA already held -> not re-entered; BBB still entered long.
    state = _release_state()
    state = make_state(
        bar_index=0,
        market=state.market,
        ranks=state.current_ranks_row,
        is_release=True,
        positions={"AAA": position_view("AAA", 0.1, 100.0, 100.0)},
    )
    orders = t.run(state)
    longs = {o.asset for o in orders if o.side == 1}
    assert "AAA" not in longs
    assert "BBB" in longs


def _prime_long(trader, asset="AAA", entry=100.0):
    """Run an entry release bar so the trader records entry_px for `asset`."""
    trader.run(_release_state(bar_index=0))
    assert math.isclose(trader._entry_px[asset], entry)


def _exit_state(asset, size, entry, mark, bar_index=5):
    return make_state(
        bar_index=bar_index,
        market={asset: market_view(asset, mark)},
        positions={asset: position_view(asset, size, entry, mark)},
        is_release=False,
    )


def test_bracket_long_take_profit():
    t = load("SLTP_Bracket.py")  # sl=0.05, tp=0.10
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 111.0))  # +11%
    assert len(orders) == 1 and orders[0].reduce_only and orders[0].side == -1


def test_bracket_long_stop_loss():
    t = load("SLTP_Bracket.py")
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0))  # -6%, mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_bracket_long_inside_band_no_exit():
    t = load("SLTP_Bracket.py")
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 103.0))  # +3%
    assert orders == []


def test_bracket_short_take_profit():
    t = load("SLTP_Bracket.py")
    t.run(_release_state(bar_index=0))  # primes DDD short at 100
    assert math.isclose(t._entry_px["DDD"], 100.0)
    orders = t.run(_exit_state("DDD", -0.2, 100.0, 88.0))  # short +12% favorable, notional 17.6 >= min
    assert len(orders) == 1 and orders[0].reduce_only and orders[0].side == 1


def test_expiry_forces_close():
    t = load("SLTP_Bracket.py", bars_per_day=24)  # expiry_bars = 240
    _prime_long(t)
    # inside band but past expiry -> close
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 102.0, bar_index=240))
    assert len(orders) == 1 and orders[0].reduce_only


def test_exit_skipped_below_min_notional():
    t = load("SLTP_Bracket.py", min_notional_usd=10.0)
    _prime_long(t)
    # size*mark = 0.05*94 = 4.7 < 10 -> no order despite stop breach
    orders = t.run(_exit_state("AAA", 0.05, 100.0, 94.0))
    assert orders == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sltp_strategies.py -v`
Expected: FAIL — collection error / `FileNotFoundError: strategy file not found: ...SLTP_Bracket.py`.

- [ ] **Step 3: Implement `SLTP_Bracket.py`**

Create `GioTester/Traders/SLTP_Bracket.py`:

```python
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry with a symmetric stop-loss / take-profit bracket."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        sl_pct: float = 0.05,
        tp_pct: float = 0.10,
        **_,
    ) -> None:
        self.name = "BracketSLTP"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}

    def _should_exit(self, ret: float) -> bool:
        return ret <= -self.sl_pct or ret >= self.tp_pct

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (self._should_exit(ret) or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sltp_strategies.py -v`
Expected: PASS (all tests in the file green).

- [ ] **Step 5: Commit**

```bash
git add GioTester/Traders/SLTP_Bracket.py GioTester/tests/test_sltp_strategies.py
git commit -m "feat: add BracketSLTP strategy with SL+TP exits"
```

---

## Task 2: StopOnly (SL only)

**Files:**
- Create: `GioTester/Traders/SL_Only.py`
- Test: append to `GioTester/tests/test_sltp_strategies.py`

- [ ] **Step 1: Write failing tests**

Append to `GioTester/tests/test_sltp_strategies.py`:

```python
def test_stop_only_closes_on_stop():
    t = load("SL_Only.py")  # sl=0.05, no TP
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0))  # -6%, mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_stop_only_lets_winner_run():
    t = load("SL_Only.py")
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 130.0))  # +30%, no TP
    assert orders == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sltp_strategies.py -k stop_only -v`
Expected: FAIL — `FileNotFoundError: ...SL_Only.py`.

- [ ] **Step 3: Implement `SL_Only.py`**

Create `GioTester/Traders/SL_Only.py` — identical to `SLTP_Bracket.py` except the class docstring, `self.name`, the `__init__` signature (drop `tp_pct`, keep `sl_pct`), and `_should_exit`. Full file:

```python
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry; hard stop-loss only — winners ride to expiry."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        sl_pct: float = 0.05,
        **_,
    ) -> None:
        self.name = "StopOnly"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.sl_pct = float(sl_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}

    def _should_exit(self, ret: float) -> bool:
        return ret <= -self.sl_pct

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (self._should_exit(ret) or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_sltp_strategies.py -k stop_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add GioTester/Traders/SL_Only.py GioTester/tests/test_sltp_strategies.py
git commit -m "feat: add StopOnly strategy (stop-loss only)"
```

---

## Task 3: TargetOnly (TP only)

**Files:**
- Create: `GioTester/Traders/TP_Only.py`
- Test: append to `GioTester/tests/test_sltp_strategies.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_target_only_closes_on_target():
    t = load("TP_Only.py")  # tp=0.10, no SL
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 111.0))  # +11%
    assert len(orders) == 1 and orders[0].reduce_only


def test_target_only_rides_loser():
    t = load("TP_Only.py")
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 70.0))  # -30%, no SL
    assert orders == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sltp_strategies.py -k target_only -v`
Expected: FAIL — `FileNotFoundError: ...TP_Only.py`.

- [ ] **Step 3: Implement `TP_Only.py`**

Create `GioTester/Traders/TP_Only.py` — identical to `SL_Only.py` except docstring, `self.name = "TargetOnly"`, `__init__` takes `tp_pct: float = 0.10` instead of `sl_pct`, stores `self.tp_pct`, and:

```python
    def _should_exit(self, ret: float) -> bool:
        return ret >= self.tp_pct
```

Full file:

```python
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry; take-profit only — losers ride to expiry."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        tp_pct: float = 0.10,
        **_,
    ) -> None:
        self.name = "TargetOnly"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.tp_pct = float(tp_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}

    def _should_exit(self, ret: float) -> bool:
        return ret >= self.tp_pct

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (self._should_exit(ret) or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_sltp_strategies.py -k target_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add GioTester/Traders/TP_Only.py GioTester/tests/test_sltp_strategies.py
git commit -m "feat: add TargetOnly strategy (take-profit only)"
```

---

## Task 4: AsymRR (tight SL, wide TP)

**Files:**
- Create: `GioTester/Traders/SLTP_Asym.py`
- Test: append to `GioTester/tests/test_sltp_strategies.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_asym_tight_stop_fires():
    t = load("SLTP_Asym.py")  # sl=0.04, tp=0.12
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 95.5))  # -4.5%, mark notional 19.1 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_asym_wide_target_holds_then_fires():
    t = load("SLTP_Asym.py")
    t.run(_release_state(bar_index=0))
    assert t.run(_exit_state("AAA", 0.1, 100.0, 108.0)) == []   # +8% < 12%
    assert len(t.run(_exit_state("AAA", 0.1, 100.0, 113.0))) == 1  # +13%
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sltp_strategies.py -k asym -v`
Expected: FAIL — `FileNotFoundError: ...SLTP_Asym.py`.

- [ ] **Step 3: Implement `SLTP_Asym.py`**

Create `GioTester/Traders/SLTP_Asym.py` — identical to `SLTP_Bracket.py` except docstring, `self.name = "AsymRR"`, and the `__init__` defaults `sl_pct: float = 0.04, tp_pct: float = 0.12`. `_should_exit` is the same bracket predicate. Full file:

```python
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry; asymmetric bracket — tight stop, wide target (1:3)."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        sl_pct: float = 0.04,
        tp_pct: float = 0.12,
        **_,
    ) -> None:
        self.name = "AsymRR"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}

    def _should_exit(self, ret: float) -> bool:
        return ret <= -self.sl_pct or ret >= self.tp_pct

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (self._should_exit(ret) or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_sltp_strategies.py -k asym -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add GioTester/Traders/SLTP_Asym.py GioTester/tests/test_sltp_strategies.py
git commit -m "feat: add AsymRR strategy (tight stop, wide target)"
```

---

## Task 5: TrailStop (trailing stop)

**Files:**
- Create: `GioTester/Traders/Trailing.py`
- Test: append to `GioTester/tests/test_sltp_strategies.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_trailing_ratchets_and_gives_back():
    t = load("Trailing.py")  # trail=0.05
    t.run(_release_state(bar_index=0))  # entry AAA @ 100
    # climb to +10% (peak), no exit
    assert t.run(_exit_state("AAA", 0.1, 100.0, 110.0, bar_index=1)) == []
    # give back to +6%: 10% - 6% = 4% < 5% -> no exit
    assert t.run(_exit_state("AAA", 0.1, 100.0, 106.0, bar_index=2)) == []
    # give back to +4%: 10% - 4% = 6% >= 5% -> exit
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 104.0, bar_index=3))
    assert len(orders) == 1 and orders[0].reduce_only


def test_trailing_initial_stop():
    t = load("Trailing.py")
    t.run(_release_state(bar_index=0))  # entry AAA @ 100
    # never profitable; peak stays 0; -6% <= 0 - 5% -> exit
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0, bar_index=1))  # mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sltp_strategies.py -k trailing -v`
Expected: FAIL — `FileNotFoundError: ...Trailing.py`.

- [ ] **Step 3: Implement `Trailing.py`**

Create `GioTester/Traders/Trailing.py`. Differences from the others: adds `trail_pct`, a `self._peak_ret` dict, peak tracking + pruning in `_exit_orders`, and seeds `_peak_ret[asset] = 0.0` on entry. Full file:

```python
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src import OrderCommand, OrderType


class Trader:
    """Rank-based L/S entry; trailing stop that ratchets up with favorable moves."""

    def __init__(
        self,
        *,
        n: int = 3,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        bars_per_day: int = 24,
        trail_pct: float = 0.05,
        **_,
    ) -> None:
        self.name = "TrailStop"
        self.n = int(n)
        self.leverage = float(leverage)
        self.margin_mode = str(margin_mode).lower()
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.bars_per_day = int(bars_per_day)
        self.expiry_bars = 10 * self.bars_per_day
        self.trail_pct = float(trail_pct)
        self._entry_bar: Dict[str, int] = {}
        self._entry_px: Dict[str, float] = {}
        self._peak_ret: Dict[str, float] = {}

    def _signed_ret(self, size: float, mark_px: float, entry_px: float) -> float:
        side = 1.0 if size > 0 else -1.0
        return side * (mark_px / entry_px - 1.0)

    def _rank_candidates(self, state) -> List[Tuple[str, float]]:
        if not state.current_ranks_row:
            return []
        cands: List[Tuple[str, float]] = []
        for asset, preds in state.current_ranks_row.items():
            if asset not in state.market:
                continue
            sig = preds[0]
            if math.isfinite(sig):
                cands.append((asset, float(sig)))
        return cands

    def _exit_orders(self, state) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._entry_bar):
            if asset not in live:
                self._entry_bar.pop(asset, None)
                self._entry_px.pop(asset, None)
                self._peak_ret.pop(asset, None)

        for asset, pos in state.positions.items():
            entry_bar = self._entry_bar.setdefault(asset, state.bar_index)
            entry_px = self._entry_px.get(asset)
            mv = state.market.get(asset)
            if entry_px is None or entry_px <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = self._signed_ret(pos.size, mark_px, entry_px)
            peak = max(self._peak_ret.get(asset, 0.0), ret)
            self._peak_ret[asset] = peak
            stopped = ret <= peak - self.trail_pct
            expired = (state.bar_index - entry_bar) >= self.expiry_bars
            if not (stopped or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=-1 if pos.size > 0 else 1,
                    order_type=OrderType.MARKET.value,
                    size=abs(pos.size),
                    leverage=self.leverage,
                    reduce_only=True,
                )
            )
        return orders

    def _entry_orders(self, state) -> List[OrderCommand]:
        cands = self._rank_candidates(state)
        if len(cands) < 2:
            return []
        cands.sort(key=lambda t: (t[1], t[0]))
        shorts = cands[: self.n]
        longs = cands[-self.n :]
        long_set = {p for p, _ in longs}
        shorts = [(p, s) for p, s in shorts if p not in long_set]
        held = set(state.positions)
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            if asset in held:
                return
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
            self._entry_bar[asset] = state.bar_index
            self._entry_px[asset] = float(mv.mark_px)
            self._peak_ret[asset] = 0.0
            orders.append(
                OrderCommand(
                    asset=asset,
                    side=side,
                    order_type=OrderType.MARKET.value,
                    notional=notional,
                    leverage=self.leverage,
                )
            )

        if self.notional_long >= self.min_notional_usd:
            for asset, _ in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset, _ in shorts:
                _open(asset, -1, self.notional_short)
        return orders

    def run(self, state) -> List[OrderCommand]:
        orders = self._exit_orders(state)
        if state.is_release_bar and state.current_ranks_row:
            orders.extend(self._entry_orders(state))
        return orders
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_sltp_strategies.py -k trailing -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `python -m pytest tests/test_sltp_strategies.py -v`
Expected: PASS (all tests across the five strategies green).

- [ ] **Step 6: Commit**

```bash
git add GioTester/Traders/Trailing.py GioTester/tests/test_sltp_strategies.py
git commit -m "feat: add TrailStop strategy (trailing stop)"
```

---

## Task 6: End-to-end integration via tester.py

**Files:**
- None created. Verifies the five files run inside the real backtest loop and produce result JSON.

> Requires the `../data/` CSVs to be present (see `GioTester/CLAUDE.md` Data Requirements). If data is absent, this task is blocked — note it and stop; do not fabricate data.

- [ ] **Step 1: Confirm data is available**

Run: `python test_load.py`
Expected: completes without assertion errors (data loads).

- [ ] **Step 2: Run the full backtester**

Run: `python tester.py`
Expected: tqdm progress bars for each discovered strategy, including `BracketSLTP`, `StopOnly`, `TargetOnly`, `AsymRR`, `TrailStop`, with no exceptions and no `cash went non-finite` error.

- [ ] **Step 3: Verify result JSON written for each new strategy**

Run: `python -c "from pathlib import Path; names=['BracketSLTP','StopOnly','TargetOnly','AsymRR','TrailStop']; missing=[n for n in names if not (Path('results')/f'{n}.json').exists()]; print('MISSING:',missing) if missing else print('all 5 result files present')"`
Expected: `all 5 result files present`.

- [ ] **Step 4: Sanity-check exit asymmetry in the rejected/closed stats**

Run: `python -c "import json; d=json.load(open('results/StopOnly.json')); print('StopOnly closes:', d.get('n_closed'))"`
Expected: prints an integer ≥ 0 without error (confirms schema-valid output; deeper behavioral comparison is done visually in GioVisualizer).

- [ ] **Step 5: Commit (only if results are tracked for your workflow; otherwise skip)**

Per `GioTester/CLAUDE.md`, `results/` is generated output and should not be committed. No commit in this task unless you have explicitly decided to track results.

---

## Self-Review

**Spec coverage:**
- Shared rank entry (top-`n`/bottom-`n`, `pred_10d`, skip-held, fixed notional) → every task's `_entry_orders`; asserted in Task 1 `test_entry_*`.
- Signed-return side-symmetric exit → `_signed_ret`; asserted via long + short tests (Task 1 `test_bracket_short_take_profit`).
- Five exit variants (bracket / SL-only / TP-only / asym / trailing) → Tasks 1–5, each with targeted tests.
- 10d expiry backstop → `expiry_bars`; asserted Task 1 `test_expiry_forces_close`.
- Fire-and-forget (no rotation) → no ranking-driven close path exists in `_exit_orders`; entries only added on release bars.
- Sub-min-notional close skip → asserted Task 1 `test_exit_skipped_below_min_notional`.
- Deterministic / frozen-view conformance → strategies read `state` only; no mutation; integration run (Task 6) exercises the real engine.
- JSON schema v5 untouched → no engine changes; Task 6 confirms result files load.

**Placeholder scan:** No TBD/TODO; every code step contains full file or full test content; commands have explicit expected output.

**Type consistency:** All five files use the same names — `_entry_bar`, `_entry_px`, `_signed_ret(size, mark_px, entry_px)`, `_rank_candidates`, `_exit_orders`, `_entry_orders`, `run`. `TrailStop` adds `_peak_ret`. Tests reach into `_entry_px` (priming assertion) — defined in all five `__init__`. Harness `load()` passes only kwargs the `__init__`s accept (others absorbed by `**_`).

**Note on duplication:** The five files intentionally repeat the skeleton — standalone strategy files are the project's UX contract (no cross-file imports in `Traders/`). This is deliberate, not a DRY violation to refactor.
