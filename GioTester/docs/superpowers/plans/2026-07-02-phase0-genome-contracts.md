# Phase 0 — Genome & Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a compositional-genome layer so a strategy is `{universe_filter, signal, entry_timing, sizing, exit_rule}`, and a `build_trader(genome)` adapter produces a standard `Trader` object the existing engine runs unchanged.

**Architecture:** A new `src/genome/` package. Five typed gene Protocols; a slot→kind registry; a reference gene library that registers on import; a `ComposedTrader` whose `run(state)` orchestrates the genes in a fixed pipeline (prune ledger → exits → gated entries), maintaining a central `EntryLedger` so exit genes need no private entry-tracking. Proven by asserting a genome reconstruction of `SLTP_Bracket` emits byte-identical orders to the legacy `Traders/SLTP_Bracket.py`.

**Tech Stack:** Python 3.12, stdlib (`dataclasses`, `typing.Protocol`, `math`), `pytest`. Genes import only from the `src` public API (`OrderCommand`, `OrderType`).

## Global Constraints

- Python 3.12; run tests with `python -m pytest` from the `GioTester/` directory.
- Genes and the adapter import trading types **only** from the `src` public API: `from src import OrderCommand, OrderType`.
- **Do not modify `src/__init__.py`** — the top-level public API is a project invariant. The genome surface is exposed via `from src.genome import ...` only.
- **Determinism is a hard invariant.** Candidate selection sorts by `(score, asset)` ascending; never rely on dict insertion order for ranking.
- Engine invariants (matching, margin, funding, liquidation, schema v5) are untouched: the adapter only produces standard `Trader` objects consumed by the existing `run_backtest`.
- Test files live flat in `tests/` and import fixtures via `from strategy_harness import ...` (matches `tests/test_sltp_strategies.py`).
- Commit steps assume the user has opted into execution; git is not run while authoring this plan.

---

### Task 1: EntryLedger + gene Protocols

**Files:**
- Create: `src/genome/__init__.py` (empty for now — populated in Task 6)
- Create: `src/genome/genes.py`
- Test: `tests/test_genome_ledger.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EntryRecord(bar_index: int, price: float)` — frozen dataclass.
  - `EntryLedger` with `record(asset: str, bar_index: int, price: float) -> None`, `get(asset: str) -> EntryRecord | None`, `prune(live_assets: Set[str]) -> None`, `__contains__(asset) -> bool`.
  - Protocols `UniverseFilter`, `Signal`, `EntryTiming`, `Sizing`, `ExitRule` (typing only).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_ledger.py
from __future__ import annotations

from src.genome.genes import EntryLedger


def test_record_and_get():
    led = EntryLedger()
    led.record("BTC", bar_index=5, price=100.0)
    rec = led.get("BTC")
    assert rec is not None
    assert rec.bar_index == 5
    assert rec.price == 100.0
    assert "BTC" in led
    assert led.get("ETH") is None


def test_prune_drops_absent_assets():
    led = EntryLedger()
    led.record("BTC", 1, 100.0)
    led.record("ETH", 1, 200.0)
    led.prune({"BTC"})
    assert "BTC" in led
    assert "ETH" not in led


def test_record_overwrites():
    led = EntryLedger()
    led.record("BTC", 1, 100.0)
    led.record("BTC", 9, 150.0)
    rec = led.get("BTC")
    assert rec.bar_index == 9 and rec.price == 150.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genome'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/genome/__init__.py
# (intentionally empty; public surface added in Task 6)
```

```python
# src/genome/genes.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Set, runtime_checkable

from src import OrderCommand


@dataclass(frozen=True)
class EntryRecord:
    bar_index: int
    price: float


@dataclass
class EntryLedger:
    """Central record of when/at-what-price each open position was entered.

    The adapter maintains one ledger per backtest so exit genes stay stateless
    with respect to entry tracking.
    """

    _by_asset: Dict[str, EntryRecord] = field(default_factory=dict)

    def record(self, asset: str, bar_index: int, price: float) -> None:
        self._by_asset[asset] = EntryRecord(bar_index=int(bar_index), price=float(price))

    def get(self, asset: str) -> EntryRecord | None:
        return self._by_asset.get(asset)

    def prune(self, live_assets: Set[str]) -> None:
        for asset in list(self._by_asset):
            if asset not in live_assets:
                del self._by_asset[asset]

    def __contains__(self, asset: object) -> bool:
        return asset in self._by_asset


@runtime_checkable
class UniverseFilter(Protocol):
    def eligible(self, state) -> Set[str]: ...


@runtime_checkable
class Signal(Protocol):
    def score(self, state, universe: Set[str]) -> Dict[str, float]: ...


@runtime_checkable
class EntryTiming(Protocol):
    def should_enter(self, state) -> bool: ...


@runtime_checkable
class Sizing(Protocol):
    def orders_for(self, state, longs: List[str], shorts: List[str]) -> List[OrderCommand]: ...


@runtime_checkable
class ExitRule(Protocol):
    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_ledger.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/genome/__init__.py src/genome/genes.py tests/test_genome_ledger.py
git commit -m "feat(genome): EntryLedger + gene protocols"
```

---

### Task 2: Slot→kind registry

**Files:**
- Create: `src/genome/registry.py`
- Test: `tests/test_genome_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SLOTS: tuple[str, ...]` = `("universe_filter", "signal", "entry_timing", "sizing", "exit_rule")`.
  - `register(slot: str, kind: str) -> Callable[[type], type]` — class decorator; raises `KeyError` on unknown slot or duplicate kind.
  - `build_gene(slot: str, kind: str, params: dict) -> object` — instantiates `cls(**params)`; raises `KeyError` if unregistered.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_registry.py
from __future__ import annotations

import pytest

from src.genome.registry import SLOTS, build_gene, register


def test_register_and_build():
    @register("signal", "dummy_test")
    class Dummy:
        def __init__(self, *, k=1):
            self.k = k

    obj = build_gene("signal", "dummy_test", {"k": 7})
    assert obj.k == 7


def test_slots_are_the_five_expected():
    assert SLOTS == (
        "universe_filter",
        "signal",
        "entry_timing",
        "sizing",
        "exit_rule",
    )


def test_unknown_slot_raises():
    with pytest.raises(KeyError):
        register("not_a_slot", "x")


def test_duplicate_kind_raises():
    @register("sizing", "dup_test")
    class A:
        pass

    with pytest.raises(KeyError):
        @register("sizing", "dup_test")
        class B:
            pass


def test_build_unregistered_raises():
    with pytest.raises(KeyError):
        build_gene("exit_rule", "does_not_exist", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genome.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/genome/registry.py
from __future__ import annotations

from typing import Callable, Dict, Type

SLOTS = (
    "universe_filter",
    "signal",
    "entry_timing",
    "sizing",
    "exit_rule",
)

_REGISTRY: Dict[str, Dict[str, Type]] = {slot: {} for slot in SLOTS}


def register(slot: str, kind: str) -> Callable[[Type], Type]:
    if slot not in _REGISTRY:
        raise KeyError(f"unknown slot: {slot!r}; expected one of {SLOTS}")

    def _decorator(cls: Type) -> Type:
        if kind in _REGISTRY[slot]:
            raise KeyError(f"duplicate gene kind {kind!r} in slot {slot!r}")
        _REGISTRY[slot][kind] = cls
        return cls

    return _decorator


def build_gene(slot: str, kind: str, params: dict) -> object:
    try:
        cls = _REGISTRY[slot][kind]
    except KeyError as exc:
        raise KeyError(f"no gene registered for slot {slot!r} kind {kind!r}") from exc
    return cls(**params)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/genome/registry.py tests/test_genome_registry.py
git commit -m "feat(genome): slot->kind gene registry"
```

---

### Task 3: Entry-side reference genes (filter, signal, timing, sizing)

**Files:**
- Create: `src/genome/library.py`
- Test: `tests/test_genome_library.py`

**Interfaces:**
- Consumes: `register` (Task 2); `OrderCommand`, `OrderType` from `src`.
- Produces registered genes:
  - `universe_filter` / `"all_tradable"` → `AllTradableFilter`, `eligible(state) -> set(state.market)`.
  - `signal` / `"rank"` → `RankSignal(field_index: int = 0)`, `score(state, universe) -> {asset: float}` for finite signals of assets in `universe`.
  - `entry_timing` / `"release_bar"` → `ReleaseBarTiming`, `should_enter(state) -> bool` (`state.is_release_bar and state.current_ranks_row`).
  - `sizing` / `"fixed_notional"` → `FixedNotionalSizing(notional_long, notional_short, min_notional_usd, leverage)`, `orders_for(state, longs, shorts) -> [OrderCommand]` (longs first, then shorts; notional-gated; mark-valid guard).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_library.py
from __future__ import annotations

import src.genome.library  # noqa: F401  (registers genes on import)
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view


def _release_state():
    ranks = {
        "AAA": (0.90, 0.0),
        "BBB": (0.80, 0.0),
        "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0),
        "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(market=market, ranks=ranks, is_release=True)


def test_all_tradable_filter_returns_market_keys():
    gene = build_gene("universe_filter", "all_tradable", {})
    state = _release_state()
    assert gene.eligible(state) == set(state.market)


def test_rank_signal_scores_finite_in_universe():
    gene = build_gene("signal", "rank", {})
    state = _release_state()
    universe = set(state.market)
    scores = gene.score(state, universe)
    assert scores["AAA"] == 0.90
    assert set(scores) == universe


def test_rank_signal_excludes_out_of_universe():
    gene = build_gene("signal", "rank", {})
    state = _release_state()
    scores = gene.score(state, {"AAA", "BBB"})
    assert set(scores) == {"AAA", "BBB"}


def test_release_bar_timing():
    gene = build_gene("entry_timing", "release_bar", {})
    assert gene.should_enter(_release_state()) is True
    non_release = make_state(market={}, ranks=None, is_release=False)
    assert gene.should_enter(non_release) is False


def test_fixed_notional_sizing_longs_then_shorts():
    gene = build_gene(
        "sizing",
        "fixed_notional",
        {"notional_long": 10.0, "notional_short": 10.0, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA", "BBB"], ["CCC", "DDD"])
    assert [o.asset for o in orders] == ["AAA", "BBB", "CCC", "DDD"]
    assert [o.side for o in orders] == [1, 1, -1, -1]
    assert all(o.order_type == "market" and not o.reduce_only for o in orders)
    assert all(o.notional == 10.0 for o in orders)


def test_fixed_notional_gates_below_min():
    gene = build_gene(
        "sizing",
        "fixed_notional",
        {"notional_long": 5.0, "notional_short": 10.0, "min_notional_usd": 10.0, "leverage": 1.0},
    )
    state = _release_state()
    orders = gene.orders_for(state, ["AAA"], ["CCC"])
    # long gated out (5 < 10); only the short survives
    assert [o.asset for o in orders] == ["CCC"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_library.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genome.library'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/genome/library.py
from __future__ import annotations

import math
from typing import Dict, List, Set

from src import OrderCommand, OrderType

from .registry import register


@register("universe_filter", "all_tradable")
class AllTradableFilter:
    """Every asset that has a market view this bar is eligible."""

    def eligible(self, state) -> Set[str]:
        return set(state.market.keys())


@register("signal", "rank")
class RankSignal:
    """Score = the prediction at `field_index` from the release-bar ranks row."""

    def __init__(self, *, field_index: int = 0) -> None:
        self.field_index = int(field_index)

    def score(self, state, universe: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not state.current_ranks_row:
            return out
        for asset, preds in state.current_ranks_row.items():
            if asset not in universe:
                continue
            sig = preds[self.field_index]
            if math.isfinite(sig):
                out[asset] = float(sig)
        return out


@register("entry_timing", "release_bar")
class ReleaseBarTiming:
    """Enter only on a signal-release bar."""

    def should_enter(self, state) -> bool:
        return bool(state.is_release_bar and state.current_ranks_row)


@register("sizing", "fixed_notional")
class FixedNotionalSizing:
    """Open a fixed USD notional per selected asset; longs first, then shorts."""

    def __init__(
        self,
        *,
        notional_long: float = 10.0,
        notional_short: float = 10.0,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.notional_long = float(notional_long)
        self.notional_short = float(notional_short)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)

    def orders_for(self, state, longs: List[str], shorts: List[str]) -> List[OrderCommand]:
        orders: List[OrderCommand] = []

        def _open(asset: str, side: int, notional: float) -> None:
            mv = state.market.get(asset)
            if mv is None or not (math.isfinite(mv.mark_px) and mv.mark_px > 0):
                return
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
            for asset in longs:
                _open(asset, 1, self.notional_long)
        if self.notional_short >= self.min_notional_usd:
            for asset in shorts:
                _open(asset, -1, self.notional_short)
        return orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_library.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/genome/library.py tests/test_genome_library.py
git commit -m "feat(genome): entry-side reference genes"
```

---

### Task 4: Exit genes — Bracket family + stateful Trailing

**Files:**
- Modify: `src/genome/library.py` (append exit genes)
- Test: `tests/test_genome_exits.py`

**Interfaces:**
- Consumes: `EntryLedger` (Task 1); `register`; `OrderCommand`, `OrderType`.
- Produces registered genes:
  - `exit_rule` / `"bracket"` → `BracketExit(sl_pct, tp_pct, expiry_bars, min_notional_usd, leverage)`. Exits when signed return ≤ −`sl_pct`, ≥ `tp_pct`, or `bar_index − entry_bar ≥ expiry_bars`; emits a reduce-only market order of full size; skips positions below `min_notional_usd`. Set `tp_pct=math.inf` for stop-only, `sl_pct=math.inf` for take-profit-only, asymmetric via unequal pcts.
  - `exit_rule` / `"trailing"` → `TrailingExit(trail_pct, expiry_bars, min_notional_usd, leverage)`. Tracks per-asset peak favorable price internally; exits when price retraces `trail_pct` from peak or on expiry. Demonstrates a stateful exit gene.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_exits.py
from __future__ import annotations

import math

import src.genome.library  # noqa: F401
from src.genome.genes import EntryLedger
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view, position_view


def _held_state(mark, *, size=1.0, bar_index=0, entry=100.0):
    market = {"BTC": market_view("BTC", mark)}
    positions = {"BTC": position_view("BTC", size=size, entry_price=entry, mark_price=mark)}
    return make_state(bar_index=bar_index, market=market, positions=positions)


def _ledger(entry_bar=0, entry_px=100.0):
    led = EntryLedger()
    led.record("BTC", entry_bar, entry_px)
    return led


def test_bracket_take_profit_fires():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=111.0), _ledger())  # +11% long
    assert len(orders) == 1
    o = orders[0]
    assert o.asset == "BTC" and o.side == -1 and o.reduce_only and o.size == 1.0


def test_bracket_stop_loss_fires():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=94.0), _ledger())  # -6% long
    assert len(orders) == 1 and orders[0].side == -1


def test_bracket_holds_inside_band():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    assert gene.exits(_held_state(mark=103.0), _ledger()) == []  # +3%, no expiry


def test_bracket_expiry_fires_even_inside_band():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 10, "min_notional_usd": 10.0})
    orders = gene.exits(_held_state(mark=101.0, bar_index=10), _ledger(entry_bar=0))
    assert len(orders) == 1


def test_bracket_stop_only_ignores_upside():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": math.inf, "expiry_bars": 240, "min_notional_usd": 10.0})
    assert gene.exits(_held_state(mark=130.0), _ledger()) == []  # +30% never hits tp=inf
    assert len(gene.exits(_held_state(mark=94.0), _ledger())) == 1


def test_bracket_skips_below_min_notional():
    gene = build_gene("exit_rule", "bracket",
                      {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240, "min_notional_usd": 10.0})
    # size 0.05 * mark 111 = 5.55 < 10 -> skip despite tp hit
    assert gene.exits(_held_state(mark=111.0, size=0.05), _ledger()) == []


def test_trailing_exits_on_retrace():
    gene = build_gene("exit_rule", "trailing",
                      {"trail_pct": 0.05, "expiry_bars": 240, "min_notional_usd": 10.0})
    led = _ledger()
    # ride up to 120 (peak), then retrace to 113 (>5% off peak) -> exit
    assert gene.exits(_held_state(mark=110.0), led) == []
    assert gene.exits(_held_state(mark=120.0), led) == []
    orders = gene.exits(_held_state(mark=113.0), led)
    assert len(orders) == 1 and orders[0].reduce_only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_exits.py -v`
Expected: FAIL — `KeyError: "no gene registered for slot 'exit_rule' kind 'bracket'"`

- [ ] **Step 3: Write minimal implementation** (append to `src/genome/library.py`)

```python
# --- append to src/genome/library.py ---
from .genes import EntryLedger


def _reduce_order(asset: str, size: float, leverage: float) -> OrderCommand:
    return OrderCommand(
        asset=asset,
        side=-1 if size > 0 else 1,
        order_type=OrderType.MARKET.value,
        size=abs(size),
        leverage=leverage,
        reduce_only=True,
    )


def _signed_ret(size: float, mark_px: float, entry_px: float) -> float:
    side = 1.0 if size > 0 else -1.0
    return side * (mark_px / entry_px - 1.0)


@register("exit_rule", "bracket")
class BracketExit:
    """Symmetric-or-asymmetric stop/take-profit bracket with a time expiry.

    tp_pct=inf -> stop-only; sl_pct=inf -> take-profit-only.
    """

    def __init__(
        self,
        *,
        sl_pct: float = 0.05,
        tp_pct: float = 0.10,
        expiry_bars: int = 240,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self.expiry_bars = int(expiry_bars)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)

    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        for asset, pos in state.positions.items():
            rec = ledger.get(asset)
            mv = state.market.get(asset)
            if rec is None or rec.price <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = _signed_ret(pos.size, mark_px, rec.price)
            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            hit = ret <= -self.sl_pct or ret >= self.tp_pct
            if not (hit or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(_reduce_order(asset, pos.size, self.leverage))
        return orders


@register("exit_rule", "trailing")
class TrailingExit:
    """Exit when price retraces `trail_pct` from the best favorable price seen."""

    def __init__(
        self,
        *,
        trail_pct: float = 0.05,
        expiry_bars: int = 240,
        min_notional_usd: float = 10.0,
        leverage: float = 1.0,
    ) -> None:
        self.trail_pct = float(trail_pct)
        self.expiry_bars = int(expiry_bars)
        self.min_notional_usd = float(min_notional_usd)
        self.leverage = float(leverage)
        self._peak_ret: Dict[str, float] = {}

    def exits(self, state, ledger: EntryLedger) -> List[OrderCommand]:
        orders: List[OrderCommand] = []
        live = set(state.positions)
        for asset in list(self._peak_ret):
            if asset not in live:
                del self._peak_ret[asset]
        for asset, pos in state.positions.items():
            rec = ledger.get(asset)
            mv = state.market.get(asset)
            if rec is None or rec.price <= 0 or mv is None:
                continue
            mark_px = mv.mark_px
            if not (math.isfinite(mark_px) and mark_px > 0):
                continue
            ret = _signed_ret(pos.size, mark_px, rec.price)
            peak = max(self._peak_ret.get(asset, ret), ret)
            self._peak_ret[asset] = peak
            expired = (state.bar_index - rec.bar_index) >= self.expiry_bars
            retraced = (peak - ret) >= self.trail_pct
            if not (retraced or expired):
                continue
            if abs(pos.size) * mark_px < self.min_notional_usd:
                continue
            orders.append(_reduce_order(asset, pos.size, self.leverage))
        return orders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_exits.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/genome/library.py tests/test_genome_exits.py
git commit -m "feat(genome): bracket + trailing exit genes"
```

---

### Task 5: Genome, GeneSpec, selection, and the ComposedTrader adapter

**Files:**
- Create: `src/genome/adapter.py`
- Test: `tests/test_genome_adapter.py`

**Interfaces:**
- Consumes: `build_gene` (Task 2), `EntryLedger` (Task 1), all registered genes (Tasks 3–4).
- Produces:
  - `GeneSpec(kind: str, params: dict = {})` — frozen.
  - `Genome(name, universe_filter, signal, entry_timing, sizing, exit_rule, n_long=2, n_short=2, margin_mode="cross")` — frozen; slots are `GeneSpec`.
  - `select_longs_shorts(scores: Dict[str,float], n_long: int, n_short: int, held: Set[str]) -> tuple[list[str], list[str]]` — ascending `(score, asset)` sort; `shorts = first n_short`, `longs = last n_long`; drop long/short overlap from shorts; drop held from both; returns `([], [])` if fewer than 2 candidates.
  - `ComposedTrader` with `name`, `margin_mode`, and `run(state) -> list[OrderCommand]`.
  - `build_trader(genome: Genome) -> ComposedTrader`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_adapter.py
from __future__ import annotations

import src.genome.library  # noqa: F401
from src.genome.adapter import GeneSpec, Genome, build_trader, select_longs_shorts
from strategy_harness import make_state, market_view, position_view


def _bracket_genome():
    return Genome(
        name="GenomeBracket",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional", {"notional_long": 10.0, "notional_short": 10.0,
                                           "min_notional_usd": 10.0, "leverage": 1.0}),
        exit_rule=GeneSpec("bracket", {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240,
                                       "min_notional_usd": 10.0, "leverage": 1.0}),
        n_long=2,
        n_short=2,
    )


def _release_state(bar_index=0):
    ranks = {
        "AAA": (0.90, 0.0), "BBB": (0.80, 0.0), "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0), "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_select_top_bottom_n2():
    scores = {"AAA": 0.90, "BBB": 0.80, "EEE": 0.40, "CCC": 0.20, "DDD": 0.05}
    longs, shorts = select_longs_shorts(scores, 2, 2, held=set())
    assert set(longs) == {"AAA", "BBB"}
    assert set(shorts) == {"CCC", "DDD"}


def test_select_excludes_held():
    scores = {"AAA": 0.90, "BBB": 0.80, "CCC": 0.20, "DDD": 0.05}
    longs, shorts = select_longs_shorts(scores, 2, 2, held={"AAA"})
    assert "AAA" not in longs


def test_select_needs_two_candidates():
    assert select_longs_shorts({"AAA": 0.5}, 2, 2, set()) == ([], [])


def test_run_emits_entries_on_release():
    trader = build_trader(_bracket_genome())
    orders = trader.run(_release_state())
    longs = {o.asset for o in orders if o.side == 1}
    shorts = {o.asset for o in orders if o.side == -1}
    assert longs == {"AAA", "BBB"}
    assert shorts == {"CCC", "DDD"}
    assert trader.margin_mode == "cross"
    assert trader.name == "GenomeBracket"


def test_run_no_entries_off_release():
    trader = build_trader(_bracket_genome())
    non_release = make_state(market={}, ranks=None, is_release=False)
    assert trader.run(non_release) == []


def test_run_exits_tracked_entry_on_tp():
    trader = build_trader(_bracket_genome())
    trader.run(_release_state())  # opens AAA long @100 (ledger records entry)
    # next bar: AAA up 12% -> bracket take-profit exit
    market = {"AAA": market_view("AAA", 112.0)}
    positions = {"AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0)}
    state = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    orders = trader.run(state)
    assert any(o.asset == "AAA" and o.reduce_only for o in orders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.genome.adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/genome/adapter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from src import OrderCommand

from .genes import EntryLedger
from .registry import build_gene


@dataclass(frozen=True)
class GeneSpec:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Genome:
    name: str
    universe_filter: GeneSpec
    signal: GeneSpec
    entry_timing: GeneSpec
    sizing: GeneSpec
    exit_rule: GeneSpec
    n_long: int = 2
    n_short: int = 2
    margin_mode: str = "cross"


def select_longs_shorts(
    scores: Dict[str, float],
    n_long: int,
    n_short: int,
    held: Set[str],
) -> Tuple[List[str], List[str]]:
    cands = sorted(scores.items(), key=lambda t: (t[1], t[0]))
    if len(cands) < 2:
        return [], []
    shorts = [a for a, _ in cands[:n_short]]
    longs = [a for a, _ in cands[-n_long:]]
    long_set = set(longs)
    shorts = [a for a in shorts if a not in long_set]
    longs = [a for a in longs if a not in held]
    shorts = [a for a in shorts if a not in held]
    return longs, shorts


class ComposedTrader:
    """A Trader assembled from five genes; run(state) drives a fixed pipeline."""

    def __init__(self, genome: Genome, uf, sig, timing, sizing, exit_rule) -> None:
        self.name = genome.name
        self.margin_mode = genome.margin_mode
        self._genome = genome
        self._uf = uf
        self._sig = sig
        self._timing = timing
        self._sizing = sizing
        self._exit = exit_rule
        self._ledger = EntryLedger()

    def run(self, state) -> List[OrderCommand]:
        self._ledger.prune(set(state.positions))
        orders: List[OrderCommand] = list(self._exit.exits(state, self._ledger))
        if self._timing.should_enter(state):
            universe = self._uf.eligible(state)
            scores = self._sig.score(state, universe)
            held = set(state.positions)
            longs, shorts = select_longs_shorts(
                scores, self._genome.n_long, self._genome.n_short, held
            )
            entry_orders = self._sizing.orders_for(state, longs, shorts)
            for order in entry_orders:
                mv = state.market.get(order.asset)
                if mv is not None:
                    self._ledger.record(order.asset, state.bar_index, mv.mark_px)
            orders.extend(entry_orders)
        return orders


def build_trader(genome: Genome) -> ComposedTrader:
    uf = build_gene("universe_filter", genome.universe_filter.kind, genome.universe_filter.params)
    sig = build_gene("signal", genome.signal.kind, genome.signal.params)
    timing = build_gene("entry_timing", genome.entry_timing.kind, genome.entry_timing.params)
    sizing = build_gene("sizing", genome.sizing.kind, genome.sizing.params)
    exit_rule = build_gene("exit_rule", genome.exit_rule.kind, genome.exit_rule.params)
    return ComposedTrader(genome, uf, sig, timing, sizing, exit_rule)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_adapter.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/genome/adapter.py tests/test_genome_adapter.py
git commit -m "feat(genome): Genome/GeneSpec + ComposedTrader adapter"
```

---

### Task 6: Public surface + equivalence proof vs legacy SLTP_Bracket

**Files:**
- Modify: `src/genome/__init__.py`
- Test: `tests/test_genome_equivalence.py`

**Interfaces:**
- Consumes: everything above; the legacy `Traders/SLTP_Bracket.py` via `strategy_harness.load`.
- Produces: `src.genome` public surface exporting `Genome`, `GeneSpec`, `build_trader`, `build_gene`, `register`, `EntryLedger`.

**Note on equivalence scope:** Both traders track entries from the same emission events, so they agree across normal replay. The legacy file lazily `setdefault`s an entry bar for positions it never opened; that path is not exercised when entries flow through the pipeline, so the test drives entries through `run()` on both and asserts identical order streams. This is equivalence within the operating regime, stated explicitly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_genome_equivalence.py
from __future__ import annotations

from src.genome import Genome, GeneSpec, build_trader
from strategy_harness import load, make_state, market_view, position_view


def _genome_bracket():
    return Genome(
        name="BracketSLTP",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional", {"notional_long": 10.0, "notional_short": 10.0,
                                           "min_notional_usd": 10.0, "leverage": 1.0}),
        exit_rule=GeneSpec("bracket", {"sl_pct": 0.05, "tp_pct": 0.10, "expiry_bars": 240,
                                       "min_notional_usd": 10.0, "leverage": 1.0}),
        n_long=2,
        n_short=2,
    )


def _key(orders):
    return sorted(
        (o.asset, o.side, o.order_type, o.notional, o.size, o.reduce_only) for o in orders
    )


def _release_state(bar_index=0):
    ranks = {
        "AAA": (0.90, 0.0), "BBB": (0.80, 0.0), "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0), "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_entry_orders_match_legacy():
    legacy = load("SLTP_Bracket.py")
    genome = build_trader(_genome_bracket())
    state = _release_state()
    assert _key(genome.run(state)) == _key(legacy.run(state))


def test_exit_orders_match_legacy_over_sequence():
    legacy = load("SLTP_Bracket.py")
    genome = build_trader(_genome_bracket())

    # bar 0: both open entries at px 100
    s0 = _release_state(bar_index=0)
    assert _key(genome.run(s0)) == _key(legacy.run(s0))

    # bar 1: AAA & BBB (longs) +12% -> tp; CCC & DDD (shorts) -12% price move
    marks = {"AAA": 112.0, "BBB": 112.0, "EEE": 100.0, "CCC": 88.0, "DDD": 88.0}
    market = {a: market_view(a, px) for a, px in marks.items()}
    positions = {
        "AAA": position_view("AAA", size=0.1, entry_price=100.0, mark_price=112.0),
        "BBB": position_view("BBB", size=0.1, entry_price=100.0, mark_price=112.0),
        "CCC": position_view("CCC", size=-0.1, entry_price=100.0, mark_price=88.0),
        "DDD": position_view("DDD", size=-0.1, entry_price=100.0, mark_price=88.0),
    }
    s1 = make_state(bar_index=1, market=market, positions=positions, is_release=False)
    assert _key(genome.run(s1)) == _key(legacy.run(s1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_genome_equivalence.py -v`
Expected: FAIL — `ImportError: cannot import name 'Genome' from 'src.genome'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/genome/__init__.py
"""Compositional-genome layer.

A strategy = {universe_filter, signal, entry_timing, sizing, exit_rule}.
build_trader(genome) returns a standard Trader the existing engine runs unchanged.
"""

from . import library  # noqa: F401  — side effect: registers reference genes
from .adapter import Genome, GeneSpec, ComposedTrader, build_trader, select_longs_shorts
from .genes import EntryLedger
from .registry import SLOTS, build_gene, register

__all__ = [
    "Genome",
    "GeneSpec",
    "ComposedTrader",
    "build_trader",
    "select_longs_shorts",
    "EntryLedger",
    "SLOTS",
    "build_gene",
    "register",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_genome_equivalence.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full genome suite**

Run: `python -m pytest tests/test_genome_ledger.py tests/test_genome_registry.py tests/test_genome_library.py tests/test_genome_exits.py tests/test_genome_adapter.py tests/test_genome_equivalence.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/genome/__init__.py tests/test_genome_equivalence.py
git commit -m "feat(genome): public surface + legacy equivalence proof"
```

---

## Manual Verification (not a committed test — requires `../data/` CSVs)

After Task 6, confirm full-backtest equivalence against real data:

1. In a scratch script, load market data as `tester.py` does.
2. Run `run_backtest(load_trader("Traders/SLTP_Bracket.py", **shared), market, ranks_path, bt_cfg, verbose=False)` and `run_backtest(build_trader(genome_bracket), market, ranks_path, bt_cfg, verbose=False)`.
3. Assert `numpy.allclose(a.total_equity, b.total_equity)`.

If the curves diverge, the divergence is the first real bug to investigate — do not paper over it.

---

## Self-Review

**Spec coverage (Phase 0 of the design doc):**
- "Define gene slots with typed interfaces" → Tasks 1 (Protocols), 2 (registry/SLOTS).
- "Define genome-config format" → Task 5 (`Genome`/`GeneSpec`).
- "Genome → runnable `Trader` adapter, engine runs unchanged" → Task 5 (`ComposedTrader`/`build_trader`) + manual verification.
- "Prove the model by re-expressing the current SLTP family as genomes" → Task 6 equivalence (Bracket, verified in full); SL_Only/TP_Only/Asym expressible as `bracket` params (Task 4 `test_bracket_stop_only_ignores_upside` demonstrates the mechanism); Trailing as a distinct stateful gene (Task 4).

**Placeholder scan:** none — every step carries runnable code and exact commands.

**Type consistency:** `EntryLedger.record/get/prune`, `GeneSpec(kind, params)`, `Genome(...)`, `build_gene(slot, kind, params)`, `build_trader(genome)`, `select_longs_shorts(scores, n_long, n_short, held)`, and gene method names (`eligible`, `score`, `should_enter`, `orders_for`, `exits`) are used identically across tasks.

**Deferred (correctly, per design doc):** the full gene taxonomy (more options per slot) is a separate "universe" brainstorm; Phase 0 ships one reference option per slot plus the exit family — enough to prove the contract end-to-end.
