# Design — Five Stop-Loss / Take-Profit Trader Strategies

Date: 2026-06-15
Status: Approved (design), pending implementation plan
Location: `GioTester/Traders/`

## Purpose

Add five standalone strategy files to `GioTester/Traders/`, each implementing a
deterministic stop-loss / take-profit exit discipline on top of a shared,
HODL-style rank-based entry. The five differ **only** in their exit rule so they
form a clean A/B comparison against `HODL10` (same entry, same signal horizon).

No new engine code. Each file is a standalone `Trader` class consumed by
`tester.py`, importing only from `src`.

## Signal background

`state.current_ranks_row[asset]` is a `(pred_10d, pred_30d)` tuple of CrowdCent
predictions of an asset's relative over/underperformance versus the tradeable
universe, released at midnight UTC (`state.is_release_bar == True`). Higher value
⇒ expected to outperform the universe; lower ⇒ underperform. All five strategies
use index 0 (`pred_10d`), matching `HODL10`.

## Shared skeleton (identical across all five)

- **Entry trigger**: only on `is_release_bar` with a non-empty `current_ranks_row`.
- **Ranking**: collect `(asset, pred_10d)` for assets present in `state.market`
  with finite signal; sort. Long the top `n`, short the bottom `n`
  (`n = 3` default, kwarg). Remove overlap so a name is never both long and short
  (long set wins, mirroring `HODL10`).
- **No stacking**: skip any candidate asset already held in `state.positions`.
- **Sizing**: fixed `notional_long` / `notional_short` USD per slot, injected as
  kwargs (same source as `HODL10`). A side is skipped if its notional is below
  `min_notional_usd`.
- **Signed return** (side-symmetric, the core exit primitive):
  `ret = side * (mark_px / entry_price - 1)` where `side = +1` long, `-1` short.
  `ret > 0` is favorable for both longs and shorts; `ret < 0` is adverse. This
  removes all short-side special-casing.
- **Exit evaluation**: every bar, for each held position, compute `ret` from
  `state.market[asset].mark_px` and the strategy-tracked `entry_price`. On a
  breach, emit a `reduce_only` market `OrderCommand` with `size = abs(pos.size)`.
- **Time-expiry backstop**: force-close any position still open at
  `expiry_days = 10` (`expiry_bars = 10 * bars_per_day`), reusing the `HODL10`
  expiry mechanic. Guarantees no position outlives its 10-day signal horizon.
- **Fire-and-forget**: a name dropping out of the top/bottom-`n` does **not**
  close the position. Only SL, TP, or expiry close it. Lowest turnover; isolates
  the exit logic under test.
- **Per-asset strategy state**: track `entry_bar` and `entry_price` per held
  asset in a dict; prune entries when the asset is no longer in
  `state.positions`. Strategy #5 additionally tracks `peak_ret`.

### Order of operations in `run(state)`

1. Build exit orders for currently-held positions (SL/TP/trail/expiry checks).
2. Prune internal tracking dicts against `state.positions`.
3. If `is_release_bar` and ranks present: rank, select, skip-if-held, append
   entry orders and record `entry_bar` / `entry_price` for new names.
4. Return the combined order list.

Entry records use the bar's mark/trade price as `entry_price` reference. Because
engine fills happen within the same bar at `trade_px`, the strategy seeds
`entry_price` from `state.market[asset].mark_px` at entry; the exit math is a
relative move from that seed and remains deterministic.

## The five strategies

All share the skeleton above. They differ only in the exit rule. All thresholds
are `__init__` kwargs (tunable); each `__init__` absorbs unknown shared kwargs
via `**_`.

| # | File | `name` | Exit rule | Defaults |
|---|------|--------|-----------|----------|
| 1 | `SLTP_Bracket.py` | `BracketSLTP` | both SL and TP | SL −5%, TP +10% |
| 2 | `SL_Only.py` | `StopOnly` | hard SL; winners ride to expiry | SL −5% |
| 3 | `TP_Only.py` | `TargetOnly` | TP only; losers ride to expiry | TP +10% |
| 4 | `SLTP_Asym.py` | `AsymRR` | tight SL, wide TP (1:3) | SL −4%, TP +12% |
| 5 | `Trailing.py` | `TrailStop` | trailing stop, ratchets up | trail 5% |

### Exit math

- **Stop-loss**: close when `ret <= -sl_pct`.
- **Take-profit**: close when `ret >= tp_pct`.
- **Trailing** (#5): maintain `peak_ret = max(peak_ret, ret)` per position,
  initialized at entry to `0.0`. Close when `ret <= peak_ret - trail_pct`. With
  `peak_ret` starting at 0 the initial stop sits at `-trail_pct` (acts as the
  first stop), then ratchets only upward as the trade moves favorably. Standard
  trailing-stop behavior.

`sl_pct`, `tp_pct`, `trail_pct` are fractions (e.g. `0.05` = 5%).

## Interface conformance

Each file defines exactly one class `Trader` with:

```python
class Trader:
    name = "<as table>"
    margin_mode = "cross"   # default; kwarg-overridable like HODL10

    def __init__(self, *, n, leverage, notional_long, notional_short,
                 min_notional_usd, blackout_days_end, bars_per_day,
                 # strategy-specific: sl_pct / tp_pct / trail_pct as applicable
                 **_): ...

    def run(self, state) -> list[OrderCommand]: ...
```

Imports: `from src import OrderCommand, OrderType` only.

## Simplifications (explicit, per project transparency rules)

- **Mark-based, once per hour**: SL/TP/trail are evaluated on the per-bar
  `mark_px` the strategy can see — not intrabar OHLC high/low (not exposed to
  strategies). Deterministic; slightly optimistic when a bar gaps through a
  level. Documented, accepted.
- **Same-bar fill**: reduce-only exit orders emitted in `run()` are filled within
  the same bar at `trade_px` by the engine loop.
- **Entry price seed**: `entry_price` is seeded from `mark_px` at the entry bar,
  not the realized fill price. The two coincide under the engine's market-fill
  model closely enough for relative-move exits; any divergence is bounded by the
  bar's trade/mark spread and is deterministic.
- **`CloseReason`**: reduce-only closes are attributed by the engine; the
  `CloseReason.STOPLOSS` enum value already exists and is the natural label.

## Out of scope (deliberately not built)

- ATR / volatility-scaled stops.
- Daily rotation / re-ranking exits.
- Multiple signal horizons (30d) or variable breadth across strategies.
- Limit / trigger order types (engine accepts market only).

## Invariants preserved

- Deterministic replay: all exit logic is a pure function of `state` plus
  strategy-local tracking seeded deterministically.
- `StrategyState` immutability: strategies read frozen views only; no engine
  state mutation.
- Public API: imports limited to `src`.
- No engine changes ⇒ JSON schema v5 and matching behavior untouched.

## Validation plan

- `python test_load.py` — data loads clean (unchanged).
- `python tester.py` — discovers the five new files plus existing, all complete,
  `results/<name>.json` written for each.
- Sanity per strategy: inspect that `StopOnly` never closes on the upside,
  `TargetOnly` never closes on the downside, `BracketSLTP` closes on either,
  `AsymRR` shows wider winning exits than losing exits, `TrailStop` shows exits
  at a give-back from peak rather than fixed levels.
