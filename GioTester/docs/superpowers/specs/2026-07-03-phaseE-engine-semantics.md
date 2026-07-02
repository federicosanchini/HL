# Phase E — Engine Execution Semantics v2 (spec, rev. 2)

**Status:** rev. 2 after 3-lens adversarial review (2026-07-03; 4 criticals + 9 importants absorbed, no G1 decision revisited).
**G1 (user-locked):** (a) decide-close/fill-next-open, (b) SL-wins same-bar, (c) liquidation-wins, (d) limits dropped.
**This is a deliberate, user-approved matching-behavior change.** `ENGINE_SEMANTICS_VERSION` bumps 1 → 2.

## v1 defects being fixed (measured)

- `trade_px = open[i]`, `mark_px = close[i]`; strategy sees close[i] and fills at open[i] **of the same bar** — a look-ahead an evolution loop will exploit.
- Exits trigger on close-based returns and fill at open; no intrabar fidelity.

## v2 normative rules

### R1. Market orders: decide at close[i], fill at open[i+1]
- Strategy runs at bar i end seeing completed bar i (as today). Market orders are queued:
  `StateBucket.pending_orders: List[PendingOrder]`, `PendingOrder = {order: OrderCommand, queued_ts: pd.Timestamp, queued_bar: int}`.
- **Fill pass (start of bar i+1):** consume FIFO in strategy-emission order — matching v1 `execute_orders` semantics; this list order is an **explicit exception** to the asset-sorted rule (it is itself deterministic).
- **Price pinning (anti-look-ahead):** before the pending-fill pass, all open positions are re-marked at `open[i+1]` (guaranteed priceable by the gap-scan rule below), and the mark price supplied to ALL margin/hypothetical-position logic during this pass is `open[i+1]` — **never** `close[i+1]`. Concretely: the execute path used for pending fills takes `mark_px = trade_px` as a parameter instead of reading `ms.mark_px()`. The later `mark_positions(close[i+1])` re-marks everything.
- All margin/reduce-only/leverage/min-notional checks run at fill time under those open[i+1] prices. Rejection events carry `queued_ts`.
- **Unpriceable fill bar:** rejected during bar i's gap scan (the same place `_force_close_positions_before_price_gap` already knows open[i+1] is missing), reason `"pending order unpriceable at fill bar"`, timestamp = `queued_ts`. Single rejection site; no fill-bar rejection path for this case.
- **Last bar:** pending orders are NOT filled; each is rejected with reason `"backtest end"` (timestamp `queued_ts`) before `close_all`. No literal `break`: last bar = close_all → skip strategy/funding/gap-scan → fall through to the equity snapshot exactly as the current runner does.
- Engine-internal closes remain immediate (last-bar `close_all`, force-close-before-gap, liquidation) — they are not strategy decisions.

### R2. Trigger orders (new)
- `OrderCommand` gains `trigger_px: Optional[float]` and `trigger_direction: Optional[str]` ∈ {`"stop"`, `"tp"`}. `OrderType.TRIGGER` exists but is **NOT** added to `SUPPORTED_ORDER_TYPES` (that set remains the market-fill whitelist; `execute_order` keeps rejecting TRIGGER defensively). Strategy output is partitioned by `normalized_type()`: TRIGGER → placement path; everything else → R1 queue.
- v2 restriction: `reduce_only=True`, explicit `size > 0`, finite `trigger_px > 0`, `trigger_direction` required, side must oppose the current position at placement. Open-via-trigger and limits: out of scope (G1d).
- Resting store: `StateBucket.resting_triggers: Dict[str, List[RestingTrigger]]`, `RestingTrigger = {order, placed_bar}`.
- **Replace rule:** replacement fires iff the bar's strategy output contains ≥1 trigger order for asset A, valid or not: A's resting set is cleared, then each new trigger is validated individually — invalid ones produce `OrderRejectedEvent`s (loud), and an all-invalid set leaves A with no resting triggers. Market orders for A do not touch A's triggers.
- **Auto-cancel:** position reaches size 0 or flips sign (any path: fill, liquidation, force-close) → all resting triggers for that asset cancelled.
- **Activation:** a trigger placed at bar i is active from bar i+1 (bar i is complete when the strategy sees it).
- **One-bar protection gap (accepted v2 behavior, documented):** an entry queued at bar i fills at open[i+1]; the gene first sees the position at bar i+1 and places triggers then; they are active from bar i+2. Every new position is exposed to bar i+1's full range. Test required asserting a bar-(i+1) breach does NOT fill.

### R3. Trigger evaluation & fill (bar i, high[i]/low[i], direction-dependent)
Evaluated per asset in `sorted(asset)` order, triggers in placement order. Fire/fill matrix:

| side, direction | fires iff | gap-through (fill at open[i]) iff | else fill at |
|---|---|---|---|
| sell (close long), stop | `low[i] <= px` | `open[i] <= px` | `px` |
| sell (close long), tp | `high[i] >= px` | `open[i] >= px` | `px` |
| buy (close short), stop | `high[i] >= px` | `open[i] >= px` | `px` |
| buy (close short), tp | `low[i] <= px` | `open[i] <= px` | `px` |

- Gap-through is always worse-for-trader by construction.
- **Size clamp:** executed size = `min(trigger.size, |position.size|)` at fire time (position may have shrunk since placement); `|position| <= EPS` is already handled by auto-cancel. The clamped size feeds the fill event and R4 comparison. Trigger fills bypass `_assert_reduce_only`'s flip rejection (the clamp guarantees legality).
- **min_notional_usd does NOT apply to trigger fills** (reduce-only closes, like `close_position`). It applies at **placement**: a trigger with `abs(size) * close[i] < min_notional_usd` is rejected at placement (loud).
- Fees: taker. Events: `ExecutionEvent(event_type="position_reduced"/"position_closed", reason=CloseReason.TRIGGER.value)`; `CloseReason.TRIGGER = "trigger"` is added.
- **NaN high/low fallback (uniform):** if `high[i]` (resp. `low[i]`) is non-finite, substitute `max(open[i], close_fallback[i])` (resp. `min(...)`), where `close_fallback` uses the existing mark_px open-fallback. Applies to R3 firing, R3 gap-through, R5 marking, and R7's `high_px`/`low_px`.

### R4. Same-bar multi-trigger rule (G1b, role-free)
If ≥2 resting triggers on the SAME asset fire on the same bar: execute only the one with the **worst fill price for the trader** (long-close: lowest; short-close: highest); cancel the others for that asset that bar. For a bracket this is exactly SL-wins. Tie key, exactly: `(fill_price worst-first, client_id is None, client_id or "", placement_index)` — real client_ids sort before None; None-vs-None falls to placement order.

### R5. Liquidation precedence (G1c, conservative, no free option)
Before trigger fills at bar i, per position in sorted-asset order: conservative breach scan — portfolio equity with THIS position marked at its adverse extreme (long: low[i]; short: high[i], with the R3 NaN fallback) and all others at open[i]; maintenance at the same marks. Breach → the asset's triggers are **suppressed** for the R3 pass.
**Deferred execution (anti-exploit):** if asset A was suppressed at bar i and the close-mark liquidation pass does NOT liquidate A that bar, A's suppressed-but-fired triggers execute immediately after that liquidation pass, same bar, at their normal R3/R4 fill price. Liquidation strictly wins whenever it fires; a recovered bar no longer forgives the stop. Isolated mode: same rule on isolated equity. Documented approximation: single-position-extreme marking; exact multi-asset intrabar paths are unknowable from OHLC.

### R6. Bar pipeline v2 (runner.py)
```
for i in bars:
    ms = market_state(i)
    if last_bar:
        reject_pending("backtest end"); fill_triggers?  -> NO: triggers evaluate first? see below
    # normal bars:
    remark_positions(open[i]); fill_pending_market_orders(open[i])   # R1, FIFO, mark=open
    suppress = liquidation_precedence_scan(high/low[i])              # R5
    fill_triggers(high[i], low[i], suppress)                         # R3 + R4
    mark_positions(close[i])
    liquidate_if_needed()                                            # unchanged close-mark pass
    execute_deferred_suppressed_triggers()                           # R5 deferred clause
    if last_bar: close_all(FORCE); snapshot; continue-to-end
    orders = strategy.run(state_i)
    partition: queue markets (R1) / place-replace triggers (R2)
    apply_funding(); liquidate_if_needed()
    gap_scan: force-close positions + reject unpriceable pendings    # R1
    snapshot equity
```
Last-bar ordering, normative: reject pendings ("backtest end") → evaluate resting triggers on bar N-1's range (R3-R5) → `close_all(FORCE)` → snapshot. No strategy call, no funding, no gap scan — mirroring today's last-bar structure.
Funding runs AFTER trigger fills (assert in tests). No step consumes a price later than its information set.

### R7. Strategy surface & gene contract v2
- `MarketAssetView` gains `high_px: float`, `low_px: float` (completed bar i; NaN fallback per R3). `dto.py` + `state.asset_view` + harness `market_view` fixture (kwargs defaulting to `mark_px`).
- **ExitRule v2 contract:** `exits(state, ledger) -> List[OrderCommand]`; genes emit their FULL desired trigger set every bar a position exists (R2 replace makes re-emission idempotent); expiry remains a market order and may appear in the same list; min-notional placement gate per R3.
- **Level anchoring:** genes derive levels from `PositionView.entry_price` (the realized open[i+1] fill). The adapter refreshes each `EntryLedger` record's price from `state.positions[asset].entry_price` on first observation (bar_index kept for expiry); ledger records whose emission never became a position are pruned as today.
- BracketExit(long): sell-stop at `entry*(1-sl_pct)` + sell-tp at `entry*(1+tp_pct)` (mirrored short); `tp_pct=inf` → SL-only emitted. TrailingExit: peak ratchets from `high_px` (long) / `low_px` (short), monotonic; re-emits stop at `peak*(1∓trail_pct)` each bar.

### R8. Versioning, equivalence, and test migration
- `ENGINE_SEMANTICS_VERSION = 2` exported from `src`, stamped into `SimResult`.
- **Equivalence, corrected:** (1) ENTRY equivalence genome-vs-legacy holds under v2 unchanged (decision logic identical). (2) EXIT equivalence: new v2-native comparator — a test-local hand-written trigger-emitting trader — proving the genome trigger stream identical (`_key` extended with `trigger_px`, `trigger_direction`). (3) `test_exit_orders_match_legacy_over_sequence` is REPLACED by an intentional-**divergence canary**: same sequence, assert genome emits triggers, legacy emits market exits, streams differ by design (cited in the phase report; legacy Traders keep running under v2 with expected metric shifts).
- **Test inventory:** UNCHANGED — `test_sltp_strategies.py`, `test_genome_ledger/registry/library/selection_guard.py`, entry-side adapter/equivalence tests. MECHANICAL — `strategy_harness.market_view` high/low kwargs. REWRITTEN — `test_genome_exits.py` (trigger-placer contract: exact trigger sets, tp=inf → SL-only, re-emission, expiry market order, placement min-notional, trailing ratchet tested with `high_px != mark_px`), `test_run_exits_tracked_entry_on_tp` (assert exact SL+TP trigger set; companion: entry decision bar emits NO triggers). NEW — `tests/engine_harness.py`: minimal synthetic `SimData` (3–6 bars, 1–2 assets, hand-set OHLC/funding/oracle) + scripted trader through `run_backtest`, covering: R1 queue→next-open fill (fees/margin at open, queued_ts in events), unpriceable-pending rejection, backtest-end rejection, R3 all four fire/fill cases + gap-through + NaN fallback, R4 worst-fill + tie key, R5 suppression + deferred execution, R2 replace/auto-cancel/activation-delay + one-bar gap test, last-bar ordering, funding-after-triggers, determinism ×2 byte-identical, version stamp == 2.
- Schema v5: `reason="trigger"` additive on ExecutionEvent — verify GioVisualizer `parseResult.ts` tolerates unknown reasons; else reuse `"stoploss"`.

## Out of scope
Limits, open-via-trigger, maker fees, sub-bar paths beyond high/low, intrabar funding, exact cross-margin intrabar liquidation.

## Task decomposition
1. DTO/fixture: high_px/low_px + NaN fallback helper. 2. OrderCommand trigger fields + CloseReason.TRIGGER + validation + partition rule. 3. StateBucket pending/resting stores, replace/auto-cancel, placement gate. 4. Trigger fire/fill pure functions (R3 matrix + clamp + R4) + unit tests. 5. R5 scan + deferred execution + tests. 6. Runner pipeline v2 (R1 pinning, last-bar path) + engine_harness + integration cases. 7. Genes → trigger-placers + ledger price-refresh + rewritten exit tests + equivalence trio. 8. Version stamp, determinism ×2, full suite, real-data smoke (skip-if-missing).
