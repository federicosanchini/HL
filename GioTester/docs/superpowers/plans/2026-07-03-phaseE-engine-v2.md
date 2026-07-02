# Phase E — Engine v2 Implementation Plan

> **For agentic workers:** execute task-by-task via fresh implementer subagents. The NORMATIVE source is `docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md` (rev. 2) — every rule cited as R1..R8 below refers to it. Where this plan and the spec disagree, the spec wins. Steps use checkbox syntax.

**Goal:** implement engine semantics v2: decide-close/fill-next-open market orders, intrabar trigger fills, liquidation precedence, version stamp — per the reviewed spec.

**Architecture:** pending-order queue + resting-trigger store on `StateBucket`; pure trigger fire/fill functions in a new `src/triggers.py`; runner pipeline reorder; genome exit genes become trigger-placers. All engine invariants except matching (deliberately versioned to 2) preserved.

**Tech stack:** Python 3.12, pytest, existing src/ modules only. No new deps.

## Global Constraints
- Normative spec: `docs/superpowers/specs/2026-07-03-phaseE-engine-semantics.md` rev. 2. Implementers MUST read it before coding their task.
- TDD per task: failing tests first (from the spec's enumerated cases), then implementation, then full suite green.
- `src/__init__.py` MAY be modified in Task 8 ONLY (export `ENGINE_SEMANTICS_VERSION`) — one added export, nothing removed.
- Determinism: asset-sorted iteration for triggers/liquidation; FIFO emission order for pending fills (explicit R1 exception); tie key per R4 exactly.
- Commit per task, staged files only, never `git add -A`. Test cmd: `python -m pytest` from `GioTester/`.
- Existing-test policy per R8 inventory: UNCHANGED files must stay green untouched; MECHANICAL = harness fixture only; REWRITTEN = test_genome_exits.py + adapter TP test + exit-equivalence trio.

---

### Task 1: DTO surface + NaN-fallback helper [S]
**Files:** modify `src/dto.py` (MarketAssetView + high_px/low_px), `src/state.py` (asset_view + `bar_range(asset) -> (eff_high, eff_low)` helper implementing R3's NaN fallback), `tests/strategy_harness.py` (market_view kwargs `high_px=None, low_px=None` defaulting to mark). Test: `tests/test_engine_v2_dto.py`.
**Produces:** `MarketAssetView.high_px/.low_px`; `MarketState.bar_range(asset)`.
**Tests:** view carries high/low; NaN high → eff_high = max(open, close_fb); NaN low → eff_low = min(open, close_fb); NaN close too → open used; existing suite green (fixtures default high=low=mark).

### Task 2: OrderCommand trigger fields + partition rule [S]
**Files:** modify `src/position.py` (add `trigger_px`, `trigger_direction`; `CloseReason.TRIGGER = "trigger"`; `validate_trigger()` per R2: reduce_only, explicit size>0, finite trigger_px>0, direction in {stop,tp}; SUPPORTED_ORDER_TYPES UNCHANGED — market-only). Test: `tests/test_engine_v2_orders.py`.
**Tests:** valid trigger passes; each violated constraint raises/returns reason; TRIGGER routed to execute_order still rejected (defensive); market orders unaffected.

### Task 3: StateBucket stores + placement path [M]
**Files:** modify `src/state.py` (`PendingOrder`, `RestingTrigger` dataclasses; `pending_orders: List[PendingOrder]`; `resting_triggers: Dict[str, List[RestingTrigger]]`), new `src/triggers.py` with `place_triggers(orders, ms, bucket, min_notional_usd) -> None` implementing R2 (replace-iff-any-trigger-for-asset, individual validation w/ loud rejects incl. side-opposes-position + placement min-notional at `abs(size)*close[i]`, placed_bar recorded) and `auto_cancel(bucket, asset)`. Test: `tests/test_engine_v2_placement.py`.
**Tests:** replace clears old set; all-invalid set leaves none + rejection events; market order for A leaves A's triggers; placement gate; side validation; auto-cancel on flat/flip.

### Task 4: Trigger fire/fill core (R3 matrix + clamp + R4) [M]
**Files:** extend `src/triggers.py`: `fired(trigger, open_, eff_high, eff_low) -> Optional[fill_px]` implementing the 4-row fire/fill matrix + gap-through; `select_and_fill(bucket, ms, fee_bps, suppress: Set[str])` — asset-sorted, R4 worst-fill selection with exact tie key, size clamp `min(size, |pos|)`, fills via `close_position`-style partial close (reduce path bypassing `_assert_reduce_only`), events `reason="trigger"`, activation check `placed_bar < i`. Test: `tests/test_engine_v2_triggers.py`.
**Tests:** all 4 matrix rows fire/fill correctly; gap-through fills at open each direction; not-fired cases; activation delay (placed_bar == i doesn't fire); clamp after shrink; R4 SL-wins bracket case; tie key incl. None client_id; suppressed asset skipped; NaN fallback path.

### Task 5: Liquidation precedence scan + deferred execution [M]
**Files:** extend `src/triggers.py`: `precedence_scan(bucket, ms) -> Set[str]` (R5 single-position-extreme marking, cross + isolated); runner hook for deferred execution after close-mark liquidation (R5 deferred clause). Test: `tests/test_engine_v2_precedence.py`.
**Tests:** breach at extreme suppresses; close-liquidation fires → trigger never executes; close recovers → suppressed-fired trigger executes after liq pass at R3 price; isolated variant; no-breach → no suppression.

### Task 6: Runner pipeline v2 [L — most capable model]
**Files:** modify `src/runner.py` per R6 exactly (remark at open → FIFO pending fills w/ `mark_px = open` pinning per R1 → scan → trigger fills → mark close → liquidate → deferred triggers → strategy/partition/queue/place → funding → liquidate → gap-scan w/ unpriceable-pending rejects → snapshot; last-bar: reject pendings "backtest end" → triggers on range → close_all → snapshot, no strategy/funding/gap). Modify `src/execution.py`: `execute_order` takes optional `mark_px_override` (pending path passes open). New `tests/engine_harness.py` (synthetic SimData builder: 3–6 bars, 1–2 assets, hand-set OHLC/funding/oracle + scripted trader) + `tests/test_engine_v2_pipeline.py`.
**Tests (R8 NEW list):** queue→next-open fill w/ fees+margin at open + queued_ts in events; margin accept/reject independent of close[i+1] (two datasets differing only in close ⇒ identical accept/reject); unpriceable-pending reject at gap scan; backtest-end reject; one-bar protection gap (bar i+1 breach doesn't fill); funding-after-triggers ordering; last-bar ordering; determinism ×2 byte-identical SimResult.

### Task 7: Genes → trigger-placers + equivalence trio [M]
**Files:** modify `src/genome/library.py` (BracketExit/TrailingExit per R7 contract: full set every bar, entry_price anchoring, tp=inf → SL-only, trailing ratchet from high_px/low_px, expiry market order), `src/genome/adapter.py` (ledger price-refresh from `state.positions[..].entry_price` on first observation). REWRITE `tests/test_genome_exits.py` + `test_run_exits_tracked_entry_on_tp` (+ companion no-triggers-on-decision-bar) per R8. REWRITE exit-equivalence: v2-native trigger comparator (test-local) + divergence canary replacing `test_exit_orders_match_legacy_over_sequence`. Entry-side tests untouched.

### Task 8: Version stamp + gates [S]
**Files:** `src/__init__.py` (+`ENGINE_SEMANTICS_VERSION = 2`), `src/result.py` (SimResult field + serialization; check GioVisualizer `parseResult.ts` tolerance for `reason:"trigger"` — if intolerant, use "stoploss" and note), determinism gate, real-data smoke test `tests/test_real_data_equivalence.py` (legacy vs genome through run_backtest on `../data/`, allclose on ENTRY-only genome variant; skip-if-data-missing). Full suite ×2. Phase-final opus review + ledger.

## Self-review
Spec coverage: R1→T3/T6, R2→T2/T3, R3→T1/T4, R4→T4, R5→T5, R6→T6, R7→T7, R8→T8 + inventories embedded. No placeholders — code bodies are normatively defined in the spec; tasks name exact functions/files/cases. Types consistent: PendingOrder/RestingTrigger defined T3, consumed T4–T6; `bar_range` T1 consumed T4/T5.
