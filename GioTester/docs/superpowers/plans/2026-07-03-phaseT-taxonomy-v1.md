# Phase T — Gene Taxonomy v1 Implementation Plan

> G2 approved 2026-07-03 (full proposed list). Normative context: macro-plan rev2 §6-T; engine v2 spec R7 (gene contract). Deviation (orchestrator call, documented): per-asset *pullback* does not fit the global `should_enter() -> bool` timing contract (and rank rows are None off-release) — v1 ships `delay` timing (enter k bars after release); per-asset pullback deferred to a future `entry_filter` slot (Gene-Author scope). Everything else as approved.

**Goal:** grow the gene library to ~48 kind-tuples so Phase 1's sweep has a population; hoist cross-cutting knobs to genome level; close Phase E Minor M1.

## Global constraints
- Genes import only from `src` public API; stateful genes allowed (Trailing precedent); determinism (no set iteration, seeded nothing).
- Distinctness bookkeeping: new signal kinds registered as DISTINCT kinds (`rank_30d` separate from `rank`) — mechanism accounting per macro-plan §2.
- All existing tests stay green; full suite after each task; commit per task, staged files only.
- `src/__init__.py` untouched.

### Task T1 — M1 closure + knob hoist [S]
- New runner-level test (in `tests/test_engine_v2_pipeline.py` or new file): re-emitting bracket genome (`build_trader`) through `run_backtest_prepared` on synthetic bars where bar i+2 low breaches SL → assert a `reason="trigger"` fill occurs (locks fill-before-re-emission ordering against regression).
- `Genome` gains `leverage: float = 1.0`, `min_notional_usd: float = 10.0`; `build_trader` injects both into every slot gene's params (dict-merge: genome-level value OVERRIDES per-gene param if the gene's __init__ accepts it — inject only keys present in the gene class signature via inspect, or simpler: genes' __init__ all accept **_ absorbing extras; verify and inject unconditionally). Existing genome constructions in tests get identical behavior (defaults match current values). Tests: leverage=2 genome → orders carry leverage 2 from sizing AND exit genes; min_notional propagates.

### Task T2 — Signal + timing genes [M]
- `signal`/`rank_30d`: registered kind, RankSignal with field_index locked to 1.
- `signal`/`momentum(lookback_bars=72)`: stateful; per asset, append mark_px each bar (deque maxlen); score = px/px[-lookback] - 1 when window full, asset skipped otherwise. Scores computed only when called (release bars under release_bar timing) but accumulation must happen EVERY bar → genes only see state when adapter calls them... PROBLEM: signal.score() is only called on should_enter bars. FIX (contract-preserving): momentum accumulates inside score() calls AND ComposedTrader calls a new optional gene hook `observe(state)` every bar for all genes that define it (adapter: `for g in genes: if hasattr(g,'observe'): g.observe(state)` at top of run()). Deterministic, additive, no protocol break (hook optional). Document in genes.py.
- `signal`/`funding_carry`: score = -funding_rate per asset (short expensive-funding, long negative-funding); stateless.
- `entry_timing`/`delay(bars_after_release=24)`: stateful; observes release bars (via observe hook or should_enter logic), returns True exactly `bars_after_release` bars after the most recent release bar. PROBLEM as above: rank signals return {} off-release. PAIRED REQUIREMENT: `signal`/`rank_cached` + `rank_30d_cached` kinds — RankSignal variant caching the last seen ranks row in observe(); delay timing only composes with *_cached or non-rank signals; document composition constraint in registry docstring (Phase 1 sweep generator must respect it — note for Phase 1 plan).
- Tests per gene: momentum window math incl. insufficient-window skip; funding sign; delay fires exactly once k bars post-release; cached rank survives off-release bars; observe hook called every bar (adapter test).

### Task T3 — Sizing + exit genes [S]
- `sizing`/`percent_of_equity(pct=0.5%, min_notional_usd, leverage)`: notional per position = pct × state.account.equity; skip below min_notional; longs-then-shorts order preserved.
- `exit_rule`/`time_only(expiry_bars)`: market reduce-only exit at expiry; no triggers ever.
- Tests: pct sizing math vs equity; below-min skip; time_only emits nothing pre-expiry, market exit at expiry, no TRIGGER orders ever.
- Update taxonomy table in macro-plan §6-T status + ledger.

Phase-final: full suite, phase review (sonnet sufficient — additive genes only, no engine surface), ledger.
