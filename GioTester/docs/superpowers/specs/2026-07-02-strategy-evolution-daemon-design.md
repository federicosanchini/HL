# Strategy-Evolution Research Pipeline — Design (Macro-Plan, rev. 2)

**Date:** 2026-07-02 (rev. 2: 2026-07-03, after 5-lens adversarial review)
**Status:** Revised. Phase 0 implemented (commits `4e5020ba..ef547f53`, review-clean). Awaiting user approval of rev. 2, then decomposition proceeds per the bootstrap spec.
**Scope:** A closed-loop, session-resumable research system that generates, backtests, scores, interprets, and evolves HyperLiquid trading strategies on top of the deterministic GioTester engine.

**Rev. 2 changes (from adversarial review, 5 parallel lenses, 2026-07-03):** replaced the "overnight daemon" model with a resumable crank; inserted Phases 0h/E/T before the seed sweep; promoted resumability into the ledger's defining requirement; hardened the objective function (normalized PnL, CI-banded ranking, holdout, placebo baselines); made anti-fine-tuning mechanically enforced; made skeptics diagnostics-driven.

---

## 1. Problem & Goal

GioTester is a deterministic single-shot evaluator with no research loop around it. Goal: the closed loop — idea generation → implementation → testing → interpretation → reformulation — running with minimal supervision, producing a steady flow of explored, distinct ideas.

**Operational reality (binding):** Pro-plan sessions die every ~5 hours and resume only when the user re-triggers. There is no long-lived daemon. The pipeline is a **resumable crank**: every invocation loads all state from a durable ledger, performs bounded work, persists atomically, and can die at any moment losing at most one in-flight work unit.

## 2. Objective Function (what "good" means)

- **PnL is the primary discriminator — but normalized, not raw USD.** Score = PnL / peak gross notional deployed over the run (return on capital actually at risk). Raw USD PnL is gameable by cranking notional/leverage; normalized PnL is not.
- **Statistical tie handling:** each genome's equity curve gets a block-bootstrap PnL confidence interval (weekly blocks, fixed RNG seed — deterministic). Genomes rank by **lower CI bound**. Overlapping CIs = statistical tie → fall through to **Sortino, then max-DD** as secondary discriminators. This operationalizes "significant (>1% monthly) PnL differences are the real signal": differences within the CI are noise and must not drive selection.
- **Liquidations = heavy penalty.** Additionally, digests carry **margin headroom** (min distance to maintenance margin over the path); near-liquidation paths are penalized continuously — "no liquidation fired" is not a pass for strategies that skirted it.
- **Holdout discipline:** the data window is split — evolve/score on the earlier ~75%, hold out the final ~25%, with an embargo gap ≥ max holding period (30 days) between folds. **Holdout metrics never enter the Strategist's view.** They are computed by the driver and visible only to the Skeptic panel and the report. In-sample normalized PnL drives evolution; the holdout makes champion claims falsifiable.
- **Null baseline:** every evaluation batch includes placebo genomes (identical slot structure, seeded random-signal gene) and the EWP benchmark. "Top tier" means: exceeds the placebo population's 95th percentile AND beats EWP on the same window. Without this, top-of-N selection bias makes "top tier" unfalsifiable.
- **Anti-overfitting is mechanically enforced, not prompted:** two genomes are the *same mechanism* iff they share the gene-kind tuple across all five slots. Hard budget: **max 3 param-variants per mechanism** — the driver rejects over-budget proposals before they run. Breadth = new kind-tuples; the "unexplored gene combos" ledger query is the sanctioned frontier.
- **Complexity penalty defined:** complexity = total free parameters across the genome's GeneSpecs. Genes are capped on param count at admission; a Gene-Author gene is admitted only if it improves lower-CI score on ≥2 different genome backbones.
- The single composite score is defined once, here, and referenced by name (`score_v1`) everywhere (scorer, ledger queries, convergence). No phase restates it.

## 3. Core Design Decision — Compositional Genome

```
genome = { universe_filter, signal, entry_timing, sizing, exit_rule }   # 5 slot genes
       + { n_long, n_short, margin_mode, leverage }                      # genome-level discrete knobs
```

- Each gene = small typed Python component (implemented: `src/genome/`).
- **Cross-cutting params (leverage, min_notional_usd) are hoisted to genome level** so crossover cannot produce desynchronized entry/exit legs. (Phase 0 shipped them inside genes; Phase T hoists them.)
- **Three operators:** *gene mutation* = swap one slot gene; *crossover* = take slot genes from two parents; *knob mutation* = step one genome-level knob along a fixed discrete grid (grids, not continuous search — consistent with anti-fine-tuning).
- Escape hatch: the Gene-Author writes new genes directly into the library (no free-form Trader intermediate — the pipeline stores genomes only).

## 4. Where Intelligence Is Needed — three roles, fixed firing points

| Role | Fires when | Job |
|---|---|---|
| **Strategist** | after each evaluation batch | Read ledger digests → propose next batch (mutations, crossovers, knob steps, prunes). Distinctness budget enforced by the driver, not by the prompt. |
| **Gene-Author** | population saturates existing kind-tuples | Write a new gene + unit test conforming to the slot contract; admission rule per §2. |
| **Skeptic panel** | a genome reaches top tier (per §2 null-baseline definition) | **Diagnostics-driven, not opinion-driven.** The driver computes deterministically: holdout performance, PnL-after-dropping-top-K-trades, param-perturbation spread (±20% jitter — robustness check, not tuning), sub-period consistency. Each skeptic receives a different diagnostic slice and votes on thresholds; the LLM interprets borderline cases and writes disqualification rationales. Independence comes from independent evidence. |

**Brain-call discipline (write-ahead):** before any LLM call, persist a request record `{role, cycle_id, input-hash}`; persist the raw structured response before anything consumes it. On resume, an existing response for `(role, cycle_id)` skips the call. Exactly-once decisions, full audit trail, deterministic resume through the only stochastic component. Roles sit behind a file-based request/response boundary so the backend (SDK API-key vs a subscription-backed Claude Code session draining pending requests) is swappable — **backend decision is made before Phase 3 build starts**, since it sets the budget model.

## 5. Execution Model — the resumable crank

One entry point: `python evolve.py {step|resume|report|status}`.

- `step`/`resume`: load ALL state from ledger → perform one bounded work unit (minutes, not hours) → persist transition atomically → exit 0.
- `report`: pure function of the ledger, generatable at any moment (there is no "end of run" to report at).
- Stop/pause policy: max-cycles, token budget, convergence (**no lower-CI score_v1 gain of the top tier on the validation fold in K cycles**), **wall-clock check before each unit** (insufficient window remaining → persist PAUSED, exit 0), **rate-limit → PAUSED_RATE_LIMIT with resume-after hint** (never a failure). Engine-invariant violations remain the only hard-fail.
- Crash cost ≤ one in-flight work unit. Backtests are deterministic → evaluation is idempotent: every digest is keyed by `(genome content-hash, compatibility fingerprint)` and all evaluation is skip-if-digest-exists.
- **Compatibility fingerprint** = {engine-semantics version, data-window hash, scorer version, gene-library version}. Stamped on every digest; the scorer refuses to rank across fingerprints. A semantics change invalidates stale digests loudly, never silently.
- A RESUME runbook ships with Phase 4: fresh zero-context session needs only "run `python evolve.py resume` and read its output"; the command itself validates preconditions (ledger readable, fingerprint matches engine commit + data hashes) and refuses loudly otherwise.

## 6. Phases (rev. 2)

Dependency-ordered. Effort: XS/S/M/L.

### Phase 0 — Genome & contracts ✅ DONE
`src/genome/` shipped and review-clean (commits `4e5020ba..ef547f53`; equivalence proof vs legacy SLTP_Bracket; 29/29 tests).

### Phase 0h — Adapter hardening [XS] — **gates everything downstream**
Fix `select_longs_shorts`: `n_long=0` → `cands[-0:]` selects the whole list (long-only/short-only genomes broken); guard both sides (`[] if n <= 0`); revisit the `len(cands) < 2` early-return that wrongly blocks legitimate one-sided single-candidate genomes. Regression tests: (0, k), (k, 0), (1 candidate, one-sided), plus the untested overlap-drop branch.

### Phase E — Engine execution semantics [M] — **the invariant-sensitive one; own spec; gates the seed sweep**
This is a deliberate, spec'd matching-behavior change with explicit approval — not a silent alteration. Scope:
1. **Look-ahead audit & fix.** Today the strategy decides on `mark_px = close[i]` but fills at `trade_px` of the same bar — an exploitable channel a PnL-maximizing loop WILL find. Pick one consistent convention (decide-on-close → fill at next bar open is the standard), re-validate the Phase 0 equivalence proof under it.
2. **Intrabar trigger fills.** Add `trigger_px` to OrderCommand; admit TRIGGER order type; resting-order store on StateBucket; expose bar `high_px`/`low_px` on MarketAssetView; fill pass checks high/low, fills **at trigger price**, gap-through fills at open (worse-for-trader). Deterministic same-bar SL-vs-TP ambiguity rule (worst-case-first) and asset-sorted fill order — documented. First cut: reduce-only triggers (no margin check at trigger time). Exit genes become trigger-placers instead of close-based market emitters.
3. **Liquidation-vs-stop precedence**, explicit: if the bar's adverse extreme breaches liquidation price, liquidation wins and the stop is suppressed (conservative), or the bar-granularity approximation is documented as known-optimistic. Written down, either way.
4. **Limit orders: decide build-or-drop here.** If dropped, limit-based genes are removed from the taxonomy; if built, OHLC-touch fills, same machinery.
5. **Regression gate:** all existing market-only strategies produce byte-identical results, else the engine-semantics version bumps. Version stamp lands in every digest (see §5).

### Phase T — Gene taxonomy v1 [S-M]
The current combinatorial product is 2 genomes — there is nothing to sweep. Ship: `percent_of_equity` sizing (engine-independent — `AccountView.equity` already exposed; sequence its digests after Phase E's look-ahead fix); genome-level discrete `leverage` knob {1,2,3} + hoist `min_notional_usd`; 2–3 options per remaining slot (from the deferred "universe" brainstorm with the user — this phase has a user-input gate); trigger/limit-based genes register only if engine version ≥ Phase E. Gene-library version bumps the fingerprint.

### Phase 1 — Batch evaluation & scoring [M] *(no LLM)*
- **Task 1 = the real-data seam gate:** pytest running legacy SLTP_Bracket and its genome through `run_backtest` on real `../data/`, asserting `allclose` on equity curves (skip-if-data-missing). This was Phase 0's "manual verification" — now owned and mandatory. Re-run as Phase E's acceptance gate too.
- **Prepared-SimData batch API:** `run_backtest_prepared(strategy, sim_data, bt_cfg)` — pure refactor; `prepare_sim_data` currently runs per call, which a sweep cannot afford. Persistent worker prep on 4c/8t Windows (spawn semantics); measure per-backtest wall time before fixing sweep size.
- **Scorer** implementing `score_v1` (§2) + **digest-maker**: normalized PnL + CI, Sortino, max-DD, margin headroom, n_liq, trade count, PnL concentration (top-5 trade share), long/short attribution, per-month PnL signs, benchmark-relative vs EWP, *when* PnL was earned. Digest-and-discard — no per-genome v5 JSON by default (tens of MB each).
- Digests written durably on completion, append-only JSONL keyed `(genome_hash, fingerprint)`, skip-if-exists — the sweep itself is resumable and this record format IS the ledger seed (Phase 2 builds on it unchanged).
- The **seed sweep executes after Phase 2's ledger exists** (it's one-time and expensive — it lands directly in durable storage); Phase 1 validates the machinery on a handful of genomes.

### Phase 2 — Ledger: population memory + durable pipeline state [M]
Two roles, one store:
1. **Population memory:** genes, genomes, digests, lineage (parent→child + operator), lessons. Strategist queries: top-N by `score_v1`, unexplored kind-tuples, genes correlated with liquidation, per-mechanism variant counts (budget enforcement).
2. **Pipeline state machine:** cycle_id, step, per-work-item status (`proposed → queued → evaluating → scored → judged`, with failure reasons), pending brain requests/responses (§4 write-ahead), budget counters (tokens per role, backtests run), compatibility fingerprint. Resume queries: "what is incomplete in cycle N", "is this ledger compatible with the current fingerprint".
- Corruption-safe: SQLite (WAL) or append-only JSONL with temp-write+rename. A crash mid-write must never leave the ledger unreadable.
- Substrate decision (SQLite vs JSONL vs graphify) made in this phase's sub-plan. **The seed sweep runs at the end of this phase**, into the ledger, with placebo + EWP baselines included.

### Phase 3 — Brain roles [M-L]
Strategist / Gene-Author / Skeptic-panel per §4, behind the file-based request/response boundary, with write-ahead persistence and per-call token/cost recorded in the ledger. **Precondition: backend decision (API-key SDK vs subscription-backed sessions) — different billing, budget model, and failure modes.**

### Phase 4 — The crank [M]
`evolve.py` per §5: state machine over the ledger, bounded work units, pause states (wall-clock, rate-limit), budget/convergence stops, `report` subcommand (ledger-derived, any time), RESUME runbook in `docs/`. Anti-degenerate-loop guardrails. Crash-isolation per genome; engine-invariant violations hard-fail.

### Phase 5 — Surfacing [S, deferred]
How champions surface into GioVisualizer. (The launch-mechanism question formerly here moved into Phase 3's precondition; the session-chunked execution model is now §5, not deferred.)

## 7. Build-Order Rationale

0h before anything (one-sided genomes poison any sweep). E before T's trigger genes and before ANY digest that matters (fill-semantics change invalidates every prior digest — never spend sweep budget on a fill model known to change). T before 1 (nothing to sweep otherwise). 1's record format seeds 2; the expensive one-time sweep waits for 2's durable ledger. 3 needs 2's state machine + backend decision. 4 composes 1+2+3. 5 last.

## 8. Invariants

- Phase 0 preserved all engine invariants (verified: equivalence proof + final review).
- **Phase E deliberately changes matching behavior** under its own spec, with explicit user approval, regression gates for market-only paths, and an engine-semantics version bump. This is the sanctioned path for invariant-sensitive change — the "no silent logic changes" rule holds precisely because the change is loud, versioned, and gated.
- Determinism everywhere: seeded bootstrap, deterministic intrabar rules, asset-sorted iteration, content-hash keys.
- `StrategyState` immutability, margin semantics separation, schema v5 (new event types in Phase E need an explicit schema decision), public-API discipline (`src/__init__.py` untouched; genome surface via `src.genome`).

## 9. Open Items

- Gene taxonomy v1 contents — user brainstorm gate inside Phase T.
- Ledger substrate — Phase 2 sub-plan.
- Brain backend (API-key vs subscription sessions) — decided before Phase 3.
- Phase E same-bar ambiguity + liquidation precedence — decided inside Phase E's spec (options pre-written there).
- Whether CSVs contain history before the 2025-10-10 config default — check in Phase 1; extend the window before splitting if so.
