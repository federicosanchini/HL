# Phase 1 — Batch Evaluation & Scoring Implementation Plan

> Normative: macro-plan rev2 §2 (objective `score_v1`), §5 (fingerprint/idempotency), §6-Phase-1. Engine v2 + taxonomy v1 landed (165 tests). `run_backtest_prepared(strategy, sd, bt_cfg)` already exists (E-T6). New package: `src/evolution/` (evaluation layer — NOT engine; engine invariants untouched; imports engine via public surfaces).

**Goal:** deterministic machinery that turns a genome list into durable, scored, fingerprinted digests — resumable, holdout-disciplined, placebo-baselined. No LLM anywhere.

## Global constraints
- Determinism absolute: seeded `numpy.random.Generator(PCG64(seed))` only; content-hash = sha256 of canonical-JSON genome; no wall-clock in outputs.
- Digests: append-only JSONL, one line per (genome_hash, fingerprint); writer skips if key exists; corruption-safe via temp-write+rename... APPEND-ONLY files can't rename-swap per line — rule: append with `\n`-terminated json lines, reader tolerates+reports a trailing partial line (crash artifact) and ignores it; a repair pass truncates it. Document.
- Fingerprint = {engine_semantics_version, data_window_hash (sha256 of timeline bounds+perp list), scorer_version="s1", gene_library_version="t1"}.
- Holdout: last 25% of bars + 30d (720-bar) embargo before it; in-sample = bars before embargo. Holdout metrics NEVER in the strategist-facing digest record (separate record type `holdout`, joined only by Skeptic/report tooling).
- src/__init__.py untouched. Commit per task.

### Task P1 — Sweep generator [M] — `src/evolution/sweep.py`
`generate_seed_genomes(grid_cfg) -> List[Genome]`: cartesian over kind-tuples × discrete knob grids (n {1,2,3} incl. one-sided (n_long=0|n_short=0 variants), leverage {1,2,3}), respecting: composition constraint (delay timing ⇔ cached/momentum/funding signals only; release_bar ⇔ any), max 3 param-variants per kind-tuple (hard-coded variant tables per gene kind, e.g. bracket {(0.05,0.10),(0.03,inf),(inf,0.10)}, trailing {0.05,0.10}, time_only {240,720}, momentum {72,168}, delay {24,72}), margin_mode fixed "cross" for v1. `genome_hash(genome) -> str` canonical sha256. Deterministic order (sorted by hash). Tests: constraint respected, variant budget enforced, hash stable across runs/process restarts, count sanity (log the number).

### Task P2 — Scorer [M] — `src/evolution/scoring.py`
Pure functions over `SimResult`:
- `gross_notional_series(result) -> ndarray` from `per_perp_position` (sum |signed notional| per bar); `normalized_pnl = (final_equity - initial) / max(peak_gross, initial)` (guard zero-position genomes).
- `bootstrap_pnl_ci(total_equity, block_bars=168, n_boot=500, seed=<derived from genome_hash>) -> (lo, hi)`: circular block bootstrap on per-bar equity diffs, 5th/95th pct of resampled total PnL (normalized by same peak_gross). Seed = int from genome_hash prefix → deterministic per genome, no global RNG.
- `margin_headroom(result, mm_rates) -> float`: min over bars of (equity - maintenance_est)/equity, maintenance_est = Σ |per_perp_position_qty×?|... SimResult lacks mark prices per bar; APPROXIMATE with |per_perp_position| (invested notional) × mm_rate — document as entry-notional approximation, scorer_version-stamped.
- `score_v1(digest) -> float`: lower-CI normalized PnL, −0.5×score penalty if n_liquidated>0 (heavy), −complexity×0.001 (complexity = total free params), headroom penalty when headroom<0.1 (linear to −0.05). Constants documented as s1; changes bump scorer_version.
Tests: hand-built SimResult fixtures; CI determinism (same seed → same interval); liquidation penalty ordering; zero-trade genome scores 0 not NaN.

### Task P3 — Digest-maker + store [M] — `src/evolution/digest.py`
`make_digest(result, genome, fingerprint) -> dict`: normalized PnL + CI, Sortino, max-DD (from metrics_total), margin headroom, n_liq, trade count (n_opened+n_closed), PnL concentration (share of |realized| from top-5 events), long/short attribution, per-month PnL signs, when-earned (equity quartile timestamps), benchmark fields left null (joined later), genome spec + hash + kind-tuple + fingerprint.
`DigestStore(path)`: append(digest) skip-if-(hash,fingerprint)-exists; load() → dict keyed (hash,fp); partial-trailing-line tolerance + `repair()`. Tests: roundtrip, skip-if-exists, crash-artifact tolerance (write garbage tail).

### Task P4 — Windows, placebo, EWP [M]
- `src/evolution/windows.py`: `split_windows(sd) -> (insample_sd, holdout_sd)` slicing SimData arrays by bar index (75% − 720-bar embargo | last 25%); tests on synthetic timelines (boundaries exact, embargo length, no overlap).
- Placebo: `signal`/`placebo_random(seed=0)` gene in library — score = deterministic hash(asset, bar_index, seed)→[0,1); observe-free, stateless, works on release_bar timing. Registered kind; documented as baseline-only (excluded from evolution ops later).
- EWP baseline: run existing `Traders/EWP.py` through the same digest pipeline (trader-not-genome digest record type `baseline`).
### Task P5 — Batch CLI + real-data run [S-M] — `GioTester/evaluate.py`
`python evaluate.py --out results/digests_v1.jsonl [--limit N] [--holdout]`: loads data once, `prepare_sim_data`, splits windows, iterates genomes from sweep (+3 placebo seeds + EWP), skip-if-exists, per-genome try/except (failure recorded as digest {status:"error", reason} — loud, not hidden; engine RuntimeError re-raised only for cash-non-finite invariant which is recorded then re-raised), tqdm, prints wall-time/genome. Smoke test with --limit 5 on real data (skip-if-missing). NOT the full sweep run — that's Phase 2's seeded-ledger step per macro-plan.

Phase-final: full suite, 3-lens plan-vs-implementation review (statistical lens on P2 mandatory), ledger.
