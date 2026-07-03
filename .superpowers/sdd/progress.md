# Phase 0 SDD Progress
Base: 281c1017 (branch giorgio)
Task 1: complete (commits 281c1017..4e5020ba, review clean)
Task 2: complete (commits 4e5020ba..b41b7523, review clean)
  Minor(final-review): global _REGISTRY shared across test session (by design)
Task 3: complete (commits b41b7523..01012d18, review clean)
Task 4: complete (commits 01012d18..1efd866f, review clean)
  Minor(final-review): TrailingExit peak carry-over on same-bar direction flip (not reachable via ledger)
Task 5: complete (commits 1efd866f..c0855f09, review clean)
  Minor(final-review): select_longs_shorts overlap-drop branch untested
Task 6: complete (commits c0855f09..ef547f53, review clean, full suite 29/29)
FINAL whole-branch review: READY TO MERGE (opus). No Critical/Important. 3 logged Minors confirmed non-blocking.
Phase 0h: complete (fix + 6 regression tests, suite 51/51, commit 9acb1d24)
G1 ANSWERED (2026-07-03): a) decide-close/fill-next-open b) SL-wins same-bar c) liquidation-wins d) limits dropped from taxonomy
Phase E: spec rev2 written+reviewed (3-lens, 4 crit+9 imp absorbed), committed 4761e32c. Next: author Phase E task plan (8 tasks per spec decomposition), then subagent execution.
PhaseE Task1: complete (c1ece10d, dto high/low + bar_range, 58/58, review clean)
PhaseE Task2: complete (9a129b7f, trigger fields + CloseReason.TRIGGER, 77/77). NOTE: commits c1ece10d+9a129b7f fold in USER PRE-EXISTING uncommitted rework of state.py/position.py (dirty tree since session start) - intentional preservation, not implementer scope creep. Remaining user-dirty src files (config.py, data_loader.py, __init__.py...) untouched.
PhaseE Task3: complete (b815c75c, stores + placement, 93/93, review clean)
PhaseE Task4: complete (7c1fa8b3, R3/R4 core, 114/114, opus review clean). Minor deferred: use CloseReason.TRIGGER.value not literal in triggers.py:205 (sweep in T8).
PhaseE Task5: complete (5e8997d3, R5 scan + deferred exec, 123/123, review clean)
PhaseE Task6: complete (ca0d7c2e, runner v2 + run_backtest_prepared + engine harness, 132/132, opus review clean; position.py-uncommitted claims by 2 agents verified FALSE)
PhaseE Task7: complete (6971a37b, trigger-placer genes + v2 equivalence trio, 138/138, review clean)
PhaseE Task8: complete (f68d9b22, version stamp + gates, 140/140 x2, real-data smoke RAN 44s).
Phase E: COMPLETE - final opus review READY. Minors logged: M1 synthetic re-emit trigger-fill test (add in Phase T), M2 GioVisualizer parseResult tolerance for reason=trigger unverified (dir deleted), M3 last-bar suppression inert (accepted).
Next: G2 gate (taxonomy v1 list) then Phase T.
G2 ANSWERED: taxonomy v1 approved as proposed. Phase T plan committed. Deviation: pullback->delay timing (slot contract), pullback deferred to entry_filter slot.
PhaseT Task1: complete (26dbe39a, knob hoist + M1 integration test, 144/144, review clean)
PhaseT Task2: complete (42930c6b, 5 signal kinds + delay timing + observe hook, 158/158, review clean; nit: momentum set-iteration -> fix in T3)
PhaseT Task3: complete (617cd165, pct sizing + time_only + hygiene, 165/165).
Phase T: COMPLETE - phase review READY. 14 kinds: uf{all_tradable} sig{rank,rank_30d,rank_cached,rank_30d_cached,momentum,funding_carry} tim{release_bar,delay} siz{fixed_notional,percent_of_equity} exit{bracket,trailing,time_only}. Next: Phase 1 (batch eval+scoring, no gate).
SESSION HANDOFF (2026-07-03 ~08:00): Phase 1 plan committed; 3-lens review FAILED on usage limit (0/3 done, nothing cached). Retry cron e2cb6ee2 @ 10:52 Rome: re-run review workflow, absorb findings, execute P1-P5. Phases 0/0h/E/T all COMPLETE, suite 165/165.
Phase 1 plan REV2 committed (absorbed 3-lens: 3 crit + 8 imp). Tasks P0-P5. Key: additive liq penalty, per-window data hash + record_type key, hashlib placebo, exact headroom via per_perp_maintenance, null-sentinel inf legs, status-aware skip, repair-before-append, holdout warm-up, real-data seam gate (legacy allclose replaced - v2 divergence). Executing P0-P5 now.
PhaseE Task... P0: complete (05a20dd6, real-data seam gate 3/3 on real data, 164 fast green)
Phase1 P0: complete (05a20dd6, real-data seam gate 3/3, review clean).
Phase1 P1: complete (22958ef7, sweep 3570 genomes + canonical hash + per-bar maintenance, 195 fast/199 all green, review clean). Minor: min_notional_usd not in genome_hash payload (inert - sweep does not vary it).
Phase1 P2: complete (4cc2b954, score_v1 additive-liq + CRN bootstrap + exact headroom, 215/219 green, opus review clean - sign bug empirically guarded)
