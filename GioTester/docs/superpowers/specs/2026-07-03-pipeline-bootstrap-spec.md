# PIPELINE BOOTSTRAP SPEC — Strategy-Evolution Build

**Purpose:** self-contained instructions for a fresh Claude Code session (zero prior context) to continue building and eventually operating the strategy-evolution pipeline. The user re-triggers one session per usage window; this file + the ledger are the only memory.

**User invocation (paste this each session):**
```
Read GioTester/docs/superpowers/specs/2026-07-03-pipeline-bootstrap-spec.md and continue the pipeline. ultracode
```

---

## 1. Ground truth — read in this order, act on nothing until done

1. `GioTester/docs/superpowers/specs/2026-07-02-strategy-evolution-daemon-design.md` — the macro-plan (rev. 2). Authoritative for WHAT to build.
2. `.superpowers/sdd/progress.md` (repo root) — the build ledger. Authoritative for WHERE we are. Trust it + `git log` over any recollection.
3. `git log --oneline -15` on branch `giorgio` — verify the ledger's claims exist in git.
4. `GioTester/docs/superpowers/plans/` — existing sub-plans. A phase with a plan file but no ledger completion lines is mid-execution: resume at the first task not marked complete. NEVER re-dispatch a task the ledger marks complete.

## 2. Standing orders (verbatim authority from the user, 2026-07-03)

> "i want to defer to you the orchestration of the plan creation and implementation pipeline: … handle plan creation, supervision, approval, execution, checks, cleaning of obsolete. then you can deploy the agents to start research."

Meaning: the orchestrator (you) authors each phase's sub-plan, self-reviews it, executes it via subagent-driven development, reviews results, updates the ledger, and proceeds — **without** pausing for per-task user approval. Exceptions — HARD user gates, stop and ask:
- **G1. Phase E semantic decisions** (fill convention, same-bar SL/TP rule, liquidation precedence, limit build-or-drop): present the options from the macro-plan §6-E with a recommendation; user picks. These change matching behavior — the one invariant that must never change silently.
- **G2. Phase T taxonomy contents**: the user wants to brainstorm the gene universe. Present a proposed v1 list; user edits/approves.
- **G3. Brain backend** (API-key SDK = pay-per-token, vs subscription-backed sessions): billing decision, user's money, user picks. Ask before Phase 3 build starts.
- **G4. Deploying live research** (first real seed sweep + brain cycles): confirm once before the first spend.
- Anything requiring destructive git (revert/reset/force) or touching files outside GioTester/{src,tests,Traders,docs,evolve.py area} → ask.

Git: commits per completed task are authorized (established pattern: `feat(genome): …` style, staged files only, never `git add -A`, never touch the user's unrelated dirty tree or top-level `src/__init__.py`).

## 3. Build sequence (from macro-plan rev. 2 — do not reorder)

| # | Phase | Gate | Size |
|---|---|---|---|
| 1 | 0h — adapter hardening (n_long=0 guard + one-sided tests) | none | XS |
| 2 | E — engine semantics (look-ahead fix, intrabar trigger fills, precedence, engine-version stamp) | **G1 before implementation** | M |
| 3 | T — taxonomy v1 (percent_of_equity, leverage knob hoist, 2-3 options/slot) | **G2 for gene list** | S-M |
| 4 | 1 — batch eval (real-data seam gate first, prepared-SimData API, score_v1, digest-maker) | none | M |
| 5 | 2 — ledger (population memory + pipeline state machine) + seed sweep at end | none | M |
| 6 | 3 — brain roles (write-ahead discipline) | **G3 before build** | M-L |
| 7 | 4 — evolve.py crank + RESUME runbook | none | M |
| 8 | research ops — cycles of the crank | **G4 once** | ongoing |

Per-phase loop: **(a)** write sub-plan to `docs/superpowers/plans/YYYY-MM-DD-phaseX-*.md` (bite-sized TDD tasks, complete code in steps, exact commands — follow the Phase 0 plan's format, it worked); **(b)** adversarial plan review — one Workflow with 3 lenses (spec-coverage, correctness-of-included-code, sequencing), fix findings; **(c)** execute via fresh implementer subagent per task + reviewer per task (sonnet implementers when the plan contains the full code; sonnet reviewers; escalate to opus only for invariant-touching diffs — Phase E); **(d)** final whole-branch review (opus) per phase; **(e)** ledger lines per task + phase-complete line; **(f)** clean obsolete: delete superseded plan drafts, stale scratch files, tasks list entries.

## 4. Session protocol (5h windows)

- **Start:** read ground truth (§1). One line to the user: current phase, resume point, intended work this session.
- **Work units are tasks, not phases.** Commit + ledger line after every task. A session dying mid-task costs that task only.
- **~30 min before window feels tight** (or when a long task finishes late): stop dispatching, write a `SESSION HANDOFF:` line to the ledger (phase, next task number, any decision pending), tell the user what to expect next session. Do not start a task you cannot finish.
- **End:** final message = what was completed, what is next, which gate (if any) the user must answer next session — so they can answer it in the same paste that resumes work.

## 5. Model / effort / ultracode policy (this answers the user's standing question)

- **Session model: Fable, effort medium.** Orchestration, synthesis, and plan-writing at medium is the right default; raise your own effort only while authoring the Phase E spec (invariant-sensitive).
- **Implementers: sonnet** (plans carry complete code — transcription+testing). **Task reviewers: sonnet. Final phase reviews: opus.**
- **Ultracode: yes, but only at gates** — plan reviews (3 lenses, ~150k tokens) and phase-final reviews. Do NOT run workflows for implementation or routine checks; the Phase 0 experience showed per-task subagents suffice. The macro-plan-review pattern (5 lenses, ~280k) is reserved for spec-level milestones: after Phase E lands and before research ops begin.
- Budget sanity: a phase ≈ 400-700k subagent tokens at this policy. If a session's remaining budget clearly cannot finish the current task, hand off instead of starting it.

## 6. Research-ops protocol (after Phase 4, gate G4 passed)

Each session: `python evolve.py resume` → read its output → drain pending brain requests if the backend is subscription-mode (G3) → run cycles until the wall-clock rule (§4) fires → `python evolve.py report` → relay the report's headline (new champions, disqualifications, frontier status) to the user in plain language. The ledger + evolve.py own all state; the session is a disposable operator.

## 7. Current state snapshot (update this section every session — it is the compaction anchor)

- **2026-07-03 (i):** Phase 0 DONE (`4e5020ba..ef547f53`, 29/29, opus READY). Macro-plan rev. 2 after 5-lens review. Phase 0h DONE (`9acb1d24`). Full permissions incl. git granted; caveman ultra active.
- **2026-07-03 (ii):** G1 answered (decide-close/fill-next-open; SL-wins; liquidation-wins; limits dropped). Phase E spec rev2 (3-lens reviewed) + **Phase E COMPLETE**: 8 tasks, `c1ece10d..f68d9b22`, suite 140/140 ×2, real-data smoke ran, final opus review READY. `ENGINE_SEMANTICS_VERSION=2`. Minors deferred: M1 synthetic re-emit fill test (add early Phase T), M2 GioVisualizer reason="trigger" tolerance unverified. NOTE: task commits progressively folded in the user's pre-existing uncommitted engine rework (state/position/execution/runner/result.py) — intentional preservation. Next: **G2 gate** (taxonomy v1), then Phase T. Gates G3-G4 open.
