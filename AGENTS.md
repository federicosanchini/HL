# AGENTS.md — HYPERLIQUID VAULT

## Purpose

Build a deterministic HyperLiquid trading-engine simulator and research framework.

Users write standalone Python strategy files:

- final strategy class: `Trader`
- main strategy method: `run()`
- engine handles replay, matching, synchronization, bookkeeping, execution, and orchestration

Architecture is inspired by IMC-style trading challenge simulators.

---

## Non-Negotiable Invariants

Never break:

- exchange semantics
- matching behavior
- deterministic replay
- timestamp consistency
- shared timestamp synchronization
- existing strategy compatibility, unless explicitly approved

Never:

- silently alter trading logic
- introduce speculative exchange behavior
- weaken correctness for optimization
- hide failures with fallback logic
- change simulator semantics during refactors
- change public strategy interfaces without explicit approval

Prefer:

- explicit behavior
- deterministic behavior
- local reasoning
- maintainability
- debuggability
- simple architecture

Avoid:

- speculative abstractions
- enterprise patterns
- unnecessary indirection
- framework-heavy solutions
- broad rewrites when a local fix is enough

---

## Environment

```txt
OS       : Windows 11
CPU      : Intel Core i7, 4c/8t
RAM      : 15.6 GB
GPU      : GTX 1650 Max-Q, 4 GB VRAM
Python   : use `python`
```

Defaults:

- use CPU + NumPy unless GPU is clearly justified
- assert CUDA availability before CUDA use
- respect 4 GB VRAM limit
- batch GPU workloads
- profile before optimizing

---

## Priorities

In order:

1. Correctness
2. Determinism
3. Simplicity
4. Readability
5. Performance

Guidelines:

- prefer `O(n log n)` over `O(n²)`
- use vectorization when readable
- use numerically stable formulations
- type annotate public functions
- keep nesting shallow
- names should encode semantics and units
- avoid hidden global state
- avoid premature generalization
- prefer boring, inspectable code

---

## Simulator Rules

The simulator is correctness-sensitive.

High-risk areas:

- matching
- replay
- timestamps
- fills
- order lifecycle
- liquidation
- funding
- margin
- position/equity bookkeeping
- synchronization
- serialization

When touching these areas:

- inspect existing behavior before editing
- preserve deterministic ordering
- preserve timestamp alignment
- preserve fill/cancellation semantics
- prefer explicit errors over silent correction
- do not add queue, latency, impact, priority, or real-exchange assumptions unless requested
- do not optimize by weakening correctness

Only encode behavior that is already part of this simulator’s intended semantics or explicitly requested.

---

## Strategy Interface

Strategies are user-facing.

Assume:

- standalone Python files
- class named `Trader`
- main method named `run()`
- users should not need simulator internals

When changing strategy-facing code:

- preserve backward compatibility by default
- keep APIs minimal and stable
- do not leak engine internals into strategy code
- document intentional interface changes

---

## Data, I/O, and Schemas

Never:

- overwrite raw data
- write raw API responses to disk
- store secrets or API keys
- modify `.gitignore`
- silently change serialized schemas
- silently reinterpret output fields

When changing exported formats:

- preserve old fields where possible
- add new fields explicitly
- update readers/writers together
- handle optional fields explicitly
- document schema changes

---

## Research Code

Research code may be exploratory, but must not contaminate simulator semantics.

Rules:

- separate exploratory assumptions from engine logic
- control random seeds where relevant
- prefer reproducible scripts over notebook-only logic
- save processed outputs separately from raw data
- do not overwrite source data
- state when a result is diagnostic rather than production logic

---

## Visualizer Rules

The visualizer must display simulator output faithfully.

Rules:

- preserve expected JSON schema unless explicitly approved
- do not silently rename fields
- do not reinterpret simulator metrics in the frontend
- handle missing optional fields gracefully
- keep calculations explicit
- prefer simple components over highly abstract chart frameworks

---

## Refactoring Rules

A refactor must preserve behavior.

When refactoring:

- keep changes local
- avoid broad rewrites
- preserve public APIs
- preserve simulator semantics
- preserve serialized schemas
- avoid mixing refactor and feature work
- avoid mixing replication and extension phases

If behavior changes, call it a behavior change, not a refactor.

---

## Terminal Rules

request user approval before running:

- scripts
- installs
- git commands
- shell commands

When terminal use is requested:

- use the smallest command that answers the question
- ask before mutating files
- do not run broad test suites when a focused test is enough
- ask before installing packages
- never use git commands unless requested

---

## Testing and Validation

When editing code:

- run tests only if explicitly allowed
- prefer focused tests over broad suites
- if tests are not run, state what should be run
- use exact expected values for deterministic behavior
- control seeds for stochastic/research code
- do not weaken tests to make code pass

Review changes for:

- determinism
- timestamp consistency
- matching semantics
- strategy compatibility
- numerical stability
- memory/performance regressions
- serialization compatibility

---

## Execution Style

Prioritize:

- implementation
- code inspection
- direct fixes
- concrete diffs

Minimize:

- narration
- long reports
- repeated observations
- architectural essays
- verbose summaries
- generic advice
- speculative future work

Explain only:

- major decisions
- non-obvious tradeoffs
- invariant-sensitive changes
- user-visible behavior changes

Spend tokens on inspecting and editing code, not reports.

---

## Confidence Protocol

Before non-trivial implementations or bug fixes, provide:

```txt
Confidence : HIGH | MEDIUM | LOW
Uncertainty: <missing information>
Resolves if: <what removes uncertainty>
```

If LOW confidence:

- do not implement immediately
- ask for approval or the smallest blocking clarification

If MEDIUM confidence:

- proceed only when the change is local and reversible
- state the assumption

If HIGH confidence:

- proceed directly

---

## Change Summary Format

For non-trivial coding tasks, use:

```txt
Confidence : HIGH | MEDIUM | LOW
Uncertainty: ...
Resolves if: ...

Changed:
- path/to/file.py::function_name
  - what changed
  - why

Validation:
- what was checked
- what still needs to be run
```

Keep summaries short unless a detailed report is requested.

---

## Claude-Specific Files

Do not use `.claude/` commands, Claude skills, or Claude-only workflows as part of the Codex workflow.

They may exist in the repo, but are not authoritative for Codex unless explicitly requested.
