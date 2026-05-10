# Claude Code — Operating Contract

You are working on a research codebase. The following rules are **mandatory**. If a rule cannot be satisfied, halt and ask the user. Do not improvise. Do not run experiments "to see what happens."

A task is not complete until every applicable section of this document has been satisfied and a report has been written.

---

## Operating principles

These hold above any individual section. If a specific rule below appears to permit a shortcut, it does not.

- **Be systematic, not lazy.** When a procedure exists in this document, follow it. Shortcuts accumulate into broken experiments. If a rule feels inconvenient, that is a sign the rule is doing work — not a sign to skip it.
- **Write the plans down.** Plans live on disk (Section 2), not in the working memory of a single chat turn. An unwritten plan has no continuity, no review surface, and no audit trail.
- **No improvisation under pressure.** "I'll just try it and see" is the failure mode this document exists to prevent.

---

## 0. Workflow (top-level)

For every task, in this order:

1. **Read `CORE.YAML` and `tools.yaml`** at the repository root.
2. **Plan** the change (Section 2) before touching code.
3. **Implement** the change outside of `CORE.YAML`-protected files (Section 1).
4. **Write tests** at all required layers (Section 3).
5. **Run tests** and **measure** memory + latency (Sections 3, 4).
6. **Write the report** (Section 9).
7. Only then, return control to the user with a summary.

If any step fails, stop and report the failure. Do not proceed past a failing step.

---

## 1. Core Framework Protection

The repository contains a `CORE.YAML` file at the root. It enumerates files, modules, crates, and packages that constitute the **core framework**. These are read-only by default.

- If a task can be solved **without** modifying anything in `CORE.YAML`: proceed.
- If a task **requires** modifying a `CORE.YAML` item: **STOP**. Produce a written justification and a migration plan, and wait for explicit user approval. Do not edit.
- Adding, removing, or upgrading a dependency (Cargo, pip, apt, npm, system package) is treated as a core change. Same protocol.
- If `CORE.YAML` is missing, malformed, or unreadable: halt and ask. Do not assume a default.

This rule overrides convenience. A workaround in non-core code is always preferred over a "small" core edit.

---

## 2. Plan Before You Act

For any non-trivial change (anything beyond a single-file local fix or a typo), produce a **plan document** before writing code. Commit it to:

```
docs/plans/<YYYY-MM-DD>-<slug>/
```

The plan must be produced in **all four** formats:

- `plan.tex`   — LaTeX source, compilable standalone (`pdflatex` or `lualatex`).
- `plan.pdf`   — built from `plan.tex`.
- `plan.tikz`  — TikZ figure(s) for architecture / dataflow / module boundaries.
- `plan.mmd`   — Mermaid diagram(s) for sequence / state / dependency views.

The plan must state, at minimum:

- **Scope** and goal.
- **Affected files** (full list).
- **CORE.YAML items touched** — must be the empty list, or escalate per Section 1.
- **Interface changes** (signatures, types, contracts).
- **Test strategy** (which tests at which layer; see Section 3).
- **Performance budget** (peak RSS, wall time, inference latency).
- **Rollback path**.
- **Risk anticipation** — what could go wrong at production scale that won't show up in unit tests? Specifically:
  - **Performance-contract preservation** — does this change preserve every existing contract on the touched code path? Caps (e.g. `max_cycles`, batch_size), memory budgets, time bounds, and sample-size limits are contracts. Adding a new branch beside an existing one is the highest-risk case: the new branch must inherit all the contracts the old one enforced. Grep the file you are about to edit for aspirational comments ("should never materialise", "respect cap X", "TODO: push the cap into …"). Those are flags for latent contract gaps — if the new branch doesn't honor the comment, write the test that proves it does, or escalate the comment into a real precondition check.
  - **What does the worst-case input look like?** State the production-scale dataset, cap, and config the plan is intended to run under, not just the unit-test fixture.
- **Empty-plan-dir hygiene** — if a plan dir is created but the work is abandoned, delete the dir before the next session. Empty plan dirs are noise that look like in-progress work.

**No implementation begins until the plan exists on disk and compiles.** This is non-negotiable. Back-dating a plan (writing it alongside or after the implementation) defeats the purpose — the plan exists to surface risk *before* code is touched. A plan→report gap shorter than the time it would take to honestly implement the change is a back-dating red flag.

---

## 3. Testing Protocol

Every change ships with tests, and tests are executed and pass before reporting success. A passing implementation without a passing test suite is a protocol violation.

### Required layers

- **Unit tests** — per function / per module, pure, deterministic, fast (target < 1 s each). Cover normal cases, boundary cases, and at least one failure case per public function.
- **Integration tests** — exercise module boundaries on realistic inputs. End-to-end paths that match real usage.
- **Performance tests** — measure:
  - **Peak resident memory (RSS)** — must be under the budget declared in the plan and never exceed the global cap (Section 4).
  - **Inference / runtime latency** — wall time and, where relevant, throughput (samples/s, tokens/s).
  - Each performance test asserts a numerical budget. A test that only prints numbers is not a performance test.

### Rules

- New public code without tests is rejected by you, not approved.
- Tests must run in CI-equivalent isolation (clean working dir, no hidden global state).
- Flaky tests are bugs, not noise. Mark them, do not retry-until-green.
- **Production-scale smoke before queuing a long run.** A new code path or env-var branch must be exercised at **production scale** (real dataset, real cap, real wall-clock, real RSS) for at least 1 seed / 1 arity *before* it is queued in a multi-seed or overnight run. Unit tests at toy scale do **not** substitute — they will not surface memory bloat, missing-cap regressions, or per-stage wall costs that only appear at the real input size. If the smoke can't be afforded in ≤ 10 % of the queued run's wall budget, write a single-stage smoke (one arity, one dataset, no training) that costs less than that.
- **In-flight experiment claims must cite a verifiable disk artifact.** Before writing "an experiment is in flight / running / queued" in a report or memory, the claim must reference one of: a log file path that exists and is growing; an output jsonl path (zero bytes is acceptable — the file existing proves the script started); a PID/jobspec captured at launch; a `systemd-run --user` unit name; or an orchestrator `/tmp/<slug>/` dir. An ID-only string (e.g. a hash, a job slug) without a corresponding path is **unverified** and must be flagged as such, not asserted as in-progress.
- **Run new modules before queuing.** Before any overnight/multi-hour run touches a new module, run that module's unit tests in the same environment as the queued run. Untested rewired call paths turn latent bugs into 90-min OOMs (see `signedkan_wip/tests/test_cycle_cache.py` for the precedent).

### Coverage rule (new and modified code)

- Every new function, method, or struct (**public or private**) must be exercised by at least one test added in the same change. A private helper may share a test with its public caller, but a test that drives the new path must exist.
- Every modification to the **observable behavior** of an existing function requires a new **regression test** — one that would have failed against the prior implementation. "It still passes the old tests" is not sufficient.
- Indirect coverage via an unchanged integration test does **not** satisfy this rule. The integration test must be new, extended, or have its assertions strengthened.
- Pure renames, formatting, and comment-only changes are exempt — but must be declared as such in the report (Section 9).

### Determinism and reproducibility

- Every test and experiment fixes its random seed explicitly. No reliance on system entropy.
- Tests must be order-independent. If `pytest -p no:randomly` or `cargo test` parallelism breaks them, the tests are wrong, not the runner.
- Floating-point determinism: where parity with prior runs matters (RTL fixtures, published benchmarks), pin BLAS thread count, math mode (`MKL_CBWR`, `CUBLAS_WORKSPACE_CONFIG`), and library versions; document in the test or the report.
- Test inputs are either generated deterministically from a seed, or committed as fixtures with content hash.

### Benchmark stability

- Performance tests run a minimum of **5 iterations after warm-up** and report **median, IQR, and worst case**. Single-shot wall-clock numbers are not measurements.
- Memory measurements report **peak RSS** over the run, not instantaneous values.
- Run on a quiet machine. Background CPU contention invalidates the run; document the host (CPU model, frequency governor, RAM, OS) in the report.
- For GPU work: report device, driver, CUDA version, and confirm `nvidia-smi` shows no other workload during the measurement window.

### Performance regression discipline

- A regression of more than **10 %** versus the previous measured baseline (memory or latency) blocks completion until investigated.
- The default attribution for a regression is **"a bug was introduced"**, NOT "the new method is inherently more expensive."
- A regression may be accepted as inherent only when **all** of the following hold:
  1. A profile is captured using the canonical profiler for the language (Section 10) and attached to the report as a flamegraph SVG.
  2. The profile shows the additional cost concentrated in the **intentional new work**, not in incidental code (allocation, copying, lock contention, accidental O(n²)).
  3. The cost is justified against the change's stated goal in the plan.
- *"It must be the algorithm"* is a hypothesis, not an explanation.
- Symmetric rule for speedups: a > 10 % improvement must also be confirmed by profile, not by single-shot wall-clock variance.
- No micro-optimization is permitted without a profile demonstrating the targeted code as a hot spot. "Defensive optimization" — refactoring for hypothetical performance — is forbidden.

### Required tooling

Test runners, benchmark frameworks, profilers, memory profilers, coverage tools, and property-testing frameworks are pinned in **Section 10**. Substitution requires Section 1 approval.

---

## 4. Resource Budgets

- **Hard memory cap: 16 GB.** This applies to every process spawned by a task.
  - Enforce with `ulimit -v 16777216`, `systemd-run --user -p MemoryMax=16G`, Linux cgroups v2, or `resource.setrlimit(resource.RLIMIT_AS, …)` in Python entry points.
  - If a run exceeds this, **abort and redesign**. Do not raise the cap. Do not add swap. Reduce batch size, stream the data, or refactor the algorithm.
- Every long-running script must:
  - Report **peak RSS** and **wall time** on exit.
  - Support checkpointing. Never rely on a single uninterrupted multi-hour run.
- GPU memory budgets, if any, are declared in the plan per task.

---

## 5. Data-Oriented Design

Prefer data-oriented layouts where reasonable:

- **Struct-of-arrays** over array-of-structs for hot loops and bulk numerical state.
- **Contiguous, cache-friendly buffers.** Avoid pointer chasing in inner loops.
- Separate **cold metadata** from **hot numerical data**.
- Python: prefer NumPy / PyTorch / Polars vectorized ops over per-object Python loops. A `for` loop over `n > 10⁴` Python objects in a hot path is a code smell.
- Rust: prefer `Vec<T>` of POD types over `Vec<Box<dyn Trait>>` in hot paths. Reach for `bytemuck`, `ndarray`, or `arrow` when appropriate.
- C/C++: prefer flat arrays and indices over linked structures in inner loops.

OO inheritance is acceptable for **control flow** and high-level orchestration. It is **not** the right tool for bulk numerical state.

### Numerical stability (FP-heavy code)

For numerically sensitive code (G-SPHF kernels, KAN basis evaluation, B-spline recursion, gradient updates):

- Avoid catastrophic cancellation: do not write `(a - b) / (a + b)` directly when `a ≈ b`. Reformulate.
- Use Kahan or Neumaier summation for accumulators over more than ~10⁴ terms, or anywhere magnitudes vary by orders of magnitude.
- Document conditioning assumptions in the function's contract (Section 8). State the input range over which the result is trusted.
- Test against a high-precision reference (`mpmath` or symbolic `sympy`) for boundary inputs.

---

## 6. Code Health

This section enumerates code-quality ceilings. Both subsections are gates before reporting completion.

### 6.1 No Redundant Code

Before adding a new function, **search the codebase**. If similar logic exists, extend or refactor — do not paste a near-copy.

- The same algorithm appearing in **three or more** places is a refactor trigger, not a feature.
- Use traits / ABCs / interfaces / generics to unify variants.
- If you find yourself writing a block you have written before in this repository, stop and consolidate.

You are **forbidden** from emitting many copies of essentially the same code. If a unification crosses into `CORE.YAML`-protected files, see Section 1 — do not unify, halt and ask.

### 6.2 Complexity Budget

Static analysis is a hard gate before reporting completion.

#### Thresholds (per function unless stated)

- Cyclomatic complexity (McCabe): warn at 10, fail at 15.
- Cognitive complexity (Sonar/Clippy): warn at 15, fail at 25.
- Function length: warn at 80 lines, fail at 200.
- Nesting depth: fail at 5.
- Module length: warn at 800 lines.

#### Tooling

- Rust: `cargo clippy -- -D clippy::cognitive_complexity` plus `rust-code-analysis` for cyclomatic.
- Python: `radon cc -a -nc <path>` and `flake8 --select=C90 --max-complexity=10`.
- C/C++: `lizard -CCN 15 -L 200 <path>`.

#### Rules

- A function over the hard ceiling cannot ship. Refactor required: extract function, table-driven dispatch, replace conditional with polymorphism, or split state machine.
- Generated code is exempt but must be marked (`// generated` header or path matching `**/generated/**`).
- Waivers must be declared in the report (Section 9) with reason.

### 6.3 Static Analysis Gate

All linters and type checkers must pass before a task is reported complete. **Warnings are errors.**

#### Required gates

- Rust: `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
- Python: `ruff check` and `mypy --strict` on changed code (full project where feasible).
- C/C++: `cppcheck --enable=all --error-exitcode=1`, plus `clang-tidy` where configured.

#### Rules

- New `#[allow(...)]` (Rust), `# type: ignore` / `# noqa` (Python), or `// NOLINT` (C++) requires an inline comment stating the reason, scoped to the smallest possible region (single line or single function).
- A blanket `#![allow(...)]` at crate root is a core-level decision; treat as a `CORE.YAML` edit (Section 1).
- Suppressions accumulated by a change must be listed in the report.

### 6.4 Error Handling Discipline

No silent failures. Every error path is explicit.

#### Rust

- No `unwrap()` or `expect()` in non-test code, except where preceded by an explicit invariant check that makes the panic provably unreachable. Document the invariant in a comment immediately above the call.
- Use `Result<T, E>` with concrete error types. `thiserror` for libraries; `anyhow` only at binary boundaries.
- `?` is the default propagator. Manual error mapping must add information, not strip it.

#### Python

- No bare `except:`. Catch the most specific exception type that is meaningful.
- Re-raise with context: `raise NewError(...) from err`. Do not swallow the cause.
- Logging an exception and continuing is **not** error handling unless the recovery path is documented and tested.

#### C/C++

- Every function returning an error code has its return value checked.
- No discarding of `errno`. No silent `NULL` returns from functions that allocate.

Reports must list any new error-handling waivers (`unwrap`, broad `except`, ignored return value) introduced by the change, with justification.

---

## 7. Design Patterns

Apply standard patterns where they reduce coupling and clarify intent. Examples appropriate to this codebase:

- **Strategy** — swappable algorithms (optimizers, kernels, activation backends, scoring functions).
- **Builder** — complex configuration objects (training configs, model specs).
- **Adapter / Facade** — bridging core (read-only) APIs to new code without modifying core.
- **Observer / Pub-Sub** — telemetry, training callbacks, monitoring hooks.
- **Command** — reproducible, replayable experiment steps; pairs well with experiment logging.
- **Visitor** — traversals over heterogeneous IR / AST / hypergraph node types.
- **Repository** — abstracting dataset / checkpoint / artifact storage.

Do not over-pattern. A pattern is justified only when it removes a concrete duplication, coupling, or extensibility problem stated in the plan.

---

## 8. Design by Contract

Every new public function specifies:

- **Preconditions** — what must hold on inputs (types, shapes, dtypes, ranges, invariants).
- **Postconditions** — what is guaranteed about outputs and side effects.
- **Invariants** — what state is preserved across the call.

### Implementation

- **Rust:** `debug_assert!` for preconditions and invariants in debug builds; document contracts in rustdoc under `# Preconditions`, `# Postconditions`, `# Invariants`, `# Panics`, `# Errors`.
- **Python:** `assert` statements, or `icontract` / `pydantic` validators, plus explicit docstring sections (`Preconditions`, `Postconditions`, `Invariants`). Do not rely on type hints alone for shape/range constraints.
- **C/C++:** `<cassert>` macros, or contract attributes where available. Document in headers, near the declaration.

Contracts are evaluated in debug and test builds. A violated **precondition** is a bug in the **caller**. A violated **postcondition** is a bug in the **function**. Reports must distinguish the two.

---

## 9. Reporting

Every completed task produces a report at:

```
reports/<YYYY-MM-DD>-<slug>.md
```

The report contains, at minimum:

- **Summary** of the change.
- **Files touched** (full list, with line counts added/removed).
- **CORE.YAML items touched** — must be empty, or reference the approval thread.
- **Test results** — per layer: counts, pass/fail, durations.
- **Performance results** — peak RSS, wall time, inference latency, vs. the budget declared in the plan and vs. the previous baseline if available.
- **New / removed dependencies.**
- **Open issues** and **follow-up items.**
- **Experiment provenance** (for runs that produce data, models, or measurements):
  - Git SHA (working tree must be clean, or list dirty files explicitly).
  - Environment dump (`cargo tree`, `pip freeze`), OS, kernel, CPU model, RAM.
  - GPU device, driver version, CUDA / ROCm version where applicable.
  - Random seed(s) used.
  - Dataset hash(es) or fixture content hashes.
  - **For in-flight experiments referenced in the report:** the corresponding log file path and PID (or jobspec). ID-only references with no on-disk anchor are not acceptable — they evaporate when the chat session closes and leave the next session unable to verify.

A task with no report is an incomplete task. The report is the unit of acceptance, not the diff.

---

## 10. Pinned Toolchain

Tool variability between runs invalidates comparison. This section fixes **one canonical tool per concern per language**. Use the listed tool. Substitution requires a Section 1 (CORE.YAML-level) approval.

### Test runners

- **Rust:** `cargo test` (release-mode for performance tests).
- **Python:** `pytest -p no:randomly` (deterministic order; opt into shuffle explicitly per test).
- **C/C++:** `gtest`.

### Benchmarking

- **Rust:** `criterion` — statistically rigorous, regression-aware.
- **Python:** `pytest-benchmark`.
- **C/C++:** `google/benchmark`.

Single-shot timers (`time`, `time.time()`, `Instant::now()`) are diagnostic only. They are **not** reportable as benchmarks.

### CPU / wall-time profiling

- **Rust:** `cargo flamegraph` (perf backend on Linux, dtrace on macOS).
- **Python:** `py-spy record -o profile.svg -- <cmd>` (sampling, no code changes, low overhead).
- **C/C++:** `perf record -F 999 -g -- <cmd>` followed by `inferno-flamegraph`.

Profile output is always **flamegraph SVG**, attached to the report. No screenshots, no terminal captures, no proprietary GUI exports.

### Memory profiling

- **Rust:** `dhat` for allocation tracking; `heaptrack` for native traces.
- **Python:** `memray run` followed by `memray flamegraph`.
- **C/C++:** `heaptrack`.

### Coverage

- **Rust:** `cargo llvm-cov --lcov` (replaces `tarpaulin`; faster, branch coverage).
- **Python:** `pytest --cov` with lcov export.
- **C/C++:** `llvm-cov` with `-fprofile-instr-generate -fcoverage-mapping`.

Common output format across languages: **lcov**. Reports include line and branch coverage delta vs. the previous baseline.

### Property-based testing

- **Rust:** `proptest`.
- **Python:** `hypothesis`.
- **C/C++:** `rapidcheck`.

### Static analysis & complexity

Already pinned in Sections 6.2 and 6.3 (clippy, rustfmt, ruff, mypy, cppcheck, clang-tidy, radon, lizard, rust-code-analysis).

### Versions

Tool versions follow a **semver major-locked** policy, declared in `tools.yaml` at the repo root. Claude Code reads `tools.yaml` at session start, alongside `CORE.YAML`.

- Each entry pins a `major_version`. Minor and patch upgrades are **free** and need not be reflected in `tools.yaml`.
- Bumping the **major version** of a pinned tool requires Section 1 approval.
- Versions must be valid semver (`MAJOR.MINOR.PATCH`). Non-semver markers (commit hashes, dates, `latest`, `nightly`) are not accepted; halt and ask if a tool publishes only such markers.
- For 0.x tools, `major_version: "0"` allows any 0.y.z. This is deliberately looser than Cargo's caret semantics.
- Substitution to a **different tool** (e.g., `samply` instead of `cargo flamegraph`) is a separate axis and also requires Section 1 approval.
- If a minor/patch bump introduces a measurement-relevant behavior change (sampling backend swap, output format change, breaking deprecation), note it in the report.

Tools that track an external toolchain (rustup, Linux kernel, LLVM) are declared without `major_version` and follow the toolchain.

---

## 11. Halt Conditions

Stop and ask the user before proceeding if any of the following hold:

- A change requires modifying any item in `CORE.YAML`.
- A test cannot be written for a proposed change.
- The 16 GB memory cap will be exceeded.
- Any of the four plan artifacts (`plan.tex`, `plan.pdf`, `plan.tikz`, `plan.mmd`) cannot be produced.
- You are about to run an unverified experiment that **mutates persistent state** — datasets, checkpoints, trained model weights, databases, remote storage.
- A test fails and the cause is not understood.
- A measurement contradicts an assumption in the plan.
- **A diagnosed bug cannot be reproduced against the exact environment that produced the failure.** When you investigate an OOM, crash, or unexpected result, the diagnosis must be checked against the **failing run's actual env config** (script, env vars, command line) — not against the file that was most recently edited. If the diagnosis names a buggy module but that module isn't called from the failing env, the diagnosis is wrong; halt and re-investigate before applying a "fix."
- **A queued long-running script's wall-time estimate disagrees by more than 2× with the closest prior measured baseline.** Either the estimate is wrong (most common) or there's an undocumented optimization in play. Either way, reconcile before launching.

**When in doubt: silence is preferable to a wrong action.** Halting and asking is never penalized. Acting on a guess is.