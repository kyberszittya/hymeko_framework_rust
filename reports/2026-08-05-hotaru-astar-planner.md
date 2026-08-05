# HOTARU kickoff — an A* planner over the AKOIRE cognitive loop

**Date:** 2026-08-05 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov`
· **Git SHA at start:** `aff5057c` (clean working tree at start; this change adds/edits `akoire/` only).
**Plan:** `docs/plans/2026-08-05-hotaru-astar-planner/` (tex/pdf/tikz/mmd; gitignored per repo convention).

---

## Summary

HOTARU (`akoire::hotaru`) had only a `ScriptedHotaru` stand-in — a fixed delta queue; the
`HotaruPlanner::next_delta` seam was never driven by a real planner. This change **kicks HOTARU off with
a real planner**: `SearchHotaru` **derives** its delta sequence by **A\* search over the implicit
HIVE-delta space** toward the `Objectives` goal, then streams it through the unchanged
`HotaruSynthesizer` / `CognitiveLoop` / `HymekoEngine` gatekeeper — no hand-written script anywhere.

This is the "reuse the RRT/A\* planner framework to kickoff HOTARU" increment, on the cognitive /
structure-synthesis side (the user's chosen layer).

### Why HOTARU carries its own A\* (not `hymeko_graph::astar`)

Reusing the concrete `hymeko_graph::astar` was **doubly blocked**, and this is the honest engineering
call, not a shortcut:

1. **Architectural mismatch.** `hymeko_graph::astar` runs on a fixed integer-indexed **CSR** (unweighted,
   `max_depth`). The HIVE-delta space is an **implicit** graph: nodes are cognitive states (HyQL source
   strings) *generated on the fly*, edges are named deltas, each candidate vetted by the parser. A CSR
   cannot represent an unbounded on-the-fly node space.
2. **CORE (§1) dependency.** `akoire → hymeko_graph` is a **new dependency = a `CORE.YAML` change**
   requiring escalation.

So `akoire` carries a small `std`-only **generic implicit-graph A\***, mirroring the framework's other
implicit-graph A\* (`hymeko_rl.control.graph_planner.astar`): the same algorithm, a node model that fits,
**zero new dependencies**. The design-level reuse is real; the concrete-crate reuse was infeasible.

## What was built

| piece | what it does |
|---|---|
| `search::astar<N,E>` | generic implicit-graph A\*; `neighbours(n)→(edge,next,cost)` on demand; returns `AstarResult{actions, expansions, frontier_peak}` (optimal path + effort stats) |
| `SearchHotaru` (`impl HotaruPlanner`) | plans a delta path by A\*; the **parser is the feasibility oracle** (a successor is kept only if it parses), so ordering (host block before a referencing edge) is enforced for free; streams the plan via `next_delta` |
| `engine::preview_edge_names` | non-mutating dry-run parse (`Some(edge_names)`/`None`); the search's feasibility + edge oracle, through the *same* parser boundary the loop gate-keeps with (reuses `collect_edge_names`) |
| `Objectives::satisfied_edges` / `missing_edges` | the goal predicate (factored out of `Goal::satisfied`, §6.1 single source of truth) and the **admissible** heuristic |

**Heuristic admissibility (measured claim → proof):** each menu edge-delta adds ≤1 required edge, the base
adds none, every delta costs 1 ⇒ `h = #missing required edges` never overestimates remaining steps, and is
consistent ⇒ A\* returns a shortest delta path. The empty-menu / `ZeroHeuristic` cases are tested too.

**Load-bearing result (measured, not assumed):** the parser *rejects* a top-level edge with no host block
(`preview_edge_names("@e_ab : a, b { }") == None`), so the search **cannot** place an edge before the base
— `base-first` ordering is enforced by the feasibility oracle. The A\* plan for `{e_ab, e_bc}` is
`[base, e_ab, e_bc]` (the distractor `@dist` is pruned as never-needed).

## Files touched

| File | +/− | notes |
|---|---|---|
| `akoire/src/search.rs` | +234 / new | generic implicit-graph A\* + `AstarResult` + 5 unit tests |
| `akoire/src/hotaru.rs` | +218 / −1 | `SearchHotaru` (`plan`/`search`/`remaining`/`next_delta`) + 6 unit tests |
| `akoire/src/context.rs` | +71 / −3 | `satisfied_edges` + `missing_edges`; `Goal::satisfied` refactored to reuse; +3 unit tests |
| `akoire/src/engine.rs` | +17 / 0 | `preview_edge_names` (dry-run parse) |
| `akoire/src/lib.rs` | +4 / −2 | `pub mod search;` + re-exports (`astar`, `AstarResult`, `SearchHotaru`, `preview_edge_names`) |
| `akoire/tests/hotaru_search.rs` | +81 / new | 2 integration tests (Converged / Exhausted through the loop) |
| `akoire/benches/loop_bench.rs` | +28 / −2 | `criterion` case `hotaru_search_plan_two_edge` (latency provenance) |

## CORE.YAML items touched

**None.** `parser` (CORE, `lockdown: full`) is used through its public `parse_description` only, unedited
(`git status parser/` clean). **No new dependency** (§1 avoided by design). `akoire` is non-core.

## Test results (`cargo test -p akoire`)

| layer | count | result |
|---|---|---|
| unit — `search::astar` | 5 | pass (shortest weighted path; `h≡0` optimal; unreachable→None; budget respected; start-is-goal) |
| unit — `SearchHotaru` | 6 | pass (base-first forced; optimal plan skips distractor; all prefixes feasible; unreachable→None; budget→None; **expansion budget**) |
| unit — `Objectives` | 3 | pass (satisfied_edges normal/negative/empty; missing_edges count) |
| unit — pre-existing | 7 | pass (no regression from the `Goal::satisfied` refactor) |
| integration — `hotaru_search.rs` | 2 | pass (planned HOTARU → `Converged`, rounds==accepted==plan.len, rejected==0; short plan → `Exhausted`) |
| integration — pre-existing | 2 | pass |
| doctest | 1 | ignored (existing `ignore` block) |

**Total: 25 pass / 0 fail** (21 lib + 4 integration), 0.04 s. `cargo clippy -p akoire --no-deps
--all-targets -- -D warnings` → clean. `cargo fmt -p akoire -- --check` → clean.

> Note: `cargo clippy` *with* deps surfaces a pre-existing `clippy::needless_return` in the CORE `parser`
> crate (2 sites, untouched by this change, `lockdown: full` ⇒ §1). Out of scope; `--no-deps` lints only
> `akoire`, which is clean.

## Performance results

- **Deterministic budget (asserted, CI-safe):** planning `{e_ab, e_bc}` expands **≤16 nodes**
  (`frontier_peak ≤16`), plan length **3** — asserted in `search_meets_expansion_budget`.
- **Latency provenance (`criterion`, 100 samples, §10 canonical tool):**
  `hotaru_search_plan_two_edge` median **31.4 µs** [31.39, 31.48] — the A\* plan, dominated by ~7
  `preview_edge_names` parser calls; well under the 1 ms plan budget.
  `cognitive_loop_converge_1_round` median **863 ns** (unchanged baseline ⇒ the loop path is untouched, no
  regression).
- **Peak RSS:** negligible (a Rust unit/bench process, ≪ 16 GB cap).

## §6.5 anti-patterns

None introduced. No Cartesian API surface (one `astar` + one `SearchHotaru`), no string-typed config, no
globals, the A\* algorithm is not duplicated (design-level reuse of the framework's implicit-graph A\*; the
concrete CSR variant genuinely does not fit — documented in the module header). No new `unwrap`/`expect` in
non-test code; the one indexing site (`nodes[seq]`) is guarded by a documented invariant (§6.4).

## Open issues / follow-up

- **Delta cost / weighting.** Costs are uniform (1 per delta); a future menu can attach per-delta costs
  (arity, risk) — the `astar` API already takes `f64` costs.
- **Larger menus.** The implicit graph is exponential in the menu; `max_expansions` bounds it (returns
  `None`, no spin). A domain heuristic beyond `#missing edges` would scale it.
- **Kyosei.** The planner ignores `Kyosei::max_arity`; wiring it as a neighbour filter (prune deltas that
  exceed the arity bound) is the next honest increment.
- **Bridge to motion.** The unifying HOTARU-planner-framework option (one planner interface spanning
  HIVE-delta *and* humanoid footstep planning) remains offered, not built.

## Provenance

Git SHA `aff5057c` (branch `research/humanoid-com-lyapunov`); working tree at start clean except this
change (all under `akoire/`). Toolchain: `rustc 1.96.1`, `cargo test`/`clippy`/`rustfmt`/`criterion` (§10
pins). Host: macOS (darwin 25.5), Apple Silicon. Deterministic: A\* is seed-free (monotonic-counter
tie-break); no dataset, no GPU.

---

## Update — Kyosei arity filter (#2) + unifying `SearchProblem` framework (#3)

Two additive increments on top of the kickoff (planned in `docs/plans/.../plan.tex` §Addendum, committed
after the first commit `2b2b2c52`). Both non-core, no new dependency.

### (2) Kyosei arity filter
The planner now respects `Kyosei::max_arity` (the physical arity bound). Wired as a **neighbour filter**:
a successor whose applied state contains any hyperedge of arity > `max_arity` is dropped, so the search
only traverses arity-respecting states. Arity = `EdgeInner.bases.len()` (referenced-vertex count) read
from the parsed AST via the new `engine::preview_edges(source) -> Option<Vec<(String, usize)>>`;
`preview_edge_names` / `collect_edge_names` were refactored to derive from the one traversal (§6.1).
`SearchHotaru::plan`/`search` gained a `kyosei: &Kyosei` parameter (5 params, under the §6.5 config-struct
threshold). It is a **filter, not a heuristic** — the parser's arity is exact.

- **Regression test** (`kyosei_arity_bound_prunes_high_arity_edges`): a menu whose only route to the goal
  needs a 3-ary edge → `max_arity 2` ⇒ `plan` returns `None` (goal unreachable); `max_arity 3` ⇒ the plan
  reaches it. Would have failed against the prior (arity-blind) implementation.

### (3) Unifying `SearchProblem` framework
Extracted the planner interface the "reused by both" ask needs: `trait SearchProblem { type Node; type
Edge; start; neighbours; is_goal; heuristic }` + `fn solve<P: SearchProblem>(&P, max_expansions) ->
AstarResult<P::Edge>` (delegates to the closure-based `astar`). HOTARU's search is now a
`HiveDeltaProblem` implementing `SearchProblem` (structure-synthesis instance), solved by `solve`. A
second instance — a 4-connected occupancy-**grid** `SearchProblem` (the shape a footstep planner takes) —
is added under test and proves the *same* framework spans **structure** (HIVE-delta) and **motion**.

- **Honest scope**: the humanoid footstep planner lives in Python (`scenarios/humanoid`); it *mirrors*
  this interface rather than sharing the Rust trait across the FFI boundary. The in-crate grid instance
  (`solve_drives_a_motion_grid_problem`: detours a two-cell wall, optimal path length 7) is the Rust-side
  generality proof.

### Files touched (delta vs `2b2b2c52`)
| File | +/− | notes |
|---|---|---|
| `akoire/src/search.rs` | +90 / 0 | `SearchProblem` trait + `solve` + grid `SearchProblem` test |
| `akoire/src/hotaru.rs` | +112 / −43 | `HiveDeltaProblem` (`impl SearchProblem`) + `kyosei` param + `search` via `solve` + Kyosei test |
| `akoire/src/engine.rs` | +27 / −11 | `preview_edges` (arity); `preview_edge_names`/`collect_edge_names` derive from one traversal |
| `akoire/src/lib.rs` | +2 / −2 | export `solve`, `SearchProblem`, `preview_edges` |
| `akoire/benches/loop_bench.rs` | +8 / −2 | `kyosei` arg |
| `akoire/tests/hotaru_search.rs` | +4 / −2 | `kyosei` arg |

### Test / gate results (after #2 + #3)
- `cargo test -p akoire` → **29 pass / 0 fail** (23 lib + 6 integration across 3 binaries) + 1 ignored
  doctest. New: `solve_drives_a_motion_grid_problem`, `kyosei_arity_bound_prunes_high_arity_edges`.
- `cargo clippy -p akoire --no-deps --all-targets -- -D warnings` → clean; `cargo fmt -- --check` → clean.
- **Perf:** `hotaru_search_plan_two_edge` median **32.0 µs** [31.90, 32.07] (was 31.4 µs; arity filter adds
  `preview_edges` per neighbour — within noise, no regression). Deterministic expansion budget unchanged
  (≤16).

### §6.5 anti-patterns (update)
None. `SearchProblem` is the Strategy/Template interface the user asked for (not over-patterning — §7); the
Kyosei axis is a `kyosei` parameter, not a new `plan_*` variant (§6.5 #1/#5); the AST traversal is unified
(§6.1). No new `unwrap`/`expect` in non-test code.
