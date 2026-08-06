# HOTARU plans over the HyMeKo-described graph elements → a semantic plan

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `1a069c8c`.

---

## Summary

HOTARU now plans over the **graph a HyMeKo model describes** — its declared nodes and hyperedges — and
produces a **semantic plan**: a sequence of graph operations grounded in the model's real elements, toward
a *topological* goal. Previously `SearchHotaru` searched a hand-provided menu of opaque source-string
deltas toward an edge-name objective; it now also reads the parsed graph structure and composes
node-grounded `add_edge` operations toward goals like "connect node `a` to node `c`". The
`candidate_edges` + goal are exactly the **specification an LLM would emit from intent** — HOTARU turns
that spec into a verified, minimal, gatekeeper-valid plan.

## What was built (all in `akoire`, non-core, no new dependency)

| piece | what it does |
|---|---|
| `engine::preview_graph(source) -> GraphView` | dry-run parse into the model's **graph elements**: declared nodes + hyperedges (name + the nodes each connects), read from the parser AST (`header` node decls + `EdgeInner.bases`) |
| `GraphView` | `nodes()`, `edges()`, `has_node/has_edge`, and **`connected(a,b)`** — union-find over the hyperedge incidence (each edge fully connects its nodes) |
| `hotaru::HiveDelta::add_edge(name, nodes)` | a **semantic** graph operation (`@name : n0, n1 { }`) grounded in real node names |
| `hotaru::GraphGoal` | a topological goal: `RequiredEdges(names)` or **`Connect(a, b)`** (nodes in one component) |
| `hotaru::SearchHotaru::plan_graph(seed, candidate_edges, goal, kyosei, max)` | reads the seed's nodes, **grounds** candidates (drops any referencing a node the model does not declare), and searches — via the shared `SearchProblem`/`solve` A\* — for the shortest grounded `add_edge` sequence reaching `goal` |

The neighbour/feasibility relation (parse + Kyosei arity) is factored into one `feasible_successors`
shared by the edge-name planner and the graph planner (§6.1). The A\* heuristic is admissible (missing
required edges, or 0/1 for connectivity), so the semantic plan is minimum length.

## The LLM seam (why this is the right shape)

HOTARU is the **deterministic, verifiable** half; an LLM is the **open-ended** half. The division:

- **LLM** reads intent + the model, and emits a *specification*: a `GraphGoal` (what topology to reach)
  and a `candidate_edges` vocabulary (which edges are worth considering). This is the ambiguous,
  natural-language-grounded step.
- **HOTARU** grounds that spec against the model's real nodes, searches the shortest sequence, and every
  step passes the parser gatekeeper — so a hallucinated node or an unparseable edge simply never enters
  the plan. The output is a *correct* semantic plan, not a guess.

So the LLM proposes a plan/spec; HOTARU disposes — over the actual graph elements.

## Full flow (verified)

`plan_graph` drives the unchanged `CognitiveLoop`: from a committed seed model (nodes `a, b, c`, no
edges) and the goal `Connect(a, c)`, HOTARU derives the plan `[add e_ab, add e_bc]`; the loop gate-keeps
each add-edge, and the final committed model **actually connects `a` and `c`** (checked on the parsed
graph, not just by edge name). Integration test: `graph_plan_drives_loop_to_a_connected_model`.

## Files touched

| File | +/− | notes |
|---|---|---|
| `akoire/src/engine.rs` | +147 / −1 | `GraphView` + `GraphEdge` + `preview_graph` + `connected` (union-find) + `collect_graph` |
| `akoire/src/hotaru.rs` | +224 / −18 | `HiveDelta::add_edge`, `GraphGoal`, `GraphProblem`, `plan_graph`, `feasible_successors` refactor, 6 graph tests |
| `akoire/src/lib.rs` | +7 / −2 | export `GraphView`, `GraphEdge`, `GraphGoal`, `preview_graph` |
| `akoire/tests/hotaru_search.rs` | +40 / −2 | the full-flow graph integration test |

## CORE.YAML / dependencies

None. `parser` used through its public AST only (read-only); no new dependency; `akoire` is non-core.

## Test / gate results

- `cargo test -p akoire` → **36 pass / 0 fail** (29 lib incl. 6 new graph tests + 7 integration across the
  test binaries) + 1 ignored doctest. New graph tests: node/edge/connectivity extraction; `plan_graph`
  connects two nodes via the minimal edge chain; prefers a direct edge when available; **grounds
  candidates against real nodes** (a candidate over an absent node is dropped ⇒ unreachable); required-edges
  goal; unparseable-seed ⇒ `None`; and the loop-driven connected-model integration.
- `cargo clippy -p akoire --no-deps --all-targets -- -D warnings` → clean; `cargo fmt --check` → clean.

## §6.5 anti-patterns

None. The graph planner reuses the shared `SearchProblem`/`solve` A\* and one `feasible_successors`
oracle (§6.1); `GraphGoal` is a sealed enum-with-data, not a string mode (§6.5 #7); no globals; no new
`unwrap`/`expect` in non-test code (union-find indexes a map key proven present).

## Open issues / follow-up

- **Signed / directed goals**: `add_edge` emits neutral references; signed hyperedges (`+ a, − b`) and
  arc-directed goals are a natural extension (the AST already carries `SignedRef`).
- **Richer topological goals**: cycle-freeness, k-connectivity, a required subgraph — each a new `GraphGoal`
  variant over `GraphView`.
- **The LLM half**: an `LlmSynthesizer` (sketched in `synthesize.rs`) that emits `(GraphGoal,
  candidate_edges)` from intent — the open-ended front-end this planner is the verifier for.
- **Node-adding operations**: today the plan only adds edges (nodes are fixed by the seed); an `add_node`
  delta would let HOTARU grow the vertex set too.

## Provenance

Git SHA `1a069c8c` at start. Rust `rustc 1.96.1`, `cargo`/`clippy`/`rustfmt` (§10). Host macOS (darwin
25.5), Apple Silicon. Deterministic: the A\* is seed-free; union-find is order-independent.
