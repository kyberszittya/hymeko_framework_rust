# P-graph process info as signed-incidence hypergraph queries — 2026-05-25

## Summary

Reverted the `ProcessDefinition` semantic layer added earlier the same day and,
per user direction, made the Friedler Definition-1 *process information* a set of
**queries over the signed-incidence hypergraph** rather than a stored side table.

The directed edge set in `PGraphSchema` already *is* the signed incidence
(`m → u` = consumed/`-`, `u → m` = produced/`+`). The per-unit `unit_inputs` /
`unit_outputs` maps on `LoweredPGraph` were a redundant denormalisation of that
edge set, independently built in three places (lowering, builder, pgip reader) and
able to drift. They are removed. `PGraphSchema::try_new` now derives an adjacency
index in the single pass it *already* makes for the bipartite check, and exposes
`predecessors` / `successors`; `LoweredPGraph` exposes `inputs` / `outputs` as thin
delegations. `raws` / `products` remain stored (tag information is not in the edge
set). This supersedes and removes `reports/2026-05-25-pgraph-process-definition.md`.

Plan: `docs/plans/2026-05-25-pgraph-hypergraph-queries/` (tex/pdf/tikz/mmd).

## Files touched

| File | Δ | Change |
| --- | --- | --- |
| `hymeko_pgraph/src/schema.rs` | +92/−2 | adjacency derived in `try_new`; add `predecessors`/`successors` + 2 unit tests |
| `hymeko_pgraph/src/lowering.rs` | +38/− | drop `unit_inputs`/`unit_outputs` fields + construction; add `inputs`/`outputs` methods |
| `hymeko_pgraph/src/builder.rs` | +/−12 | reverted to direct construction; build only the edge set |
| `hymeko_pgraph/src/pgip_io.rs` | +/−45 | reader builds only edges; writer reads via `inputs`/`outputs` |
| `hymeko_pgraph/src/msg.rs` | +/−37 | `p.unit_inputs.get(u)` → `p.inputs(u)` (idem out); fixpoint passes |
| `hymeko_pgraph/src/ssg.rs` | +/−11 | `is_feasible` via queries |
| `hymeko_pgraph/src/dump.rs` | +/−8 | `project_subschema` via queries |
| `hymeko_pgraph/tests/{pgraph_e2e,pgip_io,axiom_witness}.rs` | +/−54 | retarget field access to queries |
| `hymeko_pgraph/tests/hypergraph_queries.rs` | **new**, 100 | 3 integration tests (signature recovery, disposal-sink empty-outputs, HDA solve) |
| **deleted** | — | `src/process_def.rs`, `tests/process_def.rs`, `data/pgraph/meta_process.hymeko`, `reports/2026-05-25-pgraph-process-definition.md`, `docs/plans/2026-05-25-pgraph-process-definition/` |

`lib.rs` is byte-identical to HEAD: the revert dropped the `process_def` exports,
and the new queries are methods on the already-exported `PGraphSchema` /
`LoweredPGraph`, so no export change was needed.

Not part of this change: `Core.yaml` / `Tools.yaml` / `tools.yaml` working-tree
edits predate this task (CORE restoration + tools.yaml population from a prior
session); left untouched.

## CORE.YAML items touched

Empty list. `hymeko_pgraph` is not a core crate; `hymeko_core` used read-only; no
pinned-dependency change.

## Interface changes

- **Added** `PGraphSchema::predecessors(DeclId) -> &BTreeSet<DeclId>` and
  `successors(...)`; `LoweredPGraph::inputs(u)` / `outputs(u)` (delegations).
- **Removed** public fields `LoweredPGraph::unit_inputs` / `unit_outputs`, and the
  `ProcessDefinition` / `ProcessDefError` types and exports. In-crate breaking
  change only; grep confirms no other workspace crate read them.

## Test results

| Layer | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` | **101 passed, 0 failed, 1 ignored** |
| New `schema` unit tests | `predecessors_successors_match_incidence`, `empty_and_absent_neighbours` — pass |
| New `tests/hypergraph_queries.rs` | 3 pass (incl. disposal-sink empty-vs-absent regression) |
| Existing suites (e2e, pgip_io, relaxed_msg, axiom_witness, multi_objective, builder, byproduct, gomb) | unchanged pass (now driven through the queries) |

Per-target wall times all ≤ 0.07 s (toy scale). The behaviour-preservation is
gated by the existing suites, which were rewired onto the queries: a wrong
adjacency derivation fails them.

## Performance

Adjacency is built inside the `try_new` edge pass that already ran for the
bipartite check: **O(E) build, no asymptotic change**; queries are an O(1) map
lookup returning a reference (the old call sites cloned). No new bench added —
structural refactor at toy scale (`|O| ≤ 29` across shipped Chapter fixtures),
consistent with prior pgraph reports; ≪ 16 GB cap. A > 10 % regression was not
observed in suite wall time. No profile attached (no perf claim made either way).

## Static analysis

- `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` — **passes** (exit 0).
- New code adds **no** `#[allow]`, `unwrap`/`expect` in non-test code, or other
  suppressions. The `static EMPTY_NEIGHBOURS: BTreeSet<DeclId>` uses the const
  `BTreeSet::new()` to return a shared empty borrow (no allocation, no `unsafe`).

### rustfmt — pre-existing crate-wide drift (flagged, out of scope)

`cargo fmt -p hymeko_pgraph --check` is **red**, but this predates the change:
~120 diffs span files this task never opened (`axioms.rs`, `axiom_extensions.rs`,
`relaxed_msg.rs`, `multi_objective.rs`, `byproduct_filter_phase11.rs`, the dump
binary). Every diff in the files I *did* edit was verified to be pre-existing code
(e.g. `per_unit_dim` declarations, `self.materials.get(...)` chains) — not my
edits. My own contributions (`schema.rs`, `tests/hypergraph_queries.rs`) are
rustfmt-clean. Reformatting the whole crate would add unrelated churn across ~15
files and bury the change; recommend a **separate formatting-only commit** to
bring the crate to rustfmt-clean. No fmt regression introduced by this task.

## §6.5 anti-patterns

None introduced; this change *removes* a duplication (anti-pattern #1/§6.1: the
same incidence stored in four representations). Single source of truth is now the
edge set.

## Dependencies

None added or removed.

## Open issues / follow-up

1. **Crate-wide rustfmt sweep** — separate formatting-only commit (see above).
2. **`materials` / `units` fields retained** on `LoweredPGraph` as cached views of
   `schema.m_nodes()` / `o_nodes()`. Not the flagged duplication (built from the
   same `kinds` source, single construction), left to bound churn. Could become
   `m_nodes()`/`o_nodes()` delegations in a later pass if desired.
3. **`Tools.yaml` vs `tools.yaml`** casing duplicate still present (pre-existing).

## Experiment provenance

N/A (no training runs / no persistent-state mutation). Working tree dirty:
the `hymeko_pgraph` changes above plus the pre-existing `Core.yaml`/`*ools.yaml`
edits. Git SHA at task start: `db99de0` (HEAD).
