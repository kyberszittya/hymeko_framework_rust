# P-graph meta-model resolution adapter — 2026-05-25

## Summary

Added a non-core adapter, `hymeko_pgraph::meta_resolve`, that lets a P-graph be
authored in the general HyMeKo meta-model style — an `@"meta_pgraph.hymeko"`
include, `using pgraph.raw as raw` aliases, and instances typed by `<isa>`
(`A: + <isa> raw {}`) — and lowers it to a `LoweredPGraph` for MSG/SSG/ABB.

Previously such files parsed and lowered to an **empty** graph: the existing
`lower(&Description)` reads the raw parser AST and recognises only the literal
role tags `<material>`/`<unit>`/`<raw>`/`<product>`. The adapter instead reuses
`hymeko_core`'s public `ModuleStore::compile` (read-only) to obtain a fully
resolved `Ir` (includes loaded, `using` aliases applied), then walks the IR's
`<isa>` ancestry to classify declarations into the Friedler sets `M`
(materials, with raw/product roles `R`/`P`) and `O` (operating units). HyMeKo
gains no P-graph keywords; only the archetype-name contract lives in the adapter.

Plan: `docs/plans/2026-05-25-pgraph-meta-resolve/` (tex/pdf/tikz/mmd).

## Design

```
*.hymeko ──ModuleStore::compile──▶ resolved Ir ──meta_resolve──▶ LoweredPGraph ──▶ MSG/SSG/ABB
          (hymeko_core, read-only)              (new, non-core)
```

- **Material** (`M`): a `Node` decl whose `<isa>` ancestry reaches `raw` /
  `product` / `intermediate`; role from which archetype (reaching both `raw`
  and `product` ⇒ `ConflictingRole`).
- **Unit** (`O`), **hybrid rule** (per user decision): an `Edge` decl is a unit
  iff its `<isa>` ancestry reaches `process`, *or* its arcs are non-empty and
  every arc target is a material (so a `@dataflow`-style edge over non-materials
  is skipped). Incidence: `-m` ⇒ `m→u`, `+m` ⇒ `u→m`, `~` ⇒ `NeutralRef`; cost =
  edge's numeric value, default `1.0`.
- `<isa>` walking is cycle-safe (visited set) and does not count a decl as its
  own ancestor, so archetypes never self-classify.

## Files touched

| File | Δ | Change |
| --- | --- | --- |
| `hymeko_pgraph/src/meta_resolve.rs` | **new**, 376 | parser shim, `compile_to_lowered`, `lower_resolved`, archetype lookup, `<isa>` walker, hybrid classifier, IR→`LoweredPGraph` |
| `hymeko_pgraph/src/lib.rs` | +2 | export `compile_to_lowered`, `lower_resolved`, `MetaResolveError` |
| `hymeko_pgraph/tests/meta_resolve.rs` | **new**, 119 | 4 integration tests |
| `hymeko_pgraph/src/pgip_io.rs` | +/−4 | `explicit_counter_loop` fix (`.enumerate()`) — a regression from the prior hypergraph-queries simplification |
| `hymeko_pgraph/tests/{multi_objective,axiom_witness}.rs` | +/−8 | pre-existing clippy lints fixed (see below) |

Used read-only (no edit): `hymeko_core::module_store::{ModuleStore, StdFsProvider,
HymekoParser}`, `hymeko_core::ir`, `hymeko_core::resolution::Interner`,
`parser::parse_description`. No new dependency (both already depended on).

## CORE.YAML items touched

Empty list. `hymeko_pgraph` is not core; `hymeko_core`/`parser` are used via
public API only; no pinned-dependency change.

## Verification of foundations (before coding)

`cargo run -p hymeko_cli -- validate hymeko_pgraph/data/prgraph_ex_3_1.hymeko`
⇒ `✅ valid` (exit 0): the include + `using` aliases resolve through core today.

## Test results

| Layer | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` | **105 passed, 0 failed, 1 ignored** (TEST_EXIT=0) |
| New `tests/meta_resolve.rs` | 4 pass: example-3.1 classification (M/R/P + hybrid units + incidence + default cost); MSG prunes the unproduced-`F` branch to `{u1,u4,u5}`; meta-alone ⇒ empty graph; non-meta file ⇒ `MissingArchetype` |
| Existing suites + hypergraph_queries | unchanged pass |

`prgraph_ex_3_1.hymeko` (the user's file) classifies to `M={A,B,C,D,E,F,G}`,
`R={A,B}`, `P={G}`, units `u1..u5` via the structural branch; MSG correctly
drops `u2`,`u3` because `F` is declared intermediate but produced by no unit.

## Performance

Toy scale (`|M|=7`, `|O|=5`). `compile` cost is core resolution + file IO
(already exercised by the CLI); adapter passes are `O(decls + arcs)` with
`O(depth)` `<isa>` walks. Full suite wall time unchanged (per-target ≤ 0.08 s).
No criterion bench (structural, toy scale; consistent with prior pgraph reports);
≪ 16 GB.

## Static analysis

- `cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` — **passes**
  (CLIPPY_EXIT=0, verified by real exit code, not a piped `tail`).
- My library + new/edited test targets each verified clippy-clean individually
  (lib, `--test meta_resolve`, `--test hypergraph_queries`, `--test pgraph_e2e`,
  `--test pgip_io` all exit 0).
- New code adds no `#[allow]`, no `unwrap`/`expect` in non-test code. The
  `build_incidence` tuple return uses a `type Incidence` alias to satisfy
  `type_complexity`.

### Corrections / pre-existing debt fixed

- **Prior-report correction:** the 2026-05-25 hypergraph-queries report claimed
  "clippy passes (exit 0)"; that check piped clippy through `tail`, so the
  reported code was `tail`'s, not clippy's. Re-checked here with real exit codes.
- Under a correct check, the crate had **pre-existing** clippy `-D warnings`
  failures in two untouched test files: `multi_objective.rs:40`
  (elidable lifetime) and `axiom_witness.rs:375-376` (needless borrow). Both are
  trivial and now fixed, so the gate is genuinely green. They are unrelated to
  this feature (flagged here for honesty).
- `pgip_io.rs` `explicit_counter_loop`: the hypergraph-queries simplification
  shrank that reader loop enough to newly trip the lint; fixed with `.enumerate()`.

### rustfmt — pre-existing crate-wide drift (unchanged, out of scope)

`cargo fmt --check` remains red crate-wide (≈120 diffs across files this task
never opened), as documented in the hypergraph-queries report. My new files
(`meta_resolve.rs` src + test) are rustfmt-clean. Recommend a separate
formatting-only commit.

## §6.5 anti-patterns

None introduced. The adapter **reuses** core resolution rather than
reimplementing include/alias handling (avoids the parallel-machinery
anti-pattern). Archetype-string knowledge is centralised in four `const`s, not
scattered string matches.

## Open issues / follow-up

1. **CLI / dump wiring** — `hymeko_pgraph_dump` and `analyze_source` still lower
   via the literal-tag AST path, so they show meta-model files as empty. A small
   follow-up: have the dump binary call `compile_to_lowered` when the source has
   includes/`using`/`<isa>`.
2. **Crate-wide rustfmt + remaining clippy sweep** — separate cleanup commit.
3. **Material name collisions** — `name_to_decl` keys on short names; cross-
   namespace collisions would overwrite. Fine for instance files; revisit if
   multi-namespace P-graphs appear.

## Experiment provenance

N/A (no training runs / no persistent-state mutation). Git SHA at task start:
`db99de0`. Working tree dirty: this task's `hymeko_pgraph` changes, the prior
hypergraph-queries changes, and the orthogonal pre-existing `Core.yaml`/`*ools.yaml`
edits.
