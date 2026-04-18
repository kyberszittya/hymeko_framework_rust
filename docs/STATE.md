# HyMeKo Framework — State Snapshot

**Date:** 2026-04-18
**Branch:** `refactor/extract-hymeko-hre` (off `dev/query_engine`)
**Workspace root:** `hymeko_framework_rust/`

A point-in-time map of what's integrated, what's in-flight, and what's planned. Renew whenever a major phase lands. This doc is derived from `cargo test --workspace`, the git tree, and the authoritative plans under `docs/plans/`.

---

## Workspace crates

| Crate | Purpose | Notes |
|-------|---------|-------|
| `parser` | LALRPOP grammar + SIMD lexer | Source of the `.hymeko` AST. 76 lexer tests + 4 using-alias tests passing. |
| `hymeko_core` | IR, resolution, module store, tensor primitives, HGNN/mesh ops, traversal, writers | 133 tests passing. Lib crate-name `hymeko`. |
| `hymeko_hre` | Hypergraph Rewriting Engine orchestrator (IR → `TensorCoo` expansions, `iceoryx2` subscriber) | Extracted from `hymeko_core` on 2026-04-18. 2 tests. |
| `hymeko_query` | Predicate / query engine / rewrite / formats / kinematics | 101 tests passing. Includes URDF/SDF/MJCF/DOT/ROS2-launch transforms. |
| `hymeko_daemon` | Worker, IR-CBOR pipeline, shared-memory gates | Depends on `hymeko_core`; does not yet use `hymeko_hre` directly. |
| `hymeko_client` | Subscriber shell | |
| `hymeko_cli` | `emit`/`query`/`compile` entry points + robotics workflow | Consumes `hymeko_query` (`QueryEngine`) and `hymeko_hre` indirectly via Python. |
| `hymeko_py` | PyO3 bindings (`PyHypergraphEngine`, `PyHypergraphIR`, etc.) | Now depends on both `hymeko_core` and `hymeko_hre`. |

## Test status

`cargo test --workspace` → **347 tests passing**, 0 failures, 3 ignored doc-tests.

- `parser`: 76 lexer + 11 using-alias + 15 query-variable + 1 doc-test = 103
- `hymeko_core`: 133 (no engine tests anymore — moved to `hymeko_hre`)
- `hymeko_hre`: 2 (engine registry + hashing load)
- `hymeko_query`: 106 (90 existing + 16 alias-parity incl. SDF/MJCF/per-link-mass)
- `hymeko_cli`: 3 (integration)

## Integrated features

| Area | Feature | Status | Reference |
|------|---------|--------|-----------|
| Language | `node` / `edge` / `arc` declarations | ✅ shipped | parser grammar |
| Language | Signed refs `+ - ~` | ✅ shipped | `SignedRefR` in `hymeko_core::ir` |
| Language | Tag annotations `<tag>` | ✅ shipped | |
| Language | Containment `{ children }` | ✅ shipped | |
| Language | Weight annotations `[[xyz],[rpy]]` on arcs | ✅ shipped | |
| Language | Import `@"path.hymeko";` | ✅ shipped | `ImportStmt` |
| Language | **Namespace alias `using path as alias;`** | ✅ **shipped + tested** | `UsingStmt` + `apply_usings` + 15 tests (4 parser, 11 parity) — 2026-04-18 audit |
| Language | Wildcard `_` in queries | ✅ shipped | interpreter special-cases `"_"` |
| Language | Query variable binding `?x` | 🟡 **partial** (parser surface landed 2026-04-18; not yet consumed by query engine) | T10 — `docs/examples/query_variables.md`, `parser/tests/query_variable.rs` |
| Language | Deep containment `.. { child }` | ❌ planned | |
| Language | Comparison ops `>=` / `<=` | ❌ planned — lexer disambiguation needed vs `<tag>` | |
| Language | Negation prefix `!tagged` | ❌ planned (query layer has `Not` predicate internally) | |
| IR | Arena-backed `Ir` with `DeclId`/`NodeId`/`EdgeId`/`HyperArcId`/`SymId` | ✅ | `hymeko_core::ir::ir` |
| IR | Canonical hash (Blake3) | ✅ | `hymeko_core::ir::hash`, `canonical_hash` |
| IR | Zero-copy IR hand-off via `take_last_ir()` | ✅ | `ModuleStore::take_last_ir` |
| IR | CBOR serialize / deserialize | ✅ | `hymeko_core::writers::cbor_writer`, `HymekoDaemon::deserialize_cbor_ir` |
| Compile | `ModuleStore::compile` pipeline (parse → intern → resolve → lower → apply_usings) | ✅ | |
| Engine | `HypergraphEngine` with node/edge registries + `TensorCoo` builder | ✅ in `hymeko_hre` | Extracted 2026-04-18 |
| Engine | Star expansion `compile_star_expansion_core<F: Real>` | ✅ | |
| Engine | Clique expansion `compile_clique_expansion_core<F: Real>` | ✅ | |
| Engine | `iceoryx2` zero-copy stream via `write_*_into_raw` (feature `ipc`) | ✅ | |
| Tensor | COO, CSR representations | ✅ | `hymeko_core::tensor::representations` |
| Tensor | Deterministic weight init (Van der Corput, Xavier, Kaiming, Zeros/Ones/Constant) | ✅ | `hymeko_core::tensor::conv::weight_init` |
| Tensor | HGNN / signed HGNN / clique GCN / mesh conv | ✅ (in `hymeko_core`, follow-up extraction planned) | |
| Query | `Predicate` enum (17 variants) + `QueryEngine` | ✅ | `hymeko_query::engine` + `predicate` |
| Query | AST-to-predicate interpreter | ✅ | `hymeko_query::interpret` |
| Query | Kinematic model extraction (links, joints, geometries, axes) | ✅ | `hymeko_query::kinematics::kinematic` |
| Query | Template-driven rewrite engine | ✅ | `hymeko_query::rewrite` |
| Transforms | URDF | ✅ | `transforms/urdf/`, `hymeko_query::formats::urdf` |
| Transforms | SDF | ✅ | `transforms/sdf/` |
| Transforms | MJCF (MuJoCo) | ✅ | `transforms/mjcf/` |
| Transforms | DOT (Graphviz) | ✅ | `transforms/dot/` |
| Transforms | ROS2 launch | ✅ | `transforms/ros2_launch/` |
| Transforms | Gazebo world | ❌ planned (T11) | |
| Transforms | Isaac Sim USD | ❌ planned (T12) | |
| Transforms | SysML v2 | ❌ planned (Plan 06 §2) | Ground truth in `docs/examples/hymeko_to_sysmlv2.md` |
| Transforms | Mermaid hypergraph render | ❌ planned | Hand-authored examples in `docs/examples/visualizations.md` |
| Daemon | Worker + IR-CBOR + shared-memory gates | ✅ | `hymeko_daemon::worker` |
| Daemon | `iceoryx2` publish loop | ✅ (feature-gated) | |
| Python | `PyHypergraphEngine`, `PyHypergraphIR`, `PySharedExpansion` | ✅ | Uses `hymeko_hre::HypergraphEngine` as of today |
| Python | Arrow / DLPack zero-copy tensor export | ✅ | `hymeko_py::interface_python::api` |

## In-flight (uncommitted on current branch)

All of the following is staged in the working tree of `refactor/extract-hymeko-hre` but **not yet committed**:

1. `hymeko_hre` crate extraction from `hymeko_core` — engine module moved, consumers updated, cycle avoided by keeping `traversal/` in core.
2. Architecture diagrams: `architecture/overview_crates.mermaid` (new) + refreshed `architecture/hre_rewriting_engine/architecture.mermaid` + `architecture/README.md` index entry.
3. Planning docs `docs/plans/05_hre_extraction/{plan,features}.md` and `docs/plans/06_wasm_editor/outline.md`.
4. Example docs `docs/examples/{visualizations,hymeko_to_sysmlv2}.md`.
5. Changelog `docs/changelog/changelog_20260418.md`.
6. Alias-parity tests in `hymeko_query/tests/test_transform_ecosystem.rs` (new `mod alias_parity`, 11 tests).
7. Parser grammar test in `parser/tests/using_alias.rs` (4 tests).

Commit strategy: 4 commits on `refactor/extract-hymeko-hre` — (a) HRE extraction, (b) diagrams, (c) plans + examples + changelog, (d) alias-parity tests.

## Backlog (task list)

### Near-term

| # | Subject | Size | Owner | Notes |
|---|---------|------|-------|-------|
| 14 | Commit HRE extraction | S | me | Four commits on current branch. |
| 19 / T10 | LALRPOP `?` token for query-variable binding | S | me | 3 one-line edits + grammar rule + test. `docs/plans/04_graph_query/T10_lalrpop_extension.md`. |
| 19 / T13-T16 | Remaining Paper 2 tests | M | — | Meta-kinematics schema queries, alias parity (already partly landed today), robot-specific edge cases. |
| 19 / T11 | Gazebo world transform | M | — | `hymeko_query::formats::gazebo` + `transforms/gazebo/`. |
| 19 / T12 | Isaac Sim USD transform | M | — | `hymeko_query::formats::isaac` + USD primitives. |

### Mid-term — WASM editor stack (Plan 06)

| # | Step | Description |
|---|------|-------------|
| 15 | Plan 06 §1 | `hymeko_ir` extraction design. Decide: slotmap-shaped new IR per spec vs reuse existing arena IR. |
| 16 | Plan 06 §2 | `hymeko_emitter` crate with `emit_hymeko`, `emit_sysml`, `emit_rust_stubs`, `emit_lean4`. SysML output must match the ground truth in `docs/examples/hymeko_to_sysmlv2.md`. |
| 17 | Plan 06 §3-6 | `hymeko_wasm` (wasm-bindgen) → `hymeko_server` (Axum) → `hymeko_mcp` (rmcp) → `hymeko_wire` (CBOR + zstd + xxhash). |

### Long-term — structural follow-ups

| # | Subject | Rationale |
|---|---------|-----------|
| 18 | `hymeko_hnn` extraction | Resolves the cycle noted in `docs/plans/05_hre_extraction/plan.md § Follow-up`: pull `HyperGraphView` + hypergraph-aware tensor ops (`conv/*`, `mesh_nn`, `message_passing`, representations) into their own crate. |
| — | `hymeko_ir` extraction (if Plan 06 §1 goes that route) | Will ripple through all current consumers. |
| — | `hymeko_p2p` | iroh-based P2P gossip layer per spec. |

## Key references

- `docs/plans/04_graph_query/00_MASTER_TRACKER.md` — Paper 2 master tracker (12/23 done pre-today, +15 tests today).
- `docs/plans/05_hre_extraction/plan.md` — today's HRE extraction plan.
- `docs/plans/06_wasm_editor/outline.md` — future WASM editor stack plan, indexing `steps/20260418/hymeko_claude_code_spec.md`.
- `docs/examples/visualizations.md` — visual rendering examples for robotics fixtures.
- `docs/examples/hymeko_to_sysmlv2.md` — T2M workflow with SysML v2 output contract.
- `architecture/overview_crates.mermaid` — crate-level dependency diagram after HRE extraction.
- `CHANGELOG.md` + `docs/changelog/changelog_YYYYMMDD.md` — dated changelogs.
