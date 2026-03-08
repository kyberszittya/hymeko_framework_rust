# Project Changelog — 2026-03-07

## Workspace-Wide CI Matrix & Coverage Flags
- Replaced the monolithic CI job with a `workspace-tests` matrix (multi-OS, stable + nightly) plus a Linux-only `package-check` matrix that builds/tests `hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, and `parser` independently.
- Added caching keyed by toolchain and package, ensuring each crate reuses artifacts without starving the others.
- Kept the release build job but gated it behind the matrix tests so artifacts only publish after the expanded checks pass.

## Per-Crate Coverage Pipeline
- Reworked the coverage job to loop over every crate, generating XML + HTML Tarpaulin outputs under `coverage/xml/<crate>.xml` and `coverage/html/<crate>.html`.
- Uploaded each XML to Codecov with distinct flags (`hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, `parser`) and retained the HTML directory as a downloadable artifact for manual inspection.
- Updated `codecov.yml` with the new flags, per-crate targets (parser 70%, core 65%), and clarified how the status checks map to the CI uploads.

## Documentation & Checklist Updates
- Refreshed `CI_CD_DOCUMENTATION.md`, `CODE_COVERAGE.md`, and `README_CICD.md` to explain the per-crate jobs, artifact layout, and local replication commands.
- Marked the Task 1.1 CI/CD checkbox in `docs/plans/daemon/checklist_task1.md`, confirming the daemon crate now enjoys first-class CI coverage.

## Minimal Tests Fixture Consolidation
- Expanded `tests/minimal_tests/constants.rs` with every path/name/weight used across the suite (module-store fixtures, tag-annotation nodes, smoke-test weights, edge expectations) and introduced the shared `helpers` module for field assertions.
- Refactored the traversal/tensor/edge/annotation/module-store/smoke tests to import those fixtures, eliminating inlined strings, verifying flattened weight vectors, and adding new sanity checks for HyperItem variants and DeclId lookups.
- Re-ran the focused suites via `cargo test minimal_tests::test_module_store::mod_test_module_store minimal_tests::test_smoke_test minimal_tests::annotations::test_annotations minimal_tests::edges::test_ref_values --color never` to confirm the new helpers and assertions behave as expected.

## Core Test Telemetry & Logging Cleanup
- Applied the same constant-hoisting + logging strategy to `tests/traversal`, `tests/intermediate_tests`, `tests/domain_transformations`, and `tests/aggregations`, including a new `traversal::constants` module and start/finish helpers that wrap every case with elapsed-time output.
- Added contextual `info!` statements (node/edge counts, hash digests, inheritance tallies, aggregator outcomes) so CI logs explain what each suite verified without drowning tensor dumps.
- Re-ran the full integration suite with `cargo test -p hymeko_core --tests -- --nocapture` to validate the instrumentation and give Codecov a consistent signal.

## IR Lowering: Dedicated Arc Decls
- Updated `hymeko_core/src/ir/lower.rs` so each arc allocates an explicit `DeclId`, copies its annotation via `resolve_anno`, and pushes a single `ArcRec` into both the parent edge's decl-child chain and its `EdgeRec::arcs` list.
- The helper now uses `resolve_arc_refs` once, stores the `HyperArcId` mapping in `decl_to_arc`, and guarantees downstream traversals (e.g., `ir.decl_children` + per-edge arc scans) observe the same metadata without re-resolving the AST.

## Python API Slimming for FxHash Integration
- Reworked `hymeko_py/src/interface_python/api.rs` so `PyHypergraphEngine` simply forwards star/clique tensor expansion calls to `HypergraphEngine::compile_star_expansion_core` / `compile_clique_expansion_core`, leaving Python responsible only for Arrow-backed wrappers and serialization.
- This keeps the Python surface area thin (no duplicate tensor loops), aligning Task 1.2 with a single Rust implementation that can be benchmarked independently of the bindings.

## Hypergraph Engine FxHash Adoption
- Migrated `hymeko_core/src/engine/hypergraphengine_impl.rs` so `HypergraphEngine` stores its registries (`node_registry`, `edge_registry`, `ir_repository`) and CSR sync maps in `rustc_hash::FxHashMap`, eliminating the last runtime `HashMap` bottlenecks in the inner loop.
- The constructor now seeds `TensorCoo` metadata, keeps the node/edge name caches, and exposes fast `get_or_create_*` helpers that rely on FxHash for deterministic, accelerated lookups ahead of Task 1.2’s remaining CSR swaps.
- With the Python wrapper simplified and the CSR helpers on FxHash, Task 1.2 (“FxHash Integration”) is fully checked off in `docs/plans/daemon/checklist_task1.md`.
