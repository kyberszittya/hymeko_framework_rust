# Hymeko Framework Changelog

This root changelog summarizes every dated engineering log. Full entries live under `docs/changelog/` for deep dives and diagrams.

## 2026-03-07 — Workspace-Wide CI & Coverage Flags
- Rebuilt `.github/workflows/ci.yml` so every crate (`hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, `parser`) now has its own cache-aware build/test matrix plus per-crate Tarpaulin uploads feeding Codecov flags.
- Expanded `codecov.yml`, `CI_CD_DOCUMENTATION.md`, `CODE_COVERAGE.md`, and `README_CICD.md` to document the new reports, HTML artifacts, and flag-driven targets.
- Normalized the `hymeko_core/tests/minimal_tests` suite by hoisting fixture strings, node names, and weight tables into `constants.rs`, adding the shared `helpers` module, and updating traversal/tensor/edge/annotation/module-store/smoke tests to import those definitions (plus new assertions for edges, IR lowering, and HyperItem variants).
- Normalized the rest of `hymeko_core`'s test suites (`tests/traversal`, `tests/intermediate_tests`, `tests/domain_transformations`, `tests/aggregations`) by hoisting fixture literals into shared modules, wiring every case through `log_test_header`/`log_test_footer`, and adding targeted `info!` summaries; re-ran `cargo test -p hymeko_core --tests -- --nocapture` to confirm the instrumentation passes.
- Hardened `hymeko_core/src/ir/lower.rs` so arcs get their own anonymous `DeclId`, inherit annotations via `resolve_anno`, and are linked into both the parent edge's decl-child chain and its per-edge arc list, ensuring downstream traversals and IR consumers see consistent metadata.
- Slimmed `hymeko_py/src/interface_python/api.rs` so the Python bindings now delegate star/clique tensor expansion work straight to `HypergraphEngine::compile_*_core`, keeping the bindings as thin zero-copy wrappers (Arrow exports only) for Task 1.2's FxHash acceleration effort.
- Adopted `rustc_hash::FxHashMap` throughout `hymeko_core/src/engine/hypergraphengine_impl.rs` (`node_registry`, `edge_registry`, `ir_repository`, plus the CSR sync helper) to remove the last `std::collections::HashMap` usages inside the core engine, allowing us to mark Task 1.2 fully complete.
- Details captured in [`docs/changelog/changelog_20260307.md`](docs/changelog/changelog_20260307.md).

## 2026-03-06 — Tensor Grid Telemetry & PathID Notes
- Introduced `py/coo_tensor/coo_tensor_grid_eval.py`, a PyTorch-facing grid benchmark that sweeps node/edge counts, records per-iteration parse and tensor build timings, and emits timestamped CSVs (summary + raw) for later analysis.
- Added a parser layout FAQ plus a benchmark harness overview to `README.md`, and documented why the `parser/` crate should remain nested within the workspace instead of moving to the root package.
- Captured the recent `PathKey`/`PathID` borrowable-slice work, the double-ended `DeclNode` sibling tracking, and the new measurement tooling inside [`docs/changelog/changelog_20260306.md`](docs/changelog/changelog_20260306.md).

## 2026-03-05 — Serialization & Dataset Push
- Portable CBOR snapshots through the `CborPayload` wrapper, serde derives across IDs and IR nodes, CSR tensor determinism, and curated `.hymeko` benchmark datasets.
- See [`docs/changelog/changelog_20260305.md`](docs/changelog/changelog_20260305.md) for the full breakdown.

## 2026-03-02 — Cybernetic State Compiler Foundations
- Zero-copy PyO3 bridge upgrades, the dual-frequency telemetry loop, Hyper-KA mathematical framing, and the long-horizon publication roadmap.
- Details remain in [`docs/changelog/changelog_20260302.md`](docs/changelog/changelog_20260302.md).
