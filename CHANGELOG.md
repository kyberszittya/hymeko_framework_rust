# Hymeko Framework Changelog

This root changelog summarizes every dated engineering log. Full entries live under `docs/changelog/` for deep dives and diagrams.
## 2026-04-07 — Query Engine, Tensor Initialization, and Compilation Pipeline Expansion
- Finalized the query-engine branch by splitting query/codegen/kinematics modules into the dedicated `hymeko_query` crate (`hymeko_query/src/engine.rs`, `hymeko_query/src/interpret.rs`, `hymeko_query/src/codegen.rs`, `hymeko_query/src/formats/`, `hymeko_query/src/kinematics/`) and wiring matching integration tests under `hymeko_query/tests/codegen/`.
- Reorganized tensor compute surfaces in `hymeko_core/src/tensor/` by modularizing convolution logic (`conv/gcn_clique.rs`, `conv/hgnn.rs`, `conv/signed_hgnn.rs`, `conv/traits.rs`) and adding decomposition + mesh support (`decomposition.rs`, `mesh_nn/mod.rs`).
- Added deterministic and randomizable weight initializer support in `hymeko_core/src/tensor/conv/weight_init/` (`Xavier`, `Kaiming`, `XavierRandom`, `Zeros`, `Ones`, `Constant`, `van_der_corput`) with targeted coverage in `hymeko_core/tests/computations/test_weight_init.rs`.
- Updated parser grammar/token handling (`parser/src/hymeko.lalrpop`, `parser/src/lexer/common.rs`, `parser/src/lexer/token.rs`) to align with the new query/model pipeline.
- Relocated sample datasets from `hymeko_core/data/` to top-level `data/` and refreshed robotics fixtures used by query/codegen scenarios.
- Added the articulated robotics fixture `data/robotics/anthropomorphic_arm.hymeko` with a full link/joint/control graph (revolute chain, limits, control interfaces, and simulation plugin wiring) for richer kinematics/query validation scenarios.
- Added `data/robotics/meta_kinematics.hymeko` as a reusable robotics schema layer covering units, joint archetypes, controller/sensor definitions, axis presets, and control/simulation plugin anchors for consistent model authoring.
- Enhanced `hymeko_core/src/module_store/module_store.rs` with new APIs for ownership-safe IR extraction: `ModuleStore::take_last_ir()` consumes the store to extract owned `Ir` without requiring `Clone`, enabling zero-copy IR hand-offs to daemon/worker threads in `hymeko_daemon/src/worker.rs` (`compile_to_ir_only()`) and Python bindings.
- Wired `ModuleStore::compile()` step 6b to apply using-alias resolution via `apply_usings()`, ensuring all namespace aliases are resolved during compilation before IR lowering.
- Added `compile_to_ir_only()` and `deserialize_cbor_ir()` paths in the daemon to decouple IR compilation from tensor expansion scheduling, allowing precompiled/cached IRs to flow through the query/codegen pipeline.
- Expanded transform ecosystem coverage in `hymeko_query/tests/test_transform_ecosystem.rs` with registry, validation, MJCF/DOT emission, and `using ... as` alias-parity tests across Moveo and differential-drive fixtures (`anthropomorphic_arm*.hymeko`, `robot_4wh*.hymeko`).
- Added end-to-end generation regression coverage in `hymeko_query/tests/test_generation_engine.rs` for kinematic extraction, URDF/SDF generation, cross-format link/joint parity, query predicate edge cases, and `using ... as` fixture equivalence checks.
- Details in [`docs/changelog/changelog_20260407.md`](docs/changelog/changelog_20260407.md).

## 2026-03-11 — Service-Aware Daemon Logging
- Enriched structured logging in `hymeko_daemon/src/worker.rs` with per-request/service context (`service`, `request_id`, `source`, payload-size/timing, enqueue outcomes, and debug `etag_prefix` correlation).
- Improved ingress observability in `hymeko_daemon/src/service.rs` with explicit channel/source labels (`zenoh_utf8`, `zenoh_cbor`, `iceoryx2_src`, `iox_ir`) and branch-specific receive/processing failure logs.
- Added service-scoped lifecycle logs for ingress thread start/stop and channel-close paths in `hymeko_daemon/src/iox_ingress.rs`.
- Refactored `hymeko_daemon/src/iox_ingress.rs` to normalize `RawUtf8`/`CompiledIr`/`CborEncoded` payloads into `ExecutableQuery` IR units before async handoff, with compile/deserialization errors logged and dropped safely.
- Added a standalone `hymeko_client` crate (`hymeko_client/src/main.rs`) to publish src ingress payloads, send event wakeups, and poll tensor egress responses for external smoke validation.
- Synced Task 3.2 evidence in `docs/plans/daemon/checklist_task3.md` to reflect service-aware structured logging coverage, and added Task 3.3 evidence for ingress query normalization progress.
- Details in [`docs/changelog/changelog_20260311.md`](docs/changelog/changelog_20260311.md).

## 2026-03-10 — Data-Plane Traceability & Bridge Closure
- Closed and re-traced Phase 2 daemon checklist work in `docs/plans/daemon/checklist_task2.md`, including the rewritten Task 2.3 direct-memory bridge wording.
- Updated `docs/plans/daemon/checklist_task3.md` to reflect implemented control-plane groundwork (`moka` cache init, `#[tokio::main]`, active Zenoh session + subscriber wiring in `hymeko_daemon/src/service.rs`, heartbeat `tokio::select!`, and `tracing`-based geometric/ascii logging), while keeping async-to-rayon handoff items open.
- Captured the end-to-end shared-memory delivery path across `hymeko_core/src/tensor/shared_state.rs`, `hymeko_daemon/src/main.rs`, and `hymeko_py/src/interface_python/api.rs` (`PySharedExpansion::buffers`), including subscriber gating via `number_of_subscribers()`.
- Traced `hymeko_daemon/src/worker.rs::compute_expansion` as scaffolded for the Tokio-to-Rayon bridge (structure present, execution path still pending).
- Logged fresh random COO benchmark telemetry from `hymeko_core/tests/benchmarks/bench_coo_builder_random.rs::bench_random_hypergraph_coo_builder_suite` to `hymeko_core/target/benchmarks/coo_builder_random_benchmark.csv` (28 rows; `total_ms` range `4.4408..1432.9884`; `ns_per_entry` range `10128.738..249896.226`).
- Captured daemon bootstrap modularization in `hymeko_daemon/src/main.rs`, where `main` now acts as a thin orchestrator for `config::{Args, DaemonConfig}` plus `service::HymekoDaemon::new(config).run().await` with `tracing_subscriber` env-filter setup.
- Extended architecture documentation continuity (`architecture/README.md` + sub-READMEs) and kept README-level navigation/logo touch-ups aligned with the architecture catalog.
- Details in [`docs/changelog/changelog_20260310.md`](docs/changelog/changelog_20260310.md).

## 2026-03-09 — Architecture Catalog & Branding Touches
- Added `architecture/README.md` and per-subfolder READMEs so every Mermaid/SysML diagram is self-documented and linked from a single index.
- Refreshed `README.md` with the repo logo plus a dedicated Architecture section that jumps straight to the new catalog for control/data-plane context.
- Details in [`docs/changelog/changelog_20260309.md`](docs/changelog/changelog_20260309.md).

## 2026-03-08 — Arrow Schemas for Tensor Expansions
- Added `hymeko_core/src/tensor/arrow_schema.rs`, providing `schema_expansion_3d` and `schema_expansion_2d` helpers so every crate (daemon, Python, analytics) can lock onto the same Arrow layouts for zero-copy tensor sharing.
- Checked off Task 2.2 inside `docs/plans/daemon/checklist_task2.md`, capturing that both `hymeko_core` and `hymeko_py` now depend on Arrow and that the canonical schemas live in one module for Task 2.3 to consume.
- Rewrote Task 2.3 in `docs/plans/daemon/checklist_task2.md` as **The Direct Memory Bridge**, spelling out the raw-pointer expansion hooks on the Rust side and the `pyarrow.foreign_buffer` wiring required in `hymeko_py/src/interface_python/api.rs` (including the `PySharedExpansion` scaffold) for zero-copy PyTorch ingestion.
- Landed `HypergraphEngine::write_star_expansion_into_raw` plus the updated `PySharedExpansion::buffers`, so an `iceoryx2` sample can now be filled and consumed via contiguous `[k,i,j,val]` buffers without intermediate allocations.
- `hymeko_daemon` loans `[u8]` slices from iceoryx2, writes the `ExpansionHeader + COO` payload with `HypergraphEngine::write_tensor_into_raw`, and publishes a frame on every tick where subscribers are attached, giving Task 2.1 a concrete streaming path.
- `PySharedExpansion::buffers` now hands `pyarrow.foreign_buffer` the `PySharedExpansion` object itself as the owner so the shared memory lifetime matches the Python handles.
- Noted in the checklist that `hymeko_daemon` currently advertises a raw `[u8]` slice on its publish/subscribe service so Task 2.3 can graft the typed zero-copy bridge without blocking the keepalive loop.
- Recorded follow-on work for the translation layer and iceoryx bridge inside [`docs/changelog/changelog_20260308.md`](docs/changelog/changelog_20260308.md).

## 2026-03-07 — Workspace-Wide CI & Coverage Flags
- Rebuilt `.github/workflows/ci.yml` so every crate (`hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, `parser`) now has its own cache-aware build/test matrix plus per-crate Tarpaulin uploads feeding Codecov flags.
- Expanded `codecov.yml`, `CI_CD_DOCUMENTATION.md`, `CODE_COVERAGE.md`, and `README_CICD.md` to document the new reports, HTML artifacts, and flag-driven targets.
- Normalized the `hymeko_core/tests/minimal_tests` suite by hoisting fixture strings, node names, and weight tables into `constants.rs`, adding the shared `helpers` module, and updating traversal/tensor/edge/annotation/module-store/smoke tests to import those definitions (plus new assertions for edges, IR lowering, and HyperItem variants).
- Normalized the rest of `hymeko_core`'s test suites (`tests/traversal`, `tests/intermediate_tests`, `tests/domain_transformations`, `tests/aggregations`) by hoisting fixture literals into shared modules, wiring every case through `log_test_header`/`log_test_footer`, and adding targeted `info!` summaries; re-ran `cargo test -p hymeko_core --tests -- --nocapture` to confirm the instrumentation passes.
- Hardened `hymeko_core/src/ir/lower.rs` so arcs get their own anonymous `DeclId`, inherit annotations via `resolve_anno`, and are linked into both the parent edge's decl-child chain and its per-edge arc list, ensuring downstream traversals and IR consumers see consistent metadata.
- Slimmed `hymeko_py/src/interface_python/api.rs` so the Python bindings now delegate star/clique tensor expansion work straight to `HypergraphEngine::compile_*_core`, keeping the bindings as thin zero-copy wrappers (Arrow exports only) for Task 1.2's FxHash acceleration effort.
- Adopted `rustc_hash::FxHashMap` throughout `hymeko_core/src/engine/hypergraphengine_impl.rs` (`node_registry`, `edge_registry`, `ir_repository`, plus the CSR sync helper) to remove the last `std::collections::HashMap` usages inside the core engine, allowing us to mark Task 1.2 fully complete.
- Completed Task 1.3’s deterministic hashing work by backing `Index::by_path` with a `BTreeMap` and updating `hymeko_core/src/ir/hash.rs` plus `hymeko_core/src/ir/canonical_hash.rs` to stream directly from that ordered iterator (no manual key sorting).
- Hardened `hymeko_core/tests/hash/hashing_test.rs` by hoisting all symbol/workload literals into module constants and asserting both total and per-run hashing latency stay within the CI budgets.
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
