# Changelog Index

This root changelog tracks every dated log stored in `docs/changelog/` and provides a short summary so you can jump straight to the details you need. Each entry links to the full write-up.

## 2026-04-07 — Query Engine Split, Tensor Modules, and Weight Initializers
- **Link:** [changelog_20260407.md](changelog_20260407.md)
- **Highlights:** Finalized the query-engine branch by moving query/codegen/kinematics workflows into `hymeko_query/src/` (plus new helpers/tests), reorganized tensor conv/decomposition surfaces in `hymeko_core/src/tensor/`, introduced deterministic weight initialization in `hymeko_core/src/tensor/conv/weight_init/` with coverage in `hymeko_core/tests/computations/test_weight_init.rs`, updated parser plus sample-data layout (`hymeko_core/data/` -> `data/`), added robotics fixtures (`data/robotics/anthropomorphic_arm.hymeko`, `data/robotics/meta_kinematics.hymeko`) for articulated-model and reusable kinematics schema coverage, enhanced `ModuleStore` with ownership-safe IR extraction APIs (`take_last_ir()`, `compile_to_ir_only()`, `deserialize_cbor_ir()`) plus namespace alias resolution during compilation for decoupled IR/tensor scheduling, expanded `hymeko_query/tests/test_transform_ecosystem.rs` for registry/validation/MJCF/DOT + alias parity, and added `hymeko_query/tests/test_generation_engine.rs` for extraction/generation/cross-format regression coverage.

## 2026-03-12 — Multiplexed Daemon Egress and Worker Refactor
- **Link:** [changelog_20260312.md](changelog_20260312.md)
- **Highlights:** Added multiplexed daemon dispatch in `hymeko_daemon/src/service.rs` (compiled IR + star tensor + clique tensor publishers), centralized graph-name derivation in `hymeko_daemon/src/worker.rs` via `graph_name_from_ir`, fixed Arrow serialization argument/hash type mismatches, and synced Phase 3 task evidence in `docs/plans/daemon/checklist_task3.md`.

## 2026-03-11 — Service-Aware Daemon Logging
- **Link:** [changelog_20260311.md](changelog_20260311.md)
- **Highlights:** Enriched daemon tracing with service-aware structured context across `hymeko_daemon/src/worker.rs`, `hymeko_daemon/src/service.rs`, and `hymeko_daemon/src/iox_ingress.rs` (`service`, `request_id`, ingress source labels, payload/timing metadata), introduced ingress format normalization to `ExecutableQuery` (`RawUtf8`/`CompiledIr`/`CborEncoded`) in `hymeko_daemon/src/iox_ingress.rs`, added the standalone `hymeko_client/src/main.rs` Iceoryx ingress/egress smoke harness, and synced Task 3.2/3.3 evidence in `docs/plans/daemon/checklist_task3.md`.

## 2026-03-10 — Data-Plane Traceability & Bridge Closure
- **Link:** [changelog_20260310.md](changelog_20260310.md)
- **Highlights:** Closed and traced Phase 2/3 daemon checklist updates (`docs/plans/daemon/checklist_task2.md`, `docs/plans/daemon/checklist_task3.md`), captured subscriber-gated `ExpansionHeader + COO` publishing, confirmed active Zenoh subscriber wiring in `hymeko_daemon/src/service.rs`, migrated daemon runtime output to structured geometric/ascii logging, modularized `hymeko_daemon/src/main.rs` into a thin `config` -> `service` bootstrap, recorded `worker.rs` Tokio-to-Rayon bridge status as scaffolded, and logged random COO suite telemetry in `hymeko_core/target/benchmarks/coo_builder_random_benchmark.csv`.

## 2026-03-09 — Architecture Catalog & Branding Touches
- **Link:** [changelog_20260309.md](changelog_20260309.md)
- **Highlights:** Documented every diagram under `architecture/` with inline Mermaid/SysML snippets (plus per-folder READMEs) and refreshed the repository README with the logo plus a direct Architecture section.

## 2026-03-08 — Arrow Schemas for Tensor Expansions
- **Link:** [changelog_20260308.md](changelog_20260308.md)
- **Highlights:** Centralized the Arrow schemas for 3D star/clique and 2D projected expansions in `hymeko_core/src/tensor/arrow_schema.rs`, marked Task 2.2 complete, and outlined the follow-on work for the translation layer.

## 2026-03-07 — Workspace-Wide CI & Coverage Flags
- **Link:** [changelog_20260307.md](changelog_20260307.md)
- **Highlights:** Matrixed workspace tests, per-crate Tarpaulin uploads with Codecov flags, documentation refresh (`CI_CD_DOCUMENTATION.md`, `CODE_COVERAGE.md`, `README_CICD.md`), and the daemon checklist update confirming CI parity.

## 2026-03-06 — Tensor Grid Telemetry & PathID Notes
- **Link:** [changelog_20260306.md](changelog_20260306.md)
- **Highlights:** COO tensor grid benchmarks across star/clique expansions, parser layout guidance, captured PathKey/DeclNode hygiene updates, and recorded the maturin troubleshooting notes alongside the new measurement harnesses.

## 2026-03-05 — Serialization & Dataset Push
- **Link:** [changelog_20260305.md](changelog_20260305.md)
- **Highlights:** Introduced the `CborPayload` wrapper plus `PyHypergraphIR::to_cbor / from_cbor` APIs for portable IR snapshots, derived `serde` support across IDs/IR structures, stabilized CSR tensor coalescing, and published new math notes plus curated benchmark `.hymeko` suites.

## 2026-03-02 — Cybernetic State Compiler Foundations
- **Link:** [changelog_20260302.md](changelog_20260302.md)
- **Highlights:** Documented the zero-copy PyO3 bridge, dual-frequency architecture for telemetry, Hyper-KA mathematical formalization (including NURBS activations), long-horizon roadmap, and publication strategy framing the engine as a cybernetic state compiler.

## Python Packaging Integration

- CI/CD now builds and tests Python packages using maturin.
- Python wheels are uploaded as artifacts and optionally published to PyPI.
- See `hymeko_py` crate and workflow YAML files for details.
