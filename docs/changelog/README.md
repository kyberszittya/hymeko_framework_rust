# Changelog Index

This root changelog tracks every dated log stored in `docs/changelog/` and provides a short summary so you can jump straight to the details you need. Each entry links to the full write-up.

## 2026-03-10 — Data-Plane Traceability & Bridge Closure
- **Link:** [changelog_20260310.md](changelog_20260310.md)
- **Highlights:** Closed and traced Phase 2/3 daemon checklist updates (`docs/plans/daemon/checklist_task2.md`, `docs/plans/daemon/checklist_task3.md`), captured subscriber-gated `ExpansionHeader + COO` publishing, migrated daemon runtime output to structured geometric/ascii logging, and recorded `worker.rs` Tokio-to-Rayon bridge status as scaffolded.

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
