# Hymeko Framework Changelog

This root changelog summarizes every dated engineering log. Full entries live under `docs/changelog/` for deep dives and diagrams.

## 2026-03-07 — Workspace-Wide CI & Coverage Flags
- Rebuilt `.github/workflows/ci.yml` so every crate (`hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, `parser`) now has its own cache-aware build/test matrix plus per-crate Tarpaulin uploads feeding Codecov flags.
- Expanded `codecov.yml`, `CI_CD_DOCUMENTATION.md`, `CODE_COVERAGE.md`, and `README_CICD.md` to document the new reports, HTML artifacts, and flag-driven targets.
- Normalized the `hymeko_core/tests/minimal_tests` suite by hoisting fixture strings, node names, and weight tables into `constants.rs`, adding the shared `helpers` module, and updating traversal/tensor/edge/annotation/module-store/smoke tests to import those definitions (plus new assertions for edges, IR lowering, and HyperItem variants).
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
