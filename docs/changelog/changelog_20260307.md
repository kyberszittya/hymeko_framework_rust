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

