# Report: rust-analyzer workspace memory settings

**Date:** 2026-06-08  
**Goal:** Make the IDE usable by cutting rust-analyzer RSS on the 24-crate workspace.

## Summary

Added `.vscode/settings.json` with rust-analyzer and file-watcher limits: no check-on-save, no proc-macro expansion, no build-script analysis, no all-targets, smaller LRU, excluded non-Rust trees. Optional `linkedProjects` lines are commented for per-crate scoping.

## Files touched

| File | Change |
|------|--------|
| `.vscode/settings.json` | Created (+35 lines) |
| `reports/2026-06-08-rust-analyzer-memory-settings.md` | This report |

## CORE.YAML items touched

None.

## Test results

N/A (IDE configuration only).

## Performance results

Not measured in-session. Expected: materially lower RA RSS after **Developer: Reload Window** or `rust-analyzer: Restart server`. If still high, uncomment one `linkedProjects` line for the crate you are editing.

## Trade-offs

- `buildScripts: false` — run `cargo build -p parser` (or full workspace build) once so generated `OUT_DIR` sources exist; RA may show stale errors on `parser` / `hymeko_compute` until then.
- `procMacro: false` — PyO3 / derive macros not expanded in-editor; `cargo check` remains authoritative.
- `checkOnSave: false` — no inline errors until manual `cargo check` or RA restart with check re-enabled.

## §6.5 anti-patterns

None introduced.

## Follow-up

If RSS stays above ~2 GB with defaults applied, enable exactly one `linkedProjects` entry for the active crate.

## Update (fresh-boot behaviour)

Set `rust-analyzer.enable: false` so opening the workspace after system boot does not immediately spawn RA and index all 24 crates (~3.5 GB before any edit). Start with Command Palette → `rust-analyzer: Start server` when editing Rust.
