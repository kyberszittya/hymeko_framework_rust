# Project Changelog — 2026-03-10

## Daemon Data-Plane Task Closure
- Completed and re-traced `docs/plans/daemon/checklist_task2.md` so Phase 2 now reflects the delivered direct-memory path end-to-end.
- Marked Task 2.1 as implemented with the daemon loop publishing `ExpansionHeader + COO` payloads via shared-memory loaned slices.
- Kept Task 2.3 wording focused on the concrete bridge contract (raw pointer writes in core + `pyarrow.foreign_buffer` offsets in Python).

## Zero-Copy Bridge Runtime Notes
- `hymeko_core/src/tensor/shared_state.rs` remains the canonical layout boundary for `ExpansionHeader` and transport payload framing.
- `hymeko_daemon/src/main.rs` publishes the payload as a contiguous shared-memory frame for subscribers.
- `hymeko_py/src/interface_python/api.rs` (`PySharedExpansion::buffers`) exports `(k, i, j, val)` as foreign buffers while tying ownership to the Python object for lifetime safety.

## Architecture Documentation Consolidation
- Expanded `architecture/README.md` with a stronger layer-oriented view and links to per-domain diagram docs.
- Added/extended sub-READMEs under `architecture/` so Mermaid and SysML assets are documented in place.
- Continued README polish (`README.md`) to surface the architecture index and logo-first navigation from the project landing page.

## Follow-ups
- If CI starts compiling `iceoryx2` from source in additional targets, add an explicit toolchain note (including `clang`) in CI docs and pin it in the next changelog entry.

