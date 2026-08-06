# Editor: multi-file imports + selectable vocabulary profiles

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-profiles-imports/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Two coupled gaps closed in the WASM editor:

1. **Importing `.hymeko` files "from the space."** The editor compiled a single
   in-memory source, so `@"meta_x.hymeko"` includes could not resolve. Added a
   multi-file compile path: a new Rust `compile_sources(root, &[(name, src)])`
   over the existing `MemProvider`, exposed as the WASM binding
   `parse_and_compile_files(root, files_json)`. The editor now keeps a compile
   **space** (root + auxiliary meta files) so meta vocabularies live outside the
   current context and are `@"..."`-imported.

2. **"The robot profile is used indefinitely."** The palette/default model was
   hard-wired to the kinematics `kit`. A **profile** (a meta-vocabulary) is now
   selectable: a data-driven `PROFILES` registry where each profile ships its
   meta file(s) (loaded into the space), a root template that imports them, and
   its own palette kinds. The palette is rebuilt generically from the active
   profile — no per-profile UI code. Shipped profiles: **Kinematics (robot)**
   (single-file, `kit` inlined), **HRI coalition** (imports `meta_hri.hymeko`),
   **SysML trace** (imports `meta_sysml_trace.hymeko`).

The old kit-specific `addLink`/`addJoint` collapsed into generic
`addNodeDecl`/`addEdgeDecl` driven by palette descriptors (node → `name: base {…}`;
edge → `@name[: + <isa>] base { (+P,-C); }`).

## Files touched

Rust (non-core):
- `hymeko_wasm/src/compile.rs` — `compile_sources` (multi-file); `compile_source`
  now delegates to it.
- `hymeko_wasm/src/wasm.rs` — `parse_and_compile_files(root, files_json)` binding
  (`files_json` = `{name: source}`; `serde_json` already a dep — **no new dep**).
- `hymeko_wasm/tests/test_compile.rs` — new `multi_file` module (4 tests).

New (editor + fixtures):
- `docs/editor/views/profiles.js` — `PROFILES`, `profileById`, `ROOT_NAME`,
  `EMBEDDED_FILES`. Kinematics root reuses `examples.js` (no duplication).
- `docs/editor/views/profiles.test.mjs` — registry integrity + embed ≡ fixture
  + "root imports every meta it ships" (6 tests).
- `data/profiles/{meta_hri,meta_sysml_trace,hri_cell,sysml_cell}.hymeko` —
  compact editor-tailored vocabularies + profile roots (the canonical source the
  Rust test compiles and `profiles.js` embeds).

Edited (editor):
- `editor.js` — compile `space`; multi-file `recompile` (feature-detected, falls
  back to single-file); `setProfile`; profile `<select>`; data-driven palette
  (`rebuildPalette`/`addKind`); examples + generator reset the space;
  `?profile=` deep-link.
- `index.html` — profile selector; dynamic `#paletteAdd` container; `?v=22`.
- `docs/editor/pkg/*` — **regenerated** (`wasm-pack build --target web --release`).

## CORE.YAML items touched

**None.** `hymeko_wasm` is not a CORE crate; `compile_sources` only *consumes*
the existing public `ModuleStore`/`MemProvider` API of `hymeko_core` (no edit to
the locked crate). `tests/**`, `data/**` (not in any glob), and `docs/editor/**`
are non-core. **No new dependency.** The `pkg/` rebuild uses the `wasm_build`
tool already approved in `tools.yaml` (2026-06-12).

## Test results

- **Rust (`cargo test -p hymeko_wasm --test test_compile`):** 11 pass / 0 fail.
  New `multi_file`: include resolves across the space (and *fails* without the
  meta file — proving the import is real); editor profile roots compile with
  their metas; missing include errors (no panic); single-file ≡ multi-file(1).
- **JS (`node --test`, all `views/*.test.mjs`):** 54 pass / 0 fail (6 new in
  `profiles.test.mjs`).
- **Browser (headless Chrome, `?profile=hri`):** Profile selector = "HRI
  coalition"; palette rebuilt to + Human / + Robot / + Relation; graph compiles
  (24 nodes) with `hri.*` refs resolved from the imported `meta_hri.hymeko`;
  source shows `@"meta_hri.hymeko"; using hri_meta as hri;`. No errors.

## Performance results

Multi-file compile of a root + 1 meta is the same order as single-file
(sub-millisecond to low-ms, per the 2026-06-15 measurement). No hot path; peak
RSS negligible (≪ 16 GB cap). WASM release rebuild: ~26 s wall.

## Static analysis / health

- `cargo clippy -p hymeko_wasm --lib --tests -- -D warnings`: clean.
  `rustfmt --check` on the three Rust files: clean.
- **No §6.5 anti-patterns.** Strategy/registry profiles (data-driven palette,
  not per-kind markup — collapses the old `addLink`/`addJoint` Cartesian start);
  the kinematics root reuses `examples.js` (no duplication); `compile_source`
  now a thin wrapper over `compile_sources` (DRY); no globals; string profile
  bases stay at the source-generation boundary.
- Feature-detection (`hasMultiFile`) means a stale `pkg/` degrades to single-file
  rather than breaking.

## New / removed dependencies

None.

## Open issues / follow-ups

- **Arc values cannot be set (user request, queued next):** the properties panel
  still can't edit arc endpoints/signs or the bracketed arc transforms
  (`[[…],[…]]`). Separate editing feature (string mutation on the edge body),
  independent of this change.
- A full file-space *panel* (add/remove/upload arbitrary aux files, edit a meta
  in a second buffer) is a natural extension; this change ships the profile-driven
  space + import resolution that such a panel would build on.
- More profiles (P-graph `meta_pgraph`, anatomy, NN) are each one registry entry
  + a compact meta under `data/profiles/`.

## Experiment provenance

Not an experiment. Toolchain: rustc/cargo stable, wasm-pack 0.15.0,
wasm32-unknown-unknown, node v24.14.0, Chrome (headless verify), MiKTeX pdflatex
(plan). Working tree dirty from prior session work unrelated to this change.
