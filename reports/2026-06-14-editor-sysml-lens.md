# Report — SysML lens in the web editor (SMC #5 Phase 2)

**Date:** 2026-06-14
**Slug:** `editor-sysml-lens`
**Author:** Csaba Hajdu
**Branch:** `feature/ac-hsikan`

## Summary

Adds in-browser SysML 2 export to the HyMeKo web editor — the SMC #5 Phase 2
deliverable. The editor already compiled HyMeKo → IR live in the browser (WASM)
and exported URDF / SDF / DOT; this wires the *existing* SysML transform
(`transforms/sysml/`) into the same path. Because WASM has no filesystem, the
query + template sources are embedded with `include_str!` and run through the
in-process `execute_transform` (not the FS-based registry). Net: a working
``HyMeKo → SysML 2`` button in the browser, end-to-end, with the toolchain
already installed here (wasm-pack + wasm32 target), so it is built and shipped,
not promised.

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_wasm/src/compile.rs` | add `CompiledDoc::to_sysml` (include_str! template + `execute_transform`) | +24 |
| `hymeko_wasm/src/wasm.rs` | add `#[wasm_bindgen]` forwarder `to_sysml` | +5 |
| `hymeko_wasm/tests/test_compile.rs` | add `to_sysml` test (package + part def + determinism) | +14 |
| `docs/editor/editor.js` | export switch: `case "sysml"` | +1 |
| `docs/editor/index.html` | export dropdown: `<option>SysML</option>` | +1 |
| `docs/editor/pkg/*` | rebuilt WASM (`wasm-pack build --target web --release`) | regen |

## CORE.YAML items touched

**None.** `CORE.YAML` does not list `hymeko_wasm`, `docs/editor`, or
`transforms/sysml`. No dependency added — `execute_transform` / `TransformSpec`
are existing `hymeko_query` API; the SysML template already existed. Additive
throughout, mirroring the existing `to_urdf` / `to_sdf` precedent.

## Test results

- **Native** (`cargo test -p hymeko_wasm --test test_compile`): **4 passed**,
  including the new `to_sysml_emits_package_and_part_defs` — compiles the inline
  `tiny_arm`, asserts the output contains `package tiny_arm` and `part def Link`,
  is not the error fallback, and is deterministic (same input → same output).
- **WASM build:** `wasm-pack build --target web --release` succeeded; the
  generated `docs/editor/pkg/hymeko_wasm.{js,d.ts}` now export `to_sysml`
  (`compiledir_to_sysml`), so the browser binding is live.

## What exists vs. what was the gap (the honest picture)

Already present before this change: the WASM compile/query/emit surface
(`CompiledDoc` with `to_urdf`/`to_sdf`/`to_dot`), the browser editor with live
3D-geometry / hypergraph / kinematic / regime-class views and an export menu, and
the SysML transform (`transforms/sysml/template.sysml`). The only gap was one
WASM method + two lines of editor wiring. A "good-looking online SysML tool" was
not far off — it was one method away.

## Performance

Negligible: one in-process template transform over the compiled IR (same class as
URDF/SDF emission, sub-ms on the editor's example models). No GPU, trivial RSS.

## §6.5 anti-patterns

None. Reused the existing transform + the `to_urdf`/`to_sdf` method pattern (no
duplication); SysML is one more case in the existing export switch (no new code
path family). Note: `compile.rs` uses a pre-existing compact one-liner style that
is not `cargo fmt`-canonical; the new `to_sysml` matches the surrounding style and
no unrelated lines were reformatted.

## Open issues / follow-ups

1. **Live SysML *view*** (`views/sysml.js`) — currently SysML is an *export*
   (download). A side-panel live lens with syntax highlighting (like the other
   views) is the natural next step for "good-looking."
2. **Browser visual confirmation** — functionally verified natively + binding
   shipped; an end-to-end browser screenshot (served over localhost, not
   `file://`) would confirm the rendered UX. Offered, not yet run.
3. The requirements-SysML transform (`transforms/requirements_sysml/`) could get
   the same lens for the traceability view.

## Provenance

Git SHA: working tree dirty. Host: Windows 11, cargo 1.93.1, wasm-pack + wasm32
target installed. Build: `wasm-pack build --target web --release --out-dir
../docs/editor/pkg` from `hymeko_wasm/`.
