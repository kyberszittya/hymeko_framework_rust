# Editor: 3D pan in the Kinematic view

**Date:** 2026-06-16 · **Persona:** Aiko · quick-win (BACKLOG P3)

## Summary

Ported the `panCamera` pattern from `docs/editor/views/hypergraph3d.js` to the
kinematic robot view, which was orbit/zoom only. Right-button **or** shift+drag
now translates the look-at target in the camera's screen plane (scaled by
distance so it feels consistent at any zoom); plain drag still orbits, wheel
still zooms. `contextmenu` is suppressed so the right-drag doesn't pop a menu.

The kinematic view's `target` is a `THREE.Vector3` (vs hypergraph3d's plain
object), so the port uses `subVectors` / `crossVectors` / `addScaledVector`
instead of manual component arithmetic; the math is identical (right = f×up,
up = right×f, `k = radius·0.0016`).

## Files touched

- `docs/editor/views/kinematic.js` — `panning` state; right/shift-drag routing in
  `bindPointer`; new `panCamera(dx, dy)`.
- `docs/editor/editor.js` — cache-bust `kinematic.js?v=19 → v=20`.

No CORE.YAML items; no dependency; no Rust/WASM rebuild needed (pure view JS).

## Verification

- `node --check views/kinematic.js` — clean.
- `node --test views/*.test.mjs` — **66 pass / 0 fail** (no regression).

**Test-coverage note (declared per §3).** The pan is three.js + DOM interaction
code. The editor's `node --test` suite only exercises *pure* view logic
(adapters, generators, arcs, examples); the 3D modules import `THREE`, which is
not available under `node`, so they are not unit-tested in this repo — the same
reason the original `hypergraph3d.js` `panCamera` has no unit test. This change
is verified by syntax-check, by exact parity with that proven implementation, and
by the unchanged pure-test suite; visual confirmation is a manual step in the
browser (shift-drag the robot view).

## Open / follow-up

- None for this item. (The other editor follow-ups — shareable `?src=` deep-link,
  more generator families — are separate BACKLOG entries; the latter is now
  gated behind a `hymeko_core` edit since generators moved into the core crate.)
