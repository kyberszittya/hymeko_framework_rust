# Editor 2D view: bidirectional relations

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-bidirectional-2d/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

In the 2D Cytoscape graph, a **bidirectional relation** — a reciprocal pair where
both `X→Y` and `Y→X` are drawn (mutual edge-arc references, mutual `<isa>`) — used
to render as two overlapping one-way arrows. Now such pairs are detected and
styled as a **double connector**: an arrowhead on *both* ends (accent colour) and
bezier curving so the two members bow apart and stay distinct. This satisfies
both options the user offered ("edges on both sides" and "double relationships").

## Files touched

- `docs/editor/views/adapters.js` — new pure `bidirectionalEdgeIds(edges)` → the
  set of edge ids whose reverse `(target→source)` also exists (self-loops
  excluded, one-way never flagged).
- `docs/editor/views/adapters.test.mjs` — unit test.
- `docs/editor/editor.js` — `renderGraph` tags reciprocal arc-ref/isa edges with
  class `bidir`; a `edge.bidir` Cytoscape style (source+target arrowheads, accent
  `#7c3aed`, `control-point-step-size` to bow apart).
- `docs/editor/index.html` — `editor.js?v=26 → ?v=27`.

## CORE.YAML items touched

None. `docs/editor/**`. No dependency, no WASM rebuild.

## Test results

- **JS (`node --test`, all `views/*.test.mjs`):** 66 passed / 0 failed (1 new:
  reciprocal pair detected from both sides; parallel-same-direction with a reverse
  present is bidirectional; one-way and self-loops excluded; empty/undefined → ∅).
- **Trigger compiles:** a mutual-reference model `@e1 { (+ e2); } @e2 { (+ e1); }`
  validates ✅ and produces reciprocal arcs (`e1→e2`, `e2→e1`) — the renderer will
  flag both `bidir`.
- **Syntax:** `node --check` clean on `editor.js` + `adapters.js`.

## Static analysis / health

- **No §6.5 anti-patterns:** detection is a pure, unit-tested helper in
  `adapters.js` (not inline graph surgery); the renderer just tags a class.

## Verification note (honest)

The detection logic is unit-tested and the trigger model compiles to a reciprocal
pair; the final *visual* (the double-headed bowed connector) was **not** captured
in a headless screenshot — the editor has no load-arbitrary-source-by-URL path,
so a custom reciprocal model can't be loaded headlessly. The rendering is
standard Cytoscape class styling over the verified detection.

## Open issues / follow-ups

- Reciprocity is rare in the hub-based arc rendering (arcs go edge→member); the
  feature activates for edge↔edge refs and mutual `<isa>`, and lies dormant
  otherwise (one-way graphs unaffected — confirmed on the kinematic example via
  the test suite).

## Experiment provenance

Not an experiment. Toolchain: `hymeko_cli` (cargo), node v24.14.0, MiKTeX
pdflatex. Working tree dirty from prior session work.
