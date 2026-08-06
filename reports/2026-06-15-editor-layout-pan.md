# Editor: composite (roots-centred) 2D layout + 3D pan

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-layout-pan/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Two view-ergonomics requests:

1. **Composite stacking, roots in the centre.** New 2D graph layout option
   **Composite (roots centred)** — a Cytoscape `concentric` layout keyed by
   *scope (containment) depth*: top-level container decls at the centre, nested
   children on outer rings. Selectable next to the existing **Force** (`cose`)
   layout in the View panel. Depth comes from a pure, tested helper.
2. **Pannable views.** The 2D Cytoscape graph already background-pans. The
   Hypergraph 3D view gained **translate-pan**: right-drag or shift-drag moves
   the look-at focus (orbit = plain drag, zoom = wheel, unchanged).

## Files touched

- `docs/editor/views/adapters.js` — new pure `scopeDepths(snapshot)` →
  `Map<declId, depth>` from the `scope` relationships (cycle/missing-parent safe).
- `docs/editor/views/adapters.test.mjs` — 2 new tests (chain depths, empty cases).
- `docs/editor/editor.js` — `graphLayout` state; `runGraphLayout` (cose vs
  concentric); `setConcentricLevels` (per-node `clevel = maxDepth - depth`);
  layout `<select>` wiring; `?layout=` deep-link; bumped the hypergraph3d +
  adapters import versions.
- `docs/editor/views/hypergraph3d.js` — look-at `target`; `panCamera(dx,dy)`
  (screen-plane translate, distance-scaled); right/shift-drag pan; context-menu
  suppressed; `target` recentres on mount.
- `docs/editor/index.html` — Graph-layout `<select>` in the View panel;
  `editor.js?v=23 → ?v=24`.

No `editor.css` change was needed (the selector fits the existing `.vrow`).

## CORE.YAML items touched

**None.** Pure `docs/editor/**`. No WASM rebuild, no dependency, no grammar change.

## Test results

- **JS (`node --test`, all `views/*.test.mjs`):** 65 pass / 0 fail (2 new
  `scopeDepths` tests: chain a⊃b⊃c → depths 0/1/2, edge nesting, self-scope
  ignored, multi-root, empty snapshot).
- **Browser (headless Chrome):**
  - `?layout=concentric` — the selector reads "Composite (roots centred)" and
    the graph is concentric: leaf field nodes (mass/origin/visual/dimension) on
    outer rings, containers pulled inward.
  - `?view=hyper3d` — the 3D view renders correctly after the target-based
    camera change (orbit/zoom intact; pan is right/shift-drag).

## Performance results

`scopeDepths` is O(decls); the layout is Cytoscape's own. 3D pan is a
constant-time vector update per pointer move. No hot path; RSS unaffected.

## Static analysis / health

- All JS tests green. **No §6.5 anti-patterns:** the depth logic is a pure,
  tested helper in `adapters.js` (not inlined string/DOM surgery); the layout
  dispatch is a single `runGraphLayout`; pan reuses the existing camera state.
- Two new deep-links (`?layout=`, plus the earlier `?profile=`/`?select=`)
  share the one URLSearchParams pass.

## New / removed dependencies

None.

## Open issues / follow-ups

- Pan was added to the **Hypergraph 3D** view; the Kinematic 3D view still
  orbits/zooms only (same `panCamera` pattern would port if wanted).
- The concentric layout uses scope depth; an alternative keyed by `isa` depth or
  a hybrid could be offered as another layout option.
- Remaining queued idea: the **hero demo** (`docs/plans/2026-06-13-hero-demo/`),
  now unblocked by the imports/profiles work.

## Experiment provenance

Not an experiment. Toolchain: node v24.14.0, Chrome (headless verify), MiKTeX
pdflatex (plan). Working tree dirty from prior session work unrelated to this
change.
