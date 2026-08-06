# Editor HRI profile: relations as signed hyperedges (bug fix)

**Date:** 2026-06-15
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)
**Fixes:** the HRI profile shipped in the 2026-06-15 editor profiles/imports work.

## Summary

User-reported bug: in the editor's HRI coalition cell the relations
`r_ab`/`r_ar`/`r_br` were modelled as **property nodes** (`r_ab: hri.interpersonal
{ from alice; to bob; sign 1; … }`), not as hyperedges. Corrected them to
**signed hyperedges** over the agents:

```hymeko
@r_ab: + <isa> hri.interpersonal { (+ alice, + bob); }
@r_ar: + <isa> hri.hri_relation  { (+ alice, + r1); }
@r_br: + <isa> hri.hri_relation  { (+ bob,   + r1); }
```

All-positive arcs → a balanced σ-cycle (the coalition triangle), and the
relations now render as edges in the graph / hyperedges in the 3D view, matching
the profile's "+ Relation" palette action (which already emits `@`-edges).

## Files touched

- `data/profiles/hri_cell.hymeko` — relations → `@`-hyperedges.
- `docs/editor/views/profiles.js` — the embedded `HRI_ROOT` (kept byte-consistent
  with the fixture); editor cache → `?v=26`, `editor.js?v=26`.

## CORE.YAML items touched

None. `data/profiles/**`, `docs/editor/**`.

## Test results

- `hymeko validate data/profiles/hri_cell.hymeko`: **✅ valid** (the
  `@edge: + <isa> <node-type>` form compiles, same pattern as the kit joints).
- `profiles.test.mjs`: **6/6 pass** — the embedded `HRI_ROOT` matches the edited
  fixture (embed ≡ fixture consistency holds).
- `cargo test -p hymeko_wasm --test test_compile multi_file`: **4/4 pass** — the
  HRI cell + meta still compile through the editor's multi-file pipeline.

## Static analysis / health

No code logic changed (data fixture + embedded string). No new anti-patterns.

## Open issues / follow-ups

- The standalone `Req-trace`/`HRI` *examples* and the real
  `data/coalitions/triad_hri.hymeko` still use the node-form relations
  (that's what the rapport loader reads via `from`/`to`/`sign` fields); only the
  editor HRI **profile** was switched to the hyperedge form, which is the right
  model for the editor's hypergraph view.

## Experiment provenance

Not an experiment. Toolchain: `hymeko_cli` (cargo, stable), node v24.14.0.
Verified after the harness command-safety classifier recovered from a temporary
outage. Working tree dirty from prior session work.
