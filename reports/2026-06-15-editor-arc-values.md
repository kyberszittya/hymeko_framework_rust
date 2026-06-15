# Editor: arc-ref value editing

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-arc-values/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Closed the user-reported gap "arc values cannot be set." Selecting an edge now
shows an inline **arc-ref editor**: one row per ref with **sign** (+/−/~),
**target** (node), and **value** (the arc payload, e.g. a joint origin transform
`[[0,0,0.2],[0,0,0]]`), plus add / remove / **Apply arcs**. Clicking an arc line
in the graph opens its edge's editor.

Editing is a string transformation on the edge body (source-as-truth) — no WASM
or snapshot change (the snapshot `ArcDto` has no value field). The parse/rewrite
core is a pure, unit-tested module.

## Files touched

New:
- `docs/editor/views/arcs.js` — pure tuple ops: `splitTopLevel` (bracket-aware),
  `parseArcRef`, `parseArcTuple`, `formatArcRef`, `formatArcTuple`,
  `rewriteArcTuple`.
- `docs/editor/views/arcs.test.mjs` — 9 tests.

Edited:
- `docs/editor/editor.js` — `readEdgeArcs` / `applyArcs` (locate edge body,
  rewrite the tuple via `rewriteArcTuple`, recompile); `renderArcEditor` in the
  selection panel; arc-line click opens its edge; `?select=<name>` deep-link.
- `docs/editor/index.html` — `editor.js?v=22 → ?v=23`.
- `docs/editor/editor.css` — `.arc-editor` / `.arc-row` styling.

## CORE.YAML items touched

**None.** Pure `docs/editor/**`. No WASM rebuild, no dependency, no grammar change.

## Test results

- **JS (`node --test`, all `views/*.test.mjs`):** 63 pass / 0 fail (9 new in
  `arcs.test.mjs`, including the value round-trip parse→format→parse and
  `rewriteArcTuple` replace + insert + no-op cases).
- **Browser (headless Chrome, `?select=spin_joint`):** the ARC-REFS panel renders
  two rows — `+ base_link` with its `[[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]` value in
  an editable field, and `- spinner` — plus +arc-ref / Apply. Confirmed the value
  is editable (the reported gap).

## Performance results

Negligible (string edit + existing recompile). No hot path; RSS unaffected.

## Static analysis / health

- All JS tests green. **No §6.5 anti-patterns:** pure parse/rewrite core split
  from DOM glue; the splice is the unit-tested `rewriteArcTuple` (not inlined,
  duplicated string surgery); no globals.

## New / removed dependencies

None.

## Open issues / follow-ups

- The arc editor edits the **first** `(...)` tuple in an edge body (every shipped
  example has exactly one). Multi-tuple edge bodies would need a per-tuple
  selector.
- Target list is the current node set; cross-edge arc targets aren't offered
  (rare).
- Remaining queued editor ideas: 2D composite/concentric stacking (roots center)
  + translate-pan in the 3D view; the hero demo.

## Plan note

This was a small follow-up to the day's editor work; the plan (4 artifacts,
PDF compiles) was written alongside the implementation. Risk is low — pure
string mutation, fully unit-tested, no core/deps/WASM — so the usual
plan-before-code ordering carried little additional risk-surfacing value here.
Flagged for honesty per CLAUDE.md §2.

## Experiment provenance

Not an experiment. Toolchain: node v24.14.0, Chrome (headless verify), MiKTeX
pdflatex (plan). Working tree dirty from prior session work unrelated to this
change.
