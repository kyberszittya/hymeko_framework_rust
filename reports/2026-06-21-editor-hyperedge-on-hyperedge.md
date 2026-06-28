# Editor: render hyperedge-on-hyperedge incidence (strategy.hymeko shows no hyperedges)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** [docs/plans/2026-06-21-editor-hyperedge-on-hyperedge/](../docs/plans/2026-06-21-editor-hyperedge-on-hyperedge/)

## Summary
The editor's **Hypergraph 3D** view rendered nothing for `galambos_strategy.hymeko`. Root cause (verified
at source, not guessed): `snapshotToHyperedges` resolved each arc's `target_id` only against
`snapshot.nodes` and dropped any edge with zero vertex-members. The strategy file is a **bundle of
bundles** — `@explore`/`@exploit` carry only scalar fields (no arcs), and `@strategy_spec`'s arcs
`(+ explore, + exploit)` target *edges*. All three resolved to zero members → all dropped → empty view.

The fix resolves arc targets against **edges** as well as nodes, encoding an edge member as index
`n_vertices + edgeIndex` — the same hub-position space the renderer already maintains in `P[]`. A
zero-vertex-member edge is kept only when it is part of a bundle (references an edge, or is referenced by
one); a stray type-root with no incidence stays dropped, preserving the old "never invent a target"
contract.

**Verification of the data-model premise:** `hymeko_formats/src/snapshot.rs:133` emits `target_id = DeclId`
in one unified decl space, so an arc to an edge carries that edge's `id`, distinct from any node id
([snapshot.rs:126-141](../hymeko_formats/src/snapshot.rs#L126-L141)). This was confirmed by reading the
emitter; a live in-browser end-to-end was **not** run (no headless WASM snapshot path), so that link is
verified structurally + by unit test on a faithful synthetic snapshot, not by a browser screenshot.

## Files touched (non-core; `docs/editor/` is outside CORE.YAML)
- `docs/editor/views/adapters.js` — `snapshotToHyperedges` rewritten (+33/−10): node+edge target
  resolution, bundle-keep rule, edge-member encoding.
- `docs/editor/views/hypergraph3d.js` — `infoFor` (+5/−1): edge-member label guard (tooltip/selection).
- `docs/editor/views/adapters.test.mjs` — (+44) three new tests: edge-on-edge incidence, stray-edge still
  dropped, vertex-only back-compat.

**CORE.YAML items touched:** none. **New/removed deps:** none. **WASM rebuild:** none (JS adapter only;
snapshot format unchanged).

## Test results
- `node --test docs/editor/views/adapters.test.mjs` — **17/17 pass** (4 new + the preserved
  "drop non-vertex targets" regression), 235 ms.
- `node --test docs/editor/views/*.test.mjs` (full editor suite) — **75/75 pass**, 626 ms.
- Coverage: the new `snapshotToHyperedges` branches (edge member, bundle-keep, stray-drop, back-compat)
  are each driven by a dedicated test.

## Performance
Pure `O(|V|+|E|+|arcs|)` data transform; microseconds. No render-loop cost change (`P[]` already spans
`H.n + |E|`). Peak RSS negligible (node test process), far under the 16 GB cap.

## §6.5 anti-patterns
None introduced. The fix unifies one resolution path (no per-kind wrapper, no string-typed branch); it does
not duplicate the existing `snapshotToKinematicGraph` resolver.

## Open issues / follow-ups
1. `galambos_strategy.hymeko` has `view: null` in the generated `projects_data.js`, so the user reaches the
   3D view by switching tabs manually. If a default `hyper3d` view for strategy files is wanted, that is a
   one-line change in `scripts/gen_editor_projects.py` (then regenerate) — deferred, not in scope here.
2. `computeFilter`'s `visE` treats an edge member (index ≥ `H.n`) as always-visible (its `visV[v]` is
   `undefined`), so a bundle hub ignores its parts' namespace-hide toggle. Cosmetic; left as-is.
3. A live in-browser confirmation on the actual compiled strategy snapshot remains the one unautomated
   check (see Summary).
