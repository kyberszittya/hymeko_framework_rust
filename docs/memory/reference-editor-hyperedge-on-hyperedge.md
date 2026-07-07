---
name: reference-editor-hyperedge-on-hyperedge
description: "Editor Hypergraph 3D view supports edge-on-edge incidence; a hyperedge member index ≥ n_vertices is another edge's hub (P[] index space)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3060e292-680f-4645-82c1-156ce78e537c
---

`docs/editor/views/adapters.js::snapshotToHyperedges` (fixed 2026-06-21) resolves an arc's `target_id`
against **edges as well as nodes**. A bundle-of-bundles — a hyperedge whose arcs target other hyperedges,
e.g. `galambos_strategy.hymeko`'s `@strategy_spec (+ explore, + exploit)` — previously rendered as an
**empty** view (all three edges had zero vertex-members and were dropped).

**Invariant any future editor change must respect:** a hyperedge `member` index `< n_vertices` is a vertex;
an index in `[n_vertices, n_vertices + |hyperedges|)` is **another edge's hub**, mapped into the same `P[]`
position space the renderer already maintains. A zero-vertex-member edge is kept only when it is part of a
bundle (references an edge, or is referenced by one) — stray type-roots stay dropped (the "never invent a
target" contract). `hypergraph3d.js::infoFor` guards member-label lookup for indices ≥ `H.n`.

Premise verified at source: `hymeko_formats/src/snapshot.rs:133` emits `target_id = DeclId` in one unified
decl space, so node and edge ids never collide. Tests in `adapters.test.mjs` (edge-on-edge incidence;
stray-edge dropped; vertex-only back-compat). Part of [[project-editor-mdp-project]]. Report:
`reports/2026-06-21-editor-hyperedge-on-hyperedge.md`.

**Attribute-HUD folding (2026-06-21):** the same view now folds leaf "attribute" decls (mass, dimension,
ax, type, length…) OUT of the spheres and onto the owner node's HUD with values, via a default-on
"Attributes: HUD" toggle. `NodeDto` gained `value: Option<String>` (`snapshot.rs::render_value`; needs a
`wasm-pack build --target web --release --out-dir ../docs/editor/pkg` rebuild — pkg is generated). JS:
`adapters.js::foldAttributes(nV, hyperedges, scopeSegs, vertices)` — attribute = leaf ∧ not-a-member ∧
has-parent; members/containers/roots never fold (so pure-hyperedge/flat graphs are unchanged). `?v=` bumped
adapters 28 / hypergraph3d 26. Report: `reports/2026-06-21-editor-attribute-hud.md`. To change the snapshot
shape, edit `snapshot.rs` then REBUILD the wasm or the editor serves a stale `pkg/`.
