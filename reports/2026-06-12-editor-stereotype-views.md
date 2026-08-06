# Editor 3D view tabs — transformation-stereotype viewers

**Date:** 2026-06-12 · **Plan:** `docs/plans/2026-06-12-editor-stereotype-views/`
(four-format: plan.tex/pdf/tikz/mmd) · approved plan `soft-knitting-lobster.md`

## Summary
Added a **view-tab host** to the WASM editor (`docs/editor/`) driven by the live
`snapshot_json()` / `to_urdf()` surface — no WASM rebuild. Three tabs:
1. **Graph** — the existing Cytoscape topology, now also drawing `<isa>`
   inheritance as **dashed edges** (user request).
2. **Hypergraph 3D** — a three.js star/clique force-directed viewer (the "fancy"
   general view), Star↔Clique toggle.
3. **Kinematic** — a transformation stereotype showing **both** a true robot
   render (three.js, from the emitted URDF geometry) **and** the αₖ regime
   compass + signed-topology ring.

A small `View` registry (`{name, mount, render, unmount}`) routes `recompile()`
to the active view; the 3D views mount lazily and unmount on leave so only one
render loop runs. Each stereotype is now one pluggable module — further ones
(SDF/physics, ROS, SysML) drop in the same way.

## Files touched
**Modified (non-core, static — no rebuild):**
| LOC | File | |
|---:|---|---|
| ~+45 | `docs/editor/editor.js` | view registry + tab routing; dashed `isa` edges + style; example enriched with geometry; `recompile`→`renderActiveView` |
| ~+12 | `docs/editor/index.html` | tab bar + 3 view panes; three.js r128 CDN; `editor.js?v=3` |
| ~+95 | `docs/editor/editor.css` | tab bar, view-stack, 3D chrome, kinematic panels |

**New:**
| LOC | File | |
|---:|---|---|
| 196 | `docs/editor/views/adapters.js` | pure: snapshotToHyperedges, snapshotToKinematicGraph, cycleArity, parseUrdf |
| 138 | `docs/editor/views/adapters.test.mjs` | 9 `node --test` units |
| 87 | `docs/editor/views/regime_classes.js` | `Compass`/`Topo` extracted from demo_web/index.html |
| 188 | `docs/editor/views/hypergraph3d.js` | three.js star/clique View |
| 196 | `docs/editor/views/kinematic.js` | robot render + regime panels |
| — | `docs/plans/2026-06-12-editor-stereotype-views/` | plan.{tex,pdf,tikz,mmd} + plan-tikz.pdf |

## CORE.YAML items touched
**None.** `docs/editor/` and `demo_web/` are non-core; `to_urdf()`/`snapshot_json()`
are consumed, not modified. No Rust/WASM change, no dependency add.

## Design notes
- **Geometry source:** structured kinematic data (link geometry, joint axes) is
  not in the snapshot — it lives in `KinematicModel` and surfaces only via
  `to_urdf()`. The robot render therefore parses the emitted URDF client-side
  (`parseUrdf`), so no WASM accessor/rebuild was needed.
- **Honesty:** the hypergraph-3D force layout and the topology ring are for
  legibility, not geometry; the **robot render is geometrically faithful** (it is
  the URDF the engine emits). When a source carries no `<visual>` geometry the
  robot tab degrades to joint frames + a stated note.
- **Scaffolding filter:** `snapshotToKinematicGraph` restricts links to nodes
  referenced by a joint, so the regime panels show the robot, not the inline
  `kit` namespace decls.
- **`Scene3D` deliberately not reused:** the demo's illustrative FK skeleton
  would misrepresent a real robot; the URDF-driven render replaces it. `Compass`
  and `Topo` (the honest αₖ + signed-topology analytics) are reused as-is.
- **Example enriched:** the self-contained `kit` example now defines a `geometry`
  namespace and gives `base_link` a cylinder / `spinner` a box, so the robot tab
  shows shapes out of the box (verified to emit `<cylinder>`+`<box>` URDF).

## Test results
| Layer | Tests | Result | Notes |
|---|---|---|---|
| Unit — pure adapters (`node --test`) | 9 | pass | hyperedge members/sign/arity; non-vertex arc dropped; kinematic graph + scaffolding filter; `cycleArity` triangle / 4-cycle / **K4 → 4 triangles + 3 quads**; `parseUrdf` cylinder+box+joint and the bare/no-geometry graceful case |
| Static — ESM syntax (`node --check`) | 5 files | pass | editor.js + all 4 view modules |
| Served-asset smoke (HTTP) | 6 | 200 | index, editor.js?v=3, all 4 view modules; three.js CDN reachable |
| Browser render | manual | not auto-verified | same caveat the existing demo_web viewers carry; opened at http://localhost:8000/ |

No JS linter/type-checker is pinned in `tools.yaml`; `node --check` is the
available syntax gate. No Rust/Python sources changed (probe scripts deleted), so
clippy/ruff/mypy are not applicable to this change.

## Performance
Client-side only; no spawned process, no RSS budget to assert (browser-governed).
Adapters are O(nodes+arcs) + a bounded DFS to k=6; the three.js loops are the
existing star-viewer cost. `node --test` runs in ~0.2 s.

## Verification (manual, on stage / locally)
`cd docs/editor && python -m http.server 8000`, open `http://localhost:8000/`
(hard-refresh once for the new `index.html`). Switch the three tabs; confirm the
Graph tab shows dashed `<isa>` lines (e.g. `link ⇢ meta_element`), Hypergraph 3D
toggles Star/Clique, and the Kinematic tab renders the cylinder+box robot with
live αₖ / topology panels. Edit the source and confirm the active view re-renders
on the 400 ms debounce.

## Provenance
- Git SHA `af803ee` (dirty — pre-existing untracked editor/seminar work + the
  files above). Python 3.12.13 (URDF fixtures), node v24.14.0 (tests),
  MiKTeX-pdfTeX 4.23 (plan PDFs). three.js r128, cytoscape 3.30.4 (CDN).
- Deterministic: adapters/tests use fixed inputs; the 3D force layout seeds with
  `Math.random()` (cosmetic node positions only — no reported number depends on it).

## Post-release fix — memory leak (2026-06-12)
First in-browser use showed memory climbing. Root causes + fixes:
- **Retained scenes via leaked `window` listeners** (dominant): each 3D-view
  `mount()` added `window` `pointermove`/`pointerup` handlers (and `Compass`/`Topo`
  added `resize` handlers) that were never removed on `unmount()`. Their closures
  captured the scene/renderer/geometry, so every tab switch retained the entire
  previous view. Fixed: handlers stored as named refs and `removeEventListener`-ed
  on `unmount()`; `Compass`/`Topo` got a `dispose()`.
- **Undisposed three.js GPU resources:** geometries/materials were replaced on
  each rebuild/re-render without `.dispose()`. Fixed: `clearObjects()`
  (hypergraph) and `disposeGroup()` (kinematic) dispose on every rebuild and on
  unmount; `renderer.forceContextLoss()` + `dispose()` on unmount.
- **Per-frame canvas reallocation:** `Compass`/`Topo` `resize()` ran every frame
  and reset `canvas.width/height` unconditionally (reallocating the backing
  store 60×/s). Fixed: `resize()` is now a no-op unless the pixel size changed.
- Cache: the view-module import chain is versioned (`?v=4`) so the fixes load on
  a normal refresh. Re-verified: 9 adapter tests pass, all modules `node --check`
  clean, all `?v=4` URLs serve 200.

## Enhancement batch (2026-06-12, post-initial)
User-requested 3D-view refinements, all client-side (no WASM change):
- **Hypergraph 3D:** three modes (star / clique / **prism** — translucent
  polytope over each hyperedge's members, enlarged cross-section + thicker
  extrusion); **node labels** (sprites, toggle); **attribute colouring** by first
  base / first tag with a **legend**; **hover tooltip + click-to-select**
  (raycasting); **meta `<isa>` lines** toggle (dashed amber to base types);
  **background toggle** (default **white**, switch to dark).
- **Kinematic:** **hover/click** on robot links (name + geometry), **background
  toggle** (default white).
- **Graph (Cytoscape):** **`<isa>` visibility** checkbox in the palette.
- **New pure module** `views/geometry3d.js` (prism geometry, Newell normal,
  categorical colour map, attribute value) with **6 unit tests** — total **15
  `node --test`** (9 adapters + 6 geometry), all pass. Memory hygiene preserved:
  every new three.js object (sprites/prisms/meta-lines) is disposed in
  `clearObjects()`; hover/click listeners live on the canvas (removed with it),
  not on `window`.
- Import chain versioned to `?v=6`.

## Enhancement batch 2 (2026-06-12) — relationships + source panel
- **`<isa>` as a first-class relationship layer (Hypergraph 3D):** default-on,
  brighter dashed, **and it now drives the force layout** (related nodes cluster
  instead of drifting in the floating space — the reported problem). A
  **relationship legend** (bottom-right) groups the named relationships currently
  drawn (signed membership ±, `<isa>`).
- **Floating source panel:** the `.hymeko` source moved from the bottom footer to
  a **floating, collapsible panel** (`<details class="source-float">`) overlaying
  the views — always available, collapses to a title bar.
- Import chain `?v=7`.

### Data-model finding (blocks full "group all relationships")
The WASM `snapshot_json()` exposes only **two** relationship kinds: `bases`
(`<isa>`/template) and `arcs` (signed hyperedge membership). The other *named,
annotated* relationships the user wants grouped — field references (`visual ->
link_geometry`, `limit -> joint_rev_limit`) and **containment/scoping**
(`kit ⊃ elements ⊃ link`) — are **not in the snapshot or any emitter** (verified
via `snapshot_json` + `to_dot`). Surfacing them needs a small Rust change in
`hymeko_formats/src/snapshot.rs`: emit a uniform `relationships: [{kind, from,
to, label, sign?}]` list (kinds: isa, ref, scope, membership). Then the 3D view
renders each kind as a grouped, colour-coded, toggleable layer. That change
requires a `wasm-pack` rebuild → **§1 toolchain approval** (wasm-pack +
`wasm32-unknown-unknown`, neither installed).

## Enhancement batch 3 (2026-06-12) — WASM relationship extension (approved §1)
The "group all relationships" request needed data the snapshot didn't carry, so
— with explicit user approval — the WASM/snapshot was extended and rebuilt.
- **Toolchain add (§1, approved):** installed `wasm-pack` 0.15.0 +
  `wasm32-unknown-unknown` target (neither was present). Not yet recorded in
  `tools.yaml` — follow-up if the team wants it pinned.
- **Rust (non-core — `hymeko_formats`/`hymeko_wasm` are not in CORE.YAML):**
  `hymeko_formats/src/snapshot.rs` now emits a uniform `relationships:
  [{kind, from, to}]` list keyed by `DeclId` — **`kind ∈ {scope, isa, ref}`**:
  containment from `DeclNode.parent`, inheritance from node/edge `bases`, and
  **field references** (`a -> b`) from `AnnoR.value` (`ValueR::Ref`, incl. refs
  nested in lists). Additive field; no existing consumer breaks. Rebuilt
  `docs/editor/pkg/` via `wasm-pack build --target web --release` (twice: scope+isa,
  then +ref).
- **Native test** `hymeko_wasm/tests/test_snapshot_relationships.rs` — asserts the
  real `compile_source` → `snapshot_json` path emits **`scope`, `isa`, and `ref`**
  (1 test, passes). `cargo clippy -p hymeko_formats` clean; changed files
  `fmt`-clean (pre-existing drift in `codegen.rs`/`lib.rs` is unrelated).
- **Client:** new pure `snapshotRelationships(snapshot)` adapter (groups by kind,
  maps ids→indices, drops danglers) — **2 unit tests** (17 JS total). The
  Hypergraph 3D view replaced the single isa toggle with a **relationship-layer
  system**: per-kind toggle buttons (`isa` on, `scope` off by default), each a
  dashed colour-coded layer (isa amber, scope green) that **also drives the force
  layout** so related nodes cluster; the relationship legend lists active layers.
- **Client relationship layers:** `isa` (amber, on), `ref` (purple, on),
  `scope` (green, off — it's dense). Each toggleable, colour-coded, and drives
  the force layout.
- **Source panel:** floating div (not `<details>`) anchored **top-right**,
  **draggable** by its header, **resizable** (`resize: both`), and collapsible
  via a left button; relationship legend moved bottom-left to clear it.
- Import chain `?v=9`. **Browser needs a hard-refresh** to refetch the rebuilt
  `pkg/` (the WASM URL is unversioned).

## Enhancement batch 4 (2026-06-12) — composition tree + tool pin
- **`wasm-pack` pinned** in `tools.yaml` (rust.wasm_build, `0.15`, §1-approved
  2026-06-12) — the editor rebuild is now reproducible.
- **Composition-tree layout** (Hypergraph 3D): a `Layout: force ↔ tree` toggle.
  In `tree` it lays vertices out by the **scope (containment) tree** — children
  stacked under their parent, each parent at the **barycentre** of its children
  ("barycentric closeness") — centred and camera-fitted front-on, with the scope
  layer auto-shown. Pure layout math `treePositions(n, scopeSegs)` in
  `geometry3d.js` with **2 unit tests** (barycentric root-over-children; isolated
  node → depth-0 root). 19 JS tests total. No WASM change; cache `?v=10`.
- Deferred (offered): a *transparent nested-container* notation for composition
  (translucent boxes per scope subtree) — the tree layout covers the
  "stacked under / barycentric" reading; the nested-box view is a separate add.

## Enhancement batch 7 (2026-06-12) — tree-by-relationship + nested containers
- **Tree source selector** (`Tree: scope ↔ isa ↔ ref`): the composition layouts
  (tree/cone) and the nesting are now driven by **any** named relationship, not
  just scope. The tree-source layer is the one auto-shown in tree/cone layouts.
- **Transparent nested-container notation** (`Nest: on/off`): each internal node
  of the tree-source hierarchy gets a translucent, depth-coloured box enclosing
  its whole subtree — containment reads as nested volumes. Cheap (unit-cube
  meshes transformed per frame from the live bounding box of each subtree); fully
  disposed in `clearObjects`. Works in every layout (most legible in tree/cone).
- No WASM change; cache `?v=13`. 24 JS tests (the layout math these reuse —
  `treePositions`/`coneTreePositions` — is already unit-tested).

## Enhancement batch 6 (2026-06-12) — 3D barycentric cone layout
- The Hypergraph 3D `Layout` toggle now cycles **force → tree → cone**. `cone` is
  a genuine **3D barycentric** layout: leaves spread on depth rings (deeper =
  wider), each internal node placed at the **3D centroid of its children**, one
  level up — orbit to view. Pure `coneTreePositions(n, scopeSegs)` in
  `geometry3d.js` with a unit test (root collapses to the axis by symmetry;
  internal node = mean of children X/Z; deeper = lower). 24 JS tests total. No
  WASM change; cache `?v=12`.

## Enhancement batch 5 (2026-06-12) — source syntax highlighting
- **`.hymeko` syntax highlighting** in the source panel via the overlay technique:
  a coloured `<pre>` behind a transparent `<textarea>` (caret-color visible),
  scroll-synced. The textarea stays the source of truth — **no editing logic
  changed**. New pure tokenizer `views/highlight.js` (`highlightHymeko`) —
  comments, strings, `<…>` tags, numbers, keywords (`using/const/as`), operators
  (incl. `->`), identifiers; HTML-escaped (no injection). **4 unit tests** incl.
  visible-text round-trip and keyword-vs-identifier; **23 JS tests** total. Cache
  `?v=11`; no WASM change.

## Enhancement batch 8 (2026-06-13) — root sizing + compositional dotted lines + default labels
User-requested Hypergraph-3D refinements, all client-side (no WASM change):
- **Root elements bigger + bigger gravity.** New pure `depthSizes(n, scopeSegs,
  {base,falloff,min})` in `views/geometry3d.js` (reuses the existing
  `buildScopeTree` walk — no duplicate parent/depth builder) returns a per-node
  size/mass by containment depth: roots (depth 0) largest, descendants shrink
  geometrically, floored at `min`. The view scales each vertex sphere by it
  **and** uses it as force-layout mass (`a = F/m` via `addScaledVector`), so heavy
  roots anchor the cloud while lighter descendants orbit them — that inertia is
  the "gravity". **2 new unit tests** (monotone-by-depth + floored; deep-node
  clamp + isolated-node-is-root). 26 JS tests total, all pass.
- **Compositional relationships as dotted lines, on by default.** The `scope`
  (containment) layer now defaults **on** (`relOn.scope = true`) and draws with a
  **dotted** style (`dashSize 1.5 / gapSize 4`) vs the dashed isa/ref layers;
  each `REL_KINDS` entry carries its own `dash` spec consumed in `buildRelLayers`.
- **Node labels visible by default** (`showLabels = true`; the toggle and its
  button state still work). Label sprites are lifted clear of the larger root
  spheres (`5 + size·6`).
- Memory hygiene unchanged: spheres reuse the shared geometry (per-mesh
  `.scale`), nest/label/rel objects still disposed in `clearObjects`. Import chain
  bumped to `?v=14` across `editor.js`, `index.html`, `hypergraph3d.js`,
  `kinematic.js`. `node --check` clean on all modules; all `?v=14` URLs serve 200.
- **Hyperedge hubs are cubes (2026-06-13 follow-up):** `hGeo` swapped from
  `SphereGeometry(3.4)` to `BoxGeometry(5.5³)` so vertices (spheres) vs hyperedges
  (cubes) read distinctly. Import chain bumped to `?v=15`.
- **Hub cube grows with arity (2026-06-13):** new pure `hubSize(arity)` in
  `geometry3d.js` (binary=reference 1, +0.22/extra member, clamped [0.7, 2.4]);
  each hub mesh `.scale.setScalar(hubSize(e.length))`. **1 unit test** (binary
  reference, monotone, clamp, opts). Import chain `?v=16`.
- **Description-wise viewpoint selection (2026-06-13):** the Hypergraph-3D view
  gained a **selectable, multi-toggle** filter — the MDSD move (one model, composed
  viewpoints), not a single cycle. A second toolbar row holds **one show/hide
  toggle per top-level description namespace** (keyed by namespace *name* so the
  selection survives the per-keystroke recompile), plus a **Roots only** switch;
  the two compose. Membership is by **scope (containment)**, not literal import
  provenance — a true `imported` flag would need `hymeko_core`/`parser` (both
  `lockdown: full`) and the in-browser compiler can't resolve external `@"file"`
  includes anyway, so the honest unit is the namespace root. New pure
  `scopeMembership(n, scopeSegs)` in `geometry3d.js` → per-node `{depth, root,
  roots}` (cycle-guarded); **2 unit tests** (two-namespace mapping; isolated-node +
  2-cycle termination). The view computes `visV`/`visE` (an edge is visible iff all
  its members are) and applies them to vertex spheres, hub cubes, labels,
  structural lines, relationship layers, prisms, **and the force layout** (hidden
  nodes exert no force, don't drift) — plus `pickables()` (three.js raycasts ignore
  `.visible`). The toggle bar rebuilds only when the namespace set changes; CSS
  `.view3d-filterbar`/`.view3d-filterhead` added. Stats show filtered counts. 29 JS
  tests total, all pass. Import chain `?v=18`.

## Open issues / follow-ups
- Browser render is not auto-verified — a future headless smoke (playwright)
  would close this, but that is a dependency add (§1).
- Further stereotypes (SDF/physics, ROS launch, SysML) — one `views/*.js` each on
  the registry.
- Optional: a WASM `geometry_json()` accessor if client-side URDF parsing ever
  proves limiting (would need a rebuild, §1 toolchain).
