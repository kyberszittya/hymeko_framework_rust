# Star-expansion 3D viewer — build outline

Turn the standalone prototype into a HyMeKo-fed viewer. Phased so each phase
ships on its own. **Reuse the existing pattern, do not fork it** (CLAUDE.md §6.1):
the `demo_web/` data-export + static-page convention already exists for the
kinematic compass — copy it.

## What exists now (Phase 0 — done)
- `demo_web/star_expansion_viewer.html` — self-contained Three.js page (r128 via CDN). Force-directed 3D layout, orbit/zoom/hover, presets (Fano plane, tetrahedron, 4-ary chain, random), **Star↔Clique toggle**, live counts. Graphs are generated **in-browser** (synthetic). No backend.
- Precedent to copy: `demo_web/export_kinematic_data.py` → writes `demo_web/kinematic_data.json`, consumed by the static `index.html`. Two paths: real research stack, else stdlib fallback.

## Real APIs to build on (already on disk — do not reimplement)
| Need | Entry point |
|---|---|
| parse `.hymeko` → IR | `hymeko_py` `parse_hymeko_rs` / `PyHypergraphIR` |
| star expansion COO `(k,i,j,val)` | `PyHypergraphEngine.compile_star_expansion(ir) -> PyTensorCoo3D` (`hymeko_py/src/interface_python/api.rs:578`) |
| clique expansion | `PyHypergraphEngine.compile_clique_expansion(ir) -> PySparseMatrix2D` / `compile_clique_tensor_expansion` |
| core expansion (Rust) | `hymeko_hre/src/expansion.rs::star_expansion_coo` / `clique_expansion_coo`; CSR in `hymeko_hnn` |
| COO → dense / inspect (examples) | `python/examples/parsing/example_parse_description_python.py`, `python/examples/coo_tensor/coo_tensor_pytorch.py` |
| canonical hash | `hymeko_core/src/ir/canonical_hash.rs` (already exposed through the IR) |
| live stream (optional) | daemon Arrow topics `HymekoFastStateV2` (star) / `…/tensor/clique`; HTTP/WS shim via `hymeko_server` (Axum) |

---

## Phase 1 — data contract
Define the JSON the page consumes (mirror `kinematic_data.json`). Minimal sufficient
representation: the page derives star/clique geometry in JS (the prototype already
does), but the **counts come from the engine** so they match repo numbers verbatim.

```json
{
  "schema": 1,
  "source": "data/robotics/robot_4wh.hymeko",
  "canonical_hash": "blake3:…",
  "n_vertices": 12,
  "vertex_labels": ["base", "arm", "..."],
  "hyperedges": [ { "label": "shoulder", "members": [0, 1], "sign": 1, "arity": 2 } ],
  "star":   { "incidence_nnz": 34, "coo": { "k": [], "i": [], "j": [] } },
  "clique": { "edge_count": 56 }
}
```
- `members` = vertex indices per hyperedge (drives both star hubs and clique edges in JS).
- `star.coo` optional but recommended — it is exactly the `(k,i,j)` the daemon emits; include it so the same blob can later feed a tensor view.
- **Acceptance:** schema documented in this file; one hand-written example committed at `demo_web/star_expansion_data.example.json`.

## Phase 2 — exporter (Python)
New file `demo_web/export_star_expansion.py`, structured like `export_kinematic_data.py`:
- **Real path:** import `hymeko`; parse a `.hymeko` source → IR → `compile_star_expansion` / `compile_clique_expansion`; read back vertices, hyperedge member lists + signs, the COO, the canonical hash; emit the JSON above. Counts taken from the engine (NNZ = star incidences; clique edge count from the 2D expansion).
- **Stdlib fallback:** if `hymeko` isn't importable, parse a small subset / accept a hypergraph literal and compute `members`, `Σ|e|`, `Σ C(|e|,2)` directly — same semantics, dependency-free (mirrors the kinematic exporter's fallback).
- CLI: `python demo_web/export_star_expansion.py --src data/robotics/robot_4wh.hymeko --out demo_web/star_expansion_data.json`.
- **Acceptance:** on a fixture (e.g. the Fano `.hymeko` or `data/robotics/robot_4wh.hymeko`), `star.incidence_nnz` and `clique.edge_count` match `example_parse_description_python.py` / the existing COO tests exactly; canonical hash is stable across reorderings of the same graph.

## Phase 3 — wire the page to real data
Edit `star_expansion_viewer.html`:
- Add a data source: `fetch('star_expansion_data.json')` (HTTP) **or** a `star_expansion_data.js` `<script>` blob for `file://` double-click (exactly the dual path `index.html` already uses for `kinematic_data.js`).
- When data is present, build the graph from `hyperedges[].members` instead of the synthetic generator; show `source`, `canonical_hash`, and the engine counts in the stats panel.
- Keep the in-browser generator as a "Sandbox" preset so the page still works with no data file.
- **Acceptance:** opening the page with a committed `star_expansion_data.json` renders the real graph; star/clique counts shown equal the exporter's (no in-JS recount drift — display the engine numbers, verify JS-derived counts agree).

## Phase 4 — live stream (optional, forward-looking)
Browsers cannot read Iceoryx2 / Arrow shared memory directly. Add a thin bridge:
- `hymeko_server` (Axum) endpoint that subscribes to the daemon's `HymekoFastStateV2` star topic, decodes the Arrow record batch, and re-emits the Phase-1 JSON over **WebSocket** on each `StructuralUpdate`.
- Page opens the socket and rebuilds/animates on each message (the topology-hash gate already tells you structural vs weight-only — only rebuild on structural).
- **Acceptance:** editing a `.hymeko` and recompiling pushes a new graph to the open page within the daemon's update budget; no page reload.

---

## Honest-presentation notes (carry into the talk)
- The 3D positions are a **force-directed layout for legibility, not geometric ground truth** — say so when presenting; it is not spatial data.
- The **edge-count arithmetic is exact** and sourced from the engine; that is the load-bearing claim (O(|E|·d) star vs O(|E|·d²) clique).
- Display the engine's counts, not only JS-derived ones, so the numbers match every other artifact in the repo.

## File map
```
demo_web/
  star_expansion_viewer.html        # Phase 0 (done) → edited in Phase 3
  export_star_expansion.py          # Phase 2 (new; mirror export_kinematic_data.py)
  star_expansion_data.json          # Phase 2 output (gitignore or commit a small fixture)
  star_expansion_data.example.json  # Phase 1 schema example
STAR_EXPANSION_VIEWER_OUTLINE.md    # this file
```

## Build order
1. Phase 1 (schema + example) — cheap, unblocks the rest.
2. Phase 2 (exporter) — reuse `export_kinematic_data.py` structure + `compile_star_expansion`.
3. Phase 3 (wire page) — reuse `index.html`'s dual JSON/JS load path.
4. Phase 4 (live) — only if a live demo is wanted; otherwise stop at 3.
