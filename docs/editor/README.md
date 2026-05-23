# HyMeKo Editor (WASM MVP)

Visual hypergraph design surface for `.hymeko` sources, running entirely in the browser via the `hymeko_wasm` bundle.

## Architecture

The **source text is the single source of truth**. The editor:

1. Holds the current `.hymeko` text in a textarea (canonical state).
2. After every change, calls `parse_and_compile(source)` via WASM to recompile.
3. Reads the resulting `snapshot_json()` — same JSON the Python wheel produces.
4. Renders the snapshot on a Cytoscape canvas (vertices = ellipses, edges = rounded boxes, signed arc-refs = coloured arrows).
5. Edit operations (palette buttons, properties panel) are **string transformations on the source text** + recompile + redraw.

This keeps the editor in sync with the rest of the toolchain — anything the textual `.hymeko` expresses is what the URDF / SDF / DOT / SysML emitters consume.

## Build + serve

```bash
# 1. Build hymeko_wasm into docs/editor/pkg/
cd hymeko_wasm
wasm-pack build --target web --release \
    --out-dir ../docs/editor/pkg

# 2. Serve docs/editor/ over HTTP (any static server)
cd ../docs/editor
python3 -m http.server 8000
# Open http://localhost:8000/
```

(Why HTTP and not file://? wasm-bindgen ES module loading requires a real HTTP origin.)

## What the MVP does

- **Compile + render** any `.hymeko` source on every change (debounced 400 ms)
- **Add Link** (palette) — modal asks for name + optional mass
- **Add Joint** (rev / continuous / prismatic / fixed) — modal picks parent + child from existing links
- **Selection** — click a node/edge to see its name, kind, and edit its mass (links only)
- **Delete** selected
- **Live query** with the predicate language (`KIND(...)`, `INHERITS(...)`, etc.)
- **Export** to `.hymeko` / URDF / SDF / DOT (rerouted through the same WASM emitters used everywhere else)
- **File load** for a local `.hymeko` source

## What the MVP does NOT do (yet)

- **Comment / formatting preservation** — text mutations may reflow the source. Round-trip is structural, not byte-identical.
- **Custom layer kinds** — only the standard kinematic kinds (link / *_joint) are wired into the palette. To add new kinds, edit `editor.js`'s palette buttons.
- **Multi-file / `using` import editing** — the editor edits a single source file at a time.
- **Drag-to-create edges** — edges are created via the modal, not by dragging from one node to another. This is the natural follow-up.
- **Undo / redo** — the textarea's native undo works for text edits; structural edits don't have a history stack yet.
- **SysML export** — easy to add (`lastIR.to_sysml(...)` once exposed in `hymeko_wasm/src/wasm.rs`).

## How to extend

| change | where |
|---|---|
| Add a new layer kind to the palette | `editor.js` — add a `<button data-add="my_kind">` in `index.html`, then handle the kind in the palette-button handler |
| Add a new export format | `editor.js` — extend the `<select id="exportFormat">` and add a case to `btnExport` |
| Customise canvas style | `editor.js` — the `cytoscape({ style: [...] })` array |
| Replace text-mutation engine | `editor.js` — `addLink` / `addJoint` / `setMass` / `deleteDecl`. Could later route through a Rust-side rewriter that preserves comments + formatting |

## Files

```
docs/editor/
├── index.html      Layout: toolbar + palette + canvas + properties + source
├── editor.css      Vanilla CSS, no framework
├── editor.js       Controller, ~350 lines; uses Cytoscape.js (CDN) + WASM
├── pkg/            wasm-pack output (build with `wasm-pack build hymeko_wasm`)
└── README.md       This file
```

## See also

- [Quickstart: Use the WASM bundle in a browser](../book/src/quickstart/12-wasm.md) — the underlying API
- `docs/demo/` — older display-only demo (kept for reference)
