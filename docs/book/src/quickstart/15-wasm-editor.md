# Quickstart: Visual editor (WASM, in-browser)

A point-and-click editor for `.hymeko` sources, running entirely in the browser via the `hymeko_wasm` bundle. Source-text-as-truth: every UI edit is a string transformation on the canonical `.hymeko` source plus a re-parse.

## Open the editor

```bash
# 1. Build the WASM bundle (one-time)
cd hymeko_wasm
wasm-pack build --target web --release --out-dir ../docs/editor/pkg

# 2. Serve docs/editor/ over HTTP (any static server)
cd ../docs/editor
python3 -m http.server 8000
# Open http://localhost:8000/
```

> Use HTTP, not `file://` — wasm-bindgen's ES-module loading needs a real origin.

When deployed via the GitHub Pages workflow, the editor lives at `<site>/editor/`.

## What you'll see

- **Toolbar** (top): Redraw, export-format selector, Download, Load file, Example
- **Palette** (left): Add Link / Add Joint (4 kinds) + selection panel + stats + live query
- **Canvas** (center): Cytoscape-rendered hypergraph — vertices as ellipses, hyperedges as rounded boxes, signed arc-refs as coloured arrows (blue +1, red −1, grey ~0)
- **Source pane** (bottom, collapsible): the canonical `.hymeko` text. Edit it directly; the canvas redraws on a 400 ms debounce.

## A complete edit cycle

1. Click **Example** — loads a 2-link continuous-joint robot
2. Click on `base_link` (an ellipse) — the selection panel populates with name, kind, mass field
3. Type `7.5` into Mass and click **Set mass** — the canvas redraws; the source's `mass` line updates
4. Click **+ Revolute joint** — modal asks for name + parent + child (selected from existing links)
5. Click **Download** with format=URDF — gets a URDF reflecting your edit

Every UI action is a text mutation: the source pane is the single source of truth, the canvas is a live read.

## Edit operations

| operation | what it does to the source |
|---|---|
| **+ Link** | Inserts `<name>: el.link { mass <m>; }` into the main context |
| **+ Joint** | Inserts `@<name>: + <isa> <kind> { (+ parent, − child); }` |
| **Set mass** | Find/replace `mass <num>;` in the link's body (depth-aware) |
| **Delete** | Splices out the declaration through its matching `}` (brace-counting; survives nested bodies) |
| **Source edit** | Free-form textarea editing; auto-recompile on 400 ms idle |

## Export formats

The dropdown next to **Download** offers:

- `.hymeko` — the source text as-is (round-tripped)
- `URDF` — `lastIR.to_urdf("robot")`
- `SDF` — `lastIR.to_sdf("robot")`
- `DOT` — `lastIR.to_dot("robot")`

(SysML and MJCF are easy to add — see `editor.js`'s `btnExport` handler.)

## What this MVP does NOT do

- **No comment / formatting preservation** — text mutations may reflow whitespace. Round-trip is structural, not byte-identical.
- **No drag-to-create-edges** — joints are made via the modal, not by dragging from one node onto another. Natural follow-up.
- **No undo/redo for structural edits** — the textarea's native undo works for source edits; structural ones don't have a history stack yet.
- **Standard kinematic kinds only** — `link`, `rev_joint`, `conti_joint`, `prismatic_joint`, `fixed_joint`. Adding new layer kinds is a 2-line edit in `editor.js`.

## Why source-text-as-truth

The framework's IR is rich but the textual `.hymeko` is the canonical interchange format. By making the textarea the single source of truth:

- **Round-trip is automatic** — what you see on the canvas is exactly what `hymeko emit` consumes
- **Any tool that reads `.hymeko` works on editor output** — Python wheel, CLI, other emitters
- **Power users can drop into the source pane** — fall back to text editing for anything the UI doesn't expose

The cost: no comment preservation, no per-decl positional metadata. Worth it for an MVP; later versions can route through a Rust-side rewriter that preserves both.

## Files

```
docs/editor/
├── index.html      Layout
├── editor.css      Vanilla CSS, no framework
├── editor.js       Controller (~450 lines), Cytoscape.js + wasm-bindgen
├── pkg/            wasm-pack output (built locally / in CI)
└── README.md       Build + extension notes
```

## See also

- [Use the WASM bundle in a browser](./12-wasm.md) — the underlying API surface
- [Recipes: Add a new layer kind](../recipes/add-a-layer-kind.md) — how to extend what the palette offers
