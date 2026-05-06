# Quickstart: Use the WASM bundle in a browser

`hymeko_wasm` compiles HyMeKo to WebAssembly. A user can paste a `.hymeko` source into a browser, see the IR, query it, emit URDF / SDF / DOT — zero install.

## Build

```bash
cd hymeko_wasm
wasm-pack build --target web --release
# Produces hymeko_wasm/pkg/{hymeko_wasm.js, hymeko_wasm_bg.wasm, ...}
```

## Live demo

The repo ships `docs/demo/` — drop the `pkg/` output next to `demo.html` + `demo.js` and serve:

```bash
cd docs/demo
python3 -m http.server 8000
# Open http://localhost:8000/demo.html
```

The demo:
- Textarea on the left for `.hymeko` source
- Live re-parse on every change
- Right pane shows: node/edge counts, predicate query box, snapshot JSON, DOT render

## API surface (from JavaScript)

```javascript
import init, { compile_doc } from "./pkg/hymeko_wasm.js";

await init();
const ir = compile_doc(`
hello {}
context { x: t_input { shape [3]; } y: t_output { shape [1]; } }
`);
console.log(ir.node_count(), "nodes");
console.log(ir.to_dot("hello"));
console.log(JSON.parse(ir.snapshot_json()));
console.log(ir.query("KIND(t_input)"));
```

The WASM `CompiledDoc` exposes:
- `node_count() / edge_count() / arc_count()`
- `to_urdf(name) / to_sdf(name) / to_dot(name)` — same emitters as the Python wheel
- `snapshot() / snapshot_json()` — IR introspection (after the May 2026 cleanup, both wasm + Python go through `hymeko_formats::snapshot`)
- `query(predicate) / query_count(predicate)` — the same string-predicate language as the CLI / Python

## Why WASM matters

- **No install** — collaborators can edit and visualize HyMeKo without setting up the toolchain
- **Same engine as Python + CLI** — after cleanup, all three call into shared `hymeko_formats` / `hymeko_query` modules. WASM is not a stripped-down version.
- **Editor potential** — the snapshot JSON is graph-editor-ready (nodes / edges with sign-coloured arcs). Future work: graph-editor frontend that emits HyMeKo back from canvas state.

## Next

- [Use the PyO3 wheel from Python](./11-pyo3-wheel.md) — the same engine on the server side
- [Concepts: The IR](../concepts/ir.md) — what's in `snapshot_json()`'s JSON
