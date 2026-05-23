# HyMeKo browser demo (M1)

Parse and query `.hymeko` files directly in the browser via WebAssembly.

## How to run

Serve the repo root over HTTP (the demo fetches
`examples/paper/hymeko_robot.hymeko` relative to the demo page) and open
the demo page:

```bash
# from the repo root
python3 -m http.server 8000 --bind 127.0.0.1
# then visit:
#   http://127.0.0.1:8000/docs/demo/index.html
```

## How to rebuild the wasm bundle

```bash
wasm-pack build hymeko_wasm --target web --out-dir ../docs/demo/pkg --release
```

## What the demo does

1. Load a `.hymeko` source file (file picker, canonical-example button,
   or paste into the textarea).
2. Click **Compile** — invokes `parse_and_compile(src)` in the wasm
   module and prints the IR node/edge/arc counts.
3. **Query** — run a predicate-string query against the compiled IR.
   Supported atoms: `KIND(x)`, `INHERITS(x)`, `SCOPEDIN(x)`,
   `HASARCREF(±1, inner)`, joined by `AND`. The canonical example
   reproduces the five paper queries from `queries/standard.qlist`.
4. **Emit** — URDF, SDF, DOT, or graph-viewer-ready snapshot JSON.

## Files

| Path                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `index.html`             | UI                                                  |
| `demo.js`                | ES-module glue                                      |
| `pkg/hymeko_wasm.js`     | `wasm-pack` ES-module bindings                      |
| `pkg/hymeko_wasm_bg.wasm`| Compiled WebAssembly (≈448 KB, release + wasm-opt) |
| `pkg/hymeko_wasm.d.ts`   | TypeScript definitions                              |

## Python parity

The same surface is available via the `hymeko` Python wheel
(`hymeko_py` → `maturin build`): `ir.snapshot_json()`, `ir.to_dot()`,
`ir.query()`, `ir.to_urdf()`, `ir.to_sdf()` — identical output shape.

## Next milestones

- **M2**: force-directed SVG hypergraph viewer using the snapshot JSON.
- **M3**: predicate-query matches highlighted in the M2 viewer.
- **M4**: wgpu live message-passing animation over the viewer.
- **M5**: VS Code extension using the same wasm bundle.
