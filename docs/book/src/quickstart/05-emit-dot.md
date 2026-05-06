# Quickstart: Visualize with DOT

DOT (Graphviz) is the universal graph format. Useful for inspecting any HyMeKo IR — robotics, neural-net architectures, P-graph systems all visualize with the same `--format dot` flag.

```bash
target/release/hymeko emit \
    data/robotics_imported/wam/wam.hymeko \
    --format dot \
    --name wam7 \
    -o /tmp/wam7.dot

# Render to PNG / SVG
dot -Tpng /tmp/wam7.dot -o /tmp/wam7.png
```

The emitter renders:
- **Vertices** as ellipses (filled `#EEF1F5`)
- **Hyperedges** as rounded boxes (filled `#D7E4F5`)
- **Signed arc-refs** as coloured arrows: blue (+1), red (-1), grey (~0)
- **Arrow heads** matching sign: `normal` (+), `inv` (-), `odot` (~)

## In Python

```python
import hymeko

src = open("data/hsikan/arch_mixed_k34.hymeko").read()
doc = hymeko.compile_description(src)
dot = doc.to_dot("hsikan_arch")
open("/tmp/hsikan.dot", "w").write(dot)
```

Then `dot -Tsvg /tmp/hsikan.dot -o /tmp/hsikan.svg` and view in any browser.

## Live in a browser

The WASM bundle exposes `to_dot` directly (same engine as the Python wheel). See [`docs/demo/`](https://github.com/kyberszittya/hymeko_framework_rust/tree/master/docs/demo) for a live editor that renders DOT inline as you type. After the May 2026 cleanup, both Python and WASM go through `hymeko_formats::snapshot::emit_dot_graph` — single source of truth.

## DOT for non-robotics IRs

DOT is format-agnostic — works on any HyMeKo IR:

```bash
# Visualize an HSiKAN signed-cycle architecture
target/release/hymeko emit data/nn/hsikan_mixed.hymeko --format dot -o /tmp/hsikan.dot

# Visualize a P-graph for chemical-process synthesis
target/release/hymeko emit data/pgraph/hda.hymeko --format dot -o /tmp/hda.dot

# Visualize an anatomical hypergraph
target/release/hymeko emit data/anatomy/foot.hymeko --format dot -o /tmp/foot.dot
```

## Next

- [Query an IR](./07-query.md) — filter the structure before visualizing
- [Use the WASM bundle in a browser](./12-wasm.md)
