# Quickstart: Parse a .hymeko file

What you'll do: take a small `.hymeko` source, parse it, and inspect the resulting IR.

## Prereqs

```bash
cargo build --release  # builds target/release/hymeko
```

## A minimal source

Create `/tmp/hello.hymeko`:

```hymeko
hello {
    using nn.tensors as ten;
}

context {
    x: ten.t_input  { shape [3]; }
    y: ten.t_output { shape [1]; }
}
```

This declares two tensor nodes — an input `x` and an output `y` — both inheriting from the meta-types defined in `data/nn/meta_nn.hymeko`.

## Parse it

```bash
target/release/hymeko parse /tmp/hello.hymeko
```

Output:

```
Parsed: hello
  2 nodes, 0 edges, 0 arcs
```

The CLI's `parse` subcommand runs the lexer + parser + name-resolver and prints a one-line summary. To see the full tree:

```bash
target/release/hymeko parse /tmp/hello.hymeko --json | jq .
```

## Same thing in Python

After `pip install hymeko-*.whl` (built from `hymeko_py/`):

```python
import hymeko

src = open("/tmp/hello.hymeko").read()
tree = hymeko.parse_hymeko_rs(src)
print(tree["name"])           # "hello"
print(len(tree["items"]))     # 1 (the context wrapper)

ctx = tree["items"][0]
for child in ctx["body"]:
    print(child["name"], "->", child.get("bases"))
# x -> [{'path': ['ten', 't_input']}]
# y -> [{'path': ['ten', 't_output']}]
```

## What just happened

1. **Lexer + parser** (`parser/`) tokenized and built a syntax tree.
2. **Name resolver** (`hymeko_core::resolution`) turned `ten.t_input` into a concrete `DeclId` reference, walking the `using nn.tensors as ten` import.
3. **IR builder** (`hymeko_core::ir`) populated the `Ir` struct — `decl_nodes`, `nodes`, `edges`, `arcs` arrays.

The IR is now ready for any downstream consumer: a query engine, a codegen target, or the WASM/Python bridges.

## Next

- [Emit URDF for ROS](./02-emit-urdf.md) — turn an IR into a URDF robot description
- [Query an IR](./07-query.md) — find all edges of a kind, walk inheritance
- [Concepts: The IR](../concepts/ir.md) — what's actually inside the parsed tree
