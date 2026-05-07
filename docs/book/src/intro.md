# HyMeKo Framework

**Hypergraph Model Cognition Framework** — a declarative DSL + Rust toolchain for designing, transforming, and emitting from hypergraph IRs.

## What is HyMeKo

HyMeKo describes things as **signed hypergraphs**: a vocabulary of nodes (vertices), edges (hyperedges grouping multiple nodes with `+/-/~` signs), tags, and inheritance. The same source file can compile to:

| target | use case |
|---|---|
| **URDF** | ROS robotics |
| **SDF 1.7** | Gazebo simulation |
| **MJCF** | MuJoCo rigid-body sim |
| **DOT / Mermaid** | graph visualization |
| **PyTorch nn.Module** | neural-network architectures (HSiKAN, GNNs) |
| **Gazebo world** | simulator scenes |
| **SysML 2 textual** | model-based systems engineering (Papyrus / Modelix / OMG playground) |

A single hypergraph IR feeds all of these via the **template-driven codegen pipeline**. New formats are registered as `transforms/<name>/{queries.hymeko, template.<ext>}` — no Rust changes needed for the common case.

## Why use it

- **One source, many targets** — robot in URDF and SDF and MJCF without re-modelling
- **Strongly typed** — inheritance + tags catch malformed designs before emission
- **Hypergraph-native** — first-class hyperedges with sign semantics, not "edges + groups bolted on"
- **Research-grade** — used for signed-cycle KAN architectures, P-graph axiom feasibility, tensor decomposition

## How to read this book

- **[Quickstarts](./quickstart/01-parse.md)** — one tutorial per use case. Pick the target you want and follow it end-to-end. Self-contained, runnable.
- **[Architecture](./architecture/crate-map.md)** — how the crates fit together, where extension points live.
- **[Concepts](./concepts/ir.md)** — the IR, queries, templates, codegen pipeline.
- **[Extension recipes](./recipes/add-a-format.md)** — adding formats, layer kinds, queries.
- **[Research code](./research/signedkan-overview.md)** — `signedkan_wip/` orientation; what's stable vs WIP.

## Build status & links

- Repo: <https://github.com/kyberszittya/hymeko_framework_rust>
- CLI: `cargo build --release` produces `target/release/hymeko`
- Python wheel: `cd hymeko_py && maturin build --release`
- WASM: `cd hymeko_wasm && wasm-pack build --target web`
- Tests: `cargo test --workspace` (~660 tests, ~30 s)

## Where to start

If you've never seen HyMeKo: read [Parse a .hymeko file](./quickstart/01-parse.md) first.
If you came here for ROS: jump straight to [Emit URDF](./quickstart/02-emit-urdf.md).
If you came here for ML research: [Build an HSiKAN architecture](./quickstart/08-hsikan-architecture.md).
If you want to design visually: open the [WASM editor](./quickstart/15-wasm-editor.md) (or visit `/editor/` on the published site).
