# Crate map

The framework is split across ~15 Rust crates and 2 Python packages. Each owns one concern. New users typically only need 3–4.

## Core (always)

| crate | role |
|---|---|
| **`parser`** | Lexer + parser → syntax tree |
| **`hymeko_core`** | The IR (`Ir`, `DeclId`, `nodes`, `edges`, `arcs`), name resolution, tensor representations, structural entropy + HOSVD |
| **`hymeko_query`** | Predicate language, `QueryEngine`, named queries, transform plugin API (`DomainTransform`, `TransformRegistry`), template engine |
| **`hymeko_formats`** | Built-in format plugins: URDF, SDF, MJCF, DOT, Mermaid, Gazebo world, torch_dataflow. The first consumer of the plugin API |

## Front-ends

| crate | role |
|---|---|
| **`hymeko_cli`** | `hymeko` binary: `parse`, `compile`, `emit`, `query`, `pgraph` subcommands |
| **`hymeko_py`** | PyO3 wheel: `hymeko.compile_description`, `parse_hymeko_rs`, `enumerate_k_cycles_rs`, `compile_clique_tensor_expansion`, `PyHypergraphIR` class |
| **`hymeko_wasm`** | WebAssembly bundle for browser editors / demos |

## Specialized engines

| crate | role |
|---|---|
| **`hymeko_pgraph`** | P-graph axiom feasibility (MSG, SSG, ABB) — chemical-process synthesis + neural-arch search |
| **`hymeko_compute`** | Vulkan compute kernels (vector_add, signed_spmv, force_directed) — GPU acceleration for cycle ops |
| **`hymeko_monitor`** | RV (runtime verification) STL monitor over signed-incidence hypergraphs |
| **`hymeko_hnn`** | Hypergraph-NN tensor message passing (research / experimental) |
| **`hymeko_hre`** | Hypergraph runtime engine — hypergraph operations + clique tensor expansion |
| **`hymeko_graph`** | Graph utilities (pruner, clustering algorithms) |

## Daemon / IPC (optional)

| crate | role |
|---|---|
| **`hymeko_daemon`** | Long-running server for incremental IR updates (iceoryx2 IPC) |
| **`hymeko_client`** | Daemon client |
| **`hymeko_wire`** | Shared wire format |
| **`hymeko_clifford`** | Clifford-algebra utilities |
| **`hymeko_mcp`** | MCP server (Anthropic Model Context Protocol) for AI-assisted editing |

## Python research code (separate from wheels)

| package | role |
|---|---|
| **`signedkan_wip`** | HSiKAN research code: `signedkan.py`, `mixed_arity_signedkan.py`, `hymeko_train_walker.py`, baselines (SGCN, SiGAT, SGT), benchmarks |
| **`python/ehk_torch_stub`** | Tier-3 layer surface (`SignedKANLayer`, `ArityMixer`, `WalkLayer`, `SignedClassifier`); after May 2026 cleanup, delegates to real `signedkan_wip.signedkan` when importable |

## Dependency direction

```
                  parser
                    ↓
                hymeko_core
                    ↓
                hymeko_query  ← (everything depends on this)
                  ↙   ↓   ↘
       hymeko_formats  hymeko_pgraph  hymeko_compute …
                  ↓
       hymeko_cli  hymeko_py  hymeko_wasm
                                ↑
                       signedkan_wip (consumes hymeko_py)
```

Anything below depends on anything above. There are no upward references.

## What if I just want to use HyMeKo

Most users only ever touch:
- `hymeko_cli` (the binary)
- `hymeko_py` (the wheel) — for Python-side scripting
- `hymeko_wasm` — if you embed in a browser

Adding a new format target only needs `transforms/<name>/` (a directory of plain text files — no Rust). See [Add a new format](../recipes/add-a-format.md).

## What if I want to extend the IR itself

Then you're touching `hymeko_core` and probably `hymeko_query` too. See [Extension points](./extension-points.md).
