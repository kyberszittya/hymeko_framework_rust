# Layered architecture

```
┌─────────────────────────────────────────────────┐
│ User front-ends                                  │
│ CLI · Python wheel · WASM bundle · MCP server   │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Codegen layer                                    │
│ hymeko_formats (URDF, SDF, MJCF, DOT, torch …)  │
│ hymeko_pgraph (MSG / SSG / ABB)                 │
│ hymeko_compute (Vulkan kernels)                 │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Query layer                                      │
│ hymeko_query                                     │
│ - Predicate language (string + typed)            │
│ - QueryEngine (matches, named queries)           │
│ - Template engine (#each, bind:, field:)         │
│ - DomainTransform plugin API                     │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ IR layer                                         │
│ hymeko_core::ir                                  │
│ - Ir { decl_nodes, nodes, edges, arcs }          │
│ - DeclId, NodeId, EdgeId, ArcId                  │
│ - StringTable, name resolution                   │
│ - Tensor representations (COO, dense, HOSVD)     │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Parser layer                                     │
│ parser/                                          │
│ - Tokens, syntax tree                            │
└─────────────────────────────────────────────────┘
```

Each layer talks ONLY to the layer immediately below. There are no cross-layer shortcuts. New format / new query / new layer kind almost always lives in the codegen or query layer.

## See also

- [Crate map](./crate-map.md) — concrete crate names per layer
- [Data flow](./data-flow.md) — how a `.hymeko` becomes a URDF
- [Extension points](./extension-points.md) — where to plug new components
