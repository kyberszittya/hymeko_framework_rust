# `hymeko_hre` — Feature list & code examples

The `hymeko_hre` crate (Hypergraph Rewriting Engine) owns the stateful orchestrator that compiles IR into tensor expansions and, under `ipc`, streams those expansions into shared memory for zero-copy consumers.

## Features

| # | Feature | API surface | Feature gate |
|---|---------|-------------|--------------|
| F1 | Named IR registry | `HypergraphEngine::register_ir`, `get_ir` | default |
| F2 | Idempotent node/edge interning | `get_or_create_node`, `get_or_create_edge` | default |
| F3 | Arc insertion (by name or ID) | `add_arc_by_name`, `add_node`, `add_edge`, `add_arc` | default |
| F4 | Direct IR → `TensorCoo` compilation | `compile_from_ir` | default |
| F5 | Star expansion core compute | `compile_star_expansion_core::<F: Real>` | default |
| F6 | Clique expansion core compute | `compile_clique_expansion_core::<F: Real>` | default |
| F7 | Zero-copy raw-buffer write | `write_tensor_into_raw`, `write_star_tensor_into_raw`, `write_star_expansion_into_raw` | `ipc` |
| F8 | `iceoryx2` subscriber with topology-vs-weight event classification | `HymekoSubscriber`, `MemoryEvent` | `ipc` |

All F1–F6 compile with the default feature set (no `iceoryx2` pull). F7–F8 require `--features ipc` and transitively enable `hymeko_core/ipc`.

---

## Code examples

### Example 1 — Build a small hypergraph manually (F2, F3)

```rust
use hymeko_hre::HypergraphEngine;

let mut engine = HypergraphEngine::new();
let _j1 = engine.get_or_create_edge("joint_1");
let _base = engine.get_or_create_node("base_link");

engine.add_arc_by_name(0, "base_link", "joint_1", 1.0)?;
engine.add_arc_by_name(0, "upper_arm", "joint_1", 1.0)?;

assert_eq!(engine.current_nodes, 2);
assert_eq!(engine.current_edges, 1);
```

### Example 2 — Register an IR and compile it to a COO tensor (F1, F4)

```rust
use hymeko_hre::HypergraphEngine;
use hymeko::ir::ir::Ir;                // `hymeko` is the lib-name of `hymeko_core`

let ir: Ir = /* produced by ModuleStore::compile().take_last_ir() */;
let mut engine = HypergraphEngine::new();
engine.register_ir("mini_arm", ir);

let tensor = engine.compile_from_ir("mini_arm")
    .expect("IR should be registered");
println!("nnz = {}, slices = {}", tensor.len(), tensor.num_slices);
```

### Example 3 — Star expansion in f32 for GPU hand-off (F5)

```rust
use hymeko_hre::HypergraphEngine;
use hymeko::ir::ir::Ir;

let engine = HypergraphEngine::new();
let ir: Ir = /* ... */;

let star: hymeko::tensor::representations::tensor_coo::TensorCoo<f32>
    = engine.compile_star_expansion_core::<f32>(&ir);

// Feed `star` into a DLPack exporter or nalgebra-sparse bridge downstream.
```

### Example 4 — Clique expansion for classical GNN pipelines (F6)

```rust
use hymeko_hre::HypergraphEngine;

let engine = HypergraphEngine::new();
let clique = engine.compile_clique_expansion_core::<f64>(&ir);
// Shape: (num_slices = |E|, dim_i = |V|+|E|, dim_j = |V|+|E|)
```

### Example 5 — Stream star expansion into an `iceoryx2` sample (F7)

```rust
# #[cfg(feature = "ipc")]
# fn demo() -> Result<(), &'static str> {
use hymeko_hre::HypergraphEngine;
use hymeko::tensor::shared_state::{ExpansionHeader, ExpansionKind};

let engine = HypergraphEngine::new();
let ir = /* ... */;
let coo = engine.compile_star_expansion_core::<f32>(&ir);
let header = ExpansionHeader::new(
    ExpansionKind::Star3D,
    coo.len(),
    coo.num_slices,
    coo.dim_i,
    coo.dim_j,
);

// Caller provides pointers into an iceoryx2 sample payload.
unsafe {
    HypergraphEngine::write_tensor_into_raw(
        &header, &coo,
        header_ptr, k_ptr, i_ptr, j_ptr, values_ptr,
        capacity,
    )?;
}
# Ok(()) }
```

### Example 6 — Subscribe to hypergraph memory events (F8)

```rust
# #[cfg(feature = "ipc")]
# fn demo() -> Result<(), Box<dyn std::error::Error>> {
use hymeko_hre::engine::hymeko_subscriber::{HymekoSubscriber, MemoryEvent};

let mut sub = HymekoSubscriber::new("hymeko/weights")?;
loop {
    match sub.poll_memory()? {
        Some(MemoryEvent::MappingUpdate(_)) => { /* re-map tensor layout */ }
        Some(MemoryEvent::WeightStream(_))  => { /* fast-path weight tick */ }
        None => break,
    }
}
# Ok(()) }
```

### Example 7 — CLI integration pattern (how `hymeko_cli` uses the engine)

```rust
use hymeko_hre::HypergraphEngine;
use hymeko::module_store::module_store::ModuleStore;

fn compile_and_expand(path: &str) -> anyhow::Result<()> {
    let mut store = ModuleStore::new();
    store.compile(path)?;
    let ir = store.take_last_ir()?;

    let engine = HypergraphEngine::new();
    let star = engine.compile_star_expansion_core::<f32>(&ir);
    let clique = engine.compile_clique_expansion_core::<f32>(&ir);

    println!("star nnz = {}, clique nnz = {}", star.len(), clique.len());
    Ok(())
}
```

---

## Non-features (explicit)

- **Rewrite-pattern matching** — lives in `hymeko_query::rewrite`, not here. `hymeko_hre` is the compilation/expansion engine, not the pattern-rewrite engine, despite the historical "HRE = Hypergraph Rewriting Engine" label in architecture docs.
- **Graph views / traversal traits** — stay in `hymeko_core::traversal` for now (see plan.md § "Why not the wider split").
- **Hypergraph neural-network ops** (HGNN, signed HGNN, clique GCN, message passing) — stay in `hymeko_core::tensor::conv` and `hymeko_core::tensor::message_passing` pending a follow-up `hymeko_hnn` extraction.
