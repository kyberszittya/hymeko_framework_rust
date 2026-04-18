# Plan 05 — Extract `hymeko_hre` crate from `hymeko_core`

**Branch:** `refactor/extract-hymeko-hre` (off `dev/query_engine`)
**Status:** in progress — 2026-04-18
**Deliverable:** new workspace crate `hymeko_hre` owning the hypergraph engine orchestrator, with `hymeko_core` trimmed accordingly and all consumers (query, daemon, cli, py, client) updated.

---

## Why

`hymeko_core` has grown to hold IR, resolution, module store, tensor primitives, tensor neural-network operators, hypergraph views, *and* the stateful `HypergraphEngine` orchestrator. The last of those — a registry plus a `TensorCoo` builder plus `compile_*_expansion_core` methods plus an optional `iceoryx2` subscriber — is a distinct concern that:

- does not belong on the path of anyone using `hymeko_core` purely for IR or tensor primitives,
- is a natural boundary for swapping in different expansion strategies or IPC transports,
- is the visual node labelled "Hypergraph Rewriting Engine" in `architecture/hre_rewriting_engine/architecture.mermaid`, yet currently lives inside the same crate as its dependencies.

Extracting it gives us a crate that matches the diagram and lets downstream consumers opt into the engine (with its `iceoryx2` baggage) without dragging the whole surface through every build.

## Boundary decision — tight split (engine-only)

**In scope (moves to `hymeko_hre`):**

- `hymeko_core/src/engine/mod.rs`
- `hymeko_core/src/engine/hypergraphengine.rs`
- `hymeko_core/src/engine/hypergraphengine_impl.rs`
- `hymeko_core/src/engine/hymeko_subscriber.rs` (behind `ipc` feature)

**Stays in `hymeko_core`:**

- `traversal/` (`graphview.rs`, `hypergraphview.rs`, `graph_traversal.rs`, `decltreeview.rs`)
- `tensor/` (all of it — `conv/`, `mesh_nn/`, `representations/`, primitives)
- `ir/`, `common/`, `resolution/`, `module_store/`, `writers/`

### Why not the wider split

The initial middle-option plan proposed moving `traversal/` with the engine. Inspection revealed a cyclic dependency that would break the split:

- `hymeko_core::tensor::common`, `tensor::tensor`, `tensor::common_traversal`, `tensor::message_passing`, `tensor::mesh_nn`, `tensor::conv::{hgnn,gcn_clique,signed_hgnn}`, `tensor::representations::{tensor_coo_representation,tensor_csr_representations}` all `use crate::traversal::hypergraphview::HyperGraphView`.
- `hypergraphview.rs` itself depends on `tensor::aggregation`, `tensor::common::Real`, `tensor::tensor_val`, `tensor::representations::tensor_coo::TensorInc`.

Pulling `traversal/` out would force either (a) a massive concurrent tensor split dragging HGNN/mesh/conv ops across the crate boundary, or (b) a cyclic `core ↔ hre` dependency. Option (a) is a separate refactor and belongs in a future plan (see "Follow-up" below). Option (b) is not buildable.

The tight split avoids the cycle because `engine/` is a pure consumer of `core::{ir, tensor, resolution, traversal}` with no reverse edges.

## Phases

### Phase 1 — Docs & diagrams (this doc + features.md + overview_crates.mermaid)
- `docs/plans/05_hre_extraction/plan.md` — this file.
- `docs/plans/05_hre_extraction/features.md` — feature list + code examples for the new crate's public surface.
- `architecture/overview_crates.mermaid` — new crate-level dependency diagram.
- Edit `architecture/hre_rewriting_engine/architecture.mermaid` to relabel the HRE node as `hymeko_hre`.
- Edit `architecture/README.md` to index the new diagram.

### Phase 2 — Scaffold `hymeko_hre`
- `hymeko_hre/Cargo.toml` with:
  - `hymeko_core = { path = "../hymeko_core" }`
  - `rustc-hash`, `rayon` (workspace)
  - Mirror `ipc = ["dep:iceoryx2", "hymeko_core/ipc"]` and `arrow-schema = ["hymeko_core/arrow-schema"]` features (the `shared_state` types live in core behind these feature gates).
- Register in workspace `Cargo.toml` `members`.
- `hymeko_hre/src/lib.rs` with `pub mod engine;` and a convenience `pub use engine::hypergraphengine::HypergraphEngine;`.

### Phase 3 — Move engine module
- `git mv hymeko_core/src/engine hymeko_hre/src/engine`.
- Rewrite `crate::ir::...`, `crate::resolution::...`, `crate::tensor::...`, `crate::traversal::...` → `hymeko_core::...`.
- Intra-engine refs (`crate::engine::hypergraphengine::HypergraphEngine`) remain as `crate::engine::...` since the module is now local.

### Phase 4 — Trim `hymeko_core`
- Remove `pub mod engine;` from `hymeko_core/src/lib.rs`.
- `cargo check -p hymeko_core` — green.
- Run `hymeko_core` tests to ensure no regression.

### Phase 5 — Update consumers
For each downstream crate:
1. Add `hymeko_hre = { path = "../hymeko_hre" }` to `Cargo.toml`.
2. Replace `hymeko::engine::...` (core's lib name is `hymeko`) with `hymeko_hre::engine::...`.
3. `cargo check -p <crate>`.

Consumer inventory (to be confirmed during Phase 5):
- `hymeko_query` — likely none (uses IR/view, not engine directly).
- `hymeko_daemon` — heavy user: `HypergraphEngine`, subscriber types, `write_*_into_raw`.
- `hymeko_cli` — uses engine to compile expansions.
- `hymeko_py` — exposes engine to Python.
- `hymeko_client` — subscriber path.

### Phase 6 — Verify
- `cargo check --workspace --all-features`
- `cargo test --workspace`
- CLI smoke: `cargo run -p hymeko_cli -- <flags for mini_arm.hymeko>` — produce URDF, compare to prior output shape.

### Phase 7 — Record
- `docs/changelog/changelog_20260418.md` — new entry describing the extraction.
- Three commits on the branch: crate move, diagrams, changelog.

## Risks

- **Feature-gate plumbing.** `hymeko_subscriber` and the `write_*_into_raw` methods are `#[cfg(feature = "ipc")]` and pull `iceoryx2` types via `core::tensor::shared_state`. The new crate must re-expose an `ipc` feature that enables both its own `iceoryx2` and `hymeko_core/ipc`. Missing this will manifest as "unknown type `HypergraphWeights`" during conditional builds.
- **Lib name collision.** `hymeko_core` has `name = "hymeko"` in `[lib]`. Consumers currently `use hymeko::engine::...`. After the move the imports become `use hymeko_hre::engine::...`. Any stale import will fail loudly — good, no silent breakage.
- **Python bindings.** If `hymeko_py` has `#[pyclass]` for engine types, the macro expansions stay in `hymeko_py` but the underlying types must be re-imported. Verify maturin build before declaring done.

## Follow-up (not in this plan)

- **`hymeko_hnn` crate.** The tensor/hypergraph-ops tangle identified in the cycle analysis (HGNN, mesh_nn, message passing, signed conv, COO/CSR representations) wants its own crate, but that is a concurrent refactor of ~10 tensor files and should land after this plan is merged.
- **Traversal crate.** Once the tensor split lands, `traversal/` can join `hymeko_hre` or a dedicated `hymeko_graph` crate.
