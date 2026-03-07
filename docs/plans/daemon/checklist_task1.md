# Checklist for HyMeKo Daemon Refactor and Local Loop Acceleration

## Phase 1 Execution Checklist

- [ ] **Task 1.1: Workspace Segregation**
  - [X] Create a new `hymeko_daemon` Cargo crate.
  - [X] Configure a Cargo workspace at the repository root to link the crates.
  - [X] Extract all network, server, and Python API bindings completely out of the core `hymeko` library and into the new daemon crate.
    - [X] Move all relevant code files and update module paths accordingly.
    - [X] Ensure that the core `hymeko` crate remains free of any network
    - [X] Create hymeko python bindings in `hymeko_py` crate and move all relevant code files and update module paths accordingly.
  - [X] Update CI/CD pipelines to build and test both crates independently, ensuring that the core library can be validated without the daemon dependencies.
  - [X] Verify that the new structure allows for clean separation of concerns and that the core library can be used without pulling in unnecessary dependencies.
  - [X] Extract magic numbers from test cases

- [ ] **Task 1.2: FxHash Integration (Local Loop Acceleration)**
  - [ ] Add `rustc-hash` to the dependencies in `hymeko/Cargo.toml`.
  - [ ] Open `hymeko/src/engine/hypergraphengine.rs`.
  - [ ] Replace `std::collections::HashMap` with `rustc_hash::FxHashMap` for `node_registry`, `edge_registry`, and `ir_repository`.
  - [ ] Update their initialization to use `FxHashMap::default()`.
  - [ ] Locate the tensor expansion loops and swap the temporary `decl_to_csr_node` and `decl_to_csr_edge` mappings to `FxHashMap`.

- [ ] **Task 1.3: Deterministic B-Tree Indexing (Hash Stability)**
  - [ ] Open the compiler's resolution module (`hymeko/src/resolution/resolve.rs`).
  - [ ] Change the `by_path` mapping in your `Index` struct from a standard `HashMap` to a `std::collections::BTreeMap`.
  - [ ] Open `hymeko/src/ir/hash.rs` and `hymeko/src/ir/canonical_hash.rs`.
  - [ ] Delete the manual vector allocation and sorting logic that was previously used to order `PathKey`s. Route the hasher directly through the `BTreeMap`'s native lexicographical iterator.