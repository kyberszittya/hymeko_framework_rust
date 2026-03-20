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

- [X] **Task 1.2: FxHash Integration (Local Loop Acceleration)**
  - [X] Add `rustc-hash` to the dependencies in `hymeko/Cargo.toml`.
  - [X] Open `hymeko/src/engine/hypergraphengine.rs`.
  - [X] Replace `std::collections::HashMap` with `rustc_hash::FxHashMap` for `node_registry`, `edge_registry`, and `ir_repository`.
  - [X] Update their initialization to use `FxHashMap::default()`.
  - [X] Extract current Python API implementations containing tensor expansion logic from the Python API so the Python API is essentially a thin wrapper around the core library. This will allow us to benchmark the core library's performance improvements without Python overhead.
  - [X] Locate the tensor expansion loops and swap the temporary `decl_to_csr_node` and `decl_to_csr_edge` mappings to `FxHashMap`.
  - [X] Capture randomized COO benchmark telemetry for the accelerated path via `hymeko_core/tests/benchmarks/bench_coo_builder_random.rs::bench_random_hypergraph_coo_builder_suite`, exported to `hymeko_core/target/benchmarks/coo_builder_random_benchmark.csv`.

- [X] **Task 1.3: Deterministic B-Tree Indexing (Hash Stability)**
  - [X] Open the compiler's resolution module (`hymeko_core/src/resolution/resolve.rs`).
  - [X] Change the `by_path` mapping in your `Index` struct from a standard `HashMap` to a `std::collections::BTreeMap`.
  - [X] Open `hymeko_core/src/ir/hash.rs` and `hymeko_core/src/ir/canonical_hash.rs`.
  - [X] Delete the manual vector allocation and sorting logic that was previously used to order `PathKey`s. Route the hasher directly through the `BTreeMap`'s native lexicographical iterator.

## Python Packaging Integration

- CI/CD now builds and tests Python packages using maturin.
- Python wheels are uploaded as artifacts and optionally published to PyPI.
- See `hymeko_py` crate and workflow YAML files for details.
