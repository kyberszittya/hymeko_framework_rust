# Checklist for HyMeKo Daemon Refactor: Phase 2

## Phase 2: The Data Plane (Memory & Transport)

- [ ] **Task 2.1: `iceoryx2` Integration (Zero-Copy Shared Memory)**
  - [x] Add `iceoryx2 = "0.4"` (or the latest stable version) to the dependencies in both `hymeko_core/Cargo.toml` and `hymeko_daemon/Cargo.toml`. *(Done with `0.8.1` in both crates.)*
  - [x] Create `hymeko_core/src/tensor/shared_state.rs` to define the `HypergraphWeights` struct with `#[repr(C)]` and implement the `FixedSizeByteData` trait. *(Satisfied via `#[derive(ZeroCopySend)]`, which provides the required proof.)*
  - [x] Initialize the `iceoryx2` Node and Publisher in `hymeko_daemon/src/main.rs`.
  - [ ] Implement the daemon's main event loop to keep the shared memory segment alive and monitor for PyTorch subscriber connections. *(Loop keeps the segment alive but has no subscriber monitoring yet.)*

- [x] **Task 2.2: Apache Arrow Schema Definition**
  - [x] Add the `arrow` crate dependency to `hymeko_core/Cargo.toml`. *(Done alongside `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Add the `arrow` crate dependency to `hymeko_py/Cargo.toml`.
  - [x] Define the strict Arrow schema for the 3D Star/Clique Expansions: `k` (Int64), `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_3d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Define the strict Arrow schema for the 2D Projected Expansions: `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_2d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*

- [ ] **Task 2.3: The Translation Layer (Memory Packing)**
  - [ ] Modify `PyTensorCoo3D` and `PySparseMatrix2D` in `hymeko_py/src/api.rs` to wrap the `iceoryx2` shared memory pointers rather than allocating new memory on the heap.
  - [ ] Write the serialization logic that takes the pure Rust `TensorCoo<f32>` output from `compile_star_expansion_core` and directly writes it into the `iceoryx2` segment.
  - [ ] Expose the Arrow arrays to Python via the FFI boundary so PyTorch can ingest them instantly using DLPack.