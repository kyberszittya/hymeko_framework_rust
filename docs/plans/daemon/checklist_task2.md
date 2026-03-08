# Checklist for HyMeKo Daemon Refactor: Phase 2

## Phase 2: The Data Plane (Memory & Transport)

- [ ] **Task 2.1: `iceoryx2` Integration (Zero-Copy Shared Memory)**
  - [x] Add `iceoryx2 = "0.4"` (or the latest stable version) to the dependencies in both `hymeko_core/Cargo.toml` and `hymeko_daemon/Cargo.toml`. *(Done with `0.8.1` in both crates.)*
  - [x] Create `hymeko_core/src/tensor/shared_state.rs` to define the `HypergraphWeights` struct with `#[repr(C)]` and implement the `FixedSizeByteData` trait. *(Satisfied via `#[derive(ZeroCopySend)]`, which provides the required proof.)*
  - [x] Initialize the `iceoryx2` Node and Publisher in `hymeko_daemon/src/main.rs`. *(Publisher now advertises a raw `[u8]` slice while Task 2.3 builds the typed bridge.)*
  - [x] Implement the daemon's main event loop to keep the shared memory segment alive and monitor for PyTorch subscriber connections. *(The loop now loans `[u8]` slices, writes the `ExpansionHeader + COO` buffers via `HypergraphEngine::write_tensor_into_raw`, and publishes a frame every tick while subscribers are attached.)*

- [x] **Task 2.2: Apache Arrow Schema Definition**
  - [x] Add the `arrow` crate dependency to `hymeko_core/Cargo.toml`. *(Done alongside `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Add the `arrow` crate dependency to `hymeko_py/Cargo.toml`.
  - [x] Define the strict Arrow schema for the 3D Star/Clique Expansions: `k` (Int64), `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_3d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Define the strict Arrow schema for the 2D Projected Expansions: `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_2d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*

- [x] **Task 2.3: The Direct Memory Bridge**
  - _Re-scoped on 2026-03-08 to capture the zero-copy iceoryx2 ↔ PyTorch path._
  - [x] Modify the core engine's expansion loop to accept mutable raw pointers (`*mut i64`, `*mut f32`) mapped from the `iceoryx2` slice. *(Implemented via `HypergraphEngine::write_star_expansion_into_raw`, which emits an `ExpansionHeader` + COO buffers without reallocations.)*
  - [x] Write the bridging logic in `hymeko_py/src/interface_python/api.rs` (see `PySharedExpansion` and the tensor wrappers) that takes the shared memory pointer, reads the `ExpansionHeader` for the exact size, and feeds the offsets to `pyarrow.foreign_buffer` for zero-copy PyTorch ingestion. *(`PySharedExpansion::buffers` now returns the four `pyarrow.foreign_buffer` handles and passes `self` as the owner so the shared memory lifetime matches the Python object.)*
