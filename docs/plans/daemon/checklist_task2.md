# Checklist for HyMeKo Daemon Refactor: Phase 2

PHASE 2 CLOSED at 2026-03-10, with the end-to-end zero-copy path now implemented and traced across the core engine, daemon, and Python bindings. The checklist below reflects the final state of the tasks as of that date, with all items marked complete.

## Phase 2: The Data Plane (Memory & Transport)

- [x] **Task 2.1: `iceoryx2` Integration (Zero-Copy Shared Memory)**
  - [x] Add `iceoryx2 = "0.4"` (or the latest stable version) to the dependencies in both `hymeko_core/Cargo.toml` and `hymeko_daemon/Cargo.toml`. *(Done with `0.8.1` in both crates.)*
  - [x] Create `hymeko_core/src/tensor/shared_state.rs` to define the `HypergraphWeights` struct with `#[repr(C)]` and implement the `FixedSizeByteData` trait. *(Satisfied via `#[derive(ZeroCopySend)]`, which provides the required proof.)*
  - [x] Initialize the `iceoryx2` Node and Publisher in `hymeko_daemon/src/main.rs`. *(Publisher advertises a raw `[u8]` slice and emits `ExpansionHeader + COO` frames.)*
  - [x] Implement the daemon's main event loop to keep the shared memory segment alive and monitor for PyTorch subscriber connections. *(The loop uses `service.dynamic_config().number_of_subscribers() > 0` for gating and calls `publish_star_expansion` to loan/send frames.)*

- [x] **Task 2.2: Apache Arrow Schema Definition**
  - [x] Add the `arrow` crate dependency to `hymeko_core/Cargo.toml`. *(Done alongside `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Add the `arrow` crate dependency to `hymeko_py/Cargo.toml`.
  - [x] Define the strict Arrow schema for the 3D Star/Clique Expansions: `k` (Int64), `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_3d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*
  - [x] Define the strict Arrow schema for the 2D Projected Expansions: `i` (Int64), `j` (Int64), and `val` (Float32). *(Implemented via `schema_expansion_2d` in `hymeko_core/src/tensor/arrow_schema.rs`.)*

- [x] **Task 2.3: The Direct Memory Bridge**
  - _Re-scoped on 2026-03-08 to capture the zero-copy iceoryx2 ↔ PyTorch path._
  - [x] Modify the core engine's expansion loop to accept mutable raw pointers (`*mut i64`, `*mut f32`) mapped from the `iceoryx2` slice. *(Implemented via `HypergraphEngine::write_star_expansion_into_raw`, which emits an `ExpansionHeader` + COO buffers without reallocations.)*
  - [x] Write the bridging logic in `hymeko_py/src/interface_python/api.rs` (see `PySharedExpansion` and the tensor wrappers) that takes the shared memory pointer, reads the `ExpansionHeader` for the exact size, and feeds the offsets to `pyarrow.foreign_buffer` for zero-copy PyTorch ingestion. *(`PySharedExpansion::buffers` now returns the four `pyarrow.foreign_buffer` handles and passes `self` as the owner so the shared memory lifetime matches the Python object.)*


