# Project Changelog — 2026-03-08

## Arrow Schema Module for Tensor Expansions
- Added `hymeko_core/src/tensor/arrow_schema.rs`, which exposes `schema_expansion_3d` and `schema_expansion_2d` so every consumer (daemon, Python bindings, analytics tools) can request a shared Arrow schema instead of re-declaring field layouts.
- Locked the 3D Star/Clique expansion schema to `(k, i, j, val)` with 64-bit indices and 32-bit weights, matching the tensor compiler output while avoiding downstream casts.
- Locked the 2D Projected expansion schema to `(i, j, val)` so projection pipelines and visualization clients can share buffers with the daemon without reallocations.

## Daemon Star Expansion Stream
- `hymeko_daemon` now compiles a star-expansion tensor once on boot, loans `[u8]` slices from its iceoryx2 publisher, and copies the `ExpansionHeader + (k,i,j,val)` buffers into the shared segment every tick whenever subscribers are attached.
- Publishing reuses `HypergraphEngine::write_tensor_into_raw`, so the daemon and Python bridge rely on a single layout contract and no longer allocate intermediary host buffers.

## Checklist Impact
- Marked Task 2.2 in `docs/plans/daemon/checklist_task2.md` as complete: `hymeko_core` now carries the Arrow dependency next to the schema helpers, `hymeko_py` already had the crate wired, and both schemas are implemented in one module for easy reuse.
- Documented the schema functions inside the checklist entry to help Task 2.3 engineers link directly to the zero-copy building blocks they must wrap for `PyTensorCoo3D`/`PySparseMatrix2D`.
- Re-scoped Task 2.3 as **The Direct Memory Bridge**, focusing on raw pointer expansion hooks inside `hymeko_core` and the `pyarrow.foreign_buffer` wiring in `hymeko_py/src/interface_python/api.rs` (e.g., the `PySharedExpansion` scaffold plus the tensor wrappers) so future work items reference the precise zero-copy objectives.
- Completed that bridge by adding `HypergraphEngine::write_star_expansion_into_raw` (writes directly into `ExpansionHeader` + COO buffers) and `PySharedExpansion::buffers`, which converts any mapped `iceoryx2` slice into four `pyarrow.foreign_buffer` handles for PyTorch.
- Swapped the daemon's publish/subscribe type to a raw `[u8]` slice so the service stays live while Task 2.3 wires the structured view; the checklist now calls out this interim state under Task 2.1.
- Checked off Task 2.1 now that the daemon loop publishes real frames and monitors subscriber lifetimes; noted that `PySharedExpansion` passes `self` into `pyarrow.foreign_buffer` so ownership matches the Rust sample lifetime.

## Next Steps
- Finish Task 2.3 by refitting the Python tensors and serializer to write directly into the `iceoryx2` segment using the Arrow schemas.
- Extend the daemon event loop to emit readiness telemetry once multiple Arrow subscribers attach (PyTorch + OpenGL client), ensuring the shared memory lifecycle lines up with the schema contracts.
