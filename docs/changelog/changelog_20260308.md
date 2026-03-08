# Project Changelog — 2026-03-08

## Arrow Schema Module for Tensor Expansions
- Added `hymeko_core/src/tensor/arrow_schema.rs`, which exposes `schema_expansion_3d` and `schema_expansion_2d` so every consumer (daemon, Python bindings, analytics tools) can request a shared Arrow schema instead of re-declaring field layouts.
- Locked the 3D Star/Clique expansion schema to `(k, i, j, val)` with 64-bit indices and 32-bit weights, matching the tensor compiler output while avoiding downstream casts.
- Locked the 2D Projected expansion schema to `(i, j, val)` so projection pipelines and visualization clients can share buffers with the daemon without reallocations.

## Checklist Impact
- Marked Task 2.2 in `docs/plans/daemon/checklist_task2.md` as complete: `hymeko_core` now carries the Arrow dependency next to the schema helpers, `hymeko_py` already had the crate wired, and both schemas are implemented in one module for easy reuse.
- Documented the schema functions inside the checklist entry to help Task 2.3 engineers link directly to the zero-copy building blocks they must wrap for `PyTensorCoo3D`/`PySparseMatrix2D`.

## Next Steps
- Finish Task 2.3 by refitting the Python tensors and serializer to write directly into the `iceoryx2` segment using the Arrow schemas.
- Extend the daemon event loop to emit readiness telemetry once multiple Arrow subscribers attach (PyTorch + OpenGL client), ensuring the shared memory lifecycle lines up with the schema contracts.

