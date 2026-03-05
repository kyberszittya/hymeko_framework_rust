# Project Changelog - 2026-03-05
## Portable CBOR snapshots of the IR
- Added a formal CborPayload (src/writers/cbor_writer.rs) that packages the Ir, Index, canonical hash, interned strings, and import table so binary exports contain all compiler context needed for faithful reconstruction.
- Wired PyHypergraphIR::to_cbor / from_cbor (src/interface_python/api.rs) to the new payload, using ciborium for serialization so Python can persist and reload fully-resolved graphs, including the module key of the root artifact.
- Exposed the feature in a quick Python harness (py/cbor/write_cbor.py) that loads the sample Fano graph, emits CBOR bytes, and reports compressed sizes for regression tracking.
## serde coverage across compiler data structures
- Introduced serde::{Serialize, Deserialize} derives across IDs, PathKey, HashId, IR nodes (DeclNode, NodeRec, EdgeRec, ArcRec, SignedRefR, ValueR, AnnoR), Meta, ModuleKey, and the resolver Index, enabling CBOR snapshots without manual encode glue.
- Added the corresponding serde and ciborium dependencies in Cargo manifests, ensuring the feature is available to both the engine crate and the parser workspace via workspace inheritance.
## Tensor CSR builder determinism
- Replaced the previous global sort in TensorCsrBuilder::finalize_coalesced (src/tensor/representations/tensor_csr_builder_impl.rs) with row-local sorting that respects the precomputed row_ptr boundaries, resulting in O(N * d log d) behavior and stable row pointers even when rows are empty.
## Research collateral and datasets
- Authored docs/math/kan_math.md, formally stating the Hypergraph-KA operator and tying KA inner/outer sums to our CSR-backed incidence tensors for future publication.
- Added the curated data/benchmarks/*.hymeko fixtures (dense, sparse, clique, etc.) so simulation, compression, and serialization benchmarks have reproducible inputs going forward.
- Captured the 2026-03-05 architecture blueprint in `docs/plans/plan_20260305.md`, detailing expansion strategies, CBOR/Zlib serialization, spectral math, and the NURBS-powered HyperKAN roadmap.

## Parsing grid benchmarks
- Introduced `py/parsing/benchmarks/grid_expansion_bench.py`, which sweeps configurable node/edge/density grids, records parse/expansion timings, and prints a summary table for each configuration.
- Added timestamped CSV exports for both aggregate statistics and raw per-iteration measurements (`grid_expansion_<timestamp>.csv` and `grid_expansion_raw_<timestamp>.csv`) so downstream tooling can ingest the benchmark data without rerunning experiments.
- Captured the iteration count in both the tabular summaries and CSV output, and persisted every per-trial parse/expansion measurement (including NNZ counts) for deeper statistical post-processing.

## Path indexing and IR tree hygiene
- Let `PathKey` implement `Borrow<[SymId]>`, so the resolver/index no longer clones temporary vectors when doing lookups, and wired the lowering phase to query paths by slice directly.
- `DeclNode` now tracks both `first_child` and `last_child`, plus `Ir::decl_node(_unchecked)` and `Ir::ensure_decl_capacity` helpers centralize arena growth; the lowering pass uses these helpers and keeps siblings wired in O(1) without re-scanning child lists.
- Added `ResolveError::UnexpectedTopLevelArc` so stray arcs at the module root throw a descriptive error instead of silently mutating IR state, and tightened HyperArc lowering so edge arc lists stay coherent.

## Tensor CSR math helpers
- Introduced `TensorCsr::spmv` and `TensorCsr::spmm`, giving the Python/Hutchinson pipelines ready-to-use sparse mat-vec/mat-mat kernels that run directly on the coalesced CSR data without rebuilding dense tensors.
