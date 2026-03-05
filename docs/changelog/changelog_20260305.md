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
