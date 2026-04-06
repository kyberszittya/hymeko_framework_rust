# Project Changelog — 2026-04-07

## Query Engine Branch Integration
- Landed the new query stack and follow-up refactors across `hymeko_query/src/` (`engine.rs`, `interpret.rs`, `predicate.rs`, `codegen.rs`, `formats/`, and `kinematics/`) plus matching tests under `hymeko_query/tests/codegen/`.
- Split query functionality out of `hymeko_core` into the dedicated `hymeko_query` crate, including path/module moves and workspace wiring updates in `Cargo.toml` and `hymeko_core/src/lib.rs`.
- Added missing helper surfaces for downstream query workflows in `hymeko_query/src/traits.rs` and expanded test helpers in `hymeko_query/tests/test_helpers.rs`.

## Tensor Convolution and Decomposition Expansion
- Reorganized tensor convolution into modular components under `hymeko_core/src/tensor/conv/` (`gcn_clique.rs`, `hgnn.rs`, `signed_hgnn.rs`, and shared `traits.rs`) and exported them via `mod.rs`.
- Added decomposition and mesh-oriented tensor plumbing in `hymeko_core/src/tensor/decomposition.rs` and `hymeko_core/src/tensor/mesh_nn/mod.rs`.
- Updated tensor/common representation support (`hymeko_core/src/tensor/common.rs`, `tensor_coo_representation.rs`, `tensor_csr_representations.rs`) and related imports/usages.

## Deterministic Weight Initialization Coverage
- Added weight initializer primitives under `hymeko_core/src/tensor/conv/weight_init/` with deterministic sequence support (`van_der_corput`) and initializer variants (`Xavier`, `Kaiming`, `XavierRandom`, `Zeros`, `Ones`, `Constant`).
- Registered the new module in `hymeko_core/src/tensor/conv/mod.rs` and added coverage tests in `hymeko_core/tests/computations/test_weight_init.rs`.

## Parser and Dataset Layout Updates
- Extended parser token/grammar handling in `parser/src/hymeko.lalrpop`, `parser/src/lexer/common.rs`, and `parser/src/lexer/token.rs` to support the updated query/model authoring flow.
- Moved sample `.hymeko` assets from `hymeko_core/data/` to top-level `data/` and kept robotics fixtures updated for query/codegen scenarios (notably `data/robotics/robot_4wh.hymeko`).

