#[path = "minimal_tests/mod.rs"]
mod minimal_tests;

#[path = "typical_graphs/mod.rs"]
mod typical_graphs;

#[path = "intermediate_tests/mod.rs"]
mod intermediate_tests;

#[path = "traversal/mod.rs"]
mod traversal;

pub mod test_helpers;
pub mod test_asserts;

mod test_tensor_representations;
mod aggregations;
mod domain_transformations;
mod engine;
mod hash;
mod benchmarks;
mod basic;
