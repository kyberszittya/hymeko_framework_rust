//! Operator catalogue for Nagare.
//!
//! Each operator is a forward + backward pair of plain Rust functions
//! over SoA buffers (`&[f32]`, `&[u32]`, etc.). There is **no Op
//! trait**; we deliberately avoid an autograd graph. Operators
//! compose by direct function call + intermediate `Vec<f32>` storage,
//! and the training loop in `training.rs` orchestrates them.
//!
//! Per-operator gradients are derived once analytically and hand-coded
//! (see plan.tex Section "Per-primitive forward + backward kernels"
//! for the closed-form expressions for FIR, scatter-mean, linear,
//! BCE-with-logits).

pub mod adam;
pub mod clifford_fir;
pub mod linear;
pub mod loss;
pub mod scatter;
