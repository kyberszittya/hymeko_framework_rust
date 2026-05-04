//! Compute kernels.
//!
//! Each module here is one shader + a thin Rust dispatch wrapper.
//! Shaders are GLSL `.comp` files compiled to SPIR-V at compile-time
//! via the `vulkano_shaders::shader!` proc macro inline in each module.
//!
//! Kernel inventory:
//!
//! - [`vector_add`] — proof-of-life elementwise add. Used by the
//!   smoke tests to confirm the device + dispatch plumbing is alive.
//! - [`force_directed`] — naïve $O(N^2)$ Fruchterman-Reingold force
//!   summation per vertex; the first KEPAF §IV deliverable.
//! - [`signed_spmv`] — signed-incidence sparse matrix-vector product
//!   used by `hymeko_hnn` convolution variants.

pub mod force_directed;
pub mod signed_spmv;
pub mod vector_add;
