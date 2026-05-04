//! # `hymeko_compute`
//!
//! Vulkan compute kernels for hypergraph computations on the
//! canonical signed-incidence IR. Self-contained crate; consumes the
//! flat SoA buffer layouts already produced by
//! `hymeko_hnn::traversal::HyperGraphView` and
//! `hymeko_core::tensor::TensorCsr`, and runs compute shaders on the
//! native Vulkan device.
//!
//! ## Boundary with the rest of the workspace
//!
//! - `hymeko_wasm` keeps the browser / WebGPU visualisation path. No
//!   shader sharing in this revision (a follow-up may target wgpu's
//!   WGSL→SPIR-V to share shaders cross-API).
//! - `hymeko_hnn` keeps the CPU compute paths. This crate is the GPU
//!   path; downstream wiring (a `hymeko_hnn` feature flag) is a
//!   follow-up.
//!
//! ## What ships in 0.1
//!
//! - [`context::VulkanContext`] — instance / physical device / logical
//!   device / compute queue / command pool, all initialised once and
//!   shared across kernels.
//! - [`buffers`] — typed buffer wrappers for upload + readback.
//! - [`kernels::vector_add`] — proof-of-life elementwise add, used by
//!   the smoke test that the harness is alive on a fresh box.
//! - [`kernels::force_directed`] — naïve $O(N^2)$ Fruchterman-Reingold
//!   force summation, the first KEPAF §IV deliverable.
//! - [`kernels::signed_spmv`] — signed-incidence SpMV
//!   $\mathbf{y} = \mathbf{B}\mathbf{x}$ with sign-aware accumulation;
//!   the natural primitive used by every `hymeko_hnn` convolution
//!   variant.
//!
//! Octree-acceleration, integration step, and shader sharing with the
//! browser side are deferred to a follow-up plan.
//!
//! ## Hardware / SDK assumptions
//!
//! Tested against Vulkan SDK 1.3.x with a discrete NVIDIA GPU. CI
//! without a GPU should mark the smoke tests `#[ignore]`; the device
//! init in [`context::VulkanContext::new`] returns a typed error that
//! callers can swallow gracefully.

// `unsafe` is required for `vulkano::shader::ShaderModule::new`, which
// trusts caller-supplied SPIR-V to satisfy Vulkan invariants. Each use
// site is scoped narrowly to the SPIR-V word slice produced by our own
// `build.rs` via `glslc`, so the unsafety is contained.
#![warn(missing_docs)]

pub mod buffers;
pub mod context;
pub mod kernels;

pub use context::{ComputeError, VulkanContext};
