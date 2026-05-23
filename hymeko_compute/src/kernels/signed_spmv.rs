//! Signed-incidence sparse matrix-vector product, GPU-side.
//!
//! Given a flattened CSR-like layout of the signed-incidence matrix
//! $\mathbf{B}$ (rows = vertices, columns = hyperedges, entries
//! $\sigma_{ve}\in\{-1,0,+1\}$ scaled by an optional weight), computes
//! $\mathbf{y} = \mathbf{B}\mathbf{x}$ where $\mathbf{x}\in\mathbb{R}^{|E|}$
//! is a per-hyperedge signal and $\mathbf{y}\in\mathbb{R}^{|V|}$ is a
//! per-vertex aggregate.
//!
//! This is the workhorse primitive used by every `hymeko_hnn`
//! convolution variant (signed_hgnn, hgnn, gcn_clique). The CSR layout
//! matches `hymeko_core::tensor::TensorCsr`:
//!
//! - `row_ptr[v]` / `row_ptr[v+1]` bracket the entries of vertex `v`;
//! - `col_ind[k]` is the hyperedge column for the `k`th non-zero;
//! - `val[k]` is the signed (and possibly weighted) entry.
//!
//! One workgroup-thread per vertex. Each thread iterates its row.

use bytemuck::{Pod, Zeroable};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::{
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::GpuFuture,
};

use crate::buffers;
use crate::context::VulkanContext;

const SIGNED_SPMV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/signed_spmv.spv"));

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "SPIR-V byte length not multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// CSR view of the signed-incidence matrix $\mathbf{B}$ for use with
/// the SpMV kernel. Field semantics match
/// `hymeko_core::tensor::TensorCsr` exactly.
#[derive(Debug, Clone)]
pub struct SignedIncidenceCsr {
    /// `row_ptr` of length `n_rows + 1`.
    pub row_ptr: Vec<u32>,
    /// Hyperedge column indices, length `nnz`.
    pub col_ind: Vec<u32>,
    /// Signed (and possibly weighted) entries, length `nnz`.
    pub val: Vec<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SpmvParams {
    n_rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Compute `y = B * x` where `b` carries the CSR view of the
/// signed-incidence matrix. `x` is the per-hyperedge signal; the
/// returned vector is per-vertex aggregate of length `b.row_ptr.len() - 1`.
pub fn run(ctx: &VulkanContext, b: &SignedIncidenceCsr, x: &[f32]) -> Result<Vec<f32>, String> {
    let n_rows = b.row_ptr.len().saturating_sub(1) as u32;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let buf_row_ptr = buffers::upload(ctx, &b.row_ptr);
    let buf_col_ind = buffers::upload(ctx, &b.col_ind);
    let buf_val = buffers::upload(ctx, &b.val);
    let buf_x = buffers::upload(ctx, x);
    let buf_y = buffers::download::<f32>(ctx, n_rows as usize);
    let buf_params = buffers::upload(
        ctx,
        &[SpmvParams {
            n_rows,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }],
    );

    let words = bytes_to_words(SIGNED_SPMV_SPV);
    let shader =
        unsafe { ShaderModule::new(ctx.device().clone(), ShaderModuleCreateInfo::new(&words)) }
            .map_err(|e| e.to_string())?;
    let entry_point = shader.entry_point("main").ok_or("missing entry point")?;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        ctx.device().clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(ctx.device().clone())
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let pipeline = ComputePipeline::new(
        ctx.device().clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
    )
    .map_err(|e| e.to_string())?;
    let dsl = layout.set_layouts().first().ok_or("missing dsl")?;
    let descriptor_set = PersistentDescriptorSet::new(
        ctx.descriptor_set_allocator().as_ref(),
        dsl.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_row_ptr.clone()),
            WriteDescriptorSet::buffer(1, buf_col_ind.clone()),
            WriteDescriptorSet::buffer(2, buf_val.clone()),
            WriteDescriptorSet::buffer(3, buf_x.clone()),
            WriteDescriptorSet::buffer(4, buf_y.clone()),
            WriteDescriptorSet::buffer(5, buf_params.clone()),
        ],
        [],
    )
    .map_err(|e| e.to_string())?;

    let workgroups = n_rows.div_ceil(64);
    let mut cb = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator().as_ref(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(|e| e.to_string())?;
    cb.bind_pipeline_compute(pipeline.clone())
        .map_err(|e| e.to_string())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            descriptor_set,
        )
        .map_err(|e| e.to_string())?
        .dispatch([workgroups, 1, 1])
        .map_err(|e| e.to_string())?;
    let cmd = cb.build().map_err(|e| e.to_string())?;
    let future = cmd
        .execute(ctx.queue().clone())
        .map_err(|e| e.to_string())?
        .then_signal_fence_and_flush()
        .map_err(|e| e.to_string())?;
    future.wait(None).map_err(|e| e.to_string())?;

    Ok(buffers::read_back(&buf_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny worked example: 3 vertices, 2 hyperedges, signed entries.
    /// $B = \begin{pmatrix} +1 & 0 \\ -1 & +1 \\ 0 & -1 \end{pmatrix}$,
    /// $\mathbf{x} = (2, 3)^T$ ⇒ $\mathbf{y} = (2, 1, -3)^T$.
    #[test]
    #[ignore = "requires Vulkan-capable GPU"]
    fn signed_spmv_tiny() {
        let ctx = VulkanContext::new().expect("vulkan init");
        let b = SignedIncidenceCsr {
            row_ptr: vec![0, 1, 3, 4],
            col_ind: vec![0, 0, 1, 1],
            val: vec![1.0, -1.0, 1.0, -1.0],
        };
        let x = [2.0_f32, 3.0];
        let y = run(&ctx, &b, &x).expect("dispatch");
        assert_eq!(y.len(), 3);
        assert!((y[0] - 2.0).abs() < 1e-6, "y[0] = {} (expected 2)", y[0]);
        assert!((y[1] - 1.0).abs() < 1e-6, "y[1] = {} (expected 1)", y[1]);
        assert!(
            (y[2] - (-3.0)).abs() < 1e-6,
            "y[2] = {} (expected -3)",
            y[2]
        );
    }
}
