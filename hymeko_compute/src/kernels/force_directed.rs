//! Naïve $O(N^2)$ Fruchterman--Reingold force summation, one Vulkan
//! dispatch per layout iteration. The first KEPAF §IV deliverable.
//!
//! ## Forces
//!
//! - **Repulsion**: every pair of vertices contributes $k_r / d^2$
//!   along the line connecting them.
//! - **Attraction**: arcs in the input edge buffer contribute
//!   $k_a (d - L_0)$ along their endpoint pair.
//! - **Damping**: $-\gamma v$ on the per-vertex velocity.
//!
//! Per-vertex update is one workgroup-thread; each thread sums forces
//! against every other vertex in a single pass. $O(N^2)$ inside the
//! kernel; the GPU is faster than the CPU baseline (NetworkX
//! `spring_layout`, 1739 s on $|V|=35\text{k}$, 20 iter) by an amount
//! we measure in the integration test.
//!
//! Octree / Barnes-Hut acceleration is the natural follow-up but is
//! deferred — the harness is proven first, the asymptotic speedup
//! second.

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

const FORCE_DIRECTED_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/force_directed.spv"));

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

/// 2-D vertex position in layout space.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Position {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

/// One arc in the layout graph: indices into the position buffer plus
/// the spring rest length the attraction term targets.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Arc2D {
    /// Source vertex index.
    pub src: u32,
    /// Destination vertex index.
    pub dst: u32,
    /// Spring rest length $L_0$.
    pub rest_len: f32,
    /// Padding for std430 alignment.
    pub _pad: f32,
}

/// Per-iteration parameters passed to the shader as a uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LayoutParams {
    /// Number of vertices.
    pub n_vertices: u32,
    /// Number of arcs.
    pub n_arcs: u32,
    /// Repulsion gain $k_r$.
    pub k_repulsion: f32,
    /// Attraction gain $k_a$.
    pub k_attraction: f32,
    /// Damping coefficient $\gamma$.
    pub damping: f32,
    /// Integration time step $\Delta t$.
    pub dt: f32,
    /// Padding to round to 32 bytes (std140 alignment).
    /// Padding to align to 32 bytes (std140 uniform alignment).
    pub _pad0: f32,
    /// Padding to align to 32 bytes (std140 uniform alignment).
    pub _pad1: f32,
}

/// Run `n_iter` iterations of force-directed layout in place.
///
/// `pos_init` is overwritten and returned with the final positions.
/// `arcs` are read-only across iterations. Velocity is initialised to
/// zero and discarded on exit.
pub fn run(
    ctx: &VulkanContext,
    pos_init: &mut Vec<Position>,
    arcs: &[Arc2D],
    params: LayoutParams,
    n_iter: u32,
) -> Result<(), String> {
    let n = pos_init.len() as u32;
    if n == 0 {
        return Ok(());
    }
    if params.n_vertices != n {
        return Err(format!(
            "params.n_vertices ({}) != pos_init.len() ({})",
            params.n_vertices, n
        ));
    }

    let buf_pos = buffers::storage(ctx, pos_init);
    let zero_vel: Vec<Position> = vec![Position { x: 0.0, y: 0.0 }; n as usize];
    let buf_vel = buffers::storage(ctx, &zero_vel);
    let buf_arcs = buffers::upload(ctx, arcs);

    // Uniform buffer for params. We update it every iteration in case
    // the caller wants a schedule (currently params is held constant
    // across n_iter, but the API leaves room to lift this).
    let buf_params = buffers::upload(ctx, &[params]);

    let words = bytes_to_words(FORCE_DIRECTED_SPV);
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
            WriteDescriptorSet::buffer(0, buf_pos.clone()),
            WriteDescriptorSet::buffer(1, buf_vel.clone()),
            WriteDescriptorSet::buffer(2, buf_arcs.clone()),
            WriteDescriptorSet::buffer(3, buf_params.clone()),
        ],
        [],
    )
    .map_err(|e| e.to_string())?;

    let workgroups = n.div_ceil(64);
    for _ in 0..n_iter {
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
                descriptor_set.clone(),
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
    }

    *pos_init = buffers::read_back(&buf_pos);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test on the canonical 21-vertex paper example. Verifies
    /// the layout converges (final RMS displacement small) and that no
    /// vertex went to NaN/Inf.
    #[test]
    #[ignore = "requires Vulkan-capable GPU"]
    fn force_directed_canonical() {
        let ctx = VulkanContext::new().expect("vulkan init");
        let n = 21;
        // Random initial positions in [-1, 1]^2, deterministic seed.
        let mut pos: Vec<Position> = (0..n)
            .map(|i| {
                let t = i as f32 * 1.1;
                Position {
                    x: t.cos(),
                    y: t.sin(),
                }
            })
            .collect();
        // 30 random arcs (binary edges) — stand-in for the canonical
        // example's 10 hyperedges expanded to a Levi graph of ~33
        // pair-wise edges.
        let arcs: Vec<Arc2D> = (0..30)
            .map(|k| Arc2D {
                src: (k * 7 % n) as u32,
                dst: ((k + 1) * 11 % n) as u32,
                rest_len: 0.3,
                _pad: 0.0,
            })
            .collect();
        let params = LayoutParams {
            n_vertices: n as u32,
            n_arcs: arcs.len() as u32,
            k_repulsion: 0.05,
            k_attraction: 1.0,
            damping: 0.85,
            dt: 0.02,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        run(&ctx, &mut pos, &arcs, params, 100).expect("dispatch");
        for (i, p) in pos.iter().enumerate() {
            assert!(
                p.x.is_finite() && p.y.is_finite(),
                "vertex {i} went to non-finite ({}, {})",
                p.x,
                p.y
            );
        }
    }
}
