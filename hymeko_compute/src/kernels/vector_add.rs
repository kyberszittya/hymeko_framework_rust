//! Proof-of-life: elementwise vector add `c[i] = a[i] + b[i]` on the
//! Vulkan compute queue. Used by tests to verify the harness is alive
//! on a fresh box; not a real workload.

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    pipeline::{
        ComputePipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::GpuFuture,
};

use crate::buffers;
use crate::context::VulkanContext;

/// Compiled SPIR-V for the vector-add kernel; produced by `build.rs`
/// from `shaders/vector_add.comp`.
const VECTOR_ADD_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/vector_add.spv"));

/// Run `c = a + b` on the device, returning `c` to the host.
pub fn run(ctx: &VulkanContext, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vector_add: input length mismatch");
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    let buf_a = buffers::upload(ctx, a);
    let buf_b = buffers::upload(ctx, b);
    let buf_c = buffers::download::<f32>(ctx, n);

    let words = bytes_to_words(VECTOR_ADD_SPV);
    let shader = unsafe {
        ShaderModule::new(
            ctx.device().clone(),
            ShaderModuleCreateInfo::new(&words),
        )
    }
    .expect("create vector_add shader module");
    let entry_point = shader.entry_point("main").expect("entry point");

    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        ctx.device().clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(ctx.device().clone())
            .unwrap(),
    )
    .unwrap();
    let pipeline = ComputePipeline::new(
        ctx.device().clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
    )
    .unwrap();

    let descriptor_set_layout = layout.set_layouts().get(0).unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        ctx.descriptor_set_allocator().as_ref(),
        descriptor_set_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_a.clone()),
            WriteDescriptorSet::buffer(1, buf_b.clone()),
            WriteDescriptorSet::buffer(2, buf_c.clone()),
        ],
        [],
    )
    .unwrap();

    let workgroups = ((n as u32) + 63) / 64;
    let mut cb = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator().as_ref(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    cb.bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            descriptor_set,
        )
        .unwrap()
        .dispatch([workgroups, 1, 1])
        .unwrap();
    let cmd = cb.build().unwrap();

    let future = cmd
        .execute(ctx.queue().clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();

    buffers::read_back(&buf_c)
}

/// Re-interpret a SPIR-V byte slice as the `&[u32]` slice vulkano
/// expects. Panics if the byte length is not a multiple of 4 (which
/// would mean a corrupt `.spv`).
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V byte length not multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable GPU"]
    fn vector_add_round_trip() {
        let ctx = VulkanContext::new().expect("vulkan init");
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32).collect();
        let c = run(&ctx, &a, &b);
        assert_eq!(c.len(), 1024);
        for (i, &v) in c.iter().enumerate() {
            assert!((v - 1024.0).abs() < 1e-5, "c[{i}] = {v} (expected 1024.0)");
        }
    }
}
