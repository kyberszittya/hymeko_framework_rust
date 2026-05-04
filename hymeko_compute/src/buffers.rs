//! Typed buffer helpers for kernel I/O.
//!
//! Wraps [`vulkano::buffer::Buffer`] with two ergonomic shapes:
//!
//! - [`upload`] — host-visible, write-once, read-by-shader; used for
//!   small constants and inputs the host produced.
//! - [`download`] — host-visible, written-by-shader, read-by-host;
//!   used for results.
//! - [`storage`] — device-local storage buffer the shader reads and
//!   writes; the workhorse for every per-vertex / per-edge array.
//!
//! All variants live in `MemoryUsage` regions that are coherent — no
//! manual flush is required, but a queue-side memory barrier is.

use bytemuck::Pod;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::context::VulkanContext;

/// Allocate a host-visible buffer initialised from `data` and usable
/// as a shader storage source (read by shader). Convenient for inputs
/// the host wants the GPU to read once.
pub fn upload<T: Pod + Send + Sync + 'static>(
    ctx: &VulkanContext,
    data: &[T],
) -> Subbuffer<[T]> {
    Buffer::from_iter(
        ctx.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .expect("buffer upload failed")
}

/// Allocate a host-readable buffer the shader writes into. Use for
/// outputs the host wants to read back.
pub fn download<T: Pod + Send + Sync + 'static>(
    ctx: &VulkanContext,
    len: usize,
) -> Subbuffer<[T]> {
    Buffer::from_iter(
        ctx.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        std::iter::repeat(T::zeroed()).take(len),
    )
    .expect("buffer download alloc failed")
}

/// Allocate a host-visible storage buffer the shader both reads and
/// writes, initialised from `initial`. Used for per-vertex / per-edge
/// arrays the host needs to read back after the dispatch completes.
///
/// Currently this is host-visible (PREFER_HOST + HOST_RANDOM_ACCESS) so
/// `read_back` works directly without a staging copy. For workloads
/// where the same buffer survives many dispatches without ever leaving
/// the device, a device-local variant with an explicit GPU→staging copy
/// is the right follow-up.
pub fn storage<T: Pod + Send + Sync + 'static>(
    ctx: &VulkanContext,
    initial: &[T],
) -> Subbuffer<[T]> {
    Buffer::from_iter(
        ctx.memory_allocator().clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        initial.iter().copied(),
    )
    .expect("storage buffer alloc failed")
}

/// Read a host-visible buffer back into a Rust `Vec<T>` after the
/// shader has signalled completion. Caller is responsible for
/// ensuring the GPU is done with this buffer (typically via a
/// `wait_for_fences` on the dispatch's future).
pub fn read_back<T: Pod + Send + Sync + Clone + 'static>(
    buf: &Subbuffer<[T]>,
) -> Vec<T> {
    let view = buf.read().expect("buffer read lock failed");
    view.iter().cloned().collect()
}
