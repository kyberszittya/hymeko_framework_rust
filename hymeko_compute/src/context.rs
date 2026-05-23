//! Vulkan device + queue initialisation.
//!
//! [`VulkanContext`] owns the Vulkan instance, the chosen physical
//! device, the logical device, the compute queue, and a memory
//! allocator + command-buffer allocator + descriptor-set allocator.
//! It is the handle every kernel takes as its first argument.
//!
//! Construction errors flow through [`ComputeError`] so callers on
//! GPU-less machines (CI runners) can degrade gracefully.

use std::sync::Arc;

use thiserror::Error;
use vulkano::{
    VulkanLibrary,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
};

/// Errors returned by [`VulkanContext::new`].
#[derive(Debug, Error)]
pub enum ComputeError {
    /// The Vulkan loader could not be initialised (no driver / no
    /// libvulkan on the system).
    #[error("vulkan loader unavailable: {0}")]
    LoaderUnavailable(String),

    /// `vkCreateInstance` failed.
    #[error("vulkan instance creation failed: {0}")]
    InstanceCreate(String),

    /// No physical device exposes a queue family with `COMPUTE` bit
    /// set. Practically this happens only on degraded headless setups.
    #[error("no Vulkan-capable compute device found")]
    NoComputeDevice,

    /// `vkCreateDevice` failed.
    #[error("vulkan device creation failed: {0}")]
    DeviceCreate(String),
}

/// Vulkan context shared across kernels.
///
/// Owns:
/// - the [`Instance`] and the chosen [`PhysicalDevice`];
/// - the logical [`Device`] + compute [`Queue`];
/// - a [`StandardMemoryAllocator`] for buffer allocation;
/// - a [`StandardCommandBufferAllocator`] for short-lived command
///   buffers;
/// - a [`StandardDescriptorSetAllocator`] for kernel descriptor sets.
///
/// Construction picks the lowest-bandwidth-cost device that exposes a
/// compute queue: discrete GPU first, then integrated, then virtual,
/// then CPU. The chosen device is logged via [`Self::device_name`].
#[derive(Clone)]
pub struct VulkanContext {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VulkanContext {
    /// Initialise a Vulkan instance + device + compute queue.
    ///
    /// Returns a typed [`ComputeError`] if any step fails so callers
    /// (tests, optional GPU paths) can degrade gracefully.
    pub fn new() -> Result<Self, ComputeError> {
        let library = VulkanLibrary::new().map_err(|e| {
            ComputeError::LoaderUnavailable(e.to_string())
        })?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                application_name: Some("hymeko_compute".into()),
                ..Default::default()
            },
        )
        .map_err(|e| ComputeError::InstanceCreate(e.to_string()))?;

        // Pick (physical_device, queue_family_index) — prefer discrete
        // GPU. Selection is deterministic so tests reproduce.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .map_err(|e| ComputeError::InstanceCreate(e.to_string()))?
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .ok_or(ComputeError::NoComputeDevice)?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| ComputeError::DeviceCreate(e.to_string()))?;

        let queue = queues.next().ok_or(ComputeError::DeviceCreate(
            "no queue returned by Device::new".into(),
        ))?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            device.clone(),
        ));
        let command_buffer_allocator = Arc::new(
            StandardCommandBufferAllocator::new(device.clone(), Default::default()),
        );
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(device.clone(), Default::default()),
        );

        Ok(Self {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        })
    }

    /// Vulkan instance handle (for surface creation, tests, etc.).
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Logical device handle.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Compute queue handle.
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Memory allocator for buffer construction.
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }

    /// Command buffer allocator.
    pub fn command_buffer_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.command_buffer_allocator
    }

    /// Descriptor set allocator.
    pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.descriptor_set_allocator
    }

    /// Name of the chosen physical device, for logging.
    pub fn device_name(&self) -> String {
        self.device
            .physical_device()
            .properties()
            .device_name
            .clone()
    }

    /// Vulkan API version reported by the loader.
    pub fn api_version(&self) -> String {
        let v = self.device.physical_device().api_version();
        format!("{}.{}.{}", v.major, v.minor, v.patch)
    }
}
