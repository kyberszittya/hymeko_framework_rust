//! Smoke test: report what the loader picks. Useful to confirm
//! the chosen device matches expectations on a developer box.

use hymeko_compute::VulkanContext;

#[test]
#[ignore = "requires Vulkan-capable GPU"]
fn report_device() {
    let ctx = VulkanContext::new().expect("vulkan init");
    println!("device: {}", ctx.device_name());
    println!("api version: {}", ctx.api_version());
    println!(
        "queue family: {}",
        ctx.queue().queue_family_index()
    );
}
