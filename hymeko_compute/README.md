# `hymeko_compute`

Vulkan compute kernels for hypergraph computations on the canonical
signed-incidence IR.

## Boundary with the rest of the workspace

| crate | role |
|-------|------|
| `hymeko_wasm` | browser / WebGPU **visualisation**, no GPU compute |
| `hymeko_hnn` | hypergraph NN convolution variants on the **CPU** |
| `hymeko_compute` (this crate) | **GPU compute** path on native Vulkan |

This split is intentional: WebGPU stays for browser-side visualisation
where it is the only viable choice; Vulkan owns native compute where
control over queue submission, memory residency, and synchronisation
matters.

## What ships in 0.1

- `VulkanContext` — instance / physical device / logical device /
  compute queue / memory + command + descriptor allocators, all
  initialised once and shared.
- Typed buffer helpers (`upload`, `download`, `storage`).
- Three kernels:
  - `vector_add` — proof-of-life (1024-element elementwise add).
  - `force_directed` — naïve $O(N^2)$ Fruchterman-Reingold force
    summation, the first KEPAF §IV deliverable.
  - `signed_spmv` — signed-incidence sparse mat-vec
    $\mathbf{y} = \mathbf{B}\mathbf{x}$ on a CSR layout matching
    `hymeko_core::tensor::TensorCsr`.

## Run the tests

The crate's tests are gated `#[ignore]` so they don't run on
GPU-less CI runners. On a developer box with a Vulkan-capable GPU:

```bash
cargo test -p hymeko_compute -- --ignored --nocapture
```

This will print the chosen device name (e.g.
`NVIDIA GeForce RTX 2070 SUPER`) and run all three kernel smoke tests.

## Toolchain assumptions

- Vulkan SDK 1.3.x with `glslc` and `glslangValidator` on `PATH` (the
  `vulkano-shaders` proc macro uses them at compile time to lower
  GLSL to SPIR-V).
- `libvulkan` available at link time. On Debian/Ubuntu:
  `apt install libvulkan-dev mesa-vulkan-tools`.
- A discrete GPU is not strictly required (an integrated GPU works);
  the device picker prefers discrete when both are present.

## Roadmap (not in 0.1)

- **Octree spatial-acceleration kernel** for the force-directed
  layout — replaces the $O(N^2)$ naïve sum with $O(N \log N)$
  Barnes-Hut. Promised in KEPAF §IV.4 / §IV.5.
- **Integration step** as a second dispatch — currently the
  integration is fused into the force kernel; splitting it allows
  per-step Δt scheduling.
- **Shader sharing with the WebGPU side** — wgpu's WGSL→SPIR-V
  pipeline can be retargeted, but for 0.1 we keep the GLSL→SPIR-V
  Vulkan path independent.
- **`hymeko_hnn` GPU feature flag** — wires `signed_spmv` into the
  CPU SpMV call site behind `--features gpu`.
- **Multi-GPU / distributed** — long-term, not on the radar.

## Validation layers

For development with the Khronos validation layer enabled:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    cargo test -p hymeko_compute -- --ignored
```

Validation layer messages go to stderr.
