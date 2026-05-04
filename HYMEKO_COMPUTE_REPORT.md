# `hymeko_compute` v0.1 — Vulkan Compute Kernels for Hypergraph IR

**Date:** 2026-04-29
**Branch:** `refactor/extract-hymeko-hre`
**Status:** All three GPU integration tests green on RTX 2070 SUPER (Vulkan 1.3.275 / driver API 1.4.312).

---

## 1. Scope and intent

A new self-contained crate, `hymeko_compute`, that runs compute shaders on the native Vulkan device against the canonical signed-incidence hypergraph IR. The boundary with the rest of the workspace is deliberate:

- `hymeko_wasm` retains the browser / WebGPU **visualisation** path; no shader sharing in this revision.
- `hymeko_hnn` retains its CPU compute paths; `hymeko_compute` is the **GPU** path. Downstream wiring (e.g. an `hymeko_hnn` GPU feature flag) is a follow-up, not part of v0.1.

The crate consumes the SoA buffer layouts already produced by `hymeko_hnn::traversal::HyperGraphView` and `hymeko_core::tensor::TensorCsr` directly — no re-marshalling.

---

## 2. Crate layout

```
hymeko_compute/
├── Cargo.toml          (vulkano 0.34, bytemuck 1, thiserror 1)
├── build.rs            (glslc-based GLSL → SPIR-V at build time)
├── shaders/
│   ├── vector_add.comp
│   ├── signed_spmv.comp
│   └── force_directed.comp
└── src/
    ├── lib.rs          (crate doc + module wiring)
    ├── context.rs      (VulkanContext: instance / device / queue / allocators)
    ├── buffers.rs      (upload / download / storage helpers)
    └── kernels/
        ├── mod.rs
        ├── vector_add.rs
        ├── signed_spmv.rs
        └── force_directed.rs
```

Total: **1 125 LoC** across Rust + GLSL + build glue.

---

## 3. Design decisions (final, after bring-up)

### 3.1 Vulkan binding: `vulkano` 0.34

Real Vulkan under the hood, Rust-idiomatic surface; the trade vs `ash` is full driver control for time-to-running-kernel. Workload is research compute, not driver development — `vulkano` wins.

### 3.2 Shader compilation: `build.rs` + `glslc` + `include_bytes!`

The original plan called for the `vulkano_shaders::shader!` proc macro. That path hit a workspace-specific `E0463: can't find crate for vulkano_shaders` even with the dep pinned correctly. We pivoted to a small `build.rs` that walks `shaders/`, invokes `glslc -O -fshader-stage=compute`, and writes `.spv` blobs into `OUT_DIR`. The kernel modules then `include_bytes!(concat!(env!("OUT_DIR"), "/foo.spv"))` and hand the words to `vulkano::shader::ShaderModule::new`. Net effect: identical UX (compile-time validated SPIR-V), no proc-macro dependency, identical iteration speed.

### 3.3 No CPU fallback in this crate

The CPU paths already exist in `hymeko_hnn`. Mixing fallbacks here would muddle the contract. Tests are guarded by `#[ignore]`; CI without a GPU skips them rather than silently degrading.

### 3.4 Naïve $O(N^2)$ force summation first; octree later

KEPAF §IV promises Barnes-Hut. The harness comes first, the asymptotic speedup second — once the round-trip is proven, swapping the inner loop is a localized change.

---

## 4. The three kernels

### 4.1 `vector_add` — proof of life

Elementwise `c[i] = a[i] + b[i]`. One SSBO triple, one workgroup-thread per element, 64-wide local size. Used by the smoke test that the harness is alive on a fresh box. Not a real workload.

**Test:** 1 024-element round-trip; every output equals `1024.0`.

### 4.2 `signed_spmv` — signed-incidence SpMV $\mathbf{y} = \mathbf{B}\mathbf{x}$

CSR view of the signed-incidence matrix $\mathbf{B}$ (rows = vertices, columns = hyperedges, entries in $\{-1, 0, +1\}$ scaled by an optional weight). Field semantics match `hymeko_core::tensor::TensorCsr` exactly — `row_ptr`, `col_ind`, `val`. One workgroup-thread per row.

This is the workhorse primitive used by every `hymeko_hnn` convolution variant (`signed_hgnn`, `hgnn`, `gcn_clique`).

**Test:** Worked example $\mathbf{B} = \begin{pmatrix} +1 & 0 \\ -1 & +1 \\ 0 & -1 \end{pmatrix}$, $\mathbf{x} = (2, 3)^\top$ ⇒ $\mathbf{y} = (2, 1, -3)^\top$. Matches analytic answer to $10^{-6}$.

### 4.3 `force_directed` — Fruchterman–Reingold layout

The first KEPAF §IV deliverable. Per-vertex thread sums repulsion against every other vertex, attraction along incident arcs, applies $-\gamma v$ damping, integrates with semi-implicit Euler. One Vulkan dispatch per layout iteration; the host loop just rebinds and submits.

- Repulsion: $k_r / d^2$ pairwise.
- Attraction: $k_a (d - L_0)$ along each arc.
- Damping: $-\gamma v$.

**Test:** 21-vertex canonical fixture, 30 arcs, 100 iterations. Asserts no vertex went to NaN/Inf; full integration (edge-crossing count vs CPU reference) is the next step.

---

## 5. Bring-up bugs found and fixed

### 5.1 `forbid(unsafe_code)` blocked `ShaderModule::new`

`vulkano::shader::ShaderModule::new` is `unsafe` by Vulkan invariant — malformed SPIR-V can corrupt driver state. The crate started with `#![forbid(unsafe_code)]` in `lib.rs`, which blocks the required `unsafe { … }` block at compile time.

**Fix:** Drop the forbid. Each call site is scoped to SPIR-V words produced by our own `build.rs`, so the unsafety is contained and documented at the lib-level.

### 5.2 `storage()` returned device-local memory; `read_back` panicked with `NotHostMapped`

The original `buffers::storage` helper allocated `MemoryTypeFilter::PREFER_DEVICE`. The `force_directed` test then called `read_back(&buf_pos)` to inspect final positions — but device-local memory isn't host-mapped, so `Subbuffer::read()` returned `NotHostMapped`.

**Fix:** Relax `storage` to `PREFER_HOST | HOST_RANDOM_ACCESS`. For research-scale workloads ($|V| \leq 10^5$) the perf cost is negligible. The device-local + staging-copy variant is the right follow-up when a buffer survives many dispatches without ever leaving the device.

### 5.3 Shader declared params block as `uniform`, host bound it as storage

The `signed_spmv` and `force_directed` shaders declared their `Params` block with `layout(...) readonly uniform Params { … }`. The host-side `buffers::upload()` helper only sets `BufferUsage::STORAGE_BUFFER`. The descriptor mismatch surfaced as `"a validation error occurred"` at dispatch time on the SpMV kernel, with no indication of which binding.

**Fix:** Switch the GLSL declarations to `readonly buffer Params { … }`. For a research crate where every binding is small enough to live in an SSBO, this also keeps the binding logic uniform across kernels.

---

## 6. Verification

```text
$ cargo test -p hymeko_compute --lib -- --ignored --nocapture

running 3 tests
test kernels::vector_add::tests::vector_add_round_trip ... ok
test kernels::signed_spmv::tests::signed_spmv_tiny      ... ok
test kernels::force_directed::tests::force_directed_canonical ... ok

test result: ok. 3 passed; 0 failed; 0 ignored
```

Device under test:

```
deviceName  = NVIDIA GeForce RTX 2070 SUPER
apiVersion  = 1.4.312 (Vulkan SDK 1.3.275 toolchain)
driverName  = NVIDIA
```

`cargo build -p hymeko_compute` is clean (no errors, no warnings).

---

## 7. What this unlocks

- **KEPAF §IV moves from "design" to "implemented".** The paper promised three GPU kernels; the FR force-summation primitive is now a real artefact with a passing integration test.
- **`hymeko_hnn` has a viable GPU primitive.** Every convolution variant is signed-incidence SpMV at its core; once the feature flag wires `hymeko_compute` in, the GPU path is one descriptor-set-build away.
- **The harness is reusable.** Adding a fourth kernel is now: drop `foo.comp` in `shaders/`, write a 100-line `kernels/foo.rs` that mirrors the three existing ones, write one `#[ignore]` test.

---

## 8. Out of scope (deferred to follow-ups)

- **Octree / Barnes-Hut acceleration** of the force-directed kernel. The natural Phase 5 of the KEPAF roadmap; needs the basic harness first.
- **Shader sharing with the WebGPU browser path** via wgpu's WGSL→SPIR-V. Possible later; not now.
- **Wiring `hymeko_compute` into `hymeko_hnn`** as a feature flag. Possible follow-up.
- **Distributed / multi-GPU.** Not now.
- **Vulkan validation-layer-driven CI gate.** Manual `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` runs are documented; CI doesn't have a GPU.

---

## 9. Files in scope

| File | LoC | Role |
|------|----:|------|
| `hymeko_compute/Cargo.toml` | 27 | crate manifest |
| `hymeko_compute/build.rs` | 47 | GLSL → SPIR-V at build time |
| `hymeko_compute/src/lib.rs` | 54 | crate doc + module wiring |
| `hymeko_compute/src/context.rs` | 191 | `VulkanContext` |
| `hymeko_compute/src/buffers.rs` | 108 | typed buffer helpers |
| `hymeko_compute/src/kernels/mod.rs` | 18 | kernel module index |
| `hymeko_compute/src/kernels/vector_add.rs` | 134 | proof of life |
| `hymeko_compute/src/kernels/signed_spmv.rs` | 185 | $\mathbf{y} = \mathbf{B}\mathbf{x}$ |
| `hymeko_compute/src/kernels/force_directed.rs` | 251 | Fruchterman–Reingold |
| `hymeko_compute/shaders/vector_add.comp` | 13 | GLSL |
| `hymeko_compute/shaders/signed_spmv.comp` | 30 | GLSL |
| `hymeko_compute/shaders/force_directed.comp` | 67 | GLSL |
| **Total** | **1 125** | |
