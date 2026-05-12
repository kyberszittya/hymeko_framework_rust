# KEPAF → T-SMC-S journal extension plan (path 2)

## Why

The KEPAF 2026 short paper (`paper/kepaf_v1/`) lands the Vulkan compute
kernels (`hymeko_compute`: `force_directed`, `signed_spmv`) with measured
$343\times$ / $1\,654\times$ speed-ups vs the CPU baseline at MNIST and
synthetic-$|V|=10^4$ scales. The §VII-G "Portability and threats to
validity" subsection explicitly names three gaps that this journal
extension closes:

1. **Vendor portability**: only NVIDIA + Linux measured.
2. **Compute API**: only native Vulkan measured; the browser WebGPU
   path uses WGSL (not the same source as the GLSL `.comp` files in
   `hymeko_compute/shaders/`) and has unmeasured per-submit IPC cost.
3. **Form-factor**: mobile / Apple-silicon unmeasured; "browser
   deployable" implies these should at least degrade gracefully.

Plus a missing dogfood loop:

4. **Headless rendering**: the browser viewer renders to live SVG/canvas
   only; there is no offscreen path that produces a PNG/SVG for paper
   figures. The KEPAF figures are matplotlib-rendered over GPU-computed
   positions, which is a half-dogfood. The journal version eliminates
   the matplotlib dependency from the figure pipeline.

## What ships in the journal extension

### M1 — WGSL ports of the two kernels

Translate `shaders/force_directed.comp` and `shaders/signed_spmv.comp`
into WGSL. Validate against the existing GLSL output on the same
fixtures (canonical / mnist_adj / synthetic-35k); positions and SpMV
results must agree to $10^{-4}$ relative.

Files to add:
- `hymeko_compute/shaders/force_directed.wgsl`
- `hymeko_compute/shaders/signed_spmv.wgsl`
- `hymeko_compute/src/kernels/force_directed_wgsl.rs` (alternate dispatch
  path via `wgpu` 0.19+, gated behind a `wgsl-backend` feature)
- `hymeko_compute/src/kernels/signed_spmv_wgsl.rs`

Acceptance: `cargo test -p hymeko_compute --features wgsl-backend
-- --ignored` runs both kernel pairs and asserts numerical agreement.

### M2 — Cross-vendor measurement matrix

Run the existing `bench_kepaf` example on three vendor classes via
`wgpu`'s adapter selection:

| vendor              | api      | runtime          |
|---------------------|----------|------------------|
| NVIDIA RTX 2070 SUPER | Vulkan | native           |
| NVIDIA RTX 2070 SUPER | WebGPU | wgpu native      |
| Intel iGPU (UHD 6xx)  | Vulkan | native           |
| Intel iGPU            | WebGPU | wgpu native      |
| AMD (RDNA2/3, if available) | Vulkan / WebGPU | both     |

Plus one mobile or Apple-silicon target via someone's M1/M2 box, even
if just spot-measured.

Files to add:
- `hymeko_compute/examples/bench_kepaf_wgsl.rs` (sibling to
  `bench_kepaf.rs`, dispatches via `wgpu` instead of `vulkano`)
- `paper/arxiv_v1/results/cross_vendor_bench.csv` (or whichever paper
  tree the journal version lives in)

### M3 — Headless SVG / PNG export from the WASM viewer

Add `render_to_svg(snapshot, positions, grammar) -> String` to
`hymeko_wasm`, exporting the three-colour-band sign-aware grammar
already implemented for the live canvas. CLI wrapper:

```bash
cargo run -p hymeko_wasm --example svg_dump -- \
    --hymeko data/robotics/anthropomorphic_arm.hymeko \
    --positions paper/.../mnist_adj.positions.json \
    --out fig.svg
```

This eliminates `matplotlib` from the figure pipeline and lets the
journal version's figures be produced **by the same code path the
browser viewer runs**, not by a Python re-render.

Files to add:
- `hymeko_wasm/src/render_svg.rs`
- `hymeko_wasm/examples/svg_dump.rs`
- `scripts/regenerate_journal_figures.sh` (orchestrator that calls
  layout binary + svg_dump for each fixture)

### M4 — Browser end-to-end benchmark

Instrument the WASM bundle in `docs/demo/` to time:

- WGSL compile-and-cache (one-shot, per page load)
- per-frame dispatch (force_directed step, signed_spmv aggregation)
- per-frame render (positions → SVG via M3)

Run in Chromium and Firefox, log to `docs/demo/results/`. Compare to
the native Vulkan numbers from M2 — expected: WebGPU within $\approx
2\times$ at $|V|>10^3$, dominated by IPC at smaller $|V|$.

Acceptance: a JSON report file per browser, with per-frame timings
across the three fixtures.

## Out of scope

- Barnes-Hut / octree spatial-acceleration kernel. (Phase 5 of the
  KEPAF roadmap; needs the WGSL ports and cross-vendor proof first.
  The headline $1\,654\times$ already comes without it.)
- Mobile-specific optimisations (subgroup operations, half-precision).
- Distributed / multi-GPU.

## Dependencies on the conference version

The conference version's figures and §VII-F numbers stand. The
journal version adds — does not replace — the cross-vendor /
cross-runtime / browser-end-to-end measurements. The §VII-G
"Portability and threats to validity" disclosures stay; they explain
why the conference version is honest about what it does and does not
prove.

## Owner / timeline

- M1 (WGSL ports): 1–2 days, owner self.
- M2 (cross-vendor matrix): 1 day, owner self + ad-hoc Apple/mobile
  collaborator.
- M3 (SVG export): 1 day, owner self.
- M4 (browser end-to-end): 1 day, owner self.

Total ≈ 1 working week. Target: T-SMC-S submission window after the
KEPAF feedback round.
