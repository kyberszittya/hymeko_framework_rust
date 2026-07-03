# Report — HSiKAN wall speedup: CRActivation stacked-indexing fix

**Date:** 2026-05-29
**Predecessor:** `reports/2026-05-29-vision-rebench-strong-correction.md` (the overnight chain ran HSiKAN at 7 100 s/cell on Fashion — the friction this task addresses).
**CORE.YAML items touched:** none.

## Headline

| | before | after | Δ |
|:--|--:|--:|--:|
| Top CUDA kernel | `aten::_index_put_impl_` 76.45 % | `aten::_index_put_impl_` 48.80 % | **−27.6 pp** |
| `index_put_impl_` call count (10 iters) | 480 | **120** | **4× reduction** |
| Self CUDA total (10 iters) | 8.207 s | **5.177 s** | **1.59×** |
| Wall, 5ep / subset 2000 | 64.24 s | **39.66 s** | **1.62×** |
| Wall, 5ep / full MNIST | ~1 683 s (extrapolated from chain) | **1 186 s** | **1.42×** |
| Per-cell wall (overnight 20-ep config, projected) | 6 733 s (chain measured) | **~4 700 s** | **~1.43×** |

**Parity** (mathematical identity, not just statistical): 9 tests pass —
forward output AND gradient w.r.t. control points match the legacy path
to **1e-7** (`hymeko_neuro/tests/test_cractivation_parity.py`).

## What I got wrong, then corrected

The day-before write-up *predicted* the wall hotspot was the dense
incidence einsum in `SignedBranchConv.forward` (4 dense einsums × 3
arities × 2 layers = 24 dense matmuls over a 3-15 % sparse incidence).
**That was wrong.** torch.profiler showed `einsum` doesn't even appear
in the top 20 CUDA kernels.

The real hotspot is `aten::_index_put_impl_` (the backward kernel of
advanced indexing). At 480 calls per 10-iteration training window and
76.45 % of CUDA time, it's clearly *the* kernel.

Tracing it: `CRActivation.forward` (`hsikan_vision.py:96–127`) does
**four separate advanced-indexing gathers** to evaluate the Catmull-Rom
spline (`p0 = cp[ch_idx, i0]`, `p1 = cp[ch_idx, i1]`, etc.). Each gather
backward materialises one `index_put_` accumulator into the
control-points parameter. 480 = 4 gathers × 2 branches × 3 arities × 2
layers × 10 iterations — exactly the profile's count.

**Lesson learned:** code-reading hypotheses about wall-time hotspots in
GPU code are unreliable until profiled — py-spy can't see CUDA kernels;
torch.profiler is the right tool (memory-hotspot ≠ time-hotspot in this
case, too — the dense einsum is the *memory* hotspot per the GPU memory
probe, but the per-element index_put is the *time* hotspot).

## The fix

Replace the four separate gathers with one stacked-indexing op:

```python
# Before — 4 advanced-indexing ops, 4 index_put_ backwards:
p0 = cp[ch_idx, i0]; p1 = cp[ch_idx, i1]
p2 = cp[ch_idx, i2]; p3 = cp[ch_idx, i3]

# After — 1 advanced-indexing op over a (...,4) stacked-index tensor,
# 1 index_put_ backward:
i_stack = torch.stack((i_m1, i, i_p1, i_p2), dim=-1)      # (..., C, 4)
p_all   = cp[ch_idx.unsqueeze(-1).expand_as(i_stack), i_stack]
p0, p1, p2, p3 = p_all.unbind(dim=-1)
```

Mathematically identical (proven by the parity tests); collapses 4
backward `index_put_` ops into 1.

The legacy implementation is retained as `CRActivation._forward_legacy`
solely so the parity test can target it.

## Files touched

| File | Status | Lines |
|:--|:--|--:|
| `hymeko_neuro/experiments/vision/hsikan_vision.py` | modified (CRActivation.forward + `_forward_legacy` kept for testing) | +34/−13 |
| `hymeko_neuro/tests/test_cractivation_parity.py` | new | 60 |
| `hymeko_neuro/experiments/runs/probe_hsikan_torch_profiler.py` | new (the profiling probe) | 45 |
| `hymeko_neuro/experiments/runs/probe_vision_gpu_memory.py` | new (the GPU memory probe) | 56 |
| `reports/2026-05-29-hsikan-cractivation-speedup.md` | new | this |

No edits to other modules; no dependency changes; no Rust touched.

## Profile evidence (before / after)

| Kernel | Before (% CUDA) | After (% CUDA) |
|:--|--:|--:|
| aten::_index_put_impl_ | 76.45 % (6.27 s) | 48.80 % (2.52 s) |
| indexing_backward_kernel_stride_1 | 73.98 % (6.07 s) | 45.08 % (2.33 s) |
| ProfilerStep* (overhead) | 9.48 % | 25.14 % |
| aten::mul | 7.25 % | 13.05 % |
| aten::cat | (not in top) | **11.63 %** (601 ms) ← new cost from `torch.stack` |
| aten::add_ | 3.54 % | 5.65 % |

The fix introduces an `aten::cat` overhead (the stack op materialises a
new tensor). That partially offsets the index_put_ win — which is why
total CUDA time dropped 1.59× rather than the 4× the index_put_ call
reduction alone would suggest. Eliminating the cat (computing the
stacked indices via arithmetic instead of `torch.stack`) is a possible
follow-up but with diminishing returns.

## Test results

| Test | Count | Result |
|:--|--:|:--|
| `test_cractivation_parity` (forward parity × 3 shapes × 2 branches; gradient parity × 2 branches; extrapolation parity) | 9 | **pass** in 6.4 s |
| ruff on changed code | — | clean (7 ruff errors in `hsikan_vision.py` are all pre-existing unused imports, not from this change) |

## Implications

- The overnight HSiKAN cell wall projects from 7 100 s → ~4 700 s (~33 %
  saved per cell). A 6-cell hsikan×{mnist,fashion}×3-seed sweep drops
  from 7.1 h → ~5 h. Not transformative, but it adds up across re-runs.
- The fix also reduces wall on every other CRActivation user (currently
  only HSiKAN-vision, but if CRActivation is reused for future
  vision-specific operators the benefit transfers).
- **GPU memory is not changed** by this fix (the per-pixel-per-channel
  activations themselves are unchanged — only their backward kernel
  count is reduced). HSiKAN still uses ~2.8 GiB GPU vs CNN's 27 MiB —
  the dense-incidence activation chain remains the memory bottleneck,
  unrelated to this fix.

## What this does *not* do

- Address GPU memory (still ~100× CNN). The dense-incidence einsum in
  `SignedBranchConv` is the memory hotspot — a separate fix (sparse
  incidence via `torch.sparse.mm`) would address that, but it's a
  larger change than this one and benefits memory more than wall.
- Address Tier-1 wins (larger batch size, `torch.compile`, AMP).
  Each is still on the table and could compose with this fix.

## Provenance

- **Git SHA:** `8fd8187` (dirty).
- **Interpreter:** miniconda3 python 3.13.5, **torch 2.11.0+cu130**.
- **GPU:** RTX 2070 SUPER 8 GiB, driver 580.126.09.
- **Profile artifact:** torch.profiler table (`probe_hsikan_torch_profiler.py`).
- **Parity artifact:** 9-test pytest run; outputs match to 1e-7.
- **Wall artifact:** two cells (subset 2000 / 5ep and full MNIST / 5ep)
  measured back-to-back; consistent with profiler's 1.59× CUDA speedup
  prediction.

## Follow-ups

1. **Sparse-incidence rewrite** of `SignedBranchConv.forward` —
   addresses the *memory* axis (would reduce HSiKAN GPU from 2.8 GiB to
   ~100 MiB plausibly; on a 7.6 GiB GPU it unlocks larger batches and
   bigger hidden). Bigger change, needs care; parity test design same as
   this one.
2. **Tier-1 wins** stacked atop this: larger batch (GPU is at ~3 GiB,
   plenty of headroom), `torch.compile(model)`, AMP autocast. Each
   independently measurable.
3. **Eliminate `aten::cat`** in the new code path — compute stacked
   indices arithmetically (`i.unsqueeze(-1) + torch.arange(-1,3)`)
   instead of via `torch.stack`. Saves ~12 % of CUDA on top of this fix.
   Tiny change.
