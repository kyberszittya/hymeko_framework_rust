# Pool-scatter Triton stack — final state of the 2026-06-05/06 optimisation arc

**Date:** 2026-06-06 afternoon (closing the two-day overnight + day session)
**Scope:** Final state report of the AC-HSiKAN pool-scatter backward
optimisation arc. Documents the headline measurement (Triton-stack path
on full IMDB 5-seed) that the project adopts as the production number.

## Adopted headline

Full IMDB, 25 000 train / 5 000 val, L = 200, 8 epochs, batch 64,
lr 3e-3, AdamW(wd=1e-4), CUDA RTX 2070 SUPER, AC v1.6
(pool-scatter + entropy Hamilton rotor + Triton bwd + Triton Hamilton fwd):

| config | val_acc (5-seed) | params | wall (5 seeds) | Δ vs Transformer |
|---|---:|---:|---:|---:|
| Transformer baseline       |   0.8535 ± 0.0051 | 166 594 |   122 s |   0       |
| **AC-HSiKAN v1.6 (Triton)** | **0.8480 ± 0.0046** | 164 668 |   423 s | **−0.0055** |

Paired t = −2.94 (df=4), p slightly below 0.05; wins 1/5 for AC.
The −0.0055 effect size is **on the order of seed-noise** (σ ≈ 0.005);
the t-statistic crosses the α=0.05 threshold by a thin margin.

## Why the Triton-optimised path is the adopted reference

| path | acc | wall | reasoning |
|---|---:|---:|---|
| PyTorch closed-form (2026-06-06 morning) | 0.8489 ± 0.0037 | 568 s | numerically bit-equal to original autograd; statistical paritás (p>0.05) |
| **Triton stack (2026-06-06 afternoon)**   | **0.8480 ± 0.0046** | **423 s** | numerical drift ≈ 4e-7 / call from FMA ordering; 1.34× wall; adopted |

Per-seed Δ between paths: ranges from −0.0028 to +0.0014, mean
−0.0009 (one-fifth of a seed σ). **Not a regression**; numerical
drift from Triton-fused FMA ordering across ~10⁷ float operations per
training step. The downstream effect on validation accuracy is below
the granularity that distinguishes architectures on this task.

## Optimisation arc — 41× cumulative on pool-scatter bwd

| step | bwd (component) | end-to-end IMDB smoke wall |
|---|---:|---:|
| 2026-06-05 evening baseline (autograd-through-PyTorch-ref) | 114.0 ms | 165 s |
| + closed-form CR-coef scatter_add (12×)        | 9.6 ms | 23 s |
| + packed 8 scatter_adds into 1 (1.94×)         | 5.0 ms | 17 s |
| + Hamilton coeffs LRU cache + ctx-save Q/K/V/pool/scatter | 4.5 ms | 17 s |
| **+ Triton fused CR-coef kernel (3.84×)**       | **2.78 ms** | 14.9 s |
| **+ Triton Hamilton rotor (forward only, 11×)** | **2.78 ms** | **14.5 s** |

Cumulative: **41× bwd component speedup**, **11.4× end-to-end IMDB smoke wall**.

Full-IMDB 5-seed: 568 → 423 s = **1.34×**, or 20× vs the original
PyTorch-autograd-reference path.

## Final profile (after all opts, classifier-level)

```
forward                 4.24 ms  (was 4.29)
forward + backward     17.79 ms  (was 18.53)
  ↳ backward alone    13.55 ms  (was 14.24)
```

Top CUDA-time consumers (10-iter profile):
- `_cr_coef_backward_kernel`              27.6 %   (Triton, atomic-throughput floor)
- `Hamilton rotor cat` (worker-thread)     8.7 %   (PyTorch fallback in bwd)
- `mm + addmm + gemvx + gemmk1`           22.0 %   (per-arity + sign-attn matmuls)
- `reduce_kernel`                           5.3 %   (LayerNorm + dropout)
- `_fused_pool_scatter_backward_kernel`     2.4 %   (Triton dQ/dK/dV)
- `_fused_pool_scatter_forward_kernel`      1.5 %   (Triton fwd)
- other                                   ~32.5 %

The CR-coef kernel is now atomic-throughput-bound on the small
`(2·h·G,) = 128`-entry target; block-reduction Triton variant
deferred (estimated ≤ 2× upper bound for 200-300 LOC of kernel work).

## What did NOT pay off

- **`_cr_coef_backward_both` (one combined scatter_add)** — 0 % gain;
  the scatter is atomic-bound, not launch-bound (verified: 0.55 ms
  for 128-entry vs 0.095 ms for 4096-entry target on same op count,
  5.8× contention floor).
- **Per-arity Linear → batched matmul** — net SLOWDOWN (14.2 → 14.8 ms);
  the (16, 16) matmuls are too small to benefit from batching, the
  stack/unbind overhead dominates.
- **Hamilton rotor einsum** — 1.05× (already well-optimised by PyTorch).
- **`torch.compile(fullgraph=False)`** — at component level 1.48× faster
  steady-state; at the IMDB smoke level **2.8× SLOWER** with a small
  numerical drift that hurt accuracy (−0.012). Disabled by default;
  `--compile` flag retained for L ≥ 512 experiments.
- **CUDA Graphs** — blocked: the autograd-managed custom Function
  raises ``operation not permitted when stream is capturing``.
- **Sparse sign-head** — 1.07× wall but **5× higher seed variance**
  (.020 vs .003) on smoke; not a good trade.
- **`torch.cuda.set_device` + `empty_like` workarounds** for Triton in
  the autograd worker thread — still hit ``invalid device context``.
  Hamilton Triton kernel kept main-thread-only (MainThread guard).

## Tests

- 50 / 50 passes on `test_pool_scatter_rotor_parity` (3) +
  `test_evolvent_telemetry` (6) + `test_ac_hsikan` (41).
- All parity tests at numerical noise: ∂L/∂x = 0.0, W_{q,k,v,back}
  at 1e-10, coef_{pos,neg} at 0.0, entropy_axis at 4.5e-13, entropy_beta
  at 0.0. End-to-end on IMDB: mean per-seed Δ from PyTorch-reference
  path is −0.0009 (within 1/5 of seed σ).

## Files touched (across the two-day arc)

- [hymeko_neuro/models/ac_hsikan/components/pool_scatter.py](../hymeko_neuro/models/ac_hsikan/components/pool_scatter.py) —
  closed-form CR-coef bwd, packed scatter_add, ctx-save, Triton
  CR-coef + Hamilton dispatch, MainThread guard for worker-thread
  fallback.
- [hymeko_neuro/kernels/triton_kernels/fused_pool_scatter.py](../hymeko_neuro/kernels/triton_kernels/fused_pool_scatter.py) —
  new `_cr_coef_backward_kernel`, new `_hamilton_rotate_kernel`,
  Python wrappers `cr_coef_backward_triton` + `hamilton_rotate_triton`.
- [hymeko_neuro/models/ac_hsikan/layer.py](../hymeko_neuro/models/ac_hsikan/layer.py) —
  pre-norm + LayerScale opt-in (deep-stability fix), `_local_indices`
  buffer-cache, entropy-scalar pass to pool_scatter rotor.
- [hymeko_neuro/models/ac_hsikan/config.py](../hymeko_neuro/models/ac_hsikan/config.py) —
  `use_pool_scatter_rotor`, `use_pre_norm`, `use_layer_scale`,
  `layer_scale_init` flags.
- [hymeko_neuro/models/ac_hsikan/telemetry.py](../hymeko_neuro/models/ac_hsikan/telemetry.py) —
  `EvolventTelemetry` context manager + JSONL sink + ctx-snapshot
  pattern to survive PyTorch's autograd worker thread.
- [hymeko_neuro/experiments/ac_hsikan_imdb_smoke.py](../hymeko_neuro/experiments/ac_hsikan_imdb_smoke.py) —
  new `--pool-scatter`, `--rotor`, `--telemetry-out`, `--clifford-fir`,
  `--fused-walk`, `--sparse-sign-head`, `--compile`, `--d-model`,
  `--n-layers`, `--pre-norm`, `--layer-scale` flags.
- 3 new test files: `test_pool_scatter_rotor_parity.py`,
  `test_evolvent_telemetry.py`, plus updates to `test_ac_hsikan.py`.

## CORE.YAML items touched

None. `hymeko_neuro/` is non-core.

## Open / next

- **Architecture-side scaling**: 4 axes (d_model width, depth L=4
  via pre-norm + LayerScale, FFN-add, lr-shift) tested in the same
  arc — none unlocked accuracy gains above the d=16 L=2 sweet spot.
  The current AC-HSiKAN architecture is parameter-efficient but does
  not capacity-scale on IMDB. Next directions in this space:
  L = 400+ (where pool-scatter's wall advantage scales), Long Range
  Arena tasks, or revisiting wider top_k_per_position.
- **Block-reduce Triton kernel for CR-coef bwd**: ~200-300 LOC, est.
  ≤ 2× on the C component, ~10% on classifier. Deferred (cost vs gain).
- **CUDA Graphs / custom autograd Function rewrite**: blocks the
  next ~2× on classifier wall. Multi-day effort.

## Result artefacts (transient)

- `/tmp/imdb_full_5seed_today.json` — adopted headline run
- `/tmp/imdb_full_5seed_optimized.json` — pre-Triton-kernels baseline
- `/tmp/imdb_full_evolvens.seed*.jsonl` — per-step rotor + gradient/error
  telemetry from the prior run
