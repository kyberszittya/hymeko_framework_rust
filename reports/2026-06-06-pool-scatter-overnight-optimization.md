# Pool-scatter backward optimization — overnight session 2026-06-05/06

**Date:** 2026-06-06 (started evening of 2026-06-05)
**Scope:** Profile-driven optimisation of the AC-HSiKAN pool-scatter
backward path, after the user's morning observation that pool-scatter
slowed IMDB training by 19× over walk-op only. **No accuracy regression
at any step**; parity verified via the 9-parameter rotor-parity test
plus end-to-end IMDB val_acc match.

## Headline

| step | pool-scatter bwd | cumulative | accuracy (5k smoke) |
|---|---:|---:|---:|
| baseline (autograd-through-reference) |  114.0 ms | 1.00× | 0.7452 ± 0.006 |
| 1. closed-form CR-coef scatter_add    |    9.64 ms | **12×** | 0.7455 ± 0.011 |
| 2. pack 8 scatter_adds into 1         |    5.00 ms | 23× | (same) |
| 3. LRU-cache Hamilton coeffs          |    4.59 ms | 25× | (same) |
| 4. save Q_signed/K/V/pool_h/scatter_h |    4.46 ms | **26×** | (same) |

End-to-end IMDB smoke (5 000 train / 2 000 val / 4 epochs / 2 seeds):

| run | val_acc | wall | × baseline |
|---|---:|---:|---:|
| original v1.6 + rotor                       | 0.7452 ± .006 | 165.4 s | 1.0× |
| optimized v1.6 + rotor (no compile)         | 0.7430 ± .011 |  17.0 s | **9.7×** |
| optimized + `torch.compile(fullgraph=False)` | 0.7338 ± .008 |  66.5 s | 0.4× (worse) |

Pool-scatter is now **2× slower than walk-op only** instead of 19×.
A full-IMDB 5-seed run (25k/5k/8ep/L=200) becomes ≈ 8–10 min wall, not 90.

## What was the bottleneck

Initial profile (B=64, L=128, K=8, h=8, n_quat=2, G=8 — IMDB shape):

```
forward (full)                   0.30 ms
forward + backward (full)      114.32 ms
  A. Triton fwd re-run (in bwd)  0.28 ms  (0.2%)
  B. Triton dQ/dK/dV bwd         0.25 ms  (0.2%)
  C. CR-coef autograd-ref      111.45 ms  (97.7%)
  D. matmul chain                2.04 ms  (1.8%)
```

97.7 % of the backward was spent in the autograd-through-PyTorch-reference
path computing the CR-coefficient gradients. The Triton-fused dQ/dK/dV
kernel was 0.25 ms — the rest was building the full computation graph
of the PyTorch reference and `torch.autograd.grad`-ing through it,
solely to differentiate two tiny (h, G) = (8, 8) = 64-element tensors.

## Optimisation 1 — closed-form CR-coef gradient (12×)

The Catmull–Rom interpolation is

  S(R) = w_{−1}·P_{−1} + w_0·P_0 + w_{+1}·P_{+1} + w_{+2}·P_{+2}

where P_g are gathered from `coef_{pos|neg}[c, idx_g(R)]` and w_g are
fixed cubic functions of `t = frac(|R|·(G−1))`. The gradient is therefore

  ∂L/∂coef[c, idx_g] += Σ_{...} (∂L/∂S)[..., c] · w_g[..., c]

with the sign of R choosing the buffer. This is a `scatter_add_` into
the flat `(h·G,)` accumulator. No autograd graph, no reference forward
re-run.

Backward time dropped from 114 ms to 9.64 ms. The remaining 4 ms is
the scatter_add itself plus auxiliary tensor ops.

## Optimisation 2 — pack 8 scatter_adds into 1 (1.94×)

The naïve version did 8 launches: 4 CR knots × 2 sign branches. Packing
all into a single combined target of shape `(2·h·G,)` and stacking the
four knot indices/contribs along a new axis collapsed it to one
`scatter_add_` per call. **1.94× win** despite no algorithmic change —
purely kernel-launch overhead.

```
# Old: 8 scatter_adds per call
for idx_x, w_x in knots:
    pos_flat.scatter_add_(0, idx_x_flat, (dL_pos * w_x).flat)
    neg_flat.scatter_add_(0, idx_x_flat, (dL_neg * w_x).flat)

# New: 1 scatter_add per call
combined = torch.zeros(2 * h * G)
combined.scatter_add_(0, stacked_idx.flat, (stacked_w * dL_dS).flat)
```

## Optimisation 3 — LRU-cache Hamilton coefficients (10 %)

The `(1, -1, -1, -1) × n_quat` coefficients tensor was reallocated on
every forward AND backward call (3 sites). `@functools.lru_cache` keyed
on `(n_quat, device, dtype)` is a one-line fix.

## Optimisation 4 — save Q_signed/K/V/pool_h/scatter_h in ctx (3 %)

The backward path was recomputing `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
and re-running the Triton forward to recover `pool_h`, `scatter_h`. All
already exist in forward. Saving them costs ~0.4 MB / call (5 tensors of
shape (B, L, h) at IMDB shape); the matmul + Triton fwd re-run was
~0.13 ms / call. Small but free.

The pre-rotor `scatter_h` is what backward needs (the closed-form CR-coef
path operates on the pre-rotation field; the rotor's M* reverse-rotation
on grad_scatter_h already accounts for the rotor's effect on the
gradient). Captured before the `_hamilton_rotate_static` call in forward.

## What was tried and did NOT help

### Fuse pool + scatter into one `_cr_coef_backward_both` call

Stacking R_pool and R_scatter along a new axis to do both paths in one
scatter_add: **no measurable win** (4.96 → 5.11 ms). The scatter_add
is atomic-throughput-bound on the 128-entry target, not launch-bound.
Verified directly: `scatter_add to 128-entry target = 0.55 ms` vs
`scatter_add to 4096-entry target = 0.095 ms` for the same 2 M
contributions — 5.8× difference confirms atomic contention is the floor.

### Block-reduce Triton kernel for the scatter

Would reduce per-element atomics by aggregating per program block into
a small local accumulator before the global atomic. Estimated upper-bound
gain: ~2× on the C component. Cost: 200–300 LOC of Triton + parity
tests. Given the C component is already 3.6 ms and the **end-to-end
classifier** is 21.5 ms (only 17 % is C), the expected end-to-end gain
is < 10 %. **Not done this session**; deferred as future work.

### `torch.compile(mode="default", fullgraph=False)` on the classifier

At the FusedPoolScatter component level: 1.48× faster steady-state.
At the IMDB smoke level: **2.8× SLOWER** with **−0.012 acc drift**.
The compile warmup overhead doesn't amortise over 4 epochs on 5k train
(~16 s extra on seed 0), and the slight numerical drift (probably
reduction-order changes in fused kernels) hurt training trajectory.
**Not enabled by default**. May be worth revisiting for L ≥ 512 +
8 + epochs.

## Files touched

- [signedkan_wip/src/ac_hsikan/components/pool_scatter.py](../signedkan_wip/src/ac_hsikan/components/pool_scatter.py)
  - new `_hamilton_coeffs` LRU helper
  - new `_cr_coef_backward` (closed-form, packed scatter_add)
  - new `_cr_coef_backward_both` (kept as a helper, unused — tried, no win)
  - `fused_pool_scatter_reference` gained optional `rotor_M` kwarg (used by the
    coef-grad reference path; landed during the 2026-06-05 rotor work).
  - `_FusedPoolScatterTritonFn.forward` now saves Q_signed/K/V/pool_h/scatter_h(pre-rotor).
  - `_FusedPoolScatterTritonFn.backward` skips matmul + Triton fwd re-run;
    calls `_cr_coef_backward` instead of autograd-through-reference.
- [signedkan_wip/src/ac_hsikan/layer.py](../signedkan_wip/src/ac_hsikan/layer.py)
  - `_local_indices_cache` buffer pre-computed at `__init__` from `cfg.n_positions`.
    Fast path used when `L == cfg.n_positions` (the IMDB case).
- [signedkan_wip/experiments/ac_hsikan_imdb_smoke.py](../signedkan_wip/experiments/ac_hsikan_imdb_smoke.py)
  - new `--compile` flag (defaults off; included for benchmarking).

## Tests

- `signedkan_wip/tests/test_pool_scatter_rotor_parity.py` (3 tests) — all pass
  - `test_forward_parity_with_rotor` — Triton fwd vs PyTorch ref < 1e-4
  - `test_backward_parity_with_rotor` — 9-parameter gradient parity to numerical
    noise (max 1e-10 on W_q/W_k/W_v, 0 on coef_pos/neg/x/entropy_beta,
    4.5e-13 on entropy_axis)
  - `test_rotor_identity_when_beta_zero`
- `signedkan_wip/tests/test_evolvent_telemetry.py` (6 tests) — all pass
- `signedkan_wip/tests/test_ac_hsikan.py` (41 tests) — all pass
- End-to-end IMDB val_acc parity verified: 0.7452 (before) → 0.7455 (after CR-coef fix)
  → 0.7430 (after full stack). Within seed noise (σ ≈ 0.006-0.011).

**50 / 50 tests green throughout the session.**

## Open / next

- Block-reduce Triton kernel for CR-coef bwd — 1.5-2× more on C, ~10%
  end-to-end. Deferred (LOC cost vs gain).
- Full-IMDB 5-seed GPU run (25k/5k/8ep/L=200) is in flight as of this
  report's writing; estimated wall ~8-10 min total. Result will land in
  a follow-up.
- `torch.compile` at larger L / longer training might net-positive.
  Worth re-evaluating at the journal/full-IMDB stage.
- The Hamilton `_hamilton_rotate_static` PyTorch path has visible
  Python overhead in cProfile (~0.33 ms / call). Could be `torch.jit.script`'d
  or fused into the Triton forward (would also avoid the pre-rotor save
  trick). Deferred.

## CORE.YAML items touched

None. `signedkan_wip/` is non-core.
