# AC-HSiKAN v1.6 IMDB smoke — pool-scatter, entropy rotor, evolvens telemetry

**Date:** 2026-06-05
**Scope:** Three-way GPU smoke on real IMDB sub-sample comparing
(a) walk-op-only AC-HSiKAN, (b) v1.6 dual-path pool-scatter without rotor,
(c) v1.6 dual-path + entropy Hamilton rotor — all vs. iso-parameter
Transformer baseline. Evolvens telemetry attached to (c).

## Headline

| config | val_acc (2-seed) | params | wall (2 seeds) | Δ vs Transformer |
|---|---:|---:|---:|---:|
| Transformer baseline      | **0.7772 ± 0.0088** | 166 594 |   4.8 s |   0     |
| AC walk-op only           |   0.6980 ± 0.0071   | 163 372 |   8.7 s | −0.0792 |
| AC v1.6 pool-scatter      |   0.7438 ± 0.0067   | 164 654 | 161.8 s | −0.0335 |
| AC v1.6 + entropy rotor   |   0.7452 ± 0.0060   | 164 668 | 165.4 s | −0.0320 |

Two seeds, IMDB 5 000 train / 2 000 val, L_max 128, 4 epochs, batch 64,
lr 3e-3, AdamW(weight_decay=1e-4), CUDA (RTX 2070 SUPER), Triton-fused
pool-scatter on the v1.6 paths.

## What moved, what didn't

- **Pool-scatter is the lift.** Walk-op → +pool-scatter buys **+0.0458**
  validation accuracy (+5σ paired). The synaptic primitive carries the
  bulk of the gap-closure on this task.
- **Rotor is in the noise on accuracy.** +pool-scatter alone vs +rotor
  is **+0.0014** (≈ 0.2σ). The rotor does fire on every backward step
  (632 records / seed, see telemetry), but the validation-accuracy
  benefit at this scale is below seed noise.
- **AC still loses by Δ = −0.032** at smoke scale. Closer than the
  2026-06-04 CPU smoke (−0.151 → −0.032, **4.7× tighter**), close
  enough that longer training or larger scale might flip it, but
  there is no clean win yet.
- **Wall is the cost.** Pool-scatter ON is **20× slower** than walk-op
  only at L=128 / batch=64. Triton kernel launch overhead at small
  shapes dominates; the kernel's per-FLOP advantage shows up at
  larger L (the deferred plan target is L ≥ 1024).

## Evolvens telemetry (rotor variant, 632 backward calls × 2 seeds)

Per-step scalar metrics streamed to JSONL:

| metric | seed 0 mean | seed 1 mean | reading |
|---|---:|---:|---|
| `rotor_angle_rad`         | 0.36 (21°) | 0.26 (15°) | β·H drifts as α / gate distributions update |
| `cos_scatter_pre_post`    | 0.93       | 0.96       | matches cos(angle) — forward rotor preserves norm + twists exactly |
| `scatter_norm` vs `_mod`  | 56.4 ≡ 56.4 | 55.7 ≡ 55.7 | Hamilton rotation is norm-preserving ✓ |
| `grad_x_norm`/`grad_out`  | 3.5×       | 2.7×       | layer amplifies the upstream gradient — pool-scatter contributes on top of residual |
| `cos_grad_x_grad_out`     | 0.88 ± .11 | 0.91 ± .11 | gradient direction **substantially twisted** by layer (was 1.0 in synthetic 30-step demo) |

Two readings worth noting:

1. The rotor's input-angle (`rotor_angle_rad`) and feature-space twist
   (`cos_scatter_pre_post`) agree numerically — the closed-form
   `M = (cos θ, sin θ · n)` construction does on real workloads exactly
   what it does on synthetic toys.

2. The gradient/error coupling (`cos_grad_x_grad_out`) is flat at 1.0
   in early training and on synthetic data, but during real IMDB
   training it drops to ~0.88 with σ ≈ 0.11 — meaning the dual-path
   layer is now genuinely twisting the gradient direction (the residual
   short-circuit no longer dominates). The "evolvens" behaviour the
   plan describes is observable on a trained model.

The rotor moves *measurably* on every step, but the val_acc impact at
this 5 000-sample / 4-epoch budget is +0.0014, within noise. The β=0.1
initialisation likely under-uses the rotor's range; a learnable β that
grew past 0.5 could change this. Untested.

## Comparison to prior numbers

- **2026-06-04 CPU smoke** (sub-sampled 3 000 train, AC pre-pool-scatter):
  Δ = −0.151, Transformer wins clean.
- **2026-06-05 GPU smoke** (this report, AC v1.6 + rotor): Δ = −0.032.
  Gap **closed 4.7×** by the Triton-fused dual-path + rotor stack.
- **2026-05-18 full IMDB** (pre-AC HSiKAN, iso-param 321 k):
  HSiKAN edged Transformer by z ≈ +1.12. Different architecture
  family, full IMDB scale.

## Why this is a smoke, not a verdict

- n = 2 seeds (noise σ ≈ 0.006 on AC, 0.009 on Transformer).
- 5 000 train / 2 000 val — 20 % of full IMDB.
- 4 epochs (Transformer is already over-fitting by ep 4 on seed 0:
  0.783 → 0.772; AC is still climbing: 0.524 → 0.741).
- Triton kernel overhead dominates the small-L wall budget; the plan's
  wall claim is for L ≥ 1024 where the sequential cost of attention
  attention amortises differently. L = 128 here.

The 5-seed full-IMDB GPU run that would actually decide this still
needs scheduling. Within the existing smoke this run shows AC v1.6
**within striking distance** of Transformer, with the pool-scatter
primitive responsible for most of the closure and the rotor doing
visible-but-tiny extra work.

## Files

- [signedkan_wip/src/ac_hsikan/config.py](../signedkan_wip/src/ac_hsikan/config.py)
  — added `use_pool_scatter_rotor: bool = False`.
- [signedkan_wip/src/ac_hsikan/layer.py](../signedkan_wip/src/ac_hsikan/layer.py)
  — compute per-layer H from (α, gate), pass to `FusedPoolScatter`.
- [signedkan_wip/experiments/ac_hsikan_imdb_smoke.py](../signedkan_wip/experiments/ac_hsikan_imdb_smoke.py)
  — new `--pool-scatter`, `--rotor`, `--telemetry-out` flags.
- [signedkan_wip/src/ac_hsikan/telemetry.py](../signedkan_wip/src/ac_hsikan/telemetry.py)
  — `EvolventTelemetry` context manager (see 2026-06-05 telemetry note).

## Result artefacts (transient, on /tmp)

- `/tmp/imdb_smoke_walkop.json` — walk-op-only run
- `/tmp/imdb_smoke_v16_norotor.json` — pool-scatter w/o rotor
- `/tmp/imdb_smoke_v16_rotor.json` — pool-scatter + rotor (+ telemetry summary)
- `/tmp/imdb_v16_rotor_evolvens.seed{0,1}.jsonl` — per-step evolvens traces

## CORE.YAML items touched

None. `signedkan_wip/` is non-core.

## Open / next

- Decide whether β init / learnable scale needs adjustment (rotor angle
  stayed at 15-21° mean — under-used range).
- Schedule a 5-seed full-IMDB GPU comparison (USER gate) — the smoke
  gap of −0.032 ± 0.008 paired isn't a verdict.
- Investigate whether Triton kernel wall scales sub-linearly to L,
  where the plan claims AC's advantage should appear (L = 1024+).
- The +0.046 pool-scatter lift over walk-op is the **largest single
  lever** we have on this task; worth replicating at L = 256 and 512
  before scheduling the full run.
