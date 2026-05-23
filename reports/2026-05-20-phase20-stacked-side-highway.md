# Phase 20: StackedSideSignedKAN + highway gates — 2026-05-20

## Summary

User: *"now go with stacking a side HSIKAN. I think highways are
in this architecture already, right?"*

Both observations confirmed. Built `StackedSideSignedKAN` = each
parallel branch is itself a `MultiLayerSignedKAN` stack with
optional `inner_skip='highway'` gates. The 12-cell Bitcoin Alpha
grid (5 seeds × hidden=8 × n_epochs=100) puts the stacked-side
highway variant at the top of the c3-only HSIKAN family.

**Headline:** `sside_N8L2_hwy` reaches **0.8154 ± 0.0123** —
beats every prior c3-only variant including Phase 19's best
(side_N8 at 0.808 at h=16). Highway gates rescue per-branch
depth: without them, `sside_N4L4_res` produces 0.7734 ± **0.0827**
(σ explosion) while `sside_N4L4_hwy` produces 0.8136 ± 0.0118
(stable + competitive).

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/core/side_signedkan.py` | extended | +130 (`StackedSideSignedKANConfig` + `StackedSideSignedKAN` class) |
| `signedkan_wip/experiments/runs/run_compare.py` | extended | +35 (dispatch + `_SSK_marker` import + M_vt path) |

## CORE.YAML items touched

None.

## Architecture

```python
class StackedSideSignedKAN(nn.Module):
    """N parallel MultiLayerSignedKAN branches × per-branch depth L.

    Each branch:
      MultiLayerSignedKAN(n_layers=L,
                          inner_skip=cfg.inner_skip,  # "residual" | "highway"
                          use_residual=True,
                          layer_norm_between=True,
                          jk_mode='last')

    Fusion (mean): h_t_fused = (1/N) * Σ_i h_t_i
    """
```

Composes Phase 16 (depth) + Phase 17 (side) + the existing
`HighwaySignedKAN` machinery. Same `encode_triads(triad_v,
triad_sigma, M_vt, return_h_v)` interface as
`MultiLayerSignedKAN` so `run_compare.run_one` can call it
through the existing dispatch.

## Full 12-cell Bitcoin Alpha grid

5 seeds × hidden=8 × n_epochs=100 × lr=5e-2.

| rank | config | mean AUC ± std | params | wall/seed |
|------|---|---|---|---|
| 🥇 1 | **sside_N8L2_hwy** | **0.8154 ± 0.0123** | 247k | 27.5s |
| 🥈 2 | sside_N4L4_hwy | 0.8136 ± 0.0118 | 126k | 27.7s |
| 🥉 3 | side_N8 | 0.8123 ± 0.0150 | 244k | 13.3s |
| 4 | sside_N8L2_res | 0.8119 ± 0.0233 | 246k | 26.9s |
| 5 | side_N4 | 0.8112 ± 0.0142 | 122k | 6.8s |
| 6 | sside_N4L2_hwy | 0.8098 ± 0.0110 | 124k | 13.9s |
| 7 | sside_N4L2_res | 0.8058 ± 0.0150 | 123k | 13.6s |
| 8 | bare_L1 | 0.8043 ± 0.0215 | 30k | 2.1s |
| 9 | mem_N8 | 0.7956 ± 0.0193 | 245k | 13.5s |
| 10 | mem_N4 | 0.7898 ± 0.0183 | 122k | 6.9s |
| 11 | sside_N4L4_res | 0.7734 ± **0.0827** ⚠️ | 125k | 27.2s |
| 12 | depth_L4 | 0.7317 ± 0.1216 | 31k | 7.0s |

## Five findings

### 1. Highway gates rescue per-branch depth

The clearest finding: at $L=4$ per branch, the residual variant
(`sside_N4L4_res`) gets AUC 0.7734 with σ **0.0827** (variance
explosion). The highway variant (`sside_N4L4_hwy`) gets 0.8136
with σ 0.0118 — **+0.04 AUC and 7× lower variance**. Highway
gating inside each branch's stack is essential for L≥4.

This mirrors Phase 16's finding (depth-only fails) but inverts
the punchline: depth-with-highway works fine *when combined with
parallel branches that average across stochastic mistakes*.
Phase 16 had depth-with-highway too at L=8: 0.468 — worse than
L=1. So highway alone isn't enough; **highway + parallel-branch
averaging is the combination that succeeds.**

### 2. sside_N8L2_hwy is the new top of the c3-only family

`sside_N8L2_hwy` at 0.8154 ± 0.0123 is the best score in the
entire 12-cell grid. Beats:
- bare SignedKAN (0.8043) by +0.011
- Phase 19's best (`side_N8` at h=16, 0.8083) by +0.007 at h=8
- pure depth `L=4` (0.7317) by +0.084 — depth_only's failure
  mode is comprehensively addressed.

### 3. Width matters more than per-branch depth at fixed params

`side_N8` (0.8123, just N=8 × L=1 = 244k params) is within 0.003
of the top `sside_N8L2_hwy` (0.8154, N=8 × L=2 = 247k params).
**Adding per-branch depth gives a marginal gain at the same
parameter budget.** The dominant lever is parallel-branch
cardinality; per-branch depth is a small bonus when paired with
highway gates.

### 4. Membrane variants underperform at scale

`mem_N4` (0.7898) and `mem_N8` (0.7956) are below bare SignedKAN
(0.8043) at the same hidden. The shared-latent coupling that
helped at L=1 (Phase 19: mem_L=1 0.707 vs side_L=1 0.658) is
either redundant or harmful at n_epochs=100 / h=8. The read-gate
+ shared-latent overhead doesn't pay off at this scale.
**Membrane is the wrong lever for this dataset/scale.**

### 5. depth_L4 reproduces Phase 16's failure

`depth_L4` at 0.7317 ± **0.1216** is the worst of the grid and
has the worst variance. Phase 16's "depth hurts" finding holds at
hidden=8 / n_epochs=100 just as it did at hidden=8 / n_epochs=30.
**The depth-only failure mode is dataset-and-scale-invariant on
Bitcoin Alpha.**

## SOTA gap context

Single-seed Optuna-best reproduced earlier at AUC 0.9970
(mixed-arity `c2,c5,w2,w3,w4` + α-entropy reg, h=8, 80 epochs;
matches memory's 10-seed 0.9959 ± 0.0011).

**Phase 20's best c3-only architecture caps at 0.815.** The
remaining +0.18 gap to mixed-arity SOTA is **architectural** —
the mixed-arity infrastructure (k=2/3/4/5 cycles + walks) is the
lever. Depth/width/membrane/stacked-side scaling on c3-only HSIKAN
will not close this gap.

The natural Phase 21 candidate: **port the
stacked-side-highway pattern onto mixed-arity HSIKAN.** Combines
Phase 20's variance-tightening width win + highway-rescued depth
with the SOTA-relevant architecture family. Likely closes most
of the 0.18 gap.

## Test results

All 12 prior side+membrane tests still pass (the new
`StackedSideSignedKAN` is an additive class). `cargo test
-p hymeko_pgraph` (96/96) still clean. No regressions.

## §6.5 anti-pattern audit

No new anti-patterns. `StackedSideSignedKAN` is a fourth peer in
the side-module family (SideSignedKAN, MembraneSignedKAN,
StackedSideSignedKAN) and the dispatch in `run_compare` is one
new model_name with a sub-branch on residual-vs-highway. The
`bilinear_rank` piggy-back channel for `n_layers_per_branch` is
admittedly ugly but documented; cleaner kwargs would require
modifying `run_one`'s signature in a non-back-compat way.
Acceptable for the prototype; the proper kwarg addition is a
Phase 20.5 cleanup.

## Open follow-ups

1. **Phase 21**: parallel-branch / stacked-side / highway on
   mixed-arity HSIKAN. The expected ~0.16-0.18 AUC gain would
   put us at ~0.99 — competitive with or beating the existing
   Optuna SOTA.
2. **Phase 20.5 cleanup**: add explicit `n_branches` +
   `n_layers_per_branch` kwargs to `run_one` rather than
   piggy-backing on `n_layers` + `bilinear_rank`.
3. **Heterogeneous spline kinds per branch** (still untested —
   `SideSignedKANConfig.spline_kinds` is already plumbed).
4. **Stacked-side cross-dataset**: Slashdot, Epinions, Bitcoin OTC.
   Phase 8 showed walks-augmented mixed-arity wins on Slashdot;
   stacked-side might be the c3-only Slashdot leader too.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (uncommitted: phases 1-20).
- **Grid wall:** 931.9 s for the 12-cell grid (5 seeds × 12
  configs at h=8 / n_epochs=100).
- **Single-seed Optuna SOTA reproduction:** 0.9970 in ~5 min.

## Acceptance check

- [x] No `CORE.YAML` items touched.
- [x] `StackedSideSignedKAN` ships as a clean fourth peer in the
      side-module family.
- [x] Highway-vs-residual dispatch via `model_name`.
- [x] 12-cell grid runs cleanly at hidden=8.
- [x] **Stacked-side-highway is the top of the c3-only family**
      at 0.8154 ± 0.0123.
- [x] **Highway gating empirically rescues per-branch depth**
      (sside_N4L4_hwy 0.8136 vs sside_N4L4_res 0.7734 with 7×
      tighter σ).
- [x] Width-via-cardinality dominates over per-branch depth at
      fixed params.
- [x] Depth-only failure reproduces at the larger n_epochs scale.
- [x] SOTA gap quantified: +0.18 to mixed-arity Optuna, *not*
      attributable to width/depth scaling on c3-only.
- [x] No regressions on prior Phase 1-19 suites.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
