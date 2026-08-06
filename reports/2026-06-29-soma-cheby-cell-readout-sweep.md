# Chebyshev-CR Cell × Readout Sweep for Gömb-Soma Vision

**Date:** 2026-06-29
**Plan:** [docs/plans/2026-06-29-soma-cheby-cell-readout-sweep/](../docs/plans/2026-06-29-soma-cheby-cell-readout-sweep/) (tex/pdf/tikz/mmd)
**Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Summary

After the holonomy re-test confirmed the Gömb-Soma walk-vision falsification on
the *sign-handling* axis, the question moved to **expressivity vs architecture**:
the Soma-vision pipeline never used the framework's HSiKAN cell (bare `Linear`
patch embed, fixed `GELU` message, **global mean-pool** readout). Two hypotheses
for the 0.52 ceiling:

- **H1 — expressivity-bound:** the weak cell is the limit; the learnable
  Chebyshev-CR cell (CR patches + CR messages — the user's proposal) lifts it.
- **H2 — pooling-bound:** the global mean-pool discards *which* patch held
  *what*; no activation restores discarded position.

A 2×2 factorial — cell (baseline vs Chebyshev-CR) × readout (mean-pool vs
position-preserving flatten) — on the exact MNIST harness discriminates them.

## Result (MNIST 5000/1000, 5 epochs, Adam 3e-3, CPU)

| arm | cell | readout | params | test acc (5-seed mean ± pstd) |
|---|---|---|---:|---|
| Linear control | — | — | 7850 | 0.9056 ± 0.0079 |
| base-Soma (`gomb_soma`) | GELU | mean-pool | 2010 | 0.5186 ± 0.0204 |
| `gomb_soma_cheby` | Chebyshev-CR | mean-pool | 2170 | 0.5530 ± 0.0433 |
| `gomb_soma_flat` | GELU | flatten | 9690 | **0.9450 ± 0.0072** |
| `gomb_soma_cheby_flat` | Chebyshev-CR | flatten | 9850 | **0.9476 ± 0.0076** |

Figure: `reports/figures/soma_cheby_cell_readout_20260629.png`.

### Factorial decomposition (the discriminator)

| main effect | size |
|---|---|
| **readout** (mean-pool → flatten), averaged over cell | **+0.410** |
| **cell** (GELU → Chebyshev-CR), averaged over readout | +0.018 |
| interaction | small: cell adds +0.034 at mean-pool, +0.003 at flatten |

**Verdict — H2 (pooling-bound), decisively.** The readout main effect is
**~23× the cell effect**. The 0.52 ceiling was the global mean-pool discarding
*which patch held what*, not the activation's expressivity. The Chebyshev-CR
cell on its own (mean-pool) gives 0.553 — inside noise of base-Soma's 0.519.

**Two consequences that change the Soma-vision story:**

1. **The 2026-06-15 falsification was substantially a readout artifact.** The
   walk-conv was never given a fair readout; fix the pooling and the gap to
   linear closes entirely.
2. **Position-preserving Soma *beats* the linear control** (0.945 / 0.948 vs
   0.9056) at comparable capacity. The signed walk-conv structure *is* doing
   useful work once the readout stops throwing it away. This rehabilitates
   Gömb-Soma for vision on the *readout* axis — the opposite outcome to the
   holonomy (sign-handling) re-test, which stayed negative.

Caveat (measured / inferred / open): *measured* — the five 5-seed means;
*inferred* — position is the binding constraint and the structure is net-useful
once kept (from the factorial); *still open* — whether the flatten-readout lift
holds at scale / on cluttered or natural images (MNIST is small and centred;
flatten's `n_patches·d_hidden` head does not scale to large grids — a learned
position-aware pool, not raw flatten, is the scalable form). The user's
Chebyshev-CR intuition was half-right: the cell adds a little, but it surfaced
the architectural lever that actually mattered.

## Files touched

| file | change |
|---|---|
| `hymeko_neuro/models/hymeko_gomb/soma/hg_conv.py` | `MessageActivation` enum + `message_activation` config field |
| `hymeko_neuro/models/hymeko_gomb/soma/walk_layer.py` | build activation module at init (Strategy), `_build_message_activation` |
| `hymeko_neuro/models/hymeko_gomb/soma/vision/walk_conv_classifier.py` | `PatchEncoder`/`Readout` enums + Strategy modules (`_Mean/_Flatten` readout, CR patch encoder), head width from readout |
| `hymeko_neuro/models/hymeko_gomb/soma/vision/train_mnist.py` | 3 new arms (cheby / flat / cheby_flat) |
| `hymeko_neuro/experiments/runs/soma_holonomy_ab_plot.py` | extend ARM_ORDER/LABEL/COLOR (reuse shaper+renderer) |
| `hymeko_neuro/tests/test_gomb_soma_cheby_cell_readout.py` | new: 10 cell/readout tests |
| `hymeko_neuro/tests/test_soma_holonomy_ab_plot.py` | 2 assertions relaxed to present-arm subset |

## CORE.YAML items touched

None. All edits in `hymeko_neuro/`. `ChebyshevCRActivation` reused from
`hymeko_neuro/core/splines.py` — no new spline code, no new dependency.

## Test results

- **New cell/readout tests:** 10 passed — activation builders; default stays
  GELU (regression); Chebyshev-CR walk forward finite/shaped; flatten preserves
  position while mean-pool is invariant (operationalises H2); readout out-dims +
  head width; all four arms forward; cheby adds params; cheby cell trainable.
- **Regression:** `test_gomb_soma_hg_conv`, `test_gomb_soma_walk_layer`,
  `test_gomb_soma_vision_walk_conv_classifier`, `test_gomb_soma_holonomy_aggregation`,
  `test_soma_holonomy_ab_plot` — all pass.
- **Static analysis:** `ruff check` clean on all changed files.

## Performance

- Smoke: cheby ~68 s/epoch, cheby_flat ~62 s/epoch (vs GELU baseline ~28 s/epoch —
  the ChebyshevCR per-element spline gather, expected, not a regression). Full
  sweep ≈ 66 min (3 new arms × 5 seeds × 5 ep). Peak RSS < 1 GB (largest arm
  9850 params on a 5k MNIST subset) — far under the 16 GB cap.

## §6.5 anti-patterns

None. Cell/readout/patch-encoder are config enums + Strategy modules built at
construction — structural variants are classes, not forward-time flags (§6.5 #8);
plot module reused, not duplicated (§6.1).

## Graphical output (§9)

Numerical (table + JSONL), plotted (bar chart). No GIF: static classification A/B.

## Experiment provenance

- Git SHA: 9aea4f6 (working tree dirty — this feature diff).
- JSONL: `reports/soma_cheby_cell_readout_mnist_20260629.jsonl` (3 new arms × 5
  seeds). Linear + base-Soma anchors reuse
  `reports/soma_holonomy_vs_routing_mnist_20260629.jsonl` (same machine/torch,
  reproduced exactly earlier this session).
- Seeds 0–4; Adam lr 3e-3, batch 64, 5 epochs; MNIST 5000/1000 sub-sampled by seed.
- Env: torch 2.12.0 (cu132), CPU, Windows 11, `.venv`.

## Open issues / follow-up

- **Scalable position-aware readout.** Raw `flatten` works on MNIST (49 patches)
  but its `n_patches·d_hidden` head does not scale to large/variable grids.
  Replace with a learned position-aware pool (attention pool or per-position
  weights) — the proper form of "keep position." This is the natural next step
  if the Soma-vision line is revived.
- **Re-test on cluttered / non-centred images.** The MNIST lift may partly ride
  on digit centring; the honest test is Cluttered-MNIST (where RicciStim scored
  0.14) and a non-centred set, with the position-aware readout.
- **Wire the readout fix into RicciStim.** The 3-branch backbone
  (`ricci_stim_backbone.py`) also pools; the same readout axis likely applies.
- The Chebyshev-CR cell stays available (`MessageActivation.CHEBY_CR`,
  `PatchEncoder.CHEBY_CR`) but is not the lever — keep it, don't lead with it.
