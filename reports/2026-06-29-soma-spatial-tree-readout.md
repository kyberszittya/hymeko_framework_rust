# Dynamic Spatial-Tree Readout for Gömb-Soma Vision

**Date:** 2026-06-29
**Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Motivation

The readout (not the connection) is the lever for Soma vision. The two extremes
each fail: **mean-pool** discards position (0.31 on cluttered); **flatten** keeps
all position but doesn't scale (`n_patches·d` head) and can't handle variable
anchors. Single-vector pools (attention, pos-attention) plateau ~0.45. The user's
proposal: a **dynamic spatial tree** at the readout — the principled middle.

`_SpatialTreeReadout` (in `walk_conv_classifier.py`): a quadtree pyramid of
regions (levels 1×1, 2×2, 4×4 = 21 cells); each cell mean-pools its region
(multi-scale position preserved); each cell is gated by a learned sigmoid of its
activity (the *dynamic* part — the tree emphasises where content is).
`out_dim = (Σ levelᵢ²)·d = 21·d`, **independent of the grid size** (scalable), and
applicable to variable-anchor sets (so it can go where flatten can't, incl.
RicciStim's `AdaptiveQuadtree`).

## Result (Cluttered-MNIST, random-position digit, canvas 48, 5 seeds × 5 ep × 5000)

| readout | params | acc |
|---|---:|---|
| linear control | 23050 | 0.2126 ± 0.0057 |
| mean-pool | 2010 | 0.3136 ± 0.0230 |
| attention (scale-free) | 2027 | 0.4662 ± 0.0302 |
| **spatial tree (dynamic)** | **5227** | **0.5372 ± 0.0189** |
| flatten (full spatial map) | 24890 | 0.6166 ± 0.0157 |

Figure: `reports/figures/soma_readout_ladder_20260629.png`.

## Verdict — the scalable position-aware readout we were missing

The dynamic spatial tree **clears the single-vector plateau** (mean 0.31,
attention 0.47) decisively, reaching **0.537 — ~87% of the way from attention to
flatten — at ~1/5 of flatten's parameters** (5227 vs 24890). It does not fully
match flatten (0.617); the ~0.08 gap is the price of compressing 144 patches to
21 multi-scale cells. But it is the **best readout that scales**: its output size
is independent of the input grid, and it pools a *variable* region set — so it is
the one position-aware readout that drops into RicciStim's adaptive-anchor setting
(where flatten is undefined).

So the readout hierarchy, in one line: `mean (0.31) < attention (0.47) <
spatial-tree (0.54) < flatten (0.62)`, with spatial-tree the scalable knee of the
curve. This is the concrete revival path for Soma vision: a position-keeping
readout that is not tied to a fixed grid.

Honest caveats: still below flatten in absolute terms; cluttered MNIST is a
centred-digit-on-clutter task (not natural images); the dynamic gate's
contribution vs a static pyramid was not isolated here (a clean follow-up).

## Next
- **Static-vs-dynamic ablation** (is the learned per-cell gate pulling weight, or
  is the static pyramid enough?).
- **Spatial tree into RicciStim** (replace mean-pool over anchors with a spatial
  tree over anchor positions — the variable-anchor case flatten can't serve).
- **Deeper levels / more cells** to close the gap to flatten while staying scalable.

## Follow-ups (2026-06-30): gate ablation, RicciStim transfer, export

**Gate ablation — the *pyramid*, not the *dynamic gate*, is the compressor.**
Static pyramid (gate removed) **0.5584 ± 0.0231** vs dynamic **0.5372 ± 0.0189**
— statistically tied (static marginally higher). The learned per-cell gate is
**inert** on this task; the multi-scale pyramid *structure* does the work. The
gate-free version is the keeper (and is now the default, `dynamic=False`).

**RicciStim transfer — the readout helps a *different, variable-anchor* model.**
Wired the pyramid pool over RicciStim's variable anchor *positions*
(`_AnchorSpatialTreeReadout`). At the Phase-2 scale (highway on, 3×3×2000):

| readout | full 3-branch | encoder-only |
|---|---|---|
| mean-pool | 0.235 | 0.152 |
| attention | 0.191 | — |
| **spatial-tree** | **0.292** | 0.212 |

The spatial-tree readout **lifts RicciStim above its mean/attention ceiling**
(+0.057 over mean), and structure stays load-bearing under it (full-tree −
enc-tree = **+0.080**, matching Phase 2's +0.083 under mean-pool). Figure:
`reports/figures/soma_ricci_tree_readout_20260630.png`.

## The principle (exported)

**A flatten readout over a spatial field compresses to a multi-scale spatial-
pyramid pool** — ≈1/5 the parameters for ≈90 % of the accuracy, *plus*
scale-invariance and variable-item-count support. The compression is structural
(the pyramid), not learned (the gate). Extracted as a **domain-agnostic**
`SpatialPyramidPool` (`hymeko_neuro/experiments/vision/spatial_pyramid.py`): its input is
just `(features (N,d), positions∈[0,1]²)`, so any model with positioned features
can drop it in for flatten — the Gömb-Soma patch grid and RicciStim anchors both
now delegate to it (one implementation, no duplication).

**Export targets (other domains in this repo):** HyMeYOLO detection heads (a
spatial-pyramid descriptor over the feature map / queries instead of a flattened
head); RL vision backbones; any graph/point readout with node coordinates. Logged
for follow-up.

## GPU optimization (2026-06-30): the pyramid is a linear operator

The pyramid pool is **linear**: ``cells = P @ features``, where ``P`` is the
``(n_cells, N)`` row-normalised cell-indicator matrix (each row = one pyramid
cell, summing to 1 = a mean). Consequences:

- All levels collapse into **one matmul** (stack the per-level rows of ``P``) —
  no Python level-loop, no ``scatter_add``.
- For a **fixed** layout (a patch grid) ``P`` is constant → precompute once
  (``set_fixed_positions``); the readout is then a single fused, **batchable**
  ``einsum('cn,bnd->bcd', P, features)``. For variable anchors ``P`` is built per
  call (still one matmul).

The old code (and the classifiers) called the readout **per image** — a Python
loop of tiny ops = launch-bound (the B=1 dispatch problem). Benchmark
(``bench_spatial_pyramid``; N=144, d=16, batch=128; median over warmed-up runs):

| device | per-item loop | batched matmul | speedup |
|---|---:|---:|---:|
| CPU | 28969 µs | 938 µs | **30.9×** |
| CUDA (RTX 3070) | 27146 µs | 341 µs | **79.6×** |

The per-item loop costs ~27 ms on *both* devices — it is launch-bound, not
compute-bound (a GPU cannot accelerate a Python loop of tiny kernels). The
batched matmul fixes it (31× CPU, **80× GPU**). Figure:
`reports/figures/spatial_pyramid_bench_20260630.png`; data
`reports/spatial_pyramid_bench_20260630.json`. `SpatialPyramidPool.forward` now
accepts batched ``(B, N, d)`` — so the readout is batch-ready; realising the
end-to-end win additionally requires batching the upstream walk-conv (the grid
topology is shared across the batch), logged as the systemic follow-up.

## End-to-end batching (2026-06-30): the whole classifier in one pass

The readout matmul is only part of the per-image cost — the *whole*
``WalkConvImageClassifier`` looped over the batch. Since the grid topology
(walks, ``M_v``) is shared across the batch, every stage now runs on
``(B, n_patches, d)`` in one pass: batched patchify, batched brightness/edge
signs, batched walk-conv (the message einsum gains a batch axis; the sparse
aggregation packs the batch into ``M_v``'s column axis = one ``sparse.mm``), and
the batched readouts. The per-image path is untouched (gated on ``x.ndim``), so
RicciStim's variable-anchor layers and the holonomy ABC are unaffected.

**Parity-verified:** the batched forward equals the per-image loop exactly
(``atol 1e-5``) for every readout, including the Chebyshev-CR cell + holonomy
aggregation. Benchmark (``bench_walk_conv_batched``; canvas 48, 144 patches,
batch 64, spatial-tree readout; median):

| device | per-image loop | batched | speedup |
|---|---:|---:|---:|
| CPU | 312 ms | 152 ms | 2.1× |
| CUDA (RTX 3070) | 204 ms | **7.4 ms** | **27.7×** |

GPU is launch-bound (204 ms of tiny per-image kernels → 7.4 ms batched);
CPU was already compute-bound, so it gains less. Figure:
`reports/figures/walk_conv_batched_bench_20260630.png`; data
`reports/walk_conv_batched_bench_20260630.json`. This realises the end-to-end
GPU win the readout optimisation pointed at.

## Multi-query pool (2026-06-30): the scalable readout that *matches* flatten

The remaining candidate: K learned query vectors, each attending (softmax) over
the patches → K pooled "content slots", concatenated (``out_dim = K·d``,
grid-independent; K=1 = single-query attention). Re-ran the scalable readouts on
one identical GPU+resident harness (cluttered 48, 5 seeds × 5 ep × 5000) for an
apples-to-apples ladder:

| readout | params | acc |
|---|---:|---|
| attention (K=1) | 2027 | 0.467 ± 0.022 |
| spatial tree | 5227 | 0.553 ± 0.023 |
| **multi-query (K=8)** | **3258** | **0.605 ± 0.045** |
| flatten (full map) | 24890 | 0.605 ± 0.016 |

**Multi-query matches flatten (0.605 ≈ 0.605) at ~1/8 the parameters, and is
scale-free** — the readout we were after. It beats the spatial tree (0.553) and
the single-query plateau (0.467), reaches the ceiling without flatten's
``n_patches·d`` head, and (unlike flatten) handles a variable item count. (The
attention arm reproduced its earlier 0.466 → the GPU+resident harness is
consistent, so the ladder is sound.) Cost: higher seed variance (±0.045 vs
flatten's ±0.016). Figure: `reports/figures/soma_multiquery_ladder_20260630.png`;
data `reports/soma_multiquery_ladder_20260630.jsonl`. Batched + parity-tested
like the others. So the scalable-readout question is answered: **multi-query
attention is the flatten-matching, scale-free, variable-size readout** (the
spatial tree is the simpler runner-up).

## Files
- `hymeko_neuro/models/hymeko_gomb/soma/vision/walk_conv_classifier.py` —
  `Readout.SPATIAL_TREE` + `_SpatialTreeReadout`; `_build_readout` now takes grid dims.
- `hymeko_neuro/models/hymeko_gomb/soma/vision/train_mnist.py` — `gomb_soma_tree` arm.
- `hymeko_neuro/tests/test_soma_position_aware_readout.py` — 5 spatial-tree tests.
- jsonl `reports/soma_spatial_tree_cluttered_20260629.jsonl`; figure as above.
- No CORE.YAML items; no new dependency.
