---
name: project-soma-vision-readout-bound
description: Gömb-Soma walk-vision falsification was substantially a READOUT artifact; position-preserving walk-conv BEATS linear on MNIST (2026-06-29)
metadata: 
  node_type: memory
  type: project
  originSessionId: 3b1d3a94-913b-4c0f-9bc1-e8c767463815
---

2026-06-29: tested the user's "use Chebyshev-CR + Chebyshev-CR patches" idea as a
2×2 factorial — cell {GELU, Chebyshev-CR} × readout {mean-pool, flatten} — on the
exact MNIST harness. The Soma-vision pipeline had never used the framework's
HSiKAN cell (bare `Linear` patch embed, fixed `GELU` message, **global
mean-pool** readout).

**Result — pooling-bound (H2), decisively.** base/mean **0.519**, cheby/mean
**0.553**, GELU/flatten **0.945**, cheby/flatten **0.948** (5-seed). Readout main
effect **+0.410**, cell main effect **+0.018** (~23×). The 0.52 ceiling was the
mean-pool discarding *which patch held what*, NOT activation expressivity.

**Key reframe:** **position-preserving Soma (0.945–0.948) BEATS the linear
control (0.906)** at comparable capacity — the signed walk-conv *is* net-useful
once the readout stops throwing away spatial layout. So "walks-only falsified for
vision" (2026-06-15, [[project-backlog-done-tracking]]) holds **only for the
mean-pool readout**; it was substantially a readout artifact. This is the
*opposite* outcome to the holonomy/sign-handling re-test
([[project-soma-holonomy-vision-falsified]]), which stayed negative.

**Phase 1+1.5 update (2026-06-29, Cluttered-MNIST random-position):** linear
0.213 < mean-pool 0.314 < attention 0.466 ≈ pos-attention 0.446 < **flatten
0.617**. The scale-free single-vector pools (content-attention, and attention +
learned positional embedding) **plateau ~0.45** — the bottleneck is single-vector
*compression*, not absence of position (adding position did NOT help). Flatten's
full `n_patches·d` spatial map wins on bounded grids. **Scalable readout that
might match flatten = multi-query attention (K·d, K≪n_patches)** — untried,
logged. **RicciStim caveat:** its adaptive quadtree has a *variable* anchor count
→ flatten cannot wire in; a set-pool (attention/mean) is forced there. So the
"wire flatten into RicciStim" plan step is blocked by RicciStim's variable-anchor
design — Phase 2 must use attention pool (which plateaus) or rethink.

**Multi-query pool = flatten-matching scalable readout (2026-06-30):** K learned
query vectors softmax-attend over patches → K·d slots (grid-independent). Same
GPU harness, 5-seed cluttered: attention(K=1) 0.467 < spatial-tree 0.553 <
**multi-query(K=8) 0.605 ≈ flatten 0.605** at ~1/8 flatten's params + scale-free
+ variable-size. So the scalable-readout question is ANSWERED: multi-query
attention matches the flatten ceiling without its huge head (spatial tree is the
simpler runner-up). Cost: higher seed variance. `Readout.MULTI_QUERY`.

**Spatial-pyramid EXPORTED (2026-06-30):** (a) gate ablation — static pyramid
0.558 ≈ dynamic 0.537 → the learned per-cell GATE is INERT; the multi-scale
PYRAMID is the compressor (gate-free default). (b) RicciStim transfer — pyramid
over variable anchor positions lifts it: full-tree 0.292 > mean 0.235 > attn
0.191; structure load-bearing (full-tree−enc-tree=+0.080). (c) PRINCIPLE: flatten
over a spatial field compresses to a multi-scale spatial-pyramid pool (~1/5
params, ~90% acc, +scale-invariance +variable-N). Extracted domain-agnostic
`SpatialPyramidPool` in `signedkan_wip/src/vision/spatial_pyramid.py` (input
`(features, positions∈[0,1]²)`); soma-grid + RicciStim-anchor readouts both
delegate (no dup). Export targets logged: HyMeYOLO heads, RL vision, graph/point
readouts. User's "export to other domains" — done + evidenced.

**Spatial-tree readout (2026-06-29, the win):** the scalable position-aware
readout. `_SpatialTreeReadout` (quadtree pyramid 1+4+16=21 cells, each mean-pools
its region, learned per-cell gate = dynamic; out_dim=21·d, grid-independent).
Cluttered ladder: mean 0.314 < attention 0.466 < **spatial-tree 0.537 (5227p)** <
flatten 0.617 (24890p). Clears the single-vector plateau, ~87% of the way to
flatten at ~1/5 params, AND scales / handles variable anchors (flatten can't).
This is the concrete revival readout — drops into RicciStim's AdaptiveQuadtree
anchors. Open: static-vs-dynamic-gate ablation; tree-into-RicciStim. The user's
"dynamic spatial tree at the readout" idea landed.

**Phase 2 update (2026-06-29, RicciStim, highway ON):** the structural branches
ARE load-bearing. Corrected 2×2 (highway carries encoder features so encoder-only
is a real control, not zero-input): full-mean **0.235** vs enc-mean **0.152**
(+0.083, ~3σ), full-attn 0.191 vs enc-attn 0.141 (chance 0.10). So walk/polygon/
triangle add real value over the encoder. Attention readout still ≤ mean (no
lift — same single-vector plateau). Absolute ~0.235 ≈ linear, < WalkConv+flatten
0.62 (reduced/undertrained 3ep×2000). **Lesson the user flagged:** the highway
skip is the whole point of the encoder-only control — without it the ablation
zeroes the head input and the test is meaningless. Open: multi-query pool;
non-centred + full-scale re-test.

**Why / How to apply:** the user's Chebyshev-CR intuition
([[feedback-user-intuition-is-calibrated]]) was half-right — the cell adds a
little, but it *surfaced the architectural lever* (readout) that actually
mattered. Don't repeat "hypergraph-vision is just falsified" without this caveat.
The live revival path is a **scalable position-aware pool** (attention /
per-position weights — raw `flatten`'s `n_patches·d_hidden` head doesn't scale),
re-tested on Cluttered-MNIST (RicciStim's 0.14) + non-centred images, and wired
into `ricci_stim_backbone.py` (it also pools). Reusable knobs now in code:
`MessageActivation.{CR,CHEBY_CR}`, `PatchEncoder.CHEBY_CR`, `Readout.FLATTEN` in
`signedkan_wip/.../soma/`. Report: `reports/2026-06-29-soma-cheby-cell-readout-sweep.md`.
