# GömbSoma-Ricci-Stim Phase 2 — AdaptiveQuadtree

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 2 of 8
**Prior phase:** [Phase 1 — FormanCurvatureHead](2026-05-14-gomb-soma-ricci-stim-phase1.md)

## 1. Summary

Built `AdaptiveQuadtree`, the content-driven anchor-selection
mechanism that replaces YOLO's uniform-grid rolling-shutter tiling.
The quadtree recursively subdivides patches whose subdivision score
exceeds a threshold; the score combines pixel-variance (content
complexity) and Forman-Ricci curvature magnitude (structural
bottleneck) via configurable weights. Output is a deterministic
multi-scale `AnchorTree` with positions, sizes, scales, and
parent-child links.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/quadtree.py](../signedkan_wip/src/hymeko_gomb/soma/vision/quadtree.py) | 198 | `AdaptiveQuadtree` + `AnchorTree` dataclass |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +7 / -0 | re-exports |
| [signedkan_wip/tests/test_gomb_soma_vision_quadtree.py](../signedkan_wip/tests/test_gomb_soma_vision_quadtree.py) | 261 | 17 unit tests |

## 3. CORE.YAML items touched

None.

## 4. Scoring policy

Per anchor $v$ in the current frontier:
$$
s(v) \;=\; \alpha \cdot \mathrm{std}\bigl(\text{pixels in } v\bigr)
       \;+\; \beta \cdot |\kappa_v|
$$
where $\alpha$ = `variance_weight` (default 1.0), $\beta$ = `curvature_weight` (default 0.0), and $\kappa_v$ is the Forman vertex curvature on the same-scale 4-connected frontier graph.

The anchor subdivides if $s(v) >$ `score_threshold`. Children inherit half the parent's side length and are positioned in the four quadrants of the parent's region.

## 5. Honest deviation from the plan

The plan §3.2 specified pure $|\kappa_v|$-driven subdivision. As Phase 1's report noted, **pure combinatorial Forman κ is degenerate on a uniform 4-connected grid** — every interior anchor has the same degree and zero triangles, so $|\kappa_v|$ alone cannot discriminate between anchor positions on the first iteration. Phase 1 documented this and noted the κ signal becomes meaningful only after SDRF rewiring (Phase 6) or with feature-weighted Forman.

The implementation therefore makes the score a **hybrid** with configurable weights. Default is variance-only (the natural baseline for content-driven anchor selection); κ-only (`curvature_weight > 0, variance_weight = 0`) and combinations are accessible. This is documented in the source docstring and the report rather than papered over.

## 6. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_quadtree.py -v
=========== 17 passed in 1.96s ===========
```

### What each test pins

| Test | Property |
|---|---|
| `test_rejects_misaligned_image` | image_h % patch_size_initial must be 0 |
| `test_rejects_bad_patch_size` | patch_size_initial ≥ 1 |
| `test_rejects_bad_min_size` | patch_size_min ≤ patch_size_initial |
| `test_rejects_all_zero_weights` | at least one of (α, β) must be positive |
| `test_uniform_image_no_subdivision` | constant image → only scale-0 anchors |
| `test_scale_0_uniform_tiling` | scale-0 covers image with no gaps / overlaps |
| `test_high_variance_region_subdivides` | bright spot in one quadrant → subdivision only there |
| `test_subdivision_respects_depth_bound` | max_depth honored |
| `test_subdivision_respects_anchor_budget` | max_anchors honored |
| `test_parent_child_tiling` | 4 children at half side length, in 4 quadrants of parent |
| `test_size_consistent_with_scale` | size(v) = patch_size_initial / 2^scale(v) |
| `test_scale_0_anchors_have_no_parent` | parent_idx = -1 iff scale = 0 |
| `test_determinism` | same input → same output, every call |
| `test_curvature_weight_path_runs` | κ-only mode runs and terminates |
| `test_hybrid_score_combines_variance_and_curvature` | (α, β) both positive works |
| `test_output_is_anchor_tree` | output is `AnchorTree` with consistent shapes |
| `test_rejects_wrong_image_shape` | (C, H, W) required |

The key behavioural test is `test_high_variance_region_subdivides`:
a 32×32 image with one bright pixel in the upper-left quadrant
yields exactly four scale-0 anchors (initial tiling at 16×16) PLUS
four scale-1 children of the upper-left scale-0 anchor only — the
other three scale-0 anchors don't subdivide because they have
zero variance. The selectivity is exactly what we want: subdivision
fires where content is, not uniformly.

## 7. Architecture commentary

### 7.1 Why this is the right primitive for object detection

YOLO's bottleneck is anchor uniformity: every grid cell pays the
same compute cost regardless of whether it contains a complex
foreground or featureless background. Most compute is wasted on
background. The quadtree fixes this at the root:

* **Featureless region (sky, blank wall)**: variance ≈ 0, no
  subdivision, stays at coarse scale 0. One anchor covers the
  whole region.
* **Object-rich region (digit, vehicle, face)**: high variance,
  subdivision fires, recursive refinement until features are
  locally homogeneous or budget is exhausted.

This is the multi-resolution behaviour FPN tries to deliver via a
separate top-down pathway; the quadtree gives it for free in a
single bottom-up pass.

### 7.2 The bottleneck signal

When the κ weight is non-zero, the score also reflects graph-
theoretic bottleneck pressure. On a uniform 4-connected grid every
interior anchor has $\kappa = -6$; border anchors $-5$, $-4$, $-3$
depending on degree pair. After SDRF rewiring (Phase 6) the
distribution changes and κ becomes a meaningful subdivision signal.
For Phase 2, variance carries the load; κ is available for ablation.

### 7.3 Multi-scale graph structure for downstream layers

The output `AnchorTree` records `parent_indices` explicitly. Phase 5
(`StimulusGraphBuilder`) will use these to construct cross-scale
edges in the multi-scale hypergraph — edges linking an anchor to
its parent at the coarser scale, in addition to same-scale 4-conn
edges between siblings.

### 7.4 Topology-agnostic downstream

The Walk / Polygon / Triangle layers (Phases 2 / 3-G / 4 of the main
GömbSoma plan) consume any (primitives, signs, M_v) tuple. They
don't care that the multi-scale hypergraph has anchors at varying
sizes or that some anchors are children of others. The quadtree's
geometric structure is invisible to them; only the hypergraph
topology matters.

## 8. Performance

CPU-only, 1.96 s for all 17 tests. Per-image quadtree construction
is dominated by the variance computation ($O(N \cdot P^2)$ where $N$
is the anchor count and $P$ is the largest patch side); cheap
relative to the downstream conv layers.

## 9. Numerical stability

All scoring uses `region.std()` on float pixel data. FP32-stable.
No catastrophic cancellation. The integer arithmetic on positions
and sizes is exact.

## 10. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/quadtree.py
   (clean)
```

No new suppressions.

## 11. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — single class, single forward |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 198 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — typed dataclass + numeric weights |
| 8 | Structural forward flag | NO — `variance_weight` / `curvature_weight` are parametric |
| 9 | Bypassing strategy traits | N/A — quadtree is geometry, not a HypergraphConv |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 12. Phase 2 acceptance

- [x] `AdaptiveQuadtree` recursively subdivides anchors by score threshold.
- [x] Multi-scale `AnchorTree` output with parent-child links.
- [x] Depth bound honored.
- [x] Anchor budget honored.
- [x] Deterministic.
- [x] Variance + curvature scoring modes both work.
- [x] 17 unit tests pass.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 13. Next phase

**Phase 3: HodgeLaplacian0/1/2.** Signed boundary operators
$\partial_k$ and Hodge Laplacians $\Delta_k$ at each simplex
dimension. Unit tests including:
* $\partial_k \partial_{k+1} = 0$ (the fundamental identity, pinned exactly);
* Hodge decomposition theorem on small examples (reconstruction of arbitrary chains);
* `Δ₀` reduces to the standard graph Laplacian on vertices.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py`
* `signedkan_wip/tests/test_gomb_soma_vision_hodge.py`

No phase 3 work in this commit, per the one-phase-per-session rule.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import AdaptiveQuadtree

qt = AdaptiveQuadtree(
    image_h=28, image_w=28, patch_size_initial=4,
    patch_size_min=1, max_depth=2, max_anchors=256,
    variance_weight=1.0, curvature_weight=0.0,
    score_threshold=0.05,
)
tree = qt(image)  # image: (C, H, W) FloatTensor
# tree.positions, tree.sizes, tree.scales, tree.parent_indices
```

No new dependencies.

---

*End of Ricci-Stim phase 2 report.*
