# GömbSoma-Ricci-Stim Phase 5 — StimulusGraphBuilder

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 5 of 8
**Prior phases:** [1 Forman](2026-05-14-gomb-soma-ricci-stim-phase1.md), [2 Quadtree](2026-05-14-gomb-soma-ricci-stim-phase2.md), [3 Hodge](2026-05-14-gomb-soma-ricci-stim-phase3.md), [4 Bochner](2026-05-14-gomb-soma-ricci-stim-phase4.md)

## 1. Summary

Built `StimulusGraphBuilder`, the bridge between an `AnchorTree`
(Phase 2 output, multi-scale anchor geometry) and the GömbSoma layer
stack (Walk / Polygon / Triangle, optionally Bochner-wrapped). For
each `(AnchorTree, anchor_features)` pair the builder produces a
`StimulusGraph` containing:

- Combined edge list: same-scale 4-connected siblings + cross-scale
  parent-child links.
- Edge signs from feature-inner-product polarity: $\sigma(u, v) = \text{sign}(\langle f_u, f_v \rangle - \theta)$.
- Walks (length 2), polygons (4-cycles), triangles (3-cliques) over
  the combined graph, with σ-products per primitive.
- Per-primitive incidence matrices $M_v$.
- Per-primitive Forman κ curvatures (mean over constituent edges).
- The Hodge Laplacian $\Delta_0$ on the anchor set.

The output is consumed directly by Bochner-wrapped Walk / Polygon /
Triangle layers via `prepare()` then `forward()`.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py](../signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py) | 297 | `StimulusGraphBuilder` + `StimulusGraph` dataclass |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +6 / -0 | re-exports |
| [signedkan_wip/tests/test_gomb_soma_vision_stim_graph.py](../signedkan_wip/tests/test_gomb_soma_vision_stim_graph.py) | 281 | 20 tests including end-to-end integration with `AdaptiveQuadtree` |

## 3. CORE.YAML items touched

None.

## 4. Architecture and data flow

```
   AdaptiveQuadtree(image)  →  AnchorTree
                                   │
                                   │   StimulusGraphBuilder
                                   │   (this phase)
                                   ▼
                              StimulusGraph
                                   │
                                   ├─→ edges, edge_signs, edge_curvatures
                                   ├─→ walks, walk_signs, M_v_walks, walk_curvatures
                                   ├─→ polygons, polygon_signs, M_v_polygons, polygon_curvatures
                                   ├─→ triangles, triangle_signs, M_v_triangles, triangle_curvatures
                                   └─→ hodge_laplacian_0
                                   │
                                   │   BochnerHypergraphConv.prepare(
                                   │       hodge_laplacian=...,
                                   │       primitive_curvatures=...,
                                   │   )
                                   ▼
                              Walk / Polygon / Triangle forward
```

### 4.1 Edge sources

Same-scale: two anchors with identical (size, scale) and centers
exactly one side-length apart in horizontal or vertical direction
→ 4-connected edge. Deduplicated to undirected.

Cross-scale: every anchor with `parent_indices[i] >= 0` contributes
an edge `(i, parent)`. The cross-scale edges are what turn the pure
4-conn grid (triangle-free) into a graph with non-trivial triangle
structure.

### 4.2 Edge signs

$$\sigma(u, v) = \begin{cases} +1 & \text{if } \langle f_u, f_v \rangle \geq \theta \\ -1 & \text{otherwise} \end{cases}$$

with $\theta$ = `sign_threshold`, default 0. Aligned features →
positive edge; anti-aligned → negative. This is the
canonical sign function for content-driven hypergraph construction.

### 4.3 Primitive σ-products

For walks $w = (a, b, c)$: σ-product = $\sigma(a, b) \cdot \sigma(b, c)$.
For polygons $p = (v_0, v_1, v_2, v_3)$: σ-product = $\prod_{i} \sigma(v_i, v_{i+1 \mod 4})$.
For triangles $t = (a, b, c)$: σ-product = $\sigma(a, b) \cdot \sigma(a, c) \cdot \sigma(b, c)$.

Computed in tensor form via `edge_signs[walk_eidx].prod(dim=1)`.

### 4.4 Per-primitive curvatures

For each primitive, the curvature is the mean of Forman κ over its
constituent edges. Sourced from `FormanCurvatureHead` (Phase 1) on
the same combined edge list. Used downstream by
`BochnerHypergraphConv` as the Ricci-correction weight.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_stim_graph.py -v
=========== 20 passed in 1.95s ===========
```

### 5.1 Edge construction (5 tests)

| Test | What it pins |
|---|---|
| `test_same_scale_edges_on_flat_grid` | 3×3 grid → 12 undirected 4-conn edges |
| `test_cross_scale_edges_added` | multiscale tree (4 scale-0 + 4 scale-1 children) → 12 edges total (4 same + 4 child-grid + 4 cross-scale) |
| `test_edge_signs_positive_when_features_aligned` | aligned features → all σ = +1 |
| `test_edge_signs_negative_when_features_anti_aligned` | alternating features in a row → all σ = -1 |
| `test_edge_curvature_shape` | one κ per edge |

### 5.2 Primitive enumeration (5 tests)

| Test | What it pins |
|---|---|
| `test_walks_have_length_3` | walks have shape (·, 3) |
| `test_walks_no_backtracking` | walk endpoints ≠ |
| `test_polygons_are_4cycles_in_grid` | 2×2 grid → exactly one 4-cycle plaquette |
| `test_no_triangles_in_pure_flat_grid` | 4-conn-only grid is triangle-free |
| `test_triangles_exist_when_cross_scale_edges` | quadtree subdivision produces 4 triangles per parent (each adjacent sibling pair forms one with the parent) |

### 5.3 σ-products and curvatures (3 tests)

| Test | What it pins |
|---|---|
| `test_walk_sign_is_sigma_product_of_edges` | walk σ = $\sigma_{e_1} \cdot \sigma_{e_2}$ |
| `test_walk_curvature_is_mean_of_edges` | walk κ = $\frac{1}{2}(\kappa_{e_1} + \kappa_{e_2})$ |
| `test_M_v_polygons_uniform_weight` | M_v entries are $1/k$ where $k$ = primitive arity |

### 5.4 Robustness (7 tests)

| Test | What it pins |
|---|---|
| `test_M_v_walks_shape` | M_v shape (n_anchors, n_walks) |
| `test_hodge_laplacian_shape` | (n_anchors, n_anchors) sparse |
| `test_determinism` | same input → same output |
| `test_walks_respect_budget` | n_walks ≤ max_walks |
| `test_output_is_stimulus_graph` | output is `StimulusGraph` with `n_anchors` set |
| `test_rejects_mismatched_features` | features-tensor shape validation |
| `test_integration_with_quadtree` | end-to-end: AdaptiveQuadtree(img) → StimulusGraphBuilder → StimulusGraph |

## 6. Bug found and fixed during testing

`_enumerate_walks` initially rebuilt its own edge-index lookup from
the adjacency. The lookup keyed canonical pairs to a *different*
ordering than the caller's combined edge list, so the `walk_edge_idx`
indices pointed to wrong entries in `edge_signs` / `edge_curvatures`.
The σ-product test caught this immediately (sign-product mismatch
in `test_walk_sign_is_sigma_product_of_edges`).

Fixed by threading the caller's `edge_lookup` dict into
`_enumerate_walks` as a parameter, eliminating the local rebuild.
The σ-product and curvature pinning tests both pass after the fix.

This is exactly why we pin σ-products with explicit recompute-from-edges
in a test — it catches any drift between primitive enumeration and
edge ordering.

## 7. Architectural commentary

### 7.1 Why this is the central plumbing piece

Phases 1–4 each built one component:

- Phase 1 (Forman κ): per-edge curvature
- Phase 2 (AdaptiveQuadtree): per-image anchor geometry
- Phase 3 (HodgeLaplacian): boundary operators + Δ_k
- Phase 4 (BochnerHypergraphConv): three-term message passing

None of them on their own *use* the GömbSoma layer machinery on
real input. Phase 5 is what wires everything together: it takes
the anchor geometry, produces the full hypergraph, computes every
geometric quantity the downstream conv layers need, and packages
it as one dataclass. The next phases (SDRF rewiring, MNIST
classifier, Cluttered MNIST detector) consume `StimulusGraph`
directly.

### 7.2 Cross-scale edges are what create triangles

A pure 4-connected grid is triangle-free, so the Cartwright–Harary
balance gate (Phase 4, Triangle-conv) would have no triangles to
operate on if we used only same-scale edges. The cross-scale parent-
child edges from the quadtree change that: every adjacent sibling
pair under a parent forms a triangle with the parent (3 edges,
3 vertices). The multiscale-tree test confirms this — 4 triangles
per fully-subdivided parent.

### 7.3 Determinism

Every step uses sorted iteration over Python lists/sets. No
random sampling. Same `(AnchorTree, features)` input produces
the same `StimulusGraph` byte-for-byte. Pinned by
`test_determinism`.

## 8. Performance

20 tests, 1.95 s CPU. Primitive enumeration is $O(\bar{d}^2 \cdot n)$
for walks and triangles (where $\bar{d}$ is mean degree), $O(n)$ for
polygons (one plaquette per same-scale top-left corner). At
Cluttered MNIST scale (~50-200 anchors), all enumerations are
fast even before the budget caps kick in.

## 9. Numerical stability

Sign computation uses `torch.where` over inner-product values —
FP32-stable. σ-products are integer products in $\{-1, +1\}$ — exact.
Curvature averaging is FP32 mean of small integers — stable.

## 10. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py
   (clean)
```

No new suppressions.

## 11. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — single builder class, single forward |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | 297 LOC — under the 400-LOC heuristic |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — numeric weights and caps |
| 8 | Forward-time structural flags | NO |
| 9 | Bypassing strategy traits | N/A — this is a graph-builder, not a HypergraphConv |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 12. Phase 5 acceptance

- [x] `StimulusGraph` dataclass with all per-primitive tensors.
- [x] `StimulusGraphBuilder` produces all fields deterministically.
- [x] Same-scale + cross-scale edges combined correctly.
- [x] Edge signs from feature inner products.
- [x] σ-products computed correctly (walk-sign test pinned).
- [x] Per-primitive curvatures from Forman κ.
- [x] Hodge Laplacian Δ_0 included.
- [x] 20 unit tests pass.
- [x] Integration smoke with `AdaptiveQuadtree` passes.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 13. Next phase

**Phase 6: SDRF (Stochastic Discrete Ricci Flow) rewiring.** Take
the `StimulusGraph` and add shortcut edges where Forman κ is most
negative; iterate until $\min_e \kappa(e)$ is bounded away from the
worst-case. This addresses the over-squashing problem
(Topping et al. 2022) on the multi-scale patch hypergraph.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py`
* `signedkan_wip/tests/test_gomb_soma_vision_sdrf.py`

No phase 6 work in this commit, per the one-phase-per-session rule.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import (
    AdaptiveQuadtree, StimulusGraphBuilder,
)

qt = AdaptiveQuadtree(image_h=28, image_w=28, patch_size_initial=4,
                       patch_size_min=1, max_depth=2)
tree = qt(image)
features = patch_encoder(extract_patches(image, tree))

builder = StimulusGraphBuilder()
sg = builder(tree, features)
# sg.walks / sg.polygons / sg.triangles consumed by Walk/Polygon/Triangle layers.
# sg.hodge_laplacian_0 + sg.walk_curvatures fed to BochnerHypergraphConv.prepare().
```

No new dependencies.

---

*End of Ricci-Stim phase 5 report.*
