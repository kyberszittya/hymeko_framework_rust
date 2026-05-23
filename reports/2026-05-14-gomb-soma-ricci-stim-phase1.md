# GömbSoma-Ricci-Stim Phase 1 — FormanCurvatureHead

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 1 of 8

## 1. Summary

Built `FormanCurvatureHead`, the bottom rung of the Ricci-Stim
differential-geometric stack. Forman-Ricci curvature (Forman 2003)
is a combinatorial graph invariant computed from local degree and
triangle counts. It will drive (a) the quadtree subdivision
decision in phase 2, (b) the Bochner-coupled message-passing
correction in phase 4, and (c) the SDRF curvature-rewiring
preprocessor in phase 6.

The module is stateless — no learnable parameters. It computes a
deterministic graph-theoretic invariant. Wrapping it as `nn.Module`
keeps the API consistent and allows learnable variants (weighted /
signed Forman, Ollivier-Ricci approximation) to subclass cleanly in
later phases.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/forman.py](../signedkan_wip/src/hymeko_gomb/soma/vision/forman.py) | 152 | `FormanCurvatureHead` + `FormanCurvature` dataclass |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +6 / -2 | re-export |
| [signedkan_wip/tests/test_gomb_soma_vision_forman.py](../signedkan_wip/tests/test_gomb_soma_vision_forman.py) | 235 | 16 tests against classical graph invariants |

No edits to Gömb / Phase 1-3-G / Phase 3-V / CORE.YAML.

## 3. CORE.YAML items touched

None.

## 4. The Forman-Ricci formula

For an edge $e = (u, v)$ in an undirected graph:

$$
\kappa(e) = 2 - \deg(u) - \deg(v) + 2 \cdot |\Delta(u, v)|
$$

where $|\Delta(u, v)|$ is the number of triangles incident on $e$.

* Positive κ: dense / clique-like neighbourhood. Maximum on complete graphs $K_n$ where κ = 0 (the formula's "flat-Riemannian" zero).
* Zero κ: triangle-balanced. K_n always achieves this.
* Negative κ: bottleneck. Cycles $C_n$ (n ≥ 4) and stars $S_n$ give strongly negative κ.

Per-vertex κ_v is the mean of incident edge κ values.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_forman.py -v
=========== 16 passed in 2.03s ===========
```

### Pinned graph-theoretic invariants

| Test graph | Topology | Expected κ | Pinned |
|---|---|---|---|
| $K_3$ (triangle) | deg=2, 1 triangle/edge | $2 - 2 - 2 + 2 = 0$ | ✓ |
| $K_4$ | deg=3, 2 triangles/edge | $2 - 6 + 4 = 0$ | ✓ |
| $K_5$ | deg=4, 3 triangles/edge | $2 - 8 + 6 = 0$ | ✓ |
| $C_4$ | deg=2, no triangles | $2 - 4 + 0 = -2$ | ✓ |
| $C_6$ | deg=2, no triangles | $-2$ | ✓ |
| $P_4$ (path) | mixed deg, no triangles | -1 / -2 / -1 | ✓ |
| $S_5$ (star) | centre deg 5, leaves deg 1 | $2 - 5 - 1 = -4$ | ✓ |
| 3×3 grid | mixed deg, no triangles | -3 corner-side, -5 side-centre | ✓ |

### Module contract

| Test | What it pins |
|---|---|
| `test_isolated_vertices_have_zero_kappa` | disconnected vertex → vertex_κ = 0 |
| `test_self_loops_dropped` | (u, u) entries silently ignored |
| `test_empty_graph` | 0-edge input → zero-tensor outputs of right shape |
| `test_rejects_wrong_edge_shape` | non-(n, 2) input → ValueError |
| `test_undirected_handling` | duplicate (u, v) + (v, u) merged for degree count |
| `test_vertex_kappa_is_mean_of_incident` | per-vertex aggregation matches manual calc |
| `test_dataclass_fields_present` | `FormanCurvature` has edge_kappa, vertex_kappa, degree, triangle_count |
| `test_no_learnable_parameters` | head is stateless (0 params) |

## 6. Architectural commentary

### 6.1 Why Forman, not Ollivier

Ollivier-Ricci uses optimal-transport (Wasserstein-1) between
neighbourhood distributions; it's the "correct" Riemannian-curvature
analogue but expensive (per-edge Wasserstein). Forman is purely
combinatorial — degree counts and triangle counts. For our use case
(per-layer recomputation on a possibly-thousands-of-edges graph),
Forman is the only feasible choice. The κ-driven SDRF rewiring
(phase 6) and quadtree subdivision (phase 2) only need a *signed,
locally-consistent* curvature signal, which Forman provides.

### 6.2 Why this matters for the 4-connected grid

A 4-connected patch grid is triangle-free. Forman κ then reduces to
$2 - \deg(u) - \deg(v)$, which depends only on local degree. On a 3×3
grid this gives the expected pattern:

| Edge type | κ |
|---|---|
| corner ↔ side | -3 |
| side ↔ side | -4 |
| side ↔ centre | -5 |
| centre ↔ centre (if exists in larger grid) | -6 |

All negative. The interior of the grid is a global bottleneck under
Forman — exactly what we want for an SDRF-rewiring target (phase 6
will add shortcut edges to alleviate it).

### 6.3 Why this is the right substrate for the quadtree

The phase-2 `AdaptiveQuadtree` will subdivide patches where $|\kappa_v|$
is large (or where κ is locally inhomogeneous). On the uniform-feature
parts of an image, every patch has identical κ; no subdivision
happens; the grid stays coarse. On a textured / object-boundary
region, feature gradients shift the degree distribution and κ
becomes inhomogeneous; subdivision fires. This is exactly the
behaviour we want from a content-driven anchor selector.

## 7. Performance

CPU-only, 2.03 s for all 16 tests. The Forman computation is
$O(|E| \cdot \bar{d})$ (each edge intersects two adjacency sets);
fits easily in the per-layer budget for any hypergraph that fits in
GPU memory at all.

## 8. Numerical stability

All arithmetic is FP32 over small-integer quantities (degrees, triangle
counts). No catastrophic cancellation. Exact-equality assertions in
tests pass.

## 9. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/forman.py
   (clean)
```

No new suppressions.

## 10. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 152 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Structural-flag forward branches | NO — the only flag (`treat_as_undirected`) is parametric, not structural |
| 9 | Bypassing existing strategy traits | N/A — this is a new graph-functional, not a HypergraphConv subclass |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 11. Phase 1 acceptance

- [x] `FormanCurvatureHead` computes Forman-Ricci κ on a graph.
- [x] Per-edge κ and per-vertex κ_v outputs.
- [x] 16 unit tests pass against classical graph invariants.
- [x] No learnable parameters (stateless functional).
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 12. Deviation from plan

The plan specified `src/hymeko_gomb/soma/vision/forman.py`. Followed
exactly. (Forman is general-purpose for any signed hypergraph; later
phases may lift it from `vision/` to `soma/` if Phase 4 / TriangleConv
needs it outside the vision subpackage. Deferring that decision until
the use case is concrete.)

## 13. Next phase

**Phase 2: `AdaptiveQuadtree`.** Recursive κ-driven subdivision of
patches. Each level computes Forman κ on the current anchor set,
splits anchors with $|\kappa| > \tau$, recomputes, and iterates until
budget or depth bound is reached. Tests:

* Determinism: same input → same anchor tree.
* Depth bound: never exceeds the user-set max depth.
* Subdivision-on-bottleneck: a high-feature-variance region subdivides;
  a uniform region does not.
* Total-anchor budget cap.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/vision/quadtree.py`
* `signedkan_wip/tests/test_gomb_soma_vision_quadtree.py`

No phase 2 work in this commit, per the one-phase-per-session rule.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import FormanCurvatureHead

head = FormanCurvatureHead()
out = head(edges_tensor, n_nodes=N)
print(out.edge_kappa)    # Tensor[n_edges]
print(out.vertex_kappa)  # Tensor[N]
```

No new dependencies. Pure PyTorch + Python stdlib.

---

*End of Ricci-Stim phase 1 report.*
