# GömbSoma Phase 3-G — PolygonConvLayer

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma/](../docs/plans/2026-05-14-gomb-soma/)
**Phase:** 3-G (main-plan polygon layer)
**Prior phases:** [Phase 1](2026-05-14-gomb-soma-phase1.md), [Phase 2](2026-05-14-gomb-soma-phase2.md), [Phase 3-V](2026-05-14-gomb-soma-phase3v-vision.md)

## 1. Summary

Built `PolygonConvLayer`, the second concrete `HypergraphConv`
subclass. Operates on closed cycles ($c_k$ for $k \geq 3$). Sign-
branched and **cyclic-and-reflection-invariant** — the message
function gives the same answer regardless of which vertex is
chosen as the cycle's starting point or which traversal direction
is used.

This is the second rung of the GömbSoma compositional ladder:
walks (open) → polygons (closed) → triangles (closed + balance
gate) → abstractions.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/polygon_layer.py](../signedkan_wip/src/hymeko_gomb/soma/polygon_layer.py) | 121 | `PolygonConvLayer` — cyclic/reflection invariant, sign-branched |
| [signedkan_wip/src/hymeko_gomb/soma/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/__init__.py) | +1 / -0 | re-export |
| [signedkan_wip/tests/test_gomb_soma_polygon_layer.py](../signedkan_wip/tests/test_gomb_soma_polygon_layer.py) | 210 | 10 unit tests including planted-4-cycle SBM smoke |

## 3. CORE.YAML items touched

None.

## 4. Architecture and the cyclic-invariance argument

A closed polygon $c = (v_0, v_1, \ldots, v_{k-1})$ has no canonical
starting vertex: it is identical to any cyclic shift, and (for
undirected graphs) to the reversal. The message function must be
invariant under both operations.

WalkConvLayer's position-aware mechanism $\sum_i W_i x_{v_i}$ would
collapse under cyclic averaging to $\left( \frac{1}{k} \sum_i W_i \right) \sum_j x_{v_j}$
— a position-agnostic projection. So position-aware weights add
**no information** under cyclic symmetry; they only inflate
parameter count.

PolygonConvLayer therefore uses the **cycle-mean** of vertex
features as its first-moment invariant:

$$
\text{msg}(c) = \text{GELU}\!\left( W^{\pi(c)} \cdot \frac{1}{k} \sum_{i=0}^{k-1} x_{v_i} + b^{\pi(c)} \right)
$$

* $W^{\pm} \in \mathbb{R}^{\text{in} \times \text{out}}$ — sign-branched.
* $b^{\pm} \in \mathbb{R}^{\text{out}}$ — sign-branched.
* GELU activation, identical on both branches.

Cyclic invariance and reflection invariance are guaranteed by
construction (the mean is symmetric in all $k$ arguments). The
discriminative power comes from sign-branching (positive vs
negative σ-product polygons) — which is what Cartwright–Harary
balance gives us at the architectural level.

**Higher-order invariants** (sum of adjacent products, DFT magnitudes,
etc.) are left to future phases. For triangles (k=3), the
TriangleConvLayer in Phase 4 adds back discriminative power via the
explicit Cartwright–Harary balance gate.

## 5. Parameter count

Independent of `k_arity`: $2 \cdot \text{in} \cdot \text{out} + 2 \cdot \text{out}$.

| Config | Params |
|---|---|
| `in=out=16, k=4` | 544 |
| `in=out=16, k=5` | 544 |
| `in=out=16, k=6` | 544 |

Compare to WalkConvLayer at the same shape: 1 568 params. Polygons
are 65 % lighter because position-aware weights are useless under
cyclic symmetry.

## 6. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_polygon_layer.py -v
test_rejects_k_arity_below_3                  PASSED
test_construction_and_param_count             PASSED
test_param_count_independent_of_k_arity       PASSED
test_forward_shape                            PASSED
test_cyclic_invariance                        PASSED
test_reflection_invariance                    PASSED
test_sign_branching_actually_branches         PASSED
test_permutation_equivariance                 PASSED
test_gradient_flow_on_every_branch            PASSED
test_sbm_smoke_with_planted_4cycles           PASSED
============================ 10 passed in 1.99s ============================
```

What each test pins:

| Test | What it pins |
|---|---|
| `test_rejects_k_arity_below_3` | a polygon needs ≥ 3 vertices |
| `test_construction_and_param_count` | $2 \cdot \text{in} \cdot \text{out} + 2 \cdot \text{out}$ |
| `test_param_count_independent_of_k_arity` | constant across $k \in \{3,4,5,6\}$ |
| `test_forward_shape` | output is $(n_\text{nodes}, \text{out})$ |
| **`test_cyclic_invariance`** | rolling each polygon's vertex order by 1 leaves the message identical (atol $10^{-6}$) |
| **`test_reflection_invariance`** | reversing each polygon's vertex order leaves the message identical |
| `test_sign_branching_actually_branches` | flipping all signs moves the output by $> 10^{-3}$ |
| `test_permutation_equivariance` | global vertex permutation passes through |
| `test_gradient_flow_on_every_branch` | both sign-branch weight slices receive non-zero gradient |
| `test_sbm_smoke_with_planted_4cycles` | end-to-end on a 40-vertex SBM with 30 planted 4-cycles (half balanced, half not); produces non-degenerate output |

The two invariance tests are the headline: cyclic and reflection
invariance are properties of the polygon as a graph object, not
of any one tensor representation. The tests pin both at atol $10^{-6}$
(numerical floor) — they pass *exactly*, not approximately.

## 7. Performance

Tests run in 1.99 s on CPU. No GPU. Performance contracts apply at
the hierarchy level (Phase 6), not at the per-layer level.

## 8. Numerical stability

Forward path: `gather → mean → einsum → bias → GELU`. All operations
FP32-stable. No catastrophic cancellation; the cycle-mean is a
well-conditioned averaging operation.

## 9. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/polygon_layer.py
   (clean)
```

No new suppressions.

## 10. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — single class, single config struct |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 121 LOC |
| 5 | New axis = new function name | NO — class-per-structural-variant per plan |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Forward-time structural flags | NO — sign-branching is parametric |
| 9 | Bypassing strategy traits | NO — implements HypergraphConv |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 11. Phase 3-G acceptance

- [x] `PolygonConvLayer(HypergraphConv)` implements `_forward_messages`.
- [x] Sign-branched.
- [x] Cyclic-invariant: tested numerically at atol $10^{-6}$.
- [x] Reflection-invariant: tested numerically at atol $10^{-6}$.
- [x] k_arity-independent parameter count.
- [x] 10 unit tests pass.
- [x] SBM smoke produces non-degenerate output.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 12. Architectural ladder so far

| Phase | Layer | Primitive | Closure | Position info | Param-count formula |
|---|---|---|---|---|---|
| 2 | `WalkConvLayer` | open walk $w_k$ | none | yes (directed) | $2 k_\text{arity} \cdot \text{in} \cdot \text{out} + 2 \cdot \text{out}$ |
| 3-G | `PolygonConvLayer` | closed cycle $c_k$ | ring closure | no (cyclic-inv) | $2 \cdot \text{in} \cdot \text{out} + 2 \cdot \text{out}$ |
| 4 (next) | `TriangleConvLayer` | $c_3$ + Cartwright–Harary | full balance | TBD | TBD |
| 5 | `InterLayerCliffordFIR` | inter-layer transfer | grade lift | n/a | TBD |
| 6 | `AbstractionConv` | derivative-nodelet | contraction | n/a | TBD |
| 7 | `GombSomaHierarchy` | top-level cascade | --- | --- | --- |

## 13. Next phase

**Phase 4: TriangleConvLayer.** Specialised handling of $c_3$ with
the Cartwright–Harary balance gate: balanced triads ($\pi(c) = +1$)
route through one bank that exploits the *all-positive* structural
constraint; frustrated triads route through another. The triangle
is small enough that the full vertex-tuple (not just the mean) is
worth exploiting.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/triangle_layer.py`
* `signedkan_wip/tests/test_gomb_soma_triangle_layer.py`
* SBM smoke with planted balanced/frustrated triads.

No phase 4 work in this commit, per the rule.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConvConfig, PolygonConvLayer,
)

cfg = HypergraphConvConfig(in_features=16, out_features=16, k_arity=4)
layer = PolygonConvLayer(cfg)
y = layer(x, polygons, signs, M_v)
```

---

*End of phase 3-G report.*
