# GömbSoma-Ricci-Stim Phase 3 — HodgeLaplacian

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 3 of 8
**Prior phases:** [Phase 1 — FormanCurvatureHead](2026-05-14-gomb-soma-ricci-stim-phase1.md), [Phase 2 — AdaptiveQuadtree](2026-05-14-gomb-soma-ricci-stim-phase2.md)

## 1. Summary

Built `HodgeLaplacian`, the algebraic backbone of the Ricci-Stim
geometric stack. Given a signed simplicial complex
$(V, E, T)$ — vertices, edges, triangles — the module produces:

* signed boundary operators $\partial_1: C_1 \to C_0$ and $\partial_2: C_2 \to C_1$;
* Hodge Laplacians $\Delta_0$ (vertex-level), $\Delta_1$ (edge-level), $\Delta_2$ (triangle-level).

The headline contract pinned by tests is the **fundamental identity**

$$\partial_1 \partial_2 = 0$$

which holds *exactly* (not within tolerance — in integer arithmetic).
This identity is what makes the Hodge Laplacians well-defined and
what underpins the Hodge decomposition theorem we use in Phase 4
for Bochner-coupled message passing.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py](../signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py) | 200 | `HodgeLaplacian` + `HodgeOperators` dataclass |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +7 / -0 | re-exports |
| [signedkan_wip/tests/test_gomb_soma_vision_hodge.py](../signedkan_wip/tests/test_gomb_soma_vision_hodge.py) | 290 | 18 tests including Hodge-decomposition reconstruction |

## 3. CORE.YAML items touched

None.

## 4. The operators

### 4.1 Signed boundaries

For canonical edges (sorted vertex pair $u < v$):

$$\partial_1 \, [u, v] = +[v] - [u]$$

For canonical triangles ($v_0 < v_1 < v_2$):

$$\partial_2 \, [v_0, v_1, v_2] = +[v_1, v_2] - [v_0, v_2] + [v_0, v_1]$$

Both are sparse $\{-1, 0, +1\}$-valued matrices in COO format.

### 4.2 Hodge Laplacians

| Dimension | Formula | Shape | Acts on |
|---|---|---|---|
| 0 | $\Delta_0 = \partial_1 \partial_1^\top$ | $n_v \times n_v$ | vertex features |
| 1 | $\Delta_1 = \partial_1^\top \partial_1 + \partial_2 \partial_2^\top$ | $n_e \times n_e$ | edge features |
| 2 | $\Delta_2 = \partial_2^\top \partial_2$ | $n_t \times n_t$ | triangle features |

$\Delta_0$ reduces *exactly* to the standard graph Laplacian
$D - A$. Verified by test on a triangle and a path.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_hodge.py -v
=========== 18 passed in 1.94s ===========
```

### 5.1 The fundamental identities (pinned exactly)

| Test | Identity | Where |
|---|---|---|
| `test_partial_partial_is_zero_on_triangle` | $\partial_1 \partial_2 = 0$ | K₃ (1 triangle, 3 edges) |
| `test_partial_partial_is_zero_on_K4` | $\partial_1 \partial_2 = 0$ | K₄ (4 triangles, 6 edges, 4 vertices) |

Both pinned at `torch.equal` (exact equality). The identity is built
into the integer-arithmetic boundary formulas; it cannot drift.

### 5.2 The graph-Laplacian reduction

| Test | What it checks |
|---|---|
| `test_laplacian_0_equals_D_minus_A_on_triangle` | $\Delta_0$ on K₃ = `diag([2,2,2]) - all-ones-off-diag-1` |
| `test_laplacian_0_equals_D_minus_A_on_path` | $\Delta_0$ on P₃ = `diag([1,2,1]) - bidiag` |
| `test_laplacian_0_eigenvalue_zero_for_each_component` | two disjoint triangles → 2 zero eigenvalues |
| `test_laplacian_0_symmetric` | $\Delta_0 = \Delta_0^\top$ |
| `test_laplacian_0_positive_semidefinite` | all eigenvalues $\geq 0$ |

These pin $\Delta_0$ to the standard graph-theoretic Laplacian.
Anything downstream that uses $\Delta_0$ inherits the correctness.

### 5.3 The Hodge decomposition

`test_hodge_decomposition_reconstruction_on_triangle` verifies that
on the filled triangle (one 2-simplex), an arbitrary 1-form $\omega$
decomposes uniquely as

$$\omega = \omega_{\rm harm} + \partial_1^\top \alpha + \partial_2 \beta$$

with each piece in an orthogonal subspace. We construct the
projectors via rank-filtered SVD, project $\omega$ onto each, sum
the three pieces, and check the result equals $\omega$. We also
check $\omega_{\rm harm} \in \ker \Delta_1$ — which on the filled
triangle (Betti-1 = 0) means $\omega_{\rm harm} = 0$.

The triangle is *contractible*, so $\beta_1 = 0$; no harmonic
1-forms exist. The test confirms this empirically by getting
$\omega_{\rm harm} = 0$ within FP32 noise.

### 5.4 Boundary orientation

| Test | What it pins |
|---|---|
| `test_partial_1_canonical_orientation` | Edge $(0, 1)$: $\partial_1 [0,1] = +[1] - [0]$, i.e., row 0 has -1, row 1 has +1 |
| `test_partial_2_triangle_alternating_signs` | Triangle $[0,1,2]$: $+e_{(1,2)} - e_{(0,2)} + e_{(0,1)}$ |

### 5.5 Robustness

| Test | What it pins |
|---|---|
| `test_self_loops_dropped` | $(u, u)$ entries silently ignored |
| `test_degenerate_triangles_dropped` | triangles with a repeated vertex silently ignored |
| `test_empty_input` | 0-edge input → consistent-shape empty sparse outputs |
| `test_rejects_wrong_edge_shape` | non-$(n, 2)$ edge tensor → ValueError |
| `test_rejects_wrong_triangle_shape` | non-$(n, 3)$ triangle tensor → ValueError |
| `test_output_is_HodgeOperators` | output type and sparsity contract |
| `test_no_learnable_parameters` | stateless functional |

## 6. The fundamental identity is the central pin

$\partial \circ \partial = 0$ is what makes the entire Hodge / homology
framework work. It says: the boundary of a boundary vanishes; cycles
are closed-by-construction; the image of $\partial_{k+1}$ sits inside
the kernel of $\partial_k$. Everything that follows — the Hodge
Laplacian's well-definedness, the Hodge decomposition theorem, the
Betti-number readout, the Bochner identity in Phase 4 — depends on
this single identity.

We pin it exactly (not within tolerance) because the boundary
formulas use only integer arithmetic on $\{-1, 0, +1\}$. There is no
numerical excuse for the identity to drift. If a future change to
boundary canonicalisation introduces a sign error, these two tests
catch it before downstream phases can mask it.

## 7. Architectural use

* **Phase 4 (Bochner-coupled HypergraphConv):** the message-passing
  decomposition is
  $\text{msg} = \text{flat} + \alpha \cdot (\Delta_k h) + \beta \cdot \kappa$-correction.
  The $\Delta_k h$ term is computed here as a sparse mat-mul.
* **Phase 5 (StimulusGraphBuilder):** when the multi-scale signed
  hypergraph is constructed, $\partial_1$ and $\partial_2$ are
  reused for the cross-scale parent-child operator.
* **Phase 6 (SDRF rewiring):** the bottleneck-detection step ranks
  edges by Forman $\kappa$ (Phase 1) on the same complex that
  $\Delta_k$ acts on.

## 8. Performance

CPU-only, 1.94 s for 18 tests. All operations are sparse
matrix-matrix multiplies. On Cluttered MNIST scale (a few hundred
anchors and a few hundred triangles), the operators are tiny;
recomputation per training step is cheap.

## 9. Numerical stability

Boundaries are constructed in $\{-1, 0, +1\}$ integer values.
Laplacians are sums and transposes of these — small-integer
arithmetic, FP32-stable. The Hodge decomposition test passes at
atol $10^{-5}$ (FP32 noise floor).

## 10. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py
   (clean)
```

No new suppressions.

## 11. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — single class, single forward |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 200 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Structural forward flags | NO — `triangles` is parametric (optional input, same logic) |
| 9 | Bypassing strategy traits | N/A — this is a new geometric primitive |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 12. Phase 3 acceptance

- [x] `HodgeLaplacian` computes $\partial_1, \partial_2, \Delta_0, \Delta_1, \Delta_2$.
- [x] $\partial_1 \partial_2 = 0$ pinned exactly on K₃ and K₄.
- [x] $\Delta_0$ reduces exactly to $D - A$.
- [x] $\Delta_0$ has one zero eigenvalue per connected component (Betti-0 count).
- [x] Hodge decomposition reconstruction verified on the filled triangle.
- [x] $\Delta_0$ symmetric and positive-semidefinite.
- [x] 18 unit tests pass.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 13. Next phase

**Phase 4: Bochner-coupled `HypergraphConv`.** Adds a wrapper around
the Phase-1 `HypergraphConv` ABC that exposes the
Bochner–Weitzenböck decomposition:

$$\text{msg}(c) = \psi_{\pi(c)}\!\Bigl(\text{flat connection} + \alpha \cdot (\Delta_k h)_c + \beta \cdot \kappa\text{-correction}\Bigr)$$

The critical contract: with $\alpha = \beta = 0$, the wrapper
produces bit-identical output to the existing Phase 1–3-G
`HypergraphConv` subclasses. This guarantees Phase 4 is purely
additive.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/hg_conv_bochner.py`
* `signedkan_wip/tests/test_gomb_soma_bochner_conv.py`
* Phase-4 acceptance criterion: α=β=0 regression test against
  `WalkConvLayer` / `PolygonConvLayer`.

No phase 4 work in this commit, per the one-phase-per-session rule.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import HodgeLaplacian

h = HodgeLaplacian()
out = h(edges, n_vertices=N, triangles=triangles)  # all torch.long
# out.boundary_1: sparse (N, n_e)
# out.boundary_2: sparse (n_e, n_t)
# out.laplacian_0: sparse (N, N)  — equals D - A
# out.laplacian_1: sparse (n_e, n_e)
# out.laplacian_2: sparse (n_t, n_t)
```

No new dependencies.

---

*End of Ricci-Stim phase 3 report.*
