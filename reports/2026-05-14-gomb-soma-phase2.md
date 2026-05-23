# GömbSoma Phase 2 — WalkConvLayer

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma/](../docs/plans/2026-05-14-gomb-soma/)
**Phase:** 2 of 7
**Prior phase:** [phase 1 — HypergraphConv ABC](2026-05-14-gomb-soma-phase1.md)

## 1. Summary

Built `WalkConvLayer`, the first concrete `HypergraphConv` subclass —
an open-walk hypergraph convolution that is sign-branched and
position-aware. This is the bottom rung of the GömbSoma compositional
ladder; polygons and triangles follow in phases 3-4.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/walk_layer.py](../signedkan_wip/src/hymeko_gomb/soma/walk_layer.py) | 132 | `WalkConvLayer` — sign-branched, position-aware HG-conv |
| [signedkan_wip/src/hymeko_gomb/soma/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/__init__.py) | +2 / -2 | re-export `WalkConvLayer` |
| [signedkan_wip/tests/test_gomb_soma_walk_layer.py](../signedkan_wip/tests/test_gomb_soma_walk_layer.py) | 222 | 8 unit tests including SBM smoke |

No edits to Gömb / HypergraphConv ABC / CORE.YAML.

## 3. CORE.YAML items touched

None.

## 4. Architecture

For a walk $c = (v_0, v_1, \ldots, v_{k-1})$ with sign $\pi(c) \in \{-1, +1\}$:

$$
\mathrm{msg}(c) = \mathrm{GELU}\!\left( \sum_{i=0}^{k-1} W^{\pi(c)}_i \, x_{v_i} + b^{\pi(c)} \right)
$$

* $W^{\pm}_i \in \mathbb{R}^{\text{in} \times \text{out}}$ — sign-branched, position-aware weights.
* $b^{\pm} \in \mathbb{R}^{\text{out}}$ — sign-branched biases.
* GELU activation, identical on both branches; the per-branch separation is in the weights.

Aggregation: default sum-pool via $M_v$, inherited from `HypergraphConv._aggregate`.

Parameter count at $k = 3, d_\text{in} = d_\text{out} = 16$: $2 \cdot 3 \cdot 16 \cdot 16 + 2 \cdot 16 = 1\,568$. Light enough to stack many layers.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_walk_layer.py -v
test_construction_and_param_count                         PASSED
test_forward_shape                                        PASSED
test_sign_branching_actually_branches                     PASSED
test_position_awareness                                   PASSED
test_permutation_equivariance                             PASSED
test_gradient_flow_on_every_position_and_branch           PASSED
test_no_sign_branching_mode                               PASSED
test_sbm_smoke                                            PASSED
============================ 8 passed in 2.05s ============================
```

What each test pins:

1. **Param count** — $2 \cdot k_\text{arity} \cdot \text{in} \cdot \text{out} + 2 \cdot \text{out}$, no off-by-one.
2. **Forward shape** — output is $(n_\text{nodes}, \text{out})$.
3. **Sign branching actually branches** — flipping every walk's sign moves the output by $\geq 10^{-3}$. Guards against a degenerate weight collapse where positive and negative banks become identical.
4. **Position awareness** — reversing each walk's vertex order changes the output. Walks are directed; this is intentional for the sensorimotor stack (time-ordered sensor sequences). A position-blind walk would be an aggregator, not a convolution.
5. **Permutation equivariance** — vertex permutation passes through; inherited from `HypergraphConv` and confirmed at the concrete layer.
6. **Gradient flow on every position and branch** — backward populates non-zero gradients on every slice of $W^{\pm}_i$ for $i = 0, \ldots, k-1$ and both sign branches. Guards against dead-branch parameters.
7. **`use_sign_branching=False` mode** — the layer halves its parameter count and produces sign-invariant output. Documented as a fallback, not the canonical mode.
8. **SBM smoke** — synthetic two-community signed graph with $n=40$ vertices and 50 random 3-walks; the layer produces a non-degenerate embedding (std $> 0.01$, no NaNs).

## 6. Performance

CPU-only tests, 2.05 s wall. No GPU touched. Layer is light enough
that performance contracts apply at the hierarchy level (phase 6),
not at the per-layer level.

## 7. Numerical stability

The forward path is:
1. `x[primitives]` — gather, FP32 stable.
2. `torch.einsum("nki,nkij->nkj", ...)` — standard contraction.
3. `+= bias` — additive.
4. `F.gelu` — bounded, stable.

No catastrophic cancellation paths. Permutation-equivariance test
passes at atol $10^{-5}$ under FP32 — comfortable margin.

## 8. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/walk_layer.py
   (clean)
```

No new suppressions.

## 9. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — `WalkConvLayer` is one class; sign-branching is a config flag |
| 2 | Algorithm code behind Python boundary | NO — pure PyTorch, no PyO3 boundary touched |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 132 LOC |
| 5 | New axis = new function name | NO — variants are subclasses (PolygonConv, TriangleConv to come) |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — uses `HypergraphConvConfig` dataclass |
| 8 | Forward-time flags for structural variants | The `use_sign_branching` flag is parametric (same architecture, different active params), not structural. ✓ |
| 9 | Bypassing strategy traits | NO — implements the strategy trait |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals / module-level mutable state | NO |

## 10. Phase 2 acceptance

- [x] `WalkConvLayer(HypergraphConv)` implements `_forward_messages`.
- [x] Sign-branched: independent $W^{\pm}_i$ per sign per position.
- [x] Position-aware: reversal changes the output.
- [x] 8 unit tests pass.
- [x] SBM smoke produces non-degenerate output.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 11. Next phase

**Phase 3: PolygonConvLayer.** Hypergraph convolution over closed
cycles $c_k$ for $k \geq 4$. The closure constraint (cycle vs walk)
will be the key structural difference; we'll handle it by:

* enforcing rotational invariance over the cycle's vertex order
  (a polygon has no canonical "start" — unlike a walk);
* exposing a cycle-length parameter that the layer can mix over
  $\{4, 5, 6\}$.

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/polygon_layer.py`
* `signedkan_wip/tests/test_gomb_soma_polygon_layer.py`
* SBM smoke with planted polygons.

No phase 3 work in this commit, per the one-phase-per-session rule.

## 12. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConvConfig, WalkConvLayer,
)

cfg = HypergraphConvConfig(in_features=16, out_features=16, k_arity=3,
                            use_sign_branching=True)
layer = WalkConvLayer(cfg)
y = layer(x, walks, signs, M_v)
```

No new dependencies.

---

*End of phase 2 report.*
