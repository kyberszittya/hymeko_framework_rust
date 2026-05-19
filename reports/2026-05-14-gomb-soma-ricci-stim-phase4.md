# GömbSoma-Ricci-Stim Phase 4 — BochnerHypergraphConv

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 4 of 8
**Prior phases:** [Phase 1 — Forman](2026-05-14-gomb-soma-ricci-stim-phase1.md), [Phase 2 — Quadtree](2026-05-14-gomb-soma-ricci-stim-phase2.md), [Phase 3 — Hodge](2026-05-14-gomb-soma-ricci-stim-phase3.md)

## 1. Summary

Built `BochnerHypergraphConv`, the architectural realisation of the
Bochner–Weitzenböck identity on a discrete signed simplicial complex.
The wrapper composes three terms:

$$\text{msg}(c) = \underbrace{\text{msg}_{\text{inner}}(c)}_{\text{flat connection}}
                + \alpha \cdot \underbrace{\text{hodge\_proj}\bigl(\overline{(\Delta_0 x)_c}\bigr)}_{\text{Hodge smoothing}}
                + \beta  \cdot \underbrace{\kappa(c) \cdot \text{ricci\_proj}(\overline{x}_c)}_{\text{Ricci correction}}$$

where $\overline{\cdot}_c$ denotes the per-primitive mean over the
primitive's vertices.

The headline contract: **with α = β = 0 the wrapper's forward output
is bit-identical to the inner HypergraphConv subclass's output**.
This is pinned by `test_alpha_beta_zero_reproduces_inner_exactly` at
`torch.equal` (exact equality, no tolerance). Phase 4 is therefore
strictly additive — turning on Bochner coupling cannot break
existing Phase 1–3-G behaviour.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/hg_conv_bochner.py](../signedkan_wip/src/hymeko_gomb/soma/hg_conv_bochner.py) | 168 | `BochnerHypergraphConv` wrapper |
| [signedkan_wip/src/hymeko_gomb/soma/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/__init__.py) | +4 / -0 | re-export |
| [signedkan_wip/tests/test_gomb_soma_bochner_conv.py](../signedkan_wip/tests/test_gomb_soma_bochner_conv.py) | 263 | 11 tests including the regression contract |

## 3. CORE.YAML items touched

None.

## 4. Architecture

`BochnerHypergraphConv` is a `HypergraphConv` subclass that wraps
an inner `HypergraphConv` (typically a Walk / Polygon / Triangle
layer) and exposes the same `forward(x, primitives, signs, M_v)`
signature — drop-in replacement. Two extra context tensors (Hodge
Laplacian and per-primitive curvatures) are set via the `prepare()`
method before forward:

```python
layer = BochnerHypergraphConv(walk_conv, alpha=0.1, beta=0.05)
layer.prepare(
    hodge_laplacian=delta_0,         # sparse (n_v, n_v)
    primitive_curvatures=kappa_walks, # (n_walks,)
)
y = layer(x, walks, walk_signs, M_v)
```

The `prepare()`-then-`forward()` pattern is the same one used by
FiLM conditioning and attention-mask preparation: external
geometric context set once per forward pass, then consumed by
multiple layers.

### 4.1 The three terms

- **Flat connection**: delegates to `inner._forward_messages(x, primitives, signs)`. With sign-branched Walk/Polygon/Triangle, this is the existing sign-branched message.
- **Hodge smoothing**: $\alpha \cdot \text{hodge\_proj}(\text{mean}_c(\Delta_0 x))$. Applies the Hodge Laplacian as a discrete heat-flow smoothing at the vertex level, then aggregates per primitive, then projects to the output dimension. This is the $\nabla^* \nabla$ term of Bochner–Weitzenböck.
- **Ricci correction**: $\beta \cdot \kappa(c) \cdot \text{ricci\_proj}(\text{mean}_c(x))$. The per-primitive curvature scalar $\kappa(c)$ scales the projected mean of vertex features. This is the $\mathrm{Ric}(\omega^\sharp)^\flat$ term.

### 4.2 Parameter count overhead

Wrapper adds exactly $2 + 2 \cdot (d_\text{in} \cdot d_\text{out} + d_\text{out})$:

- 1 scalar for α (`nn.Parameter` if `learnable_mixing=True`, else buffer)
- 1 scalar for β
- one `Linear(in, out)` for `hodge_proj`
- one `Linear(in, out)` for `ricci_proj`

At $d_\text{in} = d_\text{out} = 8$: overhead = 2 + 2·(64 + 8) = 146 parameters. Tested by `test_param_count_overhead`.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_bochner_conv.py -v
=========== 11 passed in 1.93s ===========
```

### 5.1 The headline contract

**`test_alpha_beta_zero_reproduces_inner_exactly`** —  with α = β = 0,
*and even with non-trivial Hodge Laplacian + curvatures plumbed
through `prepare()`*, the wrapper's output equals the inner layer's
output via `torch.equal` (exact equality). Pinned at FP32 bit-level.

**`test_alpha_beta_zero_no_prepare_call`** — same contract when no
`prepare()` was ever called (the wrapper defaults to flat-connection
only).

These two tests are the central pin: a future change that introduces
a numerical drift at the α=β=0 boundary fails them immediately.

### 5.2 Coupling activity

| Test | What it pins |
|---|---|
| `test_alpha_nonzero_changes_output` | α=0.5 with Δ_0 plumbed → output moves ≥ 1e-3 vs inner |
| `test_beta_nonzero_changes_output` | β=0.5 with curvatures plumbed → output moves ≥ 1e-3 vs inner |
| `test_gradient_flow_all_components` | backward populates non-zero grads on α, β, hodge_proj, ricci_proj, AND inner params |
| `test_works_with_polygon_inner` | wrapper accepts `PolygonConvLayer` as inner; α=β=0 still bit-identical |

### 5.3 Contract preservation

| Test | What it pins |
|---|---|
| `test_forward_signature_same_as_inner` | same 4-arg `forward(x, prim, signs, M_v)` |
| `test_preconditions_inherited` | malformed primitives still rejected |
| `test_output_shape` | output is `(n_nodes, out_features)` |
| `test_param_count_overhead` | overhead is exactly `2 + 2*(in*out + out)` |
| `test_non_learnable_mixing_uses_buffers` | `learnable_mixing=False` reduces param count by 2 |

## 6. Why the wrapper pattern (not subclass / mixin)

Three options were considered. Decision: **subclass `HypergraphConv` and compose an inner `HypergraphConv`** (not pure subclass, not pure mixin).

| Option | Why rejected |
|---|---|
| Subclass each of Walk / Polygon / Triangle separately with Bochner terms baked in | Cartesian-product API anti-pattern (§6.5 #1); 3 classes per inner type |
| Pure wrapper without inheriting `HypergraphConv` | Loses preconditions check, output-shape contract, and the drop-in property |
| Subclass `HypergraphConv` and compose the inner | Inherits the sealed `forward` + preconditions; inner's `_forward_messages` is delegated and extended; one class works with any HypergraphConv inner |

The chosen design uses the sealed `forward` from `HypergraphConv` (so preconditions are checked exactly once), overrides `_forward_messages` to add the Bochner terms, and overrides `_aggregate` to delegate to the inner. The inner is recursively visible to `parameters()`, `train()`, `state_dict()`, etc.

## 7. Numerical stability

The Hodge term is `torch.sparse.mm(Δ_0, x)` followed by index gather, mean, and Linear projection — FP32-stable. The Ricci term is per-primitive elementwise multiplication by κ, also stable. The flat term inherits whatever stability the inner layer has (which we already tested in phases 2 / 3-G).

The α=β=0 regression test passes at `torch.equal` (bit-level exact), which is a stronger guarantee than `torch.allclose` and rules out any subtle FP drift.

## 8. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/hg_conv_bochner.py
   (clean)
```

No new suppressions.

## 9. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — one wrapper handles all inner types |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 168 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — uses inherited `HypergraphConvConfig` |
| 8 | Forward-time structural flags | The `learnable_mixing` flag is parametric (Parameter vs buffer); not a structural variant |
| 9 | Bypassing strategy traits | NO — IS a `HypergraphConv` subclass |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals / module-level mutable state | The `_hodge_laplacian` / `_primitive_curvatures` are instance state set by `prepare()` — not module-level globals, and not concurrency-shared. Acceptable per §6.5 #11's "context object threaded through the call site" pattern. |

## 10. Phase 4 acceptance

- [x] `BochnerHypergraphConv(HypergraphConv)` wraps any inner HypergraphConv.
- [x] Three-term decomposition: flat + α·Hodge + β·Ricci.
- [x] α = β = 0 produces bit-identical output to inner (pinned `torch.equal`).
- [x] Both coupling terms are active when their inputs are plumbed.
- [x] Gradients flow to all wrapper params AND inner params.
- [x] Works with Walk and Polygon inner layers.
- [x] 11 unit tests pass.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced (the one judgement call documented).

All acceptance criteria met.

## 11. Architectural ladder so far

| Phase | Module | What it adds |
|---|---|---|
| 1 | `HypergraphConv` ABC | sealed forward + precondition contract |
| 2 | `WalkConvLayer` | open-walk signed conv, sign-branched, position-aware |
| 3-G | `PolygonConvLayer` | closed-cycle signed conv, cyclic-invariant |
| 3-V | vision patch graph + classifier | rolling-shutter baseline |
| Ricci-Stim 1 | `FormanCurvatureHead` | combinatorial Ricci κ |
| Ricci-Stim 2 | `AdaptiveQuadtree` | content-driven anchor selection |
| Ricci-Stim 3 | `HodgeLaplacian` | $\partial_k$, $\Delta_k$ with $\partial \partial = 0$ pinned |
| **Ricci-Stim 4** | **`BochnerHypergraphConv`** | **3-term flat + Hodge + Ricci composition** |
| Ricci-Stim 5 (next) | `StimulusGraphBuilder` | multi-scale signed hypergraph from `AnchorTree` |
| Ricci-Stim 6 | `SDRF` rewiring | curvature-driven bottleneck removal |
| Ricci-Stim 7 | End-to-end MNIST classifier | first hard number |
| Ricci-Stim 8 | Cluttered MNIST detector + falsification battery | target ≥ 0.72 mAP50 |

## 12. Next phase

**Phase 5: `StimulusGraphBuilder`.** Take an `AnchorTree` (Phase 2)
and an image, produce the multi-scale signed hypergraph that the
Walk / Polygon / Triangle layers consume:

- Same-scale 4-connected edges among siblings at each depth.
- Cross-scale parent-child edges from `AnchorTree.parent_indices`.
- Edge signs from feature-inner-product polarity.
- Walk / polygon / triangle enumeration over the anchors (capped).

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py`
* `signedkan_wip/tests/test_gomb_soma_vision_stim_graph.py`

No phase 5 work in this commit, per the one-phase-per-session rule.

## 13. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma import (
    BochnerHypergraphConv, WalkConvLayer, HypergraphConvConfig,
)
from signedkan_wip.src.hymeko_gomb.soma.vision import HodgeLaplacian

cfg = HypergraphConvConfig(in_features=16, out_features=16, k_arity=3)
inner = WalkConvLayer(cfg)
layer = BochnerHypergraphConv(inner, alpha=0.1, beta=0.05)

hodge = HodgeLaplacian()
ops = hodge(edges, n_vertices=N)
layer.prepare(
    hodge_laplacian=ops.laplacian_0,
    primitive_curvatures=kappa_per_walk,
)
y = layer(x, walks, walk_signs, M_v)
```

No new dependencies.

---

*End of Ricci-Stim phase 4 report.*
