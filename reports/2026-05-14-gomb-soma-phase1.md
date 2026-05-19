# GömbSoma Phase 1 — HypergraphConv ABC

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma/](../docs/plans/2026-05-14-gomb-soma/)
**Phase:** 1 of 7

## 1. Summary

Built the shared message-passing primitive that every GömbSoma layer
(WalkConv, PolygonConv, TriangleConv, AbstractionConv) will subclass.
No concrete layers shipped yet — Phase 1 is the ABC contract only,
per the one-phase-per-session rule (CLAUDE.md §6.5 anti-pattern #11).

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/__init__.py) | 27 | package boundary, re-exports |
| [signedkan_wip/src/hymeko_gomb/soma/hg_conv.py](../signedkan_wip/src/hymeko_gomb/soma/hg_conv.py) | 235 | `HypergraphConv` ABC + `HypergraphConvConfig` dataclass |
| [signedkan_wip/tests/test_gomb_soma_hg_conv.py](../signedkan_wip/tests/test_gomb_soma_hg_conv.py) | 218 | 11 unit tests including permutation-equivariance |

No edits to Gömb files. No edits to CORE.YAML-protected items.

## 3. CORE.YAML items touched

None.

## 4. Contract sealed by the ABC

```python
class HypergraphConv(nn.Module, abc.ABC):
    def forward(
        self,
        x: Tensor[n_nodes, in_features],
        primitives: Tensor[n_prim, k_arity],
        primitive_signs: Tensor[n_prim],   # values in {-1, +1}
        M_v: SparseTensor[n_nodes, n_prim],
    ) -> Tensor[n_nodes, out_features]:
        ...
```

Subclass contract:
- `_forward_messages(x, primitives, primitive_signs) -> Tensor[n_prim, out_features]`
  — layer-specific message function (required).
- `_aggregate(messages, M_v) -> Tensor[n_nodes, out_features]`
  — defaults to sparse sum-pool; override to add normalization or
  routing.

The `forward` method is sealed: it validates preconditions, calls
the subclass hook, calls the aggregator, then checks postconditions.
Subclasses cannot override `forward` itself — they fill in the
message function and (optionally) the aggregator.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_hg_conv.py -v
test_abc_cannot_be_instantiated                       PASSED
test_config_validates_dimensions                      PASSED
test_concrete_subclass_runs                           PASSED
test_precondition_rejects_wrong_x_shape               PASSED
test_precondition_rejects_wrong_primitive_arity       PASSED
test_precondition_rejects_wrong_sign_values           PASSED
test_precondition_rejects_mismatched_M_v_shape        PASSED
test_permutation_equivariance                         PASSED
test_sparse_aggregator_does_not_materialise_dense     PASSED
test_gradient_flow                                    PASSED
test_extra_repr_documents_config                      PASSED
============================ 11 passed in 2.08s ============================
```

Coverage of the eleven tests:

1. ABC unconstructible — `pytest.raises(TypeError)`.
2. Config validates positive `in_features`, positive `out_features`, `k_arity ≥ 2`.
3. Minimal concrete subclass (`MeanConv`) runs and returns the documented shape.
4. Precondition: rejects mis-sized `x`.
5. Precondition: rejects mis-arity primitives.
6. Precondition: rejects sign values outside {-1, +1}.
7. Precondition: rejects mis-shaped `M_v`.
8. **Permutation equivariance** (the GömbSoma headline invariant): for any vertex permutation π, `forward(π(input))` equals `π(forward(input))` within atol 1e-5.
9. Sparse aggregator parity: `torch.sparse.mm(M_v, messages)` matches the manually densified `M_v.to_dense() @ messages` within atol 1e-6.
10. Gradient flow: `loss.backward()` populates non-zero gradients on every learnable parameter — no dead branches.
11. `repr()` documents the config dimensions.

## 6. Performance

ABC + a 235-LOC module + 218-LOC test file is microscopic — no
performance contract applies at this layer. Test wall time: 2.08 s
on the dev box. No GPU memory used (CPU tests).

## 7. Numerical stability

The ABC itself does no floating-point arithmetic. The fixture
`MeanConv` uses one `nn.Linear` and one `mean` — no catastrophic
cancellation risk. The permutation-equivariance test passes at
atol 1e-5, well within FP32 round-off.

## 8. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/
   (clean)
```

No new `# type: ignore`, `# noqa`, or `# NOLINT` introduced. No
clippy-equivalent waivers.

## 9. §6.5 anti-pattern review

Reviewed against the eleven anti-patterns:

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API surface | NO — single `HypergraphConv` class, config-driven |
| 2 | Algorithm code behind a Python boundary | NO — this is the contract, not an algorithm |
| 3 | Per-experiment scaffold duplication | NO — phase 1 has no runner |
| 4 | Long single-file modules | NO — 235 LOC under the 400-LOC heuristic |
| 5 | New axis = new function name | NO — `HypergraphConv` is the family entry; layer types subclass |
| 6 | `#[allow(...)]` band-aid | N/A (Python) |
| 7 | String-typed config | NO — `HypergraphConvConfig` uses proper dataclass with typed fields |
| 8 | Forward-time flags for structural variants | Mixed — `use_sign_branching` is a parametric flag, acceptable; structural variants (Walk vs Polygon vs Triangle) are class-per-variant per the plan |
| 9 | Bypassing existing strategy traits | NO — this IS the strategy trait |
| 10 | `ulimit -v` on CUDA | N/A (no GPU use) |
| 11 | Global variables / module-level mutable state | NO — no globals introduced |

## 10. Phase 1 acceptance

Phase 1 ships when:

- [x] `HypergraphConv` ABC implemented with sealed forward, abstract `_forward_messages`, default `_aggregate`.
- [x] `HypergraphConvConfig` dataclass with `validate()`.
- [x] At least one concrete subclass (test fixture `MeanConv`) exercises the contract end-to-end.
- [x] 11 unit tests pass.
- [x] Permutation equivariance verified numerically.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 11. Next phase

**Phase 2: WalkConv layer.** Hypergraph convolution over open walks
$w_k$ for $k = 2, 3, \ldots$, signed by the σ-product of their
constituent edges. Adds:

- `signedkan_wip/src/hymeko_gomb/soma/walk_layer.py`
- `signedkan_wip/tests/test_gomb_soma_walk_layer.py`
- synthetic SBM smoke

No phase 2 work in this commit, per the rule.

## 12. Reproducibility

```bash
# Run the tests:
python -m pytest signedkan_wip/tests/test_gomb_soma_hg_conv.py -v

# Use the ABC in a downstream layer:
from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConv, HypergraphConvConfig,
)

class MyLayer(HypergraphConv):
    def _forward_messages(self, x, primitives, primitive_signs):
        ...  # return (n_prim, out_features)
```

No new dependencies. Pure Python + PyTorch.

---

*End of phase 1 report.*
