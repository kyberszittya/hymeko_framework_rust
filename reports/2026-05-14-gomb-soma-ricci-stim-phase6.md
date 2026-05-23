# GömbSoma-Ricci-Stim Phase 6 — SDRF Rewiring

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 6 of 8
**Prior phases:** [1 Forman](2026-05-14-gomb-soma-ricci-stim-phase1.md), [2 Quadtree](2026-05-14-gomb-soma-ricci-stim-phase2.md), [3 Hodge](2026-05-14-gomb-soma-ricci-stim-phase3.md), [4 Bochner](2026-05-14-gomb-soma-ricci-stim-phase4.md), [5 StimulusGraph](2026-05-14-gomb-soma-ricci-stim-phase5.md)

## 1. Summary

Built `SDRFRewiring`, a discrete-Ricci-flow edge rewiring procedure
that addresses over-squashing on κ-bottlenecked graphs by iteratively
adding shortcut edges. After Topping et al.\ (NeurIPS 2022).

The monotonicity contract is pinned: every shortcut addition is
verified to *not decrease* the global min Forman κ. Pathological
graphs (paths, stars) where no shortcut can preserve monotonicity
correctly report n_added = 0 — the rewiring degrades gracefully
into a no-op rather than silently corrupting the graph.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py](../signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py) | 218 | `SDRFRewiring` + `SDRFOutput` dataclass |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +6 / -0 | re-exports |
| [signedkan_wip/tests/test_gomb_soma_vision_sdrf.py](../signedkan_wip/tests/test_gomb_soma_vision_sdrf.py) | 215 | 16 tests including butterfly-graph κ-rise |

## 3. CORE.YAML items touched

None.

## 4. The algorithm

```text
while iter < max_iters:
    κ ← Forman κ on current edges
    if min κ ≥ target: stop, converged = True
    for each edge e in ascending κ order:
        for each (a, b) ∈ (N(u_e) \ {v_e}) × (N(v_e) \ {u_e}) where a ≠ b, (a,b) ∉ E:
            tentatively add (a, b)
            new_min_κ ← Forman.min on E ∪ {(a, b)}
            if new_min_κ ≥ current min κ:
                accept best candidate by new_min_κ (tiebreak by sorted pair)
        if found a valid shortcut: add it, update adj
        else: continue to next bottleneck edge
    if no shortcut found across all edges: stop
```

The critical safeguard is the **tentative-add-then-check**: we
recompute the global min κ after each candidate addition, and only
commit if the global min didn't decrease. This converts the simple
SDRF rule into a strictly monotone graph operation.

### 4.1 Why monotonicity needed special care

A naïve "score by common-neighbour count" heuristic adds shortcuts
that *raise the new edge's own κ* but can *lower the κ of other
incident edges* (because degrees of $a$ and $b$ increase by 1).
On a pure path, no candidate creates a new triangle (no common
neighbours anywhere), so every shortcut lowers some edge's κ
without any compensating triangle gain. The first SDRF
implementation in this phase exhibited exactly this regression
on `P_8`: min κ dropped from $-2$ to $-5$. The tentative-add check
fixed it.

### 4.2 Determinism

The candidate search iterates over sorted neighbour lists,
tiebreaks by sorted pair index, and stops at the first edge whose
neighbourhood yields a valid shortcut. Same input ⇒ same output.
Pinned by `test_determinism`.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_sdrf.py -v
=========== 16 passed in 3.80s ===========
```

### 5.1 Monotonicity contract

| Test | What it pins |
|---|---|
| `test_min_kappa_never_decreases_path` | `kappa_min_after ≥ kappa_min_before` on a path graph (where no shortcut may exist) |
| `test_path_P5_monotonic` | P_5 monotonicity |
| `test_star_S5_monotonic` | S_5 monotonicity (n_added = 0 expected) |

These are the central pin: regardless of input graph, SDRF cannot worsen the bottleneck.

### 5.2 Demonstrable improvement

`test_butterfly_kappa_rises` — the butterfly graph (two triangles
sharing a hub vertex 0; 4 edges incident to the hub at κ = −2)
admits SDRF shortcuts between the two triangles' leaves. The test
verifies that at least one shortcut is added (`n_added > 0`) AND
the min κ strictly rises (`kappa_min_after > kappa_min_before`).

### 5.3 Caps and termination

| Test | What it pins |
|---|---|
| `test_max_iters_respected` | `n_added ≤ max_iters` |
| `test_zero_iters_no_change` | `max_iters=0` → graph unchanged |
| `test_convergence_flag_correct` | `converged ↔ kappa_min_after ≥ target` |
| `test_already_converged_K3_no_rewiring` | K_3 has κ = 0 everywhere → no rewiring |

### 5.4 Signs

| Test | What it pins |
|---|---|
| `test_original_signs_preserved_when_no_features` | original signs verbatim; new shortcuts default to +1 |
| `test_new_shortcut_signs_from_features` | feature-inner-product sign rule for new shortcuts |

### 5.5 Robustness

| Test | What it pins |
|---|---|
| `test_determinism` | reproducible |
| `test_output_is_sdrf_output` | dataclass shape |
| `test_edges_superset_of_input` | only adds, never removes |
| `test_rejects_wrong_edge_shape` | input validation |
| `test_rejects_mismatched_signs` | sign-length consistency |

### 5.6 Integration

`test_integration_with_stimulus_graph` — `AdaptiveQuadtree → StimulusGraphBuilder → SDRFRewiring` end-to-end on a 32×32 random image. Verifies the full Phase 2 → 5 → 6 pipeline holds together.

## 6. Honest deviation from Topping et al.

Topping's original SDRF uses **Ollivier**-Ricci curvature (optimal-transport based) and a **stochastic** Boltzmann tiebreak among candidates. Our implementation uses:

* Forman-Ricci κ (combinatorial, cheaper, integer-arithmetic-stable).
* Deterministic tiebreak (sorted pair index) instead of Boltzmann sampling.
* Tentative-add monotonicity check (an extra safeguard not in the original).

The differences are documented in the source docstring. The Forman variant is what fits in our compute budget per training step; if we ever need Ollivier we can plug it in via a subclass.

## 7. Architectural use downstream

`SDRFRewiring` is a one-shot preprocessor applied to the
`StimulusGraph.edges` before the GömbSoma layer stack runs. In the
end-to-end pipeline:

```text
AdaptiveQuadtree(image)         # Phase 2
   ↓
StimulusGraphBuilder            # Phase 5
   ↓
SDRFRewiring                    # Phase 6 — this
   ↓
Walk / Polygon / Triangle       # Phases 2 / 3-G / 4
   (Bochner-wrapped)
   ↓
classification / detection head
```

The rewired edge set lifts the κ-bottleneck floor, alleviating
over-squashing in the message passing. Phase 4's Bochner-coupled
HypergraphConv consumes the rewired graph via `prepare()`.

## 8. Performance

3.80 s for all 16 tests on CPU. Per-iteration cost is dominated by
the tentative-add + Forman recompute, which is O(|candidates| × |E|).
At Cluttered MNIST scale (~50–200 anchors, ~200–500 edges, ~20
candidates per bottleneck), each iteration is sub-second on CPU.

## 9. Numerical stability

All arithmetic is on integer-valued Forman κ outputs. The tentative-add
check uses `float(min().item())` — FP32 conversion of integer values
is exact. The `1e-9` slack in `best_min_after = current_min - 1e-9`
is a defensive margin for FP comparison; in practice κ values are
exact integers and the slack never bites.

## 10. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py
   (clean)
```

No new suppressions.

## 11. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — single rewiring class |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO — 218 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — numeric thresholds |
| 8 | Forward-time structural flags | NO |
| 9 | Bypassing strategy traits | N/A |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 12. Phase 6 acceptance

- [x] `SDRFRewiring` adds shortcut edges to lift κ-bottlenecks.
- [x] Monotonicity contract pinned: min κ never decreases.
- [x] Demonstrable improvement on butterfly graph (κ rises strictly).
- [x] Graceful no-op on pathological graphs (paths, stars).
- [x] Deterministic.
- [x] Signs computed for new shortcuts from features.
- [x] 16 unit tests pass.
- [x] Integration with `StimulusGraphBuilder` smoke-tested.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 13. Phase ladder status (6 of 8 phases complete)

| Phase | Status | What it gave us |
|---|---|---|
| 1 Forman | ✓ | per-edge combinatorial Ricci κ |
| 2 AdaptiveQuadtree | ✓ | content-driven multi-scale anchors |
| 3 HodgeLaplacian | ✓ | $\partial_k$, $\Delta_k$ with $\partial \partial = 0$ pinned |
| 4 BochnerHypergraphConv | ✓ | 3-term flat + Hodge + Ricci message passing |
| 5 StimulusGraphBuilder | ✓ | signed hypergraph from AnchorTree + features |
| **6 SDRFRewiring** | **✓** | **monotone κ-bottleneck relief** |
| 7 MNIST classifier | — | first hard number on the full pipeline |
| 8 Cluttered MNIST detector + falsification | — | target ≥ 0.72 mAP50 |

## 14. Next phase

**Phase 7: end-to-end MNIST classifier.** Assemble the full pipeline
into a working image classifier:

* Image → AdaptiveQuadtree → AnchorTree (Phase 2)
* Per-anchor patch encoder (Linear) → features
* StimulusGraphBuilder → StimulusGraph (Phase 5)
* SDRFRewiring → rewired edges (Phase 6)
* BochnerHypergraphConv(WalkConv) + BochnerHypergraphConv(PolygonConv) + BochnerHypergraphConv(TriangleConv) (Phases 1, 3-G, 4)
* Aggregate → Linear classifier head

Adds:
* `signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_classifier.py`
* `signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_classifier.py`
* Smoke train on MNIST

No phase 7 work in this commit, per the one-phase-per-session rule.

## 15. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import SDRFRewiring

sdrf = SDRFRewiring(max_iters=10, min_kappa_target=-2.0)
out = sdrf(
    edges=stim_graph.edges,
    n_vertices=anchor_tree.n_anchors,
    anchor_features=features,
    edge_signs=stim_graph.edge_signs,
)
# out.edges: rewired edge list (original + shortcuts).
# out.edge_signs: signs for all edges.
# out.kappa_min_before, out.kappa_min_after: bottleneck before / after.
# out.n_added: count of shortcuts.
# out.converged: True if min κ reached the target.
```

No new dependencies.

---

*End of Ricci-Stim phase 6 report.*
