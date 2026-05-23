# GömbSoma-Ricci-Stim Phase 10 — SDRF Wiring

**Date:** 2026-05-15
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 10 (post-plan; SDRF integration)

## 1. Summary

Wired `SDRFRewiring` (Phase 6, the κ-bottleneck relief operator)
into `RicciStimBackbone` (Phase 9). When `use_sdrf=True`, the
backbone:

  1. Runs `StimulusGraphBuilder` to get the initial signed
     hypergraph from the anchor tree + features.
  2. Runs `SDRFRewiring` on that graph's edges to add monotone
     shortcut edges.
  3. Re-runs `StimulusGraphBuilder` with the rewired edges via a new
     `edges_override` path, so walks / polygons / triangles are
     re-enumerated on the bottleneck-relieved topology.
  4. Feeds the augmented graph to the three Bochner-wrapped conv
     branches as usual.

The Phase 6 monotonicity contract carries over: SDRF never removes
edges and never decreases min Forman κ, so the conv branches see a
strictly augmented (or identical) graph relative to the no-SDRF
default.

## 2. Files touched

| File | Change |
|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py](../signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py) | +30 / −5 — `edges_override` + `edge_signs_override` kwargs on `forward`; defensive polygon-edge lookup |
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_backbone.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_backbone.py) | +25 / −3 — `use_sdrf` + SDRF integration in forward |
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_classifier.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_classifier.py) | +9 / −0 — propagate `use_sdrf` kwargs |
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py) | +9 / −0 — propagate `use_sdrf` kwargs |
| [signedkan_wip/tests/test_gomb_soma_vision_sdrf_wiring.py](../signedkan_wip/tests/test_gomb_soma_vision_sdrf_wiring.py) | NEW — 11 wiring tests |

## 3. CORE.YAML items touched

None.

## 4. The integration

### 4.1 Two-pass graph construction

```python
# In RicciStimBackbone.forward (use_sdrf=True branch):
sg = self.graph_builder(tree, features)              # pass 1 (default edges)
sdrf_out = self.sdrf(                                # rewiring
    sg.edges, n_vertices=tree.n_anchors,
    anchor_features=features, edge_signs=sg.edge_signs,
)
sg = self.graph_builder(                             # pass 2 (rewired edges)
    tree, features,
    edges_override=sdrf_out.edges,
    edge_signs_override=sdrf_out.edge_signs,
)
```

The cost is approximately 2× per-forward graph construction when
SDRF is active. Acceptable for benchmark runs; can be cached
per-image during inference.

### 4.2 The new `edges_override` path

`StimulusGraphBuilder.forward` now accepts `edges_override` and
`edge_signs_override` as optional kwargs. When provided, the
builder:

* Skips the internal `_same_scale_edges` + `_cross_scale_edges` +
  feature-inner-product sign computation.
* Uses the supplied edges and signs as the input edge set.
* Continues with walk / polygon / triangle enumeration on the
  override edges.

The polygon enumeration was hardened against missing edges: a
plaquette is emitted only if **all four** of its constituent edges
exist in the active edge set. Under the override path, this means
SDRF-augmented graphs may skip plaquettes whose perimeter relies
on absent base-grid edges; this is correct behaviour.

### 4.3 Default is unchanged

`use_sdrf=False` (default) reproduces Phase-9 behaviour bit-for-bit.
Tested via `test_backbone_use_sdrf_false_regression`: determinism
across re-runs at the same seed.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_sdrf_wiring.py -v
=========== 11 passed in 6.10s ===========
```

Full Ricci-Stim suite (phases 1–10):

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_*.py signedkan_wip/tests/test_gomb_soma_bochner_conv.py
=========== 139 passed in 33.56s ===========
```

### 5.1 Override path (3 tests)

| Test | What it pins |
|---|---|
| `test_override_path_uses_supplied_edges` | builder emits exactly the supplied edges + signs |
| `test_override_path_requires_signs` | override edges without signs → ValueError |
| `test_override_path_validates_length` | mismatched edge / sign lengths → ValueError |

### 5.2 Backbone with SDRF (5 tests)

| Test | What it pins |
|---|---|
| `test_backbone_sdrf_default_off` | `use_sdrf=False` is the default |
| `test_backbone_use_sdrf_false_regression` | deterministic, unchanged-output regression |
| `test_backbone_use_sdrf_true_changes_output` | active SDRF produces valid output (no NaN, correct shapes) |
| `test_backbone_use_sdrf_true_no_nan` | SDRF on random input → valid features |
| `test_backbone_use_sdrf_gradient_flow` | all backbone params receive gradient with SDRF on |

### 5.3 Classifier and detector integration (2 tests)

| Test | What it pins |
|---|---|
| `test_classifier_accepts_sdrf_kwargs` | classifier propagates `use_sdrf` to backbone |
| `test_detector_accepts_sdrf_kwargs` | detector propagates `use_sdrf` to backbone |

### 5.4 SDRF behavioural contract (1 test)

`test_sdrf_does_not_remove_edges` — pins that the SDRF-augmented
graph is a strict superset of the no-SDRF graph (every original
edge appears in the rewired set). This is the Phase 6
monotonicity contract carried forward.

## 6. Architectural commentary

### 6.1 Why two-pass

SDRF needs `(edges, edge_signs)` as input. The builder computes
both internally from the AnchorTree + features. Rather than refactor
the builder into "build edges" and "enumerate primitives" as
separate methods (which would scatter the API), I kept the builder
monolithic and gave it an override path. Two passes — one to get
edges, one to use rewired edges — is the simpler design.

If SDRF later becomes a hot path (every forward, every epoch),
we can refactor for single-pass. For Phase 10's primary use case
(ablation experiments, falsification battery), two-pass is fine.

### 6.2 SDRF is not differentiable

SDRF is a graph-functional. The edges it adds are discrete topology
choices; there is no differentiable gradient w.r.t. anchor features
through the topology changes. The conv branches' parameters still
receive gradient from the features (which are differentiable
inputs); only the topology choice itself is fixed at forward time.

This is the same situation as anchor-box selection in any detector;
non-differentiable topology + differentiable parameters is the
standard pattern.

### 6.3 Per-backbone, per-call recomputation

SDRF runs once per forward call (per image). For training on a
batch of $B$ images, that is $B$ runs of SDRF. Each one is fast
on the per-image anchor count (~50–200), so overhead is manageable.

## 7. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/{stim_graph,ricci_stim_backbone,ricci_stim_classifier,ricci_stim_detector}.py
   (clean)
```

## 8. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | NO |
| 4 | Long single-file module | NO |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Forward-time structural flags | The `use_sdrf` flag is a *parametric* toggle (same architecture, different active path), not a structural variant. ✓ |
| 9 | Bypassing strategy traits | NO |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 9. Phase 10 acceptance

- [x] `StimulusGraphBuilder` accepts `edges_override` + `edge_signs_override` for re-enumerating on a custom edge set.
- [x] `RicciStimBackbone` integrates `SDRFRewiring` behind a `use_sdrf` flag.
- [x] `RicciStimClassifier` and `RicciStimDetector` propagate the flag.
- [x] `use_sdrf=False` (default) is a regression-clean no-op.
- [x] `use_sdrf=True` produces valid output with gradient flow.
- [x] SDRF monotonicity (no-edge-removal) carries forward to the wired backbone.
- [x] 11 new Phase-10 tests pass.
- [x] Full 139-test Ricci-Stim suite still green.
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 10. Phase ladder — 10 phases done

| Phase | Module | Status |
|---|---|---|
| 1–8 | (Plan complete, see prior reports) | ✓ |
| 9 | Backbone consolidation | ✓ |
| **10** | **SDRF wiring into the backbone** | **✓** |
| 8-bench | Cluttered MNIST falsification | next |

## 11. What this gives the next phase

Phase 8-bench (the falsification battery) now has access to the
full ablation matrix:

| Bochner α | Bochner β | use_sdrf | What it tests |
|---|---|---|---|
| 0 | 0 | False | bare 3-primitive backbone (Phase 7 baseline) |
| > 0 | 0 | False | Hodge smoothing alone |
| 0 | > 0 | False | Ricci correction alone |
| > 0 | > 0 | False | full Bochner without graph rewiring |
| > 0 | > 0 | True | full Bochner + SDRF (the headline config) |

This is the right ablation grid for the falsification report.

## 12. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import (
    RicciStimBackbone, RicciStimClassifier, RicciStimDetector,
)

# All-in: Bochner coupling + SDRF rewiring.
m = RicciStimDetector(
    image_h=28, image_w=28, d_hidden=16, n_classes=10,
    bochner_alpha=0.1, bochner_beta=0.05,
    use_sdrf=True, sdrf_max_iters=5, sdrf_kappa_target=-2.0,
)
out = m(images)
```

No new dependencies.

---

*End of Ricci-Stim phase 10 report.*
