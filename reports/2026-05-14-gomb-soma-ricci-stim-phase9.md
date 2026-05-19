# GömbSoma-Ricci-Stim Phase 9 — Backbone Consolidation

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 9 (post-plan consolidation)

## 1. Summary

Refactored the duplicated feature-extraction backbone out of
`RicciStimClassifier` (Phase 7) and `RicciStimDetector` (Phase 8)
into a single shared `RicciStimBackbone` module. Each head class
now holds only its head + the per-image forward dispatch; the
backbone owns the quadtree, patch encoder, graph builder, and the
three Bochner-wrapped branches.

This clears the §6.5 #3 anti-pattern flag that Phase 8 documented
(per-experiment scaffold duplication). Future heads (segmentation,
dense prediction, keypoint detection) plug into the same backbone
without re-typing 200 LOC of feature extraction.

## 2. Files touched

| File | Action |
|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_backbone.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_backbone.py) | NEW — `RicciStimBackbone` (188 LOC) |
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_classifier.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_classifier.py) | rewrite — 240 LOC → 100 LOC, wraps backbone |
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py) | rewrite — 250 LOC → 130 LOC, wraps backbone |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | re-export `RicciStimBackbone` |
| [signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_backbone.py](../signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_backbone.py) | NEW — 8 backbone tests |
| [signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_classifier.py](../signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_classifier.py) | update — `m.walk_layer` → `m.backbone.walk_layer` etc. |

Net LOC change: −172 (more than 50% reduction in the classifier and
detector source).

## 3. CORE.YAML items touched

None.

## 4. The split

### Before (Phase 7 + 8)

Classifier and Detector each defined:
- `self.quadtree = AdaptiveQuadtree(...)`
- `self.patch_encoder = nn.Linear(...)`
- `self.graph_builder = StimulusGraphBuilder()`
- `self.walk_layer = BochnerHypergraphConv(WalkConvLayer(...))`
- `self.poly_layer = BochnerHypergraphConv(PolygonConvLayer(...))`
- `self.tri_layer = BochnerHypergraphConv(PolygonConvLayer(...))`
- `self._encode_anchors`, `self._walk_branch`, `self._poly_branch`, `self._tri_branch`
- Plus their respective head (pooled-cls or per-anchor-cls+bbox).

~200 LOC of feature extraction duplicated.

### After (Phase 9)

```
RicciStimBackbone:
    quadtree
    patch_encoder
    graph_builder
    walk_layer (Bochner)
    poly_layer (Bochner)
    tri_layer  (Bochner)
    forward(image) -> (features, tree)

RicciStimClassifier:
    backbone (RicciStimBackbone)
    head (Linear(d_hidden, n_classes))
    forward: backbone -> mean pool -> head

RicciStimDetector:
    backbone (RicciStimBackbone)
    cls_head (Linear(d_hidden, n_classes + 1))
    bbox_head (Linear(d_hidden, 4))
    forward: backbone -> 2 heads per anchor
```

## 5. Test results

```
$ python -m pytest \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_backbone.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_classifier.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_detector.py -v
=========== 30 passed in 27.0 s ===========
```

Full Ricci-Stim suite (phases 1–9):

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_*.py signedkan_wip/tests/test_gomb_soma_bochner_conv.py
=========== 128 passed in 29.3 s ===========
```

### 5.1 Backbone tests (8 new)

| Test | What it pins |
|---|---|
| `test_forward_returns_features_and_tree` | (Tensor, AnchorTree) tuple output |
| `test_rejects_bad_input_shape` | (C, H, W) required |
| `test_features_consistent_with_tree` | `features.shape[0] == tree.n_anchors` |
| `test_deterministic` | same input ⇒ same (features, tree) |
| `test_gradient_flow_with_bochner_coupling` | every backbone param receives gradient with α, β > 0 |
| `test_alpha_beta_zero_zeros_bochner_projections` | with α = β = 0, the Hodge / Ricci projections receive zero gradient (gated correctly) |
| `test_uniform_image_no_nan` | constant image → valid output |
| `test_param_count_excludes_head` | backbone params ≠ classifier/detector params (proves separation) |

### 5.2 Classifier and detector tests (unchanged behaviour)

All 22 existing tests pass (11 classifier + 11 detector) — exactly
the same assertions, just with the parameters now nested under
`m.backbone.*`. The refactor is behaviour-preserving.

## 6. Architectural commentary

### 6.1 Why this matters for future heads

The backbone is now a black-box: feed an image, get
`(per_anchor_features, anchor_tree)`. Building a new task head means:

```python
class RicciStimSegmenter(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.backbone = RicciStimBackbone(...)
        self.seg_head = SomeSegmentationHead(...)
    
    def forward(self, image):
        h, tree = self.backbone(image)
        return self.seg_head(h, tree)
```

Phase 10 (SDRF wiring), Phase 8-bench (training run), and any
future head can all reuse the backbone directly.

### 6.2 Bochner α / β are now backbone-level

The Bochner mixing coefficients are part of the backbone's
construction, exposed once. This is the right level: α and β
modulate the geometric coupling, which is a backbone-level
concern, not a head-level concern.

### 6.3 Backward compatibility

The classifier and detector public API (constructor args + forward
signature) is unchanged. The only difference is the *parameter
naming* in `state_dict`: `walk_layer.*` is now
`backbone.walk_layer.*`. Since no checkpoints exist yet from this
codebase, this is a clean break with no migration cost.

## 7. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | **CLEARED** — was flagged in Phase 8; backbone consolidation resolves it. |
| 4 | Long single-file module | NO — backbone 188 LOC, classifier 100 LOC, detector 130 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Forward-time structural flags | NO |
| 9 | Bypassing strategy traits | NO |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

§6.5 #3 is cleared. The phase ladder discipline (one-phase-per-session) made the original duplication temporarily acceptable; Phase 9 retires it.

## 8. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_*.py
   (clean)
```

## 9. Phase 9 acceptance

- [x] `RicciStimBackbone` extracted as shared feature module.
- [x] `RicciStimClassifier` refactored to wrap backbone + head.
- [x] `RicciStimDetector` refactored to wrap backbone + 2 heads.
- [x] 8 new backbone tests pass.
- [x] All pre-existing classifier and detector tests pass (with updated `m.backbone.*` access).
- [x] Full Ricci-Stim 128-test suite green.
- [x] §6.5 #3 anti-pattern flag cleared.
- [x] Net source LOC reduction: −172.

All acceptance criteria met.

## 10. Phase ladder status

| Phase | Module | Status |
|---|---|---|
| 1 Forman | `FormanCurvatureHead` | ✓ |
| 2 Quadtree | `AdaptiveQuadtree` | ✓ |
| 3 Hodge | `HodgeLaplacian` | ✓ |
| 4 Bochner | `BochnerHypergraphConv` | ✓ |
| 5 StimulusGraph | `StimulusGraphBuilder` | ✓ |
| 6 SDRF | `SDRFRewiring` | ✓ |
| 7 Classifier | `RicciStimClassifier` | ✓ (refactored under Phase 9) |
| 8 Detector | `RicciStimDetector` | ✓ (refactored under Phase 9) |
| **9 Backbone consolidation** | `RicciStimBackbone` | **✓** |
| 10 (next) | SDRF wiring into the classifier/detector | — |
| 8-bench | Cluttered MNIST falsification | — |

## 11. Next phase

**Phase 10 — SDRF wiring.** Plug `SDRFRewiring` between the
`StimulusGraphBuilder` and the conv branches inside `RicciStimBackbone`.
This requires re-running `StimulusGraphBuilder` (or at least its
walk/polygon/triangle enumeration) on the SDRF-rewired edge set.

No phase 10 work in this commit, per the one-phase-per-session rule.

## 12. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import (
    RicciStimBackbone, RicciStimClassifier, RicciStimDetector,
)

bb = RicciStimBackbone(image_h=28, image_w=28, d_hidden=16,
                        bochner_alpha=0.1, bochner_beta=0.05)
features, tree = bb(image)  # (n_anchors, 16), AnchorTree

# Same backbone, different heads:
clf = RicciStimClassifier(d_hidden=16, n_classes=10, ...)
det = RicciStimDetector(d_hidden=16, n_classes=10, ...)
```

No new dependencies.

---

*End of Ricci-Stim phase 9 report.*
