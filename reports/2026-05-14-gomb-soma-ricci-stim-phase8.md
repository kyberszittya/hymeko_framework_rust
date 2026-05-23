# GömbSoma-Ricci-Stim Phase 8 — RicciStimDetector

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim/](../docs/plans/2026-05-14-gomb-soma-ricci-stim/)
**Phase:** 8 of 8 — **final phase**
**Prior phases:** [1](2026-05-14-gomb-soma-ricci-stim-phase1.md), [2](2026-05-14-gomb-soma-ricci-stim-phase2.md), [3](2026-05-14-gomb-soma-ricci-stim-phase3.md), [4](2026-05-14-gomb-soma-ricci-stim-phase4.md), [5](2026-05-14-gomb-soma-ricci-stim-phase5.md), [6](2026-05-14-gomb-soma-ricci-stim-phase6.md), [7](2026-05-14-gomb-soma-ricci-stim-phase7.md)

## 1. Summary

Built `RicciStimDetector`, the per-anchor object detector that closes
the 8-phase Ricci-Stim plan. Same backbone as Phase 7's classifier
(quadtree → encoder → StimulusGraph → 3 Bochner-wrapped layers), but
the head produces per-anchor outputs:

  * class logits (n_classes + 1, including background),
  * bounding-box offsets (dx, dy, dw, dh).

The output `DetectionOutput` dataclass carries everything a
downstream NMS / mAP evaluator needs: per-anchor logits, per-anchor
offsets, anchor positions, anchor sizes. A static `decode_boxes`
utility converts offsets to absolute (cx, cy, w, h) form using
standard YOLO-style encoding.

The training-signal contract is pinned: `test_overfit_single_image_cls_and_bbox`
drives 300 Adam steps on a single image with one foreground anchor
assignment and reaches the correct class prediction plus bbox
convergence within tolerance.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py) | 250 | `RicciStimDetector` + `DetectionOutput` |
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | +5 / -0 | re-export |
| [signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_detector.py](../signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_detector.py) | 244 | 11 tests |

## 3. CORE.YAML items touched

None.

## 4. Architecture

```
image
  ↓  AdaptiveQuadtree → AnchorTree
  ↓  per-anchor adaptive-avg-pool + Linear (shared encoder)
features ∈ ℝ^(n × d_hidden)
  ↓  StimulusGraphBuilder → StimulusGraph
  ↓  Walk + Polygon + "Triangle" branches (Bochner-wrapped), summed
h ∈ ℝ^(n × d_hidden)
  ↓  cls_head: Linear(d_hidden → n_classes + 1)   per-anchor cls
  ↓  bbox_head: Linear(d_hidden → 4)              per-anchor offsets
DetectionOutput
```

The shared backbone (everything before the heads) is *identical* to
`RicciStimClassifier`. The only structural difference is replacing
`mean(dim=0) + Linear(d_hidden, n_classes)` with two parallel
per-anchor heads.

### 4.1 Box encoding

Offsets are decoded against each anchor's reference frame:

$$
c_x = r_a + s_a / 2 + dx \cdot s_a, \quad
c_y = c_a + s_a / 2 + dy \cdot s_a, \quad
w = s_a \cdot e^{dw}, \quad
h = s_a \cdot e^{dh}
$$

where $(r_a, c_a, s_a)$ are the anchor's position and side length.
Standard YOLO-style encoding, applied to multi-scale,
content-determined anchors instead of YOLO's pre-tiled fixed grid.

### 4.2 Per-image return

Because the per-image anchor count varies (the quadtree adapts to
content), the detector returns `DetectionOutput` per single image
and `list[DetectionOutput]` per batch. The downstream training
loop matches anchors to ground-truth boxes per image (the standard
detection pipeline), so the variable-anchor structure poses no
problem at the loss-computation level.

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_detector.py -v
=========== 11 passed in 7.21s ===========
```

### 5.1 The training-signal contract

**`test_overfit_single_image_cls_and_bbox`** — drive 300 Adam steps
on a single random image with one foreground assignment (anchor 0,
class 1, bbox offsets [0.1, -0.1, 0.05, -0.05]). The detector
converges to the correct class AND the bbox within tolerance.

Verifies end-to-end signal flow through both heads simultaneously.

### 5.2 Head-isolation contract

**`test_cls_head_grad_only_from_cls_loss`** — the cls head receives
gradient from cross-entropy on cls_logits, but NOT from L1 on
bbox_offsets. This pins the two-head topology: losses don't
cross-contaminate the heads (only the shared backbone does).

A bug that accidentally couples the heads (e.g., adding cls into
bbox or vice versa) would fail this test.

### 5.3 Decoding correctness

| Test | What it pins |
|---|---|
| `test_decode_boxes_zero_offsets_returns_anchor_centers_and_sizes` | (dx,dy,dw,dh) = 0 decodes to the anchor itself |
| `test_decode_boxes_nonzero_offsets_shifts_center` | (dx, dy) shifts the centre by `(dx*s, dy*s)`; (dw, dh) scales by exp |

### 5.4 Robustness

| Test | What it pins |
|---|---|
| `test_construction` | bounded param count |
| `test_rejects_bad_image_shape` | input validation |
| `test_single_image_returns_DetectionOutput` | correct dataclass + shapes |
| `test_batch_returns_list_of_outputs` | batch returns list |
| `test_gradient_flow_combined_loss` | combined loss covers every param (with α, β > 0) |
| `test_uniform_image_runs` | constant image → no NaN |
| `test_n_anchors_varies_per_image` | quadtree adapts; batch handles it |

## 6. The phase ladder is complete

| Phase | Module | What it gave us |
|---|---|---|
| 1 | `FormanCurvatureHead` | combinatorial Ricci κ |
| 2 | `AdaptiveQuadtree` | content-driven multi-scale anchors |
| 3 | `HodgeLaplacian` | $\partial_k$, $\Delta_k$, $\partial \partial = 0$ pinned |
| 4 | `BochnerHypergraphConv` | 3-term flat + Hodge + Ricci message passing |
| 5 | `StimulusGraphBuilder` | signed hypergraph from AnchorTree + features |
| 6 | `SDRFRewiring` | monotone κ-bottleneck relief |
| 7 | `RicciStimClassifier` | end-to-end image classifier |
| **8** | **`RicciStimDetector`** | **end-to-end object detector** |

Every phase shipped a working module under unit-test coverage. The
falsification battery (Cluttered MNIST, target ≥ 0.72 mAP50 vs
HyMeYOLO `+ricci-mod`) is the next milestone but lives in a
**separate Phase 8-bench** because the actual training run is hours
of GPU time, not a single-session task.

## 7. What this phase does NOT yet do

- **No actual Cluttered MNIST training run.** The architecture
  ships; the falsification benchmark is Phase 8-bench (separate
  session, ~3 hours of GPU time per the plan).
- **No anchor-target assignment (IoU-based)** — needed for training
  on real data. Standard implementation; ~150 LOC of glue code
  expected in Phase 8-bench.
- **No NMS** — output is raw per-anchor predictions; NMS belongs
  to the evaluation pipeline.
- **SDRF rewiring still not wired into the detector** (same
  deferral as Phase 7).

## 8. Performance

7.2 s for all 11 tests. The expensive ones are
`test_overfit_single_image_cls_and_bbox` (300 forward+backward
passes) and the multi-image quadtree tests.

Per-image forward at MNIST defaults: similar to Phase 7's
classifier (~15 ms CPU); the per-anchor heads add negligible
compute over the pooled head.

## 9. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_detector.py
   (clean)
```

No new suppressions.

## 10. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO |
| 2 | Algorithm code behind Python boundary | NO |
| 3 | Per-experiment scaffold duplication | The backbone duplicates `RicciStimClassifier`'s. Acceptable in this phase because phase ladder rule is one-phase-per-session; refactoring both into a shared `RicciStimBackbone` is a follow-up cleanup. Documented. |
| 4 | Long single-file module | NO — 250 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO |
| 8 | Forward-time structural flags | NO |
| 9 | Bypassing strategy traits | NO |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

The judgement call: backbone duplication between classifier and
detector. The right cleanup is a shared `RicciStimBackbone`
module they both consume. Deferred to a follow-up consolidation
phase rather than mixed into Phase 8 (which would violate
one-phase-per-session). Anti-pattern #3 partially applies but is
acceptable temporarily.

## 11. Phase 8 acceptance

- [x] `RicciStimDetector` with per-anchor cls + bbox heads.
- [x] `DetectionOutput` dataclass carries decode reference frame.
- [x] `decode_boxes` static utility for downstream NMS.
- [x] **Overfit-single-image (cls + bbox) passes** (signal-flow contract).
- [x] Head-isolation pinned (losses don't cross-contaminate).
- [x] Bochner α / β coupling exercises gradient through projections.
- [x] 11 unit tests pass.
- [x] No CORE.YAML edits.
- [x] One §6.5 anti-pattern flagged with rationale; no new ones.

All acceptance criteria met. **Ricci-Stim 8-phase plan is now complete.**

## 12. What ships after 8 phases of Ricci-Stim

A working, tested, end-to-end GömbSoma-Ricci-Stim vision stack:

- 8 modules + 4 dataclasses, all under unit-test coverage.
- ~2 500 LOC of source + ~2 200 LOC of tests across 8 phases.
- 8 phase reports + the original 4-format plan dir
  (`docs/plans/2026-05-14-gomb-soma-ricci-stim/`).
- A differential-geometry primer documenting the theoretical
  framework (`docs/differential-geometry-primer.tex`).
- The full pipeline runs end-to-end and trains: image → classification
  + detection, with α / β coupling exposed for ablation.

The architectural contribution — Forman κ-driven adaptive quadtree
anchors + Bochner-coupled message passing on a signed simplicial
complex — is now a working artefact, not just a plan. The
falsification milestone (≥ 0.72 mAP50 on Cluttered MNIST vs
HyMeYOLO `+ricci-mod` 0.723) requires an actual training run
that is the next single coherent piece of work.

## 13. Next phase (post-plan-completion)

Three natural follow-ups, each a single-session phase:

1. **Phase 8-bench:** actual Cluttered MNIST training run +
   falsification battery. The headline experiment.
2. **Phase 9 — backbone consolidation:** refactor
   `RicciStimClassifier` and `RicciStimDetector` to share a
   `RicciStimBackbone` module. Clears anti-pattern #3.
3. **Phase 10 — SDRF wiring into the classifier/detector:**
   re-run `StimulusGraphBuilder` on the SDRF-rewired edge set
   before the conv layers.

User picks.

## 14. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import RicciStimDetector

d = RicciStimDetector(
    image_h=28, image_w=28, patch_size_initial=4,
    patch_size_min=1, max_depth=2,
    in_channels=1, d_hidden=16, n_classes=10,
    bochner_alpha=0.1, bochner_beta=0.05,
)
output = d(images)  # DetectionOutput per image
boxes = RicciStimDetector.decode_boxes(output)  # decoded (cx,cy,w,h)
```

No new dependencies.

---

*End of Ricci-Stim phase 8 report — 8-phase plan complete.*
