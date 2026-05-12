# k-Cycle Vision Detection (kCVD) / HyMeYOLO vs YOLO — comparable object detection via signed-cycle aggregation — 2026-05-09

> **Update 2026-05-09 (post-context-import):** the user has substantial
> prior work in this theme that this plan must build on, not duplicate.
> See HyMeYOLO design (`notes/hymeyolo_design.md` in HyMeKoConv repo),
> IMPLEMENTATION_PLAN Phase 4, and the validated toy results below.
> The plan-as-written below was originally scoped before that context
> was imported.  The right consolidation is in §"Convergence with
> HyMeYOLO" at the end.



The signed-cycle inductive bias that wins HSiKAN on Slashdot also fits the vision-detection problem: a triangulated keypoint graph over an image *induces* candidate detection regions (faces / k-cycles) directly from the image's own structure, rather than imposing a fixed grid of YOLO-style anchors.  This plan extends the existing kCVD v1 scaffold (`signedkan_wip/src/vision/kcycle_detection.py`, 2026-05-07) into a real detection pipeline benchmarked against YOLO on PASCAL VOC and a COCO subset, with the structural-Kolmogorov--Arnold $\alpha_\kappa$ routing as the per-image-class interpretability handle.

The goal is the strongest applied paper claim from this session's work: **k-cycle detection reaches YOLO-comparable mAP at competitive parameter budget on standard object-detection benchmarks, with a fundamentally different inductive bias (faces from image structure, not anchor grids).**

## Goal

Establish kCVD as a competitive object-detection architecture with three claims:

1. **Architecture-natural fit**: image regions are k-cycles in a Delaunay triangulation over keypoints, not anchor boxes.  No retrofitting required — the cycle enumeration, σ parity, α-mixer, and Highway-attention machinery from the SMC paper apply directly.
2. **Competitive on standard benchmarks**: PASCAL VOC mAP within 5 points of YOLOv8-nano at matched parameter count.  At least one COCO subset (small-object regime) where kCVD wins.
3. **Per-class structural readout**: $\alpha_\kappa$ converges to different distributions across object classes (cars vs people vs polygonal objects), giving a categorical handle that YOLO's anchor mechanism can't.

## Why this is venue-grade novel

Object detection is dominated by anchor-grid (YOLO, SSD, RetinaNet) or anchor-free / DETR-style (DETR, Deformable DETR) architectures.  Both impose a fixed scaffold (regular grid of anchors, or N learnable queries) and learn the *content* placed on it.  None of them induce detection candidates from the image's own structural primitives.

kCVD's contribution thread:

- **Detection candidates from the image structure**: Delaunay triangulation over SLIC superpixels or learned keypoints gives faces; each face IS a candidate region.  No anchors, no NMS (faces are non-overlapping by construction).
- **Multi-scale via $\alpha_\kappa$ mixing**: small triangles (k=3) cover fine detail, larger 5-cycles cover coarser objects; $\alpha$ learns the per-dataset scale distribution.  Replaces FPN's hand-designed multi-scale hierarchy with a learnable scale weighting.
- **Sparse Hamilton attention on signed cycles**: the same attention head that beats SGT on Slashdot, applied to per-face cycle-vertex aggregation.  Sign comes from the local intensity gradient (above/below mean) — natural for vision, no retrofit.
- **Per-face classification**: each k-cycle emits a class probability + (optionally) per-vertex polygon-vertex regression for non-rectangular bounding shapes.  Sparse output (one prediction per face), no NMS pass.

Most-similar prior art: graph-based detection (HyperGraph Object Detection, etc.) but those use static graphs over object proposals from external proposal networks.  kCVD's graph IS the detection scaffold.

## Existing scaffold (2026-05-07)

`signedkan_wip/src/vision/kcycle_detection.py` (~280 LOC) ships:

- `make_image_graph(H, W, n_keypoints)` → perturbed-grid keypoints → Delaunay triangulation
- `vertex_features(img, graph)` → 5-channel per-keypoint patch features (mean RGB, gradient mag/angle)
- `KCycleDetector` — per-face HSiKAN-style aggregation + per-face classification
- `make_synthetic_polygons(n, H, W)` — synthetic dataset generator
- `smoke()` — end-to-end training + assertion

**v1 result** (2026-05-07): pipeline trains, model collapses to bg (8.7:1 imbalance), fg_acc → 0.

**v2.1 result** (2026-05-07): focal loss + 1:1 neg:pos resampling + $\alpha[\text{bg}]=0.1$ → fg_acc 0.27 → 0.46 peak at epoch 23 on synthetic polygons.  Pipeline confirmed alive; v2 levers (more keypoints, CNN features, higher arity) outstanding.

This plan picks up at v2 and pushes through to YOLO-comparable detection on real images.

## Architecture (target state)

```
image M (H × W × 3)
  ↓  keypoint detection
keypoints K = {kp_1, ..., kp_n}
  ↓  Delaunay triangulation in 2D
planar graph G = (V, E_Delaunay, signs from gradient direction)
  ↓  per-vertex features: frozen ResNet patch embedding (16-32 dim)
                       + raw 5-channel patch features
  ↓  HSiKAN encoder with K = {c_3, c_4, c_5, c_6}, h ∈ {16, 32}
                       + Highway-quat sparse attention
                       + per-arity α-mixer
  ↓  per-face classification head
per-face class probabilities
  ↓  thresholding / argmax
detection set
```

Key extensions over v1:

- **Keypoint detection**: replace perturbed-grid with a real keypoint detector (SuperPoint or ORB) for content-aware vertex placement
- **Frozen ResNet patch features**: a ResNet18-tiny backbone provides 32-dim per-keypoint features, frozen during HSiKAN training; this gives kCVD access to learned visual representations without retraining the backbone
- **Higher arity**: k=4 (quad-faces from joined triangles), k=5, k=6 (longer polygonal regions) — implemented natively by the SMC HSiKAN cycle enumerator
- **Highway-quat attention**: same sparse Hamilton-product attention as the SMC paper, scoring per-edge per-face relevance
- **Per-face polygon regression head** (optional v3): predicts per-vertex offsets to refine the bounding polygon

## Datasets

| dataset | type | size | metric |
|---|---|---|---|
| Synthetic polygons (existing) | smoke | 32 × 256² | fg_acc |
| Synthetic polygons (extended) | training | 5000 × 256² | mAP |
| PASCAL VOC 2007 | standard detection | 9963 train + 4952 test | mAP@0.5 |
| COCO val (small-object subset) | hard regime | ~5000 images | mAP@0.5 |

Pre-existing v1 scaffold handles synthetic polygons.  PASCAL VOC and COCO need: standard image loaders, ground-truth correspondence (face label = closest GT box's class via IoU > 0.5), eval harness via `pycocotools`.

## Experiments

### V1 — Synthetic polygons at scale

Extend the existing scaffold to 5000 training images (vs 32 in the smoke), 600 keypoints (vs 200), frozen ResNet18 features (vs 5-channel patch).  Train kCVD with Highway-quat attention + α-entropy aux loss.

**Baseline**: a small YOLOv8-nano (~3M params) trained on the same synthetic polygons.

**Acceptance**: kCVD mAP@0.5 within 10 points of YOLOv8-nano.  Confirms the architecture trains on real-scale image data.

### V2 — PASCAL VOC 2007

Standard 20-class object detection.  Train kCVD with the V1-validated configuration on VOC2007 train, evaluate on VOC2007 test.

**Baseline**: YOLOv8-nano at matched param count (~3M).

**Acceptance**: kCVD mAP@0.5 within 5 points of YOLOv8-nano.  This is a real claim; VOC is a settled benchmark.  Strong claim: kCVD ≥ YOLOv8-nano on at least one size-stratified metric (small / medium / large).

### V3 — COCO small-object subset

Filter COCO val to images dominated by small objects (objects < 32² pixels).  Train on a paired training subset.  This is the regime where YOLO's fixed-grid anchors struggle most — kCVD's content-aware Delaunay triangulation should help most here.

**Acceptance**: kCVD mAP@0.5 ≥ YOLOv8-nano on the small-object subset.

### V4 — Multi-scale α routing readout

For each test image, dump the converged $\alpha_\kappa$ values per detection.  Group by class.

**Hypothesis**: $\alpha$ shifts systematically per class — small fine-detail classes (e.g., bird, mouse) prefer small triangles ($k{=}3$); large coarse classes (e.g., car, sofa) prefer larger polygonal cycles ($k{=}5$, $k{=}6$).

**Acceptance**: Pearson correlation between mean object size per class and dominant $k$-arity per class > 0.5.  Confirms the routing is structurally interpretable.

### V5 — Sign-ablation control

Run V2 with all face signs forced to $+1$ (unsigned variant).  Tests whether the gradient-direction signed structure adds anything specifically to vision.

**Acceptance**: signed kCVD > unsigned kCVD by > 1 mAP point on VOC2007.

### V6 — Comparison against DETR-style baselines

DETR / Deformable DETR are anchor-free transformers.  kCVD is anchor-free hypergraph-conv.  Same niche, different inductive bias.  Compare on VOC2007 + COCO small-object subset.

**Acceptance**: kCVD within 5 mAP points of Deformable DETR at iso-param.  Differentiates kCVD from "another transformer detector."

## Implementation notes

Building on `signedkan_wip/src/vision/kcycle_detection.py`:

- Add `signedkan_wip/src/vision/keypoint_detector.py` (~150 LOC):
  - SuperPoint wrapper (or ORB fallback if no PyTorch SuperPoint available)
  - Returns (n_kp, 2) keypoint coordinates + (n_kp, descriptor_dim) features
- Add `signedkan_wip/src/vision/resnet_features.py` (~100 LOC):
  - Frozen ResNet18 backbone
  - Per-keypoint patch crop + forward + descriptor
- Extend `kcycle_detection.py` with:
  - Higher-arity cycle enumeration (call existing Rust `enumerate_k_cycles_rs`)
  - Highway-quat attention for per-face aggregation (re-use existing `_QuaternionAttentionM_e`)
  - α-entropy aux loss hook
- New runners:
  - `signedkan_wip/src/vision/run_voc_train.py` (~250 LOC): VOC dataloader, training loop, eval
  - `signedkan_wip/src/vision/run_coco_small.py` (~200 LOC): COCO subset training + eval
  - `signedkan_wip/src/vision/run_alpha_class_correlation.py` (~150 LOC): V4 routing readout
- Optional dependency: `torchvision`, `pycocotools` (both standard)
- Total: ~1000 LOC new code on top of v1

## Cost

| experiment | wall time | seeds |
|---|---|---|
| V1 synthetic at scale | ~4 hr | 3 |
| V2 PASCAL VOC | ~12-24 hr | 3 |
| V3 COCO small subset | ~12-24 hr | 3 |
| V4 α-class correlation | ~1 hr | reuse V2 |
| V5 sign ablation | ~12-24 hr | 3 |
| V6 vs DETR | ~24 hr | 3 |

Total: ~5-7 GPU-days for full sweep.  Code: ~2 weeks for implementation + benchmark wiring.  Paper draft: ~1 week.  All-in: ~4 weeks for a CVPR-shape submission.

## Risk register

| risk | probability | mitigation |
|---|---|---|
| kCVD doesn't reach YOLO-comparable mAP at scale | high | scope claim to "different inductive bias, comparable on small-object regime" not "outperforms YOLO universally" |
| Delaunay triangulation produces faces that don't align with object boundaries | medium | use SLIC superpixels (boundary-aligned) instead of generic keypoints |
| ResNet patch features dominate the architectural contribution (HSiKAN's α-routing irrelevant) | medium | ablate by training with frozen-CNN-only baseline; if HSiKAN doesn't lift, that's a real result |
| Cycle enumeration on 600-keypoint Delaunay graphs is slow | low | use existing Rust DFS enumerator from SMC paper; α-routing readout doesn't need cycle batching at this scale |
| Implementation effort overruns 2 weeks | high | hard cutoff at 4 weeks; if VOC2007 V2 not landed, scope to "synthetic + small-COCO" only |
| Compute budget for full COCO is too high | high | restrict to COCO val subset; full COCO training is out of scope |

## Acceptance for the plan as a whole

- **Tier 1** (workshop): V1 synthetic-scale lift over the v1 scaffold, V4 α-class-correlation > 0.5.  CVPR / ICCV workshop.
- **Tier 2** (CVPR / ICCV): V2 PASCAL VOC mAP within 5 points of YOLOv8-nano, V5 sign-ablation > 1 mAP, V4 α-class-correlation > 0.5.  Real applied paper.
- **Tier 3** (best paper / unique-niche claim): V3 COCO small-object regime where kCVD ≥ YOLOv8-nano.  Definitive niche.

## What this plan does NOT do

- Doesn't claim universal YOLO-replacement.  Win condition is *competitive at iso-param + interpretable routing* — different from "beats YOLO on COCO leaderboard."
- Doesn't tackle real-time inference.  YOLO ecosystem has decade-tuned inference pipelines; kCVD's cycle-enumeration step is a one-time cost per image but not optimised for real-time deployment.
- Doesn't propose new attention mechanisms — re-uses Highway-quat from the SMC paper.
- Doesn't address tracking, segmentation, or 3D detection.  Bounding-box / polygon classification only.

## Order of operations

1. **Promote v1 scaffold to v2** — implement focal loss + 1:1 resampling fix permanently, verify on synthetic polygons.  ~1 day.
2. **SuperPoint / ORB keypoint detection** — replace perturbed grid with content-aware keypoints.  ~1 day.
3. **Frozen ResNet18 vertex features** — replace 5-channel patch with 32-dim CNN descriptor.  ~1 day.
4. **Higher arity ($k{\in}\{3,4,5,6\}$)** — wire into HSiKAN encoder via the existing $\mathcal{K}$ axis.  ~half day.
5. **V1 synthetic at scale** — train + measure mAP.  ~1 day GPU.
6. **PASCAL VOC dataloader + eval harness** — `torchvision.datasets.VOCDetection` + mAP@0.5 metric.  ~2 days.
7. **V2 PASCAL VOC training** — ~2 days GPU per seed × 3 seeds = ~6 days GPU.
8. **V4 α-routing readout** — reuse V2 checkpoints.  ~half day.
9. **V5 sign-ablation** — re-train without signs.  ~6 days GPU.
10. **V3 COCO small-object** — `pycocotools` setup + filter + train.  ~6 days GPU.
11. **V6 DETR comparison** — re-train Deformable DETR baseline.  ~6 days GPU.
12. **Paper draft + writeup** — ~1 week.

Total: ~4-5 weeks to a CVPR-shape submission.

## Files this plan will touch when executed

- `signedkan_wip/src/vision/kcycle_detection.py` — extended with Highway attention + α aux + higher arity
- `signedkan_wip/src/vision/keypoint_detector.py` — new
- `signedkan_wip/src/vision/resnet_features.py` — new
- `signedkan_wip/src/vision/run_voc_train.py` — new
- `signedkan_wip/src/vision/run_coco_small.py` — new
- `signedkan_wip/src/vision/run_alpha_class_correlation.py` — new
- `paper/kcvd_vs_yolo/main.tex` — new venue submission directory
- `docs/plans_kcvd_vs_yolo_2026_05_09.md` — this file (close out with results)

## Connection to other plans

- **Mesh matching** (`plans_mesh_matching_2026_05_09.md`) — both treat 2D / 3D structure as signed-cycle hypergraphs.  kCVD on 2D images is the natural sibling of mesh matching on 3D triangulations.
- **Fractal maps** (`plans_fractal_maps_2026_05_09.md`) — IFS-driven cycle generation could replace exhaustive face enumeration on dense keypoint graphs (V2/V3 scaling).  Direct cross-pollination.
- **Time-series + frequency attention** (`plans_hsikan_time_series_2026_05_09.md`) — analogous "structure-induced hypergraph" paradigm; together with kCVD, makes the "HSiKAN as a structural primitive engine" universality argument concrete.
- **Structural-KA theorem** (`plans_structural_ka_theorem_2026_05_09.md`) — V4 α-class-correlation is empirical anchor for the per-task structural readout claim; vision is the most visually-intuitive domain to demonstrate it.
- **Existing kCVD scaffold** (`docs/plans_kcycle_vision_2026_05_07.md`) — direct continuation; v1 + v2.1 results are the starting point.

## Why kCVD vs YOLO is the right framing

YOLO is the canonical reference architecture for object detection; ``vs YOLO'' positions kCVD in a comparable niche.  But the framing is *not* ``YOLO killer'' --- it's ``different inductive bias.'' YOLO imposes a regular anchor grid; kCVD lets the image's own structure determine the grid.  Different bias, different failure modes, different strengths.

The strongest niche for kCVD is the **small-object regime** of COCO, where YOLO's fixed grid anchors perform worst.  Content-aware Delaunay triangulation places small triangles where the image has high keypoint density (typically near object boundaries), naturally giving fine-grained candidates exactly where they're needed.  V3 is the experiment that makes this concrete.

The interpretability angle (V4) is unique: $\alpha$ converging to different distributions per class gives a *categorical compass* over face arities.  YOLO has no analogous interpretation per class.  This is what the structural-Kolmogorov--Arnold framing predicts: per-class routing reflects the per-class inductive structure.

A successful kCVD vs YOLO comparison would be the strongest applied result from the post-SMC research thread.

## Convergence with HyMeYOLO (added 2026-05-09)

The user's prior HyMeKoConv / HyMeYOLO work substantially predates this plan and is the *correct* target for the YOLO-comparable claim.  Existing artifacts:

| artifact | location (in HyMeKoConv repo) | status |
|---|---|---|
| HyMeYOLO toy (single-object) | `examples/hymeyolo_toy.py` | trained 15 epochs, **mIoU 63.3%, class acc 100%, 21k params** |
| HyMeYOLO Hungarian (multi-object) | `examples/hymeyolo_hungarian.py` | trained 12 epochs, **mIoU 59.2%, recall 100%, cls acc 89.5%** |
| HyMeKoConv layer | `src/hymekoconv/torch_layer.py` + fast variant | **5/5 parity tests at <1e-10** |
| CNN equivalence proof | `examples/conv2d_equivalence.py` | machine precision (2.665e-15) |
| Polygon classification | `examples/polygon_classification.py` | 70.8% (5-way regular polygons) |
| Platonic-solid recognition | `examples/polyhedron_recognition.py` | **66.7% (5-class, 8-d pooled fingerprint)** |
| Math report | `notes/hymekoconv_math.pdf` (5 pp) | shipped |
| Implementation plan | `IMPLEMENTATION_PLAN.md` | Phase 4 = VOC, Phase 5 = Triton |

The HyMeYOLO design is **already** the object-as-hyperedge alternative to YOLO's grid head.  The kCVD vision scaffold in this repo (`signedkan_wip/src/vision/kcycle_detection.py`) was a parallel, smaller-scope exploration --- the right move is to **consolidate** the two: take HSiKAN's $\alpha$-mixer + Highway-quat sparse-attention contributions from the SMC paper / 2026-05-08 SOTA-beating work, and plug them into the HyMeYOLO head as new $\Theta_\tau$ types and per-query attention scoring.

### Updated execution path (replaces §"Order of operations" above)

1. **Bring HyMeKoConv repo into the workflow.** Either as a sibling Python package import or via a small adapter so HyMeYOLO can call into HSiKAN's `MixedAritySignedKAN` layer.
2. **Replace HyMeYOLO's per-query HyMeKoConv aggregation with the HSiKAN $\alpha$-mixer + Highway-quat attention head.**  Per-query queries become per-query *cycle slots* with their own $\alpha_\kappa$.  This is the structural-KA framing applied to detection: $\alpha$ tells you per-image-class which polytope-arity carries the signal.
3. **Phase 4 of IMPLEMENTATION_PLAN as-is**: PASCAL VOC at YOLOv5-small param budget, mAP@0.5 $\geq 0.55$ acceptance.  GPU work, ~10 days.
4. **Phase 5 Triton kernel as-is**: fuse the HSiKAN encoder forward + HyMeKoConv head scatter into one Triton kernel.  GPU work, ~1 week.
5. **Multi-primitive head** (bbox + mask + keypoint sharing corners) --- the genuine CVPR-grade contribution per the user's existing design.  Adding $\Theta_\tau$ types per primitive, ~3 days code.

### What's actually novel (re-framed)

The kCVD-from-Delaunay-keypoints framing in this plan's §3 was an alternative scaffold --- *content-aware* face proposals from the image's keypoint structure.  HyMeYOLO's framing is *query-based* --- N learnable cardinality-4 hyperedges per image, sampled bilinearly from the backbone.  These are **complementary, not redundant**:

- HyMeYOLO-Q (existing): query-based, DETR-style, cardinality-fixed at 4.
- HyMeYOLO-G (sketched in `hymeyolo_design.md` Appendix): grid-based, with learnable expansion of cardinality per cell.
- HyMeYOLO-K (this plan's kCVD scaffold reborn): keypoint-graph-based, faces of the Delaunay triangulation as detection candidates with variable cardinality $k \in \{3,4,5,6\}$ via the HSiKAN $\alpha$-mixer.

A hybrid HyMeYOLO-G $\to$ HyMeYOLO-K $\to$ HyMeYOLO-Q (proposer / refiner / verifier) is the natural endpoint and the strongest paper claim.

### What this plan now does NOT do

- Does **not** rewrite HyMeYOLO from scratch.  The existing toy + Hungarian + math + design is the foundation.
- Does **not** treat kCVD-from-Delaunay as the headline.  HyMeYOLO-Q is.  kCVD becomes a *third* hypergraph-wiring option (HyMeYOLO-K) within the same architectural family.
- Does **not** re-derive CNN equivalence or per-corner coherence --- both are already in `notes/`.
- Does **not** touch the inference-benchmark Triton story --- handled by Phase 5.

### Updated paper claim

The right paper claim is *not* "kCVD competes with YOLO" --- it's:

> **HyMeYOLO is a unified object-detection architecture in which YOLO grids, DETR queries, and Delaunay-keypoint faces are three instantiations of the same operator (HyMeKoConv) on three different hypergraph wirings.  The $\alpha_\kappa$ mixer of HSiKAN routes signal across these wirings with a per-class structural readout.  Combined with multi-primitive corner sharing (bbox + mask + keypoint), this gives the first detection architecture in which inter-primitive coherence is hard-coded by hypergraph structure rather than enforced by loss.**

This is the CVPR-grade claim once Phase 4 + Phase 5 + multi-primitive head land.  The kCVD scaffold I built today (smoke at fg_acc 0.46) is the HyMeYOLO-K seed within that framing.

### Cost estimate update

| step | wall | notes |
|---|---|---|
| 1. Adapter HyMeKoConv ↔ HSiKAN | ~2 days | Python package linkage |
| 2. HSiKAN α-mixer + Highway in HyMeYOLO head | ~3 days | replace `Theta_tau` aggregation with HSiKAN encoder call |
| 3. Phase 4 VOC training | ~10 days GPU | per IMPLEMENTATION_PLAN |
| 4. Phase 5 Triton kernel | ~7 days GPU | per IMPLEMENTATION_PLAN |
| 5. Multi-primitive head | ~3 days | typed $\Theta_\tau$ extension |
| 6. CVPR draft | ~2 weeks | full paper |

Total: ~5-6 weeks for a CVPR submission --- consistent with the IMPLEMENTATION_PLAN's "5 weeks of focused work" estimate, with the SMC paper consolidation now removed (already submitted).

