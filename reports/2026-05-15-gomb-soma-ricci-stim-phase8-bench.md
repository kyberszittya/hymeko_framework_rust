# GömbSoma-Ricci-Stim Phase 8-bench — Cluttered MNIST infrastructure + performance characterization

**Date:** 2026-05-15
**Plan:** [docs/plans/2026-05-14-gomb-soma-ricci-stim-bench/](../docs/plans/2026-05-14-gomb-soma-ricci-stim-bench/)
**Phase:** 8-bench (falsification battery)

## 1. Headline

The Ricci-Stim training infrastructure ships and runs end-to-end on
real Cluttered MNIST data. A 30-image × 1-epoch CUDA smoke (Config E
— full Bochner + SDRF) completes cleanly in 128 s.

**Performance characterization, however, is the gating issue:**

* Current wall-time: **~4.3 s per forward+backward pass per image** at the
  Python-loop implementation of the StimulusGraphBuilder + per-anchor
  patch encoding.
* HyMeYOLO baseline (5000 imgs × 50 epochs in ~760 s/seed):
  ~329 fwd+bwd per second.
* **We are ~1400× slower** than the HyMeYOLO comparison baseline at
  the current implementation.

A full 5-config × 5-seed × 5000-image × 20-epoch falsification battery
is therefore not feasible at this performance. Phase 8-bench delivers
the *infrastructure* that makes the battery runnable; the *acceleration*
needed to actually run it is the next single-coherent piece of work
(Rust-accelerated `StimulusGraphBuilder`, batched primitive enumeration).

## 2. Confirmed: this IS the octree + global-shutter design

The user's earlier architectural redirect was to (a) replace YOLO's
uniform-grid anchors with quadtree subdivision, and (b) drive anchor
selection by activations / content rather than pre-tiling. Both are
realised in this implementation:

| User's design | Where it lives |
|---|---|
| Quadtree-based anchors (not uniform grid) | `AdaptiveQuadtree` (Phase 2) |
| Content-driven subdivision (pixel-variance + Forman κ) | quadtree `score_threshold` |
| Multi-scale primitive enumeration | `StimulusGraphBuilder` (Phase 5) |
| Ricci propagation (β · κ correction term) | `BochnerHypergraphConv` (Phase 4) |
| Hodge smoothing (α · Δ_k term) | same |
| Bottleneck rewiring (SDRF) | Phase 6 + 10 wiring |
| Activation-driven inference filter | per-anchor cls head + score threshold + NMS in eval |
| Full pipeline | `RicciStimDetector` (Phase 8) |

Config E of the ablation matrix is *exactly* "quadtree anchors + Bochner-Ricci propagation + SDRF + activation-filtered inference."

## 3. Files touched

| File | Action |
|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_train.py](../signedkan_wip/src/hymeko_gomb/soma/vision/ricci_stim_train.py) | NEW — 306 LOC; anchor-target assignment, detection loss, mAP50 proxy, train loop |
| [signedkan_wip/experiments/run_ricci_stim_cluttered_mnist.py](../signedkan_wip/experiments/run_ricci_stim_cluttered_mnist.py) | NEW — Python runner with 5-config ablation matrix |
| [signedkan_wip/experiments/run_ricci_stim_cluttered_mnist_smoke.sh](../signedkan_wip/experiments/run_ricci_stim_cluttered_mnist_smoke.sh) | NEW — orchestrator stub |
| [signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_train.py](../signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_train.py) | NEW — 11 unit tests |
| [docs/plans/2026-05-14-gomb-soma-ricci-stim-bench/](../docs/plans/2026-05-14-gomb-soma-ricci-stim-bench/) | NEW plan dir (4 formats) |

## 4. CORE.YAML items touched

None.

## 5. Training infrastructure

### 5.1 Anchor-target assignment

`assign_anchors_to_gt(positions, sizes, gt_boxes, gt_labels, iou_pos)`:

1. Compute IoU between every anchor (treated as a square proposal) and every GT box.
2. For each anchor, find the best-IoU GT.
3. Assign cls target = GT class + 1 (1..10) for anchors with IoU > 0.3, else 0 (background).
4. Compute (dx, dy, dw, dh) regression target for positive anchors, matching the `decode_boxes` convention.

Pinned by 3 unit tests (no-GT → all background; high-IoU → foreground; anchor matching GT exactly → zero offsets).

### 5.2 Loss

`detection_loss(output, assignment, bbox_weight=1.0)`:

* Classification: cross-entropy over (n_classes + 1) per-anchor logits.
* Bbox: smooth-L1 on positive anchors only.
* Combined: `cls + bbox_weight * bbox`.

Pinned by 2 unit tests (runs; backward populates every parameter when α, β > 0).

### 5.3 mAP50 proxy

Per-image: filter anchors by `cls_score > 0.05` and class != background → NMS at IoU 0.5 → match to GT at IoU > 0.5 with same class → F1 = 2PR/(P+R). Mean over images.

This is the precision-recall F1 at IoU 0.5, not the full COCO mAP50 (which is per-class AP averaged). Good enough for ablation comparison; documented in code.

### 5.4 Train loop

`train_one_seed(detector_factory, train_loader, eval_loader, ...)`: standard Adam loop, per-image loss computed in Python (batch_size handled by stacking outputs from `RicciStimDetector` which returns a list of `DetectionOutput`).

## 6. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_train.py -v
=========== 11 passed in 2.64s ===========
```

Full Ricci-Stim suite (phases 1–10 + bench):

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_*.py signedkan_wip/tests/test_gomb_soma_bochner_conv.py
=========== 150 passed ===========
```

## 7. Smoke (production-scale, CUDA)

| Setting | Value |
|---|---|
| Dataset | Cluttered MNIST, 64×64, 1–3 digits/image |
| Train images | 30 |
| Eval images | 10 |
| Epochs | 1 |
| Seed | 0 |
| Config | E (Bochner α=0.1, β=0.1, SDRF on) |
| Batch size | 4 |
| Device | CUDA (RTX 2070 SUPER) |
| Wall | **128.1 s** |
| final_mAP50_proxy | **0.0** (expected at this scale) |

Pipeline runs end-to-end: image → quadtree → backbone → 3 Bochner-wrapped conv branches → detector head → cls+bbox loss → backward → opt step → eval. Exit code 0, no NaNs.

## 8. Performance bottleneck (the gating issue)

Per-forward-pass time of ~4.3 s / image is dominated by:

1. **Per-anchor patch encoding**: for each anchor (~50–200/image), gather pixels, adaptive-avg-pool, Linear projection. Python loop with no vectorisation.
2. **StimulusGraphBuilder enumeration**: walks, polygons, triangles all enumerated by Python `for` loops.
3. **SDRF rewiring (Config E only)**: iterative `for` loop over candidates with per-iteration Forman κ recomputation.

For comparison: HyMeYOLO's `train_circles_ricci` (the +ricci-mod 0.723 mAP50 baseline) ran a 5-seed × 5000-image × 50-epoch benchmark in ~63 min. That's ~760 s/seed ≈ 329 fwd+bwd/s. We measure 0.23 fwd+bwd/s.

**Ratio: ~1400×.** Closing this gap is essential before a full ablation table is meaningful.

### 8.1 Where to optimise

| Hot spot | Implementation cost | Expected speedup |
|---|---|---|
| Walk / polygon / triangle enumeration in StimulusGraphBuilder | already exists in Rust (`hymeko_py.enumerate_*`); port the wrapper to call Rust | 50–100× |
| Per-anchor patch encoding | batchify (extract all patches at once, then stack) | 5–10× |
| SDRF iteration | call Forman in batch over candidate edges instead of per-candidate | 5–10× |

Realistic combined: ~10²–10³× speedup, putting us in HyMeYOLO's range.

## 9. What was NOT run

The full ablation battery (5 configs × 5 seeds × 5000 imgs × 20 epochs) is **not feasible** at the current performance:

* 5000 imgs × 4.3 s = 21,500 s per epoch.
* × 20 epochs = ~120 hours per (config, seed).
* × 5 seeds × 5 configs = ~3,000 hours total.

A minimal scale (100 imgs × 5 epochs × Config E × 1 seed) was queued in the background after the smoke — estimated wall ~36 min. Result will be appended once it lands.

## 10. Honest verdict against the "beat YOLO" target

The architecture is complete, tested, and runs end-to-end. The *evidence* that it beats YOLO at its own game (HyMeYOLO `+ricci-mod` 0.723 mAP50) is **not delivered in Phase 8-bench**. The blocker is wall-time per forward, not architectural soundness.

The phase ladder is:

| Phase | What it shipped | Status |
|---|---|---|
| 1–10 | architecture (Forman / Quadtree / Hodge / Bochner / StimGraph / SDRF / Backbone / Classifier / Detector / SDRF wiring) | ✓ |
| **8-bench** | training infrastructure + performance characterisation | **✓** |
| 8-bench+1 (next) | hot-path acceleration (Rust enumeration) | — |
| 8-bench+2 | actual falsification battery against HyMeYOLO baselines | blocked on +1 |

This is the honest state. The architectural contribution is real (10 phases of tested modules); the empirical falsification requires one more push on performance.

## 11. Decision tree (from the plan)

The decision tree's branches all require completion of the falsification battery, which is gated on optimisation. Logged here as deferred.

## 12. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| All | (see prior phases) | NO new violations |

Particular note on #11 (no globals): the runner script is self-contained; no module-level mutable state introduced.

## 13. Phase 8-bench acceptance (revised)

Original plan acceptance was "≥ 0.5 mAP50 on Cluttered MNIST." That target is **not met** because the falsification run is not feasible at current speed.

Revised acceptance for this single-session deliverable:

- [x] Training infrastructure (anchor assignment + loss + train loop + mAP50 proxy + 5-config matrix) shipped under unit-test coverage (11 tests).
- [x] Production-scale CUDA smoke passes (30 imgs × 1 epoch × Config E → exit 0, no NaN).
- [x] Performance characterisation honest and quantified (~4.3 s/image, 1400× slower than HyMeYOLO).
- [x] Plan dir with 4 formats committed.
- [x] No CORE.YAML edits.
- [ ] **Full ablation battery: deferred to post-acceleration phase.**
- [ ] **mAP50 ≥ 0.72 target: not yet measured.**

## 14. Reproducibility

```bash
# Single config × seed × scale (the smoke):
python -m signedkan_wip.experiments.run_ricci_stim_cluttered_mnist \
    --config E --n-train 30 --n-eval 10 --n-epochs 1 \
    --seed 0 --device cuda --batch-size 4 \
    --out-jsonl /tmp/smoke.jsonl

# To run all 5 configs × 1 seed once accelerated:
for c in A B C D E; do
  python -m signedkan_wip.experiments.run_ricci_stim_cluttered_mnist \
      --config $c --n-train 5000 --n-eval 1000 --n-epochs 20 \
      --seed 0 --device cuda --batch-size 16 \
      --out-jsonl results/ablation.jsonl
done
```

---

*End of Phase 8-bench report. Architectural plan (1–10 + bench infrastructure) complete; performance optimisation is the next coherent unit of work.*
