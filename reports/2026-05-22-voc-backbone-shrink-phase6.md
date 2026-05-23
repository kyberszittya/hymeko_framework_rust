# Stage D-3-BREAK Phase 6 — backbone shrink

**Date:** 2026-05-22
**Plan:** `signedkan_wip/experiments/run_voc_backbone_shrink_2026_05_22.sh`
**Verdict:** **Backbone-shrink HURTS mAP at this hardware budget; ResNet18-IN at 224 px is the right backbone for the C9 ladder. The input-resolution axis is structurally blocked by data-preload OOM (not model OOM); the next architectural lever is a lazy-data-loader refactor of `train_voc_stagec.py`.**

## 1. Per-cell table

Sorted by mAP$_{50}$:

| cell | backbone | input | bs | ep | ckpt | **mAP$_{50}$** | mIoU | cls_acc | loss_end | wall |
|-|-|-|-|-|-|-|-|-|-|-|
| B6 | **resnet18_imagenet** | 224 | 8 | 90 | false | **0.0833** | 0.305 | 0.750 | 2.630 | 30 min |
| B1 | resnet (custom, 107k) | 224 | 8 | 90 | false | 0.0340 | 0.276 | 0.625 | 3.623 | 51 min |
| B4 | hsikan (CR, 136k) | 224 | 8 | 90 | true | 0.0241 | 0.287 | 0.625 | 3.688 | **337 min** |
| B2 | resnet | 320 | 8 | 90 | false | **OOM** at data preload | — | — | — | 2 min |
| B3 | resnet | 320 | 16 | 90 | false | **OOM** | — | — | — | 2 min |
| B5 | hsikan | 320 | 4 | 90 | true | **OOM** | — | — | — | 1 min |

C9 5-seed anchor: mAP$_{50}$ = 0.0790 ± 0.0105, band [0.0685, 0.0895].

## 2. Three findings

### 2.1 ResNet18-IN at 224 px is the right backbone (B6 = 0.0833)

B6 lands at **0.0833**, **inside** the C9 5-seed band [0.0685, 0.0895]. This validates the C9 recipe — the seed-0 of C9 reproduces the 5-seed mean within noise. The 5-seed wasn't a fluke.

### 2.2 Backbone-shrink falsified (B1, B4 both >2× worse than B6)

| comparison | Δ mAP | relative |
|-|-|-|
| B1 (tiny resnet 107k) − B6 | −0.049 | **−59%** |
| B4 (hsikan 136k + ckpt) − B6 | −0.059 | **−71%** |

Tiny from-scratch backbones cannot replace ResNet18-IN's ImageNet-pretrained weights at this dataset size + training budget (5011 images × 90 epochs).

**Interesting hsikan note:** B4's mIoU (0.287) is the highest of the three trained cells, suggesting the Catmull-Rom basis primitive does learn useful geometric features — but per-class confidence is still weak (cls_acc 0.625 vs B6's 0.750), so the overall mAP suffers. Stage D-3-quinquies measured 0.0043 at 30 ep; B4's 0.0241 at 90 ep is a 5.6× improvement, confirming hsikan was under-trained in that earlier probe — but still 3.5× below the ResNet18-IN reference.

### 2.3 Input-resolution OOM is data-preload, not model

B2, B3, B5 all OOM in the **first line of `main()`**:

```python
File ".../train_voc_stagec.py", line 171, in main
  X = torch.from_numpy(Xn).to(device)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.74 GiB.
```

The trainer pre-loads the **entire VOC2007 trainval set as a single GPU tensor**: `5011 × 3 × 320² × 4 bytes = 6.16 GiB`. The 8 GiB GPU cannot hold that plus the model + activations. **Backbone-shrink cannot fix this** — the model never gets to allocate weights because the dataset preload itself OOMs.

## 3. What this means for the C9 ladder

The within-trainer config space (backbone, input resolution, batch size at current loader) is **mapped**. The C9 recipe (ResNet18-IN, 224 px, batch 8, 6 queries, 90 ep, lam_gate_neg=2.0, nodelet head, bce gate) is the saturation point of the current trainer.

To push past 0.0790 mAP requires either:

1. **Refactor `train_voc_stagec.py` to use a lazy/batched DataLoader** (~half day's work). Unblocks 320 px (and 416 px with grad-checkpoint) — the actual input-resolution axis.
2. **Better pretraining for the from-scratch backbones** (hsikan-CR with ImageNet pretrain) — if we want to use the family-paper-purity backbone competitively.
3. **Multi-scale FPN** — but that's already enabled (fpn=2level); no further headroom on that axis without bigger backbone activations.
4. **Bigger GPU** — outside this hardware budget.

## 4. Recommendation

**Implement option 1 (lazy DataLoader) as the next experimental session.** It's the only architectural lever that genuinely unblocks the input-resolution axis on the 8 GiB GPU. Estimated cost:
- Refactor: ~3-4 hours engineering (`Dataset` + `DataLoader` wrapper around the existing per-image NumPy load)
- Validation: existing C9 recipe at 224 px under the new loader (should reproduce 0.0790)
- New probe: C9 recipe at 320 px (the cell we couldn't run before)

If the lazy-loader refactor unblocks 320 px AND the result lifts mAP, that's the next entry in the recipe ladder. If it lifts mAP but only marginally, input-resolution is also exhausted and the next axis is backbone-replacement at higher capacity.

## 5. Files

| file | role |
|-|-|
| `signedkan_wip/experiments/run_voc_backbone_shrink_2026_05_22.sh` | orchestrator |
| `signedkan_wip/experiments/results/voc_backbone_shrink_20260522T014614Z/` | per-cell logs + jsonls + grid.jsonl + orchestrator.log |
| `reports/2026-05-22-voc-backbone-shrink-phase6.md` | this file |

No code changes (only orchestration); CORE.YAML items touched = 0.

## 6. Acceptance check

- [x] 6 cells launched; 3 completed, 3 OOM'd (documented).
- [x] B6 reproduces C9 seed-0 within the 5-seed band (sanity of the C9 baseline).
- [x] B1 + B4 both confirm backbone-shrink hurts mAP by 2-3×.
- [x] B2 + B3 + B5 OOM root-caused to data-preload (Tier-2 finding, not Phase-6 deliverable).
- [x] Verdict + next lever (lazy DataLoader refactor) explicitly stated.
- [x] No CORE.YAML edits.
- [x] Report on disk.

## 7. Experiment provenance

- **Git SHA:** 507d7e24 (uncommitted; same SHA as the C9 ladder).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **Wall:** B1 51 min, B4 5h 37min, B6 30 min (= the productive cells); OOM cells ~1-2 min each. Total ~7 h.
- **Memory cap:** `systemd-run --user --scope -p MemoryMax=16G`.
- **Seeds:** single seed (seed=0) per cell.
- **Dataset:** VOC2007 trainval (5011 images, 12085 GTs, 20 classes).
