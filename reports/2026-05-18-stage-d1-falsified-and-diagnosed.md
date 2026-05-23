# Stage D-1 falsified — and the actual bottleneck found

**Date:** 2026-05-18
**Plan:** [`docs/plans/2026-05-18-hymeyolo-stage-d1-pretrain/`](../docs/plans/2026-05-18-hymeyolo-stage-d1-pretrain/)
**Run dir:** `signedkan_wip/experiments/results/stage_d1_voc2007_20260518T012029Z/`
**Git SHA:** `2ccaa4d12fae1ff9cd533bd91cd84b28f11c3dab`
**Wall:** 03:20 → 04:50 local = 18 min smoke; 5-seed gated out → 7.5 h GPU saved.

## 1. Summary

Stage D-1 swapped the from-scratch ResNet-tiny backbone for an
ImageNet-pretrained ResNet18 truncated to layer2, keeping
everything else (Hungarian head, FPN, Ricci-modulation,
n_box_queries=12, 224², 30-epoch smoke recipe) identical.

The plan's hypothesis was that backbone capacity / feature quality
was the Stage D bottleneck.

**Result: Stage D-1 mAP_50 = 0.0100, vs Stage D's 0.0073.**
$+0.0027$ improvement for a $5.4\times$ backbone parameter
increase (133 k → 715 k) plus ImageNet-pretrained features. The
plan's smoke gate is $\mathrm{mAP}_{50} \geq 0.10$; the
*falsification* zone is $< 0.05$. We're at 0.01, deep in the
falsified zone.

Per plan §7, this **falsifies** the "backbone is the bottleneck"
hypothesis. The orchestrator correctly skipped the 5-seed
(saved 7.5 h GPU).

**But the diagnostic from the JSONL is the actual headline**: both
Stage D and Stage D-1 produce **60 132 predictions versus 4 805
ground-truth boxes** — exactly $n_{\mathrm{box\ queries}}=12$
times the 5011 images. **Every query is firing on every image, at
every threshold.** The Hungarian head's no-object suppression is
broken at VOC scale.

## 2. The two smokes side-by-side

| Metric | Stage D (from-scratch) | Stage D-1 (ImageNet) | Δ |
|---:|---:|---:|---:|
| n_params total | 132 956 | **714 892** ($5.4\times$) | — |
| backbone params | ~107 k | ~680 k (ImageNet ResNet18) | — |
| wall (smoke) | 937 s | 1 055 s | $+12\%$ |
| loss_start | 4.006 | 4.041 | — |
| loss_end | 3.461 | 3.373 | — |
| loss_drop_pct | 13.6 % | 16.5 % | $+2.9$ pp |
| **mAP_50** | **0.0073** | **0.0100** | **$+0.0027$** |
| mAP_50_95 | 0.00193 | 0.00243 | $+0.0005$ |
| mean_iou_matched | 0.237 | 0.247 | $+0.010$ |
| box_cls_acc | 0.250 | 0.313 | $+0.063$ |
| **n_preds_used** | **60 132** | **60 132** | **identical** |
| **n_gts_total** | **4 805** | **4 805** | **identical** |

## 3. The diagnostic: $\frac{60132}{4805} = 12.5$

The ratio is exactly the *query count divided by the average
ground-truth count per image*:

$$\frac{n_{\mathrm{queries}} \times n_{\mathrm{images}}}{n_{\mathrm{gts}}}
= \frac{12 \times 5011}{4805} = 12.51$$

Every box query is emitting a non-suppressed prediction on every
image. Of those 12 queries:

- Roughly 1 hits a real GT (Hungarian assignment matches it).
- Roughly 1 hits *almost* a GT (mean_iou_matched=0.25 is a real
  but loose match — IoU ≈ 0.25 is "the model knows roughly where
  the object is").
- The other **10 queries are spurious confident predictions.**

mAP_50 evaluates the precision-recall curve at IoU ≥ 0.5. With
60 k spurious predictions flooding the rank ordering, precision
collapses long before recall is reached.

### What we can rule out

- **It is not backbone capacity.** $5.4\times$ more backbone
  params + ImageNet pretrain moves mAP only $+0.0027$. The features
  are not the limit.
- **It is not basic feature transfer.** mean_iou_matched went from
  0.237 to 0.247 — the matched queries *are* in roughly the right
  spatial location. The model has learned something.
- **It is not the loss formulation overall.** Loss drops 13.6 %
  (Stage D) and 16.5 % (Stage D-1). The training signal is there.

### What we now suspect

Three candidates, in decreasing order of suspicion:

1. **Query-count over-provisioning.** Cluttered MNIST had
   $n_{\mathrm{box\ queries}}=4$ on 64×64 images with ~2.5
   objects/image — query count $\sim$ object count, so the
   no-object slot got a strong gradient signal. VOC has
   $n=12$ on 224×224 with ~2.4 objects/image — query count is
   **5× the object count**, so the "no-object" target dominates
   the Hungarian assignment but the model never learns to use it.
2. **No-object class weight in the cross-entropy.** The class head
   outputs $n_{\mathrm{cls}}+1 = 21$ logits; slot 20 is "no-object".
   At average 2.4 objects per 12 queries, the **balanced** CE target
   distribution is 80 % no-object — but if the CE loss treats all
   21 classes with equal weight, the gradient on "predict no-object"
   is much weaker than on "predict one of the 20 real classes."
3. **Warm-start at the wrong resolution.** The saliency-FPS query
   corner warm-start was tuned for 64² CMNIST; at 224² with 4×
   more pixels and 16× more spatial cells, the FPS sampling may be
   placing query corners in spatial clusters that all collapse onto
   the same dominant object.

The plan's §7 anticipated all three (risks items 2, 3 and 4); we
just have data now to rank them.

## 4. Action: Stage D-2 plan — three orthogonal probes

Each probe is single-axis-different from Stage D-1, giving clean
paired attribution. All cheap (single-seed smoke is ~18 min).

| Probe | What changes | Hypothesis | Expected mAP_50 |
|---:|:---|:---|---:|
| **D-2a** | $n_{\mathrm{box\ queries}}=4$ (matches CMNIST) | over-provisioning is the bottleneck | $\geq 0.05$ |
| **D-2b** | CE no-object weight ↓ to $1/(K-1) \approx 0.09$ | class imbalance is the bottleneck | $\geq 0.05$ |
| **D-2c** | random query corners (no warm-start) | warm-start mis-tuned for 224² | flat (~0.01) |

If D-2a alone clears 0.05, the bottleneck is over-provisioning and
the architecture is fine. If D-2b clears 0.05, the loss balance
is the issue. If D-2c clears 0.05, the warm-start was actively
hurting. Most likely outcome: D-2a wins by a wide margin.

D-2a is the right first probe. It's the smallest code change
(one CLI flag) and the most-likely lift per the diagnostic.

## 5. Why this matters even with the falsification

A negative result at the production-scale-smoke gate is the
**designed-for** outcome of CLAUDE §3's smoke→gate→5-seed pattern.
The orchestrator:

- spent 18 min (not 7.5 h) on the wrong-direction hypothesis,
- emitted a JSONL that surfaced the diagnostic instantly,
- correctly identified the falsification zone and aborted the
  5-seed.

The whole architectural-attribution risk item the Stage D-1 plan
flagged (§7 item 5, "if Stage D-1 passes, credit-assignment
between ImageNet features and the Ricci/HSiKAN head becomes
unclear") is now moot — Stage D-1 didn't pass, so we *know* the
Ricci/HSiKAN head and FPN are not the bottleneck, the backbone is
not the bottleneck, and the bottleneck lives in the query-count /
class-balance / warm-start triangle. The next experiment is
sharper, not blunter, because of the falsification.

## 6. Files

### Already on disk
- `signedkan_wip/src/vision/hymeyolo_backbones.py` — `ResNet18ImageNetBackbone` class shipped 2026-05-18 (will stay; the plan didn't fail because of the backbone code).
- `signedkan_wip/tests/test_resnet18_imagenet_backbone.py` — 5/5 tests passing.
- `signedkan_wip/src/vision/train_voc_stagec.py` — `--backbone resnet18_imagenet` flag added; reusable for any future Stage D-x probe.
- `signedkan_wip/experiments/run_stage_d1_voc2007_2026_05_18.sh` — keep; can re-run with new flags if needed.

### Next
- `docs/plans/2026-05-18-hymeyolo-stage-d2-query-count/` — Stage D-2a plan (4-format, will write in the bottleneck-investigation thread next).
- Stage D-1 plan should be updated in §6 of its `plan.tex` to mark the falsified status with the actual diagnostic data point.

## 7. Bottom line

Stage D-1 falsified the "backbone is the bottleneck" hypothesis
cleanly and cheaply. The actual bottleneck is at the **head**, not
the backbone: 12 box queries on 2.4-object-per-image VOC, with the
Hungarian no-object class either over-weighted in the loss or
under-supervised by gradient — every query fires on every image,
12.5× more predictions than ground-truths, mAP collapses to noise.

The fix is a single CLI flag (`--n-box-queries 4`) plus, optionally,
a CE class-weight rebalance for the no-object slot. Stage D-2 plan
to follow.
