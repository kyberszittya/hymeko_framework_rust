# Stage D-3 — Nodelet head + gate-aware eval bugfix

**Date:** 2026-05-19
**Plan:** [`docs/plans/2026-05-19-stage-d3-nodelet-head/`](../docs/plans/2026-05-19-stage-d3-nodelet-head/) (4-format)
**Verdict:** **gates are training** (clean 4-low / 12-high separation per image) but per-image cardinality balance is off; mAP_50 lifts only marginally (0.0104 → 0.0127 with the eval bugfix; 0.053 single-class via Stage H separately). The architectural axis is correct; the loss-balance recipe needs work (Stage D-3-bis). A second smoke (Stage D-3c) tests the family-paper-pure HSiKAN-CR backbone variant.

## 1. Summary

Stage D-2 (2026-05-18) systematically tested whether the
HyMeYOLO Hungarian head's $K+1$-class softmax bottleneck could be
fixed by tuning `lam_no_obj` alone. D-2d's diagnostic was
decisive: matched-cls accuracy 0.875 (per-query *quality* is
great) but mAP = 0.0077 (every query fires; 12.5× over-provisioning).
The architectural fix was sketched as **explicit per-query
objectness gates** — and Stage D-3 implements it.

**Three smokes today**:

| Probe | Head | n_queries | Backbone | mAP_50 | Reading |
|:---|:---|---:|:---|---:|:---|
| Stage D-2d | Hungarian (lam=0.1) | 4 | ResNet18-ImageNet | 0.0077 | baseline |
| D-3 v1 (eval bug) | Nodelet | 16 | ResNet18-ImageNet | 0.0104 | trained fine but eval gate-blind |
| **D-3b** (gate-aware eval) | **Nodelet** | **16** | **ResNet18-ImageNet** | **0.0127** | gates trained, marginal lift |
| **D-3c** | **Nodelet** | **16** | **HSiKAN-CR (Catmull-Rom)** | **OOM** | basis-eval activations exceed 7.6 GB; needs ckpt or smaller input |

**The interesting finding**: gates DO train and separate cleanly,
but the per-image cardinality balance is wrong — 12 of 16 queries
fire per image instead of the ~2.4 GTs/image VOC actually has.
The architectural fix is on the right axis; the loss recipe needs
more pressure on suppression.

## 2. The gate-aware eval bugfix

D-3 v1 was the first nodelet-head training. mAP came out at 0.0104
— barely better than the legacy Hungarian D-1 (0.0100). Suspicious:
the loss dropped 23.5 % (best of any VOC probe; D-2's were 13–16 %)
but mAP didn't track.

The diagnostic: `n_preds_used = 80176 = 5011 × 16` — the eval
scored *all 16 queries per image*, regardless of gate. Looking at
the eval code path in `compute_detection_metrics`:

```python
probs = F.softmax(pred_cls, dim=-1)
obj_probs = probs[..., :n_classes]
best_score, best_class = obj_probs.max(dim=-1)
```

The eval scores every query by cls-softmax → ranks predictions →
computes precision-recall → mAP. It doesn't know about gates.
For the gated head, **a high cls confidence + a near-zero gate
should produce a near-zero effective score** — but the eval was
treating cls confidence as the whole story.

**The fix** (`signedkan_wip/src/vision/train_circles_ricci.py`):

```python
# Gates default to ones for legacy heads — backward compat preserved.
pred_gates = pred.get("box_gates", pred_boxes.new_ones(pred_boxes.shape[:2]))
...
# After computing per-query best class probability, multiply by gate:
best_score = best_score * pred_gates
```

Stage D-3b re-ran the smoke with the same checkpoint training
recipe but gate-aware eval. Result: mAP 0.0104 → 0.0127, +22 %.
Real but small.

## 3. The gate distribution diagnostic

After D-3b, I inspected the trained gate distribution on 8 VOC
images. **Gates ARE separating per image:**

```
image 0 gate values (sorted):
[0.010, 0.018, 0.064, 0.150,      ← 4 queries learned to suppress
 0.621, 0.703, 0.816, 0.854,
 0.876, 0.880, 0.889, 0.936,      ← 12 queries firing
 0.947, 0.974, 0.974, 0.975]
```

| Statistic | Value |
|:---|---:|
| Min gate | 0.010 |
| Mean gate | 0.630 |
| Max gate | 0.994 |
| Std | 0.362 |
| Fraction > 0.5 | 0.703 |
| Fraction > 0.3 | 0.742 |

So at the 0.5 threshold, ~70 % of queries fire — 11 of 16. With
VOC's mean 2.4 GTs/image, this is still 4.6× over-provisioning.
But it's **not 100 %** anymore (D-1 / D-2 had every query firing).

## 4. Why is the cardinality balance off?

The gate-BCE loss has two terms:
- $\mathcal{L}_{\text{gate}}^{+}$ pushes matched queries toward 1.
- $\mathcal{L}_{\text{gate}}^{-}$ pushes unmatched queries toward 0.

The auto-balance in `hungarian_set_loss_gated` sets
`lam_gate_neg = N_matched / N_unmatched` so the two terms
contribute roughly equally **per-sample**. With ~2.4 matched and
~13.6 unmatched per image, lam_gate_neg ≈ 0.18.

**The mistake**: per-sample balance is the wrong thing. We don't
want each unmatched query to feel as much pressure as each
matched one. We want the suppression to be the **dominant** signal
because most queries SHOULD suppress. The auto-balance suppresses
too gently — only the most over-confident wrong queries get
shoved down, the borderline ones drift to ~0.7.

**Two fixes for Stage D-3-bis (queued, not yet run)**:

1. Set `lam_gate_neg = 1.0` (equal weight per-sample), making the
   suppression-side gradient ~5× larger in aggregate. The matched
   queries should still get strong gate-up signal because their
   gradient is concentrated on few samples.
2. Use focal loss for the gates: emphasises the rare "matched=1"
   case in the loss, while still letting many unmatched samples
   contribute to suppression.

Either is ~10 LOC change.

## 5. Stage D-3 file inventory

### New
- `signedkan_wip/src/vision/nodelet_head.py` (~250 LOC) — `NodeletQueryHead`-style heads + `hungarian_set_loss_gated` matcher/loss + `filter_predictions_by_gate` inference helper.
- `signedkan_wip/tests/test_nodelet_head.py` (7 tests) — head shape, gate range, gradient flow, matcher cost, curriculum behaviour, inference filter, dispatch.

### Modified
- `signedkan_wip/src/vision/hymeyolo_circles_ricci.py` — added `query_head_kind` kwarg; when `"nodelet"`, class head emits `n_classes` only (no +1 slot) and an extra gate head emits one sigmoid scalar per query. Forward emits `box_gates` / `circle_gates` in the output dict.
- `signedkan_wip/src/vision/train_circles_ricci.py`:
  - `combined_set_loss` dispatches to `hungarian_set_loss_gated` when `box_gates` present.
  - `compute_detection_metrics` multiplies per-query score by gate (the bugfix from §2).
  - Both changes default to legacy behaviour when gates absent (byte-identical CMNIST regression-safe).
- `signedkan_wip/src/vision/train_voc_stagec.py` — `--query-head-kind` flag + `hsikan` added to `--backbone` choices.

### CORE.YAML items touched
None.

## 6. Test results

| Suite | Tests | Status |
|:---|---:|:---:|
| `test_nodelet_head.py` | 7 | ✅ |
| `test_hymeyolo_*` (existing vision regression) | 14 | ✅ |
| `test_train_voc_stagec.py` | 4 | ✅ |
| Rapport suite (unchanged) | 38 | ✅ |
| **Total** | **63** | **✅** |

CMNIST byte-identical preserved: the `combined_set_loss` dispatch
runs the legacy `hungarian_set_loss` path when `box_gates` is
absent (existing CMNIST runs don't pass `query_head_kind`,
default `"hungarian"`, no gates emitted).

## 7. The D-3c (HSiKAN backbone) probe

**Hypothesis**: the family-paper claim is *one architecture across
regimes* — signed-link prediction (Bitcoin), small-from-scratch
vision (CMNIST b_hsikan 0.9032), HRI rapport. Including
**HSiKAN-CR as the backbone for VOC detection** would make the
unification claim 100 % pure (no borrowed ImageNet pretraining).

| Variant | Backbone | n_params | Pretrained |
|:---|:---|---:|:---:|
| D-3b | ResNet18-ImageNet | 714,924 | ✅ |
| **D-3c (running)** | **HSiKAN-CR (Catmull-Rom basis)** | **136,444 (5.2× smaller)** | ❌ from-scratch |

The CMNIST baseline: b_hsikan 5-seed 0.9032 ± 0.009. Whether that
basis-function primitive transfers to natural images at small
parameter count is what D-3c tests. Honest prior: pretraining is a
large advantage on natural images, so D-3c is expected to
underperform D-3b on mAP, but the comparison itself is
informative.

**D-3c smoke result — OOM at activation budget**. The 7.6 GB RTX
2070 SUPER ran out at training step 1 in the HSiKAN-CR backbone's
first basis evaluation (`hymeyolo_backbones.py:144` — the
`theta_p1 - theta_0` arc-length segment in the Catmull-Rom basis).
The model is 136,444 params (5.2× smaller than D-3b) but the
activation memory for the basis-function path is ~6.05 GiB — the
basis primitive's intermediate buffers dominate, not the weights.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate
14.00 MiB. GPU 0 has a total capacity of 7.60 GiB of which
78.69 MiB is free. ... this process has 6.18 GiB memory in use.
Of the allocated memory 6.05 GiB is allocated by PyTorch...
```

**Honest reading**: the family-paper-purity claim (HSiKAN-CR
backbone end-to-end on natural images) is **memory-blocked at the
current consumer GPU**, not architecture-blocked. To unblock:

1. Drop `--batch-size 8` → `--batch-size 4` (~50 % activation memory).
2. Drop `--input-size 224` → `--input-size 160` (~51 % activation memory).
3. Enable PyTorch activation checkpointing on the backbone's two
   `HSiKANBlock` instances (cheap; ~70 % memory at +30 % wall).
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation,
   marginal).

Option 3 is the right answer for the family-paper claim: it
preserves batch=8 / input=224 so the D-3c numbers are directly
comparable to D-3b. Queued as part of Stage D-3-bis (see §8).

## 8. Open items

1. **Stage D-3-bis: `lam_gate_neg = 1.0`** (or focal-gate loss).
   Should push the suppression-side gradient to ~5× and bring
   the firing fraction from 70 % → 30 % or below.
2. **5-seed validation** if any single-seed smoke clears the
   0.05 gate. None has yet (best D-3b smoke = 0.013).
3. **Joint test: D-3b + D-3-bis loss** to isolate whether the
   loss-balance fix is sufficient or whether the matcher's
   gate-veto cost also needs strengthening.
4. **Integration into rapport demo**: only if a D-3 variant
   clears the 0.10 gate. Stage H's 1-class detector ships as
   the visit-demo vision in the meantime.
5. **D-3c re-run with activation checkpointing** on the two
   `HSiKANBlock` instances. Expected wall +30 %, memory −70 %,
   no AUC delta. Then the family-paper-purity comparison
   completes at batch=8 / input=224 / iso-recipe with D-3b.

## 9. Bottom line

The Stage D-3 nodelet head **trains gates that separate cleanly** —
this is the architectural fix the Stage D-2 diagnostic predicted.
But the **per-image cardinality balance is too lax**: 70 % of
queries still fire after 30 epochs. mAP barely lifts (0.013 vs
0.010 baseline) because the firing flood still dominates the
precision-recall curve. The architectural direction is correct;
**Stage D-3-bis** (loss-balance tightening) is the obvious next
step before any visit-grade VOC-grounded detector is realistic.
For the family-paper, the result so far reads:

> *Explicit per-query objectness gates train successfully under
> Hungarian matching with BCE supervision, producing clean
> gate-bimodal separation (mean 0.63, std 0.36, fraction > 0.5
> = 0.70 after 30 epochs). The auto-balanced loss recipe is
> insufficient to push the firing fraction below the
> over-provisioning threshold; a Stage D-3-bis tuning the
> negative-class weight is the natural next step.*

This is the **diagnostic ceiling test** the family paper benefits
from: the architectural axis is correct (gates ARE learnable), the
loss recipe is the remaining tunable. Both are reportable.
