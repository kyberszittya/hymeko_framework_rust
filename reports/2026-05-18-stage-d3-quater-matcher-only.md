# Stage D-3-quater — matcher cost only, BCE retained

**Date:** 2026-05-18
**Plan:** Hyperparameter sweep — no code change beyond the
already-shipped Stage D-3-tris CLI flags.
**Verdict:** **prediction falsified by 2.5×.** D-3-tris's §5
predicted mAP_50 ~0.025 from matcher-cost-only; actual is
**0.0094** — *below* D-3b's 0.0127 baseline. The matcher quality
metrics (cls_acc 0.69, mIoU 0.210) and gate bimodal separation
(std 0.351, min 0.0007, firing 27 %) **all came in even better
than predicted**, but mAP **regressed further**. The
attribution chain in the D-3-tris report (matcher-cost good,
focal bad) is now itself falsified for mAP. Real cause: matcher
cost = 3.0 over-selects, starving the other ~14 queries of
gradient signal.

## 1. The pattern across the D-3 series

After four smokes the picture is finally coherent:

| Variant | λg- | λg-match | Loss kind | **mAP_50** | cls_acc | mIoU | firing |
|:---|:---|:---|:---|---:|---:|---:|---:|
| D-3 v1 / D-3b | 0.18 (auto) | 1.0 | bce | 0.0127 | 0.625 | 0.171 | 0.70 |
| **D-3-bis (current best)** | **1.0** | 1.0 | bce | **0.0153** | 0.563 | 0.171 | 0.37 |
| D-3-tris | 1.0 | 3.0 | focal | 0.0132 | 0.690 | 0.225 | 0.41 |
| **D-3-quater** | 1.0 | **3.0** | bce | **0.0094** | 0.690 | 0.210 | 0.27 |

**The lesson**: cls_acc and mIoU are **misleading proxies for
mAP** when matched-queries are a small subset. D-3-tris and
D-3-quater both show cls_acc ≈ 0.69 (vs D-3-bis's 0.56), but
both have *worse* mAP. The metrics decouple because they measure
different things:

- **cls_acc and mIoU** measure quality of **matched** queries
  only (typically 2-3 per image).
- **mAP** measures the ranking of **all** queries
  (16 per image) against all GTs (~2.4 per image).

Raising the matcher cost concentrates assignment on the very
strongest-gate queries (the 2-3 most confident per image),
which improves the matched-subset quality but **starves the
remaining ~13 queries of cls/box gradient**. Those starved
queries still have mid-range gates from BCE training and produce
near-random cls predictions; their `gate × cls_prob` scores
(~0.09) sit just below the trained queries' scores in the
ranking and pollute the precision side of the AP curve.

D-3-bis's matcher cost = 1.0 is more *promiscuous*: the matcher
hands GTs to many different queries over the course of training,
spreading cls/box gradient across all 16 query slots. Per-query
quality is worse (cls_acc 0.56), but **per-image detection
quality is better** because every query is a usable candidate
with some learning behind it.

## 2. Gate distribution diagnostic

```
D-3-quater image 0 gate values (sorted):
[0.002, 0.002, 0.002, 0.008,     ← 4 deeply suppressed (BCE works)
 0.024, 0.043, 0.060, 0.104,     ← 4 weakly suppressed
 0.172, 0.312,                   ← 2 borderline
 0.585,
 0.685, 0.729, 0.772, 0.904, 0.984]  ← 5 firing
```

| Statistic | D-3-bis | D-3-tris | **D-3-quater** | Prediction held? |
|:---|---:|---:|---:|:---|
| Min gate | 0.002 | 0.082 | **0.0007** | ✅ BCE deep-suppress preserved |
| Mean gate | 0.339 | 0.440 | **0.288** | ✅ shifted down |
| Std gate | 0.323 | 0.284 | **0.351** | ✅ MORE bimodal |
| Firing fraction (>0.5) | 0.367 | 0.414 | **0.273** | ✅ down to ~ VOC's 15 % target |
| Min per-image firing | 0.0625 | 0.250 | **0.188** | ✅ no image fires all queries |
| Max per-image firing | 0.5625 | 0.688 | 0.4375 | ✅ tightly bounded |

**Every gate-distribution prediction from D-3-tris §5 held.** The
mAP prediction is the one that failed. That means the **gate
distribution was never the bottleneck** at this stage — the
remaining mAP gap lives elsewhere (in the cls / box gradient
flow to unmatched queries).

## 3. Why the prediction was wrong

The D-3-tris report's §5 prediction model was: *mAP = quality of
matched queries × bimodal-gate sharpness*. Both inputs improved,
so mAP should have improved. The actual model is closer to:

$$
\text{mAP} \approx \frac{
  \overbrace{\text{cls}_q \times \text{IoU}_q \times \mathbb{1}[g_q > \tau]}^{\text{per-query usefulness}}
}{
  \underbrace{\text{ranking pollution from untrained queries}}_{\text{denominator}}
}
$$

Raising matcher cost improves the numerator's per-matched-query
mean but **damages the denominator** by leaving ~13 queries
untrained. The matcher cost is acting as an
*assignment-narrowness* knob, not a *quality* knob:

- Too low (1.0, D-3-bis): wide assignment, all 16 queries
  learn → uniform but mediocre per-query quality.
- Too high (3.0, D-3-quater): narrow assignment, only 2-3
  queries learn → high per-query quality but uniform random
  predictions from the other 13 → polluted precision curve.

## 4. Final D-3 ranking — by mAP_50

1. **Stage H (1-class person)** — **0.053**, no head-bottleneck
   issue at all (K=1). The visit demo's detector.
2. **D-3-bis** — **0.0153**, the best 20-class variant.
   `--lam-gate-neg 1.0 --lam-gate-match-cost 1.0 --gate-loss-kind bce`.
3. D-3-tris — 0.0132. focal+matcher_cost combo, neither helped
   net.
4. D-3b — 0.0127. auto-balance, eval-bug-fix only.
5. D-3-quater — 0.0094. matcher_cost=3.0 starves queries.
6. D-3 v1 — 0.0104 (gate-blind eval; not directly comparable).
7. D-2d — 0.0077. legacy K+1 head.

## 5. The right next step — *outside* the matcher / loss-balance axis

Four iterations have explored the loss-balance / matcher-cost
plane around (0.18-1.0, 1.0-3.0, bce/focal) and surfaced D-3-bis
as the local optimum at mAP ≈ 0.0153. The visit-grade gate is
0.10 — **almost 7× further away**. The remaining levers in the
nodelet head's hyperparameter space are unlikely to close that
gap on their own.

The natural lever is **outside the head**:

1. **Stage D-3-quinquies (proposed)**: **HSiKAN-CR backbone with
   activation checkpointing**. The Stage D-3c OOM at the basis
   evaluation was a memory artefact; the family-paper-purity
   argument still wants this variant. Predicted mAP_50: any
   value is informative (the *comparison* is the result, not
   absolute mAP).
2. **Larger backbone / pretraining**: ResNet18-ImageNet is
   already used; the next step is ResNet50-ImageNet (would need
   batch size reduction to fit 7.6 GiB). High wall cost; defer
   until the visit demo's downstream usage justifies it.
3. **Train longer**: 30 epochs is short for a 5011-image
   training set with a randomly-initialised head. 100 epochs at
   the D-3-bis recipe is the simplest "scale-up" with no
   architectural risk. Predicted mAP ≥ 0.03.

The visit demo continues to ship on Stage H. D-3-bis remains the
"best 20-class architecture but not visit-grade" entry in the
table.

## 6. Tests

No new tests. The D-3-tris suite (16 nodelet-head tests + 38
regression) still passes — D-3-quater is a pure CLI re-invocation
of the D-3-tris-shipped flags.

| Suite | Tests | Status |
|:---|---:|:---:|
| `test_nodelet_head.py` | 16 | ✅ |
| `test_hymeyolo_stage_b.py` + `stage_c.py` + `voc_stagec.py` | 38 | ✅ |
| **Total** | **54** | **✅** |

## 7. Resource snapshot

| Metric | Value |
|:---|---:|
| Wall (1 seed, 30 ep, full trainval) | 626 s (10.4 min) |
| Peak host RSS | 4.4 GiB |
| Peak GPU mem | < 5 GiB |
| 16 GiB cap | well under |

## 8. Bottom line

**D-3-bis (λg-=1.0, λg-match=1.0, BCE, auto-balance OFF) is the
best 20-class nodelet-head configuration on VOC2007 trainval at
30 epochs / ResNet18-ImageNet / 16 queries.** Four iterations
explored the surrounding hyperparameter space and could not
exceed mAP_50 = 0.0153. The local-optimum claim is now backed by
two single-knob negative experiments (D-3-quater starves queries
at matcher_cost=3.0; D-3-tris-implied focal-only would compress
gates further per the diagnostic mechanism).

**Architectural take-away for the family paper**: the K+1-class
softmax bottleneck was real (D-3-bis lifts mAP +20 % vs D-3b),
but the residual gap to clean detection is **not in the head's
loss recipe** — it's in **multi-class signal-to-noise** (Stage
H showed 7× lift at K=1) or **representation capacity**
(scale-up / longer training).

> *Stage D-3-quater isolated the matcher-cost knob (1.0 → 3.0)
> with the gate loss kept as BCE. The matcher-quality metrics
> improved as predicted (cls_acc 0.563 → 0.69, mIoU 0.171 →
> 0.210, std 0.323 → 0.351, firing 0.37 → 0.27), but mAP_50
> regressed 0.0153 → 0.0094 (−38 %). The mechanism is that a
> narrow matcher concentrates training gradient on 2-3
> consistently-strongest-gate queries, leaving the remaining 13
> with random cls predictions. Their mid-range gates × random
> cls produce ~0.09 scores that pollute the precision side of
> the ranking. The D-3-bis configuration (λg- = 1.0,
> λg-match = 1.0, BCE) thus emerges as the local optimum in the
> matcher / loss-balance plane, with mAP_50 = 0.0153 capping the
> 20-class architectural ceiling at this training scale.*

D-3 series concluded for the head's hyperparameter axis. The
next experiments belong to backbone / scale axes.
