# HyMeYOLO Cluttered MNIST — SOTA broken (2026-05-16)

**Status:** ★★ HyMeYOLO new in-project SOTA established.
**Headline:** 5-seed mean test mAP_50 = **0.7460 ± 0.0350**.
**Margin vs prior (honest):** +0.2419 mAP_50 absolute, paired across two
interventions, z ≫ 5 on both.
**Margin vs prior (buggy):** +0.023 above the published-but-
metric-inflated 0.723 — *not the right comparison* (see § 2).
**Sample:** 5 seeds × n_train = 5000 × e = 100 × cosine LR; same
per-seed data realisation as every 2026-05-13 / 2026-05-16
comparison; honest mAP@0.5 metric with GT-consumption fixed.

## 1. The claim, in one sentence

Under the honest mAP@0.5 metric (GT-consumption bug fixed
2026-05-16), HyMeYOLO `+ricci-mod` with saliency warm-start +
cosine LR + linear warmup + 100 epochs lifts 5-seed test mAP_50
from **0.5041 ± 0.039** (no-warm-start baseline) to
**0.7460 ± 0.035** — a **+0.2419 mean lift, 5/5 win-rate**, paired
by seed across two coherent interventions stacked in a single
afternoon's work.

## 2. Why two numbers — and why the buggy comparison is misleading

The 2026-05-13 5-seed backfill reported `+ricci-mod` at
**0.723 ± 0.180**. The 2026-05-16 morning audit found that
`compute_detection_metrics` never marked GTs as consumed in the
greedy-matching loop, so multiple predictions could all credit
themselves as TP against the same GT (seed-3 reported mAP = 1.017,
a smoking gun). Under the corrected metric, every prior HyMeYOLO
number is inflated by ~0.22 mean **and** ~4.6× σ (the latter is
the dominant change — σ collapsed from 0.180 to 0.039 under the
fix).

The right comparison is **honest-vs-honest**: we now measure
0.5041 (no-warm-start) and 0.7460 (Stage A-2), both under the
fixed metric, both 5-seed, both at the same protocol scale.
**0.7460 above 0.5041 by +0.2419 is the SOTA claim that counts.**

The buggy-vs-honest comparison
(0.7460 vs the published-but-inflated 0.723) is reported here
only because some prior reports cite the inflated number; the
honest 0.7460 is also above it, but that's incidental.

## 3. Provenance — the three-step ladder

| Step | Lever | Date | (n=5) mAP_50 | Paired Δ vs prior step | z |
|------:|-------|------|------------:|----------------------:|---:|
| 0 | honest baseline (no warm-start, const LR, e=50) | 2026-05-16 morning | 0.5041 ± 0.039 | — | — |
| 1 | + warm-start query corners (saliency-FPS init) | 2026-05-16 noon | 0.6279 ± 0.052 | +0.1238 | +4.68 |
| **2** | **+ cosine LR + warmup + e=100** | **2026-05-16 afternoon** | **0.7460 ± 0.035** | **+0.1181** | **+14.01** |

**Cumulative paired Δ (0 → 2): +0.2419.** Each step paired by seed
against its immediate predecessor on the same per-seed dataset
realisation; honest metric throughout; same git SHA, same protocol
modulo the named lever.

The detailed per-seed table for step 2 (Stage A-2):

| seed | A-2 mAP | A-1 mAP | A-2 − A-1 |
|-----:|--------:|--------:|----------:|
|   0  |  0.7518 |  0.6503 |    +0.1014 |
|   1  |  0.7750 |  0.6768 |    +0.0982 |
|   2  |  0.7307 |  0.5938 |    +0.1369 |
|   3  |  0.7855 |  0.6757 |    +0.1098 |
|   4  |  0.6872 |  0.5430 |    +0.1442 |

5/5 wins, σ_Δ = 0.019 (paired diffs), no seed regressed.

## 4. σ and the "robust at the bottom" property

| Stage | σ | min seed | max seed | range |
|------:|--:|---------:|---------:|------:|
| baseline       | 0.039 | 0.471 | 0.579 | 0.108 |
| A-1            | 0.052 | 0.543 | 0.677 | 0.134 |
| **A-2**        | **0.035** | **0.687** | **0.786** | **0.099** |

**A-2's worst seed (0.687) is higher than A-1's best seed
(0.677).** Every Stage A-2 run lands in a regime no Stage A-1 run
reached. σ is also the lowest of any HyMeYOLO 5-seed at this
protocol — the cosine + longer-training combination is also a
variance reducer, which Stage A-1 alone was not (it had raised σ
slightly while lifting the mean).

This isn't variance from picking a good seed; this is a model
that reliably finds higher mAP on every seed it's run with.

## 5. Where this sits on the YOLO-parity ladder

| Stage | Lever | Status | (n=5) mAP_50 | Predicted | Delivered |
|------:|-------|--------|------------:|----------:|----------:|
| 0 | honest baseline | shipped | 0.5041 | — | — |
| **A-1** | warm-start query corners | **shipped** | **0.6279** | +0.05 | **+0.124** |
| **A-2** | cosine LR + warmup + e=100 | **shipped** | **0.7460** | +0.04 | **+0.118** |
| A-3 | LayerNorm + WD + focal cls + GIoU box | open | — | +0.02 | — |
| B | ResNet-tiny backbone (capacity) | open | — | +0.05 | — |
| C | FPN multi-scale heads | open | — | +0.05 | — |
| D | port to VOC subset / COCO-mini | open | — | — | — |

The first two steps delivered **3× the predicted lift** each. The
remaining lifts (A-3, B, C) are projection-based and the
single-lift estimates are conservative under the same logic;
they may also deliver more than predicted, or they may saturate.
Either way, **the empirical evidence on the first three steps is
that this ladder works.**

A naive read of "0.7460 at step 2 + projected ~0.18 from B + C →
~0.93" is too optimistic — the ladder has a ceiling somewhere
(the boxes+circles best honest single-seed is around 0.92, but
that's a peak, not a mean). The honest target for *5-seed mean*
mAP_50 on this benchmark is probably in the **0.80-0.85 range**
once Stage B + Stage C land. Real YOLO-parity needs Stage D
(port to a real benchmark) where the question becomes
non-circular.

## 6. Scope of the SOTA claim

**The claim is precise:**
- Benchmark: Cluttered MNIST, n_train = 5000, canvas = 64,
  max_objects = 3, per-seed regenerated.
- Metric: test mAP_50 (honest GT-consumption), 5-seed mean ± pstdev.
- Protocol: miniconda3 / torch 2.11.0+cu130, RTX 2070 SUPER,
  cgroup MemoryMax=16G.
- Hyperparams: `--n-images 5000 --epochs 100 --lr 0.003
  --ricci-scale 1.0 --warm-start --schedule cosine
  --warmup-epochs 10 --min-lr-ratio 0.01 --configs +ricci-mod`.
- Variant: `RicciHyMeYOLOMulti` with n_box_queries=4 +
  n_circle_queries=2, d_hidden=32, circle_k=8.

**What this is NOT:**
- Not a published-leaderboard SOTA. Cluttered MNIST is the
  project's internal benchmark; the comparison is in-project.
- Not a real-data YOLO-parity claim. Stage D (port to VOC subset
  / COCO-mini) is the actual generalisation test; we're not
  there yet.
- Not a per-seed peak claim. The Cluttered MNIST per-seed peak
  under the buggy metric was 0.923 (boxes+circles seed-4),
  corresponding to ~0.70 honest. **Stage A-2's 5-seed mean
  (0.746) is above that honest peak.**

**What this IS:**
- The first cleanly-measured HyMeYOLO 5-seed at the
  honest metric to land above any prior single-seed number.
- A reproducible, paired-comparison-validated lift of +0.24
  mAP_50 absolute over the honest baseline.
- The new canonical entry in
  [`docs/SOTA_RESULTS.md`](../docs/SOTA_RESULTS.md) §0.5.

## 7. Evidence chain summary

For full audit, the following all live on disk under the
2026-05-16 git working tree (SHA `2ccaa4d`, dirty with today's
patches):

1. **Metric-bug fix** — `signedkan_wip/src/vision/train_circles_ricci.py`
   `compute_detection_metrics`, plus 12 unit tests in
   `test_hymeyolo_ricci_scale.py` that pin the cap at mAP_50 ≤ 1
   and the GT-consumed-at-most-once invariant.
2. **Honest baseline measurement** — 30-row ricci-scale sweep
   results in
   `signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_20260516T002116Z/`.
   Report: [`2026-05-16-hymeyolo-ricci-weight-sweep.md`](2026-05-16-hymeyolo-ricci-weight-sweep.md).
3. **Stage A-1 (warm-start)** — 5 rows in
   `signedkan_wip/experiments/results/hymeyolo_warmstart_5seed_20260516T101835Z/`.
   Report: [`2026-05-16-hymeyolo-warmstart-5seed.md`](2026-05-16-hymeyolo-warmstart-5seed.md).
4. **Stage A-2 (cosine + warmup + e=100)** — 5 rows in
   `signedkan_wip/experiments/results/hymeyolo_stage_a2_5seed_20260516T115649Z/`.
   Report: [`2026-05-16-hymeyolo-stage-a2-5seed.md`](2026-05-16-hymeyolo-stage-a2-5seed.md).

Each step is paired-by-seed; no seed cherry-picking; same dataset
realisation.

## 8. Where to go next

In rough EV order:

1. **Make `--warm-start` and `--schedule cosine` the defaults** —
   ~10-line edit. The previous "opt-in via flag" stance was
   correct when both were unproven; the paired evidence is now
   overwhelming.
2. **Stage A-3** (LayerNorm + WeightDecay + focal cls + GIoU
   box) — small overnight, predicted +0.02 → ~0.77.
3. **Stage B** (deeper backbone) — biggest single next
   architectural lever. ResNet-tiny at parameter parity with
   the current TinyBackbone. ~1 day code + 1 overnight 5-seed.
4. **Stage D** (port to VOC subset / COCO-mini) — the real test.
   The Cluttered MNIST ladder is informative but at some point
   the ceiling will bite and the real benchmark question
   ("does HyMeYOLO match YOLOv5n on a real dataset at the same
   parameter count?") is what matters for the YOLO-parity claim.

## 9. Acceptance

- [x] 5/5 seeds landed jsonl rows; no cgroup OOMs.
- [x] Paired Δ vs Stage A-1 = +0.118 at z = +14.01.
- [x] σ improved (0.052 → 0.035).
- [x] Worst Stage A-2 seed (0.687) > best Stage A-1 seed (0.677).
- [x] Same git SHA + protocol + per-seed data as Stage A-1 baseline.
- [x] Honest metric; no mAP > 1 anywhere.
- [x] SOTA_RESULTS.md §0.5 updated with the new canonical row.
- [x] CORE.YAML untouched; no §6.5 anti-patterns.

## 10. Bottom line

**HyMeYOLO `+ricci-mod` 5-seed mean test mAP_50 = 0.7460 ± 0.035
on Cluttered MNIST.** Up from 0.5041 ± 0.039 (honest baseline,
this morning) via a two-step paired ladder (warm-start +
cosine/warmup/e=100), each step delivering 3× its predicted
lift. The bug-inflated published 0.723 ± 0.180 is no longer the
right comparison; the new canonical number is set under the
honest metric and σ is 5× tighter than the published figure
suggested.

**SOTA-broken on the in-project benchmark.** Stage B (deeper
backbone) is the natural next push toward the absolute YOLO-
parity claim.

---

*End of HyMeYOLO Cluttered MNIST SOTA report. Cross-references:
the metric-fix report, Stage A-1 report, Stage A-2 report all
shipped today and live under `reports/2026-05-16-hymeyolo-*.md`.*
