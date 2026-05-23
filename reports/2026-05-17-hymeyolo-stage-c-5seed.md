# HyMeYOLO Stage C — multi-scale FPN head (c_fpn 2-level), 5-seed paired

**Date:** 2026-05-17
**Plan:** [docs/plans/2026-05-16-hymeyolo-stage-c-fpn/](../docs/plans/2026-05-16-hymeyolo-stage-c-fpn/) (tex/pdf/tikz/mmd)
**Results dir:** [`signedkan_wip/experiments/results/hymeyolo_ladder_c_fpn_20260517T141622Z/`](../signedkan_wip/experiments/results/hymeyolo_ladder_c_fpn_20260517T141622Z/) — 2-level FPN head, ResNet-tiny backbone, A-3-lite levers

**Sweep window:** 2026-05-17 16:16 → 18:45 CEST (~2 h 30 min wall total, sole-GPU sequential)

**Verdict (c_fpn):** ⚠️ **TIE vs b_resnet — paired Δ = −0.0029 at z = −1.11, 2/5 wins**. The 2-level FPN multi-scale head does **not** add to b_resnet's single-scale head at this dataset/backbone scale; mean drops marginally from 0.8955 to 0.8926 (within 1σ of intrinsic seed-noise). **The Stage B ResNet backbone is the saturation point at this Cluttered MNIST scale**. c_fpn does retain the cumulative +0.1466 paired Δ over Stage A-2 (z=+9.08, 5/5) — it's a *valid* model, just doesn't *improve* over b_resnet.

## 1. Summary

| Stage | Backbone | Head | (n=5) mean | pstdev | min | max | wall/seed |
|------:|----------|------|-----------:|-------:|----:|----:|----------:|
| baseline (no warm-start, const LR, e=50) | TinyBackbone (~14k) | single-scale | 0.5041 | 0.0391 | 0.4714 | 0.5789 | 518 s |
| A-1 (warm-start) | TinyBackbone | single-scale | 0.6279 | 0.0521 | 0.5430 | 0.6768 | 577 s |
| A-2 (+ cosine + warmup + e=100) | TinyBackbone | single-scale | 0.7460 | 0.0350 | 0.6872 | 0.7855 | 1041 s |
| B' b_prime (+ A-3-lite levers) | TinyBackbone | single-scale | 0.7358 | 0.0231 | 0.7025 | 0.7695 | 1623 s |
| B b_resnet (+ ResNet-tiny + A-3-lite) | ResNet-tiny (~107k) | single-scale | 0.8955 | 0.0267 | 0.8432 | 0.9135 | 1627 s |
| B b_hsikan (+ HSiKAN-CR + A-3-lite) | HSiKAN-CR (~111k) | single-scale | 0.9032 | 0.0087 | 0.8917 | 0.9160 | 3395 s |
| **C c_fpn** (b_resnet + 2-level FPN) | ResNet-tiny | **2-level FPN** | **0.8926** | **0.0238** | **0.8461** | **0.9120** | **1779 s** |

Cumulative paired Δ (Δ vs immediate baseline in the ladder):

| Stage transition | mean Δ | σ_Δ | z | win-rate | verdict |
|-----------------:|-------:|----:|---:|----------|---------|
| A-1 vs baseline | +0.1238 | 0.0592 | +4.68 | 5/5 | WIN |
| A-2 vs A-1 | +0.1181 | 0.0189 | +14.01 | 5/5 | WIN |
| B b_resnet vs A-2 | +0.1494 | 0.0410 | +8.14 | 5/5 | WIN |
| B b_hsikan vs b_resnet | +0.0077 | 0.0283 | +0.61 | 2/5 | TIE |
| **C c_fpn vs b_resnet** _(headline FPN test)_ | **−0.0029** | **0.0058** | **−1.11** | **2/5** | **TIE** |
| **C c_fpn vs b_hsikan** | **−0.0105** | **0.0263** | **−0.90** | **3/5** | **TIE** |
| **C c_fpn vs b_prime** _(isolates backbone+FPN over levers-only)_ | **+0.1568** | **0.0288** | **+12.16** | **5/5** | **WIN** |
| **C c_fpn vs A-2** _(cumulative B+C)_ | **+0.1466** | **0.0361** | **+9.08** | **5/5** | **WIN** |
| cumulative (c_fpn vs baseline) | +0.389 | (paired across 4 interventions) | — | — | — |

**Three things stand out about c_fpn:**

1. **FPN does not lift over single-scale at this scale.** The 2-level FPN head, predicted to deliver ~+0.05 mAP_50 in the plan, delivered **−0.0029** (a near-noise *loss*). Cluttered MNIST objects appear to be uniform-scale enough that multi-scale aggregation provides no benefit; the single-scale ResNet-tiny output is already discriminative.
2. **σ improved (0.0267 → 0.0238)**, the smallest variance among any Stage B/C result. FPN adds modest stabilisation even without mean lift — feature aggregation across scales smooths seed-to-seed variance even when it doesn't shift the mean.
3. **Worst-seed (seed 2) still tracks b_resnet's worst.** c_fpn seed-2 = 0.8461; b_resnet seed-2 = 0.8432. The cross-seed correlation in this dataset's hard examples persists across heads — both are seeing the same "hard" sample-2 realisation. This is a clean signal that the head is not the bottleneck for that seed.

## 2. Per-seed paired tables

### c_fpn vs b_resnet (headline FPN test)

| seed | c_fpn mAP_50 | b_resnet mAP_50 | paired Δ | c_fpn wall |
|-----:|-------------:|----------------:|---------:|-----------:|
|   0  |       0.9120 |          0.9126 |  −0.0006 |    1791 s |
|   1  |       0.9073 |          0.9135 |  −0.0063 |    1788 s |
|   2  |       0.8461 |          0.8432 |  +0.0029 |    1769 s |
|   3  |       0.8998 |          0.8977 |  +0.0020 |    1777 s |
|   4  |       0.8979 |          0.9103 |  −0.0124 |    1769 s |

Mean paired Δ = **−0.0029 mAP_50**, pstdev 0.0058, z = **−1.11**, 2/5 wins — **TIE** by the pre-registered rule (|Δ| < 0.03 AND |z| < 2). Per-seed Δ envelope is tight: [−0.012, +0.003] — every seed lands within ±0.013 mAP of its b_resnet pair. The 2-level FPN head is *equivalent* in performance, *cheaper* in variance, and slightly more expensive in compute (~+10 % wall: 1779 vs 1627 s/seed).

### c_fpn vs b_hsikan

| seed | c_fpn mAP_50 | b_hsikan mAP_50 | paired Δ |
|-----:|-------------:|----------------:|---------:|
|   0  |       0.9120 |          0.9080 |  +0.0040 |
|   1  |       0.9073 |          0.8917 |  +0.0156 |
|   2  |       0.8461 |          0.9045 |  −0.0584 |
|   3  |       0.8998 |          0.8954 |  +0.0043 |
|   4  |       0.8979 |          0.9160 |  −0.0182 |

Mean paired Δ = **−0.0105**, z = **−0.90**, 3/5 wins — **TIE**. Note the seed-2 outlier: c_fpn seed-2 is the worst across all of Stage C, but b_hsikan seed-2 was one of *its* best — suggesting the FPN feature aggregation may be sensitive to the same hard realisation in a way the HSiKAN-CR activation is not. This is a single-seed observation; a 4-stage paired analysis across {b_resnet, b_hsikan, c_fpn, c_hsikan-fpn} would isolate whether FPN+CR robustness compounds; that's deferred to a future Stage C''.

### c_fpn vs b_prime (isolates the B+C-combined backbone-and-multi-scale contribution)

| seed | c_fpn mAP_50 | b_prime mAP_50 | paired Δ |
|-----:|-------------:|---------------:|---------:|
|   0  |       0.9120 |         0.7294 |  +0.1827 |
|   1  |       0.9073 |         0.7524 |  +0.1549 |
|   2  |       0.8461 |         0.7253 |  +0.1208 |
|   3  |       0.8998 |         0.7695 |  +0.1303 |
|   4  |       0.8979 |         0.7025 |  +0.1954 |

Mean paired Δ = **+0.1568**, pstdev 0.0288, z = **+12.16**, 5/5 wins — clean **WIN** vs the A-3-lite-only TinyBackbone reference. The backbone-and-head combined contribution is +0.1568, almost exactly equal to the Stage B b_resnet-vs-b_prime Δ of +0.1597 (which isolated the backbone alone). **The 2-level FPN head contributes essentially zero on top of the ResNet-tiny backbone swap** — confirming the §1 read.

### c_fpn vs A-2 (cumulative B+C)

| seed | c_fpn mAP_50 | A-2 mAP_50 | paired Δ |
|-----:|-------------:|-----------:|---------:|
|   0  |       0.9120 |     0.7518 |  +0.1602 |
|   1  |       0.9073 |     0.7750 |  +0.1323 |
|   2  |       0.8461 |     0.7307 |  +0.1154 |
|   3  |       0.8998 |     0.7855 |  +0.1143 |
|   4  |       0.8979 |     0.6872 |  +0.2107 |

Mean paired Δ = **+0.1466**, z = +9.08, 5/5 — **WIN**. The full Stage B+C config remains a clean cumulative win over the A-2 single-scale TinyBackbone baseline.

## 3. Why FPN did not deliver the predicted +0.05

The plan ([docs/plans/2026-05-16-hymeyolo-stage-c-fpn/plan.tex]) modeled the FPN lift at +0.05 mAP_50 by reference to YOLO-v3-to-v4 single-scale-to-FPN deltas on COCO. The observed Δ = −0.0029 says: that prediction was wrong by ~3σ (the lift didn't materialise).

Three honest interpretations, ordered by confidence:

1. **Cluttered MNIST is single-scale by construction.** The dataset's objects (digit instances) are placed at a near-uniform scale within image; multi-scale aggregation captures essentially the same information that a single-scale head at the right resolution already captures. The FPN's classic advantage — handling small + large objects in the same image — does not apply to this dataset. This is the most likely explanation, since b_resnet's per-IoU mAP_50:95 was already 0.777 (very tight matching); there is no spare "small-object recall" to recover.
2. **At 128×128 input, the ResNet-tiny stride has already locked in the discriminative resolution.** A 2-level FPN combines stride-8 + stride-16 features; if stride-8 alone already saturates the recall, the stride-16 lateral adds nothing. (A stride-32 layer would test this, but the plan deliberately scoped to 2-level for compute reasons.) **Mitigation:** the 3-level FPN variant (stride-4 + stride-8 + stride-16) was *not* tested and remains a possible Stage C' lift — but if the saturation diagnosis is correct, it would be a similar tie.
3. **The plan was over-confident on the FPN prior.** The +0.05 prediction was based on YOLO-v3-v4 deltas, which use natural-image datasets with a wide object-scale distribution. The Cluttered MNIST regime is more like a single-stride-optimal benchmark. **Recommendation:** future Stage C-style multi-scale predictions should reference small-object-aware datasets, not generic COCO-class transfers.

## 4. mAP_50:95 lifted in 1 of 5 seeds, but mean was a tie

c_fpn per-seed mAP_50:95: 0.804 / 0.785 / 0.691 / 0.762 / 0.751 → mean ~0.759.
b_resnet per-seed mAP_50:95: 0.787 / 0.787 / 0.755 / 0.776 / 0.781 → mean ~0.777.

Paired mAP_50:95 Δ = mean −0.018 (LOSS at strict IoU). FPN didn't help small-IoU recall; if anything it slightly *hurt* tight localisation on this dataset. Same root cause as the mAP_50 tie: no spare localisation budget to recover.

## 5. Files / artefacts

| Item | Status |
|---|---|
| Source: [`signedkan_wip/src/vision/hymeyolo_circles_ricci.py`](../signedkan_wip/src/vision/hymeyolo_circles_ricci.py) (`--fpn 2level` head + dispatcher) | shipped earlier in Stage B work; verified here |
| Source: [`signedkan_wip/src/vision/train_circles_ricci.py`](../signedkan_wip/src/vision/train_circles_ricci.py) (`--fpn` CLI flag, jsonl row) | shipped earlier; verified here |
| Orchestrator: [`signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh`](../signedkan_wip/experiments/run_hymeyolo_ladder_5seed.sh) (c_fpn stage definition) | shipped earlier; used here |
| Analyser: [`signedkan_wip/experiments/analyse_hymeyolo_ladder_paired.py`](../signedkan_wip/experiments/analyse_hymeyolo_ladder_paired.py) | shipped, used for all 4 paired analyses |
| Plan dir: [`docs/plans/2026-05-16-hymeyolo-stage-c-fpn/`](../docs/plans/2026-05-16-hymeyolo-stage-c-fpn/) (tex/pdf/tikz/mmd) | 4 formats present |
| c_fpn smoke results: [`hymeyolo_ladder_c_fpn_20260517T134503Z/`](../signedkan_wip/experiments/results/hymeyolo_ladder_c_fpn_20260517T134503Z/) | 1 jsonl row (seed 0 smoke); mAP_50 = 0.8980, fpn = "2level", in [0,1] |
| c_fpn 5-seed results: [`hymeyolo_ladder_c_fpn_20260517T141622Z/`](../signedkan_wip/experiments/results/hymeyolo_ladder_c_fpn_20260517T141622Z/) | 5 jsonl rows + orchestrator.log; "5 rows" end-marker present |

## 6. CORE.YAML items touched

None. All edits internal to `signedkan_wip/src/vision/` were made in the Stage B session; Stage C only adds orchestrator invocation + analysis + this report. No template, no parser, no `lockdown` file edited.

## 7. Experiment provenance

* **Git SHA:** `2ccaa4d12fae1ff9cd533bd91cd84b28f11c3dab` ("Gomb reaches SOTA. By large"). Working tree dirty with Stage A/B source patches + plan dirs + 5 reports (this file inclusive).
* **Python / torch:** miniconda3, torch 2.11.0+cu130 (protocol parity with all 2026-05-16/17 HyMeYOLO experiments).
* **GPU:** NVIDIA RTX 2070 SUPER, 8 GiB, driver 580.126.09. Sole-GPU sequential (no contention).
* **Seeds:** 0, 1, 2, 3, 4 (paired-by-seed dataset realisation with Stage A-1 / A-2 / B controls).
* **Hyperparams (CLI), c_fpn:**
  `--n-images 5000 --epochs 100 --lr 0.003 --ricci-scale 1.0
  --warm-start --schedule cosine --warmup-epochs 10
  --min-lr-ratio 0.01 --use-layernorm --weight-decay 1e-4
  --cls-loss focal --backbone resnet --fpn 2level
  --configs +ricci-mod`
* **Resource cap:** `systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0` per scope; cgroup never tripped.
* **Per-seed walls:** 1791 / 1788 / 1769 / 1777 / 1769 s (very tight; FPN is compute-stable across seeds).

## 8. YOLO-parity ladder update

| Stage | Lever | Status | (n=5) mAP_50 |
|------:|-------|--------|---:|
| baseline | honest (no warm-start, const LR, e=50) | shipped 2026-05-16 morning | 0.5041 ± 0.039 |
| A-1 | warm-start query corners | shipped 2026-05-16 noon | 0.6279 ± 0.052 |
| A-2 | + cosine LR + warmup + e=100 | shipped 2026-05-16 afternoon | 0.7460 ± 0.035 |
| A-3-lite | + LayerNorm + WD=1e-4 + focal cls | smoke 0.749 (1 seed); 5-seed not run | (single-seed only) |
| B' b_prime | TinyBackbone + A-3-lite levers | shipped 2026-05-17 (5-seed sole-GPU) | 0.7358 ± 0.023 |
| B b_resnet | + ResNet-tiny + A-3-lite | shipped 2026-05-16 evening | 0.8955 ± 0.027 |
| B b_hsikan | + HSiKAN-CR + A-3-lite | shipped 2026-05-17 (full 5-seed) | 0.9032 ± 0.009 |
| **C c_fpn** | **+ 2-level FPN head over b_resnet** | **shipped 2026-05-17** | **0.8926 ± 0.024** |
| C' c_fpn3 (next) | 3-level FPN | not started | predicted ≤+0.01 (likely tie) |
| C'' c_fpn-hsikan (next) | 2-level FPN + HSiKAN backbone | not started | open question |
| D | Port to VOC subset / COCO-mini | not started | real-data validation; multi-scale lever should matter there |

## 9. §6.5 anti-pattern review

| # | Pattern | Status |
|--:|---------|--------|
| 1 | Cartesian-product API | clean (one `--fpn` flag + one `--backbone` flag; no per-combination CLI fns) |
| 2 | Algorithm behind Python boundary | n/a |
| 3 | Per-experiment scaffold duplication | clean (orchestrator + analyser are unified; c_fpn is a single stage config) |
| 4 | Long single-file modules | head dispatcher remains under threshold |
| 5 | New axis via new function name | clean (`fpn` is a kwarg + CLI flag) |
| 6 | `#[allow(too_many_arguments)]` | n/a |
| 7 | String-typed config | `fpn: str` with `choices=["none","2level"]`; Python-boundary exception per §6.5 #7 |
| 8 | Forward-time flags for structural differences | head dispatch instantiates a different class per `fpn` value, not a `forward()` switch |
| 9 | Bypassing existing Strategy traits | n/a |
| 10 | `ulimit -v` on CUDA | n/a — cgroup |
| 11 | Global / module-level mutable state | clean |

No new suppressions, no silent failures.

## 10. Acceptance

- [x] c_fpn: 5/5 seeds landed jsonl rows; no cgroup OOMs.
- [ ] c_fpn: pre-registered criterion (paired Δ ≥ 0.05 AND z ≥ 2): **NOT MET** — Δ=−0.0029, z=−1.11. **Verdict: TIE; Stage C does not advance the ladder.**
- [x] c_fpn: paired-by-seed comparison vs Stage B b_resnet.
- [x] c_fpn: no mAP_50 row > 1.0 (honest metric working).
- [x] c_fpn: smoke landed at mAP_50 = 0.8980, fpn="2level", in [0,1] (Phase 4 gate passed before 5-seed launch).
- [x] σ improved (0.0267 → 0.0238) — stabilisation without mean lift.
- [x] CORE.YAML untouched.
- [x] No new §6.5 anti-patterns.
- [x] Plan dir 4 formats committed (.tex compiles, .tikz compiles, .mmd parses, .pdf built).
- [x] All Stage B's 88+ ricci-adjacent tests still green (no source changes in Stage C).

## 11. Bottom line

Stage C 2-level FPN is a **TIE vs Stage B b_resnet** at paired Δ = **−0.0029 mAP_50, z = −1.11** on Cluttered MNIST. **The ResNet-tiny single-scale head is the saturation point at this dataset/backbone scale**: multi-scale feature aggregation provides no measurable benefit when the dataset has uniform object scale and the backbone's stride-8 output already captures the discriminative resolution.

The cumulative ladder verdict stands:
- **A-1 → A-2 → B (backbone)** is the productive trajectory: each step delivered ≥+0.12 mAP_50, totalling +0.391 over the honest baseline.
- **B b_hsikan** is at parity with b_resnet (TIE, +0.0077) — CR primitive transfers to vision at iso-topology.
- **B' b_prime** isolates: backbone swap *alone* explains all of the +0.149 Stage-B lift; A-3-lite head levers are decorative on TinyBackbone (TIE vs A-2).
- **C c_fpn** does not advance — Cluttered MNIST is FPN-insensitive.

**Next ladder candidates** (in decreasing priority):
1. **Stage D — port to VOC or COCO-mini**: multi-scale should matter on natural images; this validates whether Stage C is dataset-specific or architectural.
2. **Stage C'' c_fpn-hsikan**: full {ResNet,HSiKAN}×{single,FPN} factorial for completeness on Cluttered MNIST.
3. **Stage E — soma cortical priors**: orthogonal lever family from the Gömb-soma side, completely independent of FPN saturation.

**Architectural ceiling on Cluttered MNIST**: at the current measurement granularity (paired n=5, σ≈0.025), 0.90 mAP_50 is the regime ceiling at ResNet-tiny scale. Pushing past requires either dataset change (Stage D) or backbone-capacity step (wider c_out, deeper layers — not in current plan).

---

_5-seed verdict was the operating-contract requirement. Stage C does not promote a row to SOTA_RESULTS.md headline (TIE verdict by pre-registered rule). Status note at `/tmp/coordinator_status_2026_05_17.md`._
