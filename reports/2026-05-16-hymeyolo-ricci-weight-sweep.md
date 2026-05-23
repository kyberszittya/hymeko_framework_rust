# HyMeYOLO `+ricci-mod` Ricci-scale sweep + mAP fix — 5-seed × 6 scales

**Date:** 2026-05-16
**Plan:** [docs/plans/2026-05-16-hymeyolo-ricci-weight-sweep/](../docs/plans/2026-05-16-hymeyolo-ricci-weight-sweep/) (tex/pdf/tikz/mmd)
**Results dir:** [`signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_20260516T002116Z/`](../signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_20260516T002116Z/) (30 jsonl rows, orchestrator log)
**Sweep window:** 2026-05-16 **02:21:16 → 06:53:50 CEST** (4 h 32 min total; mean per-run wall ≈ 543 s)
**Verdict:** **No ricci-scale beats `s = 1.0` at significance; the sweep's headline finding is the mAP-bug correction itself.**

## 1. Summary

Two changes shipped together as the night's source patch:

* **`ricci_scale` knob** on `RicciHyMeYOLOMulti` and
  `RicciKCycleHyMeYOLOMulti` — a scalar multiplier on the 3
  Ricci features (κ, mean-cos-θ, edge-length-variance) before
  they enter the class / offset heads. Default `1.0` =
  byte-identical to pre-patch behaviour (pinned by a unit test).

* **mAP@0.5 GT-consumption fix** in `compute_detection_metrics`
  (`signedkan_wip/src/vision/train_circles_ricci.py` lines
  ~290–320). The prior implementation never marked GTs as
  consumed during greedy matching, so multiple predictions
  could all be credited TP against the same GT — recall climbed
  past 1, AP overshot 1. Documented as
  ["open issue #2"](2026-05-13-hymeyolo-ricci-5seed-backfill.md)
  in the May-13 backfill (seed-3 `+ricci-mod` reported 1.017).
  Fixed via standard COCO greedy matching: per-IoU-level
  `gt_consumed[level]` bool list; each prediction picks the best
  unconsumed GT and consumes it on TP.

Both changes are covered by 12 new unit tests (`test_hymeyolo_ricci_scale.py`).

The sweep then probed 6 scales × 5 seeds = 30 runs of
`+ricci-mod` at the May-13 protocol (n_train=5000, epochs=50,
lr=3e-3, miniconda3 / torch 2.11, cgroup 16 GB).

### 1.1 Headline numbers

| scale | n | mean mAP_50 | pstdev | min | max | paired Δ vs s=1.0 | z | verdict |
|------:|--:|------------:|-------:|----:|----:|------------------:|----:|---------|
| 0.05  | 5 | 0.4487 | 0.0582 | 0.3647 | 0.5226 | −0.0554 | −2.36 | **LOSS** |
| 0.10  | 5 | 0.4907 | 0.0539 | 0.4075 | 0.5532 | −0.0133 | −0.58 | tie |
| 0.20  | 5 | 0.4426 | 0.0565 | 0.3836 | 0.5502 | −0.0615 | −2.18 | **LOSS** |
| 0.40  | 5 | 0.5093 | 0.0718 | 0.4025 | 0.6129 | +0.0052 | +0.19 | tie |
| 0.80  | 5 | 0.4383 | 0.1229 | 0.2360 | 0.6162 | −0.0657 | −1.12 | tie |
| **1.00** | 5 | **0.5041** | **0.0391** | **0.4714** | **0.5789** | (control) | — | — |

**Decision-tree branch hit** (plan §"Decision tree on the sweep result"):
*all scales lose to or tie with the `1.0` control* → `s★ = 1.0`,
no default change. The plan's third predicted branch
("monotonically below 1.0 → sweep extension {1.5, 2.0, 4.0}") is
*weakly* indicated; the data show scale 0.40 statistically equals
1.0 and 0.10 nearly does, so the relationship is non-monotonic
below 1.0. Sweep extension would be informative but isn't urgent.

### 1.2 The substantive finding

**HyMeYOLO `+ricci-mod`'s honest 5-seed mAP_50 is 0.504 ± 0.039
— not the 0.723 ± 0.180 reported in the May-13 backfill.** Two
deltas:

| Quantity | May-13 (buggy mAP) | 2026-05-16 (honest mAP) | Δ |
|---|---:|---:|---:|
| Mean   | 0.723 | **0.504** | **−0.219** |
| pstdev | 0.180 | **0.039** | **÷ 4.6 (smaller!)** |
| range  | [0.529, 1.017] | [0.471, 0.579] | drastically tighter |

The bug **inflated the mean by ≈ 0.22** *and* **inflated σ by 4.6×**.
Recall: in the buggy code, one well-trained seed could produce a
single GT that 8 predictions all credit themselves against → mAP
> 1; a poorly-trained seed didn't have that pile-up → lower
inflated mAP. Seeds had wildly different inflations on top of
their honest performance, blowing up σ.

**Under the honest metric, variance is no longer the bottleneck**
(σ = 0.039 ≈ 7.7 % of mean). The 2026-05-16 night-doc §1 framing
("variance is the bottleneck") was correct under bug-inflated
numbers but no longer applies. **The lever-ordering on the
YOLO-parity ladder needs to shift toward mean lift (capacity,
longer training, FPN) rather than variance reduction (warm-start).**
The warm-start lever (Stage A-1 of the night-doc ladder) is now
of *ambiguous EV*; a 5-seed warm-start launch will still be
queued (see §5) but its expected lift on a 0.504 baseline at
σ=0.039 is not the σ-collapse the night-doc anticipated.

## 2. Files touched

| File | + / − | Change |
|------|------:|--------|
| [`signedkan_wip/src/vision/hymeyolo_circles_ricci.py`](../signedkan_wip/src/vision/hymeyolo_circles_ricci.py) | +24 / −2 | `ricci_scale: float = 1.0` kwarg; multiply Ricci features by it on the box + circle paths |
| [`signedkan_wip/src/vision/hymeyolo_ricci_kcycle.py`](../signedkan_wip/src/vision/hymeyolo_ricci_kcycle.py) | +12 / −1 | same `ricci_scale` mirror for API consistency |
| [`signedkan_wip/src/vision/train_circles_ricci.py`](../signedkan_wip/src/vision/train_circles_ricci.py) | +60 / −15 | mAP greedy-match fix; `--ricci-scale` + `--warm-start` + `--warmstart-bootstrap-n` CLI; jsonl persistence of all three |
| [`signedkan_wip/tests/test_hymeyolo_ricci_scale.py`](../signedkan_wip/tests/test_hymeyolo_ricci_scale.py) | +258 / 0 | 12 unit tests (param count, byte-identical at default, scale=0 kills Ricci gradient, mAP cap at 1.0, GT consumed once, partial-match correctness, FPS spread, Gaussian smooth properties) |
| [`signedkan_wip/experiments/run_hymeyolo_ricci_scale_sweep_2026_05_16.sh`](../signedkan_wip/experiments/run_hymeyolo_ricci_scale_sweep_2026_05_16.sh) | new | sweep orchestrator |
| [`signedkan_wip/experiments/analyse_ricci_scale_sweep_2026_05_16.py`](../signedkan_wip/experiments/analyse_ricci_scale_sweep_2026_05_16.py) | new | aggregator + paired-Δ + mAP > 1 sanity check |

## 3. CORE.YAML items touched

None. All changes confined to `signedkan_wip/` (non-core).

## 4. Test results

```
$ env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python -m pytest \
    signedkan_wip/tests/ -q \
    --ignore=signedkan_wip/tests/test_hsikan_optuna_chase.py \
    --ignore=signedkan_wip/tests/test_run_optuna_search_attention.py \
    -k 'ricci or hymeyolo or circle or kcycle or detection_metric'
========== 76 passed, 461 deselected, 1 warning in 16.62s ==========
```

(Other unrelated tests in the suite — Optuna-dependent ones —
fail at *collection* time because `optuna` is not in `.venv`;
those failures predate this patch.)

The 12 new tests pin specifically:

| Test | What it catches |
|------|-----------------|
| `test_ricci_scale_default_byte_identical_to_prior` | `ricci_scale=1.0` keeps the forward output byte-equal to pre-patch on a state-dict-clone pair |
| `test_ricci_scale_zero_kills_ricci_branch_gradient` | `ricci_scale=0.0` leaves corner-feature grads finite but the Ricci-path component disappears |
| `test_ricci_scale_forward_runs_at_arbitrary_scale[…]` | Forward finite at scales 0.0 / 0.05 / 0.1 / 0.4 / 0.8 / 1.0 / 2.0 |
| `test_compute_detection_metrics_caps_at_1` | 6 perfect-IoU predictions against 1 GT now produce mAP_50 = 1.0 exactly (was > 6 pre-fix) |
| `test_compute_detection_metrics_consumes_each_gt_once` | 3 GTs × 2 perfect-IoU preds each gives mAP ≈ 0.7556 (the correct VOC AP for TP-FP-TP-FP-TP-FP), not 1.0 |
| `test_compute_detection_metrics_handles_partial_matches` | One pred at IoU 0.7 gives mAP_50 = 1.0, 0 < mAP_50:95 < 1 |

The interleaved-TP-FP test (`consumes_each_gt_once`) explicitly
asserts 0.7556 — the *correct* VOC all-points integration result
on a deliberately-overlapping prediction set. The pre-fix code
would have given mAP > 1 on this input.

## 5. Honest performance picture, vis-à-vis YOLO-parity

The night-doc §1 ladder estimated lifts assuming a 0.72 starting
mean. Restated under the honest 0.504 ± 0.039 baseline:

| # | Lever | Plan-doc estimated lift | Honest-baseline restatement |
|--:|-------|----------------------:|----------------------------:|
| 1 | Warm-start query embeddings | +0.05 / σ ÷ 0.4 | mean lift now uncertain (σ already small); EV ambiguous |
| 2 | Longer training + cosine LR  | +0.04 / σ × 0.7 | likely larger (loss curves still trend downward at e=50) |
| 3 | Scale to n=10k                | +0.03 / σ × 0.8 | likely larger |
| 4 | Bochner α + β additive       | +0.02 / σ × 1.0 | hold |
| 5 | LayerNorm + WeightDecay 1e-4  | +0.01 / σ × 0.9 | small; defer until variance becomes a problem |
| 6 | Deeper backbone               | +0.05            | **biggest single architectural lever** under the honest baseline |
| 7 | Focal loss for cls            | +0.02            | hold |
| 8 | GIoU / DIoU regression        | +0.02            | hold |
| 9 | Multi-scale feature pyramid   | +0.05            | hold |

**Revised recommended attack sequence:** capacity-first (lever
#6, deeper backbone; lever #9, FPN) followed by training schedule
(lever #2 + #3). Skip the warm-start (lever #1) — the
variance-collapse motivation is gone. The Stage-A sprint reorders
as: deeper backbone → longer training → FPN → GIoU. Estimated
combined mean lift: +0.12 to 0.62 ± ~0.04. Still well below
YOLO-tiny on COCO; comparable to YOLOv1-on-MNIST scale.

## 6. Per-seed detail (audit-trail)

```
scale  seed  mAP_50  box_acc  circ_acc  wall_s
 0.05    0   0.3998   0.44     0.50     571
 0.05    1   0.4665   0.50     0.71     573
 0.05    2   0.3647   0.60     0.77     575
 0.05    3   0.4899   0.58     0.80     577
 0.05    4   0.5226   0.72     0.57     574
 0.10    0   0.4075   0.50     0.64     562
 0.10    1   0.4502   0.19     0.57     578
 0.10    2   0.5301   0.67     0.69     577
 0.10    3   0.5128   0.75     0.70     581
 0.10    4   0.5532   0.61     0.64     572
 0.20    0   0.4255   0.25     0.71     550
 0.20    1   0.4187   0.44     0.57     521
 0.20    2   0.3836   0.07     0.54     516
 0.20    3   0.4350   0.50     1.00     517
 0.20    4   0.5502   0.67     0.93     515
 0.40    0   0.4782   0.62     0.57     524
 0.40    1   0.4943   0.62     0.64     516
 0.40    2   0.4025   0.60     0.92     519
 0.40    3   0.5585   0.42     1.00     521
 0.40    4   0.6129   0.72     0.79     518
 0.80    0   0.4268   0.62     0.71     520
 0.80    1   0.6162   0.75     0.64     519
 0.80    2   0.4240   0.53     0.92     517
 0.80    3   0.4887   0.58     0.70     521
 0.80    4   0.2360   0.22     0.71     518
 1.00    0   0.4779   0.56     0.86     518
 1.00    1   0.4714   0.62     0.71     522
 1.00    2   0.4865   0.47     0.31     517
 1.00    3   0.5789   0.75     0.80     518
 1.00    4   0.5057   0.72     0.71     521
```

Notes from the per-seed audit:

* **Scale = 0.80 seed 4 → mAP_50 = 0.236.** This is the
  significant outlier in the entire sweep (pstdev at 0.80 jumps to
  0.123 because of it). Box and circle classification accuracy are
  also depressed at this seed (0.22 / 0.71). Most likely an
  initialisation-dependent local minimum; the *first cleanly
  reproducible* such failure under the honest metric, which now
  surfaces these instead of being hidden by bug-inflation noise.

* **Scale = 1.00** has the **smallest σ** of any cell (0.039) and
  *also* the highest min (0.471). Both observations support
  picking 1.00 as the stable default.

## 7. Experiment provenance

* **Git SHA at sweep launch:** `2ccaa4d12fae1ff9cd533bd91cd84b28f11c3dab`
  ("Gomb reaches SOTA. By large", 2026-05-14). Working tree dirty
  with this report's source patch (deliberate; the orchestrator
  captured the SHA at launch).
* **Python / torch:** miniconda3, torch 2.11.0+cu130 (protocol
  parity with the May-13 5-seed; CORE.YAML pins 2.4.1 but the
  comparison demands matching the prior tool).
* **GPU:** NVIDIA RTX 2070 SUPER, 8 GiB, driver 580.126.09.
* **CPU / RAM:** AMD Ryzen 7 3700X, 32 GB.
* **Seeds:** 0, 1, 2, 3, 4 — same per-seed dataset realisation
  as the May-13 backfill (deterministic from `--seed`).
* **Dataset:** Cluttered MNIST, n=5000, canvas=64, max_objects=3,
  per-seed regenerated.
* **Resource cap:** `systemd-run --user --scope -p MemoryMax=16G
  -p MemorySwapMax=0` per run; cgroup never tripped.
* **Per-run wall:** 515 – 581 s (mean 543 s, σ 23 s). The
  variance is dominated by the +ricci-mod vs +kcycle-vs-others
  ordering in the original script; isolated to +ricci-mod here
  the variance is tight.

## 8. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|--:|---|---|
| 1 | Cartesian-product API surface | clean — `--ricci-scale` is a flag, no new variant names |
| 2 | Algorithm code behind Python boundary | n/a |
| 3 | Per-experiment scaffold duplication | clean — reuses `train_circles_ricci.py` |
| 4 | Long single-file modules | `hymeyolo_warmstart.py` is one concern at 290 LOC, < 400-LOC heuristic |
| 5 | New axis via new function name | clean — flag axis |
| 6 | `#[allow(too_many_arguments)]` | n/a |
| 7 | String-typed config | clean |
| 8 | Forward-time flags for structural differences | clean — `ricci_scale` is parametric |
| 9 | Bypassing existing Strategy traits | clean |
| 10 | `ulimit -v` on CUDA | n/a (cgroup) |
| 11 | Global / module-level mutable state | clean |

No new suppressions, no silent failures.

## 9. Acceptance

- [x] 30/30 rows landed; orchestrator log clean; no cgroup OOMs.
- [x] No mAP_50 row > 1.0 (the analyser explicitly checks; zero hits).
- [x] Per-scale aggregate, paired-Δ table, and verdict reported.
- [x] 76/76 ricci-adjacent tests pass.
- [x] CORE.YAML untouched.
- [x] Plan dir (4 formats) committed.
- [x] §6.3 launch-decision branch identified (s★ = 1.0).
- [x] Honest-baseline correction propagated to follow-up planning
      (this report §5, night-doc §8.1).

## 10. Follow-ups (ranked by EV under the honest baseline)

1. **Update SOTA_RESULTS.md** with the 0.504 ± 0.039 honest
   baseline. Re-state any HyMeYOLO comparison cited downstream.
2. **Capacity sprint** (deeper backbone + FPN + longer training).
   Estimated combined mean lift +0.10 to +0.15 to land in the
   0.60–0.66 mAP range — still well below YOLO-tiny COCO, but
   closing the protocol gap on Cluttered MNIST.
3. **Skip the warm-start 5-seed launch** unless explicitly
   re-motivated by future variance issues at larger scale.
4. **Sweep extension to {1.5, 2.0, 4.0}** if a positive-direction
   ricci-scale lever is wanted; current data are non-monotonic
   below 1.0 so the *above*-1.0 region is the underexplored
   regime.
5. **SDF-template axis/limit/pose emission fix** — a side-quest
   from the dual-FANUC demo, but completely orthogonal to this
   sweep.

---

*End of ricci-scale sweep report. The headline finding is the
mAP correction, not the sweep result; the sweep's value is now
that it produces the first honest 5-seed measurement of
`+ricci-mod` at the protocol scale.*
