# Stage D-3-BREAK — C8 5-seed validation + Phase 3 launch

**Date:** 2026-05-21
**Plan:** [`docs/plans/2026-05-21-voc-d3-break-phase3/`](../docs/plans/2026-05-21-voc-d3-break-phase3/) (4-format)
**Verdict:** **CONFIRMED WIN.** C8 recipe 5-seed mean
mAP$_{50}$ = **0.0552 ± 0.0146** on VOC2007 trainval — **3.6× over
the published D-3-bis baseline of 0.0153**.  All 5 seeds clear the
single-seed-luck falsifier; all 5 clear the partial-win threshold
(0.020).  Phase 3 launched to probe whether the unplateaued descent
at ep 60 has more headroom.

## 1. Summary

Phase 1 (PID 893163, 2026-05-21 03:03–05:15 CEST) ran an 8-cell
single-seed grid over the axes ($\lambda_{\text{gate}^-} \in \{1, 2, 5\}$,
epochs $\in \{30, 60\}$, $n_q \in \{6, 12\}$).  Winner: **C8** at
$\lambda{=}2.0$, 60 epochs, $n_q{=}6$, single-seed mAP$_{50}$ = 0.0567.

A 5-seed re-run of C8 (PID 924539, 2026-05-21 12:14–14:02 CEST)
turned that single-seed win into a publication-quality claim:

| metric | value |
|---|---|
| n            | 5 |
| mean mAP$_{50}$ | **0.0552** |
| pstdev       | **0.0146** |
| per-seed     | 0.0358, 0.0753, 0.0434, 0.0673, 0.0540 |
| lift vs D-3-bis (0.0153) | **+0.0399 (3.6×)** |
| lift / sd ratio | 2.73 |

### Falsifier gates — all passed

| gate | threshold | actual | result |
|---|---|---|---|
| single-seed luck | mean ≥ 0.030 | 0.0552 | ✓ 1.8× |
| brittle recipe   | σ ≤ 0.020    | 0.0146 | ✓ |
| partial-win      | mean ≥ 0.020 | 0.0552 | ✓ 2.8× |
| all-seeds-clear-baseline | all 5 > 0.0153 | min = 0.0358 (2.3×) | ✓ |

## 2. The winning recipe (drop-in CLI)

```
python -m signedkan_wip.src.vision.train_voc_stagec \
  --image-set trainval --epochs 60 --input-size 224 \
  --batch-size 8 --n-box-queries 6 \
  --lr 0.003 --seed {0..4} \
  --device cuda \
  --backbone resnet18_imagenet \
  --query-head-kind nodelet \
  --lam-gate-neg 2.0 \
  --gate-loss-kind bce
```

vs the published D-3-bis recipe, the four changes were:
- `--n-box-queries 12 → 6` (provisioning 5× → 2.5×)
- `--epochs 30 → 60`
- `--lam-gate-neg 1.0 → 2.0` (gate suppression 5× stronger)
- single-seed → 5-seed paired

## 3. Phase 1 diagnostic — why $n_q = 6$ wins

From the Phase 1 8-cell grid (sorted by mAP$_{50}$):

| cell | $\lambda$ | ep | $n_q$ | mAP | mIoU | cls_acc | over-prov | loss_end | drop % |
|---|---|---|---|---|---|---|---|---|---|
| **C8**  | **2.0** | **60** | **6** | **0.0567** | 0.291 | 0.875 | **2.49×** | **3.094** | **34.3** |
| C5  | 2.0 | 60 | 12 | 0.0510 | 0.270 | 1.000 | 4.98× | 3.147 | 30.9 |
| C4  | 1.0 | 60 | 12 | 0.0505 | 0.267 | 1.000 | 4.98× | 2.977 | 29.0 |
| C7  | 1.0 | 60 | 6  | 0.0412 | 0.303 | 1.000 | 2.49× | 3.150 | 27.9 |
| C3  | 5.0 | 30 | 12 | 0.0178 | 0.238 | 0.750 | 4.98× | 4.100 | 21.1 |
| C2  | 2.0 | 30 | 12 | 0.0159 | 0.230 | 0.625 | 4.98× | 3.712 | 18.9 |
| C6  | 1.0 | 30 | 6  | 0.0119 | 0.280 | 0.800 | 2.49× | 3.621 | 17.0 |
| C1  | 1.0 | 30 | 12 | 0.0094 | 0.233 | 0.750 | 4.98× | 3.457 | 17.7 |

Three concurrent mechanisms explain the C8 win:

1. **Provisioning is the right knob.**  At $n_q{=}12$, the
   matcher concentrates on $\sim 2{-}3$ queries per image
   (VOC averages 2.4 GT/image), and $9{-}10$ "dead-weight" queries
   accumulate spurious-high gates that pollute the precision-recall
   curve.  The "perfect" cls_acc=1.000 at $n_q{=}12$ is a *symptom*
   of over-selection, not a feature.  At $n_q{=}6$, all queries are
   exercised — better gradient distribution, better localization
   (mIoU 0.291 vs 0.270).
2. **Stronger gate suppression compounds with longer training.**
   $\lambda{=}2.0$ at 30 ep (C2) gives 0.0159 (close to baseline);
   $\lambda{=}2.0$ at 60 ep (C8) gives 0.0567 (3.6× lift).  The
   gate-balance recipe needs $> 30$ epochs to converge.
3. **The match-cls-noise / overall-mAP decoupling.**  C8's matched
   cls_acc is 0.875 — *lower* than C5/C4/C7's 1.000 — yet C8 has
   the highest mAP.  The classifier is allowed to spread its
   per-class confidence more honestly; the gate handles the false-
   positive suppression separately.  This is the inverse of the
   D-3-tris / D-3-quater "cls_acc improves but mAP regresses"
   regime.

## 4. Loss trajectory — none of the 60-epoch cells have plateaued

```
cell                  ep 0   ep10   ep20   ep30   ep40   ep50   ep59
C8 (λ=2, n_q=6)       4.71   4.27   4.03   3.82   3.60   3.34   3.09
C5 (λ=2, n_q=12)      4.56   4.05   3.84   3.64   3.48   3.30   3.15
C4 (λ=1, n_q=12)      4.20   3.72   3.51   3.36   3.23   3.08   2.98
C7 (λ=1, n_q=6)       4.37   3.98   3.78   3.62   3.46   3.30   3.15

Δ (ep50 → ep59):
  C8: −0.245     ← steepest recent descent
  C7: −0.149
  C5: −0.157
  C4: −0.101
```

**C8 has not plateaued at ep 60** and has the steepest recent slope.
This is the headline argument for Phase 3 cell C9 (90 epochs).

## 5. Phase 3 — 3-cell extrapolation grid (running)

Launched 2026-05-21 17:48 CEST as PID 941355 from
`signedkan_wip/experiments/run_voc_d3_break_phase3_2026_05_21.sh`.
Log dir
`signedkan_wip/experiments/results/voc_d3_break_phase3_20260521T154837Z/`.

| cell | $\lambda_{\text{gate}^-}$ | epochs | $n_q$ | tests |
|---|---|---|---|---|
| C9  | 2.0 | **90** | 6 | unplateaued-descent hypothesis (Section 4) |
| C10 | 2.0 | 60 | **4** | further provisioning reduction (VOC median ~1.7 GT/img) |
| C11 | **5.0** | 60 | 6 | Phase-1 C3 "λ=5 at 30 ep was undertrained" hypothesis |

Falsifier verdicts embedded in the orchestrator's aggregator:

- C9 > 0.06 → more epochs help; queue 5-seed of C9.
- C10 > 0.055 → push $n_q \to 3$; otherwise provisioning is at the floor.
- C11 > 0.055 → $\lambda{=}5$ is the actual sweet spot.

Wall budget ~75-85 min (C9: ~33 min, C10: ~22 min, C11: ~22 min).

## 6. Files touched

| File | Status | LOC |
|---|---|---|
| `docs/plans/2026-05-21-voc-d3-break-phase3/{plan.tex,plan.pdf,plan.tikz,plan.mmd}` | new | 4-format plan |
| `signedkan_wip/experiments/run_voc_c8_5seed_2026_05_21.sh` | new | 5-seed validation orchestrator (already done) |
| `signedkan_wip/experiments/run_voc_d3_break_phase3_2026_05_21.sh` | new | Phase 3 orchestrator |
| `signedkan_wip/experiments/results/voc_c8_5seed_20260521T101436Z/` | new | 5-seed result JSONL |
| `signedkan_wip/experiments/results/voc_d3_break_phase3_20260521T154837Z/` | running | Phase 3 result JSONL |
| `reports/2026-05-21-voc-d3-break-c8-5seed.md` | new | this report |

## 7. CORE.YAML items touched

None.

## 8. §6.5 anti-pattern audit

- **No new code.** Phase 3 reuses the existing
  `train_voc_stagec.py` CLI surface that was already exercised by
  Phase 1 and the C8 5-seed.
- **No new tests.** The Phase 1 grid + C8 5-seed are the validation
  artefacts.
- **Orchestrator is shell-only**, ~80 LOC, single-purpose.  No
  Cartesian-product wrapper functions; the cells are explicit
  `run_cell <args>` lines.
- The recipe knobs (`--lam-gate-neg`, `--gate-loss-kind`,
  `--n-box-queries`, `--epochs`) were *already* config flags
  (anti-pattern §5 was avoided when D-3 and D-3-bis added them).

Clean.

## 9. Open follow-ups

1. **Phase 3 aggregation** (when it finishes ~19:05 CEST today): if
   any cell beats C8 by > +0.005, 5-seed-validate that cell.
   Otherwise C8 is the recipe.
2. **5-seed paired vs published baseline.**  The current 5-seed
   uses seeds 0-4 at the *new* recipe.  A paired-Δ vs D-3-bis seeds
   0-4 (already on disk) would tighten the statistical claim from
   "+0.0399 lift / sd 0.0146" to a real paired-z.
3. **5-seed of seeds at higher input resolution.**  D-3 series was
   all at 224×224; YOLOv5/v8 conventions use 320 or 416.  Higher
   resolution is a known mAP lever.
4. **Visit-gate gap.**  Current best is 0.0552; visit gate is 0.20
   (3.6× short).  Phase 3 + resolution sweep are the next levers
   before pivoting to backbone / FPN.

## 10. Experiment provenance

- **Git SHA:** `507d7e24` (uncommitted; same SHA across Phase 1,
  C8 5-seed, Phase 3).
- **GPU:** RTX 2070 SUPER 8 GiB.
- **OS:** Ubuntu 24.04, kernel 6.17.
- **Memory cap:** `systemd-run --user --scope -p MemoryMax=16G`
  (cgroups v2 RSS gate).  Observed peak ~6 GiB per cell.
- **Seeds:** 0-4 for the 5-seed validation.
- **Dataset:** VOC2007 trainval, 5011 images, 12085 GT boxes,
  20 classes, 80/20 train/val split inside the trainval set.
- **Wall:** ~22 min/seed × 5 = ~110 min C8 5-seed; ~75 min Phase 3.

## 11. Acceptance check

- [x] Plan in 4 formats on disk.
- [x] CORE.YAML items touched = 0.
- [x] C8 5-seed validation: passes all 3 falsifier gates.
- [x] Phase 3 orchestrator launched, results being written to disk.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
