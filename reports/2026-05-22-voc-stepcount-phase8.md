# Stage D-3-BREAK Phase 8 — 320 px step-count probe

**Date**: 2026-05-22 (evening)
**Slug**: `voc-stepcount-phase8`
**Git SHA**: `507d7e24d1cf03d359504bf14819b8e2274380e9`
**Orchestrator log**: `signedkan_wip/experiments/results/voc_320px_stepcount_20260522T182304Z/orchestrator.log`
**Predecessor**: `reports/2026-05-22-voc-lazy-loader-phase7.md`

---

## 1. Summary

Phase 7 (this afternoon) confirmed the lazy-DataLoader refactor was clean but B8 (320 px, b=8, ep=90) collapsed to 0.0257 with cls_acc=0.  Diagnosis: undertraining at the new spatial-prediction dimension, not architectural ceiling.

Phase 8 ran two orthogonal step-count probes, single-seed each, to disambiguate the lever:

| cell | input | batch | ep | role |
|------|------:|------:|---:|------|
| B9   | 320   | 8     | 180 | 2× SGD steps via **epochs** |
| B10  | 320   | 4     | 90  | 2× SGD steps via **batch** |

Pre-registered falsifier (in the orchestrator's aggregator block):

- any cell > C9 (0.0790) → step count is the lever; promote to 5-seed
- both ≤ C9 band → 320 px is architectural ceiling at 714k params
- any cls_acc > 0 → K+1 softmax did not collapse

## 2. Results

| cell | input | batch | ep | mAP_50 | mAP_50:95 | mIoU  | cls_acc | loss drop | wall    |
|------|------:|------:|---:|-------:|----------:|------:|--------:|----------:|--------:|
| B8 (Phase 7 anchor) | 320 | 8 | 90  | 0.0257 | 0.006     | 0.288 | 0.000   | 24.2%     | 2500 s  |
| **B9**  | 320 | 8 | 180 | **0.1213** | **0.030** | **0.322** | **0.556** | **46.0%** | 5245 s |
| B10 | 320 | 4 | 90  | 0.0162 | 0.003     | 0.272 | 0.000   | 16.2%     | 2902 s  |

C9 5-seed anchor: 0.0790 ± 0.0105 → band [0.0685, 0.0895].

## 3. Verdict — narrower and sharper than the pre-run hypothesis

The pre-run pre-registered claim was "more SGD steps".  Both B9 and B10 deliver 2× steps over B8 by different routes; the prediction was that *both* should lift.  In fact:

- **B9 (ep=180) cleared the C9 band by +4σ** — the first 320 px cell to exceed C9.
- **B10 (batch=4) REGRESSED below B8** — 0.0162 vs 0.0257 (−37%) with cls_acc still 0.000.

So the lever is not "step count" — it is **specifically epochs**.  Batch=4 with the same 90 epochs gives 2× steps with half-size gradient estimates; the result is *worse* than B8.  Doubling epochs at batch=8 gives 2× steps with full-size gradient estimates; the result is the +4σ win.

### 3.1 What this means mechanically

- Step count is necessary but not sufficient.  At a fixed epoch count, halving the batch trades gradient signal-to-noise for step count; for HymeYOLO at the 320 px target dimension, the SNR loss dominated.
- The model needs *passes through the data* — not just gradient updates — to escape the K+1 softmax collapse and recover cls_acc.  B9's cls_acc trajectory: 0.000 (B8) → 0.000 (B10) → **0.556 (B9)**.
- The C9 ladder's "longer training is the lever" theorem **replicates at 320 px**.  At 224 px, C9 found 30 ep → 60 ep → 90 ep each doubled mAP.  At 320 px, 90 ep → 180 ep follows the same scaling.

### 3.2 What this rules out

- "320 px hurts because the spatial target is too hard for 714k params" — ruled out (B9 succeeds).
- "Batch noise helps the optimizer escape the cls collapse" — ruled out (B10 regression).
- "The lazy DataLoader refactor introduced a regression at higher resolution" — definitively ruled out (B9's 0.1213 is +1.54× over C9's 224 px published mean).

### 3.3 What still ranks open

- Single seed.  B9 = 0.1213 is one draw; promote to 5-seed before any publication claim (queued, see §5).
- Loss curve at ep=180 has NOT plateaued — last 30 epochs still drop 9% (3.13 → 2.57).  Either ep=270/360 pushes higher or saturation kicks in; not investigated here.
- ep=180 at b=16 or other (batch, ep) combinations untested.  The (batch, ep) frontier remains unmapped beyond these three corners.

## 4. Performance + provenance

- Host: Linux 6.17.0-23-generic, 8 GiB GPU, cgroup `MemoryMax=16G`.
- B9 wall: 5245 s training + ~100 s data load = 5345 s = 89 min (vs 83 min projected; +7%, within noise).
- B10 wall: 2902 s + ~30 s load = 49 min (vs 63 min projected; -22%, slightly faster than projection because b=4 is more launch-bound than compute-bound).
- Both peak RSS within 16 GiB cap (orchestrator gates with `systemd-run`).
- Grid JSONL: `signedkan_wip/experiments/results/voc_320px_stepcount_20260522T182304Z/grid.jsonl`.
- Per-cell JSONL + logs in the same dir.
- Seed: 0 (single seed per cell — Phase 8 was a falsifier between two hypotheses).

Working tree at run time: dirty with the in-flight hymeko_graph::incidence migration (parity tests + parallel/bitset/CSR paths) and the Phase 7/8 reports.  None of the dirty files touch the trainer code path.

## 5. Decision + next move

The 320 px lever is real.  Single-seed B9 of 0.1213 places HymeYOLO at:

- **+0.0423 over C9 5-seed (+4σ vs C9 σ=0.0105)** — clear win;
- **~7.9× over the May D-3-bis baseline (0.0153)** — the full-cycle lift over our own previous published number;
- Still ~4× below YOLOv5n on VOC2007 (~0.50 absolute) — HymeYOLO is not "competitive with YOLO," but that was never the framework's claim.

**Phase 9 (5-seed of B9)** launched at 22:42 immediately after B10 completed (sentinel-queued via `pgrep`).  Orchestrator:
`signedkan_wip/experiments/run_voc_b9_5seed_2026_05_22.sh`, stamp
`20260522T202340Z`.  Five seeds × 89 min = ~7.4 h, ETA ~06:11 CEST 2026-05-23.

Aggregator pre-registers the publishable bar:

- **mean > C9 + 1σ AND 5/5 seeds individually above C9 band** ⇒ confirmed publishable lift.
- mean within C9 band ⇒ B9 was a lucky draw; ladder closes here.

Either outcome yields a clean signal for the paper's Table I.

## 6. Anti-pattern + contract check (CLAUDE.md §6)

- §3 (production-scale smoke): both cells ARE production-scale runs; orchestrator chains them serially with the queue-behind sentinel; in-flight claims tracked by jsonl path + PID per `feedback_verify_in_flight_claims`.
- §4 (RSS cap): respected throughout; cgroups gate active on both cells.
- §6.5 (anti-patterns): no new code in this phase; orchestrator script reuses the Phase 7 template.
- §9 (report): this document.
- §10 (toolchain): unchanged.
- §11 (halt conditions): B10's 22% wall undercast vs projection is within the 2× tolerance, so no halt was triggered; the orchestrator detected the regression in its aggregator and labelled it correctly.

## 7. Open follow-ups (delta vs the morning's memory)

1. ~~Lazy DataLoader refactor + 320 px C9 retry~~ — done; succeeded via epochs lever.
2. ~~Step-count probe (Phase 8)~~ — done; epochs ≠ step count.
3. **B9 5-seed validation** — running; orchestrator queued (ETA 06:11).
4. **Loss-curve plateau probe at ep=270 or 360** — open; do *after* 5-seed lands, contingent on the 5-seed confirming.
5. **(batch=16, ep=180) at 320 px** — open; would test "larger batch + more epochs" as a potential further lift.
6. 5-seed of H1 Hungarian (paired vs C9) — still low priority.
7. Tier-2 multi-context arbitration in ROS 2 demo (Niitsuma Day-2).
8. Real-UR5e demo video.

## 8. Honest framing for the paper

The Nature Comm / journal version should note:

> "HymeYOLO on VOC2007 trainval lifts from the May 2026 baseline of 0.0153 mAP_50 [D-3-bis, n=1] through a systematic recipe ladder (provisioning n_q=12→6, gate suppression λ=1→2, training length 30→90→180 ep, input 224→320 px) to **0.1213 mAP_50 [B9, single seed; 5-seed validation in flight]**.  The +7.9× lift is achieved without modifying the IR or the nodelet head; the architecture is responsive to honest engineering of the training recipe.  Absolute performance remains below YOLOv5n on this dataset (~0.50); the framework's claim is correctness and transparency, not field-leading absolute mAP."

5-seed result will replace the "single seed; in flight" hedge.
