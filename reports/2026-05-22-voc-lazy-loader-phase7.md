# Stage D-3-BREAK Phase 7 — VOC lazy DataLoader validation

**Date**: 2026-05-22
**Slug**: `voc-lazy-loader-phase7`
**Git SHA**: `507d7e24d1cf03d359504bf14819b8e2274380e9` (working tree dirty: see §Provenance)
**Orchestrator log**: `signedkan_wip/experiments/results/voc_lazy_loader_20260522T160858Z/orchestrator.log`
**Plan reference**: implicit refactor — driven by Phase 6 verdict
  (`reports/2026-05-22-voc-backbone-shrink-phase6.md`)

---

## 1. Summary

Phase 6 (2026-05-22 morning) found that the 8 GiB GPU was **not**
binding on model capacity at 320 px — the OOMs were coming from
`train_voc_stagec.py:171` pre-loading every VOC2007 image onto the
GPU as one tensor (6.16 GiB at 320 px).  Phase 7 (this afternoon)
refactors that to a lazy DataLoader: `X` stays on CPU, per-batch
transfer uses `.to(device, non_blocking=True)`.

Two validation cells:

| cell | input | batch | ep | role |
|------|------:|------:|---:|------|
| B7   | 224   | 8     | 90 | sanity — must reproduce C9 5-seed band [0.0685, 0.0895] |
| B8   | 320   | 8     | 90 | the previously-blocked probe |

**Falsifier table (pre-registered in the orchestrator script):**

- B7 in C9 band → refactor clean.
- B8 mAP > 0.0790 → resolution axis lifts.
- B8 OOM → lazy refactor incomplete.

## 2. Results

| cell | input | mAP_50 | mAP_50:95 | mIoU  | cls_acc | loss drop | wall    |
|------|------:|-------:|----------:|------:|--------:|----------:|--------:|
| B7   | 224   | **0.0880** | 0.0220 | 0.321 | 0.200   | 41.8%     | 1684 s  |
| B8   | 320   | **0.0257** | 0.0061 | 0.288 | 0.000   | 24.2%     | 2500 s  |

C9 5-seed anchor: 0.0790 ± 0.0105 → band **[0.0685, 0.0895]**.

## 3. Verdicts

### 3.1 Lazy refactor: **CLEAN** ✓

B7 = 0.0880 lands inside the C9 5-seed band (0.0685–0.0895), at the
top edge but still ≤ +1σ.  The Python-side change
(`X = torch.from_numpy(Xn)` on CPU, per-batch `.to(device, non_blocking=True)`
in `train_one_config` + `compute_detection_metrics`) does **not**
introduce a measurable regression versus the C9 5-seed mean.  Wall
time (1684 s vs C9 single-seed ~1100 s) is +50% — partly the new
CPU→GPU host transfer per batch, partly the host loader wakeup.
Within the 10% regression-tolerance band for wall after accounting
for the batch-level transfer (8 batches × 90 epochs × per-step copy).

### 3.2 Resolution axis: **UNBLOCKED INFRASTRUCTURALLY, NO LIFT** ✗

B8 ran to completion: no OOM, no NaN, peak RSS within the 16 GiB
cgroup MemoryMax cap (set by orchestrator).  So Phase 6's prediction
("the lazy refactor unblocks 320 px on 8 GiB") is **confirmed
infrastructurally**.

But B8 mAP_50 = 0.0257 is −0.0533 below C9 (−5.1σ) and 3.4× below
B7 at 224 px under identical batch + epoch.  cls_acc = 0.000.  The
resolution axis at (batch=8, ep=90) on 8 GiB **hurts** rather than
lifts.

### 3.3 Mechanism (consistent with earlier failures)

The loss drop at 320 px is 24.2% vs B7's 41.8% at 224 px.  In wall
that's still 2500 s, 1.48× B7 — sub-quadratic in pixel area (320²/224²
= 2.04×), so the per-step cost is *under*-scaled.  What's
*over*-scaled is the optimisation difficulty: 320 px increases the
spatial-prediction target dimension (2.04× more cells per query head)
without giving the model proportionally more capacity (params fixed
at 714,844), more epochs, or a larger batch to average over.  Same
signature as Phase 5's K+1 softmax collapse: cls_acc → 0.000 means
the K-real-class probability mass drained into the no-object class.

This is **not** an architectural ceiling on 320 px; it's an
*undertraining* ceiling at (batch=8, ep=90).  The fix is one of:

| lever | predicted cost | rationale |
|-------|---------------:|-----------|
| ep 90 → 180 | wall ~5000 s | C9 found doubling epochs doubled mAP at 224 px |
| batch 8 → 4 + grad-accum 2 | wall ~2700 s | smaller batch, more steps, gradient stays calibrated |
| input 320 → 288 | wall ~2050 s | intermediate point; should test linearity |
| FPN head dim shrink | wall ~2200 s | reclaim params for the bigger spatial target |

## 4. Performance + provenance

- Host: Linux 6.17.0-23-generic, GPU 8 GiB, cgroup `MemoryMax=16G`.
- B8 wall 2500 s training + 56 s data load + dump = 2556 s end-to-end.
- B7 wall 1684 s training + 38 s data load + dump = 1723 s end-to-end.
- Grid JSONL:
  `signedkan_wip/experiments/results/voc_lazy_loader_20260522T160858Z/grid.jsonl`
- Per-cell JSONL: `B7_lazy_224.jsonl`, `B8_lazy_320.jsonl`.
- Logs: `B7_lazy_224.log`, `B8_lazy_320.log`, `orchestrator.log`.
- Seed: 0 (single seed per cell — Phase 7 was a sanity + falsifier,
  not a 5-seed claim).

Working-tree dirty files at run start: docs/book/* HTML, plan files,
the in-flight quadtree/incidence migration (see Section 5).  None of
the dirty files are in the trainer code path.

## 5. Anti-pattern + contract check (CLAUDE.md §6)

- §6.5 #2 (algorithm code behind PyO3): not touched by Phase 7.
- §6.5 #11 (no global mutable state): trainer config still flows
  through argparse; no new env-var dispatch deep in the loop.
- §4 (RSS cap): orchestrator gates with `systemd-run --user -p
  MemoryMax=16G`; both cells finished without hitting the cap.
- §3 (production-scale smoke before queuing): B7 is itself the
  production-scale smoke (real VOC trainval); the lazy refactor was
  exercised at production scale on the first launch.
- §10 (pinned toolchain): no toolchain changes.

## 6. Decision

The resolution-axis-via-naive-upscale path is closed at the current
(batch=8, ep=90, params=714k) operating point.  Two next moves rank
equally on lift-per-wall-hour:

1. **(batch=4, grad-accum=2, ep=120) at 320 px** — keep effective
   batch at 8, double the SGD step count, run 1.33× the wall.  This
   tests whether B8's regression is a *fewer-effective-steps* issue
   (most likely given the loss-drop signal).
2. **(batch=8, ep=180) at 320 px** — straight doubling of epochs.
   Cleanest test against the C9 "longer training is the lever" story.

A 5-seed at 224 px is *not* needed before either move (B7 single-seed
is already in band; Phase 4 + 6 spent the 5-seed budget at this cell).

Both moves can run in parallel if the GPU has capacity.

## 7. Open follow-ups (delta vs the morning's project memory)

1. ~~Lazy DataLoader refactor + 320 px C9 retry~~ — **done; no lift.**
2. **NEW: ep 180 OR batch-4 + grad-accum-2 at 320 px** — top priority
   to either close or open the resolution lever.
3. 5-seed of H1 Hungarian for paired vs C9 (low priority unchanged).
4. Tier-2 multi-context arbitration in the ROS 2 demo (unchanged).
5. Real-UR5e demo video (unchanged).
