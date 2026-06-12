# VOC2007 Held-Out Test Baseline for B9 (reduced ep60) + Seminar Panels

**Date:** 2026-06-10
**Plan:** `docs/plans/2026-06-10-voc-test-baseline/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan`

## Summary

Produced the first **held-out VOC2007 test-split** number for the
HyMeYOLO B9 recipe — the prior work's top-ranked, never-done next move
(all committed B9 numbers were on `trainval`). Because no B9 checkpoint
existed on disk (the 5-seed run saved only JSONLs), one seed was
retrained. On the user's decision (laptop GPU makes full ep180 a ~4 h
run, 2.8× the original hardware), a **reduced ep60** model was trained.

**Held-out result: VOC2007 test mAP_50 = 0.0149** (4952 images), vs the
ep60 trainval 0.0195 and the full-B9 ep180 trainval 0.1092. The
train→test gap is small; the ep60-vs-ep180 undertraining gap dominates
(ep60 loss dropped 20% vs B9's 47%). Six real-photo test panels were
rendered for the seminar slide. The number is low and honest — accepted
for the slide framing (structural transfer, not accuracy).

## Files touched

| File | Change | Lines |
|---|---|---|
| `signedkan_wip/src/vision/eval_voc.py` | **new** — held-out eval + VOC panel render | +228 |
| `signedkan_wip/tests/test_eval_voc.py` | **new** — 6 tests (unit/integ/cuda regression) | +175 |
| `signedkan_wip/src/vision/train_voc_stagec.py` | edit — save `query_head_kind`; fix F541 | +4/−1 |
| `docs/plans/2026-06-10-voc-test-baseline/` | **new** — 4 plan artifacts | — |

`eval_voc` reuses `VocPersonDetector` (model reconstruction + nodelet
decode), `compute_detection_metrics` (corrected COCO matcher),
`load_voc_hungarian`, and `_peak_rss_mb` — no new algorithm code.

## CORE.YAML / dependencies

None touched; no new dependency (VOC via torchvision, already present).

## Test results

`pytest -p no:randomly`. **6 passed** (`test_eval_voc.py`): checkpoint
guard, `n_panels` guard, `_draw_voc_panel`, synthetic-ckpt eval + panels
on real test images, and a **cuda regression test** added after a device
bug (GT tensors must follow the model to cuda — the cpu test could not
catch it; fixed in `eval_checkpoint_on_split`).

## Performance / provenance

- Train (ep60, 1 seed): 4444 s train + 75 s load = **74 min**, RTX 3070
  Laptop, 2.0/8.2 GB VRAM (no OOM), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.
- Test eval (4952 img, 320px): wall 105 s, peak RSS 7.3 GB (< 16 GB cap).
- §11 wall reconciliation: measured 79 s/epoch here vs 28.5 s/epoch on
  the original card (2.8×, hardware — laptop throttling + host-bound
  per-batch transfer). Reconciled before launch; reduced to ep60.
- Windows note: `systemd-run` RSS cap unavailable; the Phase-7 lazy
  loader bounds RSS structurally (X stays on CPU); peak monitored via
  `_peak_rss_mb` ctypes probe. Unicode `→` print crash fixed with
  `PYTHONIOENCODING=utf-8`.
- Checkpoint: `signedkan_wip/experiments/results/voc_b9_ep60_seed0/stage_d1_voc_seed0.pt`
  (label `stage_d1_voc`, ep60, nodelet, resnet18_imagenet, 320px).
- Panels: `demo_out/voc/voc_panel_{00..05}.png` (gitignored, threshold 0.10).
- Seed 0; git SHA `af803ee` (dirty tree — this change set).

## §6.5 anti-patterns

None introduced. Reused the existing detector + metric rather than
re-implementing (avoids #1/#9); `eval_voc` is one module with a
`--mode {metrics,panels,both}` arg (#13).

## Open issues / follow-up

- **ep60 is undertrained** (0.0149 test). The full ep180 B9 (~4 h here,
  or Komondor HPC) would land nearer the trainval 0.11 band; not run by
  user choice. GCP was considered and declined — the job is host-bound
  (37% GPU util), so a faster GPU would not help without a `DataLoader`
  (pinned-memory prefetch) rewrite.
- The device bug shows the integration test should exercise cuda when
  present; the regression test now does.
