---
name: project-voc-hymeyolo-baseline
description: HyMeYOLO VOC2007 held-out test baseline status + eval_voc tool
metadata: 
  node_type: memory
  type: project
  originSessionId: 7d38ee1f-9710-4c95-87d5-425cc349680a
---

On 2026-06-10 the first held-out VOC2007 **test** number for HyMeYOLO was
produced: **mAP_50 = 0.0149** (4952 images), from a **reduced ep60** B9
seed (full ep180 is ~4 h on the RTX 3070 laptop — 2.8× the original
hardware; user chose ep60). The full-B9 ep180 trainval reference is
0.1092. ep60 is undertrained (loss drop 20% vs B9's 47%), so this number
is a floor, not the recipe's ceiling.

New tool: `signedkan_wip/src/vision/eval_voc.py` — held-out eval +
real-photo panel render for any VOC checkpoint, reusing
`VocPersonDetector` + `compute_detection_metrics`. CLI:
`--mode {metrics,panels,both}`. 6 tests pass incl a **cuda regression
test** (GT tensors must move to the model device — a cpu-only test missed
this device bug). Checkpoint self-description fix: `train_voc_stagec.py`
now saves `query_head_kind` so nodelet checkpoints reload correctly.

Checkpoint: `signedkan_wip/experiments/results/voc_b9_ep60_seed0/stage_d1_voc_seed0.pt`.
Report: `reports/2026-06-10-voc-test-baseline.md`.

Follow-up for a publishable number: full ep180 on Komondor HPC (not GCP —
the job is host-bound at 37% GPU util; a faster GPU won't help without a
pinned-memory `DataLoader` rewrite). Related: [[project-sisy2026-control-paper]].
