---
name: project-seminar-demos-and-hymeyolo-plan
description: Plan to finish the seminar inference demos (6) + get HyMeYOLO right; spec in docs/INFERENCE_DEMOS_OUTLINE.md
metadata: 
  node_type: memory
  type: project
  originSessionId: 80ba693f-ca45-4545-9682-55fb60a97d78
---

Plan to land the **seminar inference demos** and **HyMeYOLO** for the talk.
Spec: `docs/INFERENCE_DEMOS_OUTLINE.md` (6 demos, inference-first build
order, single-CLI). Cross-cutting acceptance every demo: one CLI
(`python -m signedkan_wip.src.demo <mode> --device auto --seed 0`),
deterministic, print peak RSS + wall + assert < 16 GB, write to
`demo_out/<mode>/`, and only print a number if it can name the
checkpoint/file it came from.

**The real build lives in `signedkan_wip/demos/seminar/`** (one CLI:
`python -m signedkan_wip.demos.seminar <demo>`, `SeminarDemo` Protocol +
`DemoRunner` enforcing seed/device/peak-RSS/16 GB). Spec: `signedkan_wip/demos/
SEMINAR_DEMOS.md` (5 demos) + plan `docs/plans/2026-06-10-seminar-demos-remaining/`
(items 4–7). NOTE: the INFERENCE_DEMOS_OUTLINE.md numbering differs from the
seminar-package numbering — go by the registered `DEMOS` dict, not the outline.

**Status (2026-06-12):**
- `link` (link-prediction inference) — **DONE** (build-item 1, AUC 0.9957 otc).
- `hive` (HIVE compile + canonicalisation) — **DONE** (build-item 3).
- `latency` (forward-latency bars) — **DONE 2026-06-12**: wraps the 2026-06-11
  measurement (`run_inference_bench.py`+`bench_to_png.py`) as a CLI mode; reads
  committed `inference_bench.json`, headline mean joint/lean **3.58× CPU /
  1.14× CUDA**; honest "not faster than SGCN" line. `reports/2026-06-12-seminar-latency-demo.md`.
- `balance` (item 5, affective-balance opener) — **DONE 2026-06-12**: cycles via
  `enumerate_top_k_cycles_rs`, balance = fraction of cycles with +sign-product
  (classified on the **product in Python**, NOT the `fraction_negative` score —
  a 2-neg-edge cycle is balanced). Fixtures `--graph {planted(0.5,triad{0,1,2}),
  camps(1.0),karate(1.0)}`. `reports/2026-06-12-seminar-balance-demo.md`.
- **`mesh`** (item 6, the only new algorithm — Sinkhorn matcher + signed-vs-
  unsigned chiral ablation; discovery-grep `sinkhorn|optimal.transport` first)
  — **NEXT, TODO**.
- `bridge` (item 7, `.hymeko`→IR→perception thin closer) — TODO.

Old INFERENCE_DEMOS-numbered TODOs still outside the seminar package: kinematic
compass/pose, Gömb signature+label-shuffle, HyMeYOLO detection (heaviest;
quote corrected mAP 0.903), `.hymeko`→URDF/SDF/MJCF transform.

**HyMeYOLO ("get it right"):** demo at
`signedkan_wip/src/vision/demo_hymeyolo_tk.py` (+ `vision/DEMO_README.md`,
`launch_demo.sh`). Finish = (a) `--checkpoint checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt`
**skips training** (no-ckpt path quick-trains 30 ep — always pass the ckpt
for the demo); (b) headless mode rendering N panels (GT cyan / pred red) to
`demo_out/yolo/` + mAP_50 + latency line for slide capture. Acceptance:
mAP_50 ~0.90 band on the demo split, honest consumed-GT COCO matching.
**Quote the corrected mAP 0.903 ± 0.009, never the bug-inflated 0.723.**
Separate track — the VOC held-out baseline is still a floor (ep60 = 0.0149,
needs full ep180 on Komondor, DataLoader is host-bound): see
[[project-voc-hymeyolo-baseline]]. Conv-as-hypergraph plan on disk:
`docs/plans/2026-06-11-conv-as-hypergraph-hymeyolo/`.

**The "other things" in flight (don't drop):** the SISY control paper is
6 pp, review-corrected, ROS demo runs — [[project-sisy2026-control-paper]];
the no-leakage E1 grid (Bitcoin done, Epinions/Slashdot next) —
[[project-no-leakage-benchmark-resume]].

**Why:** the seminar talk needs runnable, honest, reproducible demos with
named-source numbers; HyMeYOLO is the "primitive transfers to vision"
slide and must show the corrected mAP, not the old inflated bug number.

**How to apply:** build in the listed order (Demo 1 → 3 → 5 → 4 → 6),
each as a ~20-line config on the shared harness (CLAUDE.md §6.5 #3 — no
per-demo scaffold reimplementation). Confirm every demo meets the
cross-cutting acceptance before moving on.
