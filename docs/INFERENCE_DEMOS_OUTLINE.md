# HyMeKo — Inference Demos Outline

A build spec for finishing the seminar demos, **inference-first**. Each demo
maps to a slide in `HyMeKo_Seminar.pptx`. Most scaffolding already exists —
this document says what to *reuse*, what to *finish*, and the acceptance gate
for each. **Do not rebuild what is listed under "Already on disk."**

> Ground rules (from `CLAUDE.md`)
> - Discovery pass before any new file; extend the `hymeko_neuro/experiments/demo/` package, do not fork it.
> - Fixed seeds, deterministic output, peak-RSS + wall-time reported on exit.
> - Honest reporting: every printed number maps to a checkpoint/artifact path; absolute latency is reported next to accuracy, not in place of it.
> - 16 GB RSS cap; demos must run on CPU (cuda optional, auto-detected).

---

## 0. Shared inference harness (already on disk — reuse)

| Component | Path | Role |
|---|---|---|
| `ModelBundle`, `predict_test_edges` | `hymeko_neuro/experiments/demo/inference.py` | load checkpoint → run forward → AUC/F1, ROC, αₖ vector |
| `load_checkpoint`, `CheckpointMeta`, `InferenceBundle` | `hymeko_neuro/experiments/demo/checkpoint.py` | checkpoint (de)serialisation |
| model registry | `hymeko_neuro/experiments/demo/registry.py`, `models.yaml`, `kinematic_models.yaml` | name → constructor + checkpoint path |
| plotting helpers | `hymeko_neuro/experiments/demo/plotting.py`, `kinematic_plotting.py` | ROC / αₖ / pose figures |
| checkpoints | `checkpoints/hsikan/bitcoin_{alpha,otc}_optuna_best.pt`, `checkpoints/kinematic/family_classifier_k{4,6}.pt`, `checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt` | trained weights |

**Gap to close once, used by all demos:** a single thin CLI entry
`python -m hymeko_neuro.experiments.demo` (mode argument: `link | latency | kinematic | yolo | signature`)
that dispatches to the demos below. One file with a mode arg — not five scripts
(CLAUDE.md §6.5 #13).

---

## Demo 1 — Signed-graph link-prediction inference  ★ priority
**Slide:** "Signed-graph link prediction results" (17)

- **What it shows:** load a trained checkpoint, predict held-out edge signs, print AUC/F1, draw the ROC, and surface the learned αₖ "regime" vector.
- **Already on disk:** `demo/inference.py::predict_test_edges`, both Bitcoin checkpoints, `demo/plotting.py`.
- **To finish:**
  - CLI wrapper: `python -m hymeko_neuro.experiments.demo link --dataset bitcoin_otc --checkpoint checkpoints/hsikan/bitcoin_otc_optuna_best.pt`.
  - Print a table: dataset · n_test · AUC · F1 · n_params · fwd_ms.
  - Save `ROC.png` + `alpha_k.png` to `demo_out/link/<dataset>/`.
- **Inputs:** `checkpoints/hsikan/bitcoin_{alpha,otc}_optuna_best.pt`; datasets via `hymeko_neuro/data/datasets.load`.
- **Acceptance:** reproduces the committed AUC for that checkpoint within ±0.002 over the fixed test split; figure files written; exits with peak-RSS + wall-time line.
- **Honest note:** these checkpoints are the `optuna_best` (transductive convention) — label the figure as such; the strict-protocol number is the architectural baseline.

---

## Demo 2 — Forward-latency benchmark  ★ priority  ·  **DONE (2026-06-11)**
**Slide:** "SOTA range — at a fraction of the cost" (18)

- **What it shows:** measured forward-pass latency — SGCN vs HSiKAN-lean (h4) vs HSiKAN-joint (h16) per (dataset, device).
- **On disk:** `hymeko_neuro/experiments/runs/run_inference_bench.py` (extended), `bench_to_png.py` (renderer), `experiments/results/inference_bench.json` + `inference_bench_{cpu,cuda}.png`. See `reports/2026-06-11-latency-bench-extension.md`.
- **Done:**
  - Two HSiKAN width cells per dataset, **lean (h=4)** and **joint (h=16)**, width as a config axis; cycle pool built once and shared.
  - Re-emitted `inference_bench.json`; `bench_to_png.py` renders the slide-18 horizontal bars.
- **Acceptance:** ≥ 5 repeats after warmup; median, IQR, worst case (CLAUDE.md §3); both devices; numbers in the json. ✓ Peak RSS 1.49 GB, wall 103 s.
- **Measured gap (corrects the earlier "11×"):** the within-family h4-vs-h16 width gap is **~3.5× on CPU and ≈1× on CUDA** (same device, both Bitcoin graphs). The deck's 30.5-vs-342 "11×" was the *optuna_best_otc-vs-joint* result — real but **OTC-specific and tuple-set-driven, not a general width claim** (Alpha's optuna config runs ~2× *slower*).
- **Honest note (surfaced in output):** on absolute forward latency, **SGCN and MLP-blind are lightest**; SiGAT and SGT are **heavier than HSiKAN-lean**. The defensible claims are (a) **accuracy-per-parameter** and (b) the measured **within-family width gap** above — not "faster than SGCN."
- **SOTA accuracy-vs-cost (2026-06-11):** `bench_to_png.py --all` also emits an **accuracy-vs-params Pareto** (`inference_bench_pareto_<dataset>.png`) and a merged **numbers table** (`inference_bench_table_<dataset>.md`) covering SGCN / SiGAT / SGT / MLP-blind / GCN-blind / HSiKAN. **Honesty caveat (must say aloud):** optuna-HSiKAN dominates accuracy-per-parameter (OTC 0.9933 @ 23.8k vs SGCN 0.942 @ 203k) **but it is tuned and the baselines are not** — the untuned HSiKAN in the same panel (leanest, 0.851) sits *below* the baselines, and is plotted as a second point so the claim is not oversold. Cost axis = params (committed); latency lives in the bars. See `reports/2026-06-11-slide18-sota-comparison.md`.

---

## Demo 3 — Kinematic regime compass + pose inference
**Slide:** "From structure to inductive prior" (12) / "The bridge" (19)

- **What it shows:** (a) cycle-arity compass (four-bar→k4, delta/Stewart→k6, serial→flat); (b) family classification from topology; (c) HSiKAN graph-only XYZ vs MuJoCo (~5 cm L2).
- **Already on disk:** `hymeko_neuro/demos/demo_kinematic_mujoco.py`, `demo_kinematic_pose.py`, `examples/pose_demo.py`, `demo/kinematic_classifier.py`, `checkpoints/kinematic/family_classifier_k{4,6}.pt`, web demo `demo_web/` (compass + 3D), and the existing render `demo_video/hsikan_mujoco_4dof.mp4`.
- **To finish:**
  - Wire `family_classifier_k{4,6}.pt` into a `kinematic` mode that loads the checkpoint (skip training) and prints predicted family + αₖ compass per mechanism.
  - Regenerate `demo_out/sim.mp4` headless (EGL) and confirm the overlay L2 stays ≈ 5 cm.
  - Refresh `demo_web/kinematic_data.json` from the canonical URDF fixtures so the browser compass matches.
- **Acceptance:** classifier loads from checkpoint without retraining; per-frame L2 reported; mp4 + compass json regenerated deterministically (seeded).
- **Display:** `demo_web/index.html` (no server needed) for the talk; mp4 already embedded in the deck.

---

## Demo 4 — HyMeYOLO detection inference
**Slide:** "The primitive transfers: HyMeYOLO" (18→ vision slide)

- **What it shows:** the `+ricci-mod` + HSiKAN-CR detector finding digits in Cluttered MNIST; boxes + per-image mAP + forward latency.
- **Already on disk:** `hymeko_neuro/experiments/vision/demo_hymeyolo_tk.py`, `vision/DEMO_README.md`, `vision/launch_demo.sh`, `checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt`.
- **To finish:**
  - Ensure `--checkpoint checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt` **skips training** (README says no-checkpoint path quick-trains 30 epochs — for the demo always pass the checkpoint).
  - Add a headless mode that renders N example panels (GT cyan / pred red) to `demo_out/yolo/` plus a mAP_50 + latency line, for slide capture.
- **Acceptance:** loads checkpoint, no training on launch; mAP_50 in the 0.90 band on the demo split; honest-metric (consumed-GT COCO matching, not the pre-2026-05-16 bug).
- **Honest note:** quote the corrected mAP (0.903 ± 0.009), never the bug-inflated 0.723.

---

## Demo 5 — Gömb structural signature + label-shuffle (honesty)
**Slide:** "Honesty as a protocol" (17→ honesty slide) / "Gömb" (16)

- **What it shows:** the strict cycle-pool signature for a positive / negative / boundary query, and the **label-shuffle collapse to ≈0.540** that proves the prior is structural, not leakage.
- **Already on disk:** `hymeko_neuro/experiments/runs/demo_gomb_signature.py`, `demo_fuzzy_signature.py`; shuffle flags `run_final_cell.py --shuffle-train-signs`, `run_gomb_smoke.py --shuffle-train-signs`.
- **To finish:** a `signature` mode that runs one real-label inference and one shuffled-label inference back to back and prints both AUCs side by side (real ≈ 0.94 vs shuffled ≈ 0.54).
- **Acceptance:** shuffled AUC within ±0.02 of chance; figures to `reports/figures/gomb_signature_*/`.

---

## Demo 6 — Query-driven transform (framework, optional)
**Slide:** "Query-driven transforms" (9)

- **What it shows:** one `.hymeko` graph → URDF/SDF/MJCF live, no recompile.
- **Already on disk:** `transforms/{urdf,sdf,mjcf,dot,ros2_launch}/`, `hymeko_cli`, `scripts/demo_*.sh`.
- **To finish:** a scripted `hymeko transform data/robotics/robot_4wh.hymeko -t urdf` capture + a diff showing the same IR → three targets.
- **Acceptance:** emitted URDF parses (e.g. `check_urdf`); identical link/joint counts across targets.

---

## Build order (inference-first)
1. **Demo 1** (link-prediction) — smallest, validates the shared harness.
2. **Demo 2** (latency) — needed for the slide-18 numbers; reuses Demo 1's load path.
3. **Demo 3** (kinematic) — highest visual payoff for the talk.
4. **Demo 5** (signature/shuffle) — short, reinforces the honesty slide.
5. **Demo 4** (HyMeYOLO) — heaviest; checkpoint wiring is the only blocker.
6. **Demo 6** (transform) — optional framework-side closer.

## Cross-cutting acceptance (every demo)
- Single CLI: `python -m hymeko_neuro.experiments.demo <mode> [...]`, `--device auto`, `--seed 0`.
- Deterministic: seed fixed, no system entropy; outputs reproducible.
- On exit: print peak RSS + wall time; assert RSS < 16 GB.
- Each demo writes to `demo_out/<mode>/` and prints the artifact paths it produced.
- A number is only printed if it can name the checkpoint/file it came from.
