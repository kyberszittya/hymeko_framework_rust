# HyMeKo — Done (completed work)

**Last updated:** 2026-06-15 · Companion: [BACKLOG.md](BACKLOG.md) (open work).

Curated milestone log. The **full archive is `reports/`** (~250 reports — every
completed task has one); this file is the human-scannable highlight reel, newest
first. When a backlog item lands, move its line here with the report link.

---

## 2026-06-15 — Editor + hero-demo sprint

- ✅ **Editor: hypergraph examples gallery** (Fano / sunflower / K₄³ / generic) — `reports/2026-06-15-editor-hypergraph-examples.md`
- ✅ **Editor: parametric generators** (Steiner S(2,3,n), sunflower, complete Kₙ⁽ʳ⁾) + progress bar — `reports/2026-06-15-editor-hypergraph-generators.md`
- ✅ **Editor: multi-file imports + vocabulary profiles** (WASM `parse_and_compile_files`, rebuilt pkg) — `reports/2026-06-15-editor-profiles-imports.md`
- ✅ **Editor: arc-ref value editing** (sign/target/value, source-as-truth) — `reports/2026-06-15-editor-arc-values.md`
- ✅ **Editor: 2D composite (roots-centred) layout + 3D pan** (hyper3d) — `reports/2026-06-15-editor-layout-pan.md`
- ✅ **Editor: robot-arm hero-cell profile** (imported kinematics) — `reports/2026-06-15-editor-robot-arm-profile.md`
- ✅ **Editor: 2D bidirectional relations** (reciprocal-pair detection + double connector) — `reports/2026-06-15-editor-bidirectional-2d.md`
- ✅ **HRI profile fix** — relations as signed hyperedges — `reports/2026-06-15-editor-hri-hyperedges.md`
- ✅ **Hero demo Phase 1** (robotics spine: one model → URDF/SDF/MJCF/DOT/Mermaid, gated; broken-twin rejected) — `reports/2026-06-15-hero-demo-phase1.md`
- ✅ **Hero demo Phase 2** (hybrid: robots + learners via `torch_dataflow`) — `reports/2026-06-15-hero-demo-phase2.md`
- ✅ **Hero demo gate hardening** (exit-code-authoritative; corrected a wrong "exit 0" finding) — `reports/2026-06-15-hero-demo-gate-exitcode.md`
- ✅ **Hero demo Phase 3** (Gömb structural parity 6/6 + minimal Soma 3/3, torch-free) — `reports/2026-06-15-hero-demo-phase3.md`
- ✅ **Seminar deck additions** (star/clique tensor-view heatmaps + Reference & Future-work slides; python-pptx approved) — `reports/2026-06-15-seminar-deck-additions.md`
- ✅ **SysML requirements-trace lens** (SMC #5) + **editor SysML lens** — `reports/2026-06-14-sysml-requirements-trace.md`, `reports/2026-06-14-editor-sysml-lens.md`

## Soma-vision line (built; vision hypothesis falsified — see BACKLOG decision)

- ✅ **Base-Soma vs Linear control (MNIST, 2026-06-15)**: walk-conv base-Soma **0.52** vs linear **0.91** (−0.387 paired, all 5 seeds) — walks-only falsified for vision too (2.2× more param-efficient, but far lower absolute). Completes the Soma-vision vision-falsification. `reports/2026-06-15-base-soma-vs-linear-mnist.md`
- ✅ **Cichy-92 cortical-prediction article** (why brain-predictivity, not accuracy, is the honest test) — `docs/articles/cichy-cortical-prediction/article.pdf`
- ✅ **Seminar: Publications & submissions slide + portfolio figure** (8 venue families, status-coloured) — `docs/seminar/{make_publications_figure.py, insert_into_deck.py}` → `HyMeKo_Seminar.with_refs.pptx` (37 slides)

- ✅ **RicciStim stack**: Forman κ, Hodge Laplacians, adaptive quadtree, Bochner-coupled hg-conv, stim-graph builder, SDRF — all implemented + 168+ tests — plans `2026-05-14-gomb-soma-ricci-stim(+bench)`
- ✅ **Perf passes 1–4**: 296× speedup (8.3 s → 28 ms/image) — `reports/2026-05-15-gomb-soma-ricci-stim-sdrf-optimization.md`
- ✅ **Hodge boundary₂ vectorize + dead-code cleanup** — `reports/2026-05-16-gomb-soma-hodge-vectorize.md`
- ✅ **Rust quadtree port** (PyO3, 3.9–9.8×) — `reports/2026-05-16-gomb-soma-quadtree-rust.md`
- ✅ **Cortical infra (Slice 1)**: scorer + ResNet-tiny baseline + synthetic Cichy smoke (21 tests) — `reports/2026-05-19-gomb-soma-cortical-implementation.md`
- ✅ **Fair vision re-bench** (CNN/MLP/HSiKAN/HGNN/RicciStim × MNIST/Fashion): **negative result, well-engineered** — `reports/2026-05-28-vision-hypergraph-vs-cnn-rebench.md`

## HyMeYOLO line (Cluttered-MNIST stages landed; VOC transfer open)

- ✅ Stage A-2 (cosine+e100, +0.118), Stage B (ResNet-tiny, +0.149), Stage C (FPN), warm-start (+0.124), ricci-weight sweep — `reports/2026-05-16/17-hymeyolo-*-5seed.md`
- ✅ Stage D / D-1 **falsified with diagnostic** (from-scratch & ImageNet both ≈0.01 mAP on VOC → head is the bottleneck); Stage D-2 head-bottleneck diagnosis complete
- ✅ VOC test baseline (ep60 floor 0.0149) + `eval_voc` tool + headless capture for slides — `reports/2026-06-10-voc-test-baseline.md`, `…-hymeyolo-headless-capture.md`

## Gömb signed-link prediction (results on disk)

- ✅ Gömb-strict link prediction across Bitcoin/OTC/Slashdot/Epinions/wiki_elec (~0.90–0.94 AUC, config-dependent; leakage control → 0.540 = chance); ~30k params — artifacts under `signedkan_wip/experiments/results/*.jsonl`, `reports/gomb_tune_*`, `reports/hsikan_*5seed*`

## Framework / systems (selected)

- ✅ P-graph engine + A1–A5/MSG/SSG/ABB + Pimentel book validation; reachability-rules unification — plans `2026-05-19-pgraph-*`, `2026-06-14-reachability-rules-audit-pgraph`
- ✅ WASM editor MVP + stereotype views; canonical Blake3-hash IR; query-driven transforms (urdf/sdf/mjcf/gazebo/ros2/dot/mermaid/sysml/torch_dataflow)
- ✅ SISY 2026 control paper (review-fixed, ROS pick-and-place demo) — `[[project-sisy2026-control-paper]]`

> For anything not listed here, search `reports/<date>-<slug>.md`.
