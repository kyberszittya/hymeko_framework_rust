# HyMeKo — Done (completed work)

**Last updated:** 2026-06-18 · Companion: [BACKLOG.md](BACKLOG.md) (open work).

Curated milestone log. The **full archive is `reports/`** (~250 reports — every
completed task has one); this file is the human-scannable highlight reel, newest
first. When a backlog item lands, move its line here with the report link.

---

## 2026-06-18 — Rotor signed-link: input enrichment + inductive transfer + the rotor ablation

- ✅ **Strengthened inductive transfer — the learned increment clears σ on harder pairs.** 3-arm decomposition (real / shuffle-gate / random-init floor) on train-small (bitcoin) → eval-large (slashdot/epinions), 5-seed. The learned-from-source increment (real − max(shuffle, random-init)) is **positive in 8/8 cells and ≥ 1σ_shuffle in 7/8**, reaching **+2.90σ** on the cleanest pair (`bitcoin_otc→epinions` +0.0261, σ_sh 0.009) — promoting the bitcoin-pair's borderline "~1–1.5σ, suggestive" to a real result. Key insight: **variance, not effect size, separates the train graphs** — otc-trained shuffle arms are tight (σ_sh≈0.009 → +2.3–2.9σ) while alpha-trained are noisy (σ_sh≈0.05 → ~1.1σ despite a *larger* raw increment). Random-init stays at/below chance everywhere, so learned = real − shuffle. Engineering: consolidated the previously-ad-hoc 3-arm decomp into the canonical driver (`Arm` enum + unified `run_grid`, §6.5 #3/#13) and **fixed a latent resumption-key collision** (real↔random-init shared a key → random-init silently dropped on resume), pinned by a regression test. Peak RSS 1.9 GB (heaviest cell). `reports/2026-06-18-transfer-grid-strengthen.md`
- ✅ **Is the Cayley-rotor geometry load-bearing? — NO (foundational ablation).** `MLPEmbedSignedModel` control: identical proj/SGCN/classifier, rotor embedding swapped for a higher-capacity MLP of the same output dim (16,661 ≥ 15,761 params — generous). 5-seed deduped gated: rotor − MLP = ±0.002–0.005 (all inside σ 0.01–0.027), both leakage-clean. So the rotor's S³ geometry adds nothing measurable on signed-link prediction (consistent with the head ablation: rotor *algebra* not load-bearing at the readout either). **The line's wins (leakage-free, inductive, param-light) come from replacing the node-ID table with structural features — NOT from the rotor; the MLP-embed control has all three too.** Reframes the contribution as structural-feature signed-link prediction. `reports/2026-06-18-rotor-vs-mlp-embed-ablation.md`
- ✅ **Inductive transfer test (nuanced positive — the line's distinctiveness validated).** Cross-graph transfer (train weights on graph A, evaluate frozen model on B's strict deduped split). **Mechanism + discrimination win:** the rotor transfers (0.81–0.86 AUROC on the *unseen* graph) while transductive `dadsgnn` **cannot** (`nn.Embedding(n_A)` can't index B). The naive shuffle-A gate is confounded (B's real signed adjacency carries the signal), so a **random-init control** was added (below chance, 0.38–0.47 — the true floor); the learned-from-A increment is **+0.038–0.063** (4/4 positive, ~1–1.5σ — suggestive, not strong). Bulk of transfer AUROC is the eval graph's own structural signal, exposed by training but not A-specific. Train loop refactored into a reusable, behaviour-preserving `_train`. `reports/2026-06-18-inductive-transfer-test.md`
- ✅ **Leakage-free input enrichment — `StructuralFeature` registry + exact k-walk signed `A^k` profile (new).** Extensible Strategy registry (degree/cycle single-sourced; new k-walk profile via sparse `A^k` mat-vecs, cap-free; new clustering ratios). 5-seed deduped gated: the walk profile lifts the **non-propagated** rotor baseline (audit `cyc+walk` otc **+0.0168**, gate-clean) but is **parity on the slerp-propagated line** (+0.003 otc) — slerp propagation already performs signed-`A^k` aggregation, so the feature is redundant there. A seed-0 otc 0.9131 (apparently beating pure SiGAT) **did not replicate** (5-seed 0.882) — recorded as variance, not a win. The residual gap to pure SiGAT (0.895) is **expressivity** (learned attention), not input features. `reports/2026-06-18-leakage-free-input-enrichment.md`

## 2026-06-17 — Rotor signed-link: the input fix (computer-graphics × AI)

- ✅ **Adaptive rotor propagation (neutral / knob-removal).** Phase 1 learnable per-block self-retention `α_b = retention_floor + exp(θ_b)` (log-space — softplus is for heads, not scale-free manifold mixing; ≡ a per-block sigmoid residual gate): 5-seed A/B **parity** with fixed sw=4 (alpha +0.0002 / otc −0.0010, gates clean) → removes the per-dataset `sw` knob at no AUROC cost. Phase 2 depth scan {2,3,4,6}: **no gain** (alpha over-smooths past r2; otc flat). Third independent confirmation the val ceiling is **input-bounded** (degree-only STRUCT_DIM=6), not retention/depth/readout. Next numbers lever = leakage-free input enrichment. `reports/2026-06-17-adaptive-rotor-propagation.md`
- ✅ **Protocol-matched honest SiGAT comparison — the "0.04 gap" was a dedup artifact.** `--dedup` true-held-out filter added to `run_baseline_audit` (single-sourced into the datasets layer); 5-seed grid, 4 models × 2 Bitcoin graphs × {deduped, non-deduped}, all shuffle-gated. On the matched **deduped** protocol the gap **inverts vs `sigat_rotor`** (0.833/0.868 < our 0.850/0.879) and the real residual is vs **pure SiGAT** only (+0.036 alpha / +0.016 otc). The non-deduped protocol leaks under shuffle (gate ⚠) → deduped is the honest protocol. Deduped rotor numbers reproduce the slerp report exactly. `reports/2026-06-17-protocol-matched-sigat.md`
- ✅ **Signed slerp/nlerp rotor propagation — first confirmed lift on the leakage-free rotor line.** Diagnosed the ~0.04 SiGAT gap as **input-bounded** (rotor input was a 6-dim degree-only feature; `val` pinned across 4 readout levers). Fix = propagate rotors over the signed graph on S³ (nlerp = "interpolate normals then renormalise"; edge-signed). r2 sw4: alpha 0.8455→**0.8500** (+0.0045) / otc 0.8685→**0.8790** (+0.0105), 5-seed, gates ≈0.5. `reports/2026-06-17-signed-rotor-slerp-propagation.md`
- ✅ **Link-head ablation (fair, on the same propagated rotors)** — real bilinear > complex > geodesic > quat (5-seed, gaps ~0.13, not a param artifact). Expressivity beats algebra at the readout; the head is **not** the gap. Side finding: real-on-q alone ≈ full triad-encoder pipeline. `reports/2026-06-17-link-head-ablation.md`
- ✅ **Diagnostic chain that localised the bottleneck** — geom-attn gate-collapse, woken score, rotor-relative projection, k=4 Berge cycles (all flat, all leakage-clean). `reports/2026-06-17-{geom-gate-inspection,geom-attention-wake-score,rotor-relative-projection,berge-kcycle-rotor,hsikan-rotor-sota-levers}.md`
- ✅ **Editor 3D hyperedge labels** — were never rendered (only vertex sprites); now sign-tinted at the member centroid, toggle with Labels, all modes. `docs/editor/views/hypergraph3d.js`
- ▶ **Next:** protocol-matched honest SiGAT comparison + adaptive propagation / propagation-as-encoder — plans saved, see BACKLOG.md.

## 2026-06-16 — Param-efficiency baseline + framework maturity

- ✅ **DETR baseline fairness guard fixed** — from-scratch MiniDETR could not overfit (box IoU plateaued at 0.46 < 0.5); fix = additive `l1giou` box loss + `lr 1e-3 / grad-clip 0.1` recipe → overfit guard `mAP→1.0`, PASSES (was `xfail`). Unblocks the RicciStim-vs-DETR head-to-head. `reports/2026-06-16-detr-overfit-fix.md`
- ✅ **Generators into `hymeko_core`** (single-sourced; 28 tests) — `reports/2026-06-16-generators-into-core.md`
- ✅ **RicciStim topology cache** (made the 40-ep full vision run feasible) — `reports/2026-06-16-stim-geometry-cache.md`
- ✅ **Full vision run (cached, upgraded RicciStim)** — Cluttered-MNIST config F, 5000 img / 40 ep, **mAP50_proxy 0.228** @ 5 896 params (single seed; beats prior hypergraph-vision number, not a parity claim) — perf log in `ROADMAP.md`
- ✅ **Technology-milestone snapshot** (proven / prototype / program maturity table) — `reports/2026-06-16-technology-milestone.md`

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
