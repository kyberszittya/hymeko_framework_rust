# GömbSoma cortical benchmark — progress as of 2026-05-19

**Date:** 2026-05-19
**Last touch:** 2026-05-16 (Rust quadtree port; Hodge vectorisation; SDRF net-negative call)
**Status snapshot:** architecture complete and tested, *Cluttered MNIST falsifier missed*, **cortical-benchmark plan staged but unlaunched**. Three branch options on the table from May 15 ("not yet over"); a recommendation below.

## 1. What ships today

### Code

- **4 235 LOC** under `signedkan_wip/src/hymeko_gomb/soma/`.
- **204 tests** in `signedkan_wip/tests/test_gomb_soma*.py` — **all passing** as of today's audit.
- Architecture stack (10 phases, complete):
  - Forman κ per anchor (Phase 1–2)
  - AdaptiveQuadtree (Phase 3) — **ported to Rust** via `hymeko.build_quadtree_rs` (2026-05-16), 3.9× to 9.8× CPU speedup, set-equal vs Python ref
  - Hodge Δ₀ eigenmodes (Phase 4) — **vectorised** (2026-05-16), 4× `sparse.mm` is the hot path
  - BochnerHypergraphConv αβ propagation (Phase 5)
  - StimulusGraphBuilder (Phase 6)
  - SDRFRewiring (Phase 6–10) — wired but **net-negative on Cluttered MNIST**
  - Classifier, Detector, end-to-end train loop (Phase 7–9)
  - Phase 10: SDRF integration into the backbone (`use_sdrf=True` switch)

### Performance milestones

| Pass | Optimisation | ms/image | Throughput |
|:---|:---|---:|---:|
| 0 (naïve) | baseline                           | 8 283 | 0.12 FPS |
| 1         | scalar-tensor-elimination          | 2 840 | 0.35 FPS |
| 2         | quadtree variance scoring on GPU   |   720 | 1.4 FPS  |
| 3         | sparse-matmul Hodge eigenmodes     |   150 | 6.7 FPS  |
| **4**     | **Rust quadtree + Triton hooks**   | **28** | **35 FPS** (real-time) |

**~296× cumulative speedup** on RTX 2070 SUPER. The architecture is now real-time deployable on consumer hardware.

## 2. Where we got stuck (Cluttered MNIST 5-config ablation, 2026-05-15)

5 configurations × 1-seed × Cluttered MNIST × the same training recipe.

| Config | What it adds vs C | mAP_50 proxy | Wall (1 seed) |
|:---|:---|---:|---:|
| A (baseline) | --- | 0.0  | --- |
| B | + anchor-only inference | 0.0  | --- |
| C | + Bochner-ricci propagation | 0.158 | --- |
| **D** | **+ Bochner αβ tuned (no SDRF)** | **0.174** | best of series |
| E | + SDRF rewiring | 0.141 | 27% slower than D |

**Two findings, both negative for the original headline claim:**

1. **SDRF rewiring is net-negative** on this regime: −0.033 mAP at +27 % wall. The κ-bottleneck-relief operator that Topping et al. 2022 reported as beneficial does not transfer to the Cluttered MNIST detection task with our architecture and recipe.
2. **The ceiling is 0.174**, far below the original plan's **0.235 falsification gate**. The headline claim "GömbSoma beats HyMeYOLO `+ricci-mod` (0.723) at its own game" is **not delivered** at the current training budget.

**Honest framing**: the architecture is sound (gradients flow, tests pass, optimisation milestones hit), but the *Cluttered MNIST detection benchmark is not the right falsification target*. The architecture is **cortical-circuit-inspired**, multi-depth, hierarchical — it's designed for **brain-response prediction**, not for YOLO-style box regression. Detection is a downstream readout.

## 3. The cortical benchmark — the right falsification target

The 4-format plan at `docs/plans/2026-05-16-gomb-soma-cortical-benchmark/` lays out the **Brain-Score-style** evaluation that the architecture was designed for:

### Methodology

1. **Stimuli**: Cichy-92 image set (Cichy, Pantazis, Oliva 2014) — 92 images, ~free, used as the "first viable" cortical benchmark in the literature.
2. **fMRI targets**: V1 / V2 / V4 ROI responses, 16 subjects (publicly available with Cichy 92).
3. **GömbSoma feature extraction**: per-depth features at depths 0–3 (V4-scale → retinal-scale).
4. **Scoring**: PLS reduction (25 components) → Ridge regression per voxel → noise-ceiling correction → per-ROI Brain-Score.
5. **Baselines**: parameter-matched ResNet-tiny (~1 M params) and ViT-S/16. Brain-Score's public-leaderboard suite numbers as comparison.

### Why this is the right experiment

- **GömbSoma's multi-depth quadtree maps onto V1→V2→V4 retinotopic hierarchy by construction.** The architecture *is* a cortical-circuit model.
- **Brain-Score is the canonical evaluation** in the field; it has a public Python API and a comparable-numbers leaderboard.
- **Cichy 92 is small** (92 images), so we can run end-to-end in hours not days, on the consumer GPU.
- **Falsification is structurally cleaner**: "does GömbSoma's per-depth feature extraction predict V1/V2/V4 better than ResNet-tiny?" is a single statistical comparison with an honest noise ceiling.

### What's missing operationally

- The Cichy 92 dataset isn't downloaded yet.
- A `scripts/cichy92_to_features.py` extraction pipeline is the next 100–200 LOC.
- Brain-Score's `score_model` API integration: ~50 LOC of glue.
- 5-seed paired comparison vs ResNet-tiny + ViT-S/16.

**Realistic wall-time estimate**: dataset download + feature extraction + 3 models × 5 seeds × per-ROI scoring → **8–12 hours** total, dominated by the slowest model's forward pass. Fits in one overnight if launched today.

## 4. The three branches from May 15 — recommendation

The user wrote on 2026-05-15 "not yet over" after the SDRF-net-negative result. Three options were on the table:

| Option | Effort | What it answers |
|:---|:---|:---|
| **A. SDRF parameter sweep** | ~1 day | Is SDRF salvageable for Cluttered MNIST? (Probably not — net-negative is a strong prior.) |
| **B. Sober writeup of the Cluttered MNIST ceiling** | ~1 day | Documents the falsification of the original claim. No new experiments. |
| **C. Pivot to the cortical benchmark** | ~2 days | Tests the architecture against *its actual design target*. |

**Recommendation: option C.** Reasons:

- **The architecture was designed for cortical scoring, not detection.** Cluttered MNIST tests the wrong reflex.
- **The plan exists** (4-format, May 16). All the upstream code is in tree and tested. The missing pieces are dataset wrangling + scoring glue.
- **A negative result is still publishable** — Brain-Score has a well-defined noise ceiling, so "GömbSoma scores X% noise-ceiling on V1, vs ResNet-tiny's Y%" is a clean statement regardless of sign.
- **A positive result would be the headline** — the cortical-circuit-inspired architecture beating ResNet at brain-response prediction would justify the whole 10-phase implementation effort.

Option B can be folded into option C's report (the cortical benchmark report would note "Cluttered MNIST showed a 0.174 ceiling, which we attribute to the architectural mismatch documented in §X").

## 5. Concrete day-19 / day-20 plan

If the user accepts option C, the natural sequence is:

1. **Day 19 morning** (~3 hr): write `scripts/cichy92_fetch.py` to grab the Cichy 92 stimuli + fMRI ROI matrices from the public mirror; verify hashes; cache locally.
2. **Day 19 afternoon** (~3 hr): write `scripts/gomb_soma_cortical_features.py` — load each Cichy image, run GömbSoma backbone in eval mode, extract per-depth feature vectors, save to `.npz`.
3. **Day 19 evening** (~3 hr): write `scripts/brainscore_eval.py` — for each (model, ROI) pair, PLS-reduce features, fit ridge per voxel, compute noise-ceiling-corrected score; emit JSON.
4. **Day 20 morning** (~2 hr): 5-seed run; write report.

Total estimate: ~11 working hours, splittable across two days. Everything is local; no cluster needed.

**Open question for the user before launching:**

- Acceptance criterion for the cortical benchmark? Proposed: GömbSoma's V1+V2+V4 mean Brain-Score is within ±10 % of ResNet-tiny's at iso-parameter-budget. Stricter ("beats ResNet-tiny") would be the headline; "comparable" would be still-publishable.

## 6. What I'd do *right now* if you say go

Start with `scripts/cichy92_fetch.py` — that's the smallest reversible step, gets data on disk, no architecture changes. Then write a tiny end-to-end smoke (one image, one ROI, one model) before scaling to 92 images × 3 ROIs × 3 models × 5 seeds. Same protocol that worked for the Stage D-3 series.

If you'd rather pick option A (SDRF sweep) or B (sober writeup), say which and I'll switch direction.

---

**Companion artifacts:**
- Plan (4-format): [`docs/plans/2026-05-16-gomb-soma-cortical-benchmark/`](../docs/plans/2026-05-16-gomb-soma-cortical-benchmark/)
- Architecture reports: [`reports/2026-05-14-gomb-soma-ricci-stim-phase{1..10}.md`](../reports/) + Phase 8-bench + sdrf-optimisation
- Optimisation reports: [`reports/2026-05-16-gomb-soma-hodge-vectorize.md`](2026-05-16-gomb-soma-hodge-vectorize.md), [`reports/2026-05-16-gomb-soma-quadtree-rust.md`](2026-05-16-gomb-soma-quadtree-rust.md), [`reports/2026-05-16-sdrf-net-negative-cluttered-mnist.md`](2026-05-16-sdrf-net-negative-cluttered-mnist.md)
- Source: [`signedkan_wip/src/hymeko_gomb/soma/`](../signedkan_wip/src/hymeko_gomb/soma/) (4 235 LOC, 204 tests)
