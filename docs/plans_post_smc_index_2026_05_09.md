# Post-SMC research direction index — 2026-05-09

The SMC HSiKAN paper was submitted previously.  This index lists every research direction proposed during the 2026-05-08/09 sessions, scoped *after* SMC.  Each direction lives in its own plan doc; this file is just the table of contents + venue-fit + leverage-on-existing-work cross-reference.

| direction | plan doc | venue | code distance | risk |
|---|---|---|---|---|
| **HSiKAN on tabular benchmarks** (Iris, housing, MNIST clustering) | `plans_hsikan_tabular_benchmarks_2026_05_09.md` | workshop / journal §V | ~300 LOC | low (architecture stretch) |
| **Mesh matching via signed-cycle attention** | `plans_mesh_matching_2026_05_09.md` | SIGGRAPH / CVPR | ~600 LOC | medium (real benchmark) |
| **Structural Kolmogorov–Arnold theorem** | `plans_structural_ka_theorem_2026_05_09.md` | JMLR / NeurIPS theory | 0 (pure theory) | high (math risk) |
| **Predictive coding on signed graphs** (entropy replaces backprop) | `plans_predictive_coding_signedgraph_2026_05_09.md` | NeurIPS / ICML | ~500 LOC | medium-high |
| **General-graph extensions** (node-class, contrastive, masked-cycle, bipartite) | `plans_general_graph_extensions_2026_05_09.md` | workshop / journal §V | ~700 LOC over 4 sub-experiments | low |
| **Time-series via sequence-induced hypergraphs + frequency attention** | `plans_hsikan_time_series_2026_05_09.md` | NeurIPS / ICML | ~600 LOC | medium (FEDformer / PatchTST baselines well-tuned) |
| **Fractal maps for cycle generation + evaluation** | `plans_fractal_maps_2026_05_09.md` | NeurIPS workshop / NeurIPS | ~750 LOC | medium (IFS-on-cycles design risk) |
| **kCVD vs YOLO** — k-cycle vision detection on PASCAL VOC / COCO | `plans_kcvd_vs_yolo_2026_05_09.md` | CVPR / ICCV | ~1000 LOC + benchmark wiring | high (real benchmark, mature competition) |

## What each direction needs from the existing codebase

| direction | reuses | requires new |
|---|---|---|
| Tabular | full HSiKAN encoder, run_final_cell pattern | `tabular_signed_graph.py`, `cell_tabular` |
| Mesh matching | full encoder, sparse attention infrastructure | mesh loader, face-sign assignment, Sinkhorn loss, FAUST harness |
| Structural-KA theorem | nothing (theory only) | nothing |
| Predictive coding | full architecture, $M_{vt}^\top$ generative path | PC inner-loop, local update rules, eval harness |
| General-graph extensions | full encoder | per-task heads + 4 task runners |

## Cross-direction synergies

- **Predictive coding's** Tier-1 gate is best tested on the **tabular** Iris dataset (~100s of samples, fast convergence).  Run them together.
- **Mesh matching** uses the **bipartite matching** machinery from the **general-graph extensions** plan as scaffolding.
- **Structural-KA theorem** uses the empirical compounding ladder from *every* other plan as constructive anchor.  Pure-theory paper but needs the empirical anchor citations.
- **Tabular benchmarks** node-classification and **general-graph extensions** node-classification overlap; tabular extends with sklearn datasets, general-graph uses graph datasets.  Same `cell_node_classification` head.

## Suggested execution order

If running one at a time:
1. **Tabular benchmarks** (1 week) — fastest, lowest-risk, immediate paper-bullet for journal §V.
2. **General-graph extensions** (1-2 weeks) — compounds with (1) for a "HSiKAN universality" story.
3. **Mesh matching** (4-6 weeks) — biggest applied paper; standalone venue.
4. **Predictive coding** (4-6 weeks if Tier 1 lands) — methods/theory paper.
5. **Structural-KA theorem** (4-6 months) — pure theory; can run in parallel, independent of code.

If running in parallel (one per researcher):
- (1) + (2) on the same code track.
- (3) on its own track.
- (4) gated on (1)'s tabular sanity-check.
- (5) ongoing throughout.

## What's NOT in this index

- The **SMC paper** revisions / proceedings work.  That's already submitted and frozen.
- The **GrafGeo 2026** paper (per memory `project_grafgeo_submitted`) — a different track.
- The **Phase A entropy aux losses** (`plans_entropy_learning_2026_05_08.md`) — that's already partially executed (A1 validated tonight at +5σ); no new plan needed.

## Validated empirical anchors at the time of writing (2026-05-09)

These are the locked-in 5-seed numbers that any post-SMC paper can cite without further validation:

| dataset | HSiKAN result | recipe |
|---|---|---|
| Bitcoin Alpha | 0.9845 ± .0028 | joint c3,c4,w2,w3, h=16 |
| Bitcoin OTC | 0.9801 ± .0057 | same |
| Slashdot | **0.9050 ± .0050** with α-entropy aux at λ=0.01 | c2,c3,c4,c5,w2,w3 + Highway-quat at h=4 |
| SBM 200/400 | 0.911/0.962 (paper baseline) | cycle-only |
| Epinions | best single-seed 0.8409 (bigger_caps) | overnight grid in flight; 5-seed validation pending |

Slashdot's result at 0.9050 ± .005 places it **+1.5σ above SGT 0.897 ± .002** at **1/8 the parameter count** on a **consumer 8 GiB GPU**.  Phase A1 α-entropy aux loss is 5σ-paired-significant on top of the Highway-quat baseline.

## Acceptance for the index itself

This file is a navigational artifact.  The "acceptance" is just that the plans linked here are concrete enough to execute when the user picks one up.  No experiments are run from this doc.
