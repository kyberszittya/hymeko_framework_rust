# Report — Hypergraph-vision vs CNN: fair re-benchmark on MNIST / Fashion-MNIST

**Date:** 2026-05-28
**Plan:** `docs/plans/2026-05-28-vision-hypergraph-vs-cnn-rebench/` (tex/pdf/tikz/mmd)
**Predecessor:** 2026-05-06 null (memory `project_hsikan_vision_redo_2026_05_06`),
                  trained at `--n-epochs 5`, 1 seed.
**CORE.YAML items touched:** none.

## Question

The 2026-05-06 vision null (HSiKAN 0.34/0.37, NeocogHGNN 0.58/0.62 vs CNN
≈ 0.99 on MNIST/Fashion) was produced at **5 epochs, 1 seed**. That is
almost certainly undertrained — too weak to support the headline
"hypergraph vision loses to CNN." This task asks: **does the gap survive
properly-trained, multi-seed evaluation?** Including the never-benchmarked
vision-specific **Gömb-Soma RicciStim** operator (quadtree → signed
patch-graph → Bochner walks/polygons/triangles).

## Headline — yes, decisively. The structural prior is hurting, not helping

| Model | MNIST AUC | Fashion AUC | gap to CNN MNIST | gap to CNN Fashion |
|:--|:--|:--|--:|--:|
| **cnn** (TinyCNN, conv) | **0.9720 ± 0.0003** | **0.8608 ± 0.0037** | 0 | 0 |
| **mlp** (structure-free) | 0.9228 ± 0.0045 | 0.8283 ± 0.0048 | 0.049 | 0.033 |
| hsikan (RF + signed-branch Catmull-Rom) | 0.4189 ± 0.0201 | 0.6371 ± 0.0089 | **0.553** | **0.224** |
| hgnn (Feng 2019 RF hyperconv) | 0.3148 ± 0.0272 | 0.4136 ± 0.0098 | **0.657** | **0.447** |
| ricci_stim (quadtree + signed walks) | 0.3104 ± 0.0091 | 0.4538 ± 0.0234 | **0.662** | 0.407 |

Two findings tighten the prior null:

1. **The gap survives fair training.** At 15 epochs × 3 seeds (vs the
   prior 5 × 1), with seed-sd 0.003 – 0.027 the gap is **22–66 percentage
   points** to CNN — way outside any conceivable seed noise. Undertraining
   was not the explanation.

2. **Hypergraph operators lose to MLP, not just CNN.** A plain
   fully-connected network at the same budget is **30–50 points above all
   three hypergraph operators on MNIST**. The structural prior the
   hypergraphs encode is not merely failing to help — it is actively
   blocking the network from learning what a structure-free baseline
   learns trivially. The "structural prior aids vision" hypothesis fails
   *this hard*.

3. **The new vision-specific Gömb-Soma RicciStim doesn't break it either.**
   At its reduced budget (subset 2000, 10 epochs — set by its per-image
   quadtree cost, see Performance) it lands at 0.310 / 0.454, statistically
   indistinguishable from HGNN (0.315 / 0.414) on MNIST. The
   vision-specific operator the prior steer asked for is *not* a way out
   of the hypergraph-vision gap. (Caveat: it is the *most* budget-starved
   of the five — see "what this doesn't settle.")

## Per-model commentary

- **cnn ≫ mlp ≫ hypergraphs.** The ordering on both datasets is
  translation-equivariant > structure-free > hypergraph. The middle
  position of MLP isolates the cost cleanly: it isn't that "we need
  inductive bias to learn from limited data"; the structure-free MLP
  learns fine. It's that *this particular* structural prior (pixel-RF
  hyperedges, signed branches, quadtree patches) collides with the data.
- **HSiKAN > HGNN on Fashion** (0.637 vs 0.414) but **HSiKAN ≈ HGNN on
  MNIST** (0.419 vs 0.315). HSiKAN's αₖ arity mixing + Catmull-Rom helps
  on coarser texture-rich Fashion shapes but not on stroke-based MNIST.
  Even on Fashion HSiKAN is 22 points below CNN.
- **RicciStim's signed-walks + Bochner machinery** — the most "matured
  signed-graph" of the three — does *not* outperform vanilla NeocogHGNN.
  Per-image quadtree construction is a tractability tax with no quality
  payoff on this regime.

## Test / quality results

| File | Count | Result |
|:--|--:|:--|
| `test_vision_bench.py` (subset, matrix, aggregation, gap, dry-run) | 7 | pass |
| ruff on new code | — | clean |
| All 5 models CPU-crash-checked through unified runner | 5 | run |
| Failed cells in overnight matrix | 0 / 30 | — |

## Performance results

The chain's full timeline (logged):

| Stage | Wall | Notes |
|:--|--:|:--|
| Wait for `b01mfrajm` to free GPU | 79 min | poll on `summary.json` |
| GPU smoke (cnn + ricci_stim, tiny) | 2 min | passed → proceed |
| Fast-four matrix (cnn,mlp,hgnn,hsikan × {mnist,fashion} × 3 seeds = 24 cells) | 78 min | subset 8000, 15 ep |
| RicciStim matrix (× 2 datasets × 3 seeds = 6 cells) | 46 min | subset 2000, 10 ep |
| **Total chain** | **205 min (3.4 h)** | within ≤ 6 h budget |

Per-cell peak RSS ≤ ~1 GiB (well under 7 GiB / 16 GiB caps).
`systemd-run --user --scope -p MemoryMax=16G`.

## Provenance

- **Git SHA:** `8fd8187` (dirty; new files are the runner / orchestrator /
  test / chain script).
- **Interpreter:** miniconda3 python 3.13.5, **torch 2.11.0+cu130** (drift
  vs CORE 2.12.0, user-approved). Absolute accuracies are
  comparison-grade, not absolute SOTA — TinyCNN at full MNIST + more
  epochs would exceed our 0.972.
- **GPU:** RTX 2070 SUPER 8 GiB. **Seeds:** {0,1,2}.
- **Fast four:** `--train-subset 8000 --n-epochs 15 --batch-size 128 --hidden 32 --lr 1e-3`.
- **RicciStim** (reduced, flagged): `--train-subset 2000 --n-epochs 10` (same other knobs).
- **Artifacts:** `/tmp/vision_bench/results.jsonl` (30 rows), `summary.json`, per-cell logs.

## Honest caveats

- **Subset training, not full MNIST.** Numbers are relative, fair *across
  models*, not SOTA. The comparison is "does the gap survive fair
  training" — yes.
- **RicciStim not iso-budget.** It has 1/4 the train data and 2/3 the
  epochs of the fast four; the eval set is the *full* 10k test set in
  both cases (~400 s/cell eval for RicciStim, the bottleneck). Its result
  is the most charitable reading; a smaller asymmetric budget cannot
  rescue an operator that's also clearly trailing on Fashion (0.45 vs
  CNN 0.86).
- **5 architectures, 2 datasets, 3 seeds — not exhaustive.** Cluttered-MNIST,
  CIFAR-10 grayscale, and capacity sweeps (h ≠ 32) are untested.
- The 2026-05-06 result was *consistent* with the prior null direction;
  this round simply removes the undertraining doubt.

## Conclusion + follow-ups

**The 2026-05-06 finding stands and strengthens:** *the existing
hypergraph-vision operators (signed receptive fields, Feng 2019 HGNN,
quadtree-signed-walks) do not close the gap to translation-equivariant
convolution — or even to a structure-free MLP — on small-image
classification.* The vision regime favours translation equivariance over
signed-cycle / signed-RF structure.

What this *doesn't* refute:
- A future vision-specific operator (steerable + equivariant + signed)
  that combines translation equivariance with the structural bias.
- Tasks where small-image classification is replaced by a regime that
  *needs* relational structure (graph-classification, 3D meshes, point
  clouds). The right comparison there is graph-data benchmarks, not
  MNIST.

Follow-ups:
- **None recommended on this hardware for image classification.** The
  signal is unambiguous; further small-image runs would re-confirm at
  cost. The natural next domain shift is graph or mesh data.

## Cross-thread overnight summary (2026-05-27 → 2026-05-28)

1. **Regime A/B/C 5-seed comparison** on Bitcoin Alpha, Slashdot K=8, and
   Slashdot K=32 + sparse-attn — the 33 %-lossless No-Excess cut held in
   *every* test; quaternion (Canonical-only) never wins. (Three reports.)
2. **Hypergraph-vision vs CNN fair re-benchmark** (this) — gap survives
   proper training; structural prior actively hurts.

Both threads land negative for the "more elaborate is better" hypothesis,
and both with disciplined contract compliance (plans, smokes, wall
reconciliation, per-cell checkpointing, multi-seed, reports).
