# Gömb as a learned approximator for NP-hard signed-graph problems

**Date:** 2026-05-14
**Audience:** Niitsuma presentation + future paper.
**Status:** plan v1.

## The claim

> Gömb's cycle pool computes σ-products in poly time, which is a
> *learnable approximation* of objectives that are NP-hard to optimise
> exactly. Three concrete NP-hard targets are reachable through this
> framing: **maximum balanced clique**, **correlation clustering**,
> and **signed community detection / faction recovery**.

This is the research-shaped framing of the cliques demo. The
descriptive demo (generator + Bron-Kerbosch + balance check) is the
*input* surface; this plan is the *output* — what Gömb-with-a-cycle-
pool actually buys you that classical algorithms don't.

## What Gömb does and does NOT do

**Does:** For fixed *k*, enumerate all *k*-cycles in time
`O(|V|^k)` (in practice much faster with per-vertex top-K Rust
enumeration). Compute σ-products on those cycles. Aggregate into
vertex embeddings via `M_vt`. Edge-level prediction via the
classifier head.

**Does not:** Enumerate cliques. Solve max-clique. Solve max-balanced-
clique. These are NP-hard; no poly-time architecture can solve them
exactly unless P = NP.

**The bridge:** σ-products around triangles (k=3 cycle pool) are the
standard "balance feature" used by correlation-clustering
approximators (Bansal et al. 2004, Charikar et al. 2005). Gömb learns
**weighted combinations** of these features that are tuned to a
training distribution — i.e., a *learned approximation* of objectives
that have to otherwise be hand-designed.

## Prerequisites

**Must land first:**
`docs/plans/2026-05-14-cliques-generation-detection/plan.md` — adds
planted-clique generators + a four-detector benchmark harness
(Bron-Kerbosch / triangle-density / greedy-balanced / spectral). Until
that ships, "Gömb beats baseline X" claims have no baseline X.

## Compose-with

**Self-evolving cycle sampling:**
`docs/plans/2026-05-14-self-evolving-cycle-sampling/plan.md` — Gömb's
own gradients used as a learned importance-sampling distribution
over candidate cycles. Independent of this plan but the two compose:
the NP-hard Stage 1 / Stage 2 metrics may improve once the cycle
features Gömb consumes are themselves chosen by Gömb. Run them
independently first; compose only after both succeed in isolation.

## CORE.YAML items touched

**Empty list.** All work under `signedkan_wip/src/demo/cliques*.py`
and the existing `mixed_arity_signedkan` (no architectural changes
to Gömb itself — we're *applying* it, not modifying it).

## Three stages, three deliverables

### Stage 1 — Faction recovery on synthetic SBM

The simplest NP-hard target. Faction recovery is the signed-graph
analogue of community detection; on a stochastic block model with
*k* factions and a noisy observation channel, it is equivalent to
the planted-coloring problem (NP-hard in the worst case).

**Task:** generate `make_robot_network(n_factions=2, noise_prob=0.1)`
networks, predict edge signs with Gömb, measure faction-recovery
accuracy.

**Metric:**
- Edge-sign AUC (the v0.5 cliques predictor already does this).
- Faction-recovery accuracy: cluster vertex embeddings with k-means
  on k=`n_factions`, match against ground-truth labels via Hungarian
  algorithm.

**Baselines to beat:**
- Random baseline: 1/k.
- Spectral clustering on the *unsigned* graph (ignores sign).
- Spectral clustering on the *signed Laplacian* (Kunegis et al. 2010).
- Bron-Kerbosch + balance check (current `enumerate_balanced_cliques`):
  use largest balanced clique as a faction seed, grow greedily.

**Headline target:** Gömb beats signed-Laplacian on n_factions ∈ {2, 3}
across n_robots ∈ {20, 50, 100} at noise_prob ∈ {0.05, 0.10, 0.20}.
A 3×3 grid, ~9 cells × 5 seeds = 45 runs, each ~3 s on CPU.

**Files:**
- New: `signedkan_wip/src/demo/cliques_eval.py` — faction-recovery
  metric, Hungarian-matched accuracy, all baselines.
- New: `signedkan_wip/experiments/cliques_faction_recovery_2026_05_14.py`
  — sweep runner that emits JSONL.
- New: `signedkan_wip/tests/test_cliques_faction_recovery.py` — unit
  tests for the metric + baseline correctness on hand-built cases.
- Modified: `signedkan_wip/src/demo/gui.py` — extend the Cliques tab
  with a "Train predictor + recover factions" sub-section that reports
  side-by-side Gömb vs. signed-Laplacian numbers.

**Effort:** ~1 day. The training loop already exists
(`train_edge_sign_predictor`); we add the cluster-matching evaluation
on top.

### Stage 2 — Balanced-clique extraction via Gömb features

**Task:** use Gömb's trained vertex embeddings to seed a greedy
balanced-clique-expansion algorithm. Compare against the exact
baseline (Bron-Kerbosch + balance check) and the obvious heuristic
baselines.

**Algorithm sketch (greedy balanced expansion):**

```
1. Sort vertices by ‖embedding‖ descending (or by triangle-balance score).
2. For each candidate seed v:
   3.   clique ← {v}
   4.   repeat:
   5.     pick the vertex w maximising
            σ_product(w, clique) × cosine_similarity(emb_w, mean(emb_clique))
   6.     if extending the clique keeps σ-product = +1 AND w is
            adjacent to all of clique: add w; else stop.
   7.   record clique
8. Deduplicate, sort by size, return top-K.
```

**Metric:**
- Largest balanced clique size recovered.
- Wall-time vs. graph size (Bron-Kerbosch should blow up
  exponentially; Gömb-greedy stays poly).
- Recall@k against ground-truth (planted) balanced cliques in
  hand-crafted networks.

**Baselines:**
- Bron-Kerbosch + balance check (exact, exponential — the existing
  `enumerate_balanced_cliques`).
- Pure triangle-density vertex ranking + greedy expansion (no
  learned embeddings).
- Spectral relaxation of max-balanced-clique (Hochbaum 1998-style).

**Headline target:** On networks of n_robots ∈ {50, 100, 200} where
Bron-Kerbosch becomes prohibitive, Gömb-greedy recovers ≥ 80% of the
largest balanced clique's size in ≤ 1 s. Show a wall-time vs. graph-
size plot where Bron-Kerbosch crosses 1 minute around n=80–100 while
Gömb-greedy stays flat.

**Files:**
- New: `signedkan_wip/src/demo/cliques_extract.py` — greedy expansion
  algorithm, embedding-similarity scoring, exhaustive-vs-greedy harness.
- New: `signedkan_wip/tests/test_cliques_extract.py` — invariants
  (returned cliques really are cliques, really are balanced, sorted
  by size).
- New: `signedkan_wip/experiments/cliques_extraction_walltime_2026_05_14.py`
  — wall-time scaling sweep.
- Modified: `signedkan_wip/src/demo/gui.py` — add to the Cliques tab
  a "Greedy extraction from Gömb features" view alongside the exact
  Bron-Kerbosch one.

**Effort:** ~1 day. The expansion algorithm is the bulk; the test
suite locks the invariants.

### Stage 3 — Real-data validation (optional, for paper)

**Task:** apply Stage 1 + Stage 2 to Bitcoin Alpha / OTC / Slashdot
where we have real signed-graph data and trained checkpoints.

**Bitcoin Alpha** has a known balance structure (Cartwright-Harary
analysis published in [memory project_axiom_beats_attention_2026_05_05]).
Test: do Gömb-recovered factions / balanced cliques correspond to
known trader clusters?

**Metric:** Qualitative — do recovered factions cluster sensibly? Plot
the network with vertex colour = recovered faction. Look for
structural agreement with Davis 1967 / Cartwright-Harary 1956 patterns.

**Files:**
- New: `signedkan_wip/experiments/cliques_bitcoin_validation_2026_05_14.py`
  — load the Bitcoin Optuna-best checkpoints from
  `checkpoints/hsikan/`, extract vertex embeddings, run Stage 1 + 2
  pipelines, dump results.
- New: `reports/2026-05-14-cliques-bitcoin-validation.md`.

**Effort:** ~half day. Most of the infrastructure is reused; the new
work is plotting + writing.

## Test strategy

- **Unit:** faction-recovery accuracy metric (Hungarian matching);
  greedy-expansion invariants (clique + balanced); embedding
  extraction is deterministic per checkpoint.
- **Integration:** end-to-end on a 30-robot 2-faction network: train
  Gömb → recover factions ≥ 0.85 accuracy in ≥ 4/5 seeds.
- **Performance:** wall-time sweep, n_robots ∈ {50, 100, 200, 500},
  Gömb-greedy must stay sub-second; Bron-Kerbosch is allowed to
  blow up (that's the point).

## Performance budget

- Stage 1 sweep: ~45 runs × 3 s = ~2 min CPU.
- Stage 2 wall-time sweep: scales with `n_robots`; budget < 30 min.
- Stage 3 Bitcoin run: ~5 min (forward pass + clustering).
- Peak RSS: under 2 GB throughout (Gömb at hidden=12 is tiny).
- GPU optional, CPU sufficient.

## Risk anticipation

- **Stage 1 worst case:** Gömb plateaus at random performance like the
  partial v0.5 results above. Mitigation: more training data
  (multi-graph corpus), bigger hidden, longer training. If Gömb still
  can't beat signed-Laplacian on a clean SBM, the headline claim is
  dead and we pivot to a viz-only narrative for Niitsuma.
- **Stage 2 worst case:** greedy expansion is myopic; the largest
  balanced clique is often not reachable from any vertex by greedy
  σ-product steps. Mitigation: multi-seed greedy + cross-validation
  against ground truth on planted-clique benchmarks. If greedy
  doesn't recover ≥ 60% of the largest, fall back to spectral
  relaxation seeded by Gömb embeddings (still novel — pos relaxation
  + learned features is fresh).
- **Stage 3 worst case:** Bitcoin balance structure doesn't replicate
  cleanly with our recovered factions. Acceptable — Stage 3 is for
  "this transfers to real data" credibility, not load-bearing.

## Why no TikZ/PDF/Mermaid plan (yet)

This is still in research-explore mode — same rationale as the
kinematic and cliques v0/v1 plans. If Stage 1 produces a credible
beat-the-baseline result, **upgrade to a four-format plan immediately**
because this then becomes paper-shaped work.

## Empty-plan-dir hygiene

If abandoned, delete `docs/plans/2026-05-14-gomb-np-hard-approximation/`.

## Order of work

1. Plan reviewed by user (this doc).
2. Stage 1 — faction recovery.
3. **GO / NO-GO decision after Stage 1**: if Gömb beats signed-Laplacian
   on the 3×3 sweep, continue to Stage 2; if not, pivot back to
   descriptive demos.
4. Stage 2 — balanced-clique extraction.
5. **Niitsuma talk** — present Stage 1 + 2 deliverables.
6. Stage 3 — Bitcoin validation (for paper writeup, not the talk).
