# Self-evolving cycle sampling — Gömb-guided cycle enumeration

**Date:** 2026-05-14
**Status:** **research direction**, sized at 2-3 weeks for V1 (static).
**Origin:** user observation during the cycle-enumeration scaling
sweep — "if Gömb can detect cycles, it can accelerate k-enumeration,
follow a graph distribution and sample." The cap on
`enumerate_k_cycles_rs` throws away 99.6% of cycles on Slashdot k=4
(measured 2026-05-14: 200k retained out of an estimated 55M+
total). Gömb-guided sampling is the principled answer to *which*
99.6% to throw away.

## The claim

> Cycle enumeration on large signed graphs is bounded by a hard cap
> (`max_cycles`) that discards information indiscriminately. Gömb's
> own gradients tell us **which cycles its predictions depend on**.
> Feeding those importance signals back into the enumerator turns
> blind brute-force truncation into **learned importance sampling**
> — fewer cycles retained at equal-or-better downstream accuracy.

This is bootstrap / EM-style training. It is also, more
ambitiously, a step toward **architectures that learn to allocate
their own compute** — meta-learning for graph neural networks.

## Why this matters

Measured 2026-05-14 (cycle-detection sweep):

| Dataset | k | Wall (cap=200k) | Cap-bounded? | Memory log |
| --- | ---: | ---: | --- | --- |
| bitcoin_alpha | 3 | 0.011 s | no (22k cycles total) | full enum |
| bitcoin_alpha | 4 | 0.062 s | **yes** | hits cap |
| slashdot | 3 | 0.495 s | **yes** | hits cap |
| slashdot | 4 | 5.034 s | **yes** | hits cap (memory: ~55M total uncapped) |
| epinions | 3 | 1.174 s | **yes** | hits cap |
| epinions | 4 | 36.846 s | **yes** | hits cap |

For every large graph at k ≥ 3 the cap is binding. The current
remedy is the per-vertex top-K mode (`HSIKAN_TOPK_MODE=per_vertex`,
`HSIKAN_TOPK_K=128`) which uses hand-picked scorers (`balance`,
`fraction_negative`, `entropy`). Two memories
(`project_global_topk_ladder_null_2026_05_10`,
`project_abb_global_topk_2026_05_10`) record that hand-picked
**global** top-K *fails* to replace per-vertex top-K. The natural
follow-up: **let Gömb learn the scorer.**

## CORE.YAML items touched

**Empty for V1.** All work under `signedkan_wip/src/demo/` and the
existing Rust enumerator's optional weighted-DFS path. No CORE
crate modified, no pinned-dep changes.

**V3 (joint differentiable sampling) may touch CORE** — would need a
differentiable DFS approximation, likely a new Rust path. Escalate
under §1 when V3 is the active priority.

## Three implementation designs, ranked by feasibility

### Design A — Static, two-pass (V1, ~2 weeks)

The cheapest design. Validates the core claim before any
infrastructure investment.

**Procedure:**

1. Train Gömb on **uniformly-sampled** cycles at the standard cap
   (the current pipeline).
2. After training, compute per-vertex importance scores:
   `s(v) = ‖∂L_test / ∂node_embed[v]‖`
   on a held-out edge set. These are the gradients telling Gömb
   "this vertex matters to predictions".
3. **Re-enumerate**, biasing the DFS: at each branch in the cycle
   enumerator, the probability of continuing through vertex `v` is
   proportional to `s(v) / Σ_u s(u)` (softmax with temperature `τ`).
4. Retrain Gömb on the biased sample.
5. Compare: AUC at fixed cap (does the biased sample beat
   uniform?), AUC at half the cap (do we get equal AUC at half the
   compute?).

**Hyperparameters:**
- Sampling temperature `τ ∈ {0.5, 1.0, 2.0}`. Low τ = aggressive
  importance sampling; high τ = closer to uniform.
- Uniform exploration mix `λ ∈ {0.0, 0.1, 0.2}`. The biased
  sample is `λ · uniform + (1 - λ) · importance` — `λ > 0` is the
  guard against model collapse (sampler over-emphasises easy cycles
  and loses generalisation).

**Headline target:** on Slashdot k=4, recover ≥ 95% of the
full-cap AUC at **half** the cap.

**Files (V1):**

- New: `signedkan_wip/src/demo/cycle_importance_sampler.py` —
  Python wrapper that (a) computes the importance scores from a
  trained model, (b) builds a vertex-biased sampling distribution,
  (c) calls into a slightly-extended Rust enumerator that accepts
  per-vertex weights.
- Extended: `hymeko/src/...` Rust enumerator — add an optional
  `weights: Option<Vec<f32>>` parameter to `enumerate_k_cycles_rs`.
  When `Some`, the DFS branch ordering is biased by those weights;
  when `None`, behaviour is unchanged. **Additive API — fits within
  `hymeko` crate's `lockdown: implementation`.**
- New: `signedkan_wip/experiments/cycle_importance_v1_2026_05_14.py`
  — sweep script: 4 datasets × 3 (`τ`, `λ`) × 5 seeds, JSONL output.
- New: `signedkan_wip/tests/test_cycle_importance_sampler.py` —
  deterministic-with-seed sampling, weight-zero vertex correctly
  excluded, weight-uniform recovers baseline.

### Design B — Iterative, EM-style (V2, ~3 weeks)

Builds on V1. The static design uses *one* importance pass; the
iterative design alternates sampling and training over multiple
outer rounds, with Gömb's gradients getting progressively sharper.

**Procedure:**

```
init: uniform cycle sample
for outer_round in 1..R:
    train Gömb on current sample
    compute importance scores from gradients
    re-enumerate with new bias
    record validation AUC
```

- Outer rounds `R ∈ {3, 5, 8}`.
- Per-round AUC trajectory: does it improve monotonically or
  saturate / collapse?
- Compare to a "stuck on round 1" V1 baseline.

**Risks (explicit):**

- **Mode collapse.** Iterative re-sampling tends to over-fit to the
  cycles Gömb already predicts well, losing generalization. The
  `λ` uniform-mix from V1 generalises here, but its right value
  may change with outer-round number.
- **Compute amplification.** R = 8 rounds means 8× the enumeration
  cost (offset by smaller cap per round if the bias works).

**Headline target:** on Slashdot k=4, V2 with `R=5` beats V1 at
the same total cycle-enumeration budget.

### Design C — Joint, end-to-end differentiable (V3, ~4-6 weeks)

The hardest, cleanest version. The sampler and the model are
trained jointly with a Gumbel-softmax (or Bernoulli relaxation)
over candidate DFS continuations. The model's loss gradient flows
through the sampling distribution to update which cycles get
enumerated next.

**Procedure (sketch — needs more thought before commit):**

1. For each DFS frontier vertex set, parameterise a categorical
   distribution over which neighbour to extend the cycle into.
2. Sample via Gumbel-softmax; the forward pass enumerates a
   *soft* cycle (weighted combination of candidates).
3. The model's gradient flows backward through the soft cycle into
   the categorical parameters, updating them.
4. Anneal the Gumbel temperature toward zero during training to
   make cycles hard at convergence.

**Why this is harder:**

- DFS is intrinsically discrete; relaxation introduces bias that
  needs careful analysis.
- Rust-side soft-cycle support is non-trivial. Likely a Python-side
  prototype first, then Rust port.
- Convergence guarantees? Gumbel-softmax has known training
  stability issues at low temperatures.

**Connections to literature:**

- *Learning to search* (Chen & Bansal 2018, RetroXpert).
- *Differentiable random subgraph sampling* — GraphSAINT (Zeng et
  al. 2020), GraphSAGE (Hamilton et al. 2017), FastGCN.
- *Neural architecture search via gradient* (DARTS) — same
  relaxation trick on a different combinatorial structure.

## What this is NOT

- **Not** a Niitsuma talk deliverable. The talk should focus on
  the cliques v0.5 + NP-hard Stage 1 demos. Self-evolving sampling
  is the *closing slide* alongside the belief-planning bridge.
- **Not** a replacement for the per-vertex top-K path. V1 *augments*
  the existing scorers (balance, entropy) with a learned variant.
  If V1 beats the hand-picked scorers, we have a real result; if
  not, the hand-picked scorers stay.
- **Not** a build commitment for the next session. V1 is the
  smallest viable step but requires a Rust extension to
  `enumerate_k_cycles_rs` — that's a careful change, not a
  weekend project.

## Test strategy

### V1

- **Unit:** importance score computation matches a hand-computed
  reference on a 3-vertex toy graph; uniform-weight sampler
  recovers baseline cycle counts; zero-weighted vertex never
  appears in returned cycles.
- **Integration:** end-to-end pipeline (train → score → re-enumerate
  → retrain) runs in < 5 min on Bitcoin Alpha; reported AUC delta
  is reproducible to 1e-4 with fixed seed.
- **Performance:** sampling-biased DFS on Slashdot at cap=100k
  must finish in ≤ 1.5 × the wall of uniform DFS at the same cap.
  If the weight-lookup overhead exceeds that, the DFS path is too
  slow and needs Rust-level optimisation.

### V2 + V3

Defined when those plans become active.

## Performance budget

- V1 single training cycle: ~5 min on Bitcoin (small graphs),
  ~20 min on Slashdot, ~45 min on Epinions (training only,
  enumeration is small fraction).
- V1 sweep (4 datasets × 3 (`τ`, `λ`) × 5 seeds = 60 runs):
  ~12 h total wall, GPU optional.
- Peak RSS: under 4 GB (Gömb at hidden=12 + Slashdot graph).

## Risk anticipation

- **Importance scores are noisy.** Gradients from a single trained
  model are noisy. Mitigation: average over multiple seeds before
  building the sampler.
- **Cap-bounded baseline already loses information.** The "biased
  sample beats uniform" claim is only meaningful if the uniform
  baseline is itself a meaningful target. On graphs where the cap
  retains < 1% of cycles, the uniform sample is so lossy that
  beating it isn't a strong claim. Mitigation: pick experimental
  caps where uniform retains 5-20% of total cycles, so there's room
  to demonstrate genuine improvement.
- **Rust API change.** Extending `enumerate_k_cycles_rs` with
  weighted DFS adds a new internal branch. Lockdown is
  `implementation`, so additive API is allowed, but every new
  branch needs regression tests on the unweighted path.
- **Stochasticity vs. caching.** The cycle cache
  (`~/.cache/hymeko/cycles_v1/`, controlled by
  `HYMEKO_CYCLE_CACHE=1`) keys off graph hash + topk fingerprint.
  Per the cycle-cache-fingerprint memory, any new env var that
  affects enumeration must be added to `_topk_fingerprint`. The
  importance-weighted sampler counts: its seed + temperature +
  uniform-mix must be in the cache key.

## Connection to other plans

- **`docs/plans/2026-05-14-gomb-np-hard-approximation/plan.md`** —
  the NP-hard pivot's Stage 1 (faction recovery) and Stage 2
  (balanced-clique extraction) both consume cycle samples. If
  self-evolving sampling improves the cycle-feature distribution,
  it tightens the NP-hard claims. The two plans are independent
  but compose: Stage 1 establishes Gömb beats hand-picked scorers
  on faction recovery; this plan establishes Gömb beats hand-picked
  scorers on its own training-data selection. Both make the same
  argument from different angles.
- **`docs/plans/2026-05-14-cliques-generation-detection/plan.md`** —
  the foundation harness measures detection recall vs. wall-time;
  the same harness extends naturally to "wall-time at fixed
  recall", which is the metric this plan optimises.
- **`docs/plans/2026-05-14-gomb-belief-planning-bridge/plan.md`** —
  V2 of the bridge plan (Gömb on contracted hypergraphs) is closely
  related: both are "Gömb's own outputs feed back into Gömb's
  inputs". Bridge V2 contracts the graph topology; this plan
  reweights the sampling. They could compose into V4: contracted
  hypergraph + importance-weighted cycles at every scale.

## Why no TikZ/PDF/Mermaid plan (yet)

Same rationale as the other 2026-05-14 plans — exploratory research,
small interface surface (additive Rust API, additive Python
wrapper). Upgrade to four-format when V1 shows a positive result
and we're committing to the architectural follow-on.

## Empty-plan-dir hygiene

If permanently abandoned, delete
`docs/plans/2026-05-14-self-evolving-cycle-sampling/`. But this is
load-bearing for the broader "Gömb evolves itself" narrative; it
should outlive any single experiment failure.

## Slide-shaped narrative for the closing of the Niitsuma talk

Three sentences:

> The cap on cycle enumeration discards 99% of cycles
> indiscriminately. Gömb's own gradients tell us *which* cycles its
> predictions depend on — and feeding those scores back into the
> enumerator turns blind truncation into learned importance
> sampling. Same architecture, fewer cycles, better accuracy:
> **Gömb learns to allocate its own compute.**

## Order of work — V1 only, gated on the cliques foundation

V1 should not start until:

1. The cliques generation/detection foundation lands
   (`docs/plans/2026-05-14-cliques-generation-detection/plan.md`).
2. The NP-hard Stage 1 (faction recovery) returns a sensible
   result on faction-based synthetic networks.

Both prerequisites give us baseline numbers to compare against. Without
them, V1's "X% improvement" claim has no X.
