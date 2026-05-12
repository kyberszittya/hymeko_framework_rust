# Epinions infrastructure plan — sparse attention + learnable incidence

*Written 2026-05-10, while the edge_cr + balance pruner 5-seed runs on Epinions overnight. This plan exists so the next architectural step is staged regardless of which way the morning result lands.*

---

## Where we are

| signal | current value |
|---|---|
| Epinions baseline (bigger_caps single-seed, scalar gate, default enum) | 0.8409 |
| Epinions today's edge_cr seed-0 (kernel ON, default enum) | **0.8611** (+0.020) |
| Epinions edge_cr 5-seed in flight | landing ~05:30 |
| Epinions edge_cr + balance pruner + cache 5-seed queued | landing ~10:00 |
| SGT Epinions reference | 0.941 |
| Current gap to SGT | −0.10 (after edge_cr seed-0: −0.08) |

Two morning outcomes drive the next move:

```
                        Morning result
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
          ≥ 0.870           0.84-0.86       < 0.84
        breakthrough        partial gain    null
                │             │             │
        (defer infra,    (try one infra    (need infra,
         push for better  path; pick the    BOTH paths
         seeds / longer  cheaper one)       in flight)
         training)
```

This document plans for the right two columns. The left column ("breakthrough") is the happy path and needs no new code.

---

## Path A — sparse attention scaling

### Hypothesis

The dense Hamilton attention pool computes a per-query weight over **every** cycle in the pool. On Epinions with 100-200K cycles, the gate-never-opens diagnosis of `project_epinions_ceiling` may stem from gradient signal being diluted across too many low-quality cycles. Top-K attention would let the model commit gradient signal to a small subset of cycles per query, recovering the inductive bias that "only a few cycles matter for any given edge."

### Synthetic test design — `needle_in_haystack`

A graph constructed so dense attention provably fails and top-K attention provably succeeds:

- $N = 1000$ vertices, partitioned into two communities $A, B$ of size 500 each.
- $S = 50$ **signal cycles**: balanced 4-cycles entirely within $A$ or entirely within $B$. The community label is decodable from these.
- $M = 5000$ **noise cycles**: random 4-cycles with signs sampled uniformly from $\{+1, -1\}$. Independent of community labels.
- Test edge labels: predict whether two vertices are in the same community.
- Cycle pool: $S + M = 5050$ cycles, of which $1\%$ carry signal.

**Expected behavior:**
- Dense attention with current pool: AUC near 0.5 (signal swamped by noise).
- Top-K attention at K=8 with a sane scoring head: AUC > 0.9 once the head learns to score signal cycles above noise.
- Linear oracle on signal cycles only: AUC = 1.0 (sanity ceiling).

If this test fails, the infrastructure is wrong. If this test passes, deploy on Epinions.

### Implementation sketch

1. New module `signedkan_wip/src/sparse_attention.py`. Adds a `SparseAttentionPool(nn.Module)` parallel to the existing dense attention path.
2. Per-query scoring head: `score = h_query @ W_score @ h_cycle + b` (cheap, $O(d \cdot d')$ per cycle).
3. Top-K selection: `torch.topk(scores, K)` with a straight-through estimator for gradient flow ($\partial K / \partial \text{score}$ ignored; standard practice). Optional: differentiable Gumbel-top-K for full gradient flow if straight-through stalls training.
4. Full attention computed only over the K selected cycles per query.
5. Env-gated: `HSIKAN_SPARSE_ATTN=1`, `HSIKAN_SPARSE_ATTN_K=8`.

Estimated effort: **1 day** (new module + integration + sanity test on `needle_in_haystack`).

### Ablations once baseline lands

| variant | K | expected use |
|---|---|---|
| dense (current) | all | baseline |
| top-K | 8, 32, 128 | sweep — too low under-fits, too high reverts to dense |
| top-K + balance pruner | 8, 32 | composition test — does axiom selection at the cycle level + top-K at the query level stack? |

---

## Path B — learnable incidence

### Hypothesis

The current $M_e$ matrix encodes the **fixed** signed-incidence between vertices and cycles: $M_e[v, c] \in \{+1, 0, -1\}$ depending on whether vertex $v$ is on cycle $c$ with which sign. This treats every (vertex, cycle) incidence as equally informative. A **learnable** $M_e$ would re-weight or re-route the incidence based on features — closer to a graph attention over the bipartite vertex↔cycle bipartite graph.

### Synthetic test design — `feature_conditioned`

A graph where cycle importance depends on **vertex features**, not just cycle structure:

- $N = 1000$ vertices, each with a 4-dim feature vector.
- Features drawn from two modes (clusters in feature space). The mode label is the prediction target.
- Cycles generated to have **mixed-mode membership**: each cycle touches both feature modes.
- Signs assigned so that for vertices in **mode 0**, the cycle's "positive" subset is informative; for **mode 1**, the "negative" subset is informative.
- Fixed $M_e$ (uniform incidence) cannot distinguish which subset of any given cycle to use.
- Learnable $M_e$ conditioned on vertex features should recover near-perfect AUC.

**Expected behavior:**
- Fixed $M_e$ HSiKAN: AUC ≈ 0.5 + ε (cycle-level signal is mode-canceling).
- Learnable $M_e$ HSiKAN: AUC > 0.9 once the M_e module learns to weight by feature.

### Implementation sketch

1. New module `signedkan_wip/src/learnable_incidence.py`.
2. Several variants:
   - **Linear**: $M_e^{learned}[v, c] = M_e^{fixed}[v, c] \cdot \sigma(W \cdot h_v + b)$. Cheapest; one scalar per vertex.
   - **Bilinear**: $M_e^{learned}[v, c] = M_e^{fixed}[v, c] \cdot \sigma(h_v^{T} W h_c)$ where $h_c$ is a per-cycle embedding. Captures interaction.
   - **MLP**: small MLP over $(h_v, h_c, M_e^{fixed}[v, c])$. Most expressive.
3. Replace the fixed $M_e$ construction in the encode_edges path with the learnable variant. Backward compatibility via env var `HSIKAN_LEARNABLE_M_E={off,linear,bilinear,mlp}`.
4. Gradient flows through both the spline parameters and the M_e weights — same Triton kernel can be reused (the M_e is already a runtime tensor).

Estimated effort: **2 days** (more involved than Path A; requires touching the encode_edges sparse-mm path).

### Ablations once baseline lands

| variant | what it adds | expected gain on `feature_conditioned` |
|---|---|---|
| fixed $M_e$ | baseline | AUC ~ 0.5 (predicted) |
| linear $M_e$ | per-vertex scalar weight | AUC ~ 0.7 (partial recovery) |
| bilinear $M_e$ | (vertex × cycle) interaction | AUC ~ 0.9+ |
| MLP $M_e$ | full expressivity | AUC ~ 0.9+ (might over-fit small graphs) |

---

## Synthetic test harness — common to A and B

One generator script: `signedkan_wip/src/synthetic_signed_graphs.py`.

| dataset | size | role |
|---|---|---|
| `easy_sbm` | $N=200$, balanced 2-block | sanity: any architecture should solve |
| `needle_in_haystack` | $N=1000$, 1% signal | tests Path A (sparse attention) |
| `feature_conditioned` | $N=1000$, feature-dependent | tests Path B (learnable M_e) |
| `dense_walk` | $N=2000$, dense | tests scaling of both |

Each generator:
- Returns a `SignedGraph` plus a ground-truth label tensor.
- Provides an `oracle_auc(model_pred)` that reports how close the model is to the achievable ceiling.

This is the morning's first deliverable: synthetic harness is fast (~minutes per test), gives ground-truth-bounded AUC, and we know the answer before running on Epinions.

---

## Decision tree

```
edge_cr + balance + cache 5-seed result on Epinions
    │
    ├── ≥ 0.87 (breakthrough)
    │     → write up; defer Path A and B
    │
    ├── 0.85 - 0.87 (partial)
    │     → run synthetic A + B harness today
    │     → ship whichever validates faster
    │     → 5-seed Epinions with that lever next night
    │
    └── < 0.85 (null on balance pruner too)
          → run synthetic A + B harness today
          → ship BOTH (parallel work; Path A is faster)
          → 5-seed Epinions with sparse attention next night
          → 5-seed Epinions with learnable M_e the night after
```

---

## What lands this morning regardless of result

1. `signedkan_wip/src/synthetic_signed_graphs.py` — generator with three test sets + oracle baselines.
2. `signedkan_wip/tests/test_synthetic_signed_graphs.py` — sanity tests for the generator itself (does the SBM look like an SBM? Are signal cycles labelled correctly?).
3. This document, frozen.

These are useful even if Epinions breaks through — they become the standard regression suite for any future architectural lever (e.g., the Triton kernel work could have been validated against `easy_sbm` before going to Slashdot).

---

## Files of record

- `docs/plans_epinions_infrastructure_2026_05_10.md` — this document
- `signedkan_wip/src/synthetic_signed_graphs.py` — generator (to land)
- `signedkan_wip/tests/test_synthetic_signed_graphs.py` — generator tests (to land)
- `signedkan_wip/src/sparse_attention.py` — Path A implementation (planned)
- `signedkan_wip/src/learnable_incidence.py` — Path B implementation (planned)
