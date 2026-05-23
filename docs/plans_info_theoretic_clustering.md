# Information-theoretic clustering for connection pruning

User idea, 2026-05-05 evening:
> "what about pruning or distributing connections based on connection
> clusters based on information theory"

This connects to four existing threads in the codebase:
- `signedkan_wip/src/run_path_i_total_correlation.py` — Path I total
  correlation MI experiments (memory entry from 2026-04-26)
- `signedkan_wip/src/cross_branch_reg.py` — cross-branch info regulariser
- `signedkan_wip/src/entropy_reg.py` — Lyapunov-safe spectral entropy
- The empirical "regime split" we locked in today: balance vs unbalanced
  axiom is graph-conditional

## The core observation

Today's regime-split finding (Slashdot wants `unbalanced`, Bitcoin Alpha
wants `balance`) is at the **graph level** — one global axiom per
dataset. But:

- Real graphs are **heterogeneous**: dense conflict regions next to
  cooperative regions in the same network (e.g., a Slashdot
  sub-community of activists, a sub-community of casual contributors).
- Today's pipeline applies *one* axiom globally, missing the
  community-conditional structure.
- The user's idea: **detect communities, apply per-community axiom
  selection, prune connections by per-community information content**.

This is the natural next step beyond "graph-conditional axiom" to
**"community-conditional axiom"**. And it's information-theoretically
grounded: a community where balanced cycles dominate the prediction
signal should use `balance`; one where frustrated cycles dominate
should use `unbalanced`. Mutual information $I(\text{cycle features};
\text{edge sign})$ measures this directly.

## Concrete scheme

Three layers stack from cheap-to-expensive:

### Layer 1 — community detection (cheap)

Apply Louvain (or signed-Louvain, or spectral clustering) to the
**unsigned** adjacency to get communities $C_1, \ldots, C_K$. Each
edge belongs to either an intra-community or an inter-community
class. Cycles inherit a community-membership signature
$\mathrm{comm}(\text{cycle}) = $ multiset of communities its
vertices touch.

For each community $C_i$, compute its local balance ratio
$\beta_i = $ (\#balanced triads in $C_i$) / (\#triads in $C_i$).

### Layer 2 — community-conditional axiom selection

For each community $C_i$:

- if $\beta_i \geq 0.85$: apply `balance` pruner to cycles internal
  to $C_i$
- if $\beta_i \leq 0.75$: apply `unbalanced` pruner internal to $C_i$
- else (intermediate): no axiom pruning, just BFS distance

Inter-community cycles are scored separately --- they may carry the
strongest sign-prediction signal because they bridge regimes.

This is essentially the existing `CompositePruner` infrastructure
applied **conditional on cycle community membership**. Implementation
just needs an extension to `hymeko_graph::cycle_enum` so the pruner
function takes the cycle's community-membership tuple.

### Layer 3 — information-theoretic scoring (the deepest version)

For each candidate cycle, compute its **mutual information** with
the test-edge label distribution:

$$
I(\text{cycle}; \text{label}) =
\sum_{c, y} p(c, y) \log \frac{p(c, y)}{p(c) p(y)}
$$

where $c$ is a discrete signature of the cycle (its sign-product, its
balance class, its community signature) and $y$ is the sign of edges
incident to it.

In practice, MI is estimated from the training set:

1. For each cycle in the candidate pool, count $(c, y)$ co-occurrences
   in the training set
2. Compute empirical MI via the standard plug-in estimator
3. Use $I(\text{cycle}; \text{label})$ as the **scorer** in the existing
   top-K vertex-stratified framework

This replaces the heuristic `fraction_negative` scorer with a learned
information-theoretic one. The promise: cycles that are **independent
of the label** ($I \approx 0$) get pruned regardless of axiom; cycles
that are **highly informative** ($I \gg 0$) get kept regardless of
balance class.

### Connection to existing Path I total correlation

The `project_path_i_total_correlation_mi.md` memory describes:

> L-way joint TC + KL-feedback λ + 3-mode variance momentum

This is the same mathematical apparatus, applied at the loss-function
level (regularising the joint distribution of αₖ-weighted arity
representations). Layer 3 above applies it at the **cycle-selection
level**, complementing the αₖ regularisation.

## Why this should help, mechanistically

1. **Bitcoin Alpha is a cooperative network with mostly trust edges.**
   Most cycles are balanced. The `balance` pruner globally is right.
   But: there are sub-communities of bad actors. Cycles in those
   sub-communities are predictively informative *because* they are
   unbalanced. Today's global `balance` pruner removes them.
   Community-conditional pruning keeps them.

2. **Epinions is the dense intermediate.** Today's m=16 result
   (0.7088) shows it's signal-poor at small m. A community detection
   step would expose that "review-of-review" sub-communities have
   very different balance characteristics from "trust-of-reviewer"
   sub-communities. Per-community top-K + per-community axiom would
   give richer cycle population per community at the same global
   `m × |V|` budget.

3. **Slashdot's headline 4.4σ regime split likely understates the
   true effect.** It's averaged across the whole graph; the
   community-conditional version would have larger pure effects in
   the most-conflict-dense communities and near-zero effects in the
   most-cooperative ones. Cleaner story for paper.

## Implementation plan

### Phase A — minimal viable (this week)

1. Add Louvain community detection to `hymeko_graph` (or use existing
   `petgraph` integration). Cluster the **unsigned** edge list.
2. Compute per-community balance ratio in one BFS pass.
3. Extend `CompositePruner` to support per-cycle community-membership
   conditioning.
4. PyO3 bridge: `enumerate_top_k_per_vertex_cycles_community_signed_rs`
   that takes a community-id-per-vertex array.
5. Smoke test on Bitcoin Alpha; expect minor lift since it's
   already small/uniform.

### Phase B — full information-theoretic scoring (next 2 weeks)

1. Implement the empirical MI estimator over the training set.
2. Wire it as a new scorer choice (`HSIKAN_TOPK_SCORER=mutual_info`).
3. Test on all 3 datasets at the locked-in best configs.
4. Compare MI-scored top-K to fraction_negative top-K.

### Phase C — community-conditional MI (the full proposal)

1. Per-community MI estimation (cycle features × per-community edge
   labels).
2. ABB over `(community, axiom-choice)` pairs; the P-graph becomes
   community-stratified.
3. Final scorer: combine community MI + global MI with a
   learnable weight (the αₖ pattern).

### Phase D — research extensions

1. **Information bottleneck** on the cycle embeddings: train them
   to maximise $I(\text{embedding}; \text{label}) - \beta I
   (\text{embedding}; \text{full cycle})$. This formalises "compress
   cycles to their predictively-informative content."
2. **Total correlation across arities**: extend the αₖ mixer to
   minimise total correlation between cycle representations at
   different arities, encouraging each arity to capture
   *complementary* information. Connects directly to the existing
   Path I work.

## Expected lift

- Phase A (community-conditional axiom): **+0.005–0.020 AUC** on
  heterogeneous datasets (Epinions especially), small lift on
  uniform ones (Bitcoin Alpha).
- Phase B (MI-scored top-K): **+0.010–0.030 AUC** as long as the
  empirical MI estimator is well-calibrated (n_train ≥ 1000).
- Phase C (community-conditional MI): hard to predict; the synergy
  could be additive or sub-additive. Best case: **+0.020–0.040 AUC**.
- Phase D: research-grade contribution. Not gunning for AUC on the
  next iteration but for theoretical clarity + potential journal
  paper.

## Risk and caveats

1. **Empirical MI estimator bias** at small training sets.
   Bitcoin Alpha has only ~24K training edges; MI estimates from
   that few samples are noisy. Need bias-corrected estimators
   (e.g., Miller-Madow, NSB) if we go below ~1000 samples per
   community.

2. **Community detection stability**. Louvain is non-deterministic;
   would need to fix the seed or ensemble across multiple runs.
   Spectral clustering on the signed Laplacian is more stable but
   more expensive.

3. **Combinatorial blow-up of community signatures**. With $K$
   communities and arity $k$, there are $\binom{K+k-1}{k}$ possible
   community-membership multisets per cycle. For $K=10, k=4$ that's
   715. Manageable, but the per-signature MI estimate becomes
   noisy. Solution: collapse signatures by symmetry classes or by
   "purity" (fraction of vertices in the dominant community).

4. **Potential null result**. If the global axiom is already
   close-to-optimal (Bitcoin Alpha case), community-conditional
   axiom adds nothing. We'd expect the **biggest gains on Epinions**
   (the dense intermediate where global axiom doesn't help) and
   modest gains on Slashdot.

## Connection to the P-graph framework

This extension fits *naturally* into the P-graph picture we developed
this morning:

- **Materials**: per-community resource budgets (cycle quota,
  training time)
- **Operating units**: per-(community, axiom-choice) cycle-selection
  units
- **Axioms**: community-internal feasibility (balance ratio $\to$
  axiom choice)
- **MSG**: trim community-axiom pairs that don't reach the
  prediction target
- **ABB**: pick the cost-minimum set of (community, axiom) pairs
  that produces $\text{auc\_score}$

The framework's expressiveness has been the central message all day.
This extension keeps the P-graph methodology as the architectural
glue while adding **information-theoretic semantics** to the
operating-unit-cost function.

## Action: start Phase A tomorrow

If time permits, while the queued GPU experiments finish overnight:

1. Add `petgraph` (or equivalent) Louvain to `hymeko_graph`.
2. Compute per-community balance ratio for Slashdot/Epinions.
3. Sketch the `enumerate_top_k_community_aware` Rust function.

Phase A should land within 1 day. Phase B + C are 1-2 weeks.

---

**Bottom line**: this is a real research direction, well-aligned with
both the P-graph methodology (community-stratified MSG/ABB) and the
existing total-correlation work in the codebase. It's the next step
after "graph-conditional axiom" → **"community-conditional axiom +
information-theoretic scoring"**.
