# SpherePlanner — hypergraph-native task/motion planning

**Date:** 2026-05-16
**Status:** research design + grounded reuse map + smallest viable experiment
**Companion artefacts:**
[`meta_task.hymeko`](../../data/robotics/meta_task.hymeko),
[`handover_task.hymeko`](../../data/robotics/sim/dual_fanuc/handover_task.hymeko)
(both from earlier tonight — the substrate this proposal builds on).

## TL;DR

Instead of emitting HymeKo task descriptions to PDDL / BT.CPP /
ROS 2 (the *adapter* route — useful but not novel), build a
**hypergraph-native planner** that uses the same machinery the
codebase has shipped for signed-cycle link prediction:

* signed hyperedges with σ-product semantics (HSiKAN),
* Forman / Ricci curvature κ per edge (`hymeko_gomb/soma/vision/forman.rs`),
* Hodge Laplacian Δ_k for spectral structural decomposition
  (`hodge.py`),
* top-k cycle enumeration with branch-and-bound, in Rust
  (`hymeko_py.enumerate_top_k_cycles_rs`).

The proposal name **SpherePlanner** (a deliberate callback to
GömbSoma = "sphere-soma" in Hungarian) names a planner that
treats the state-action hypergraph as its primary structural
object — not a graph compiled down from STRIPS operators.

Three falsifiable hypotheses (§3) about *what specifically*
the hypergraph structure buys you over A* / FastDownward /
PDDLStream. One smallest-viable experiment (§5) on a
multi-agent blocksworld variant that can decisively support or
falsify the κ-as-heuristic claim in a 1-week sprint.

## 1. The gap — where existing planners struggle

Pick the two domains where this codebase is already deeply
invested (multi-arm coordination, demonstrated in the
2026-05-16 dual-FANUC handover) and the two settings where
classical planners are weakest:

### 1.1 N-ary coordination is bolted on, not native

PDDL operators are unary in their preconditions and effects:
each precondition is a literal predicate over symbols. To express
"both arms must be at their handover poses *at the same time*",
classical PDDL needs auxiliary state variables (a `phase` counter
or per-agent boolean state predicates). The plan has to
*manufacture* synchrony at the symbol level.

Behavior trees handle parallel sub-plans natively (the
`<Parallel>` node) but the parallelism is human-authored, not
synthesised — you can't ask a BT-based system to *plan* a
parallel sub-tree that needs synchronisation.

A signed hyperedge of arity 2 — one vertex per arm's pose — is
the relation "these two arms are *both* at their respective
handover poses, *at the same time*". The hypergraph carries
N-ary synchrony as a structural primitive.

### 1.2 Disjunctive constraints are awkward

"Object should be at location A OR location B" is a disjunctive
precondition. Classical planners handle it by branching the
state space (compile to two operator variants, or via PDDL 3.0's
disjunctive preconditions which most planners support partially).

A signed-arity-3 hyperedge `(+ object, + loc_A, - loc_B)` with
σ-product semantics encodes "object at A OR (object at B and
loc_B is a fine choice)" naturally — the +/- pattern is the
disjunctive structure, not an annotation on top of it.

### 1.3 Plan-quality measures are local

A* / GOAP score plans by cumulative action cost. Quality
measures like "is this plan minimal under reordering?" or "does
it have parallelisable sub-plans?" are second-order analyses
done post-hoc.

The hypergraph has *spectral* quality measures available at
plan-search time:

* **Δ_0 connectivity** of the state-action graph below the
  partial plan tells you whether the remaining sub-task is
  decomposable.
* **Δ_1 cycle component** identifies plan-fragments that
  oscillate (consume action budget without making goal-state
  progress).
* **Forman κ** per outgoing action identifies bottleneck choices
  vs flexible ones.

These are *plan-time* heuristics derivable directly from the IR,
not features you'd have to extract by traversing the partial plan.

### 1.4 The signed-cycle inductive bias *should* transfer here

The negative result `hymeyolo_kcycle_negative_2026_05_14` bounds
the σ-cycle prior: "σ-products work where data is *natively*
signed; not where signs are derived from continuous positions."

Planning IS natively signed. A precondition is positive evidence
(this fact must hold for the action to apply). A conflict / a
mutex / a delete-effect is negative evidence. The signed-cycle
work that flopped on vision should *positively* transfer to
planning. **The negative on vision is the strongest existing
indirect evidence for planning being the right next-domain.**

## 2. The substrate — what's already in tree

This proposal does not require new core machinery. Every
component below ships in the codebase as of 2026-05-16.

| Capability                                   | Where it lives                                                                | Used today by                                  |
|----------------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------|
| Hypergraph IR (vertices, signed hyperedges)  | `hymeko_core`, `hymeko_query`                                                  | every emit / query path                        |
| Forman-Ricci curvature κ per edge            | `hymeko_gomb/soma/vision/forman.py` (Python ref); vectorised Rust in `hymeko_py` | GömbSoma vision backbone                       |
| Hodge Laplacian Δ_0, Δ_1, Δ_2 (sparse)        | `hymeko_gomb/soma/vision/hodge.py`                                              | GömbSoma Bochner-Hodge propagation             |
| Top-k cycle enumeration (Rust, parallel)     | `hymeko_py.enumerate_top_k_cycles_rs`, `hymeko_py.enumerate_k_cycles_rs`        | HSiKAN cycle pool                              |
| Signed-cycle σ-product evaluation             | HSiKAN forward pass                                                            | HSiKAN paper                                   |
| Branch-and-bound on top-k scorers            | `BoundedScorer` trait in `hymeko_graph`                                         | ABB global top-K (project memory 2026-05-10)   |
| Bochner-coupled message passing on hypergraph | `hymeko_gomb/soma/bochner.py`                                                  | GömbSoma Ricci-Stim backbone                   |
| Template-driven emit pipeline                 | `transforms/`, `hymeko_formats`                                                | URDF / SDF / MJCF / DOT / Gazebo / PyTorch     |

The pieces are designed for *graphs of structural relations*.
Planning is a graph of structural relations: state-action
hypergraph nodes, action-precondition / action-effect hyperedges
between them.

The work is *putting them together in a new way*, not building
new infrastructure.

## 3. Three falsifiable hypotheses

Each is sharp enough to support or falsify within one ~1-week
sprint. Each maps to a concrete experiment (§5) and to existing
codebase machinery (§6).

### H1 — Forman κ as an A* heuristic beats Manhattan-style heuristics on N-ary-coordination domains

**Claim:** on planning domains with explicit N-ary coordination
(multi-robot handover, multi-resource lock, multi-agent
delivery), the Forman κ of each frontier edge is a *better*
heuristic for A*-style search than:

* zero heuristic (Dijkstra baseline),
* per-action additive cost (heuristic = 1 per step, GOAP-style),
* learned heuristic from a small dataset of solved plans (GNN
  baseline at iso-parameter count).

**Why it might work:** κ on a planning-domain hyperedge measures
how *bottleneck-like* that action is — high κ = many incident
constraints, low κ = many alternative actions reach the same
post-condition. A* expansion biased toward *lower κ on the
frontier* explores flexible regions first; biased toward *higher
κ on the path to goal* commits to bottleneck actions early.

**Why it might not:** κ is a *structural* property; it has no
direct connection to "distance to goal". A constraint-rich
region might be one that's *closer* to the goal (because the
goal constrains many incident edges), or one that's *further*
(because the planner is stuck in a corner). The sign of the
correlation has to be measured empirically.

### H2 — Δ_1 spectrum identifies degenerate plans before they're enumerated

**Claim:** a partial plan whose induced state-action subgraph
has a non-negligible Δ_1 cycle component (the "rotational" part
of the Hodge decomposition) is, with high probability, a plan
fragment that *cycles without making goal-state progress*.
Pruning frontier nodes whose Δ_1 ratio exceeds a threshold
reduces planning time without missing optimal plans.

**Why it might work:** the Hodge decomposition decomposes any
1-form (edge labelling) into gradient (potential flow toward a
goal-state value function) + cycle (rotational, no net
progress) + harmonic (boundary-respecting). A plan that
maximises gradient-component drift is exactly a plan that
makes net progress; a plan with high cycle-component is locally
oscillating.

**Why it might not:** in practice many useful sub-plans *do*
cycle locally (a robot rotating around a manipulation point
before grasping). Distinguishing "productive local cycle"
from "stuck cycle" requires a richer signal than just Δ_1
norm.

### H3 — Signed-cycle pool prior, trained on solved plans, gives a learned plan-completion bias that outperforms a vanilla GNN at iso-parameter

**Claim:** train HSiKAN (the signed-cycle architecture, already
shipped in `signedkan_wip`) on a corpus of (planning problem,
solved plan) pairs. The σ-product score of a candidate
action-sequence, under the trained cycle weights, predicts
"this action is in a successful plan from this state" better
than a vanilla GraphSAGE/GCN baseline at the same parameter
count.

**Why it might work:** the HSiKAN paper proved σ-cycle priors
have a Kolmogorov-Arnold representation property (project memory
`signedkan_ka_rank`). Planning has natural cycle structure
(commute-relations on actions: "do A then B" is sometimes
equivalent to "do B then A"; the σ-product encodes this as a
local symmetry).

**Why it might not:** plan diversity may mean each successful
plan has a different cycle signature, washing out the prior.
The signal-to-noise ratio of σ-cycle structure in
plan-corpus data is an empirical question.

## 4. SpherePlanner architecture

Concrete enough to start building. Each box maps to existing
machinery or a small new module.

```
┌─────────────────────────────────────────────────────────────┐
│  Input: domain (HymeKo IR) + problem (initial + goal state) │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  State-action hypergraph construction                       │
│   • vertices = world-state propositions ∪ action atoms       │
│   • hyperedges = preconditions ∪ effects ∪ mutex constraints │
│   • signs = + (asserted) / - (denied / removed)              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Per-edge structural features (computed once)               │
│   • Forman κ (sparse, vectorised; already in tree)           │
│   • Hodge Δ_0, Δ_1 decomp on the static action graph         │
│   • Top-k signed cycles per state (Rust, parallel)           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Search loop  (A*-shaped, with hypergraph-aware heuristic)  │
│                                                              │
│   priority(frontier_node) =                                  │
│     α · g(node)               (path cost so far)             │
│   + β · h_κ(node)             (H1 — Forman heuristic)        │
│   + γ · h_Δ(node)             (H2 — Δ_1 cycle penalty)       │
│   + δ · h_HSiKAN(node)        (H3 — learned signed-cycle)    │
│                                                              │
│   On expansion: enumerate applicable hyperedges by σ-product │
│   consistency with the current state's sign labelling.       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: plan = sequence of hyperedges from initial to goal │
│   • emit as BT.CPP / PDDL / ROS 2 action sequence            │
│   • OR run directly in a HymeKo-native execution harness     │
└─────────────────────────────────────────────────────────────┘
```

The four heuristic weights (α, β, γ, δ) are learnable from a
small calibration set per domain, à la the Optuna sweeps in
the existing HSiKAN benchmarks.

## 5. Smallest viable experiment

**Domain:** *dual-agent blocksworld*. Two robots, K stacks of
N blocks each, action set = {pick, place, hand_to_other_robot,
wait}. Goal = a specified final stack configuration. The
`hand_to_other_robot` action is the dual-arm coordination
primitive that classical PDDL handles awkwardly and that
the HymeKo hypergraph encodes natively.

**Domain size sweep:** {(2, 3), (2, 5), (2, 8), (3, 5),
(3, 8)} for (n_agents, n_blocks_per_stack). Plan length scales
as O(N) so even the largest is tractable for ground truth.

**Baselines:**
1. FastDownward (LM-cut heuristic) — the established A*+PDDL
   reference.
2. Dijkstra (uniform heuristic) — guarantees optimality but
   no search guidance.
3. Vanilla GNN heuristic — train a GraphSAGE on solved-plan
   demonstrations; predict node priorities for A*.

**SpherePlanner variants:**
1. κ-only (H1): α = 1, β = w, γ = 0, δ = 0.
2. κ + Δ_1 (H1 + H2): adds γ.
3. Full (H1 + H2 + H3): adds δ from a trained HSiKAN.

**Metrics:**

* **Plan length** (vs ground-truth optimal).
* **Wall-clock planning time** (per problem instance).
* **Coordination correctness** — does the planner find the
  hand-to-other-robot action when it's optimal, or does it
  prefer single-arm plans? This is the headline metric for
  the multi-agent claim.
* **Plan validity** — % of planned actions whose preconditions
  hold at execution time.

**Pass criteria for the proposal:**

* H1 passes if κ-only beats FastDownward LM-cut on
  *coordination-rate* by ≥ 0.10 (absolute) at p < 0.05 across
  the 5 (n_agents, n_blocks) cells. If H1 fails, the κ-as-
  heuristic claim is falsified and the proposal pivots.
* H2 passes if κ+Δ_1 reduces wall time by ≥ 25% over κ-only
  without degrading plan length. If H2 fails, Δ_1 pruning is
  a wash; the full variant just inherits κ.
* H3 passes if the full HSiKAN-augmented variant beats the
  vanilla-GNN baseline by ≥ 0.10 on coordination-rate at
  iso-parameter. If H3 fails, the σ-cycle bias doesn't
  transfer to planning (which would mirror the vision-corner
  negative — *informative either way*).

**Cost:** 1 week for plumbing + baseline reproduction, 1 week
for SpherePlanner variants, 1 week for ablation + writeup.
3 weeks total to settle all three hypotheses on the toy domain.

## 6. Existing-codebase reuse map

For each capability the proposal needs, the existing file /
function that supplies it:

| SpherePlanner component                                   | Reuse                                                                                                                | New code                                                  |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| State-action hypergraph IR                                 | `hymeko_core::Ir`, `meta_task.hymeko`                                                                                 | Domain-specific preconditions / effects (extend `meta_task`) |
| Forman κ computation                                       | `signedkan_wip/src/hymeko_gomb/soma/vision/forman.py`, vectorised Rust                                                | Adapter from planning-IR to (edges, n_nodes)              |
| Hodge Δ_0, Δ_1 spectral analysis                            | `signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py`                                                                  | Δ_1 cycle-component norm extractor                       |
| Top-k signed cycles per state                              | `hymeko_py.enumerate_top_k_cycles_rs`                                                                                 | Per-state caching layer                                   |
| σ-product evaluation of plan fragments                     | HSiKAN forward (`signedkan_wip.src.signedkan_layer`)                                                                  | Plan-fragment → cycle-pool adapter                        |
| Learned heuristic training (H3)                            | The 2026-05-16 ricci-scale sweep harness as a Sweep template                                                          | Plan-corpus dataset loader                                |
| Branch-and-bound search                                    | `BoundedScorer` trait in `hymeko_graph`                                                                               | A*-frontier priority queue                                |
| Multi-format emit of the resulting plan                    | `transforms/` infrastructure                                                                                          | `transforms/plan_sequence/` template (one of: BT, PDDL, ROS) |

Every row's first column is *already shipped and tested*. The
"New code" column is what an actual implementation sprint
would add — small adapters and the search-loop glue.

## 7. Risks & open questions

* **Domain transferability.** Will κ + Δ_1 + σ-cycle priors
  generalise from blocksworld to tabletop manipulation / TAMP /
  long-horizon planning? Open. The toy domain is a feasibility
  test, not a domain-general claim.

* **Scaling.** Top-k cycle enumeration is `O(N^k)` in the worst
  case. For planning problems with K = 5+ and N = 100s of state
  predicates, the enumerator's `max_cycles` cap (already plumbed)
  is the right scaling lever, but the *plan quality* under the
  cap is an open question.

* **Learning data.** H3 needs a corpus of solved (problem, plan)
  pairs. For blocksworld these are cheap (FastDownward generates
  them). For real-world TAMP corpora are scarce and noisy.

* **Comparison fairness.** "Iso-parameter" against a vanilla GNN
  baseline only matters if both use the same training set and
  the same supervision. The 2026-05-13 +ricci-mod vs
  boxes+circles paired Δ pattern (project memory
  `n_seed_before_paper_promotion`) is the protocol model.

* **Negative-result acceptance.** If H1 falsifies, the κ-as-
  heuristic claim falls; SpherePlanner reverts to a fancier-A*
  with no novel claim. That's *fine* as a science outcome —
  the negative result is a real contribution (parallel to the
  vision-corner negative). Don't bury it.

## 8. 6-week research arc (if hypotheses survive the 3-week toy)

Weeks 1-3: H1 / H2 / H3 on dual-agent blocksworld (§5).

Week 4: lift to a continuous-state TAMP variant (a single
manipulation domain — pick-and-place with collision constraints).
Use PDDLStream as the baseline.

Week 5: integrate SpherePlanner with the dual-FANUC Gazebo cell
shipped tonight. Plan a dual-arm handover with collision
avoidance + force-control phases. Demo + recording.

Week 6: writeup + (if positive) a workshop paper to ICRA / IROS
TAMP track.

## 9. Why this matters now

Three reasons it's the right moment to commit to this direction:

1. **The codebase has accumulated exactly the right machinery.**
   For 6 months the project has been building signed-hypergraph
   tools for link prediction. Planning is the natural
   second-domain that maps these tools cleanly onto a problem
   they were always shaped for.

2. **The dual-FANUC scenario and the meta_task vocabulary
   shipped tonight provide the substrate.** A blocksworld
   variant is a 1-day extension of `meta_task.hymeko`. A
   dual-arm planning problem in HymeKo is a 1-week extension.
   We are *not* starting from a blank substrate.

3. **The vision-corner negative bounds the σ-cycle inductive
   bias precisely.** "Works where data is natively signed" is
   exactly planning. The proposal does not have to argue from
   first principles for the relevance of the existing machinery
   — the negative result on vision *is* the indirect evidence.

## 10. Acceptance for this design note

* [x] Three falsifiable hypotheses each with explicit pass/fail
      criteria (§3).
* [x] Concrete architecture diagram (§4).
* [x] Smallest viable experiment with baselines, metrics, and
      cost estimate (§5).
* [x] Existing-codebase reuse map (§6) — every component sources
      to a shipped artefact.
* [x] Honest risk enumeration (§7) including the negative-result
      acceptance posture.
* [ ] **Approval to commit** — the user picks this up in the
      morning, decides whether to proceed past the design note
      to the toy-domain plumbing.

Sized roughly: 3 weeks to settle the toy claims; 6 weeks to a
demo + writeup if the claims survive.

## Bottom line

**Yes — hypergraph-native planning is the natural next research
direction**, not because hypergraphs are fashionable but because
this specific codebase has the machinery, the vision-corner
negative bounds the signed-cycle bias to exactly the right
domain, and the existing meta_task + dual-FANUC scaffolding
makes the toy experiment a 1-week setup rather than a quarter-
long infrastructure project.

The deliverable for tonight is *this proposal*. The deliverable
for the next 3 weeks would be the falsifiable answers to H1 / H2
/ H3 on dual-agent blocksworld.
