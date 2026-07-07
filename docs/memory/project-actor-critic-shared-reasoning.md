---
name: project-actor-critic-shared-reasoning
description: "Idea: actor+critic share not just reward but HSiKAN STRUCTURAL REASONING (walks/cycles/jumps over the signed robot hypergraph) — a shared reasoning trunk, dual heads"
metadata:
  node_type: memory
  type: project
  originSessionId: 3060e292-680f-4645-82c1-156ce78e537c
---

**Idea (user, 2026-06-22) — write-down, not yet implemented.** Standard actor–critic couples actor and
critic only through the **scalar reward / advantage** (the critic estimates value, the actor steps on the
advantage); at best they share a flat **feature** backbone. **With HSiKAN they can share *reasoning*, not
just reward** — a common signed-hypergraph message-passing **trunk** whose structural computation feeds both
the policy head and the value head.

**What gets shared = the structural reasoning over the kinematic hypergraph**, in the user's three terms:
- **Walks** — neighbourhood message-passing walks (k-hop aggregation along the chain): local relational
  reasoning, "how this joint's neighbours move."
- **Cycle connections** — closed loops in the (signed) hypergraph: kinematic loops + signed triads / Berge
  cycles / balance reasoning. Both actor and critic see which *cycles* are active/strained — see
  [[project-hsikan-geometric-attention-berge]] (signed-triad attention head + Berge cycles, pieces exist).
- **Jumps** — long-range / skip ("jumping-knowledge") connections: distant joints coupled directly, not just
  through the chain. Lets the value + policy reason about end-to-end coupling (base ↔ effector) in one hop.

**Why it could matter.** The critic's value estimate and the actor's action would rest on the *same*
structural understanding of the robot — richer than a shared flat feature vector and far richer than a
scalar advantage. Plausible payoffs: (1) **credit assignment** becomes structural — *which walk/cycle/jump*
drives the value, so the advantage carries relational information; (2) **sample efficiency** on
redundant/branched morphologies where the structure is load-bearing (the place HSiKAN should beat a matched
MLP, which it hasn't on serial chains — cart-pole/coin ties); (3) **auditability** — the shared reasoning is
inspectable (which cycles/paths are hot), on the "structurally accountable AI" line.

**Open design questions (for when it's built):**
- Shared trunk + dual heads (cheap, standard) vs partially-shared (actor/critic want different things — value
  wants smoothness, policy wants sharpness; full sharing can hurt). A stop-gradient or a shared-trunk +
  divergent-top-blocks split may be needed.
- Concrete HSiKAN layers for "cycle" (cycle-pooling / signed-cycle aggregation) and "jump" (JK-net-style skip
  over the hypergraph) — the walk part is the existing `_SignedConv` message passing.
- Does sharing reasoning destabilise PPO (the value loss could corrupt the policy's structural features)?
  Mirrors the Phase-2 critic-corruption finding [[project-hymeko-rl-phase2-debug]].

**STRUCTURAL CRITIC — BUILT + TESTED (2026-06-25 overnight).** `hymeko_rl/structural_critic.py`:
`StructuralCritic` (decisions resolved: both cycles+walks; aggregation ablatable {sign_mean, mlp, fir}; pooling
ablatable {attention, mean, sum}; scalar TD) + `enumerate_motifs` + `StructuralCriticConfig` + a
`NodeFeatureBackbone` Protocol. Enumerate-once over the fixed graph via the Rust binding. 11 tests pass (all 9
agg×pool configs + enumerate + validation), mypy --strict + ruff clean. The `hymeko` PyO3 binding was BUILT
(`maturin develop`; module-name="hymeko"; maturin installed via `uv pip`, non-core); `enumerate_top_k_cycles_rs`
(recompute signs) + `enumerate_top_k_walks_rs` (signs returned) verified working. NOT YET WIRED into build_policy
as the critic head + no ablation training run yet (that's the next step: a `structural_critic=True` switch in
ActorCritic + run on galambos_taskgraph where the arm-coin-zone cycle exists). Plan:

**STRUCTURAL CRITIC — 4-artifact PLAN ON DISK (2026-06-25, user-requested):**
`docs/plans/2026-06-25-structural-critic/` (plan.tex/.pdf/.tikz/.mmd, all compile/validate). Concrete first cut of
the "cycle connections" term: replace the scalar critic head with a **StructuralCritic** that decomposes V(s) over
the FIXED hypergraph's top-K signed cycles (enumerate ONCE at build — graph is fixed): per-vertex feats H via
`SignedKANBackbone.node_activations` (verified present) → sign-weighted per-cycle gather e_c=(1/k)Σ s_i H[v_i]
(mirrors `hymeko_graph::spine::SignedCycleFIR.signed_mean`) → per-cycle value v_c=g(e_c) → V=Σ α_c v_c (attention).
Actor unchanged; signed_kan core unchanged (downstream value head). With `task_graph=True` the grasp/goal
hyperedges create the arm-coin-zone cycle → routes value along the task path = structural credit for the
long-horizon wall [[project-galambos-hsikan-tie-rootcause]] [[project-fsm-structured-rl]]. v_c = the learned
[[project-fuzzy-defuzzification-heads]] / FuzzySignature object (interpretability).
**GATING PRECONDITION (do NOT build before resolving):** the cycle-enum binding is NOT available in Python —
`hymeko_py` is a Rust crate SOURCE dir (no built .pyd/.so), so `enumerate_top_k_cycles` isn't importable.
Reimplementing the DFS in Python is forbidden (§6.5#2). Decision #5 in the plan: BUILD/expose the hymeko_py PyO3
binding (canonical, a build step → §1 escalation) vs a sanctioned thin Python seam for the tiny (N≤10) fixed graph.
5 open decisions total (structures cycles/walks/both; aggregation; pooling; credit scheme; the binding). NOT built.

**Why (relevance):** directly on the HSiKAN-structure-advantage hunt — a concrete mechanism by which the
hypergraph prior could finally pay off (vs the measured ties). Ties to the shared-agent-reward-model line
[[project-xprofile-instance-refs]], the affective/interaction model [[project-kato-collaboration-grasping]],
and structural-entropy-feedback. **How to apply:** when picking it up, plan-first (§2); prototype on the
branched morphologies (Galambos two-arm / a walker), NOT a serial chain; always run the params-matched
shared-trunk MLP control before crediting the hypergraph reasoning.
