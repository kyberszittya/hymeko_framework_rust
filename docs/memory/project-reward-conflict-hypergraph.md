---
name: project-reward-conflict-hypergraph
description: "Galambos grasping SOLVED by SIMPLIFYING the reward (0.10→0.615); reward = signed hypergraph, conflict MAGNITUDE predicts grasping (not frustration-cycle); causal test confirmed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-28: the week-long Galambos "knocks-not-grasps" wall **broke** — by SIMPLIFYING the reward, not shaping it.
Reducing `galambos_task.hymeko` from ~11–14 terms to **four** (approach · both·5 · zone · oob) raised
grasp-fraction of deliveries **0.10 → 0.615** (MLP, SAC, 25k, difficulty 0.3). Reward-shaping (grasp-gate,
settle/still) had all failed; the fix was the opposite move.

**Mechanism (measured, not hunch).** Treat the reward as a **signed hypergraph** over its terms: edge sign =
sign(corr(Δf_i, Δf_j)) along a *grasping* trajectory (random-exploration probe is UNDERPOWERED — contact terms
inert; must use a trained grasping policy). The complex reward is a **frustrated** signed graph (Harary
frustration index 2, 8 negative edges); the 4-term reward is **balanced** (0). Sharpest conflict:
`arm_motion × joint_velocity = -0.24` (anti-stall vs smoothness, opposed by definition) and
`grasp_approach × arm_motion = -0.20`.

**Causal test.** Re-added exactly the conflicting pair (arm_motion + joint_velocity) to the 4-term reward →
grasp **0.615 → 0.333**. But frustration index stayed **0** (two conflict edges form a *path*, not an odd cycle →
2-colourable) even though pairwise conflict got *stronger* (`grasp_approach×arm_motion = -0.45`). So the **refined
law: CONFLICT MAGNITUDE (total negative-edge weight) predicts grasping — NOT the frustration index**, which is the
stricter cycle-holonomy special case. The Berge-cycle framing (Hajdu) named the right OBJECT (signed term graph);
the predictive invariant on it is the continuous conflict weight.

Artifacts: PDF `reports/2026-06-28-reward-conflict-hypergraph.pdf`, report
`reports/2026-06-28-overnight-collaborative-sweep.md`, probe `scratchpad/reward_frustration.py`, data
`reports/overnight/reward_frustration*.json`. Canonical `galambos_task.hymeko` = the 4-term winner (tests updated).
Matters MORE for design-fragile HSiKAN. NEXT (Hajdu-requested): **principal-axes study** — eigendecomposition of
the term co-movement matrix (signed-Laplacian smallest eigenvalue = algebraic frustration). Per the
[[feedback-reward-definition-in-hymeko]] rule the canonical reward stays in the .hymeko; sweeps use in-memory
RewardSpec variants (as `exp_reward_weight_sweep.py` does). Ties to [[project-hymeko-aggregation-semantics]]
(reward = hub-conv with algebra), [[project-gauge-holonomy-signed-hsikan]] (signed-graph holonomy),
[[project-galambos-reward-shaping]] and [[project-galambos-hsikan-tie-rootcause]].
