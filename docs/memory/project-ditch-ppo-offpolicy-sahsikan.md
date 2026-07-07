---
name: project-ditch-ppo-offpolicy-sahsikan
description: "2026-07-01 STRATEGIC PIVOT (Hajdu directive) — ditch PPO indefinitely; the RL line is off-policy (SAC/TD3/DDPG, TD3+BC) + SA-HSiKAN-based agents"
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-07-01, Hajdu directive verbatim: "Ditch PPO indefinitely. Focus on off-policy methods and SA-HSIKAN based agents."

**Why:** On galambos the BC→PPO refine *degraded* the clone — measured `single_hsikan` BC=0.125 → PPO=0.042 (3× worse) even with `bc_coef=1.0` anchor. This is the known offline→online collapse (see [[project-fanuc-offpolicy-collapse]], [[project-fsm-structured-rl]]). PPO also cost 20 min/cell (physics-bound after the vectorization fix). Off-policy (TD3+BC, SAC) does not collapse the warm-start the way on-policy PPO does; SA-HSiKAN (Bᴸ-collapse) is the cheap, structural, deploy-fast backbone ([[project-hsikan-launchbound-alternatives]]).

**How to apply:**
- **Default agent = SA-HSiKAN backbone + off-policy algo.** Do NOT reach for PPO/`train_ppo` for new RL agents or refines. PPO code stays (shared infra, tests, arm_reach `run_ppo`) but is no longer *chosen*.
- `run_galambos_bc` default `algo` flipped `ppo`→`td3` (2026-07-01). The collaborative/3ch coordination needs an **off-policy** re-build (the MultiChannelCTDE was on-policy Gaussian+value; off-policy wants deterministic multi-actor + centralized twin-Q + the MultiTreeChannel).
- Re-establish the best-arch registrations that were PPO-pinned (galambos `bc_ppo`, quadruped `ppo` in [[project-rl-evaluator-simulator-ecosystem]]) under off-policy — those were pre-pivot empirical bests; re-measure, don't just relabel (measured-not-declared).
- Pruning is a **deploy/param-count** lever, not a training-speed lever now (vectorized PPO/off-policy training is physics-bound: 3.3ms physics vs 0.67–1.5ms vectorized net). SA-HSiKAN Bᴸ-collapse is the shipped structural prune. See [[project-unify-hsikan-core]].
- Per-node action head beats mean-pool collapse (galambos hsikan_pernode 0.15 > pooled 0.0, 30k/3seed); generalization across morphologies under test (quadruped SAC). Per-node is the robotics-natural readout (action space factorizes per joint) — [[project-rl-evaluator-simulator-ecosystem]].
