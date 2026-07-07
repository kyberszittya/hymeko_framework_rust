# v2 RL-readiness gate — HyMeKo tensor contract + single-source verification

**Date:** 2026-07-07 · Gate BEFORE any RL. No training in this document — it freezes the deployable imitation
baseline, documents the HyMeKo-generated tensor contract, and verifies single-source-of-truth. The gated RL
re-entry (D) is specified but NOT yet run.

## A. Frozen deployable imitation baseline

`experiments/v2_dagger/FROZEN_selected/` — the validation-24-selected MLP+DAgger checkpoints (mean ft_dom 0.452):
`mlp_s0_selected_d2.pt` (0.438), `mlp_s1_selected_d3.pt` (0.625, best single), `mlp_s2_selected_bc0.pt` (0.292).
Diagnostic best-ckpt mean 0.514; scripted v2b ceiling 0.792. Reward `galambos_task_deliver_v2b.hymeko` (frozen);
scene v2 graded `contact_legality`.

## B. HyMeKo-generated tensor contract

| tensor | shape / dtype | code | HyMeKo source of truth |
|---|---|---|---|
| **observation** | `(n_vertices, 8)` f32 — per-vertex `[qpos, qvel, x, y, coin−x, coin−y, coin→zone−x, coin→zone−y]` | `PlanarGraspEnv.node_features()` (`_FEAT=8`) | **robot** `galambos_planar.hymeko` → `emit_arm_mjcf` → `HypergraphState.from_mjcf` (n_vertices) |
| **privileged CTDE state** | `(5,)` f32 — `[left_contact, right_contact, onehot(reach/carry/in_zone)]` | `PlanarGraspEnv.privileged_state()` (`privileged_dim=5`) | env physics (contact forces + `_ever_grasped`/`in_zone` latch); **NOT reconstructable from obs** (replay.py:35) |
| **reward-feature** | scalar per term → `Σ wₖ·termₖ` | `RewardSpec.evaluate` reading `PlanarGraspMetrics` | **task** `galambos_task_deliver_v2b.hymeko` → `RewardSpec` (terms) |
| **contact-quality** | `ContactLegalityState` — left/right fingertip, arm_body count/impulse; grade tiers | `classify_contacts(model, data, spec)` | **env** `galambos_env_v2.hymeko` (`@contact`) → `EnvSpec` → `ContactLegalitySpec.from_model` |
| **graph/hypergraph structure** | `edges (E,2)`, `signs (E,)`, incidence | `HypergraphState` (`env.hg`) | **robot** `galambos_planar.hymeko` → `emit_arm_mjcf` → `HypergraphState.from_mjcf` |
| **replay-buffer fields** | `_obs/_next (C, n_vertices, 8)`, `_act (C, action_dim)`, `_rew (C,)`, `_done (C,)`, `_priv/_priv_next (C, 5)` | `ReplayBuffer(capacity, obs_shape, action_dim, priv_dim)` | `obs_shape`←`env.observation_space` (hg); `action_dim`←`env.action_space` (actuators); `priv_dim`←`env.privileged_dim` |

## C. Single-source-of-truth verification — PASS

- **Robot** `galambos_planar.hymeko` is the *sole* source of the observation tensor, the graph-structure tensor,
  the action dimension, and the replay `obs_shape`/`action_dim` — all flow through the one `emit_arm_mjcf` →
  `HypergraphState` chain (no parallel definition).
- **Env** `galambos_env_v2.hymeko` → `EnvSpec` is the *sole* source of the contact-quality contract
  (`ContactLegalitySpec`), the zone/coin geometry, and `success_steps` — the same `EnvSpec` the metrics and the
  contact-quality tensor read.
- **Task** `galambos_task_deliver_v2b.hymeko` → `RewardSpec` is the *sole* source of the reward weighting; its
  terms read the same `PlanarGraspMetrics`/`ContactLegalityState` the eval metrics grade by (reward ≡ metric,
  by construction — the calibration guarantee).
- **Privileged state** is derived from the *same* env physics that produces the metrics and obs; the critic's
  `priv_dim` is taken from `env.privileged_dim` (5), matching `privileged_state()`. The critic and the metrics
  cannot diverge in what "contact/phase" means — one env, one source.

⇒ Reward, observation, metrics, and critic state are all generated from the same HyMeKo source of truth. **Gate
C passes.**

## D. Gated RL re-entry (specified, NOT run) — first conservative CTDE-TD3+BC

- **algo**: CTDE-TD3+BC (off-policy, asymmetric critic), **initialized from the frozen selected MLP+DAgger
  checkpoint** (per seed), **not** from scratch.
- **replay**: seeded with DAgger rollouts (the aggregated expert-labelled + clone-visited transitions), `priv_dim=5`.
- **critic**: centralized, uses the **HyMeKo-generated privileged state tensor** `z(s)` (5-dim).
- **actor**: keeps a **BC/KL regularization term toward the frozen DAgger policy** (anchor, so it cannot drift
  off the fingertip-dominant manifold).
- **budget**: small learning rate, **short** step budget (a smoke first — 1 seed).
- **metrics**: strict contact-quality split every eval (fingertip_dominant headline + assisted/exploit/body_progress).
- **RL acceptance (strict):** RL passes ONLY if it improves **fingertip_dominant_delivery over the deployable
  baseline (0.452)** *without* increasing `body_assisted_delivery`, `body_driven_exploit_delivery`, or
  `body_only_progress_to_target`. **Raw delivery alone does not count. Reward improvement alone does not count.**
- **Oracle pre-gate:** `reward_oracle.certify(galambos_task_deliver_v2b)` = delivers=True (already met); the
  contact-quality terms are metric-aligned (calibration passed).

**Status:** A–C complete (gate passes). D not started. No TD3/SAC/residual/off-policy has been run.
