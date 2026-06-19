"""hymeko_rl — a learned control head on the tensorised HyMeKo hypergraph.

The unification (the reason this package exists): a single HyMeKo hypergraph
substrate carries **four roles**, processed by the *same* HSiKAN/Gömb machinery —

  1. the **geometric description**  — the robot's kinematic hypergraph (the scene);
  2. the **actor**                  — a policy head on a HSiKAN/Gömb backbone reading
                                       that hypergraph;
  3. the **critic**                 — a value head on the *same* shared backbone;
  4. the **agent description**      — the MDP (obs/action/reward) as a `.hymeko`
                                       profile, compiled through the same IR →
                                       star-expansion → tensor bridge.

So ``ActorCritic`` is literally one backbone with two heads, and the architecture
ablation (HSiKAN-on-hypergraph vs an MLP baseline) is a *backbone swap* under one
shared PPO — fix the algorithm, ablate the architecture.

Plan: ``docs/plans/2026-06-18-mujoco-rl-grasping/`` (Kato-collab POC).
Dependencies ``mujoco`` + ``gymnasium`` added under approval token
``APPROVED-CORE-EDIT: mujoco-gymnasium-robot-rl`` (2026-06-18).
"""
from __future__ import annotations

from hymeko_rl.agent import AgentSpec
from hymeko_rl.policy import ActorCritic, POLICY_KINDS, build_policy

__all__ = ["ActorCritic", "build_policy", "POLICY_KINDS", "AgentSpec"]
