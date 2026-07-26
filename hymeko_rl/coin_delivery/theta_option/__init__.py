"""Coin 6-D torque-θ option adapter — teacher-to-RL in the frozen delivery option space.

Binds the FROZEN physical option (`forward_displacement.rollout_primitive` PUSH→BRAKE→RELEASE, K6 monitor) to the
task-independent `option_rl` engine (proposal centre θ_0 as the Bellman action + fixed bounded search → θ_exec as
provenance + semi-MDP SAC/TD3). Reuses the engine and the frozen physics; adds no external dependency and edits no core
or frozen module. See `docs/plans/2026-07-27-coin-teacher-to-rl/plan.md`.
"""
from hymeko_rl.coin_delivery.theta_option.semantics import (
    DELIVERY_CFG,
    DIM,
    THETA_NAMES,
    ThetaBox,
    ThetaProvenance,
    option_semantics,
    theta_bounds,
)
from hymeko_rl.coin_delivery.theta_option.search import (
    SEARCH_STD,
    ThetaCandidateGenerator,
    ThetaCandidateScorer,
    fixed_search_select,
)

__all__ = [
    "DELIVERY_CFG", "DIM", "THETA_NAMES", "ThetaBox", "ThetaProvenance", "option_semantics", "theta_bounds",
    "SEARCH_STD", "ThetaCandidateGenerator", "ThetaCandidateScorer", "fixed_search_select",
]
