"""HyMeKo framework option-RL engine — task-independent semi-MDP option RL with a proposal + fixed-budget-search wrapper.

The engine owns the *mechanism* (options, γ^τ semi-MDP, proposal-center-as-Bellman-action + selected-candidate provenance,
SAC/TD3, checkpoint-wise paired evaluation, hierarchical skill routing); concrete tasks (coin push-delivery is the first)
plug in via the Protocols. See `docs/plans/2026-07-24-option-rl-framework-migration/plan.md`.
"""
from hymeko_rl.option_rl.agents import (
    DetActor,
    GaussActor,
    QNet,
    SemiMDPConfig,
    make_actor,
    train_semi_mdp,
)
from hymeko_rl.option_rl.core import (
    HandoffCertificate,
    OptionEnd,
    OptionEnv,
    OptionReplayBuffer,
    OptionTransition,
    smdp_target,
)
from hymeko_rl.option_rl.eval import (
    across_seed_summary,
    bootstrap_ci,
    paired_delta,
    paired_final_score,
    preregistered_select,
)
from hymeko_rl.option_rl.hierarchy import SkillRoute
from hymeko_rl.option_rl.proposal import (
    CandidateGenerator,
    CandidateScorer,
    FixedBudgetSearch,
    ProposalPolicy,
    SelectedActionProvenance,
)

__all__ = [
    "OptionTransition", "OptionReplayBuffer", "smdp_target", "OptionEnv", "OptionEnd", "HandoffCertificate",
    "ProposalPolicy", "CandidateGenerator", "CandidateScorer", "FixedBudgetSearch", "SelectedActionProvenance",
    "QNet", "GaussActor", "DetActor", "SemiMDPConfig", "make_actor", "train_semi_mdp",
    "bootstrap_ci", "paired_delta", "preregistered_select", "paired_final_score", "across_seed_summary", "SkillRoute",
]
