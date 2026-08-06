"""spec_bench — does a HyMeKo/HTL formal task-spec describe (or augment) a task better than an LLM prompt?

Stage 0 (this module): the deterministic core — balanced synthetic labelled rollouts, HTL-predicate F1 vs native
success, and the propose→parse-gate→error-loop→faithfulness-select gate over a ``ChatModel`` seam (with a
``ScriptedModel`` so the whole path is testable without a network). Plan:
``docs/plans/2026-07-13-hymeko-vs-llm-task-spec/``.
"""
from hymeko_rl.eval.spec_bench.spec_bench import (
    ChatModel,
    ScriptedModel,
    Rollout,
    evaluate_formula,
    f1_score,
    propose_and_gate,
    score_raw,
    synth_rollouts,
)

# The close-the-loop bridge (ASSIMILATED 2026-07-15): an arbitrated HTL success spec becomes a per-step
# MetaWorld reward (ρ, with sign ρ the monitor verdict) that trains a policy. Canonical surface — import these,
# do not re-implement the reward wrapper or the reward-quality metric. See reports/2026-07-13-spec-reward-*.
from hymeko_rl.eval.spec_bench.spec_reward import (
    ARBITRATED_COFFEE_SPEC,
    RAW_COFFEE_SPEC,
    SpecRewardEnv,
    SpecRewardQuality,
    signals_from_metaworld_info,
    spec_reward_separation,
)

__all__ = [
    "ChatModel", "ScriptedModel", "Rollout", "evaluate_formula", "f1_score",
    "propose_and_gate", "score_raw", "synth_rollouts",
    # spec→reward bridge (close-the-loop)
    "SpecRewardEnv", "spec_reward_separation", "SpecRewardQuality", "signals_from_metaworld_info",
    "ARBITRATED_COFFEE_SPEC", "RAW_COFFEE_SPEC",
]
