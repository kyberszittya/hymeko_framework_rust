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

__all__ = [
    "ChatModel", "ScriptedModel", "Rollout", "evaluate_formula", "f1_score",
    "propose_and_gate", "score_raw", "synth_rollouts",
]
