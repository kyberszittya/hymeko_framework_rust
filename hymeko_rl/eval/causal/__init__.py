"""CIP / DirectLiNGAM diagnostic layer — causal experiment-prioritization over runtime-monitor variables.

The causal-discovery consumer that sits *above* the CIP-export bridge
(:mod:`hymeko_rl.eval.task_monitor.cip_export`). It aggregates per-rollout CIP + continuous variables into a
:class:`~hymeko_rl.eval.causal.frame.RolloutFrame`, ranks failures by reward–monitor disagreement
(:class:`~hymeko_rl.eval.causal.prioritize.CipPrioritizer`), and runs
:class:`~hymeko_rl.eval.causal.lingam.DirectLiNGAM` over the *continuous variables only*, stratifying
categoricals. :class:`~hymeko_rl.eval.causal.diagnose.CausalDiagnosis` packages the result.

**Doctrine:** DirectLiNGAM PROPOSES candidate structure; controlled ablations DECIDE. Nothing here is proof of
causality. Only continuous rollout variables enter the linear non-Gaussian model; categoricals stratify, never
mix in. Do not run this layer unless explicitly requested (roadmap gate).
"""
from __future__ import annotations

from .diagnose import CausalDiagnosis, DiagnosisReport
from .frame import CIP_VAR_KINDS, RolloutFrame, VarKind
from .hymeko_emit import (
    AcyclicityResult,
    CausalHypergraph,
    CrossViewReport,
    Mechanism,
    StarProjection,
    cross_view_verify,
    to_hymeko_source,
)
from .lingam import (
    DirectLiNGAM,
    IndependenceMeasure,
    LingamConfig,
    LingamResult,
    PairwiseEntropyMeasure,
    sample_linear_sem,
)
from .mechanism_factorization import (
    MechanismFactorization,
    build_pairwise_b,
    factorize_from_proposals,
    score_mechanism_set,
)
from .mechanism_proposal import (
    MechanismProposal,
    propose_common_child,
    propose_monitor_contract,
    propose_reward_terms,
    proposals_to_causal_hypergraph,
)
from .prioritize import CipPrioritizer, PrioritizedFinding

__all__ = [
    "DirectLiNGAM",
    "IndependenceMeasure",
    "LingamConfig",
    "LingamResult",
    "PairwiseEntropyMeasure",
    "sample_linear_sem",
    "RolloutFrame",
    "VarKind",
    "CIP_VAR_KINDS",
    "CipPrioritizer",
    "PrioritizedFinding",
    "CausalDiagnosis",
    "DiagnosisReport",
    "CausalHypergraph",
    "CrossViewReport",
    "Mechanism",
    "StarProjection",
    "AcyclicityResult",
    "to_hymeko_source",
    "cross_view_verify",
    "MechanismProposal",
    "propose_common_child",
    "propose_reward_terms",
    "propose_monitor_contract",
    "proposals_to_causal_hypergraph",
    "MechanismFactorization",
    "build_pairwise_b",
    "factorize_from_proposals",
    "score_mechanism_set",
]
