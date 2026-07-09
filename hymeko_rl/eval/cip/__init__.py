"""CIP intervention harnesses — turn a *proposed* DirectLiNGAM causal edge into intervention-backed evidence.

Phase 2 (``hymeko_rl/eval/causal/``) discovers a candidate DAG and stamps it PROPOSED. This package runs the
controlled interventions that would upgrade a proposal to SUPPORTED — the ``propose, then isolate`` rule.

* :mod:`contact_reward_ablation` — Stage A (cached-rollout reward recomputation, no training): recompute each
  rollout's reward under reward variants that remove/downweight the contact terms, rerun the stratified
  CIP/DirectLiNGAM pipeline, and report whether the ``contact_score → total_reward`` edge and the
  reward↔monitor disagreement collapse. Stage B (a bounded training smoke) is documented but NOT run here.
"""
from __future__ import annotations

from .contact_reward_ablation import (
    CONTACT_TERMS,
    DELIVERY_TERMS,
    AblationOutcome,
    RewardVariant,
    build_variants,
    directed_edge_weight,
    recompute_variant_reward,
    reward_delivery_alignment,
)
from .metaworld_cip import (
    TEMPLATES,
    TaskTemplate,
    run_metaworld_cip,
    run_metaworld_cip_real,
    run_metaworld_multiseed,
)
from .metaworld_generic_cip import GENERIC_TASKS, run_generic_cip, run_generic_sweep
from .metaworld_gifs import make_coffee_push_gifs, render_coffee_push_gif

__all__ = [
    "CONTACT_TERMS",
    "DELIVERY_TERMS",
    "RewardVariant",
    "AblationOutcome",
    "build_variants",
    "recompute_variant_reward",
    "directed_edge_weight",
    "reward_delivery_alignment",
    "TEMPLATES",
    "TaskTemplate",
    "run_metaworld_cip",
    "run_metaworld_cip_real",
    "run_metaworld_multiseed",
    "render_coffee_push_gif",
    "make_coffee_push_gifs",
    "GENERIC_TASKS",
    "run_generic_cip",
    "run_generic_sweep",
]
