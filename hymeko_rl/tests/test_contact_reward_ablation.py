"""Tests for the Stage-A contact-reward ablation — the pure reward-recomputation + edge-change reporting logic.

The mujoco rollout path (`run_stage_a`) is exercised at production scale by the Stage-A run itself; these tests
pin the parts a wrong edit would silently corrupt: the offline reward recomputation (must equal Σ weight·term),
the variant weight vectors (contact terms actually removed/scaled), and the edge/alignment extractors that feed
the decision rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.causal import DirectLiNGAM, sample_linear_sem
from hymeko_rl.eval.cip import (
    CONTACT_TERMS,
    RewardVariant,
    build_variants,
    directed_edge_weight,
    recompute_variant_reward,
    reward_delivery_alignment,
)

# A base spec shaped like the coin reward (the real default), for variant construction tests.
_BASE = (("grasp_approach", 4.0), ("both_contact", 5.0), ("finger_contact", 1.5),
         ("in_zone", 10.0), ("out_of_bounds", 2.0), ("arm_body_collision", 0.5))


def test_recompute_matches_weighted_term_sum() -> None:
    """Offline recomputation == Σ_steps Σ_k weight[k]·term[s,k] for a known matrix and weights."""
    recorded = ["both_contact", "in_zone", "grasp_approach"]
    matrix = np.array([[1.0, 0.0, 0.5], [1.0, 1.0, 0.2], [0.0, 0.0, 0.9]])
    variant = RewardVariant("v", {"both_contact": 5.0, "in_zone": 10.0, "grasp_approach": 4.0})
    # per step: 5*bc + 10*iz + 4*ga → 5*1+10*0+4*0.5=7 ; 5+10+0.8=15.8 ; 0+0+3.6=3.6 → 26.4
    assert recompute_variant_reward(matrix, recorded, variant) == pytest.approx(26.4)


def test_recompute_shape_guard() -> None:
    with pytest.raises(ValueError, match="term_matrix shape"):
        recompute_variant_reward(np.zeros((3, 2)), ["a", "b", "c"], RewardVariant("v", {}))


def test_build_variants_removes_and_scales_contact() -> None:
    variants = {v.name: v for v in build_variants(_BASE, downweight=0.25)}
    assert set(variants) == {"original", "contact_off", "contact_downweighted", "delivery_aligned"}
    # original untouched
    assert variants["original"].weights["both_contact"] == 5.0
    # contact-off zeros BOTH contact terms, keeps the rest
    for t in CONTACT_TERMS:
        assert variants["contact_off"].weights[t] == 0.0
    assert variants["contact_off"].weights["in_zone"] == 10.0
    # downweight scales contact terms only
    assert variants["contact_downweighted"].weights["both_contact"] == pytest.approx(1.25)
    assert variants["contact_downweighted"].weights["finger_contact"] == pytest.approx(0.375)
    assert variants["contact_downweighted"].weights["grasp_approach"] == 4.0
    # delivery-aligned: contact removed, delivery boosted + grasp_deliver added
    assert variants["delivery_aligned"].weights["both_contact"] == 0.0
    assert variants["delivery_aligned"].weights["in_zone"] == pytest.approx(20.0)
    assert variants["delivery_aligned"].weights["grasp_deliver"] == pytest.approx(2.5)


def test_build_variants_downweight_bounds() -> None:
    with pytest.raises(ValueError, match="downweight"):
        build_variants(_BASE, downweight=1.5)


def test_variant_rejects_unknown_term() -> None:
    with pytest.raises(ValueError, match="unknown reward term"):
        RewardVariant("bad", {"not_a_term": 1.0})


def test_weight_vector_alignment_and_zero_fill() -> None:
    v = RewardVariant("v", {"both_contact": 5.0, "in_zone": 10.0})
    vec = v.weight_vector(["in_zone", "both_contact", "grasp_approach"])
    assert list(vec) == [10.0, 5.0, 0.0]        # aligned to order; unlisted term → 0


def test_directed_edge_weight_reads_cause_to_effect() -> None:
    """directed_edge_weight recovers a known signed edge from a LiNGAM fit; missing var → 0."""
    x, _b = sample_linear_sem([(0, 1, 0.8)], 2, 400, seed=1, noise="uniform")
    result = DirectLiNGAM().fit(x, ["contact_score", "total_reward"])
    w = directed_edge_weight(result, "contact_score", "total_reward")
    assert w > 0.4                               # the planted +0.8 edge is recovered as a strong positive weight
    assert directed_edge_weight(result, "contact_score", "absent_var") == 0.0


def test_reward_delivery_alignment_correlation() -> None:
    reward = [1.0, 2.0, 3.0, 4.0]
    assert reward_delivery_alignment(reward, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert reward_delivery_alignment(reward, [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert reward_delivery_alignment(reward, [7.0, 7.0, 7.0, 7.0]) == 0.0     # constant delivery → 0, not NaN


def test_contact_off_collapses_reward_when_reward_is_pure_contact() -> None:
    """End-to-end offline check: if a recorded reward is PURE contact, contact-off recomputes it to ~0."""
    recorded = ["both_contact", "finger_contact", "in_zone"]
    matrix = np.array([[1.0, 2.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])  # in_zone fires once
    variants = {v.name: v for v in build_variants(
        (("both_contact", 5.0), ("finger_contact", 1.5), ("in_zone", 10.0)))}
    orig = recompute_variant_reward(matrix, recorded, variants["original"])
    off = recompute_variant_reward(matrix, recorded, variants["contact_off"])
    assert orig > off                            # removing contact must reduce the total
    assert off == pytest.approx(10.0)            # only the single in_zone step survives (10*1)
