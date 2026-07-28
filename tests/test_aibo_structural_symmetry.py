"""The HyMeKo structure carries the crab's left-right symmetry as a VERIFIED automorphism.

Locks the "use the HyMeKo structure as the equivariance signal" result: the reflection derived from the
vertex labels is an involution, an exact signed-incidence automorphism of the generated hypergraph, and
its induced action on the abduction (actuator) vertices is exactly the ``fl<->fr, bl<->br`` swap that the
hand-coded ``mirror_act`` used — so the mirror is READ from the structure, not guessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv
from scenarios.aibo.structural_symmetry import mirror_label, structural_reflection


def _full_hg_env() -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="hypergraph"), seed=0)


def test_mirror_label_involution_and_pairs() -> None:
    assert mirror_label("hip_bracket_fl") == "hip_bracket_fr"
    assert mirror_label("hip_bracket_fr") == "hip_bracket_fl"
    assert mirror_label("thigh_bl") == "thigh_br"
    assert mirror_label("eye_l") == "eye_r"
    assert mirror_label("torso") == "torso"                  # midline -> itself
    for lab in ("paw_fl", "ear_r", "shin_bl", "neck", "tail_tip"):
        assert mirror_label(mirror_label(lab)) == lab        # involution


def test_structural_reflection_is_exact_signed_automorphism() -> None:
    ref = structural_reflection(_full_hg_env().hg)
    sigma = ref.vertex_perm
    assert np.array_equal(sigma[sigma], np.arange(len(sigma)))            # involution
    # midline parts are fixed; the four legs + eyes/ears are swapped
    assert "torso" in ref.fixed and "tail" in ref.fixed
    pairs = {frozenset(p) for p in ref.swapped_pairs}
    assert frozenset(("hip_bracket_fl", "hip_bracket_fr")) in pairs
    assert frozenset(("eye_l", "eye_r")) in pairs


def test_action_perm_matches_hand_coded_mirror() -> None:
    env = _full_hg_env()
    ref = structural_reflection(env.hg)
    perm = ref.action_perm(env._abd_vtx)                     # abduction vertices [fl,fr,bl,br]
    assert np.array_equal(perm, np.array([1, 0, 3, 2]))      # == the permutation in mirror_act (-a[[1,0,3,2]])


def test_reflection_raises_when_structure_not_symmetric() -> None:
    # a hand-built hypergraph whose labels are not left-right closed must be rejected (contract)
    class _HG:
        vertex_labels = ("torso", "leg_fl")                  # fl present, fr missing
        edges = np.array([[0, 1]], dtype=np.int64)
        signs = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="absent from the structure|not L-R closed"):
        structural_reflection(_HG())
