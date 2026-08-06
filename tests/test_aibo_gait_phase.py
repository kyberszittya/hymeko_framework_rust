"""Phase A — configurable base-gait phase + the crab-symmetry mechanism it exposes.

The omni crab is one-sided because the abduction residual sits over a DIAGONAL trot
``(0,pi,pi,0)`` whose two body sides are in different gait phases at every instant (an instantaneous
left-right asymmetry). ``GAIT_PHASES`` adds a ``bound`` pattern ``(0,0,pi,pi)`` that is instantaneously
left-right symmetric (fl==fr, bl==br). These tests lock: the named patterns; the bound instantaneous
symmetry vs the diagonal asymmetry; backward-compatible ``diag`` default; and the MEASURED mechanism —
over ``bound`` the single-side abduction becomes mirror-antisymmetric (left and right push OPPOSITE
lateral directions), while over ``diag`` they push the SAME direction (the root of the one-sidedness).
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.aibo.locomotion_gait import GAIT_PHASES, SteeredTrotGait, _DIAG_PHASE
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv

_PI = float(np.pi)


def test_gait_phases_named_patterns() -> None:
    assert GAIT_PHASES["diag"] == (0.0, _PI, _PI, 0.0)
    assert GAIT_PHASES["bound"] == (0.0, 0.0, _PI, _PI)
    assert GAIT_PHASES["pace"] == (0.0, _PI, 0.0, _PI)
    assert GAIT_PHASES["pronk"] == (0.0, 0.0, 0.0, 0.0)


def test_bound_is_instantaneously_left_right_symmetric() -> None:
    fl, fr, bl, br = GAIT_PHASES["bound"]           # leg order fl,fr,bl,br
    assert fl == fr and bl == br                    # left == right at every instant -> mirror symmetric
    dfl, dfr, dbl, dbr = GAIT_PHASES["diag"]
    assert dfl != dfr                               # the diagonal trot is instantaneously ASYMMETRIC


def test_diag_is_backward_compatible_default() -> None:
    assert ResidualTrotConfig().gait_phase == "diag"              # default unchanged
    assert SteeredTrotGait().phase == _DIAG_PHASE                 # gait default unchanged
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph"), seed=0)
    assert env._phase_pat == GAIT_PHASES["diag"]                  # resolves to the prior diagonal pattern


def test_invalid_gait_phase_raises() -> None:
    with pytest.raises(ValueError, match="gait_phase"):
        ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", gait_phase="waltz"), seed=0)


def _lateral_dy(gp: str, patt: list[float], horizon: int = 600) -> float:
    """Torso lateral displacement from a constant abduction pattern over the ``gp`` scaffold."""
    e = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph", gait_phase=gp), seed=0)
    e.reset()
    e._env.goal = np.array([10.0, 0.0], np.float32)               # far straight goal -> pursuit ~ +x
    r = e._env
    y0 = float(r.data.xpos[r.torso, 1])
    a = np.array(patt, np.float32)
    for _ in range(horizon):
        e._apply(a)
    return float(r.data.xpos[r.torso, 1]) - y0


def test_bound_restores_mirror_antisymmetry_of_abduction() -> None:
    # THE mechanism claim. Over diag, left & right single-side abduction push the SAME lateral way
    # (product > 0) -> no mirror symmetry to exploit. Over bound they push OPPOSITE ways (product < 0)
    # -> the mirror antisymmetry is restored. This is why a symmetric scaffold is the real lever.
    dl_diag, dr_diag = _lateral_dy("diag", [1, 0, 1, 0]), _lateral_dy("diag", [0, 1, 0, 1])
    dl_bound, dr_bound = _lateral_dy("bound", [1, 0, 1, 0]), _lateral_dy("bound", [0, 1, 0, 1])
    assert dl_diag * dr_diag > 0                                  # diag: same sign (asymmetric)
    assert dl_bound * dr_bound < 0                                # bound: opposite signs (mirror-restored)
