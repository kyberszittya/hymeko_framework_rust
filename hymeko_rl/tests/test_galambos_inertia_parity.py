"""GALAMBOS_PLANAR_INERTIA_PARITY (2026-07-22): the v3 robot's ``galambos_inertia`` contract, applied through the
adapter, reproduces the golden ``make_planar_arms_mjcf`` COMPILED inertial (mass + diagonal inertia + COM) on the
structural links body-for-body, with the fingertip helper bodies near-massless. This is the STATIC parity gate.

KNOWN LIMITATION (see reports/2026-07-22-coin-inertia-repair-blocked.md): static parity holds, but the naive
override is DYNAMICALLY unstable (a welded fingertip CONTACT body cannot be massless without a NaN contact impulse;
the E0 clamp compounds it), so v3 is NOT yet the canonical robot — the golden-structure fix is pending. This test
guards the static-parity progress so it is not lost.
"""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.env.planar_grasp_env import emit_galambos_v2_mjcf, make_planar_arms_mjcf

_V3 = "data/robotics/galambos_planar_v3.hymeko"
_LINKS = ("base_left", "link1_left", "link2_left", "base_right", "link1_right", "link2_right")


def _models():
    g = mujoco.MjModel.from_xml_string(make_planar_arms_mjcf())
    v = mujoco.MjModel.from_xml_string(emit_galambos_v2_mjcf(_V3))   # v3 spec → golden inertial contract applied
    return g, v


def test_total_arm_mass_matches_golden():
    g, v = _models()
    # near-exact: the fingertip stability floor (2×0.001) is the only allowed excess.
    assert abs(float(g.body_mass.sum()) - float(v.body_mass.sum())) < 3e-3


def test_structural_link_mass_inertia_com_match_golden():
    g, v = _models()
    worst_m = worst_i = worst_p = 0.0
    for nm in _LINKS:
        bg = mujoco.mj_name2id(g, mujoco.mjtObj.mjOBJ_BODY, nm)
        bv = mujoco.mj_name2id(v, mujoco.mjtObj.mjOBJ_BODY, nm)
        worst_m = max(worst_m, abs(float(g.body_mass[bg]) - float(v.body_mass[bv])))
        worst_i = max(worst_i, float(np.max(np.abs(g.body_inertia[bg] - v.body_inertia[bv]))))
        worst_p = max(worst_p, float(np.max(np.abs(g.body_ipos[bg] - v.body_ipos[bv]))))
    assert worst_m < 1e-6, f"structural-link mass parity {worst_m}"
    assert worst_i < 1e-9, f"structural-link inertia parity {worst_i}"
    assert worst_p < 1e-6, f"structural-link COM parity {worst_p}"


def test_fingertip_helper_bodies_are_near_massless():
    _g, v = _models()
    for s in ("left", "right"):
        b = mujoco.mj_name2id(v, mujoco.mjtObj.mjOBJ_BODY, f"fingertip_{s}")
        assert float(v.body_mass[b]) <= 1e-3, "fingertip helper must be at/below the stability floor"


def test_control_contract_still_matches_golden_on_v3():
    # the v3 robot keeps the golden control values (kp / damping / ranges).
    g, v = _models()
    assert {round(float(g.actuator_gainprm[a][0]), 1) for a in range(g.nu)} == \
           {round(float(v.actuator_gainprm[a][0]), 1) for a in range(v.nu)} == {40.0}
    assert {round(float(g.dof_damping[j]), 2) for j in range(g.nv)} == \
           {round(float(v.dof_damping[j]), 2) for j in range(v.nv)} == {1.5}
