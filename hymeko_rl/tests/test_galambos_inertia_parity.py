"""GALAMBOS_PLANAR_INERTIA_PARITY (2026-07-22): the v3 robot's ``galambos_inertia`` contract, applied through the
adapter, reproduces the golden ``make_planar_arms_mjcf`` COMPILED inertial (mass + diagonal inertia + COM) on the
structural links body-for-body, with the fingertip helper bodies near-massless. This is the STATIC parity gate.

RESOLVED (2026-07-22, golden structure): the fingertip contact geom is folded onto link2 and the separate body
removed, so the density-derived mass/inertia match the golden EXACTLY and the frozen chain reproduces 3/9 on canonical
v3 (see reports/2026-07-22-coin-dynamic-parity-and-expert.md). v3 IS the canonical robot.
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
    # EXACT (golden structure: fingertip geom folded onto link2, no separate body, density-derived mass).
    assert abs(float(g.body_mass.sum()) - float(v.body_mass.sum())) < 1e-6


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


def test_fingertip_geom_is_on_link2_and_no_separate_body():
    # golden structure: the fingertip CONTACT geom lives on the massive link2 body (stable contact), and there is NO
    # separate fingertip body (folded, exactly like the golden). The tool site is on link2.
    _g, v = _models()
    for s in ("left", "right"):
        gid = mujoco.mj_name2id(v, mujoco.mjtObj.mjOBJ_GEOM, f"fingertip_{s}")
        assert gid >= 0, f"fingertip_{s} geom missing"
        assert mujoco.mj_id2name(v, mujoco.mjtObj.mjOBJ_BODY, int(v.geom_bodyid[gid])) == f"link2_{s}"
        assert mujoco.mj_name2id(v, mujoco.mjtObj.mjOBJ_BODY, f"fingertip_{s}") == -1, "no separate fingertip body"
        assert mujoco.mj_name2id(v, mujoco.mjtObj.mjOBJ_SITE, f"tip_{s}") >= 0, "tool site retained"


def test_control_contract_still_matches_golden_on_v3():
    # the v3 robot keeps the golden control values (kp / damping / ranges).
    g, v = _models()
    assert {round(float(g.actuator_gainprm[a][0]), 1) for a in range(g.nu)} == \
           {round(float(v.actuator_gainprm[a][0]), 1) for a in range(v.nu)} == {40.0}
    assert {round(float(g.dof_damping[j]), 2) for j in range(g.nv)} == \
           {round(float(v.dof_damping[j]), 2) for j in range(v.nv)} == {1.5}
