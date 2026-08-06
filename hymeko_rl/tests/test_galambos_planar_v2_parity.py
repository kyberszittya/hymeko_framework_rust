"""GALAMBOS_PLANAR_HYMEKO_EQUIVALENCE (2026-07-23): the load-bearing HyMeKo robot spec galambos_planar_v2.hymeko,
emitted through the canonical spec→IR→MJCF path (+ the thin control-config adapter), is equivalent to the golden
Python make_planar_arms_mjcf — structural, kinematic, contact, and control-config parity, with only the explained
fingertip-as-body (the IR carries one geom/link) and floor (scene, not robot) differences.
"""
from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np

from hymeko_rl.env.planar_grasp_env import emit_galambos_v2_mjcf, make_planar_arms_mjcf


def _golden():
    return mujoco.MjModel.from_xml_string(make_planar_arms_mjcf())


def _v2():
    return mujoco.MjModel.from_xml_string(emit_galambos_v2_mjcf())


def _arm_inventory(m):
    """(type, size, contype, conaffinity) → count, excluding the floor plane (type 0)."""
    c: Counter = Counter()
    for g in range(m.ngeom):
        if int(m.geom_type[g]) == 0:
            continue
        key = (int(m.geom_type[g]),) + tuple(round(float(x), 4) for x in m.geom_size[g]) \
            + (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
        c[key] += 1
    return c


def test_structural_parity_geom_inventory():
    assert _arm_inventory(_golden()) == _arm_inventory(_v2()), "arm geom inventory (type/size/masks) must match the golden"


def test_control_config_parity():
    g, v = _golden(), _v2()

    def jr(m):
        return {tuple(round(float(x), 3) for x in m.jnt_range[j]) for j in range(m.njnt)}

    def cr(m):
        return {tuple(round(float(x), 3) for x in m.actuator_ctrlrange[a]) for a in range(m.nu)}

    def dp(m):
        return {round(float(m.dof_damping[j]), 2) for j in range(m.nv)}

    def kp(m):
        return {round(float(m.actuator_gainprm[a][0]), 1) for a in range(m.nu)}

    assert jr(g) == jr(v) == {(-4.0, 4.0)}
    assert cr(g) == cr(v) == {(-4.0, 4.0)}
    assert dp(g) == dp(v) == {1.5}
    assert kp(g) == kp(v) == {40.0}
    assert g.nu == v.nu == 4 and g.njnt == v.njnt == 4


def _fingertips(m, d, q):
    d.qpos[:4] = np.asarray(q, float)
    mujoco.mj_forward(m, d)
    sph = sorted((gi for gi in range(m.ngeom)
                  if int(m.geom_type[gi]) == 2 and abs(float(m.geom_size[gi][0]) - 0.014) < 1e-6),
                 key=lambda gi: float(d.geom_xpos[gi][0]))
    return np.array(d.geom_xpos[sph[0]]), np.array(d.geom_xpos[sph[1]])


def test_kinematic_parity_fingertip_positions():
    g, v = _golden(), _v2()
    dg, dv = mujoco.MjData(g), mujoco.MjData(v)
    poses = [[0, 0, 0, 0], [0.5, -0.5, 0.5, -0.5], [1.0, 0.8, -1.0, -0.8], [-0.3, 1.2, 0.3, -1.2], [3.0, -3.0, 3.0, -3.0]]
    maxd = 0.0
    for q in poses:
        (gl, gr), (vl, vr) = _fingertips(g, dg, q), _fingertips(v, dv, q)
        maxd = max(maxd, float(np.linalg.norm(gl - vl)), float(np.linalg.norm(gr - vr)))
    assert maxd < 1e-4, f"fingertip world-position parity {maxd * 1000:.4f} mm exceeds tolerance"


def test_contact_parity_arm_link_collides_with_coin():
    # Every structural arm geom carries ARM_LEGALITY (contype 1 / conaffinity 3) so it collides with the coin (2/2).
    for m in (_golden(), _v2()):
        caps = [gi for gi in range(m.ngeom)
                if int(m.geom_type[gi]) in (3, 6) and int(m.geom_contype[gi]) == 1 and int(m.geom_conaffinity[gi]) == 3]
        assert caps, "no ARM_LEGALITY structural geoms found"
        for gi in caps:
            mask = (int(m.geom_contype[gi]) & 2) | (2 & int(m.geom_conaffinity[gi]))
            assert mask != 0, "arm-link<->coin collision is filtered"


def test_structural_body_count_matches_golden_exactly():
    # v3 golden STRUCTURE (2026-07-22): the fingertip contact geom is folded onto link2 and the separate fingertip
    # body is removed, so the canonical arm now has EXACTLY the golden's body count (no +2 fingertip bodies). The
    # semantic graph is therefore natively six vertices (no projection needed).
    g, v = _golden(), _v2()
    assert v.nbody == g.nbody
