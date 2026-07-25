"""H2 B_v — unit tests for the CONTROL-TO-CONTACT-VELOCITY primitives.

Two fast layers (the identification/FD path over the real coin env is exercised by
``experiments/bv_identification_benchmark.py`` as the production-scale integration + performance smoke, §3):
  * PURE kinematic transport — stationary / translation / rotation-at-offset / sign / left-right ordering (no MuJoCo).
  * THIN MuJoCo readers — a tiny synthetic planar free body with a KNOWN qvel; ``geom_planar_velocity`` / ``coin_twist``
    must recover it, and ``mj_objectVelocity`` res[2] must equal the hinge qvel (the cross-check the coin twist relies on).
"""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.contact_velocity import (
    BvConfig, contact_relative_velocity, geom_planar_velocity, planar_point_velocity, resolve_nt)


# --------------------------------------------------------------------------------------------------------------------
# PURE transport
# --------------------------------------------------------------------------------------------------------------------
def test_stationary_tip_and_coin_zero_relative_velocity():
    """A stationary tip and a stationary coin ⇒ v_rel ≈ 0 at any contact point."""
    v_rel = contact_relative_velocity(v_tip=[0.0, 0.0], v_center=[0.0, 0.0], omega=0.0, p=[0.02, 0.0], c=[0.0, 0.0])
    assert np.allclose(v_rel, 0.0, atol=1e-12)


def test_pure_coin_translation_gives_negative_center_velocity():
    """Tip still, coin translating at v_c (no spin) ⇒ v_rel = v_tip − v_c = −v_c, independent of the contact point."""
    v_c = np.array([0.3, -0.1])
    for p in ([0.02, 0.0], [-0.015, 0.011]):
        v_rel = contact_relative_velocity([0.0, 0.0], v_c, 0.0, p, [0.0, 0.0])
        assert np.allclose(v_rel, -v_c, atol=1e-12)


def test_pure_coin_rotation_at_offset_contact():
    """Tip still, coin spinning at ω about its centre ⇒ coin-surface velocity at p is ω·ẑ×(p−c); v_rel = −that.
    For c=0, p=(r,0), ω>0: surface velocity = ω·(0, r) (+y); v_rel = (0, −ωr)."""
    r, omega = 0.02, 5.0
    v_rel = contact_relative_velocity([0.0, 0.0], [0.0, 0.0], omega, [r, 0.0], [0.0, 0.0])
    assert np.allclose(v_rel, [0.0, -omega * r], atol=1e-12)
    # a contact on the −x side sees the opposite-sign tangential surface velocity
    v_rel2 = contact_relative_velocity([0.0, 0.0], [0.0, 0.0], omega, [-r, 0.0], [0.0, 0.0])
    assert np.allclose(v_rel2, [0.0, omega * r], atol=1e-12)


def test_normal_tangential_sign_convention():
    """Frozen frame n = unit(c − tip) (tip→centre), t = R₊₉₀ n. A tip on the +x side (p=(r,0), c=0) has n=(−1,0).
    A tip velocity INTO the coin (−x) ⇒ v_n > 0 (approach). A +y tip velocity ⇒ v_t along t=(0,−1) ⇒ v_t < 0."""
    c, p = np.zeros(2), np.array([0.02, 0.0])
    n = (c - p) / np.linalg.norm(c - p)                          # = (−1, 0)
    t = np.array([-n[1], n[0]])                                  # R₊₉₀ n = (0, −1)
    v_rel_in = contact_relative_velocity([-0.4, 0.0], [0.0, 0.0], 0.0, p, c)   # tip moving −x (into coin)
    v_n, v_t = resolve_nt(v_rel_in, n, t)
    assert v_n > 0 and abs(v_t) < 1e-12                          # pure approach, no tangential
    v_rel_side = contact_relative_velocity([0.0, 0.3], [0.0, 0.0], 0.0, p, c)  # tip moving +y
    v_n2, v_t2 = resolve_nt(v_rel_side, n, t)
    assert abs(v_n2) < 1e-12 and v_t2 < 0                        # pure tangential, sign set by t=(0,−1)


def test_left_right_component_ordering_is_independent():
    """The 4-vector ordering [v_n,L, v_t,L, v_n,R, v_t,R]: left and right are computed from independent frames, so a
    left-only motion leaves the right pair untouched (guards against an L/R index swap)."""
    c = np.zeros(2)
    p_l, p_r = np.array([-0.02, 0.0]), np.array([0.02, 0.0])
    n_l, n_r = (c - p_l) / 0.02, (c - p_r) / 0.02
    t_l, t_r = np.array([-n_l[1], n_l[0]]), np.array([-n_r[1], n_r[0]])
    # a coin translation excites BOTH; a fabricated tip-left-only velocity excites only the left pair
    vrel_l = contact_relative_velocity([0.1, 0.0], [0.0, 0.0], 0.0, p_l, c)
    vrel_r = contact_relative_velocity([0.0, 0.0], [0.0, 0.0], 0.0, p_r, c)
    vn_l, vt_l = resolve_nt(vrel_l, n_l, t_l)
    vn_r, vt_r = resolve_nt(vrel_r, n_r, t_r)
    vrel4 = np.array([vn_l, vt_l, vn_r, vt_r])
    assert abs(vrel4[0]) + abs(vrel4[1]) > 1e-6                  # left pair non-zero
    assert abs(vrel4[2]) < 1e-12 and abs(vrel4[3]) < 1e-12       # right pair untouched


def test_planar_point_velocity_rigid_body_transport():
    """Rigid-body transport: the velocity of a point offset from a body's origin is v_origin + ω·ẑ×(x−origin).
    Origin at rest, ω>0, point at +x offset ⇒ the offset point moves +y at ω·|offset| (the transport the common-contact
    measurement relies on — using the geom ORIGIN instead of the contact point would drop this term)."""
    v = planar_point_velocity(v_origin=[0.0, 0.0], omega=3.0, origin=[0.0, 0.0], x_point=[0.02, 0.0])
    assert np.allclose(v, [0.0, 3.0 * 0.02], atol=1e-12)
    # pure translation: transport is offset-independent
    v2 = planar_point_velocity([0.4, -0.2], 0.0, [0.1, 0.1], [0.5, 0.9])
    assert np.allclose(v2, [0.4, -0.2], atol=1e-12)
    # combined: origin translating AND spinning
    v3 = planar_point_velocity([1.0, 0.0], 2.0, [0.0, 0.0], [0.0, 0.05])   # x=+0.05y ⇒ ω×r = 2·(−0.05,0)
    assert np.allclose(v3, [1.0 - 0.1, 0.0], atol=1e-12)


def test_geom_point_velocity_transports_spin_to_offset(planar_model):
    """On the synthetic body spinning at ω, the velocity read at a point 20 mm from the geom origin must include the
    ω×r term — i.e. differ from the geom-origin velocity (the exact error the 'use the real contact point' fix removes)."""
    mujoco, m, d = planar_model
    from hymeko_rl.coin_delivery.contact_velocity import geom_point_velocity, geom_planar_velocity
    d.qvel[:] = [0.0, 0.0, 6.0]                                   # pure spin about the coin centre
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    origin_xy = d.geom_xpos[gid][:2].copy()
    v_origin = geom_planar_velocity(m, d, gid)
    v_edge = geom_point_velocity(m, d, gid, origin_xy + np.array([0.02, 0.0]))
    assert np.allclose(v_origin, 0.0, atol=1e-9)                 # geom origin is stationary under pure spin
    assert np.allclose(v_edge, [0.0, 6.0 * 0.02], atol=1e-6)     # the 20 mm point moves at ω·r — the transported term


def test_bvconfig_frozen_defaults_are_sane():
    cfg = BvConfig()
    assert cfg.eps_scales[0] < cfg.eps_scales[1] and cfg.fn_floor > 0
    assert cfg.dq_levels[0] < cfg.dq_levels[1] < cfg.dq_levels[2]   # small < medium < near-boundary


# --------------------------------------------------------------------------------------------------------------------
# THIN readers — tiny synthetic planar free body
# --------------------------------------------------------------------------------------------------------------------
_PLANAR_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="coin" pos="0.1 -0.2 0">
      <joint name="jx" type="slide" axis="1 0 0"/>
      <joint name="jy" type="slide" axis="0 1 0"/>
      <joint name="jw" type="hinge" axis="0 0 1"/>
      <geom name="disk" type="cylinder" size="0.02 0.005" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def planar_model():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(_PLANAR_XML)
    d = mujoco.MjData(m)
    return mujoco, m, d


def test_geom_planar_velocity_recovers_known_translation(planar_model):
    """A known planar slide velocity [vx, vy] is recovered by ``geom_planar_velocity`` on the disk geom."""
    mujoco, m, d = planar_model
    d.qvel[:] = [0.37, -0.11, 0.0]
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    v = geom_planar_velocity(m, d, gid)
    assert np.allclose(v, [0.37, -0.11], atol=1e-9)


def test_coin_twist_recovers_omega_and_crosschecks_objectvelocity(planar_model):
    """``coin_twist`` returns (v_center, ω) with ω = qvel[hinge]; ``mj_objectVelocity`` res[2] must equal that hinge ω
    (the cross-check the coin-twist reader relies on — a body-vs-geom frame mismatch would break it)."""
    mujoco, m, d = planar_model
    from hymeko_rl.coin_delivery.contact_velocity import coin_twist
    d.qvel[:] = [0.05, 0.02, 4.2]
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    v_center, omega = coin_twist(m, d, 0, gid)                    # coin slide-x qvel address = 0 here
    assert np.allclose(v_center, [0.05, 0.02], atol=1e-9)
    assert abs(omega - 4.2) < 1e-9
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(gid), res, 0)
    assert abs(res[2] - omega) < 1e-9                            # rotational-z component == hinge qvel
