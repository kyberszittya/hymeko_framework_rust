"""Tests for the declarative contact-quality contract (hymeko_rl/env/contact_legality.py).

Covers the pure spec/state logic (role lookup, state-derived properties) and the model-driven pieces
(role assignment from a compiled model, geom-granular classification of real MuJoCo contacts) via a small
v2 PlanarGraspEnv. The graded/strict *consequences* live in the env and are tested in test_planar_grasp_env.
"""
from __future__ import annotations

import mujoco
import pytest

from hymeko_rl.env.contact_legality import (
    ContactLegalitySpec, ContactLegalityState, ContactMode, classify_contacts, contact_force_magnitude,
    GeomRole,
)
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv


# ── pure logic (no model) ──────────────────────────────────────────────────────────────────────────

def test_state_derived_properties() -> None:
    """``fingertip_contact`` = OR of the two sides; ``both_fingertip_contact`` = AND."""
    assert not ContactLegalityState().fingertip_contact
    assert ContactLegalityState(left_fingertip_contact=True).fingertip_contact
    assert not ContactLegalityState(left_fingertip_contact=True).both_fingertip_contact
    both = ContactLegalityState(left_fingertip_contact=True, right_fingertip_contact=True)
    assert both.fingertip_contact and both.both_fingertip_contact


def test_spec_role_lookup_and_mode() -> None:
    """``role`` maps a geom id to its declared :class:`GeomRole`; GRADED never invalidates, STRICT does."""
    spec = ContactLegalitySpec(object_geoms=frozenset({1}), left_fingertip_geoms=frozenset({2}),
                               right_fingertip_geoms=frozenset({3}), arm_body_geoms=frozenset({4, 5}))
    assert spec.role(1) is GeomRole.OBJECT
    assert spec.role(2) is GeomRole.FINGERTIP and spec.role(3) is GeomRole.FINGERTIP
    assert spec.role(4) is GeomRole.ARM_BODY and spec.role(5) is GeomRole.ARM_BODY
    assert spec.role(99) is GeomRole.OTHER
    assert spec.fingertip_geoms == frozenset({2, 3})
    assert spec.mode is ContactMode.GRADED and not spec.invalidates_on_arm_body
    strict = ContactLegalitySpec(frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}),
                                 mode=ContactMode.STRICT)
    assert strict.invalidates_on_arm_body


# ── model-driven (small v2 env) ────────────────────────────────────────────────────────────────────

def _v2_env() -> PlanarGraspEnv:
    e = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3, contact_legality=True)
    e.reset(seed=0)
    return e


def test_from_model_assigns_roles() -> None:
    """``from_model`` puts the disk in OBJECT, the two fingertip geoms in FINGERTIP (split by side), and every
    other arm-link geom in ARM_BODY — the single place the naming convention is applied."""
    e = _v2_env()
    sp = e._contact_spec
    assert sp is not None
    m = e.model
    ftl = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"))
    ftr = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right"))
    assert sp.object_geoms == frozenset({e._disk_geom})
    assert ftl in sp.left_fingertip_geoms and ftr in sp.right_fingertip_geoms
    assert sp.arm_body_geoms and ftl not in sp.arm_body_geoms and ftr not in sp.arm_body_geoms
    # every arm-body geom really is a non-fingertip arm-link geom
    for g in sp.arm_body_geoms:
        assert sp.role(g) is GeomRole.ARM_BODY
    e.close()


def test_from_model_raises_when_no_fingertips() -> None:
    """An unsatisfiable contract (no geom matches the fingertip prefix) fails loud, not silently empty."""
    e = _v2_env()
    with pytest.raises(ValueError, match="unsatisfiable"):
        ContactLegalitySpec.from_model(
            e.model, object_geoms={e._disk_geom}, arm_bodies_left=e._left_bodies,
            arm_bodies_right=e._right_bodies, fingertip_prefix="does_not_exist_")
    e.close()


def test_classify_contacts_fingertip_and_arm_body() -> None:
    """On real MuJoCo contacts at the fingertip: the fingertip sphere is classified as a FINGERTIP contact and
    the co-located distal capsule as an ARM_BODY contact (counted, finite impulse) — the two branches read from
    the same real contact set. That the coin at the fingertip centre also touches the distal capsule (rather
    than the fingertip alone) is the geometry fact that motivated the graded model: the fingertip does not
    protrude past its own link, so a fingertip grasp inevitably involves an incidental arm-body contact."""
    e = _v2_env()
    m, d = e.model, e.data
    sp = e._contact_spec
    ftl = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"))
    d.qpos[:] = 0.0
    mujoco.mj_forward(m, d)
    ft_xy = d.geom_xpos[ftl][:2].copy()
    d.qpos[e._disk_x_adr], d.qpos[e._disk_y_adr] = float(ft_xy[0]), float(ft_xy[1])
    mujoco.mj_forward(m, d)
    st = classify_contacts(m, d, sp)
    assert st.fingertip_contact                                # the sphere → FINGERTIP (not misread as arm-body)
    assert st.arm_body_contact and st.arm_body_contact_count >= 1   # the co-located capsule → ARM_BODY
    assert st.arm_body_contact_impulse >= 0.0
    e.close()


def test_contact_force_magnitude_nonnegative() -> None:
    """``contact_force_magnitude`` is finite and non-negative on a real contact (the coin resting on the floor)."""
    e = _v2_env()
    e.data.qpos[:] = 0.0
    mujoco.mj_forward(e.model, e.data)
    assert int(e.data.ncon) >= 1
    for i in range(int(e.data.ncon)):
        assert contact_force_magnitude(e.model, e.data, i) >= 0.0
    e.close()
