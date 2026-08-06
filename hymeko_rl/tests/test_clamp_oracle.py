"""Tests for CLAMP-ORACLE-0 (isolated physical clamp-transport oracle).

Covers: the isolated `diagnostic_physical_oracle` model (cylinder coin + two independent opposing pads, no arm), the
clamp+transport protocol, and the decisive facts — a POSITION-controlled parallel jaw strict-transports the cylinder
(O1 passes), the direct-object-drive calibration is NEVER credited as a grasp (O5 object-actuated), and a scrambled
(open-right) clamp does NOT pass (correct bilateral clamp beats the scramble). Deterministic (mj_step from a fixed model).
"""
from __future__ import annotations

import mujoco

from hymeko_rl.train.clamp_oracle import Actuation, OracleParams, OracleRig, build_oracle_mjcf

_P = OracleParams()


def _rig(actuation, **kw) -> OracleRig:
    return OracleRig(build_oracle_mjcf(actuation=actuation, p=_P, **kw), actuation, _P)


def test_oracle_model_is_isolated_cylinder_plus_two_pads() -> None:
    m = _rig(Actuation.POSITION).model
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}
    assert {"disk", "fingertip_left", "fingertip_right"} <= names
    assert m.geom("disk").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER      # same cylindrical coin
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert not any(b and ("arm" in b or "link" in b) for b in bodies)    # NO canonical arm (isolated)


def test_o1_position_jaw_strict_transports_the_cylinder() -> None:
    # the decisive Case-A fact: a POSITION-controlled correctly-oriented parallel jaw strict-transports the cylinder.
    r = _rig(Actuation.POSITION).run_condition()
    assert r.strict_pass is True
    assert r.both_frac >= 0.75 and r.align >= 0.7 and r.rho >= 0.30
    assert r.max_eta < 0.98 and r.min_FN >= 0.05 and r.penetration_max <= 0.02 and r.solver_ok


def test_force_clamp_also_transports() -> None:
    assert _rig(Actuation.FORCE).run_condition().strict_pass is True
    assert _rig(Actuation.FORCE, fingertip_shape="capsule").run_condition().strict_pass is True


def test_object_direct_is_never_a_grasp() -> None:
    # O5: driving the coin along its own joints reaches the target (high rho) but must NEVER count as grasp capability.
    r = _rig(Actuation.OBJECT_DIRECT).run_condition()
    assert r.object_actuated is True
    assert r.strict_pass is False                                       # object-actuated → excluded from a grasp claim
    assert r.rho >= 0.30                                                # (it does move — calibration only)


def test_box_positive_control_transports() -> None:
    assert _rig(Actuation.FORCE, coin_shape="box").run_condition().strict_pass is True


def test_scramble_open_right_does_not_pass() -> None:
    # T9: a scrambled (open-right → unilateral) clamp must NOT strict-pass; the correct bilateral clamp beats it.
    assert _rig(Actuation.POSITION).run_condition(control="open_right").strict_pass is False
    assert _rig(Actuation.FORCE).run_condition(control="open_right").strict_pass is False
