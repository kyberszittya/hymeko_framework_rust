"""U3 (R11.7A unification) — the canonical manipulation-rig generator.

Covers the single ObjectSpec→builder mapping (the reference coin maps bit-identically to the pre-unification
``reconstruct_handoff`` kwargs), the robot contract carrying the embodiment orthogonally, the fail-loud guard
on the not-yet-plumbed density/friction axis (U3b), and the stable-handle registry. The heavy reconstruct
chain is mocked — the model-level O0 parity is exercised by the production smoke (U4).
"""
from __future__ import annotations

import mujoco
import pytest

from hymeko_rl.coin_delivery import manipulation_rig as MR
from hymeko_rl.coin_delivery.manipulation_rig import (
    ManipulationRig, RigRegistry, RobotContract, build_manipulation_rig)
from hymeko_rl.env.object_spec import COIN_OBJECT, ObjectSpec, Shape

_DISK_XML = ('<mujoco><worldbody><body name="disk">'
             '<joint name="disk_x" type="slide" axis="1 0 0"/>'
             '<geom name="disk" type="cylinder" size="0.02 0.02"/></body></worldbody></mujoco>')
_NODISK_XML = ('<mujoco><worldbody><body name="puck">'
               '<joint name="j" type="slide" axis="1 0 0"/>'
               '<geom name="puck" type="cylinder" size="0.02 0.02"/></body></worldbody></mujoco>')


def _model(xml: str):
    return mujoco.MjModel.from_xml_string(xml)


class _FakeInner:
    def __init__(self, model) -> None:
        self.model = model


class _FakeRL:
    def __init__(self, model) -> None:
        self.inner = _FakeInner(model)


def _patch_rh(monkeypatch, captured: dict):
    def fake_rh(pi0, demo, **kw):
        captured.update(kw)
        return _FakeRL(_model(_DISK_XML)), "GATE", None, None
    monkeypatch.setattr(MR, "reconstruct_handoff", fake_rh)


# --- the ObjectSpec is threaded as a SINGLE carrier (not exploded into loose kwargs) --------------
def test_coin_object_threaded_as_single_carrier(monkeypatch) -> None:
    captured: dict = {}
    _patch_rh(monkeypatch, captured)
    robot = RobotContract(geom="POINT", arm_mjcf_transform="BALLTF", label="balltip")
    rig = build_manipulation_rig("PI0", "DEMO", object_spec=COIN_OBJECT, robot=robot)
    # the whole object flows as one ObjectSpec; the robot supplies geom/arm_mjcf_transform
    assert captured["object_spec"] is COIN_OBJECT
    assert captured["geom"] == "POINT" and captured["arm_mjcf_transform"] == "BALLTF"
    # no loose object kwargs leak out of the generator (the mapping happens deeper, in make_coin_env)
    assert "coin_shape" not in captured and "disk_radius_override" not in captured
    assert isinstance(rig, ManipulationRig)
    assert rig.gate == "GATE" and rig.object_spec is COIN_OBJECT and rig.robot is robot
    assert rig.registry.object_geom == "disk"


def test_variant_object_threaded(monkeypatch) -> None:
    captured: dict = {}
    _patch_rh(monkeypatch, captured)
    spec = ObjectSpec(shape=Shape.BOX, radius=0.03, radius_y=0.02, density=800.0, frictionloss=0.2)
    build_manipulation_rig("PI0", "DEMO", object_spec=spec, robot=RobotContract())
    threaded = captured["object_spec"]
    # density/friction now flow (U3b closed the gap) — no silent drop, no NotImplementedError
    assert threaded.shape is Shape.BOX and threaded.radius == 0.03 and threaded.radius_y == 0.02
    assert threaded.density == 800.0 and threaded.frictionloss == 0.2


# --- stable-handle registry -----------------------------------------------------------------------
def test_registry_resolve_requires_the_stable_disk_handle() -> None:
    RigRegistry.resolve(_model(_DISK_XML))                # present → ok
    with pytest.raises(ValueError, match="disk"):
        RigRegistry.resolve(_model(_NODISK_XML))          # renamed away → fail loud


def test_registry_ids_are_resolvable() -> None:
    m = _model(_DISK_XML)
    reg = RigRegistry.resolve(m)
    assert reg.object_geom_id(m) >= 0 and reg.object_body_id(m) >= 0


def test_robot_contract_defaults_to_clamp() -> None:
    r = RobotContract()
    assert r.geom is None and r.arm_mjcf_transform is None and r.label == "clamp"
