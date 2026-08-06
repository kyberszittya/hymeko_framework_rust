"""GALAMBOS_CONTROL_CONTRACT_RUNTIME sentinels (2026-07-22): prove the four canonical control values
(joint range, actuator ctrlrange, joint damping, servo kp) are LOAD-BEARING — an edit to the
``galambos_control { … }`` block in ``galambos_planar_v2.hymeko`` propagates all the way through the emitted MJCF
into the compiled ``MjModel`` field the runtime consumes — and that canonical mode HARD-FAILS (no Python fallback
constant) when the control block or any required field is absent.

Each sentinel writes a temp copy of the spec (same directory, so the relative ``@"meta_kinematics.hymeko"`` import
resolves) with ONE control field mutated, re-emits through :func:`emit_galambos_v2_mjcf`, compiles the MjModel, and
asserts the corresponding model field carries the mutated value — not the original.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import pytest

from hymeko_rl.env.planar_grasp_env import _CANONICAL_ROBOT_V2, emit_galambos_v2_mjcf, read_control_contract

SRC = Path(_CANONICAL_ROBOT_V2)
TMP = SRC.parent / "_control_sentinel_tmp.hymeko"   # same dir → relative @import resolves against data/robotics/


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if TMP.exists():
        TMP.unlink()


def _mutate_field(field: str, value: float) -> Path:
    """Write a temp spec identical to v2 except ``field`` inside the galambos_control block set to ``value``."""
    text = SRC.read_text(encoding="utf-8")
    new = re.sub(rf"(\bgalambos_control\s*\{{[^}}]*?\b{field}\s+)-?[\d.]+", rf"\g<1>{value}", text, count=1)
    assert new != text, f"sentinel could not mutate control field {field!r}"
    TMP.write_text(new, encoding="utf-8")
    return TMP


def _model(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(emit_galambos_v2_mjcf(str(path)))


def test_joint_range_is_load_bearing():
    base = read_control_contract(str(SRC))["joint_range"]
    m = _model(_mutate_field("joint_range", base + 1.0))
    rng = {tuple(round(float(x), 3) for x in m.jnt_range[j]) for j in range(m.njnt)}
    assert rng == {(-(base + 1.0), base + 1.0)}, f"joint range did not track the spec: {rng}"


def test_ctrlrange_is_load_bearing():
    base = read_control_contract(str(SRC))["ctrl_range"]
    m = _model(_mutate_field("ctrl_range", base - 1.0))
    cr = {tuple(round(float(x), 3) for x in m.actuator_ctrlrange[a]) for a in range(m.nu)}
    assert cr == {(-(base - 1.0), base - 1.0)}, f"actuator ctrlrange did not track the spec: {cr}"


def test_damping_is_load_bearing():
    base = read_control_contract(str(SRC))["damping"]
    m = _model(_mutate_field("damping", base + 1.0))
    dp = {round(float(m.dof_damping[j]), 3) for j in range(m.nv)}
    assert dp == {base + 1.0}, f"joint damping did not track the spec: {dp}"


def test_kp_is_load_bearing():
    base = read_control_contract(str(SRC))["kp"]
    m = _model(_mutate_field("kp", base + 15.0))
    kp = {round(float(m.actuator_gainprm[a][0]), 1) for a in range(m.nu)}
    assert kp == {base + 15.0}, f"servo kp gain did not track the spec: {kp}"


def test_missing_control_block_hard_fails():
    text = SRC.read_text(encoding="utf-8")
    TMP.write_text(re.sub(r"\bgalambos_control\s*\{[^}]*\}", "", text, count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="galambos_control"):
        read_control_contract(str(TMP))


def test_missing_required_field_hard_fails():
    text = SRC.read_text(encoding="utf-8")
    TMP.write_text(re.sub(r"\n\s*kp\s+-?[\d.]+;", "", text, count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="kp"):
        read_control_contract(str(TMP))
