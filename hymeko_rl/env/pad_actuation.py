"""Shared embodiment-actuation infrastructure — schema-aware wrist-yaw + independent pad-closure control.

Generic over any PlanarGraspEnv-style model that exposes the optional actuator groups ARM / WRIST_YAW / PAD_CLOSURE by
NAMED actuators (``a_j*`` / ``a_pad_{side}`` / ``a_close_{side}``) and joints. Reusable by Coin Delivery, PickPlaceEnv,
and future Beni/AIBO end-effectors (§13). NO task logic here — only the typed actuator discovery, the wrist-alignment
and bounded closure-force controllers (world-frame geometry + real MuJoCo contact forces), and a generalized snapshot
padding. Coin-specific oracle phases/certification live in the experiment layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mujoco
import numpy as np


def actuator_groups(model: Any) -> dict[str, list[int]]:
    """Typed actuator groups by NAMED actuator (never fixed indices). ARM = ``a_j*``; WRIST_YAW = ``a_pad_*``;
    PAD_CLOSURE = ``a_close_*``. Missing groups are empty lists."""
    groups: dict[str, list[int]] = {"ARM": [], "WRIST_YAW": [], "PAD_CLOSURE": []}
    for i in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if name.startswith("a_j"):
            groups["ARM"].append(i)
        elif name.startswith("a_pad_"):
            groups["WRIST_YAW"].append(i)
        elif name.startswith("a_close_"):
            groups["PAD_CLOSURE"].append(i)
    return groups


def pad_joint_qpos_addrs(model: Any) -> list[int]:
    """Sorted qpos addresses of the added pad joints (``pad_hinge_*`` / ``pad_slide_*``) — for generalized snapshot
    padding. All planar pad joints are 1-DOF, so ``jnt_qposadr`` are the insertion points."""
    addrs = []
    for j in range(int(model.njnt)):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("pad_hinge_") or nm.startswith("pad_slide_"):
            addrs.append(int(model.jnt_qposadr[j]))
    return sorted(addrs)


def _side_of(name: str) -> str:
    return "left" if "left" in name else "right"


def _act_by_side(model: Any, ids: list[int]) -> dict[str, int]:
    return {_side_of(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""): i for i in ids}


def _pad_body_id(model: Any, side: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pad_{side}")


def _pad_normal_force(model: Any, data: Any, side: str, disk_gid: int) -> float:
    """Sum of the NORMAL contact force between this pad's fingertip geoms and the coin (real MuJoCo contacts)."""
    total = 0.0
    for k in range(data.ncon):
        c = data.contact[k]
        if disk_gid not in (c.geom1, c.geom2):
            continue
        other = c.geom1 if c.geom2 == disk_gid else c.geom2
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        if nm.startswith("fingertip_") and side in nm:
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, k, f)
            total += abs(float(f[0]))                              # contact-frame normal component
    return total


class Phase(str, Enum):
    APPROACH = "APPROACH"
    WRIST_ALIGN = "WRIST_ALIGN"
    PAD_CLOSE = "PAD_CLOSE"
    FORCE_HOLD = "FORCE_HOLD"
    TRANSPORT = "TRANSPORT"
    BRAKE = "BRAKE"
    RELEASE = "RELEASE"
    WITHDRAW = "WITHDRAW"
    SETTLE = "SETTLE"


@dataclass
class PadLimits:
    """Bounded controller limits; the closure force target is justified from the pad slide actuator range/kp (§5)."""
    hinge_range: float = 1.2          # a_pad ctrlrange (rad)
    slide_range: float = 0.02         # a_close ctrlrange (m)
    wrist_rate: float = 0.25          # max wrist target step per env step (rad) — bounded velocity
    approach_rate: float = 0.003      # closure approach rate before contact (m/step)
    force_target: float = 2.5         # regulated bilateral normal-force target (N) — well within the slide servo
    force_kp: float = 0.0016          # slide adjustment per Newton of force error (anti-windup clamped to slide_range)
    release_rate: float = 0.004       # force-target ramp-down + pad-open rate (m/step)


@dataclass
class PadControlLog:
    wrist_err: list[float] = field(default_factory=list)
    force_left: list[float] = field(default_factory=list)
    force_right: list[float] = field(default_factory=list)
    saturated: int = 0


class WristCloseController:
    """Schema-aware wrist-align + bounded closure-force controller. ``motor_override`` takes the arm cooperative motor
    (nu-sized) and OVERWRITES the wrist/closure actuator indices with phase-appropriate targets, leaving ARM untouched.
    For a model with no wrist/closure actuators it is a no-op (E0 byte-identical)."""

    def __init__(self, planar_env: Any, limits: PadLimits | None = None) -> None:
        self._env = planar_env
        self.m, self.d = planar_env.model, planar_env.data
        self.lim = limits or PadLimits()
        self.groups = actuator_groups(self.m)
        self._wrist = _act_by_side(self.m, self.groups["WRIST_YAW"])
        self._close = _act_by_side(self.m, self.groups["PAD_CLOSURE"])
        self._disk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
        self._slide_cmd = {"left": 0.0, "right": 0.0}             # integral closure state (per side)
        self.log = PadControlLog()

    def reset(self) -> None:
        self._slide_cmd = {"left": 0.0, "right": 0.0}

    def _wrist_target(self, side: str) -> float:
        bid = _pad_body_id(self.m, side)
        if bid < 0:
            return 0.0
        R = self.d.xmat[bid].reshape(3, 3)
        normal = R[:, 0]                                          # pad contact normal (local X) in world (measured basis)
        pad_pos = self.d.xpos[bid]
        to_coin = self.d.geom_xpos[self._disk] - pad_pos
        to_coin[2] = 0.0
        nrm = np.linalg.norm(to_coin)
        if nrm < 1e-9:
            return float(self._current_hinge(side))
        desired = to_coin / nrm
        err = float(np.arctan2(normal[0] * desired[1] - normal[1] * desired[0], normal[0] * desired[0] + normal[1] * desired[1]))
        self.log.wrist_err.append(abs(err))
        step = float(np.clip(err, -self.lim.wrist_rate, self.lim.wrist_rate))   # bounded rate
        tgt = self._current_hinge(side) + step
        clamped = float(np.clip(tgt, -self.lim.hinge_range, self.lim.hinge_range))
        self.log.saturated += int(clamped != tgt)
        return clamped

    def _current_hinge(self, side: str) -> float:
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"pad_hinge_{side}")
        return float(self.d.qpos[self.m.jnt_qposadr[j]]) if j >= 0 else 0.0

    def _closure_target(self, side: str, phase: Phase) -> float:
        f = _pad_normal_force(self.m, self.d, side, self._disk)
        (self.log.force_left if side == "left" else self.log.force_right).append(f)
        cmd = self._slide_cmd[side]
        if phase in (Phase.RELEASE, Phase.WITHDRAW, Phase.SETTLE):
            cmd -= self.lim.release_rate                          # ramp force target down + open the pads
        elif phase is Phase.PAD_CLOSE and f < 1e-3:
            cmd += self.lim.approach_rate                         # bounded approach until contact
        elif phase in (Phase.PAD_CLOSE, Phase.FORCE_HOLD, Phase.TRANSPORT, Phase.BRAKE):
            cmd += self.lim.force_kp * (self.lim.force_target - f)   # regulate normal force (anti-windup via clamp)
        cmd = float(np.clip(cmd, 0.0, self.lim.slide_range))       # position limit / anti-windup
        self._slide_cmd[side] = cmd
        return cmd

    def motor_override(self, arm_motor: np.ndarray, phase: Phase) -> np.ndarray:
        full = np.asarray(arm_motor, np.float64).copy()
        if full.shape[0] != int(self.m.nu):                       # size to nu (arm motor was nu-sized already)
            sized = np.zeros(int(self.m.nu))
            sized[self.groups["ARM"]] = np.asarray(arm_motor).ravel()[:len(self.groups["ARM"])]
            full = sized
        for side, idx in self._wrist.items():
            full[idx] = self._wrist_target(side) if phase not in (Phase.APPROACH,) else self._current_hinge(side)
        for side, idx in self._close.items():
            full[idx] = self._closure_target(side, phase)
        return full.astype(np.float32)


def build_wristed_contact_env(planar_env: Any, bank: list[dict], contract: Any, *, limits: PadLimits | None = None,
                              **kw: Any) -> Any:
    """A ``ContactFormationEnv`` whose motor path is schema-aware: it drives ARM (cooperative) + WRIST_YAW + PAD_CLOSURE
    via a :class:`WristCloseController`, and pads the canonical 7-qpos snapshot into the model's typed layout on restore.
    For an E0 model (no wrist/closure actuators) the controller is a no-op → byte-identical to the canonical env."""
    from hymeko_rl.env.contact_formation_env import ContactFormationEnv
    from hymeko_rl.env.planar_snapshot import restore_planar

    class _WristedPadContactFormationEnv(ContactFormationEnv):
        def __init__(self) -> None:
            super().__init__(planar_env, bank, contract, **kw)
            self._pad = WristCloseController(planar_env, limits)
            self._phase = Phase.APPROACH
            self._qpos_addrs = pad_joint_qpos_addrs(planar_env.model)

        def set_phase(self, phase: Phase) -> None:
            self._phase = phase

        def reset(self, *, seed=None, options=None):
            self._pad.reset()
            self._phase = Phase.APPROACH
            return super().reset(seed=seed, options=options)

        def _physics_motor(self, arm_motor: np.ndarray) -> np.ndarray:
            return self._pad.motor_override(arm_motor, self._phase)     # ARM untouched; wrist/closure overwritten

        def _restore(self, item: dict) -> None:
            snap = item["snap"]
            m = planar_env.model
            idx = [a - k for k, a in enumerate(self._qpos_addrs)]        # shift each insert by prior inserts
            if len(snap.qpos) + len(idx) != int(m.nq):
                if len(snap.qpos) == int(m.nq):                          # already this layout (E0 / re-restore)
                    restore_planar(planar_env, snap)
                    return
                raise ValueError(f"pad-restore layout mismatch: {len(snap.qpos)} snapshot qpos + {len(idx)} pad DOFs "
                                 f"!= model nq {int(m.nq)} (schema/hash incompatible — never silently pad)")
            import dataclasses
            padded = dict(qpos=np.insert(snap.qpos, idx, 0.0), qvel=np.insert(snap.qvel, idx, 0.0),
                          qacc_warmstart=np.insert(snap.qacc_warmstart, idx, 0.0))
            restore_planar(planar_env, dataclasses.replace(snap, **padded))

    return _WristedPadContactFormationEnv()
