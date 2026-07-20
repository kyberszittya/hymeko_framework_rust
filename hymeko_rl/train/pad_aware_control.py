"""PAD-AWARE-COOPERATIVE-CONTROL-0 — closed-loop control of the two distal pad-orientation actuators during delivery.

CLAMP-ORACLE-0 said the canonical clamp failure is kinematic; DERIVED-SCENARIOS-1 verified the K1 distal pad-orientation
DOF geometrically (pad normal rotates 0.788 rad) but left it NOT closed-loop — the cooperative controller drove only the
4 arm DOFs, so K1 delivery == K0. This module closes the loop: it drives the two distal actuators `a_pad_{side}` toward a
geometry-based desired pad orientation, in addition to the frozen arm cooperative controller.

The K1 env is built by injecting `with_distal_pad_orientation` into `PlanarGraspEnv` (via its additive `arm_mjcf_transform`
hook; K0 byte-unchanged). `PadAwareContactFormationEnv` overrides `ContactFormationEnv._physics_motor` to append the 2
distal position targets to the 4-DoF arm motor (arm servos + `_obs` stay driven by the arm motor; only physics grows to
nu=6). The distal joints are POSITION-controlled (no force actuation). NO RL. NO canonical-arm/K0 change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np

from hymeko_rl.env.contact_formation_env import ContactFormationEnv
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

_EPS = 1e-9
_HINGE_MAX = 1.2                 # a_pad_{side} ctrlrange (rad)


class PadVariant(str, Enum):
    P0_NEUTRAL = "P0_neutral"            # capacity-matched K1 control: distal DOF exists but held at neutral (0)
    P1_GEOMETRY = "P1_geometry_aligned"  # orient each pad normal toward the local coin centre
    P2_PUSH_AWARE = "P2_push_aware"      # blend radial contact alignment with target-directed push geometry
    P3_SCRAMBLE = "P3_orientation_scramble"  # capacity-matched: the P1 target, sign-inverted (correct must beat this)
    P4_CONTACT_FB = "P4_contact_feedback"    # gate the P1 target on measured contact (position-controlled, no force)


@dataclass(frozen=True)
class PadParams:
    k: float = 1.5               # P-gain from misalignment angle to hinge target
    push_blend: float = 0.5      # P2 tangential blend weight


def _pad_facing(inner, side: str) -> np.ndarray:
    """The in-plane world direction the `pad_{side}` body currently faces (its local +x axis), unit-normalised."""
    b = mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_BODY, f"pad_{side}")
    xaxis = inner.data.xmat[b].reshape(3, 3)[:, 0][:2]
    return xaxis / (np.linalg.norm(xaxis) + _EPS)


def _pad_xy(inner, side: str) -> np.ndarray:
    b = mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_BODY, f"pad_{side}")
    return np.asarray(inner.data.xpos[b][:2], np.float64)


def _hinge_qpos(inner, side: str) -> float:
    j = mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_JOINT, f"pad_hinge_{side}")
    return float(inner.data.qpos[inner.model.jnt_qposadr[j]])


def desired_pad_dir(inner, side: str, variant: PadVariant, p: PadParams) -> np.ndarray:
    """The explicit desired pad-normal world direction. P1: toward the local coin centre (radial). P2: blend the radial
    with the target-directed push tangent. # Postconditions unit vector."""
    coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    toward_coin = coin - _pad_xy(inner, side)
    toward_coin = toward_coin / (np.linalg.norm(toward_coin) + _EPS)
    if variant is not PadVariant.P2_PUSH_AWARE:     # P1/P3/P4 use the radial toward-coin direction (P3 sign-scrambles it)
        return toward_coin
    push, _n = inner.direction_to_zone()                 # P2: blend the radial with the target-relative push (coin → zone)
    d = toward_coin + p.push_blend * np.asarray(push, np.float64)
    return d / (np.linalg.norm(d) + _EPS)


def _signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def distal_targets(inner, variant: PadVariant, p: PadParams = PadParams()) -> np.ndarray:
    """The two distal hinge position targets [left, right] for the given variant (a P-controller toward the desired pad
    direction). P0 holds neutral; P3 sign-inverts the correct target (capacity-matched scramble)."""
    if variant is PadVariant.P0_NEUTRAL:
        return np.zeros(2, np.float32)
    out = []
    for side in ("left", "right"):
        if variant is PadVariant.P4_CONTACT_FB:
            touching = getattr(inner._planar_metrics, f"{'left' if side == 'left' else 'right'}_contact")
            if not touching:                        # only orient once in contact (position-controlled, no force)
                out.append(0.0)
                continue
        ang = _signed_angle(_pad_facing(inner, side), desired_pad_dir(inner, side, variant, p))
        tgt = float(np.clip(_hinge_qpos(inner, side) + p.k * ang, -_HINGE_MAX, _HINGE_MAX))
        out.append(-tgt if variant is PadVariant.P3_SCRAMBLE else tgt)
    return np.asarray(out, np.float32)


class PadAwareContactFormationEnv(ContactFormationEnv):
    """ContactFormationEnv over the K1 (distal-pad) inner env; appends the 2 distal position targets to the 4-DoF arm
    motor in `_physics_motor`. The arm servos + `_obs` stay driven by the arm motor; only the physics command grows."""

    def __init__(self, planar_env, bank, contract, *, variant: PadVariant, params: PadParams = PadParams(), **kw) -> None:
        super().__init__(planar_env, bank, contract, **kw)
        self._variant = variant
        self._params = params
        self._pad_idx = [mujoco.mj_name2id(planar_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_pad_{s}")
                         for s in ("left", "right")]

    def _physics_motor(self, arm_motor: np.ndarray) -> np.ndarray:
        """The arm cooperative controller already returns an nu-sized command (distal actuators at 0); OVERWRITE the two
        distal actuator indices with the geometry-based pad-orientation targets. The arm servos are untouched."""
        full = np.asarray(arm_motor, np.float32).copy()
        full[self._pad_idx] = distal_targets(self._env, self._variant, self._params)
        return full

    def _restore(self, item: dict) -> None:
        """Pad the canonical (7-qpos) bank snapshot into the K1 (9-qpos) layout — insert the two distal hinge DOFs at
        neutral (0) at qpos/qvel indices [2, 5] — so the SAME held corpus is byte-paired across K0/K1."""
        import dataclasses
        from hymeko_rl.env.planar_snapshot import restore_planar
        snap = item["snap"]
        pad = dict(qpos=np.insert(snap.qpos, [2, 4], 0.0), qvel=np.insert(snap.qvel, [2, 4], 0.0),
                   qacc_warmstart=np.insert(snap.qacc_warmstart, [2, 4], 0.0))
        restore_planar(self._env, dataclasses.replace(snap, **pad))


def build_pad_aware_env(variant: PadVariant, bank, contract, *, params: PadParams = PadParams(), **planar_kw):
    """Build a K1 delivery env (canonical arm + distal pad DOF) whose step drives the distal actuators per `variant`."""
    from hymeko_rl.coin_delivery.scenarios.kinematic_variant import with_distal_pad_orientation
    planar = PlanarGraspEnv(arm_mjcf_transform=with_distal_pad_orientation, **planar_kw)
    return PadAwareContactFormationEnv(planar, bank, contract, variant=variant, params=params)


def bilateral_balance(alpha_L: float, alpha_R: float) -> float:
    """B_LR = 2·min(α_L, α_R) / (α_L + α_R + ε) ∈ [0,1]: 1 = perfectly shared load, 0 = one-finger. # Postconditions [0,1]."""
    return round(2.0 * min(alpha_L, alpha_R) / (alpha_L + alpha_R + _EPS), 3)
