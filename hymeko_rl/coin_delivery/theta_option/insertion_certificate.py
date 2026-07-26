"""TASK CONTACT-LEGALITY certificate for the coin delivery — separate from the PHYSICAL collision contract.

The physical model is realistic: every arm geom collides with the coin (see `collision_contract` + `contact_legality`).
The *task* then decides what a legitimate solution looks like — and the forbidden shortcut is NOT link contact, it is a
BALLISTIC KNOCK. Link contact (morphology-assisted guiding) is allowed; a single uncontrolled high-speed hit is not.

    CONTROLLED_INSERTION_PASS = target-directed displacement
                                AND bounded coin speed (motion contract)
                                AND active braking → low terminal speed (no fly-through)
                                AND terminal K6 dwell
                                AND NOT a ballistic knock

Delivery levels (a ladder, not a pass/fail):
    E0  WHOLE_ARM_ASSISTED_INSERTION  — link contact allowed; the current teacher (fingertip impulse share ≈ 0.06).
    E1  FINGERTIP_DOMINANT_DELIVERY   — the majority of the useful contact impulse comes from the fingertips.
    E2  FINGERTIP_ONLY_DELIVERY       — any non-fingertip coin contact invalidates the delivery.

This module only GRADES; it never disables physics. The fingertip vs arm-body impulse split is measured with the
existing `contact_legality.classify_contacts` (no re-implementation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import SETTLE_VEL
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.env.contact_legality import ContactLegalitySpec, classify_contacts, contact_force_magnitude
from hymeko_rl.env.motion_contract import MotionLimits

FINGERTIP_DOMINANT_SHARE = 0.5           # E1: fingertip impulse share must exceed this


def is_ballistic_knock(m: dict[str, Any], limits: MotionLimits = MotionLimits()) -> bool:
    """A ballistic knock = a single uncontrolled high-speed strike: the coin exceeds the hard speed cap, OR it is driven
    fast (> 3×SETTLE_VEL) yet never settles into the K6 dwell (a fly-through, not a controlled insertion)."""
    return bool(m["peak_coin_speed"] > limits.ee_speed_hard
                or (m["peak_coin_speed"] > 3.0 * SETTLE_VEL and not m["k6_delivered"]))


def controlled_insertion_pass(m: dict[str, Any], limits: MotionLimits = MotionLimits()) -> bool:
    """The CONTROLLED_INSERTION certificate over a `rollout_primitive` metrics dict. Link contact is allowed; the
    forbidden shortcut (ballistic knock) is rejected. # Postconditions: True ⇒ target-directed, speed-bounded, braked,
    K6-delivered, non-ballistic."""
    return bool(m["forward"] > 0.0
                and m["peak_coin_speed"] <= limits.ee_speed_hard
                and m["peak_qdot"] <= limits.joint_vel_hard
                and m["terminal_coin_speed"] < SETTLE_VEL
                and m["k6_delivered"]
                and not is_ballistic_knock(m, limits))


def _legality_spec(model: Any, disk: int) -> ContactLegalitySpec:
    left, right = set(), set()
    for b in range(int(model.nbody)):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.endswith("_left"):
            left.add(b)
        elif nm.endswith("_right"):
            right.add(b)
    return ContactLegalitySpec.from_model(model, object_geoms={disk}, arm_bodies_left=frozenset(left),
                                          arm_bodies_right=frozenset(right))


def contact_impulse_share(snap: CradleSnapshot, theta: Any, cfg: Any = DELIVERY_CFG) -> dict[str, Any]:
    """Roll ``theta`` through the frozen option and accumulate, per step, the coin's contact impulse split into fingertip
    vs arm-body (non-tip) using `contact_legality.classify_contacts`. Returns the impulses, the fingertip share, and the
    per-frame contact counts. # Postconditions: shares in [0,1] (or None if no coin contact)."""
    rl0 = snap.branch()
    m0 = rl0.inner.model
    disk = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    spec = _legality_spec(m0, disk)
    acc = {"ft": 0.0, "arm": 0.0, "frames": 0, "ft_frames": 0, "arm_frames": 0}

    def hook(rl: Any, _t: int) -> None:
        mm, dd = rl.inner.model, rl.inner.data
        st = classify_contacts(mm, dd, spec)
        ft = 0.0
        for i in range(int(dd.ncon)):
            c = dd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if disk in (g1, g2) and (g2 if g1 == disk else g1) in spec.fingertip_geoms:
                ft += contact_force_magnitude(mm, dd, i)
        acc["ft"] += ft
        acc["arm"] += st.arm_body_contact_impulse
        acc["frames"] += 1
        acc["ft_frames"] += int(st.fingertip_contact)
        acc["arm_frames"] += int(st.arm_body_contact)

    m = rollout_primitive(snap, tuple(np.asarray(theta, np.float64)), cfg, frame_hook=hook)
    tot = acc["ft"] + acc["arm"]
    share = (acc["ft"] / tot) if tot > 1e-9 else None
    return {"metrics": m, "fingertip_impulse": round(acc["ft"], 3), "arm_body_impulse": round(acc["arm"], 3),
            "fingertip_impulse_share": (round(share, 4) if share is not None else None),
            "arm_body_contact_frames": acc["arm_frames"], "fingertip_contact_frames": acc["ft_frames"],
            "n_frames": acc["frames"]}


@dataclass(frozen=True)
class InsertionGrade:
    """The graded verdict on a delivery: the CONTROLLED_INSERTION certificate, the ballistic-knock flag, the fingertip
    impulse share, and the delivery LEVEL (E0/E1/E2)."""

    controlled_insertion: bool
    ballistic_knock: bool
    fingertip_impulse_share: "float | None"
    level: str
    k6_delivered: bool
    peak_coin_speed: float
    terminal_coin_speed: float
    forward_mm: float


def _level(share: "float | None", ftonly_ok: bool) -> str:
    if share is None:
        return "NO_COIN_CONTACT"
    if ftonly_ok:
        return "E2_FINGERTIP_ONLY_DELIVERY"
    if share >= FINGERTIP_DOMINANT_SHARE:
        return "E1_FINGERTIP_DOMINANT_DELIVERY"
    return "E0_WHOLE_ARM_ASSISTED_INSERTION"


def grade_delivery(snap: CradleSnapshot, theta: Any, cfg: Any = DELIVERY_CFG) -> InsertionGrade:
    """Grade a delivery: run the option, measure the impulse split, and apply the CONTROLLED_INSERTION certificate + the
    E0/E1/E2 level. E2 (fingertip-only) requires ZERO arm-body contact frames; E1 requires fingertip impulse share ≥
    ``FINGERTIP_DOMINANT_SHARE``; else E0 (whole-arm assisted, link contact allowed)."""
    q = contact_impulse_share(snap, theta, cfg)
    m = q["metrics"]
    ftonly = q["arm_body_contact_frames"] == 0 and (q["fingertip_impulse_share"] or 0.0) > 0.0
    return InsertionGrade(
        controlled_insertion=controlled_insertion_pass(m), ballistic_knock=is_ballistic_knock(m),
        fingertip_impulse_share=q["fingertip_impulse_share"], level=_level(q["fingertip_impulse_share"], ftonly),
        k6_delivered=bool(m["k6_delivered"]), peak_coin_speed=round(m["peak_coin_speed"], 4),
        terminal_coin_speed=round(m["terminal_coin_speed"], 4), forward_mm=round(m["forward"] * 1000, 2))
