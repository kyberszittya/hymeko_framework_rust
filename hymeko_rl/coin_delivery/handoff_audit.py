"""R11.6D causal handoff audit + counterfactual ablation.

The R11.6C dev far-angle failures are IN-SUPPORT (a near bank demo exists) yet the retrieved theta undershoots by ~30mm.
This module tests, causally, WHICH handoff-state component governs theta-transportability: it reads the audit variables
from a capture handoff, and builds a MODIFIED handoff snapshot with ONE state component replaced by another handoff's
value (the counterfactual s_dev[c <- s_bank]), so the SAME retrieved theta can be rolled from bank / dev / dev-with-swap.

State layout (the coin is three DECOUPLED scalar joints, not a free body): qpos[0:4]=arm, qpos[4],[5]=coin x,y,
qpos[6]=coin yaw; qvel[0:4]=arm, qvel[4],[5]=coin lin-vel, qvel[6]=coin spin. Coin-state / zone swaps MUST refresh the
cached planar metrics after mj_forward (rollout_primitive anchors its metrics on that cache).
"""
from __future__ import annotations

import copy
import dataclasses
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
from hymeko_rl.coin_delivery.delivery_bc.evaluate import CLOSED_LOOP_CFG
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga

_CX, _CY, _CYAW = 4, 5, 6         # coin qpos indices (x, y, yaw); qvel[4],[5]=lin-vel, qvel[6]=spin


class SwapComponent(Enum):
    """One independently-settable handoff component to replace with the bank value (the counterfactual axes)."""

    COIN_POS = "coin_pos"          # qpos[4:6]
    COIN_YAW = "coin_yaw"          # qpos[6]
    COIN_LINVEL = "coin_linvel"    # qvel[4:6]
    COIN_SPIN = "coin_spin"        # qvel[6]
    ARM_QPOS = "arm_qpos"          # qpos[0:4]
    ARM_QVEL = "arm_qvel"          # qvel[0:4]
    PREV_TAU = "prev_tau"          # snapshot.prev_tau
    ZONE = "zone"                  # target zone (required-transport test)


_NEEDS_METRICS = {SwapComponent.COIN_POS, SwapComponent.COIN_YAW, SwapComponent.COIN_LINVEL,
                  SwapComponent.COIN_SPIN, SwapComponent.ZONE}


def _jac_cond(rl: Any, dofs: "tuple[int, int]", geom: int) -> float:
    """Condition number of the 2x2 planar fingertip Jacobian for one arm (manipulability at the handoff)."""
    m, d = rl.inner.model, rl.inner.data
    jacp = np.zeros((3, m.nv), np.float64)
    mujoco.mj_jacGeom(m, d, jacp, None, geom)
    return round(float(np.linalg.cond(jacp[:2][:, list(dofs)])), 3)


def read_audit(snap: Any) -> "dict[str, Any]":
    """The Phase-1 audit variables at a handoff: coin pose/vel/spin, target-relative velocity, contact distribution,
    grasp symmetry, Jacobian conditioning, zone. Pure read (branches, never mutates ``snap``)."""
    rl = snap.branch()
    d = rl.inner.data
    coin = np.asarray(_coin_xy(rl), np.float64)
    cv = np.asarray(rl.inner._planar_metrics.disk_vel[:2], np.float64)
    u, dtz = rl.inner.direction_to_zone()
    u = np.asarray(u, np.float64)
    tang = np.array([-u[1], u[0]], np.float64)
    con = primary_fingertip_contacts(rl)

    def fn(side: str) -> float:
        return round(float(con[side]["fn"]), 4) if con[side] else 0.0

    straddle = round(float(con["left"]["n"] @ con["right"]["n"]), 4) if (con["left"] and con["right"]) else 0.0
    grasp_angle = 0.0
    if con["left"] and con["right"]:
        axis = np.asarray(con["right"]["x_c"], np.float64) - np.asarray(con["left"]["x_c"], np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        grasp_angle = round(float(np.degrees(np.arccos(np.clip(abs(axis @ u), 0.0, 1.0)))), 2)
    gl, gr = pga._fingertip_geoms(rl.inner.model)
    return {"coin_xy": [round(float(v), 5) for v in coin], "coin_yaw": round(float(d.qpos[_CYAW]), 5),
            "coin_speed": round(float(np.linalg.norm(cv)), 5), "coin_spin": round(float(d.qvel[_CYAW]), 5),
            "dtz_mm": round(float(dtz) * 1000, 2), "coin_vel_forward": round(float(cv @ u), 5),
            "coin_vel_tangential": round(float(cv @ tang), 5), "fn_left": fn("left"), "fn_right": fn("right"),
            "fn_min": round(min(fn("left"), fn("right")), 4), "straddle_dot": straddle,
            "grasp_target_angle_deg": grasp_angle, "arm_qpos": [round(float(v), 5) for v in d.qpos[:4]],
            "arm_qvel": [round(float(v), 5) for v in d.qvel[:4]], "prev_tau": [round(float(v), 5) for v in snap.prev_tau],
            "jac_cond_left": _jac_cond(rl, (0, 1), gl), "jac_cond_right": _jac_cond(rl, (2, 3), gr),
            "zone": [round(float(rl.inner._zone_x), 5), round(float(rl.inner._zone_y), 5)]}


def _apply_swap(rl: Any, other_rl: Any, component: SwapComponent) -> None:
    d, od = rl.inner.data, other_rl.inner.data
    if component is SwapComponent.COIN_POS:
        d.qpos[_CX], d.qpos[_CY] = od.qpos[_CX], od.qpos[_CY]
    elif component is SwapComponent.COIN_YAW:
        d.qpos[_CYAW] = od.qpos[_CYAW]
    elif component is SwapComponent.COIN_LINVEL:
        d.qvel[_CX], d.qvel[_CY] = od.qvel[_CX], od.qvel[_CY]
    elif component is SwapComponent.COIN_SPIN:
        d.qvel[_CYAW] = od.qvel[_CYAW]
    elif component is SwapComponent.ARM_QPOS:
        d.qpos[:4] = od.qpos[:4]
    elif component is SwapComponent.ARM_QVEL:
        d.qvel[:4] = od.qvel[:4]
    elif component is SwapComponent.ZONE:
        rl.inner._zone_x, rl.inner._zone_y = float(other_rl.inner._zone_x), float(other_rl.inner._zone_y)


def swap_component(snap: Any, other: Any, component: SwapComponent) -> Any:
    """A NEW handoff snapshot equal to ``snap`` but with ``component`` replaced by ``other``'s value. ``prev_tau`` is a
    pure field replace; state/zone swaps edit the branched MuJoCo state, ``mj_forward``, and (for coin/zone) refresh the
    cached planar metrics that ``rollout_primitive`` anchors on."""
    if component is SwapComponent.PREV_TAU:
        return dataclasses.replace(snap, prev_tau=np.asarray(other.prev_tau, np.float64).copy())
    rl, other_rl = snap.branch(), other.branch()
    _apply_swap(rl, other_rl, component)
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    if component in _NEEDS_METRICS:
        rl.inner._planar_metrics = rl.inner._metrics()
    return kc.TransportSnapshot.from_live(copy.deepcopy(rl), snap.stack, snap.prev_tau,
                                          joint_vel_hard=snap.joint_vel_hard)


def roll(snap: Any, theta: np.ndarray) -> "dict[str, Any]":
    """Roll the retrieved theta from a handoff; return the strict-K6 verdict + gap/dtz (the transportability outcome)."""
    m = rollout_primitive(snap, clip_theta(np.asarray(theta, np.float64)), CLOSED_LOOP_CFG)
    return {"k6": bool(delivery_success(m, CLOSED_LOOP_CFG)), "dtz_mm": round(float(m["dtz_end"]) * 1000, 2),
            "gap_closed": round(float(m["gap_closed"]), 3),
            "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)}
