"""EXACT_ZERO_HOME honest full chain — multi-goal task-space fingertip planner from q=[0,0,0,0] to strict K6.

The robot starts EXACTLY at ``q=[0,0,0,0]``, ``qdot=0`` (both 2R arms fully extended). We plan in CARTESIAN fingertip
space with a per-arm THREE-CONCENTRIC-SHELL multi-goal plan around the coin centre C:

    outer approach circle (r_outer) -> middle alignment circle (r_middle) -> inner pre-straddle circle (r_inner)

Each arm spirals in ON ITS OWN SIDE of the coin (the fingertip angle sweeps from the tip's start angle to its straddle
angle, in the direction that keeps every waypoint clear of and never across the coin). Joint angles come from the exact
``PlanarArm2R.ik_cont`` (the elbow bend emerges from IK — never hand-written). Each candidate spiral is validated for
fingertip keep-out, link keep-out, joint limits and branch continuity; we execute the best one continuously through the
governed ``step_ablation`` stack (slow, so the coin is untouched before intended contact), then re-solve the capture for
the reached READY and test strict K6. First frame is exactly q=[0,0,0,0]; no snapshot teleport, no staging pose.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import HomeState, build_home_snapshot
from hymeko_rl.env.motion_contract import govern_torque
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

R_OUTER, R_MIDDLE, R_INNER = 0.120, 0.085, 0.055     # three concentric coin-shell radii (m); inner = straddle shell
N_DESCENT, N_SPIRAL = 40, 110                        # waypoints on the radial descent and the spiral-in


def _angle(p: np.ndarray, coin: np.ndarray) -> float:
    d = p - coin
    return float(np.arctan2(d[1], d[0]))


def _shell(coin: np.ndarray, r: float, a: float) -> np.ndarray:
    return coin + r * np.array([np.cos(a), np.sin(a)])


def _spiral_path(start_tip: np.ndarray, a_straddle: float, coin: np.ndarray, direction: float) -> np.ndarray:
    """Radial descent from the (far) start tip to the outer shell at its own angle, then a spiral sweeping the angle to
    ``a_straddle`` (in ``direction``) while the radius shrinks outer -> inner. Every point is >= R_INNER from the coin."""
    a0 = _angle(start_tip, coin)
    d_ang = float(np.arctan2(np.sin(a_straddle - a0), np.cos(a_straddle - a0)))     # wrapped shortest
    if direction > 0 and d_ang < 0:
        d_ang += 2 * np.pi
    if direction < 0 and d_ang > 0:
        d_ang -= 2 * np.pi
    r0 = float(np.linalg.norm(start_tip - coin))
    descent = [_shell(coin, r0 + f * (R_OUTER - r0), a0) for f in np.linspace(0, 1, N_DESCENT)]
    rs = np.concatenate([np.linspace(R_OUTER, R_MIDDLE, N_SPIRAL // 2), np.linspace(R_MIDDLE, R_INNER, N_SPIRAL // 2)])
    angs = np.linspace(a0, a0 + d_ang, N_SPIRAL)
    spiral = [_shell(coin, r, a) for r, a in zip(rs, angs)]
    return np.array(descent + spiral)


def _plan_arm(start_tip: np.ndarray, a_straddle: float, coin: np.ndarray, arm: pga.PlanarArm2R,
              cfg: pga.TransitConfig) -> "np.ndarray | None":
    """Best validated (both sweep directions) three-shell spiral, solved to joints by exact continuous IK from q=[0,0]."""
    best, best_cost = None, 1e9
    for direction in (1.0, -1.0):
        path = _spiral_path(start_tip, a_straddle, coin, direction)
        if np.linalg.norm(path - coin, axis=1).min() < cfg.keepout:              # fingertip keep-out (hard)
            continue
        traj = pga._solve_arm_traj(path, arm, np.zeros(2))                        # exact ik_cont per waypoint
        if np.abs(traj).max() > cfg.joint_hi or np.abs(np.diff(traj, axis=0)).max() > 0.3:   # limits + branch continuity
            continue
        if not pga._links_clear(traj, arm, coin, cfg):                           # link capsules clear of the coin
            continue
        cost = float(np.abs(np.diff(traj, axis=0)).sum())                        # total joint sweep (shorter is better)
        if cost < best_cost:
            best, best_cost = traj, cost
    return best


def _servo(rl: Any, qref: np.ndarray, stack: Any, lo: np.ndarray, hi: np.ndarray, cfg: pga.TransitConfig,
           coin: np.ndarray, gl: int, gr: int, frame_hook: Any = None) -> "tuple[np.ndarray, float, float]":
    """Smooth continuous tracking of the multi-goal joint path; slow enough (many substeps) that momentum stays small and
    the tips hug the collision-free spiral. Records min fingertip-coin clearance and coin displacement. ``frame_hook(rl,
    i)`` (optional) is called READ-ONLY after each waypoint for visualisation."""
    slew = float(stack.tau_rate * stack.control_dt)
    prev = np.zeros(4)
    minclr, pert = 9.0, 0.0
    mujoco.set_mjcb_control(lambda _m, d: d.ctrl.__setitem__(slice(0, 4), govern_torque(d.ctrl[:4], d.qvel[:4], stack.gov)))
    try:
        for i in range(len(qref) + cfg.hold_steps):
            target = qref[min(i, len(qref) - 1)]
            for _ in range(cfg.substeps):
                d = rl.inner.data
                dtau = np.clip(cfg.servo_kp * (target - d.qpos[:4]) - cfg.servo_kd * d.qvel[:4], -slew, slew)
                prev = np.clip(prev + dtau, lo, hi)
                step_ablation(rl, np.asarray(prev, np.float32), "A")
                cn = _coin_xy(rl)
                minclr = min(minclr, float(np.linalg.norm(d.geom_xpos[gl][:2] - cn)),
                             float(np.linalg.norm(d.geom_xpos[gr][:2] - cn)))
                pert = max(pert, float(np.linalg.norm(cn - coin)))
            if frame_hook is not None:
                frame_hook(rl, i)
    finally:
        mujoco.set_mjcb_control(None)
    return prev, minclr, pert


def _home_with_coin(rig: dict, coin_xy: "np.ndarray | None") -> "tuple[Any, np.ndarray]":
    """The exact zero home ``q=[0,0,0,0]`` with the coin placed at ``coin_xy`` (canonical if None). Returns (home, coin)."""
    stack = rig["cradle"].stack
    home = build_home_snapshot(rig["cradle"], HomeState(name="ZERO", q=np.zeros(4), mode="ZERO", description="true zero"))
    if coin_xy is None:
        return home, _coin_xy(home.branch())
    rl = home.branch()
    rl.inner.data.qpos[4:6] = np.asarray(coin_xy, float)
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    return kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, np.zeros(4)), np.asarray(coin_xy, float)


def do_reach(rig: dict, cfg: pga.TransitConfig, frame_hook: Any = None, coin_xy: "np.ndarray | None" = None) -> "dict | None":
    """The task-space 3-shell IK reach from the exact zero home. Returns the straddle READY snapshot, the reach metrics,
    and the pieces the caller needs (or None if no spiral validates). ``frame_hook`` renders the reach; ``coin_xy`` places
    the coin at an arbitrary workspace location (the reach + straddle adapt automatically)."""
    stack = rig["cradle"].stack
    home, coin = _home_with_coin(rig, coin_xy)
    arm_l, arm_r = pga.build_arms(home, coin)
    rl = home.branch()
    gl, gr = pga._fingertip_geoms(rl.inner.model)
    lo, hi = np.asarray(home.lo, float), np.asarray(home.hi, float)
    straddle = pga.CoinStraddleTargets(coin=coin).precontact()
    tl0, tr0 = rl.inner.data.geom_xpos[gl][:2].copy(), rl.inner.data.geom_xpos[gr][:2].copy()
    traj_l = _plan_arm(tl0, _angle(straddle.tip_left, coin), coin, arm_l, cfg)
    traj_r = _plan_arm(tr0, _angle(straddle.tip_right, coin), coin, arm_r, cfg)
    if traj_l is None or traj_r is None:
        return None
    qref = np.concatenate([traj_l, traj_r], axis=1)
    prev, minclr, pert = _servo(rl, qref, stack, lo, hi, cfg, coin, gl, gr, frame_hook=frame_hook)
    d = rl.inner.data
    con = primary_fingertip_contacts(rl)
    metrics = {"first_frame_q": [round(float(x), 4) for x in home.branch().inner.data.qpos[:4]],
               "reach_qerr": round(float(np.linalg.norm(d.qpos[:4] - qref[-1])), 4),
               "min_tip_coin_clr_mm": round(minclr * 1000, 1), "coin_moved_before_capture_mm": round(pert * 1000, 2),
               "reach_contacts": int(con["left"] is not None) + int(con["right"] is not None),
               "tip_l_err_mm": round(float(np.linalg.norm(d.geom_xpos[gl][:2] - straddle.tip_left)) * 1000, 1),
               "tip_r_err_mm": round(float(np.linalg.norm(d.geom_xpos[gr][:2] - straddle.tip_right)) * 1000, 1)}
    ready = kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, prev.copy())
    return {"ready": ready, "metrics": metrics, "coin": coin, "n_reach_waypoints": len(qref)}


def solve_capture(rig: dict, ready: Any, seed: int = 0) -> Any:
    """CEM re-solve of the capture for the reached READY; returns the best :class:`CaptureResult` (params + outcome)."""
    best = None
    for sd in range(seed, seed + 5):
        res = mp.plan_capture(ready, rig["ref"], rig["stack"], rig["down"], seed=sd)
        if best is None or (res.outcome.k6, -res.outcome.min_dtz_mm) > (best.outcome.k6, -best.outcome.min_dtz_mm):
            best = res
        if res.outcome.k6 and res.outcome.min_dtz_mm < 10:
            break
    return best


def reach_and_deliver(seed: int = 0) -> dict:
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)   # transit gains (kp12/kd4), gentler + settle
    r = do_reach(rig, cfg)
    if r is None:
        return {"reached": False, "reason": "no validated spiral"}
    o = solve_capture(rig, r["ready"], seed).outcome
    return {"reached": True, **r["metrics"], "capture_k6": bool(o.k6), "min_dtz_mm": round(o.min_dtz_mm, 2),
            "safe": bool(o.safe)}


if __name__ == "__main__":
    r = reach_and_deliver()
    print("ZERO-HOME 3-shell reach:", r)
    if r.get("capture_k6"):
        print(f"=> EXACT_ZERO_HOME [0,0,0,0] -> 3-shell IK reach -> re-solved capture -> STRICT K6 {r['min_dtz_mm']}mm "
              f"| coin untouched to {r['coin_moved_before_capture_mm']}mm, tip clr {r['min_tip_coin_clr_mm']}mm")
