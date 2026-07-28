"""Robust zero-home reach via RRT-Connect in the 4-DOF joint configuration space.

Replaces the hand-tuned 3-shell spiral (which was fragile — some coin positions found no valid arc) with a proper
sampling-based motion planner. From the exact zero home ``q=[0,0,0,0]`` to a collision-free straddle configuration (exact
2R IK of the coin-shell targets), RRT-Connect grows two trees and joins them; every edge is collision-checked against the
coin (link + fingertip keep-out) and inter-arm collision. The path is then shortcut-smoothed, densely interpolated, and
executed through the same governed servo. Generalises over the coin position: a valid reach is found for any reachable
coin, without per-coin tuning. (The downstream delivery is a separate matter — the learned R2 / handoff orbit are still
coin-specific.)
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.coin_zero_home_reach import _home_with_coin, _servo, solve_capture

INTER_ARM = 0.03            # min inter-arm segment clearance (m)
Q_LO, Q_HI = -3.2, 3.2      # joint sampling bounds (rad)


def _seg_seg_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Min distance between 2D segments ab and cd (0 if they cross)."""
    def _o(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    if (_o(a, b, c) * _o(a, b, d) < 0) and (_o(c, d, a) * _o(c, d, b) < 0):
        return 0.0
    return min(pga._seg_point_dist(a, b, c), pga._seg_point_dist(a, b, d),
               pga._seg_point_dist(c, d, a), pga._seg_point_dist(c, d, b))


def _collision_free(q: np.ndarray, coin: np.ndarray, arm_l: pga.PlanarArm2R, arm_r: pga.PlanarArm2R,
                    cfg: pga.TransitConfig) -> bool:
    """A 4-DOF config is valid iff both arms clear the coin (link + fingertip keep-out) and each other, within limits."""
    if float(np.abs(q).max()) > cfg.joint_hi:
        return False
    bl, el, tl = arm_l.link_points(q[:2])
    br, er, tr = arm_r.link_points(q[2:])
    for b, e, t in ((bl, el, tl), (br, er, tr)):
        if float(np.linalg.norm(t - coin)) < cfg.keepout:
            return False
        if min(pga._seg_point_dist(b, e, coin), pga._seg_point_dist(e, t, coin)) < cfg.link_keepout:
            return False
    left, right = ((bl, el), (el, tl)), ((br, er), (er, tr))
    return all(_seg_seg_dist(*ls, *rs) >= INTER_ARM for ls in left for rs in right)


def _edge_free(q0: np.ndarray, q1: np.ndarray, coll: Callable[[np.ndarray], bool], res: float = 0.05) -> bool:
    n = max(2, int(np.ceil(float(np.linalg.norm(q1 - q0)) / res)))
    return all(coll(q0 + (q1 - q0) * t) for t in np.linspace(0, 1, n))


def _straddle_goal_set(coin: np.ndarray, arm_l: pga.PlanarArm2R, arm_r: pga.PlanarArm2R, base: Any,
                       coll: Callable[[np.ndarray], bool]) -> list:
    """The GOAL SET G(coin): collision-free straddle configs from sampled precontact point-pairs (assigned-side angles
    +/- offsets x shell clearances) with exact 2R IK on both elbow branches. Never a single hand-picked READY q."""
    goals = []
    for da in (0.0, 12.0, -12.0, 24.0, -24.0):
        for shell in (0.055, 0.060, 0.065):
            tgt = pga.CoinStraddleTargets(coin=coin, shell_dist=shell, side_left_deg=base.side_left_deg + da,
                                          side_right_deg=base.side_right_deg - da).precontact()
            for bl in (1.0, -1.0):
                for br in (1.0, -1.0):
                    q = np.concatenate([arm_l.ik(tgt.tip_left, bl), arm_r.ik(tgt.tip_right, br)])
                    if coll(q):
                        goals.append(q)
    return goals


def _steer(frm: np.ndarray, to: np.ndarray, step: float) -> np.ndarray:
    v = to - frm
    n = float(np.linalg.norm(v))
    return to.copy() if n <= step else frm + v * (step / n)


def _nearest(tree: list, q: np.ndarray) -> int:
    return int(np.argmin([float(np.linalg.norm(q - n)) for n in tree]))


def _extend(tree: list, par: dict, target: np.ndarray, coll: Callable[[np.ndarray], bool], step: float) -> "tuple[str, int]":
    """Standard RRT extend: REACHED (== target), ADVANCED (one step), or TRAPPED (blocked)."""
    j = _nearest(tree, target)
    qn = _steer(tree[j], target, step)
    if not (coll(qn) and _edge_free(tree[j], qn, coll)):
        return "TRAPPED", j
    tree.append(qn)
    par[len(tree) - 1] = j
    return ("REACHED" if float(np.linalg.norm(qn - target)) < 1e-9 else "ADVANCED"), len(tree) - 1


def _path(tree: list, par: dict, k: int) -> list:
    out = []
    while k != -1:
        out.append(tree[k])
        k = par[k]
    return out[::-1]


def rrt_connect(start: np.ndarray, goals: list, coll: Callable[[np.ndarray], bool], rng: np.random.Generator,
                *, step: float = 0.25, max_iter: int = 4000, goal_bias: float = 0.15) -> "np.ndarray | None":
    """Bidirectional RRT-Connect over the 4-DOF joint box to a GOAL SET. Tree B is seeded with every goal config; the
    planner connects the start tree to any of them. Returns a collision-free waypoint path, or None."""
    ta, pa = [start.copy()], {0: -1}
    tb, pb = [g.copy() for g in goals], {i: -1 for i in range(len(goals))}
    for _ in range(max_iter):
        q_rand = goals[int(rng.integers(len(goals)))] if rng.random() < goal_bias else rng.uniform(Q_LO, Q_HI, 4)
        s, ka = _extend(ta, pa, q_rand, coll, step)
        if s != "TRAPPED":
            target = ta[ka]
            while True:                                       # CONNECT tree B all the way to the new node
                sb, kb = _extend(tb, pb, target, coll, step)
                if sb == "REACHED":
                    fwd, bwd = _path(ta, pa, ka), _path(tb, pb, kb)
                    joined = fwd + bwd[::-1]
                    return joined if np.array_equal(joined[0], start) else joined[::-1]
                if sb == "TRAPPED":
                    break
        ta, pa, tb, pb = tb, pb, ta, pa                       # swap the trees
    return None


def _shortcut(path: list, coll: Callable[[np.ndarray], bool], rng: np.random.Generator, rounds: int = 200) -> list:
    p = [q.copy() for q in path]
    for _ in range(rounds):
        if len(p) <= 2:
            break
        i, j = sorted(rng.integers(0, len(p), 2))
        if j - i >= 2 and _edge_free(p[i], p[j], coll):
            p = p[:i + 1] + p[j:]
    return p


def _densify(path: list, res: float = 0.03) -> np.ndarray:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / res)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n)[1:])
    return np.array(out)


def reach_rrt(rig: dict, cfg: pga.TransitConfig, coin_xy: "np.ndarray | None" = None, seed: int = 0,
              frame_hook: Any = None) -> "dict | None":
    """RRT-Connect reach from the exact zero home to the straddle, then the re-solved capture; returns metrics + outcome."""
    stack = rig["cradle"].stack
    home, coin = _home_with_coin(rig, coin_xy)
    arm_l, arm_r = pga.build_arms(home, coin)
    rl = home.branch()
    gl, gr = pga._fingertip_geoms(rl.inner.model)
    lo, hi = np.asarray(home.lo, float), np.asarray(home.hi, float)
    straddle = pga.CoinStraddleTargets(coin=coin).precontact()
    rng = np.random.default_rng(seed)

    def coll(q: np.ndarray) -> bool:
        return _collision_free(q, coin, arm_l, arm_r, cfg)

    goals = _straddle_goal_set(coin, arm_l, arm_r, pga.CoinStraddleTargets(coin=coin), coll)
    if not goals or not coll(np.zeros(4)):
        return None
    raw = rrt_connect(np.zeros(4), goals, coll, rng)
    if raw is None:
        return None
    qref = _densify(_shortcut(raw, coll, rng))

    prev, minclr, pert = _servo(rl, qref, stack, lo, hi, cfg, coin, gl, gr, frame_hook=frame_hook)
    d = rl.inner.data
    metrics = {"first_frame_q": [round(float(x), 4) for x in home.branch().inner.data.qpos[:4]],
               "rrt_waypoints": len(raw), "path_waypoints": len(qref),
               "reach_qerr": round(float(np.linalg.norm(d.qpos[:4] - qref[-1])), 4),
               "min_tip_coin_clr_mm": round(minclr * 1000, 1), "coin_moved_before_capture_mm": round(pert * 1000, 2),
               "tip_l_err_mm": round(float(np.linalg.norm(d.geom_xpos[gl][:2] - straddle.tip_left)) * 1000, 1),
               "tip_r_err_mm": round(float(np.linalg.norm(d.geom_xpos[gr][:2] - straddle.tip_right)) * 1000, 1)}
    ready = kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, prev.copy())
    o = solve_capture(rig, ready, 0).outcome
    return {"ready": ready, "coin": coin, "metrics": metrics,
            "capture_k6": bool(o.k6), "min_dtz_mm": round(o.min_dtz_mm, 2), "safe": bool(o.safe)}


if __name__ == "__main__":
    import dataclasses
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    canon = np.array([0.07578387, 0.14279214])
    for name, c in [("canonical", canon), ("left -3cm", canon + [-0.03, 0]), ("right +3cm", canon + [0.03, 0]),
                    ("down -2cm", canon + [0, -0.02]), ("up +2cm", canon + [0, 0.02])]:
        r = reach_rrt(rig, cfg, coin_xy=c)
        if r is None:
            print(f"{name:12s} coin={np.round(c, 3)} -> RRT: no path / no straddle", flush=True)
        else:
            m = r["metrics"]
            print(f"{name:12s} coin={np.round(c, 3)} -> RRT reach: wp {m['rrt_waypoints']}->{m['path_waypoints']} "
                  f"clr={m['min_tip_coin_clr_mm']}mm coin_moved={m['coin_moved_before_capture_mm']}mm "
                  f"tipL/R={m['tip_l_err_mm']}/{m['tip_r_err_mm']}mm | capture K6={r['capture_k6']} "
                  f"dtz={r['min_dtz_mm']}mm", flush=True)
    print("RRT_SWEEP_DONE")
