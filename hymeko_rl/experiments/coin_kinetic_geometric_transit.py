"""R10 Stage 1A — geometric-approach transit gates: HOME_V1_GENERIC -> READY, collision-free, analytic, no RL / no BC.

Rolls the explicit closed-form branch-continuous planar-IK transit from the retained generic home ``V1`` to the
pre-capture READY shell (tips on their assigned sides, 15 mm surface margin), through the frozen governed servo, and
checks the transit gate ladder. Also exercises the load-bearing validation as negative controls: (a) a transit that
starts from the singular V2 posture is *rejected*; (b) a forced over-the-top (far-side) left route is *rejected*
(the coin's far edge is beyond the arm's reach). Downstream frozen; no RL; no state edit.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_geometric_transit``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _fingertip_geoms
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.home_states import (
    HOME_STATE_V1_GENERIC, HOME_STATE_V2_READY, build_home_snapshot)
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga

OUT = Path("reports/2026-07-28-planar-geometric-approach-transit")


def _tip_polar(snapshot: Any, coin: np.ndarray, gl: int, gr: int) -> "tuple[float, float, float, float]":
    """(dist_L, side_deg_L, dist_R, side_deg_R) of the two fingertips around the coin, from a snapshot's live state."""
    data = snapshot.branch().inner.data
    tl, tr = data.geom_xpos[gl][:2], data.geom_xpos[gr][:2]
    dl, dr = float(np.linalg.norm(tl - coin)), float(np.linalg.norm(tr - coin))
    al = float(np.degrees(np.arctan2(*(tl - coin)[::-1])))
    ar = float(np.degrees(np.arctan2(*(tr - coin)[::-1])))
    return dl, al, dr, ar


def _handoff_deterministic(snapshot: Any) -> bool:
    """The READY snapshot is a deterministic handoff: two branches produce bit-identical qpos/qvel/prev_tau."""
    a, b = snapshot.branch().inner.data, snapshot.branch().inner.data
    return bool(np.array_equal(a.qpos, b.qpos) and np.array_equal(a.qvel, b.qvel)
                and np.array_equal(np.asarray(snapshot.prev_tau), np.asarray(snapshot.prev_tau)))


def _reaches_ready(dl: float, al: float, dr: float, ar: float, tgt: pga.CoinStraddleTargets) -> bool:
    """Both tips land in the READY band: within +/-10 mm of the shell radius and +/-15 deg of the assigned side."""
    dist_ok = abs(dl - tgt.shell_dist) <= 0.010 and abs(dr - tgt.shell_dist) <= 0.010
    side_ok = abs(al - tgt.side_left_deg) <= 15.0 and abs(ar - tgt.side_right_deg) <= 15.0
    return bool(dist_ok and side_ok)


def _negative_controls(home: Any, tgt: pga.CoinStraddleTargets, cfg: pga.TransitConfig) -> dict:
    """Validation is load-bearing: the singular V2 start is rejected, and the over-the-top left route is tip-infeasible.

    The over-the-top (dir = +1) left arc drives the tip past the coin's far edge, which sits at ~0.316 m from the left
    base — beyond the 0.30 m reach — so it must fail the non-singular in-annulus test.
    """
    v2_rejected = _expect_infeasible(lambda: pga.execute_transit(home, HOME_STATE_V2_READY.q, tgt, cfg))
    arm_l, _ = pga.build_arms(home, tgt.coin)
    gl, gr = _fingertip_geoms(home.branch().inner.model)
    _, home_tip_l, _ = pga.make_fk(home, tgt.coin, gl, gr)(HOME_STATE_V1_GENERIC.q)
    far_path = pga._arc_waypoints(home_tip_l, tgt.precontact().tip_left, tgt.coin, tgt.shell_dist, +1.0,
                                  cfg.n_waypoints)
    far_rejected = not pga._tip_feasible(far_path, tgt.coin, arm_l, cfg)
    return {"singular_v2_start_rejected": v2_rejected, "far_side_left_route_rejected": far_rejected}


def _expect_infeasible(thunk: Any) -> bool:
    try:
        thunk()
        return False
    except pga.TransitInfeasible:
        return True


def run(out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    coin = _coin_xy(cradle.branch())
    home = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    gl, gr = _fingertip_geoms(home.branch().inner.model)
    cfg, tgt = pga.TransitConfig(), pga.CoinStraddleTargets(coin=coin)

    res = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, tgt, cfg)
    dl, al, dr, ar = _tip_polar(res.ready_snapshot, coin, gl, gr)
    gates = {
        "SINGLE_BRANCH_MAINTAINED_PASS": bool(res.single_branch),
        "COIN_UNPERTURBED_DURING_TRANSIT_PASS": bool(res.coin_pert_mm <= 1.0 and res.contacts == 0),
        "TRANSIT_REACHES_READY_PASS": _reaches_ready(dl, al, dr, ar, tgt),
        "READY_HANDOFF_DETERMINISTIC_PASS": _handoff_deterministic(res.ready_snapshot),
    }
    gates["HOME_V1_TO_READY_COLLISION_FREE_PASS"] = bool(all(gates.values()))
    negatives = _negative_controls(home, tgt, cfg)
    verdict = ("HOME_V1_TO_READY_COLLISION_FREE_TRANSIT_PASS"
               if gates["HOME_V1_TO_READY_COLLISION_FREE_PASS"] and all(negatives.values())
               else "GEOMETRIC_TRANSIT_NEEDS_WORK")
    summary = {
        "contract": "PLANAR_GEOMETRIC_APPROACH_V1", "immutable_downstream": "d55f5017",
        "home": HOME_STATE_V1_GENERIC.name, "home_q": HOME_STATE_V1_GENERIC.q.tolist(),
        "ready_shell": {"shell_dist_m": tgt.shell_dist, "side_left_deg": tgt.side_left_deg,
                        "side_right_deg": tgt.side_right_deg},
        "transit": {"mode": res.mode, "single_branch": res.single_branch, "coin_pert_mm": res.coin_pert_mm,
                    "min_clearance_mm": res.min_clearance_mm, "contacts": res.contacts, "peak_qvel": res.peak_qvel,
                    "reached_qerr": round(res.reached_qerr, 4),
                    "ready_tips": {"dist_L_mm": round(dl * 1000, 1), "side_L_deg": round(al, 1),
                                   "dist_R_mm": round(dr * 1000, 1), "side_R_deg": round(ar, 1)}},
        "gates": gates, "negative_controls": negatives, "verdict": verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometric_transit.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    print(f"\nGEOMETRIC TRANSIT: {r['verdict']}")
    t = r["transit"]
    print(f"  mode {t['mode']} | coin_pert {t['coin_pert_mm']}mm | min_clr {t['min_clearance_mm']}mm | "
          f"contacts {t['contacts']} | branch {t['single_branch']} | qerr {t['reached_qerr']}")
    print(f"  READY tips: L {t['ready_tips']['dist_L_mm']}mm @ {t['ready_tips']['side_L_deg']}deg, "
          f"R {t['ready_tips']['dist_R_mm']}mm @ {t['ready_tips']['side_R_deg']}deg")
    for k, v in r["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'} {k}")
    print(f"  negative controls: {r['negative_controls']}")
