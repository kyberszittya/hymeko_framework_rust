"""COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1 — measurement-only analysis primitives (no training, no artifact changes).

A rollout is reduced ONCE to its per-step certificate stream (distance-to-zone, coin speed, fingertip/body contact,
body-only progress, ever-grasped). Every downstream question — re-certification under a threshold grid, the success
ladder, the touched-ever audit — is then computed offline from that stream, so the frozen controllers/artifacts are never
re-run under changed rules.

Physical units (measured): control dt = 10 ms (sim 0.5 ms × frame_skip 20); K dwell steps = K × 10 ms; coin radius
0.02 m; zone radius 0.04 m; dtz ≤ 0.02 ⟺ coin fully contained (margin = zone_half − coin_r = 0.02 m); settle 0.06 m/s
= 0.6 mm per control step.
"""
from __future__ import annotations

import numpy as np

CONTROL_DT = 0.01           # s (0.0005 × 20)
COIN_R = 0.02               # m
ZONE_HALF = 0.04            # m
FULL_CONTAIN = ZONE_HALF - COIN_R   # 0.02 m: dtz ≤ this ⟺ coin fully inside the zone


def recertify(steps, initial_clearance, *, center_tol, settle_vel, dwell_req):
    """Stream a captured CertStep list through a DeliveryCertifier with the given thresholds; return
    (delivered, cert_step, best_dwell). Read-only: the original rollout is untouched."""
    from hymeko_rl.coin_delivery.delivery_certificate import CertStep, DeliveryCertifier, DeliveryThresholds
    th = DeliveryThresholds(center_tol=center_tol, settle_vel=settle_vel, dwell_req=dwell_req)
    cert = DeliveryCertifier(initial_clearance=initial_clearance, th=th)
    for i, s in enumerate(steps):
        cert.update(CertStep(**s))
        if cert.delivery_certified:
            return True, i, cert.best_delivery_dwell
    return False, -1, cert.best_delivery_dwell


def _run_dwell(mask):
    """Longest run of consecutive True in a boolean mask (max held-dwell under a centered+settled predicate)."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def success_ladder(steps, *, entry_tol=0.05, center_tol=0.02, settle_vel=0.06):
    """Full graded description of a rollout — NOT just strict-K6. All from the certificate stream."""
    dtz = np.array([s["disk_to_zone"] for s in steps], float)
    spd = np.array([s["disk_speed"] for s in steps], float)
    touched = np.array([bool(s["left_fingertip"] or s["right_fingertip"]) for s in steps])
    n = len(dtz)
    if n == 0:
        return {}
    entered = dtz <= entry_tol
    in_zone = dtz <= center_tol                                   # fully contained
    settled = spd < settle_vel
    cs = in_zone & settled
    entry_idx = int(np.argmax(entered)) if entered.any() else None
    exit_after_entry = bool((dtz[entry_idx:] > entry_tol).any()) if entry_idx is not None else False
    reentries = int(np.sum((~entered[:-1]) & entered[1:])) if n > 1 else 0
    return {"target_entry": bool(entered.any()), "one_step_in_zone": bool(in_zone.any()),
            "k3_dwell": _run_dwell(cs) >= 3, "k6_dwell": _run_dwell(cs) >= 6, "k10_dwell": _run_dwell(cs) >= 10,
            "max_held_dwell": int(_run_dwell(cs)), "centered_fraction": round(float(in_zone.mean()), 4),
            "settled_fraction": round(float(settled.mean()), 4),
            "centered_settled_integral": round(float(cs.sum() * CONTROL_DT), 4),
            "exit_after_entry": exit_after_entry, "reentry_count": reentries,
            "final_distance": round(float(dtz[-1]), 4), "final_speed": round(float(spd[-1]), 4),
            "ever_touched": bool(touched.any()), "final_touching": bool(touched[-1])}


def dwell_k_success(steps, K, *, center_tol=0.02, settle_vel=0.06):
    """Success under a dwell-K variant only (geometry/settle fixed): coin centered+settled for K consecutive steps."""
    dtz = np.array([s["disk_to_zone"] for s in steps], float); spd = np.array([s["disk_speed"] for s in steps], float)
    return _run_dwell((dtz <= center_tol) & (spd < settle_vel)) >= K


def touched_ever_vs_current(steps, *, center_tol=0.02, settle_vel=0.06, dwell_req=6):
    """Does requiring CURRENT robot contact during the certified dwell (instead of touched-ever) change the outcome?"""
    dtz = np.array([s["disk_to_zone"] for s in steps], float); spd = np.array([s["disk_speed"] for s in steps], float)
    con = np.array([bool(s["left_fingertip"] or s["right_fingertip"]) for s in steps])
    cs = (dtz <= center_tol) & (spd < settle_vel)
    # find first window of dwell_req consecutive centered+settled; was the robot touching THROUGHOUT it?
    run = 0
    for i, v in enumerate(cs):
        run = run + 1 if v else 0
        if run >= dwell_req:
            window = con[i - dwell_req + 1:i + 1]
            return {"delivered_touched_ever": bool(con[:i + 1].any()), "current_contact_through_dwell": bool(window.all()),
                    "any_contact_during_dwell": bool(window.any())}
    return {"delivered_touched_ever": bool(con.any()), "current_contact_through_dwell": None, "any_contact_during_dwell": None}


def decompose_reward(env, dist, action, terms):
    """Named per-term reward components ``{kind: weight·term(env,dist,action)}``. Their sum equals
    ``RewardSpec.evaluate`` exactly (same computation), so it verifies the scalar reward's decomposition."""
    from hymeko_rl.env.reward import _REWARD_TERMS
    return {kind: float(w) * float(_REWARD_TERMS[kind](env, dist, action)) for kind, w in terms}


def post_stream(trace):
    """The POST-step certificate stream of a captured rollout (state_{t+1} for each transition; the terminal
    post-action state is included as the last transition's post-step) — the stream used for re-certification/ladder."""
    return [tr["post"] for tr in trace["transitions"]]


def braking_eligibility_sweep(partA_rows, v_excess_grid):
    """Step 8, corrected denominator: braking is only NEEDED where the coin is moving TOWARD the target fast
    (``target_directed_radial_velocity > v_excess``, signed — not ``abs``). Target-away (retreating) states need no
    braking and are reported separately. Report support/target-directed, support/all, prevalence, false interventions
    on already-slow states."""
    out = {}
    total = len(partA_rows)
    for v in v_excess_grid:
        toward = [r for r in partA_rows if r["pi0_radial_vel"] > v]       # approaching fast ⇒ braking-eligible
        away = [r for r in partA_rows if r["pi0_radial_vel"] < -v]        # retreating ⇒ braking not needed
        toward_support = [r for r in toward if r["support"]["n_safe_beneficial"] > 0]
        all_support = [r for r in partA_rows if r["support"]["n_safe_beneficial"] > 0]
        false_on_slow = sum(1 for r in partA_rows if abs(r["pi0_radial_vel"]) <= v and r["support"]["n_safe_beneficial"] > 0)
        out[f"v_excess={v}"] = {
            "n_target_directed": len(toward), "n_target_away": len(away),
            "prevalence_target_directed": round(len(toward) / max(total, 1), 4),
            "support_over_target_directed": round(len(toward_support) / max(len(toward), 1), 4),
            "support_over_all": round(len(all_support) / max(total, 1), 4),
            "false_interventions_on_slow_states": false_on_slow}
    return out
