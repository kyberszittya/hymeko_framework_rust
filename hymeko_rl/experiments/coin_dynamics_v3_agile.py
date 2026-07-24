"""G0-refinement — COIN_DYNAMICS_CONTRACT_V3_AGILE: task-independent agility+recovery gates so the velocity cap comes
from the GOVERNOR (per sub-step), not from crippling damping (the V2 over-damping confound C1 exposed).

CRITICAL FIX (user contract): the SAME low-level stack (`GovernedArm`, governor per sub-step) runs in calibration AND in
the coin controller — the previous per-control-step governing under-governed (free-space peak blew to 17.5 rad/s), an
artifact, not the real stack. V2 is preserved. Selection is LEXICOGRAPHIC and NEVER inspects delivery: all-safety →
agility → recovery → minimal constant damping → minimal tracking error. Every config's full A1–A6 result is saved (the
safety-vs-agility feasibility map). No direct qvel state-clamp.
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.env.governed_arm import GovernedArm, V3Stack  # noqa: E402
from hymeko_rl.env.motion_contract import MotionLimits  # noqa: E402
from hymeko_rl.experiments.coin_dynamics_calibration import _fresh_arm  # noqa: E402 — reuse the coin-arm builder
from hymeko_rl.experiments.video_coin_variants import _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
N, DT = 4, 0.01
LIM = MotionLimits()


def _reset(arm):
    arm.d.qpos[:N] = 0.0
    arm.d.qvel[:] = 0.0
    mujoco.mj_forward(arm.m, arm.d)
    arm._prev = None


# ── A3 free-space speed cap (now governed per sub-step) ──
def a3_free_space(arm):
    _reset(arm)
    hi = arm.m.actuator_ctrlrange[:N, 1].copy()
    peak, stable = 0.0, True
    for _ in range(120):
        arm.torque_step(hi)
        peak = max(peak, float(np.max(np.abs(arm.d.qvel[:N]))))
        stable = stable and np.all(np.isfinite(arm.d.qvel))
    return {"peak_vel": round(peak, 2), "ok": bool(stable and peak <= LIM.joint_vel_hard * 1.2)}


# ── A6 reversal / braking preserved ──
def a6_reversal(arm):
    _reset(arm)
    hi = arm.m.actuator_ctrlrange[:N, 1].copy()
    for _ in range(40):
        arm.torque_step(hi)
    v_fwd = float(arm.d.qvel[0])
    for _ in range(60):
        arm.torque_step(-hi)
    return {"v_fwd": round(v_fwd, 2), "v_rev": round(float(arm.d.qvel[0]), 2), "ok": bool(v_fwd > 0.3 and arm.d.qvel[0] < 0.0)}


# ── A5 sudden release / settling ──
def a5_release(arm):
    _reset(arm)
    hi = arm.m.actuator_ctrlrange[:N, 1].copy()
    for _ in range(40):
        arm.torque_step(hi)
    steps = 0
    for k in range(200):
        arm.torque_step(np.zeros(N))
        steps = k + 1
        if np.max(np.abs(arm.d.qvel[:N])) < LIM.terminal_joint_vel:
            break
    return {"settle_steps": steps, "ok": bool(np.max(np.abs(arm.d.qvel[:N])) < LIM.terminal_joint_vel)}


# ── A1 reach a normal displacement in physically-derived time ──
def a1_reach(arm, step=0.5, tol=0.06, cap=120):
    _reset(arm)
    q_des = np.full(N, step)
    t_max = step / LIM.joint_vel_hard / DT + 60
    reached = None
    for k in range(cap):
        arm.pd_step(q_des)
        if reached is None and float(np.max(np.abs(arm.d.qpos[:N] - q_des))) < tol:
            reached = k + 1
            break
    return {"reached_at": reached, "t_max": round(t_max, 1), "ok": bool(reached is not None and reached <= t_max)}


# ── A2 low-speed tracking ──
def a2_tracking(arm, rate=1.0, target=0.5, steps=90):
    _reset(arm)
    hi = arm.m.actuator_ctrlrange[:N, 1]
    errs, sat = [], 0
    for k in range(steps):
        arm.pd_step(np.full(N, min(rate * k * DT, target)))
        errs.append(float(np.max(np.abs(arm.d.qpos[:N] - min(rate * k * DT, target)))))
        sat += int(np.any(np.abs(arm.d.ctrl[:N]) >= 0.98 * hi))
    me = float(np.mean(errs))
    return {"mean_track_err": round(me, 3), "sat_frac": round(sat / steps, 2), "ok": bool(me < 0.15 and sat / steps < 0.5)}


# ── A4 contact-impulse recovery via integrated overspeed ──
def a4_recovery(arm, spike=12.0, cap=60):
    _reset(arm)
    arm.d.qvel[:N] = spike
    mujoco.mj_forward(arm.m, arm.d)
    cycles, integ = cap, 0.0
    for k in range(cap):
        arm.torque_step(np.zeros(N))
        v = float(np.max(np.abs(arm.d.qvel[:N])))
        integ += max(0.0, v / LIM.joint_vel_safe - 1.0) * DT
        if v <= LIM.joint_vel_safe:
            cycles = k + 1
            break
    return {"return_cycles": cycles, "integrated_overspeed": round(integ, 3),
            "ok": bool(cycles <= 25 and integ < 0.6 and float(np.max(np.abs(arm.d.qvel[:N]))) <= LIM.joint_vel_hard)}


def evaluate(pi0, base, forbidden, stack: V3Stack):
    m, d = _fresh_arm(pi0, base, forbidden, {"armature": stack.armature, "damping": stack.damping, "friction": stack.friction})
    with GovernedArm(m, d, stack, n=N) as arm:
        safety = {"free_space": a3_free_space(arm), "reversal": a6_reversal(arm), "release": a5_release(arm)}
        agility = {"reach": a1_reach(arm), "tracking": a2_tracking(arm)}
        recovery = {"contact_recovery": a4_recovery(arm)}
    return {"safety": safety, "agility": agility, "recovery": recovery,
            "safety_ok": all(v["ok"] for v in safety.values()), "agility_ok": all(v["ok"] for v in agility.values()),
            "recovery_ok": all(v["ok"] for v in recovery.values())}


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    grid = []
    for (qs, qh) in [(1.2, 2.5), (1.0, 2.2), (0.8, 2.0)]:
        for arm in (0.05, 0.1, 0.2):
            for damp in (2.0, 4.0, 8.0):
                for kp, kv in [(60.0, 6.0), (120.0, 12.0)]:
                    grid.append(V3Stack(qs, qh, arm, damp, 0.02, kp, kv, 30.0))
    runs = []
    for s in grid:
        r = evaluate(pi0, base, forbidden, s)
        passes = r["safety_ok"] and r["agility_ok"] and r["recovery_ok"]
        runs.append({"config": s.__dict__, "results": r, "passes": passes})
        print(f"arm{s.armature} damp{s.damping} gov{s.qdot_hard} kp{s.kp}: S={r['safety_ok']} A={r['agility_ok']} R={r['recovery_ok']} "
              f"peak={r['safety']['free_space']['peak_vel']} reach={r['agility']['reach']['reached_at']} PASS={passes}", flush=True)
    winners = [w for w in runs if w["passes"]]
    frozen, verdict = None, None
    if winners:
        winners.sort(key=lambda w: (w["config"]["damping"], w["results"]["agility"]["tracking"]["mean_track_err"]))
        frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V3_AGILE", "motion_limit_version": "V1", **winners[0]["config"]}
        verdict = "V3_AGILE_FROZEN"
    else:
        verdict = "CURRENT_ACTUATION_STACK_CANNOT_SATISFY_REALISTIC_SPEED_AND_USEFUL_AGILITY_SIMULTANEOUSLY"
    json.dump({"contract": "COIN_DYNAMICS_CONTRACT_V3_AGILE_CALIBRATION", "date": "2026-07-25",
               "discipline": "SAME GovernedArm stack in calibration + execution; per-sub-step governor; lexicographic min-damping; NO delivery",
               "preserved_v2": True, "n_configs": len(grid), "runs": runs, "frozen_contract": frozen, "verdict": verdict},
              open(f"{OUT}/dynamics_contract_v3_agile.json", "w"), indent=1, default=float)
    print(f"\n→ {verdict}" + (f"  (damp {frozen['damping']} gov {frozen['qdot_hard']} kp {frozen['kp']} arm {frozen['armature']})" if frozen else ""))
    print(f"artifact: {OUT}/dynamics_contract_v3_agile.json\nCOIN_DYNAMICS_V3_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
