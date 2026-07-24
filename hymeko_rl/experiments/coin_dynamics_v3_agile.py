"""G0-refinement — COIN_DYNAMICS_CONTRACT_V3_AGILE: add TASK-INDEPENDENT agility + recovery gates so the velocity cap
comes from the GOVERNOR, not from crippling damping (the V2 over-damping confound the C1 diagnosis exposed).

V2 is preserved unchanged (speed-safe but agility-absent). V3 adds gates a torque arm needs a POSITION loop (PD) to
exercise: A1 reach a normal manipulation distance in physically-derived time, A2 track a slow reference (bounded error,
no persistent saturation), A3 free-space speed cap (reused), A4 contact-impulse recovery via INTEGRATED overspeed (a
single peak may occur, but ∫ReLU(|q̇|/q̇_safe−1)dt must be small and velocity returns below soft), A5 braking+settling,
A6 fast reversal without stall. Selection is LEXICOGRAPHIC and NEVER inspects delivery/K6: all-safety → agility →
recovery → minimal constant damping → minimal tracking error. No direct qvel state-clamp (artificial energy removal).
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.env.motion_contract import MotionLimits, TorqueGovernorConfig, govern_torque  # noqa: E402
from hymeko_rl.experiments.coin_dynamics_calibration import (  # noqa: E402 — reuse V2 physical tests + arm builder
    _fresh_arm, _finite, test_free_space_max_speed, test_sudden_release, test_velocity_reversal)
from hymeko_rl.experiments.video_coin_variants import _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
N, DT, SUB = 4, 0.01, 20
LIM = MotionLimits()


def _pd_step(m, d, q_des, kp, kv, gov, tau_prev, tau_rate):
    """PD position command → torque-rate limit → directional governor → step. The realistic manipulation loop (a torque
    arm needs a position controller); the governor caps velocity, damping/armature are the physical layer."""
    q, qd = d.qpos[:N], d.qvel[:N]
    tau = kp * (np.asarray(q_des, np.float64) - q) - kv * qd
    if tau_rate is not None and tau_prev is not None:
        tau = tau_prev + np.clip(tau - tau_prev, -tau_rate * DT, tau_rate * DT)
    tau = govern_torque(tau, qd, gov)
    d.ctrl[:N] = np.clip(tau, m.actuator_ctrlrange[:N, 0], m.actuator_ctrlrange[:N, 1])
    for _ in range(SUB):
        mujoco.mj_step(m, d)
    return d.ctrl[:N].copy()


def test_reach(m, d, gov, kp, kv, tau_rate, step=0.5, tol=0.06, cap=120):
    """A1 — reach a normal manipulation joint displacement within a PHYSICALLY-DERIVED time window (T_min from the
    speed limit + accel/settle margin). Fails if the safe-range arm can't get there in reasonable time (over-damped)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    q_des = np.full(N, step)
    t_min = step / LIM.joint_vel_hard / DT
    t_max = t_min + 60                                       # accel/braking/settling margin (control steps)
    prev, reached_at, stable = None, None, True
    for k in range(cap):
        prev = _pd_step(m, d, q_des, kp, kv, gov, prev, tau_rate)
        stable = stable and _finite(d)
        if reached_at is None and float(np.max(np.abs(d.qpos[:N] - q_des))) < tol:
            reached_at = k + 1
            break
    return {"reached_at": reached_at, "t_min": round(t_min, 1), "t_max": round(t_max, 1),
            "ok": bool(stable and reached_at is not None and reached_at <= t_max)}


def test_tracking(m, d, gov, kp, kv, tau_rate, rate=1.0, target=0.5, steps=80):
    """A2 — track a slow ramp reference: bounded tracking error and no PERSISTENT torque saturation (shows the damping
    does not suppress useful control)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    prev, errs, sat = None, [], 0
    hi = m.actuator_ctrlrange[:N, 1]
    for k in range(steps):
        q_des = np.full(N, min(rate * k * DT, target))
        prev = _pd_step(m, d, q_des, kp, kv, gov, prev, tau_rate)
        errs.append(float(np.max(np.abs(d.qpos[:N] - q_des))))
        sat += int(np.any(np.abs(d.ctrl[:N]) >= 0.98 * hi))
    mean_err = float(np.mean(errs))
    return {"mean_track_err": round(mean_err, 3), "sat_frac": round(sat / steps, 2),
            "ok": bool(mean_err < 0.15 and sat / steps < 0.5)}


def test_recovery_integrated(m, d, gov, kp, kv, tau_rate, spike=12.0, cap=60):
    """A4 — after a contact-impulse velocity spike (zero position command), a single peak may occur but the INTEGRATED
    overspeed ∫ReLU(|q̇|/q̇_safe−1)dt must be small and velocity must return below the soft limit within N cycles (via
    inertia+damping, no artificial clamp)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    d.qvel[:N] = spike
    mujoco.mj_forward(m, d)
    prev, cycles, integ = None, cap, 0.0
    for k in range(cap):
        prev = _pd_step(m, d, np.zeros(N), kp, kv, gov, prev, tau_rate)
        v = float(np.max(np.abs(d.qvel[:N])))
        integ += max(0.0, v / LIM.joint_vel_safe - 1.0) * DT
        if v <= LIM.joint_vel_safe:
            cycles = k + 1
            break
    return {"return_cycles": cycles, "integrated_overspeed": round(integ, 3),
            "ok": bool(cycles <= 25 and integ < 0.6 and float(np.max(np.abs(d.qvel[:N]))) <= LIM.joint_vel_hard)}


def evaluate(pi0, base, forbidden, cfg):
    gov = TorqueGovernorConfig(cfg["qdot_soft"], cfg["qdot_hard"])
    kp, kv, tr = cfg["kp"], cfg["kv"], cfg["tau_rate"]
    m, d = _fresh_arm(pi0, base, forbidden, cfg)
    safety = {"free_space_speed": test_free_space_max_speed(m, d, gov, tr, DT),
              "sudden_release": test_sudden_release(m, d, gov, tr, DT),
              "reversal": test_velocity_reversal(m, d, gov, tr, DT)}
    agility = {"reach": test_reach(m, d, gov, kp, kv, tr), "tracking": test_tracking(m, d, gov, kp, kv, tr)}
    recovery = {"contact_recovery": test_recovery_integrated(m, d, gov, kp, kv, tr)}
    return {"safety": safety, "agility": agility, "recovery": recovery,
            "safety_ok": all(v["ok"] for v in safety.values()),
            "agility_ok": all(v["ok"] for v in agility.values()),
            "recovery_ok": all(v["ok"] for v in recovery.values())}


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    # sweep — governor-dominant bias (lower damping), PD gains; NO delivery in the selection
    grid = []
    for (qs, qh) in [(1.2, 2.5), (1.0, 2.2)]:
        for arm in (0.05, 0.1, 0.2):
            for damp in (2.0, 4.0, 8.0):
                for kp, kv in [(60.0, 6.0), (100.0, 10.0)]:
                    grid.append({"qdot_soft": qs, "qdot_hard": qh, "armature": arm, "damping": damp,
                                 "friction": 0.02, "kp": kp, "kv": kv, "tau_rate": 30.0})
    runs = []
    for cfg in grid:
        r = evaluate(pi0, base, forbidden, cfg)
        passes = r["safety_ok"] and r["agility_ok"] and r["recovery_ok"]
        runs.append({"config": cfg, "results": r, "passes": passes})
        print(f"arm{cfg['armature']} damp{cfg['damping']} gov{cfg['qdot_hard']} kp{cfg['kp']}: "
              f"safety={r['safety_ok']} agility={r['agility_ok']} recovery={r['recovery_ok']} "
              f"reach={r['agility']['reach']['reached_at']} trackerr={r['agility']['tracking']['mean_track_err']} PASS={passes}", flush=True)
    winners = [w for w in runs if w["passes"]]
    frozen, verdict = None, None
    if winners:
        # lexicographic: min damping, then min tracking error (NEVER delivery)
        winners.sort(key=lambda w: (w["config"]["damping"], w["results"]["agility"]["tracking"]["mean_track_err"]))
        best = winners[0]["config"]
        frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V3_AGILE", "motion_limit_version": "V1",
                  "control_dt": DT, "substeps": SUB, **best}
        verdict = "V3_AGILE_FROZEN"
    else:
        verdict = "CURRENT_EMBODIMENT_OR_ACTUATION_MODEL_INCOMPATIBLE_WITH_REALISTIC_AGILE_MANIPULATION"
    manifest = {"contract": "COIN_DYNAMICS_CONTRACT_V3_AGILE_CALIBRATION", "date": "2026-07-25",
                "discipline": "task-independent agility+recovery gates; lexicographic (min damping) selection; NO delivery",
                "preserved_v2": True, "n_configs": len(grid), "runs": runs, "frozen_contract": frozen, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/dynamics_contract_v3_agile.json", "w"), indent=1, default=float)
    print(f"\n→ {verdict}" + (f"  (damp {frozen['damping']} gov {frozen['qdot_hard']} kp {frozen['kp']})" if frozen else ""))
    print(f"artifact: {OUT}/dynamics_contract_v3_agile.json\nCOIN_DYNAMICS_V3_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
