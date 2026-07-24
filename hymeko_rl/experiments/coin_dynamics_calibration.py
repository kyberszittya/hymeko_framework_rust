"""G0 — COIN_DYNAMICS_CONTRACT_V2 calibration, INDEPENDENT of task success.

The discipline (user): do NOT tune armature/damping to the delivery rate — that is a subtler dynamics hack. Calibrate the
governor + armature + viscous damping + joint friction + torque-rate limit on PHYSICAL tests only (free-space max speed,
velocity reversal, sudden release, contact-impulse recovery, terminal settling, numerical stability), freeze the passing
config, and only THEN re-measure the task (G1) without touching the physics again.

The coin arm is driven DIRECTLY (mj_step on the coin model's first 4 dofs) with test torques — no task, no coin contact
except the explicit fixed-obstacle test — and the governor is applied per step as a pure function (no global callback,
so calibration is clean/deterministic). Contact spikes are handled by physical inertia/damping/active velocity feedback,
never by an artificial state-clamp; the hard gate may still reject an over-large spike.
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

from hymeko_rl.env.motion_contract import MotionLimits, TorqueGovernorConfig, govern_torque  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
N = 4                                                     # coin arm dofs
LIM = MotionLimits()                                      # joint_vel_hard 3.0, terminal 0.25


def _fresh_arm(pi0, base, forbidden, cfg):
    """A coin model+data with the candidate dynamics applied to the arm dofs; arm reset to home, coin parked away."""
    rl, _g = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=14000, tries=2)
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:N] = cfg["armature"]
    m.dof_damping[:N] = cfg["damping"]
    m.dof_frictionloss[:N] = cfg["friction"]
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return m, d


def _step_governed(m, d, tau_cmd, gov: TorqueGovernorConfig, tau_prev, tau_rate, dt_ctrl):
    """One control step: torque-rate limit → directional velocity governor → mj_step (frame_skip substeps). Returns the
    applied torque (for the next rate limit)."""
    tau = np.asarray(tau_cmd, np.float64).copy()
    if tau_rate is not None and tau_prev is not None:
        tau = tau_prev + np.clip(tau - tau_prev, -tau_rate * dt_ctrl, tau_rate * dt_ctrl)
    tau = govern_torque(tau, d.qvel[:N], gov)
    d.ctrl[:N] = np.clip(tau, m.actuator_ctrlrange[:N, 0], m.actuator_ctrlrange[:N, 1])
    fs = 20
    for _ in range(fs):
        mujoco.mj_step(m, d)
    return d.ctrl[:N].copy()


def _finite(d):
    return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))


def test_free_space_max_speed(m, d, gov, tau_rate, dt):
    """Command MAX torque in free space; peak joint velocity must stay near the hard limit (governor caps acceleration)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    hi = m.actuator_ctrlrange[:N, 1].copy()
    peak, prev, stable = 0.0, None, True
    for _ in range(120):
        prev = _step_governed(m, d, hi, gov, prev, tau_rate, dt)
        stable = stable and _finite(d)
        peak = max(peak, float(np.max(np.abs(d.qvel[:N]))))
    return {"peak_vel": round(peak, 2), "stable": stable, "ok": bool(stable and peak <= LIM.joint_vel_hard * 1.15)}


def test_velocity_reversal(m, d, gov, tau_rate, dt):
    """Accelerate +, then command −: braking torque must still act and the joint must reverse (governor never blocks braking)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    hi = m.actuator_ctrlrange[:N, 1].copy()
    prev = None
    for _ in range(40):
        prev = _step_governed(m, d, hi, gov, prev, tau_rate, dt)
    v_fwd = float(d.qvel[0])
    for _ in range(80):
        prev = _step_governed(m, d, -hi, gov, prev, tau_rate, dt)
    v_rev = float(d.qvel[0])
    return {"v_fwd": round(v_fwd, 2), "v_rev": round(v_rev, 2), "ok": bool(v_fwd > 0.3 and v_rev < 0.0)}


def test_sudden_release(m, d, gov, tau_rate, dt):
    """Accelerate, then release torque (0): viscous damping must decay the velocity toward the terminal band."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    hi = m.actuator_ctrlrange[:N, 1].copy()
    prev = None
    for _ in range(40):
        prev = _step_governed(m, d, hi, gov, prev, tau_rate, dt)
    v0 = float(np.max(np.abs(d.qvel[:N])))
    steps = 0
    for k in range(200):
        prev = _step_governed(m, d, np.zeros(N), gov, prev, tau_rate, dt)
        steps = k + 1
        if np.max(np.abs(d.qvel[:N])) < LIM.terminal_joint_vel:
            break
    return {"v0": round(v0, 2), "decay_steps": steps, "ok": bool(np.max(np.abs(d.qvel[:N])) < LIM.terminal_joint_vel)}


def test_contact_impulse_recovery(m, d, gov, tau_rate, dt):
    """Inject a velocity SPIKE (a contact impulse, physically) with zero command; inertia+damping must return it to the
    allowed range within a few control cycles — NOT an artificial clamp."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    d.qvel[:N] = 12.0
    mujoco.mj_forward(m, d)   # a 12 rad/s impulse
    prev = None
    cycles = 0
    for k in range(60):
        prev = _step_governed(m, d, np.zeros(N), gov, prev, tau_rate, dt)
        cycles = k + 1
        if np.max(np.abs(d.qvel[:N])) <= LIM.joint_vel_hard:
            break
    return {"recovery_cycles": cycles, "final_vel": round(float(np.max(np.abs(d.qvel[:N]))), 2),
            "ok": bool(np.max(np.abs(d.qvel[:N])) <= LIM.joint_vel_hard and cycles <= 20)}


def test_normal_range_undistorted(m, d, gov, tau_rate, dt):
    """A small (below-soft) command must pass through the governor essentially unchanged (no distortion of normal control)."""
    d.qpos[:N] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    small = np.full(N, 0.3)                                # low torque → low speed (below qdot_soft)
    prev = None
    peak = 0.0
    for _ in range(60):
        prev = _step_governed(m, d, small, gov, prev, tau_rate, dt)
        peak = max(peak, float(np.max(np.abs(d.qvel[:N]))))
    # governed vs ungoverned final velocity should match closely when never above soft
    return {"peak_vel": round(peak, 3), "ok": bool(peak < gov.qdot_soft + 0.3)}


CAL_TESTS = (test_free_space_max_speed, test_velocity_reversal, test_sudden_release,
             test_contact_impulse_recovery, test_normal_range_undistorted)


def calibrate(pi0, base, forbidden, cfg, dt):
    gov = TorqueGovernorConfig(cfg["qdot_soft"], cfg["qdot_hard"])
    m, d = _fresh_arm(pi0, base, forbidden, cfg)
    results = {t.__name__.replace("test_", ""): t(m, d, gov, cfg["tau_rate"], dt) for t in CAL_TESTS}
    results["all_ok"] = all(r["ok"] for r in results.values())
    return results


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    dt_ctrl = 0.01                                         # frame_skip 20 · sim dt 0.0005
    # sweep candidate dynamics contracts (physical calibration only — NOT delivery)
    grid = [
        {"qdot_soft": 1.5, "qdot_hard": 3.0, "armature": 0.1, "damping": 5.0, "friction": 0.0, "tau_rate": None},
        {"qdot_soft": 1.5, "qdot_hard": 3.0, "armature": 0.2, "damping": 8.0, "friction": 0.05, "tau_rate": 40.0},
        {"qdot_soft": 1.5, "qdot_hard": 3.0, "armature": 0.3, "damping": 12.0, "friction": 0.1, "tau_rate": 30.0},
        {"qdot_soft": 1.2, "qdot_hard": 2.5, "armature": 0.4, "damping": 15.0, "friction": 0.1, "tau_rate": 25.0},
    ]
    runs, frozen = [], None
    for cfg in grid:
        r = calibrate(pi0, base, forbidden, cfg, dt_ctrl)
        runs.append({"config": cfg, "results": r})
        print(f"cfg arm{cfg['armature']} damp{cfg['damping']} fric{cfg['friction']} rate{cfg['tau_rate']}: "
              + " ".join(f"{k}={v['ok']}" for k, v in r.items() if k != "all_ok") + f"  ALL={r['all_ok']}", flush=True)
        if r["all_ok"] and frozen is None:
            frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V2", "motion_limit_version": "V1",
                      "control_dt": dt_ctrl, "substeps": 20, **cfg,
                      "joint_vel_hard": LIM.joint_vel_hard, "terminal_joint_vel": LIM.terminal_joint_vel}
    manifest = {"contract": "COIN_DYNAMICS_CONTRACT_V2_CALIBRATION", "date": "2026-07-25",
                "discipline": "calibrated on PHYSICAL tests only, independent of task/delivery success",
                "runs": runs, "frozen_contract": frozen}
    json.dump(manifest, open(f"{OUT}/dynamics_contract_v2.json", "w"), indent=1, default=float)
    print(f"\nFROZEN: {frozen['dynamics_contract'] if frozen else 'NONE PASSED — widen the sweep'}")
    print(f"artifact: {OUT}/dynamics_contract_v2.json\nCOIN_DYNAMICS_CALIBRATION_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
