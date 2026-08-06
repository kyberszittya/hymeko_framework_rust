r"""Reduced-model → embodied humanoid transfer study: does the arm reaction-wheel benefit carry over? (No.)

The reduced balance model predicted that reaction-wheel arms extend the recoverable pitch basin (+0.30,
``reaction_wheel_arms.py``). This measures whether that transfers to the real MuJoCo humanoid (``balance_env``):
a **bounded arm-swing residual over the certified PD-hold scaffold** (the shoulders swing proportional to the
torso pitch / pitch-rate) vs the ``a = 0`` baseline, over a pitch-perturbation sweep.

**Measured result — it does NOT transfer.** The certified baseline already recovers pitch-rate perturbations up
to ~3 rad/s; where it fails (≥ 4 rad/s the body is pitching over and the pelvis is collapsing) the arm residual
adds **nothing** (Δ ≈ 0). Diagnosis: the arms' angular-momentum authority — small arm inertia, a ±0.4 rad target
range — is far below a whole-body base rotation, so the reduced model's favourable arm/torso inertia ratio
**over-predicted** the arms' value. The honest lesson: a reduced-model gain must be re-checked against the
embodiment's actual inertia ratios before it is claimed.

# Preconditions: the built ``hymeko`` CLI + MuJoCo (``balance_env`` builds). # Postconditions: ``transfer_study``
#   returns per-perturbation survival for the baseline and the arm residual (no benefit).
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv

_SHOULDER_L, _SHOULDER_R = 12, 14                            # actuator indices of the arm joints (shoulder_l/r)


def baseline_controller(env: HumanoidBalanceEnv) -> np.ndarray:
    """The certified scaffold: ``a = 0`` (PD-hold the nominal pose)."""
    return np.zeros(env.model.nu)


def arm_reaction_wheel_controller(k: float = 4.0, sign: int = 1, kd: float = 0.3):
    """A bounded arm-swing residual over the scaffold: swing both shoulders proportional to the torso pitch."""
    def control(env: HumanoidBalanceEnv) -> np.ndarray:
        a = np.zeros(env.model.nu)
        pitch = env.data.xmat[env._pelvis].reshape(3, 3)[0, 2]   # forward tilt (≈ sin pitch)
        s = float(np.clip(sign * (k * pitch + kd * env.data.qvel[4]), -1.0, 1.0))
        a[_SHOULDER_L] = s
        a[_SHOULDER_R] = s
        return a
    return control


def survival_rate(controller, perturb: float, n_seeds: int = 8) -> float:
    """Fraction of resets (at a fixed pitch-rate perturbation) the controller keeps upright for the full horizon."""
    env = HumanoidBalanceEnv(BalanceConfig(perturb_lo=perturb, perturb_hi=perturb))
    survived = 0
    for seed in range(n_seeds):
        env.reset(seed=seed)
        fell = False
        for _ in range(env.max_steps):
            _obs, _r, fell, _trunc, _info = env.step(controller(env))
            if fell:
                break
        survived += int(not fell)
    return survived / n_seeds


def transfer_study(perturbs=(1.0, 2.0, 3.0, 4.0, 5.0), n_seeds: int = 8,
                   arm=(4.0, 1)) -> "list[dict]":
    """Baseline vs the arm reaction-wheel residual across a pitch-perturbation sweep (survival rates + Δ)."""
    controller = arm_reaction_wheel_controller(k=arm[0], sign=arm[1])
    out = []
    for p in perturbs:
        base = survival_rate(baseline_controller, p, n_seeds)
        rw = survival_rate(controller, p, n_seeds)
        out.append({"perturb": p, "baseline": base, "arm_reaction_wheel": rw, "delta": rw - base})
    return out
