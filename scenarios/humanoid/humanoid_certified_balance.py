r"""Certified balance for the embodied humanoid (C) — the verification arc applied to the real MuJoCo humanoid.

The reduced-model verification work (viability boundary + HSTL runtime monitor) applied to ``balance_env``:
- the certified ``a=0`` PD-hold scaffold has a **recoverable region** in the perturbation space — a viability
  boundary (the max pitch-rate it catches);
- an **HSTL runtime monitor** over the balance safety spec ``G(safety_margin ≥ 0)`` watches an executing
  trajectory, where ``safety_margin = min(uprightness − upright_thr, pelvis_z − pelvis_thr)`` is the balance
  fall margin (BOTH failure modes — tipping over AND the pelvis collapsing), giving a signed robustness and an
  early warning before the fall.

This connects the M0/monitor machinery to the embodiment: the same ``make_monitor`` / robust-STL semantics used on
the reduced model now score the real humanoid's balance.

# Preconditions: MuJoCo + the built CLI (``balance_env`` builds). # Postconditions: the monitor robustness is < 0
#   exactly when the humanoid falls, and warns no later than the fall.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv
from scenarios.humanoid.hstl_monitor import make_monitor

_SPEC = "G(safety_margin >= 0)"


def safety_margin(env: HumanoidBalanceEnv) -> float:
    """The balance fall margin: min over both failure modes (uprightness and pelvis height)."""
    sig = env._com_sig()
    up = sig["uprightness"] - env.cfg.fall_uprightness
    pz = float(env.data.xpos[env._pelvis, 2]) - env.cfg.fall_pelvis_z
    return float(min(up, pz))


def scaffold_recovers(perturb: float, seeds) -> float:
    """Fraction of pitch-rate perturbations the certified ``a=0`` scaffold keeps upright (the recoverable region)."""
    env = HumanoidBalanceEnv(BalanceConfig(perturb_lo=perturb, perturb_hi=perturb))
    survived = 0
    for s in seeds:
        env.reset(seed=s)
        fell = False
        for _ in range(env.max_steps):
            _o, _r, fell, _t, _i = env.step(np.zeros(env.model.nu))
            if fell:
                break
        survived += int(not fell)
    return survived / len(seeds)


def recoverable_bound(seeds=range(8), lo: float = 2.0, hi: float = 6.0, tol: float = 0.25) -> float:
    """Bisection for the viability boundary: the largest pitch-rate the scaffold fully recovers over ``seeds``."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if scaffold_recovers(mid, seeds) >= 0.999:
            lo = mid
        else:
            hi = mid
    return lo


def lateral_push_recovers(push: float, seeds) -> float:
    """Fraction of lateral pushes the certified scaffold recovers (the push-dimension of the viability region)."""
    env = HumanoidBalanceEnv(BalanceConfig(perturb_lo=0.0, perturb_hi=0.0, push_lat_lo=push, push_lat_hi=push))
    survived = 0
    for s in seeds:
        env.reset(seed=s)
        fell = False
        for _ in range(env.max_steps):
            _o, _r, fell, _t, _i = env.step(np.zeros(env.model.nu))
            if fell:
                break
        survived += int(not fell)
    return survived / len(seeds)


def lateral_push_bound(seeds=range(8), lo: float = 1.0, hi: float = 3.0, tol: float = 0.15) -> float:
    """Bisection for the largest lateral push the scaffold fully recovers (a sharp cliff in this env, ~2.3 m/s)."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if lateral_push_recovers(mid, seeds) >= 0.999:
            lo = mid
        else:
            hi = mid
    return lo


def monitor_balance(perturb: float, seed: int, warn_band: float = 0.1) -> dict:
    """Run the scaffold under a perturbation through the HSTL monitor; return the verdict, robustness and lead."""
    env = HumanoidBalanceEnv(BalanceConfig(perturb_lo=perturb, perturb_hi=perturb))
    env.reset(seed=seed)
    monitor = make_monitor(_SPEC, "python", horizon=env.max_steps + 1)
    warn_step, fall_step, fell = -1, -1, False
    for t in range(env.max_steps):
        _o, _r, fell, _tr, _i = env.step(np.zeros(env.model.nu))
        margin = safety_margin(env)
        monitor.observe(t, {"safety_margin": margin})
        if warn_step < 0 and margin < warn_band:
            warn_step = t
        if fell:
            fall_step = t
            break
    return {"fell": fell, "robustness": monitor.robustness(), "satisfied": monitor.satisfied(),
            "warn_step": warn_step, "fall_step": fall_step,
            "lead_steps": (fall_step - warn_step) if (fell and warn_step >= 0) else -1}
