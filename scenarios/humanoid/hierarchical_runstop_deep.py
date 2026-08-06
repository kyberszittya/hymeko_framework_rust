r"""Level-2 refinement — the arm reaction wheel refined into shoulder + elbow (the humanoid's actual joints).

The Level-1 arm helped run-stop only modestly because a single reaction wheel's momentum capacity
(``I_arm·ω_max``) was smaller than the braking-induced angular momentum. The **deeper** model refines that arm
into **two joints — shoulder + elbow** (the humanoid's actual actuators, per ``runstop_humanoid.hymeko``), whose
combined momentum capacity is larger, relaxing exactly that bottleneck. The RL policy is warm-started **through
this deeper level**: the Level-1 arm channel is copied into the shoulder, the new elbow channel zero-initialised,
so the deep policy begins bit-identical to Level-1 and then learns to use the second joint.

This is the third rung of the hierarchy (L0 abstract → L1 foot/arm → L2 ankle/hip + shoulder/elbow), the RL
companion to the deep ``<isa>`` chain in ``data/robotics/runstop_ports.hymeko``.

# Preconditions: a flight-heavy ``DeepConfig``. # Post: the warm-started deep policy is initially identical to
#   the Level-1 arm policy (the elbow channel inert until learned).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeepConfig:
    """Flight-heavy run-stop with a two-segment (shoulder + elbow) reaction-wheel arm."""

    dt: float = 0.004
    inertia: float = 1.6
    l_damp: float = 1.5
    fall_pitch: float = 1.25
    ts: float = 0.10                 # flight-heavy (69%) — the arm's flight authority is the bottleneck
    tf: float = 0.22
    k_couple: float = 0.9
    a_max: float = 1.5
    fx_max: float = 2.5
    inertia_shoulder: float = 0.25   # shoulder + elbow: combined momentum capacity exceeds a single arm
    wa_shoulder_max: float = 8.0
    inertia_elbow: float = 0.18
    wa_elbow_max: float = 8.0
    arm_range: float = 1.6
    t_stop: float = 0.3
    ramp: float = 0.4
    horizon: float = 1.8
    v_stop_tol: float = 0.3
    hidden: int = 20
    pop: int = 48
    elite: int = 10
    iters: int = 30
    init_std: float = 0.4
    seed: int = 0

    @property
    def cycle(self) -> float:
        return self.ts + self.tf


# Level-1 (arm) policy: 6 inputs (vx,L,pitch,targ,phase,theta_arm) → 3 actions (fx,a,wa).
# Level-2 (deep) policy: 7 inputs (+theta_elbow) → 4 actions (fx, a, wa_shoulder, wa_elbow).
_L1 = (6, 3)
_L2 = (7, 4)


def _dims(cfg: DeepConfig, sizes: "tuple[int, int]") -> "tuple[int, int, int]":
    return sizes[0], cfg.hidden, sizes[1]


def n_params(cfg: DeepConfig, deep: bool) -> int:
    n_in, n_h, n_out = _dims(cfg, _L2 if deep else _L1)
    return n_in * n_h + n_h + n_h * n_out + n_out


def _forward(params: np.ndarray, feats: np.ndarray, cfg: DeepConfig, deep: bool) -> np.ndarray:
    n_in, n_h, n_out = _dims(cfg, _L2 if deep else _L1)
    a, b = n_in * n_h, n_in * n_h + n_h
    c = b + n_h * n_out
    h = np.tanh(feats @ params[:a].reshape(n_in, n_h) + params[a:b])
    return np.tanh(h @ params[b:c].reshape(n_h, n_out) + params[c:])


def warm_start(level1: np.ndarray, cfg: DeepConfig) -> np.ndarray:
    """Lift a Level-1 (single-arm) policy into the Level-2 (shoulder+elbow) space; the elbow channel zeroed.

    Copies the shared foot/L/shoulder weights (the L1 arm channel → the shoulder); the new elbow input-row and
    output-column start at 0 — so the deep policy is initially bit-identical to Level-1.
    """
    ni1, n_h, no1 = _dims(cfg, _L1)
    ni2, _, no2 = _dims(cfg, _L2)
    w1 = level1[:ni1 * n_h].reshape(ni1, n_h)
    b1 = level1[ni1 * n_h:ni1 * n_h + n_h]
    w2 = level1[ni1 * n_h + n_h:ni1 * n_h + n_h + n_h * no1].reshape(n_h, no1)
    b2 = level1[-no1:]
    w1d = np.zeros((ni2, n_h))
    w1d[:ni1] = w1                                           # new elbow-angle input row = 0
    w2d = np.zeros((n_h, no2))
    w2d[:, :no1] = w2                                        # new elbow-torque output column = 0
    b2d = np.zeros(no2)
    b2d[:no1] = b2
    return np.concatenate([w1d.ravel(), b1, w2d.ravel(), b2d])


def rollout(params: np.ndarray, x0: np.ndarray, cfg: DeepConfig, deep: bool) -> "tuple[np.ndarray, np.ndarray]":
    """Roll the Level-1 (single arm) or Level-2 (shoulder+elbow) policy; return (stopped, fell)."""
    v_run = x0[:, 0].copy()
    vx, ll, pitch = x0[:, 0].copy(), x0[:, 1].copy(), x0[:, 2].copy()
    th_sh, th_el = np.zeros(len(x0)), np.zeros(len(x0))
    fell = np.zeros(len(x0), dtype=bool)
    for i in range(int(round(cfg.horizon / cfg.dt))):
        t = i * cfg.dt
        stance = (t % cfg.cycle) < cfg.ts
        targ = np.where(t < cfg.t_stop, v_run, np.maximum(0.0, v_run * (1.0 - (t - cfg.t_stop) / cfg.ramp)))
        phase = np.full(len(x0), (t % cfg.cycle) / cfg.cycle)
        cols = [vx, ll, pitch, targ, phase, th_sh] + ([th_el] if deep else [])
        out = _forward(params, np.stack(cols, axis=1), cfg, deep)
        fx = np.where(stance, cfg.fx_max * out[:, 0], 0.0)
        a = np.where(stance, cfg.a_max * out[:, 1], 0.0)
        wa_sh = _limit(cfg.wa_shoulder_max * out[:, 2], th_sh, cfg)
        th_sh = np.clip(th_sh + wa_sh * cfg.dt, -cfg.arm_range, cfg.arm_range)
        arm_mom = cfg.inertia_shoulder * wa_sh
        if deep:                                             # the second joint adds momentum capacity
            wa_el = _limit(cfg.wa_elbow_max * out[:, 3], th_el, cfg)
            th_el = np.clip(th_el + wa_el * cfg.dt, -cfg.arm_range, cfg.arm_range)
            arm_mom = arm_mom + cfg.inertia_elbow * wa_el
        ll = ll + (-cfg.l_damp * ll * stance + a - cfg.k_couple * fx) * cfg.dt
        pitch = pitch + ((ll - arm_mom) / cfg.inertia) * cfg.dt
        vx = vx + fx * cfg.dt
        fell |= np.abs(pitch) > cfg.fall_pitch
    return (np.abs(vx) < cfg.v_stop_tol) & ~fell, fell


def _limit(wa: np.ndarray, theta: np.ndarray, cfg: DeepConfig) -> np.ndarray:
    """Hard arm-range limit: no outward angular velocity past the mechanical stop."""
    nxt = theta + wa * cfg.dt
    return np.where((np.abs(nxt) > cfg.arm_range) & (np.sign(wa) == np.sign(np.where(theta == 0, wa, theta))),
                    0.0, wa)


def mixed_set(cfg: DeepConfig, offset: float = 0.0, n: int = 7) -> np.ndarray:
    vs = np.array([1.5, 2.0, 2.5, 3.0]) + offset * 0.25
    ps = np.linspace(-0.3, 0.3, n) + offset * (0.6 / n)
    ls = np.linspace(-1.0, 1.0, n) + offset * (2.0 / n)
    vv, pp, ll = np.meshgrid(vs, ps, ls, indexing="ij")
    return np.stack([vv.ravel(), ll.ravel(), pp.ravel()], axis=1)


def train_cem(cfg: DeepConfig, deep: bool, init: "np.ndarray | None" = None) -> np.ndarray:
    rng = np.random.RandomState(cfg.seed)
    x0 = mixed_set(cfg, offset=0.0)
    dim = n_params(cfg, deep)
    mean = np.zeros(dim) if init is None else init.copy()
    std = np.full(dim, cfg.init_std)
    for _ in range(cfg.iters):
        pop = mean + std * rng.standard_normal((cfg.pop, dim))
        scores = np.array([rollout(p, x0, cfg, deep)[0].mean() for p in pop])
        elite = pop[np.argsort(scores)[-cfg.elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-6
    return mean


def evaluate(params: np.ndarray, cfg: DeepConfig, deep: bool, offset: float = 0.5) -> float:
    return float(rollout(params, mixed_set(cfg, offset=offset), cfg, deep)[0].mean())
