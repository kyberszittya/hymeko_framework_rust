r"""Hierarchical model refinement with RL — coarse abstraction → detailed model, warm-started through the hierarchy.

This is the RL thesis the humanoid arc points at: start from a **coarse** abstraction (the centroidal run-stop with
a single abstract angular-momentum port, controllable only in stance), and **refine** it into a more detailed,
more capable model — the abstract L-port split into a **foot port** (stance) *and* a **reaction-wheel arm port**
(flight-capable) — with RL carried **through the hierarchy** by warm-starting the detailed policy from the coarse
one (copy the shared weights; zero-init the new arm channel, so the refined policy *begins* bit-identical to the
coarse and then learns to exploit the arm). The two levels correspond to two HyMeKo models: the coarse model's
abstract L-port is ``<isa>``-refined into the foot + arm ports (see ``data/robotics/runstop_*.hymeko``).

Measured: refining the model (adding the flight-phase arm) improves the hard, flight-heavy run-stop over
foot-only, and warm-starting through the hierarchy reaches a better policy than training the detailed model from
scratch.

# Preconditions: a flight-heavy ``HierConfig`` (so the coarse foot-only model is genuinely limited). # Post: the
#   warm-started detailed policy is initially identical to the coarse one (arm channel inert until learned).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierConfig:
    """Flight-heavy run-stop + reaction-wheel arm parameters (the two refinement levels share this)."""

    dt: float = 0.004
    inertia: float = 1.6
    l_damp: float = 1.5
    fall_pitch: float = 1.25
    ts: float = 0.13                 # stance / flight — flight-heavy so foot-only is limited
    tf: float = 0.17
    k_couple: float = 0.9
    a_max: float = 1.5
    fx_max: float = 2.5
    inertia_arm: float = 0.3         # reaction-wheel arm
    arm_range: float = 1.6
    wa_max: float = 8.0
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


def _sizes(cfg: HierConfig, use_arm: bool) -> "tuple[int, int, int]":
    return (6 if use_arm else 5, cfg.hidden, 3 if use_arm else 2)


def n_params(cfg: HierConfig, use_arm: bool) -> int:
    n_in, n_h, n_out = _sizes(cfg, use_arm)
    return n_in * n_h + n_h + n_h * n_out + n_out


def _forward(params: np.ndarray, feats: np.ndarray, cfg: HierConfig, use_arm: bool) -> np.ndarray:
    n_in, n_h, n_out = _sizes(cfg, use_arm)
    a, b = n_in * n_h, n_in * n_h + n_h
    c = b + n_h * n_out
    h = np.tanh(feats @ params[:a].reshape(n_in, n_h) + params[a:b])
    return np.tanh(h @ params[b:c].reshape(n_h, n_out) + params[c:])


def warm_start(coarse: np.ndarray, cfg: HierConfig) -> np.ndarray:
    """Lift a coarse (foot-only) policy into the detailed (arm) parameter space, arm channel zeroed.

    Copies the shared foot/L weights; the new arm input-row and arm output-column start at 0 — so the detailed
    policy is initially bit-identical to the coarse one (arm inert) and then learns to use the arm.
    """
    ni_c, n_h, no_c = _sizes(cfg, False)
    ni, _, no = _sizes(cfg, True)
    w1c = coarse[:ni_c * n_h].reshape(ni_c, n_h)
    b1 = coarse[ni_c * n_h:ni_c * n_h + n_h]
    w2c = coarse[ni_c * n_h + n_h:ni_c * n_h + n_h + n_h * no_c].reshape(n_h, no_c)
    b2c = coarse[-no_c:]
    w1 = np.zeros((ni, n_h))
    w1[:ni_c] = w1c                                          # new arm-input row = 0
    w2 = np.zeros((n_h, no))
    w2[:, :no_c] = w2c                                       # new arm-output column = 0
    b2 = np.zeros(no)
    b2[:no_c] = b2c
    return np.concatenate([w1.ravel(), b1, w2.ravel(), b2])


def rollout(params: np.ndarray, x0: np.ndarray, cfg: HierConfig, use_arm: bool) -> "tuple[np.ndarray, np.ndarray]":
    """Roll the policy; return (stopped, fell). The arm reaction reduces the pitch-driving momentum, in flight too."""
    v_run = x0[:, 0].copy()
    vx, ll, pitch = x0[:, 0].copy(), x0[:, 1].copy(), x0[:, 2].copy()
    theta_a = np.zeros(len(x0))
    fell = np.zeros(len(x0), dtype=bool)
    for i in range(int(round(cfg.horizon / cfg.dt))):
        t = i * cfg.dt
        stance = (t % cfg.cycle) < cfg.ts
        targ = np.where(t < cfg.t_stop, v_run, np.maximum(0.0, v_run * (1.0 - (t - cfg.t_stop) / cfg.ramp)))
        phase = np.full(len(x0), (t % cfg.cycle) / cfg.cycle)
        cols = [vx, ll, pitch, targ, phase] + ([theta_a] if use_arm else [])
        out = _forward(params, np.stack(cols, axis=1), cfg, use_arm)
        fx = np.where(stance, cfg.fx_max * out[:, 0], 0.0)
        a = np.where(stance, cfg.a_max * out[:, 1], 0.0)
        wa = np.zeros(len(x0))
        if use_arm:                                          # reaction-wheel arm — available in flight too
            wa = np.clip(cfg.wa_max * out[:, 2], -cfg.wa_max, cfg.wa_max)
            nxt = theta_a + wa * cfg.dt
            wa = np.where((np.abs(nxt) > cfg.arm_range) & (np.sign(wa) == np.sign(np.where(theta_a == 0, wa, theta_a))),
                          0.0, wa)
            theta_a = np.clip(theta_a + wa * cfg.dt, -cfg.arm_range, cfg.arm_range)
        ll = ll + (-cfg.l_damp * ll * stance + a - cfg.k_couple * fx) * cfg.dt
        pitch = pitch + ((ll - cfg.inertia_arm * wa) / cfg.inertia) * cfg.dt
        vx = vx + fx * cfg.dt
        fell |= np.abs(pitch) > cfg.fall_pitch
    return (np.abs(vx) < cfg.v_stop_tol) & ~fell, fell


def mixed_set(cfg: HierConfig, offset: float = 0.0, n: int = 7) -> np.ndarray:
    vs = np.array([1.5, 2.0, 2.5, 3.0]) + offset * 0.25
    ps = np.linspace(-0.3, 0.3, n) + offset * (0.6 / n)
    ls = np.linspace(-1.0, 1.0, n) + offset * (2.0 / n)
    vv, pp, ll = np.meshgrid(vs, ps, ls, indexing="ij")
    return np.stack([vv.ravel(), ll.ravel(), pp.ravel()], axis=1)


def train_cem(cfg: HierConfig, use_arm: bool, init: "np.ndarray | None" = None) -> np.ndarray:
    """CEM on the policy parameters (``init`` warm-starts the mean — the hierarchy transfer)."""
    rng = np.random.RandomState(cfg.seed)
    x0 = mixed_set(cfg, offset=0.0)
    dim = n_params(cfg, use_arm)
    mean = np.zeros(dim) if init is None else init.copy()
    std = np.full(dim, cfg.init_std)
    for _ in range(cfg.iters):
        pop = mean + std * rng.standard_normal((cfg.pop, dim))
        scores = np.array([rollout(p, x0, cfg, use_arm)[0].mean() for p in pop])
        elite = pop[np.argsort(scores)[-cfg.elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-6
    return mean


def evaluate(params: np.ndarray, cfg: HierConfig, use_arm: bool, offset: float = 0.5) -> float:
    """Held-out stop-success."""
    return float(rollout(params, mixed_set(cfg, offset=offset), cfg, use_arm)[0].mean())


def hierarchical_refine(cfg: HierConfig) -> dict:
    """The full hierarchy: coarse (foot-only) → warm-start → refine with the arm; vs the detailed model from scratch."""
    coarse = train_cem(cfg, use_arm=False)
    warm = train_cem(cfg, use_arm=True, init=warm_start(coarse, cfg))
    scratch = train_cem(cfg, use_arm=True, init=None)
    return {"coarse": evaluate(coarse, cfg, False),
            "refined_warm_start": evaluate(warm, cfg, True),
            "refined_from_scratch": evaluate(scratch, cfg, True)}
