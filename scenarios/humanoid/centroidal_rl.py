r"""Neural RL for the centroidal L-regulation — a learned residual port, honestly held-out vs the scaffold.

The scripted regulator (``centroidal_step``) keeps most of the soft-regulated running basin upright, but not all.
This learns a **bounded residual** corrective L-torque ``a = a_max·π_θ(s)`` on top of it — the angular-momentum
port as a policy — trained by **CEM** (an evolutionary policy search: robust, deterministic, no autograd tuning
to confound the result) to maximise an upright-margin return. Per the repo's hard-won lesson
([[feedback-heldout-panel-is-single-use]]), the only claim that counts is **held-out**: the policy must beat the
scripted baseline on initial states it never trained on, or the scaffold wins.

The policy is a small MLP evaluated in numpy (so CEM is fast and dependency-light); TD3 / policy-gradient on the
same residual interface is the scaling follow-up. The residual enters through ``centroidal_step(l_residual=…)``.

# Preconditions: a soft-regulated ``CentroidalConfig`` (a basin with room to improve). # Postconditions:
#   ``evaluate`` returns held-out recover rates for both policy and scaffold; the policy action is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenarios.humanoid.centroidal import CentroidalConfig, centroidal_step


@dataclass(frozen=True)
class RLConfig:
    """Residual-policy + CEM hyper-parameters."""

    a_max: float = 4.0               # bound on the corrective L-torque
    hidden: int = 16
    horizon_s: float = 1.2
    w_ctrl: float = 0.02             # control-effort penalty
    pop: int = 48
    elite: int = 8
    iters: int = 25
    init_std: float = 0.5
    seed: int = 0


def regulation_task_config(l_damp: float = 2.0, **overrides) -> CentroidalConfig:
    """The angular-momentum regulation task: NO direct pitch hold (``pitch_gain=0``), so the torso pitch is
    controllable ONLY through the L port — a genuinely controllable RL task, with a weak ``l_damp`` leaving room.

    (A strong direct pitch hold makes ``L`` a spectator — verified: even ``a_max=20`` then yields Δ=0 vs the
    scaffold — so the port must be the sole authority for the RL task to be meaningful.)
    """
    return CentroidalConfig(pitch_gain=0.0, l_damp=l_damp, torque_bias=0.0, **overrides)


_N_IN = 4                            # policy features: (L, pitch, ż, phase_fraction)


def _sizes(rl: RLConfig) -> "tuple[int, int, int]":
    return _N_IN, rl.hidden, 1


def _n_params(rl: RLConfig) -> int:
    n_in, n_h, n_out = _sizes(rl)
    return n_in * n_h + n_h + n_h * n_out + n_out


def _features(state: np.ndarray, t: float, cfg: CentroidalConfig) -> np.ndarray:
    phase = (t % cfg.cycle) / cfg.cycle
    return np.stack([state[:, 2], state[:, 3], state[:, 1], np.full(len(state), phase)], axis=1)


def policy_action(params: np.ndarray, feats: np.ndarray, rl: RLConfig) -> np.ndarray:
    """Forward the numpy MLP → bounded residual L-torque in ``[−a_max, a_max]``. # Post: ``|a| ≤ a_max``."""
    n_in, n_h, n_out = _sizes(rl)
    a, b = n_in * n_h, n_in * n_h + n_h
    c = b + n_h * n_out
    w1 = params[:a].reshape(n_in, n_h)
    b1 = params[a:b]
    w2 = params[b:c].reshape(n_h, n_out)
    b2 = params[c:]
    h = np.tanh(feats @ w1 + b1)
    return rl.a_max * np.tanh(h @ w2 + b2)[:, 0]


def rollout(params: "np.ndarray | None", x0: np.ndarray, cfg: CentroidalConfig, rl: RLConfig,
            ) -> "tuple[np.ndarray, np.ndarray]":
    """Roll the residual policy (``params=None`` → the scripted scaffold) from ``x0``; return (return, fell)."""
    steps = int(round(rl.horizon_s / cfg.dt))
    state = x0.astype(float).copy()
    fell = np.zeros(len(x0), dtype=bool)
    margin, ctrl = np.zeros(len(x0)), np.zeros(len(x0))
    for k in range(steps):
        a = (np.zeros(len(x0)) if params is None
             else np.where(fell, 0.0, policy_action(params, _features(state, k * cfg.dt, cfg), rl)))
        margin += np.where(fell, -cfg.fall_pitch, cfg.fall_pitch - np.abs(state[:, 3]))
        ctrl += a ** 2
        state = centroidal_step(state, k * cfg.dt, cfg, l_residual=a)
        fell |= np.abs(state[:, 3]) > cfg.fall_pitch
    return margin / steps - rl.w_ctrl * (ctrl / steps), fell


def _grid(cfg: CentroidalConfig, n: int, offset: float) -> np.ndarray:
    ls = np.linspace(-cfg.l_max, cfg.l_max, n) + offset * (2 * cfg.l_max / n)
    ps = np.linspace(-cfg.pitch_max, cfg.pitch_max, n) + offset * (2 * cfg.pitch_max / n)
    ll, pp = np.meshgrid(ls, ps)
    return np.stack([np.full(ll.size, cfg.z0), np.zeros(ll.size), ll.ravel(), pp.ravel()], axis=1)


def train_cem(cfg: CentroidalConfig, rl: RLConfig, n_train: int = 13) -> np.ndarray:
    """CEM over the policy parameters on a TRAIN grid of initial states; return the elite-mean policy."""
    rng = np.random.RandomState(rl.seed)
    x0 = _grid(cfg, n_train, offset=0.0)
    dim = _n_params(rl)
    mean, std = np.zeros(dim), np.full(dim, rl.init_std)
    for _ in range(rl.iters):
        pop = mean + std * rng.standard_normal((rl.pop, dim))
        scores = np.array([rollout(p, x0, cfg, rl)[0].mean() for p in pop])
        elite = pop[np.argsort(scores)[-rl.elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-6
    return mean


def evaluate(params: "np.ndarray | None", cfg: CentroidalConfig, rl: RLConfig, n: int = 21,
             offset: float = 0.5) -> dict:
    """Recover rate + mean return on a (held-out by default, ``offset=0.5``) grid of initial states."""
    x0 = _grid(cfg, n, offset)
    ret, fell = rollout(params, x0, cfg, rl)
    return {"recover_rate": float(np.mean(~fell)), "mean_return": float(ret.mean()), "n": int(len(x0))}
