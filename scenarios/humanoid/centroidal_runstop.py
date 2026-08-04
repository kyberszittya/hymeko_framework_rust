r"""Run-and-stop — the hard centroidal task where a neural policy can beat a tuned linear one, with a HSTL reward.

Running then STOPPING is genuinely hard: braking (a foot force that decelerates ``vx``) sits below the CoM, so it
induces an angular momentum ``L`` — a forward pitch torque — that must be regulated *at the same time*, and the
control acts only in **stance** (no foot force in flight, where the pitch grows ballistically), under bounded
``|fx| ≤ fx_max`` / ``|a| ≤ a_max``. Across a range of stopping speeds no single linear gain is optimal
(stopping needs gain-scheduling + saturation-aware timing), so a nonlinear policy has genuine headroom — the best
single linear gain reaches only ~0.75 stop-success on the mixed set.

This ties the RL to the verification arc via **(2)**: the reward is the **HSTL monitor robustness** of the safety
spec ``G(fall_margin > 0)`` — i.e. the worst-case upright margin over the episode — which the robust-STL Globally
semantics make exactly ``min_t (fall_pitch − |pitch|)`` (verified against the actual monitor). The learned policy
is a numpy MLP trained by CEM; the honest metric is held-out stop-success vs the tuned linear baseline.

# Preconditions: a ``RunStopConfig`` in the headroom regime. # Postconditions: bounded actions; the reward equals
#   the HSTL ``G(fall_margin>0)`` robustness; ``evaluate`` returns held-out stop-success for policy and baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunStopConfig:
    """Run-stop dynamics + task parameters (the headroom regime where linear tops out near 0.75)."""

    dt: float = 0.004
    inertia: float = 1.6
    l_damp: float = 1.5
    fall_pitch: float = 1.25
    ts: float = 0.2                  # stance / flight durations (control only in stance)
    tf: float = 0.1
    k_couple: float = 0.9            # braking → angular-momentum coupling (foot force below the CoM)
    a_max: float = 1.5               # L-port bound
    fx_max: float = 2.5              # braking-force bound
    t_stop: float = 0.3              # when the stop command begins
    ramp: float = 0.4                # target-speed ramp-down time
    horizon: float = 1.8
    v_stop_tol: float = 0.3          # |vx| below this at the end = stopped

    @property
    def cycle(self) -> float:
        return self.ts + self.tf


def target_speed(t: float, v_run: np.ndarray, cfg: RunStopConfig) -> np.ndarray:
    """The commanded speed: hold ``v_run``, then ramp to 0 after ``t_stop``."""
    return np.where(t < cfg.t_stop, v_run, np.maximum(0.0, v_run * (1.0 - (t - cfg.t_stop) / cfg.ramp)))


def runstop_step(state: np.ndarray, t: float, fx: np.ndarray, a: np.ndarray, cfg: RunStopConfig) -> np.ndarray:
    r"""One step of the run-stop dynamics. ``state`` columns = (vx, L, pitch); control (fx, a) acts only in stance."""
    stance = (t % cfg.cycle) < cfg.ts
    fx = np.where(stance, fx, 0.0)
    a = np.where(stance, a, 0.0)
    vx = state[:, 0] + fx * cfg.dt
    ll = state[:, 1] + (-cfg.l_damp * state[:, 1] * stance + a - cfg.k_couple * fx) * cfg.dt   # braking induces L
    pitch = state[:, 2] + (ll / cfg.inertia) * cfg.dt
    return np.stack([vx, ll, pitch], axis=1)


@dataclass(frozen=True)
class PolicyConfig:
    hidden: int = 24
    pop: int = 56
    elite: int = 10
    iters: int = 30
    init_std: float = 0.4
    w_margin: float = 0.3            # weight on the HSTL-robustness shaping term
    seed: int = 0


_N_IN, _N_OUT = 5, 2                  # features (vx, L, pitch, target, phase); actions (fx, a)


def _n_params(pc: PolicyConfig) -> int:
    return _N_IN * pc.hidden + pc.hidden + pc.hidden * _N_OUT + _N_OUT


def policy_actions(params: np.ndarray, feats: np.ndarray, pc: PolicyConfig, cfg: RunStopConfig,
                   ) -> "tuple[np.ndarray, np.ndarray]":
    """Numpy MLP → bounded (fx, a). # Post: ``|fx| ≤ fx_max``, ``|a| ≤ a_max``."""
    a1, b1 = _N_IN * pc.hidden, _N_IN * pc.hidden + pc.hidden
    c1 = b1 + pc.hidden * _N_OUT
    w1 = params[:a1].reshape(_N_IN, pc.hidden)
    out = np.tanh(np.tanh(feats @ w1 + params[a1:b1]) @ params[b1:c1].reshape(pc.hidden, _N_OUT) + params[c1:])
    return cfg.fx_max * out[:, 0], cfg.a_max * out[:, 1]


def _features(state: np.ndarray, targ: np.ndarray, t: float, cfg: RunStopConfig) -> np.ndarray:
    phase = np.full(len(state), (t % cfg.cycle) / cfg.cycle)
    return np.stack([state[:, 0], state[:, 1], state[:, 2], targ, phase], axis=1)


def safety_shield(fx: np.ndarray, a: np.ndarray, state: np.ndarray, cfg: RunStopConfig,
                  buffer: float = 0.55, tau: float = 0.45) -> "tuple[np.ndarray, np.ndarray]":
    r"""A model-predictive safety shield — makes exploration safe by construction.

    The fall is relative-degree-2 (``a → L → pitch``), so reacting to ``|pitch|`` alone is too late: the shield
    activates on the **predicted** pitch ``pitch + (L/I)·τ`` (where the current angular momentum is taking the
    torso). As that predicted margin drops below ``buffer`` it blends the action toward the safe fallback — **stop
    braking** (the destabilising input to ``L``) and drive the L-port to **oppose the predicted pitch**
    (``a → −sign·a_max``) — so any policy explores without falling. Graded (smooth) to avoid thrashing.
    """
    pitch_pred = state[:, 2] + (state[:, 1] / cfg.inertia) * tau   # where the current L is taking the pitch
    w = np.clip((buffer - (cfg.fall_pitch - np.abs(pitch_pred))) / buffer, 0.0, 1.0)   # 0 far, 1 at the predicted edge
    fx_s = (1.0 - w) * fx                                    # cut braking as the predicted edge nears
    a_s = (1.0 - w) * a + w * (-np.sign(pitch_pred) * cfg.a_max)
    return np.clip(fx_s, -cfg.fx_max, cfg.fx_max), np.clip(a_s, -cfg.a_max, cfg.a_max)


def episode(params: "np.ndarray | None", x0: np.ndarray, cfg: RunStopConfig, pc: PolicyConfig,
            gains: "tuple[float, float]" = (4.0, 1.0), shield: bool = False,
            ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Roll the policy (``params``) or the tuned linear baseline (``params=None``) → (stopped, fell, min_margin).

    ``min_margin`` is the HSTL robustness of ``G(fall_margin>0)`` (= min over the episode of ``fall_pitch−|pitch|``).
    With ``shield=True`` every action is passed through :func:`safety_shield` (safe exploration).
    """
    v_run = x0[:, 0].copy()
    state = x0.astype(float).copy()
    fell = np.zeros(len(x0), dtype=bool)
    min_margin = np.full(len(x0), cfg.fall_pitch)
    for i in range(int(round(cfg.horizon / cfg.dt))):
        t = i * cfg.dt
        targ = target_speed(t, v_run, cfg)
        if params is None:
            fx = np.clip(-gains[0] * (state[:, 0] - targ), -cfg.fx_max, cfg.fx_max)
            a = np.clip(-gains[1] * state[:, 1], -cfg.a_max, cfg.a_max)
        else:
            fx, a = policy_actions(params, _features(state, targ, t, cfg), pc, cfg)
        if shield:
            fx, a = safety_shield(fx, a, state, cfg)
        state = runstop_step(state, t, fx, a, cfg)
        min_margin = np.minimum(min_margin, cfg.fall_pitch - np.abs(state[:, 2]))
        fell |= np.abs(state[:, 2]) > cfg.fall_pitch
    stopped = (np.abs(state[:, 0]) < cfg.v_stop_tol) & ~fell
    return stopped, fell, min_margin


def mixed_set(cfg: RunStopConfig, n: int = 8, offset: float = 0.0) -> np.ndarray:
    """A grid of initial (vx=v_run, L, pitch) spanning stopping speeds and perturbations."""
    vs = np.array([1.5, 2.0, 2.5, 3.0]) + offset * 0.25
    ps = np.linspace(-0.3, 0.3, n) + offset * (0.6 / n)
    ls = np.linspace(-1.0, 1.0, n) + offset * (2.0 / n)
    vv, pp, ll = np.meshgrid(vs, ps, ls, indexing="ij")
    return np.stack([vv.ravel(), ll.ravel(), pp.ravel()], axis=1)


def train_cem(cfg: RunStopConfig, pc: PolicyConfig, shield: bool = False) -> np.ndarray:
    """CEM on the policy parameters, maximising stop-success + the HSTL-robustness shaping term (train set)."""
    rng = np.random.RandomState(pc.seed)
    x0 = mixed_set(cfg, offset=0.0)
    dim = _n_params(pc)
    mean, std = np.zeros(dim), np.full(dim, pc.init_std)
    for _ in range(pc.iters):
        pop = mean + std * rng.standard_normal((pc.pop, dim))
        scores = np.empty(pc.pop)
        for j, p in enumerate(pop):
            stopped, _, margin = episode(p, x0, cfg, pc, shield=shield)
            scores[j] = stopped.mean() + pc.w_margin * margin.mean()
        elite = pop[np.argsort(scores)[-pc.elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-6
    return mean


def evaluate(params: "np.ndarray | None", cfg: RunStopConfig, pc: PolicyConfig, offset: float = 0.5,
             shield: bool = False) -> dict:
    """Held-out (by default) stop-success + mean HSTL robustness for a policy or the tuned linear baseline."""
    stopped, fell, margin = episode(params, mixed_set(cfg, offset=offset), cfg, pc, shield=shield)
    return {"stop_success": float(stopped.mean()), "fall_rate": float(fell.mean()),
            "mean_robustness": float(margin.mean()), "n": int(len(stopped))}
