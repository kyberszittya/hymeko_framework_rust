"""Monitor-directed CEM over a low-dimensional bounded residual — the gate-shut fallback (no critic, no gradient).

If the vector-projected gradient is not monitor-aligned, more critic-gradient RL inherits the same obstruction.
The lever then is gradient-free search directed by the monitor components — user-authorized for this run with the
strict separation preserved: the **objective** is the ``SearchObjective`` per-step component signals; the **final
verifier** is the frozen ``TaskMonitor``. The residual is deliberately low-dimensional (one bounded per-joint
offset applied only in engaged CONTACT/PUSH phases) so a handful of CEM iterations suffice and the base DAgger
policy is never destabilized in APPROACH. θ=0 reproduces the frozen policy exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.eval.critic_benchmark import phase_label
from hymeko_rl.train.search_objective import SearchObjective

_ENGAGED = ("CONTACT", "PUSH", "DELIVERY")


@dataclass
class CEMConfig:
    epsilon: float = 0.4                 # residual bound in action space (‖residual‖ ≤ epsilon per joint via tanh)
    pop: int = 16
    elite: int = 4
    iters: int = 8
    sigma0: float = 0.6
    n_eval_eps: int = 6
    eval_seed: int = 9000
    max_steps: int = 300
    progress_eps: float = 0.002
    near_coin: float = 0.06
    body_penalty: float = 2.0            # objective weight discouraging body-only progress
    arm_body_penalty: float = 0.5
    seed: int = 0
    log_every: int = 1


def make_residual_action_fn(base_actor: Any, theta: np.ndarray, epsilon: float, lo: np.ndarray, hi: np.ndarray,
                            *, progress_eps: float, near_coin: float) -> Callable[[Any, np.ndarray], np.ndarray]:
    """A phase-gated bounded-residual policy: ``clip(π(s) + gate·epsilon·tanh(θ))``; gate=1 in CONTACT/PUSH/DELIVERY,
    0 in APPROACH (so approach is never disturbed). θ=0 ⇒ exactly the base policy."""
    resid = (epsilon * np.tanh(theta)).astype(np.float32)
    state = {"toward": 0.0, "prev": None}

    def action_fn(env: Any, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            base = base_actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float32)
        z = env.privileged_state()
        both = bool(z[0] > 0.5 and z[1] > 0.5)
        in_zone = bool(z[4] > 0.5)
        m = env._planar_metrics
        phase = phase_label(min_tip=float(min(m.left_tip_dist, m.right_tip_dist)), both_contact=both,
                            toward=state["toward"], in_zone=in_zone, near_coin=near_coin, progress_eps=progress_eps)
        gate = 1.0 if phase in _ENGAGED else 0.0
        cur = float(m.disk_to_zone)
        state["toward"] = max(0.0, state["prev"] - cur) if state["prev"] is not None else 0.0
        state["prev"] = cur
        return np.clip(base + gate * resid, lo, hi).astype(np.float32)

    return action_fn


def _episode_objective(env: Any, action_fn: Callable, objective: SearchObjective, seed: int,
                       max_steps: int, cfg: CEMConfig) -> float:
    """Monitor-component objective for one episode: fingertip progress + delivery − body/arm-body penalties.
    (This is the SEARCH objective, not the verifier — the frozen TaskMonitor grades acceptance separately.)"""
    obs, info = env.reset(seed=seed)
    prev = info["disk_to_zone"]
    ft = body = deliver = arm = 0.0
    for _ in range(max_steps):
        obs2, _r, term, trunc, info = env.step(np.asarray(action_fn(env, obs), np.float32))
        m = env._planar_metrics
        sig = objective.step_signals(
            prev_dist=prev, dist=float(info["disk_to_zone"]),
            min_tip=float(min(m.left_tip_dist, m.right_tip_dist)),
            both_contact=bool(info["both_contact"]), fingertip_contact=bool(info["fingertip_contact"]),
            arm_body_contact=bool(info["arm_body_contact_this_step"]), in_zone=bool(info["in_zone"]))
        ft += sig["progress"]
        body += sig["body_progress"]
        deliver += sig["delivery"]
        arm += float(info["arm_body_contact_this_step"])
        prev = float(info["disk_to_zone"])
        obs = obs2
        if term or trunc:
            break
    return ft + 0.01 * deliver - cfg.body_penalty * body - cfg.arm_body_penalty * 0.001 * arm


def _score_theta(env: Any, base_actor: Any, theta: np.ndarray, objective: SearchObjective,
                 lo: np.ndarray, hi: np.ndarray, cfg: CEMConfig) -> float:
    fn = make_residual_action_fn(base_actor, theta, cfg.epsilon, lo, hi,
                                 progress_eps=cfg.progress_eps, near_coin=cfg.near_coin)
    return float(np.mean([_episode_objective(env, fn, objective, cfg.eval_seed + k, cfg.max_steps, cfg)
                          for k in range(cfg.n_eval_eps)]))


@dataclass
class CEMResult:
    theta: np.ndarray
    best_objective: float
    baseline_objective: float
    history: list[float]
    dim: int


def cem_optimize(env: Any, base_actor: Any, cfg: CEMConfig, *,
                 objective: SearchObjective | None = None, log=print) -> CEMResult:
    """CEM over the bounded per-joint residual θ, directed by the monitor-component objective.

    # Preconditions: ``base_actor`` maps obs→action; ``env`` is the same PlanarGraspEnv family the actor trained on.
    # Postconditions: returns the best θ found (θ=0 always evaluated as the frozen baseline, so the result can
      never be worse than the base policy on the search objective); no actor weights are modified."""
    objective = objective or SearchObjective(near_coin=cfg.near_coin, progress_eps=cfg.progress_eps)
    rng = np.random.default_rng(cfg.seed)
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)
    dim = int(env.n_actions)
    base_obj = _score_theta(env, base_actor, np.zeros(dim, np.float32), objective, lo, hi, cfg)
    mean = np.zeros(dim)
    sigma = np.full(dim, cfg.sigma0)
    best_theta, best_obj, history = np.zeros(dim, np.float32), base_obj, [base_obj]
    for it in range(cfg.iters):
        pop = rng.normal(mean, sigma, size=(cfg.pop, dim)).astype(np.float32)
        pop[0] = 0.0  # always keep the baseline in the population (elitism floor)
        scores = np.array([_score_theta(env, base_actor, th, objective, lo, hi, cfg) for th in pop])
        elite_idx = np.argsort(scores)[-cfg.elite:]
        elite = pop[elite_idx]
        mean, sigma = elite.mean(0), elite.std(0) + 1e-3
        if scores.max() > best_obj:
            best_obj, best_theta = float(scores.max()), pop[int(np.argmax(scores))].astype(np.float32)
        history.append(float(scores.max()))
        if cfg.log_every and (it + 1) % cfg.log_every == 0:
            log(f"  [cem] iter {it + 1}/{cfg.iters} | best {best_obj:+.4f} (base {base_obj:+.4f}) "
                f"| elite μ {np.round(mean, 3)}")
    return CEMResult(theta=best_theta, best_objective=best_obj, baseline_objective=base_obj,
                     history=history, dim=dim)
