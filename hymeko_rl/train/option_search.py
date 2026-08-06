"""Option-parameter search (Branch B/C) — CEM/ES over the 5-scalar PhasePushController option, NOT a per-step
residual.

The option interface is `PhasePushController(env, params=PushControllerParams.from_vector(θ))` with θ∈ℝ⁵ bounded
(`from_vector` clips to the envelope) and a PUSH/BRAKE safety arbiter that falls back to the scripted defaults — a
no-degrade shield, so a bad sample cannot drive the coin backward. This module runs CEM over θ, scored by a
**SearchObjective** built from the audit's per-episode contact statistics (sustained-PUSH windows + fingertip
progress + delivery − body-only progress − exploit − option magnitude). The frozen `TaskMonitor` is NOT the
objective — it is applied separately at acceptance (in the driver). The scripted-default θ is always kept in the
population (elitism floor → the search can never return worse-than-default on the objective).

Reuses the CEM control structure of `train/cem_residual.py` (population / elite / per-dim σ) but over θ (skill
parameters), and the single `sustained_push_windows` definition from `eval/push_audit.py`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from hymeko_rl.eval.push_audit import sustained_push_windows
from hymeko_rl.eval.task_monitor.contract import MonitorContract
from hymeko_rl.eval.task_monitor.context import MonitorContext, record_trajectory
from hymeko_rl.experiments.galambos_demo import PhasePushController, PushControllerParams

# the 5-scalar option box (matches PushControllerParams.from_vector clipping) + the scripted-default θ
PHASE_PUSH_LO = np.array([-0.010, 0.50, -0.35, 0.030, 0.004], dtype=np.float64)
PHASE_PUSH_HI = np.array([0.010, 1.50, 0.35, 0.120, 0.025], dtype=np.float64)
THETA_SCRIPTED = np.array([0.0, 1.0, 0.0, 0.06, 0.01], dtype=np.float64)   # = PushControllerParams() defaults
PARAM_NAMES = ("contact_offset", "push_gain", "direction_correction", "brake_threshold", "release_threshold")


@dataclass
class OptionSearchConfig:
    pop: int = 16
    elite: int = 4
    iters: int = 8
    sigma0: float = 0.35                 # per-dim σ as a fraction of the box width
    n_eval_eps: int = 6
    eval_seed: int = 20_000              # NOT the 9000 eval seed — keep the eval set unseen by the search
    max_steps: int = 300
    k_sustained: int = 5
    w_deliver: float = 1.0
    w_sustain: float = 0.30
    w_ftprog: float = 20.0
    w_body: float = 20.0
    w_exploit: float = 1.0
    w_magnitude: float = 0.10            # discourage large departures from the scripted option
    seed: int = 0
    log_every: int = 1


def _controller_action_fn(ctrl: PhasePushController) -> Callable[[Any, np.ndarray], np.ndarray]:
    def fn(env: Any, _obs: np.ndarray) -> np.ndarray:
        return np.asarray(ctrl.action(env), dtype=np.float32)
    return fn


def _default_make_ctrl(env: Any, theta: np.ndarray) -> Any:
    return PhasePushController(env, params=PushControllerParams.from_vector(np.asarray(theta, dtype=np.float64)))


def option_metrics(make_env: Callable[[], Any], theta: np.ndarray, contract: Any, cfg: OptionSearchConfig,
                   make_ctrl: "Callable[[Any, np.ndarray], Any] | None" = None) -> dict[str, float]:
    """Per-episode audit statistics for the controller parameterized by θ (delivery, sustained-PUSH windows,
    fingertip/body progress in contact, exploit) averaged over ``n_eval_eps``. ``make_ctrl(env, θ)`` builds the
    controller (default = ``PhasePushController(θ)``; pass a phase-gated hierarchy for option-RL v0)."""
    build = make_ctrl or _default_make_ctrl
    deliver = sustain = ft_in = body_in = exploit = arm_body = 0.0
    for k in range(cfg.n_eval_eps):
        env = make_env()
        ctrl = build(env, np.asarray(theta, dtype=np.float64))
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        traj = record_trajectory(env, _controller_action_fn(ctrl), cfg.eval_seed + k)
        if not traj:
            continue
        ctx = MonitorContext.build(traj, contract)
        delivered = bool(ctx.max_dwell >= contract.success_steps)
        deliver += float(delivered)
        exploit += float(delivered and ctx.body_prog > ctx.ft_prog)
        arm_body += float(bool(ctx.body_contact.any()))
        for s, e in sustained_push_windows(ctx, contract, cfg.k_sustained):
            sustain += 1.0
            ft_in += float(ctx.ft_prog_step[s:e].sum())
            body_in += float(ctx.body_prog_step[s:e].sum())
    ne = max(1, cfg.n_eval_eps)
    return {"delivery": deliver / ne, "sustained_push_per_ep": sustain / ne,
            "ft_progress_in_contact": ft_in / ne, "body_progress_in_contact": body_in / ne,
            "exploit_rate": exploit / ne, "arm_body_rate": arm_body / ne}


def _objective(m: dict[str, float], theta: np.ndarray, cfg: OptionSearchConfig) -> float:
    mag = float(np.linalg.norm((np.asarray(theta) - THETA_SCRIPTED) / (PHASE_PUSH_HI - PHASE_PUSH_LO)))
    return (cfg.w_deliver * m["delivery"] + cfg.w_sustain * m["sustained_push_per_ep"]
            + cfg.w_ftprog * m["ft_progress_in_contact"] - cfg.w_body * m["body_progress_in_contact"]
            - cfg.w_exploit * m["exploit_rate"] - cfg.w_magnitude * mag)


@dataclass
class OptionSearchResult:
    theta: list[float]
    theta_named: dict[str, float]
    best_objective: float
    baseline_objective: float
    best_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    history: list[float]
    improved_over_scripted: bool = field(init=False)

    def __post_init__(self):
        self.improved_over_scripted = self.best_objective > self.baseline_objective + 1e-9

    def as_dict(self) -> dict:
        return asdict(self)


def option_cem(make_env: Callable[[], Any], cfg: OptionSearchConfig, *,
               make_ctrl: "Callable[[Any, np.ndarray], Any] | None" = None, log=print) -> OptionSearchResult:
    """CEM over the 5-scalar option θ, directed by the SearchObjective.

    # Preconditions: ``make_env`` builds a PlanarGraspEnv; the scripted-default θ is a valid point.
    # Postconditions: best θ found (θ_scripted always evaluated + kept in the population → never worse than the
      default option on the objective); no actor/critic trained, no CORE mutation."""
    contract = MonitorContract.from_env(make_env())
    rng = np.random.default_rng(cfg.seed)
    sigma = cfg.sigma0 * (PHASE_PUSH_HI - PHASE_PUSH_LO)
    mean = THETA_SCRIPTED.copy()
    base_m = option_metrics(make_env, THETA_SCRIPTED, contract, cfg, make_ctrl)
    base_obj = _objective(base_m, THETA_SCRIPTED, cfg)
    best_theta, best_obj, best_m, history = THETA_SCRIPTED.copy(), base_obj, base_m, [base_obj]
    for it in range(cfg.iters):
        pop = rng.normal(mean, sigma, size=(cfg.pop, 5))
        pop = np.clip(pop, PHASE_PUSH_LO, PHASE_PUSH_HI)
        pop[0] = THETA_SCRIPTED                                   # elitism floor
        scored = [(_objective(m, th, cfg), th, m) for th, m in
                  ((th, option_metrics(make_env, th, contract, cfg, make_ctrl)) for th in pop)]
        scored.sort(key=lambda x: x[0])
        elite = np.array([th for _o, th, _m in scored[-cfg.elite:]])
        mean, sigma = elite.mean(0), elite.std(0) + 1e-4
        top_obj, top_th, top_m = scored[-1]
        if top_obj > best_obj:
            best_obj, best_theta, best_m = top_obj, top_th.copy(), top_m
        history.append(float(top_obj))
        if cfg.log_every and (it + 1) % cfg.log_every == 0:
            log(f"  [option-cem] iter {it + 1}/{cfg.iters} | best_obj {best_obj:+.3f} (base {base_obj:+.3f}) "
                f"| sustained/ep {best_m['sustained_push_per_ep']:.2f} deliv {best_m['delivery']:.2f} "
                f"| θ {np.round(best_theta, 3)}")
    return OptionSearchResult(
        theta=[float(x) for x in best_theta],
        theta_named={n: float(v) for n, v in zip(PARAM_NAMES, best_theta)},
        best_objective=best_obj, baseline_objective=base_obj, best_metrics=best_m,
        baseline_metrics=base_m, history=history)
