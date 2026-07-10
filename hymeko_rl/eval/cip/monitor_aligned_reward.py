"""A monitor-aligned, phase-aware, anti-farming reward variant for MetaWorld pick-place.

The Stage-A finding: ablating ``mw_in_place`` re-parents the reward toward delivery, and BC-anchored policies show
rising reward↔monitor disagreement under the ablated reward. This module builds a *repaired* dense reward whose
shaping cannot be decoupled from task progress: approach + gated contact/lift + **gated delivery** (only after
grasp/lift evidence) + a large success bonus − a stagnation (farming) penalty. It is NOT a sparse reward — it stays
dense — but the dominant terms (delivery, success) require real manipulation, so proximity/contact alone cannot
farm it.

Signals are geometric + monitor flags per step (verified layout): hand=obs[:3], object=obs[4:7]; info exposes
``obj_to_target``, ``near_object``, ``grasp_success``, ``success``. This variant is stateful (needs the previous
step), so it is a reward *wrapper* / trajectory function, not a HyMeKo ``Σ weight·term`` spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .metaworld_reward import _DEFAULT_SPEC, _TERM_TO_COMPONENT

_LIFT_EPS = 0.02          # object height above its rest to count as "lifted"
_MOVE_EPS = 0.002         # obj-to-target change to count the object as "moving"


@dataclass(frozen=True)
class MonitorAlignedWeights:
    """Component weights — delivery and success dominate so contact/proximity alone cannot farm the reward."""
    approach: float = 1.0
    contact: float = 0.5
    lift: float = 5.0
    delivery: float = 10.0
    success: float = 50.0
    stagnation: float = 1.0


DEFAULT_WEIGHTS = MonitorAlignedWeights()


def monitor_aligned_step(sig: "Mapping[str, float]", w: MonitorAlignedWeights = DEFAULT_WEIGHTS) -> float:
    """Per-step monitor-aligned reward from the step signals (see module docstring). Pure — used by the R1 tests.

    # Preconditions ``sig`` has d, d_prev, obj_z, obj_z0, ott, ott_prev, grasp, near, success.
    # Postconditions delivery/lift terms are gated on grasp/lift evidence; contact is capped without object motion.
    """
    approach = float(sig["d_prev"] - sig["d"])                    # potential-based; telescopes to (d0 − dT)
    grasp = float(sig["grasp"])
    lift = max(0.0, float(sig["obj_z"] - sig["obj_z0"]))
    established = grasp > 0.5 or lift > _LIFT_EPS                  # real-manipulation gate
    delivery = float(sig["ott_prev"] - sig["ott"])                # reduction in object→target distance
    obj_moving = abs(delivery) > _MOVE_EPS or lift > _LIFT_EPS
    gated_delivery = delivery if established else 0.0             # delivery only counts after grasp/lift
    gated_lift = lift * grasp                                     # lifting counts only while grasped
    contact_capped = grasp * (1.0 if obj_moving else 0.2)         # anti-farming: bare contact (no motion) is capped
    stagnation = 1.0 if (sig["near"] > 0.5 and grasp < 0.5 and not obj_moving) else 0.0   # hover-farming penalty
    return (w.approach * approach + w.contact * contact_capped + w.lift * gated_lift
            + w.delivery * gated_delivery + w.success * float(sig["success"]) - w.stagnation * stagnation)


class MonitorAlignedReward:
    """Stateful reward wrapper: tracks the previous step so the per-step monitor-aligned reward is computable live."""

    def __init__(self, w: MonitorAlignedWeights = DEFAULT_WEIGHTS) -> None:
        self.w = w
        self._d_prev: float = 0.0
        self._ott_prev: float = 0.0
        self._obj_z0: float = 0.0

    @staticmethod
    def _geom(obs: np.ndarray, info: "Mapping[str, Any]") -> "tuple[float, float, float]":
        hand, obj = obs[:3], obs[4:7]
        return float(np.linalg.norm(hand - obj)), float(obj[2]), float(info.get("obj_to_target", 0.0))

    def reset(self, obs: np.ndarray, info: "Mapping[str, Any]") -> None:
        self._d_prev, self._obj_z0, self._ott_prev = self._geom(obs, info)

    def step(self, obs: np.ndarray, info: "Mapping[str, Any]") -> float:
        d, obj_z, ott = self._geom(obs, info)
        sig = {"d": d, "d_prev": self._d_prev, "obj_z": obj_z, "obj_z0": self._obj_z0, "ott": ott,
               "ott_prev": self._ott_prev, "grasp": float(info.get("grasp_success", 0.0)),
               "near": float(info.get("near_object", 0.0)), "success": float(info.get("success", 0.0))}
        r = monitor_aligned_step(sig, self.w)
        self._d_prev, self._ott_prev = d, ott
        return r


# ---- Stage R2: offline recomputation through the CIP/LiNGAM pipeline ----

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation across rollouts (0 if either is constant)."""
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def _roll_one(task: str, policy: Any, seed_i: int, noise: float, comp_keys: "list[tuple[str, float]]",
              weights: MonitorAlignedWeights, max_steps: int) -> "dict[str, Any]":
    """One scripted+noise episode → per-step reward components, the monitor-aligned total, and CIP-frame aggregates."""
    from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
    env = ENVS[f"{task}-v3-goal-observable"](render_mode=None)
    obs, _ = env.reset(seed=seed_i)
    mar = MonitorAlignedReward(weights)
    mar.reset(obs, {"obj_to_target": float(np.linalg.norm(obs[4:7] - obs[-3:]))})
    rng = np.random.default_rng(seed_i)
    x_step: list[list[float]] = []
    y_step: list[float] = []
    csum = np.zeros(len(comp_keys))
    mon = 0.0
    n_near = n_grasp = steps = 0
    ip_sum = 0.0
    ott0: "float | None" = None
    ott_final = 0.0
    ok = False
    for _ in range(max_steps):
        act = np.clip(np.asarray(policy.get_action(obs), np.float32) + rng.normal(0, noise, 4).astype(np.float32),
                      -1.0, 1.0)
        obs, reward, term, trunc, info = env.step(act)
        cvec = [float(sign) * float(info.get(key, 0.0)) for key, sign in comp_keys]
        x_step.append(cvec)
        y_step.append(float(info.get("unscaled_reward", reward)))
        csum += np.asarray(cvec)
        mon += mar.step(obs, info)
        steps += 1
        n_near += int(info.get("near_object", 0))
        n_grasp += int(info.get("grasp_success", 0))
        ip_sum += float(info.get("in_place_reward", 0.0))
        ott = float(info.get("obj_to_target", 0.0))
        ott_final = ott
        ok = ok or bool(info.get("success", 0.0))
        if ott0 is None:
            ott0 = ott
        if term or trunc:
            break
    return {"x_step": x_step, "y_step": y_step, "totals": csum, "monitor": mon, "near": n_near / max(1, steps),
            "grasp": n_grasp / max(1, steps), "delivery": (ott0 or 0.0) - ott_final,
            "progress": ip_sum / max(1, steps), "success": int(ok)}


def _pack_rollouts(eps: "list[dict[str, Any]]") -> "dict[str, Any]":
    """Stack per-episode records into the flat/per-rollout arrays the CIP pipeline consumes."""
    def col(k: str) -> np.ndarray:
        return np.asarray([e[k] for e in eps])
    return {"x_step": np.asarray([c for e in eps for c in e["x_step"]]),
            "y_step": np.asarray([v for e in eps for v in e["y_step"]]),
            "totals": col("totals"), "monitor_aligned": col("monitor"),
            "cip": {"near_fraction": col("near"), "grasp_fraction": col("grasp"),
                    "obj_to_target_delta": col("delivery"), "progress_score": col("progress")},
            "success": col("success").astype(float)}


def _record_monitor_rollouts(task: str, policy_name: str, kinds: "list[str]", n: int, seed: int,
                             noise_max: float, weights: MonitorAlignedWeights) -> "dict[str, Any]":
    """Record ``n`` scripted rollouts → components (original/off), the monitor-aligned total, and the CIP frame."""
    import warnings
    comp_keys = [_TERM_TO_COMPONENT[k] for k in kinds]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import metaworld.policies as mp
        noises = np.random.default_rng(seed).uniform(0.0, noise_max, n)
        eps = [_roll_one(task, getattr(mp, policy_name)(), seed + i, noises[i], comp_keys, weights, 180)
               for i in range(n)]
    return _pack_rollouts(eps)


def _variant_metrics(cond: "dict[str, Any]", reward: np.ndarray, cip: "dict[str, np.ndarray]") -> "dict[str, Any]":
    """R2 metrics for one recomputed reward: disagreement, corr(delivery/progress/contact), farming candidate."""
    delivery, progress = cip["obj_to_target_delta"], cip["progress_score"]
    contact_prox = cip["near_fraction"] + cip["grasp_fraction"]
    loadings = cond["loadings"]
    strongest = max(loadings, key=lambda k: abs(loadings[k])) if loadings else None
    return {"reward_monitor_disagreement": cond["reward_monitor_disagreement"],
            "corr_delivery": _corr(reward, delivery), "corr_progress": _corr(reward, progress),
            "corr_contact_proximity": _corr(reward, contact_prox),
            "reward_farming_candidate": round(_corr(reward, contact_prox) - _corr(reward, delivery), 4),
            "loadings": loadings, "strongest_incoming_edge": strongest,
            "reward_std": round(float(np.std(reward)), 4), "cross_view_agree": cond["cross_view_agree"]}


def run_monitor_aligned_comparison(task: str = "pick-place", spec_path: str = _DEFAULT_SPEC, *, n: int = 60,
                                   seed: int = 0, out_dir: "Any" = None, noise_max: float = 0.7,
                                   weights: MonitorAlignedWeights = DEFAULT_WEIGHTS) -> "dict[str, Any]":
    """Stage R2 — recompute original / mw_in_place_off / monitor_aligned on shared rollouts, run CIP for each."""
    import json
    from pathlib import Path

    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    from hymeko_rl.eval.cip.metaworld_reward import fit_reward_weights, hymeko_reward_terms, reward_mechanism_proposal
    from hymeko_rl.eval.cip.reward_ablation_metaworld import _condition
    from hymeko_rl.eval.evaluate import experiment_dir
    out = Path(out_dir) if out_dir is not None else experiment_dir("reports/figures", "monitor_aligned_reward")
    out.mkdir(parents=True, exist_ok=True)
    kinds = [k for k, _w in hymeko_reward_terms(spec_path)]
    rec = _record_monitor_rollouts(task, GENERIC_TASKS[task], kinds, n, seed + 11_000, noise_max, weights)
    fitted, r2 = fit_reward_weights(rec["x_step"], rec["y_step"])
    off_w = np.array([0.0 if k == "mw_in_place" else fitted[i] for i, k in enumerate(kinds)])
    rewards = {"original": rec["totals"] @ fitted, "mw_in_place_off": rec["totals"] @ off_w,
               "monitor_aligned": rec["monitor_aligned"]}
    # the task monitor for disagreement is delivery success (the real task signal), shared across variants
    monitor = rec["success"] if float(np.std(rec["success"])) > 1e-9 else rec["cip"]["obj_to_target_delta"]
    prop = reward_mechanism_proposal(spec_path, available=[*rec["cip"], "total_reward"])
    variants = {name: _variant_metrics(_condition(rec["cip"], reward, monitor, prop, name, out), reward, rec["cip"])
                for name, reward in rewards.items()}
    summary = {"task": task, "n": n, "seed": seed, "reward_fidelity_r2": round(r2, 4),
               "fitted_weights": {k: round(float(w), 4) for k, w in zip(kinds, fitted)},
               "monitor": "success" if float(np.std(rec["success"])) > 1e-9 else "delivery", "variants": variants}
    (out / "monitor_aligned_comparison.json").write_text(json.dumps(summary, indent=2, default=float))
    for name, v in variants.items():
        print(f"[monitor-aligned] {name:16s} disagree={v['reward_monitor_disagreement']:.3f} "
              f"corr_delivery={v['corr_delivery']:+.2f} corr_progress={v['corr_progress']:+.2f} "
              f"farming={v['reward_farming_candidate']:+.2f} xview={v['cross_view_agree']}", flush=True)
    return summary


# ---- Stage R3: BC-anchored fine-tune smoke (live env wrapper) ----

class MonitorAlignedEnv:
    """Live reward-override: the step reward is the stateful monitor-aligned reward (env reward kept in info)."""

    def __init__(self, env: Any, weights: MonitorAlignedWeights = DEFAULT_WEIGHTS) -> None:
        self.env = env
        self._mar = MonitorAlignedReward(weights)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kw: Any) -> Any:
        obs, info = self.env.reset(**kw)
        self._mar.reset(obs, {"obj_to_target": float(np.linalg.norm(obs[4:7] - obs[-3:]))})
        return obs, info

    def step(self, action: Any) -> Any:
        obs, reward, term, trunc, info = self.env.step(action)
        r = self._mar.step(obs, info)
        return obs, r, term, trunc, {**info, "env_reward": float(reward), "monitor_aligned_reward": r}


def _eval_trained(cfg: Any, env_builder: Any, policy: Any, n: int, seed0: int = 70_000) -> "dict[str, Any]":
    """Greedy eval of a trained policy under its own reward env: behaviour metrics + reward↔success disagreement."""
    import warnings

    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor
    rows: list[dict[str, Any]] = []
    succ, grasp, near, deliv, prog = [], [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            env = env_builder()
            obs, _ = env.reset(seed=seed0 + i)
            rtot = ip = 0.0
            ng = nn = steps = 0
            ott0: "float | None" = None
            ott = 0.0
            ok = False
            for _ in range(cfg.max_steps):
                obs, r, term, trunc, info = env.step(policy.greedy(obs))
                rtot += float(r)
                steps += 1
                ng += int(info.get("grasp_success", 0))
                nn += int(info.get("near_object", 0))
                ip += float(info.get("in_place_reward", 0.0))
                ott = float(info.get("obj_to_target", 0.0))
                ok = ok or bool(info.get("success", 0.0))
                if ott0 is None:
                    ott0 = ott
                if term or trunc:
                    break
            rows.append({"policy": f"e{i}", "total_reward": rtot, "monitor_score": float(ok)})
            succ.append(int(ok))
            grasp.append(ng / max(1, steps))
            near.append(nn / max(1, steps))
            deliv.append((ott0 or 0.0) - ott)
            prog.append(ip / max(1, steps))
    dis = round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 4)
    return {"success": round(float(np.mean(succ)), 3), "grasp": round(float(np.mean(grasp)), 3),
            "near": round(float(np.mean(near)), 3), "delivery": round(float(np.mean(deliv)), 3),
            "progress": round(float(np.mean(prog)), 3), "reward_monitor_disagreement": dis}


def run_monitor_aligned_bc_smoke(task: str = "pick-place", spec_path: str = _DEFAULT_SPEC, *, budget: int = 8_000,
                                 n_eval: int = 16, out_dir: Any = None,
                                 weights: MonitorAlignedWeights = DEFAULT_WEIGHTS) -> "dict[str, Any]":
    """Stage R3 — BC-anchored PPO fine-tune under original vs monitor_aligned; does monitor_aligned keep the policy?"""
    import json
    from pathlib import Path

    from hymeko_rl.eval.cip.reward_ablation_metaworld import ablate_reward_spec
    from hymeko_rl.eval.evaluate import experiment_dir
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig, _bc_base_policy, make_training_env
    from hymeko_rl.experiments.stage_b_ppo import train_ppo_flat
    out = Path(out_dir) if out_dir is not None else experiment_dir("reports/figures", "monitor_aligned_bc_smoke")
    out.mkdir(parents=True, exist_ok=True)
    cfg = StageBConfig(task=task, optimizer="ppo", warm_start=True, total_env_steps=budget, eval_episodes_post=n_eval)
    spec = ablate_reward_spec(spec_path)
    base, bc_info = _bc_base_policy(cfg)
    base_state = base.state_dict()
    builders = {"original": lambda: make_training_env(cfg, spec),
                "monitor_aligned": lambda: MonitorAlignedEnv(make_training_env(cfg, spec), weights)}
    results: dict[str, Any] = {}
    for name, builder in builders.items():
        out_train = train_ppo_flat(cfg, builder(), base_state, cfg.seed,  # type: ignore[no-untyped-call]  # cfg is Any
                                   log=lambda _m: None)
        ev = _eval_trained(cfg, builder, out_train["policy"], n_eval)
        results[name] = {**ev, "final_return": round(out_train["returns"][-1], 2) if out_train["returns"] else None}
        print(f"[monitor-aligned R3] {name:16s} success={ev['success']} grasp={ev['grasp']} near={ev['near']} "
              f"delivery={ev['delivery']} disagree={ev['reward_monitor_disagreement']} ret={results[name]['final_return']}",
              flush=True)
    summary = {"task": task, "budget": budget, "n_eval": n_eval, "bc": bc_info, "results": results}
    (out / "monitor_aligned_bc_smoke.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
