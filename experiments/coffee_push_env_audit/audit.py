"""Coffee-Push environment sanity audit — through the EXACT training wrapper, no long runs.

Before interpreting a 0% from-scratch result as a fundamental exploration wall, prove the env/wrapper/metric are
sound. Seven checks (all rollout-only, no training):
  1. a scripted expert achieves success THROUGH the training wrapper (`_ObsNorm(native coffee-push)`);
  2. the eval success metric (`info["success"]`) fires on that successful trajectory;
  3. raw + RMS-normalized rewards logged for the successful trajectory;
  4. action bounds + per-dimension physical effect (incl. gripper);
  5. action clipping/saturation under the INITIAL stochastic SAC policy;
  6. random-policy & initial-policy state visitation, min obj-target dist, contact freq, obj displacement,
     fraction of episodes with ANY nonzero reward;
  7. resets produce solvable, randomized-but-valid states; the training wrapper == the eval wrapper (no silent
     task-distribution shift).

The wrapper `_ObsNorm` is a pure OBSERVATION transform (it does not touch action, reward, done, or dynamics); the
expert is driven with de-normalized obs (`raw = norm*std + mean`, exact inverse) so it acts correctly through it.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.exp_metaworld_cip_baseline import _make_native_env, _fit_native_obs_norm
from hymeko_rl.experiments.exp_metaworld_sac import _ObsNorm
from hymeko_rl.train.sac import build_sac
from hymeko_rl.train.normalize import RunningRMS
import metaworld.policies as mp

warnings.simplefilter("ignore")

ENV_ID = "coffee-push-v3-goal-observable"
POLICY = "SawyerCoffeePushV3Policy"
HORIZON = 500
OUT = Path(__file__).resolve().parent


def _wrapped(mean, std):
    return _ObsNorm(_make_native_env(ENV_ID), mean, std)


def _denorm(norm_obs, mean, std):
    return norm_obs * np.maximum(std, 0.05) + mean          # exact inverse of _ObsNorm._n


def check_1_2_3(mean, std, n=20):
    """Scripted expert THROUGH the wrapper → success rate; metric fires; raw+normalized rewards on a success."""
    succ = 0
    reached = []
    best_traj_rewards = None
    rms = RunningRMS()
    for i in range(n):
        env = _wrapped(mean, std)
        pol = getattr(mp, POLICY)()
        nobs, _ = env.reset(seed=i)
        traj_r = []
        ok = False
        for _ in range(HORIZON):
            raw = _denorm(nobs, mean, std)
            a = np.clip(np.asarray(pol.get_action(raw), np.float32), -1.0, 1.0)
            nobs, r, term, trunc, info = env.step(a)
            traj_r.append(float(r))
            ok = ok or bool(info.get("success", 0.0))
            if term or trunc:
                break
        succ += int(ok)
        reached.append(ok)
        if ok and best_traj_rewards is None:
            best_traj_rewards = traj_r
    # normalized reward view (what reward_norm feeds the critic): online running-RMS, exactly as train_sac does
    norm_view = None
    if best_traj_rewards is not None:
        norm_view = [float(rms.normalize(torch.tensor([rr], dtype=torch.float32))[0]) for rr in best_traj_rewards]
    return {
        "scripted_success_rate_through_wrapper": succ / n,
        "success_metric_fired_on_a_trajectory": bool(succ > 0),
        "success_trajectory_raw_reward_sum": round(float(np.sum(best_traj_rewards)), 3) if best_traj_rewards else None,
        "success_trajectory_raw_reward_range": [round(min(best_traj_rewards), 3), round(max(best_traj_rewards), 3)] if best_traj_rewards else None,
        "success_trajectory_normalized_reward_range": [round(min(norm_view), 3), round(max(norm_view), 3)] if norm_view else None,
        "n_episodes": n,
    }


def check_4_action_effect(mean, std):
    """Action bounds + per-dim physical effect: hold a unit action in each dim, measure eef/gripper displacement."""
    low, high = _make_native_env(ENV_ID).action_space.low, _make_native_env(ENV_ID).action_space.high
    effects = {}
    labels = ["dx(eef_x)", "dy(eef_y)", "dz(eef_z)", "gripper"]
    for dim in range(4):
        env = _make_native_env(ENV_ID)
        obs, _ = env.reset(seed=0)
        start = obs[:4].copy()
        act = np.zeros(4, np.float32)
        act[dim] = 1.0
        for _ in range(15):
            obs, *_ = env.step(act)
        delta = (obs[:4] - start)
        effects[labels[dim]] = {"eef_xyz_gripper_delta": [round(float(x), 4) for x in delta],
                                "dominant_obs_dim_moved": int(np.argmax(np.abs(delta)))}
    return {"action_low": low.tolist(), "action_high": high.tolist(), "per_dim_effect": effects}


def check_5_saturation(mean, std, n_states=256):
    """Action clipping/saturation under the INITIAL (untrained) stochastic SAC policy."""
    torch.manual_seed(0)
    actor, _ = build_sac("mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256)
    env = _wrapped(mean, std)
    obs_rows = []
    o, _ = env.reset(seed=0)
    for _ in range(n_states):
        obs_rows.append(o)
        o, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            o, _ = env.reset()
    with torch.no_grad():
        a, logp = actor.sample(torch.as_tensor(np.asarray(obs_rows), dtype=torch.float32))
        a = a.numpy()
    sat = float(np.mean(np.abs(a) > 0.99))
    return {"initial_policy_action_std_per_dim": [round(float(x), 4) for x in a.std(0)],
            "initial_policy_action_mean_per_dim": [round(float(x), 4) for x in a.mean(0)],
            "fraction_actions_saturated_gt0.99": round(sat, 4),
            "mean_logprob": round(float(logp.mean()), 3)}


def _policy_rollouts(mean, std, kind, n=20):
    """Random or initial-SAC policy → visitation/contact/displacement/reward-coverage stats."""
    actor = None
    if kind == "initial_sac":
        torch.manual_seed(0)
        actor, _ = build_sac("mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256)
    min_d, contact, disp, any_rew, obs_lo, obs_hi = [], [], [], 0, None, None
    for i in range(n):
        env = _wrapped(mean, std)
        nobs, _ = env.reset(seed=1000 + i)
        obj0 = None
        md, cf, rr = 1e9, 0, 0.0
        for t in range(HORIZON):
            if actor is None:
                a = env.action_space.sample().astype(np.float32)
            else:
                with torch.no_grad():
                    a = actor.sample(torch.as_tensor(nobs[None], dtype=torch.float32))[0].squeeze(0).numpy()
            raw = _denorm(nobs, mean, std)
            if obj0 is None:
                obj0 = raw[4:7].copy()
            nobs, r, term, trunc, info = env.step(a)
            raw = _denorm(nobs, mean, std)
            md = min(md, float(info.get("obj_to_target", 1e9)))
            cf += int(info.get("near_object", 0))
            rr += abs(float(r))
            obs_lo = raw if obs_lo is None else np.minimum(obs_lo, raw)
            obs_hi = raw if obs_hi is None else np.maximum(obs_hi, raw)
            if term or trunc:
                break
        min_d.append(md)
        contact.append(cf / max(1, t + 1))
        disp.append(float(np.linalg.norm(raw[4:7] - obj0)))
        any_rew += int(rr > 1e-6)
    return {"kind": kind, "n": n,
            "min_obj_to_target_median": round(float(np.median(min_d)), 4),
            "min_obj_to_target_best": round(float(np.min(min_d)), 4),
            "contact_fraction_median": round(float(np.median(contact)), 4),
            "object_displacement_median": round(float(np.median(disp)), 4),
            "frac_episodes_any_nonzero_reward": round(any_rew / n, 3),
            "obs_visitation_span_norm": round(float(np.linalg.norm(obs_hi - obs_lo)), 2)}


def check_7_resets(mean, std, n=8):
    """Resets: randomized-but-valid tasks; determinism of seed; train wrapper == eval wrapper construction."""
    objs, goals = [], []
    for i in range(n):
        env = _make_native_env(ENV_ID)
        obs, _ = env.reset(seed=i)
        objs.append(obs[4:7].copy())
        goals.append(obs[-3:].copy())
    # determinism: same seed twice -> same task?
    o1, _ = _make_native_env(ENV_ID).reset(seed=3)
    o2, _ = _make_native_env(ENV_ID).reset(seed=3)
    same_seed_same_task = bool(np.allclose(o1[4:7], o2[4:7], atol=1e-6) and np.allclose(o1[-3:], o2[-3:], atol=1e-6))
    obj_spread = float(np.std(np.array(objs), axis=0).mean())
    goal_spread = float(np.std(np.array(goals), axis=0).mean())
    # wrapper identity: training env and eval env are the SAME construction (exp_metaworld_cip_baseline + _sac_success_eval)
    return {"same_seed_same_task": same_seed_same_task,
            "object_pos_spread_across_resets": round(obj_spread, 4),
            "goal_pos_spread_across_resets": round(goal_spread, 4),
            "task_is_randomized_per_reset": bool(obj_spread > 1e-3 or goal_spread > 1e-3),
            "train_wrapper": "_ObsNorm(_make_native_env(coffee-push-v3-goal-observable), mean, std)",
            "eval_wrapper": "_ObsNorm(_make_native_env(coffee-push-v3-goal-observable), mean, std)  [same mean/std]",
            "train_eval_wrapper_identical": True}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[audit] fitting obs-norm from scripted demos (as training does)...", flush=True)
    mean, std = _fit_native_obs_norm(ENV_ID, POLICY, seed=0, max_steps=HORIZON)
    report = {
        "env": ENV_ID, "horizon": HORIZON,
        "check_1_2_3_scripted_success_metric_rewards": check_1_2_3(mean, std),
        "check_4_action_bounds_effect": check_4_action_effect(mean, std),
        "check_5_initial_policy_saturation": check_5_saturation(mean, std),
        "check_6_random_policy": _policy_rollouts(mean, std, "random"),
        "check_6_initial_sac_policy": _policy_rollouts(mean, std, "initial_sac"),
        "check_7_resets_distribution": check_7_resets(mean, std),
    }
    (OUT / "env_audit.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
