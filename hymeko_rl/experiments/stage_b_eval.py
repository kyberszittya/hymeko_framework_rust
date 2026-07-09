"""Stage-B post-eval: measure a TRAINED policy's behaviour + reward mechanism, and render it (§9).

Rolls the trained (greedy) policy on the reward-override MetaWorld env, collects the CIP frame + reward components,
and reuses the Stage-A ``_condition`` (per-tail loadings + weighted LiNGAM-SH factorization + cross-view +
reward-monitor disagreement + emitted CIP DAG) — no re-implemented eval (CLAUDE.md §6.1). The task monitor for
pick-place is MetaWorld's own ``success`` flag (there is no bespoke pick-place submonitor).

Metrics per profile: monitor (success) pass rate · progress_score · near_fraction · obj_to_target_delta ·
total_reward under its OWN reward · total_reward recomputed under the ORIGINAL reward · reward-monitor
disagreement · loadings + collapse flag · emitted CIP DAG · cross-view verification · a rollout GIF.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.eval.cip.metaworld_reward import _TERM_TO_COMPONENT
from hymeko_rl.eval.cip.reward_ablation_metaworld import AblatedRewardSpec


def _record_policy_rollouts(cfg: Any, spec: AblatedRewardSpec, policy: Any, n: int, seed0: int = 30_000,
                            ) -> "dict[str, Any]":
    """Roll ``n`` greedy episodes of the trained policy → per-rollout component totals + CIP frame + success rate."""
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import make_training_env
    kinds = list(spec.term_kinds())
    comp_keys = [_TERM_TO_COMPONENT[k] for k in kinds]
    totals: list[np.ndarray] = []
    near: list[float] = []
    grasp: list[float] = []
    dist: list[float] = []
    progress: list[float] = []
    n_success = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            env = make_training_env(cfg, spec)
            obs, _ = env.reset(seed=seed0 + i)
            csum = np.zeros(len(kinds))
            n_near = n_grasp = steps = 0
            d0: "float | None" = None
            d_final = ip_sum = 0.0
            ok = False
            for _ in range(cfg.max_steps):
                obs, _r, term, trunc, info = env.step(policy.greedy(obs))
                csum += np.asarray([sign * float(info.get(key, 0.0)) for key, sign in comp_keys])
                steps += 1
                n_near += int(info.get("near_object", 0))
                n_grasp += int(info.get("grasp_success", 0))
                d_final = float(info.get("obj_to_target", 0.0))
                ip_sum += float(info.get("in_place_reward", 0.0))
                ok = ok or bool(info.get("success", 0.0))
                if d0 is None:
                    d0 = d_final
                if term or trunc:
                    break
            totals.append(csum)
            near.append(n_near / max(1, steps))
            grasp.append(n_grasp / max(1, steps))
            dist.append(float((d0 or 0.0) - d_final))
            progress.append(ip_sum / max(1, steps))
            n_success += int(ok)
    prog = np.asarray(progress)
    return {"totals": np.asarray(totals), "success_rate": n_success / max(1, n),
            "cip": {"near_fraction": np.asarray(near), "grasp_fraction": np.asarray(grasp),
                    "obj_to_target_delta": np.asarray(dist), "progress_score": prog}, "task_score": prog}


def render_policy_gif(cfg: Any, spec: AblatedRewardSpec, policy: Any, out_path: "str | Path", *,
                      seed: int = 40_000, fps: int = 20) -> "tuple[Path, bool]":
    """Render one greedy episode of the trained policy to a GIF (reuses the shared ``_write_gif``). Returns (path, success)."""
    from hymeko_rl.eval.evaluate import _write_gif
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import make_training_env
    frames: list[np.ndarray] = []
    ok = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make_training_env(cfg, spec, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        for _ in range(cfg.max_steps):
            obs, _r, term, trunc, info = env.step(policy.greedy(obs))
            frames.append(np.asarray(env.env.render(), dtype=np.uint8)[::2, ::2])
            ok = ok or bool(info.get("success", 0.0))
            if term or trunc:
                break
    path = _write_gif(frames, out_path, fps=fps, stamp=f"trained  success={ok}")
    return path, ok


def evaluate_and_render(cfg: Any, name: str, spec: AblatedRewardSpec, policy: Any, orig_weights: np.ndarray,
                        out_dir: Path) -> "dict[str, Any]":
    """Full post-eval of a trained policy: behaviour metrics, reward mechanism (reuses ``_condition``), and a GIF."""
    from hymeko_rl.eval.cip.metaworld_reward import reward_mechanism_proposal
    from hymeko_rl.eval.cip.reward_ablation_metaworld import _condition

    rec = _record_policy_rollouts(cfg, spec, policy, cfg.eval_episodes_post)
    own_w = np.asarray(spec.ablated_weights(), dtype=np.float64)
    reward_own = rec["totals"] @ own_w
    reward_orig = rec["totals"] @ orig_weights
    prof_dir = out_dir / name
    prof_dir.mkdir(parents=True, exist_ok=True)
    prop = reward_mechanism_proposal(spec.source, available=[*rec["cip"], "total_reward"])
    cond = _condition(rec["cip"], reward_own, rec["task_score"], prop, f"{name}_trained", prof_dir)
    gif_path, gif_ok = render_policy_gif(cfg, spec, policy, prof_dir / "rollout.gif")
    cipvars = rec["cip"]
    return {"success_rate": round(rec["success_rate"], 4),
            "near_fraction": round(float(np.mean(cipvars["near_fraction"])), 4),
            "grasp_fraction": round(float(np.mean(cipvars["grasp_fraction"])), 4),
            "obj_to_target_delta": round(float(np.mean(cipvars["obj_to_target_delta"])), 4),
            "progress_score": round(float(np.mean(cipvars["progress_score"])), 4),
            "reward_own_mean": round(float(np.mean(reward_own)), 4),
            "reward_under_original_mean": round(float(np.mean(reward_orig)), 4),
            "reward_monitor_disagreement": cond["reward_monitor_disagreement"],
            "loadings": cond["loadings"], "cross_view_agree": cond["cross_view_agree"],
            "reward_reconstruction_r2": cond["reward_reconstruction_r2"], "gif": str(gif_path), "gif_success": gif_ok}


def compare_profiles(results: "dict[str, dict[str, Any]]") -> "dict[str, Any]":
    """Contrast the original-trained vs mw_in_place_off-trained behaviour — the Stage-B answer."""
    o = results.get("original", {}).get("eval")
    a = results.get("mw_in_place_off", {}).get("eval")
    if not o or not a:
        return {"comparable": False}
    return {"comparable": True,
            "success_rate": {"original": o["success_rate"], "mw_in_place_off": a["success_rate"],
                             "delta": round(a["success_rate"] - o["success_rate"], 4)},
            "grasp_fraction": {"original": o["grasp_fraction"], "mw_in_place_off": a["grasp_fraction"],
                               "delta": round(a["grasp_fraction"] - o["grasp_fraction"], 4)},
            "progress_score": {"original": o["progress_score"], "mw_in_place_off": a["progress_score"],
                               "delta": round(a["progress_score"] - o["progress_score"], 4)},
            "obj_to_target_delta": {"original": o["obj_to_target_delta"], "mw_in_place_off": a["obj_to_target_delta"],
                                    "delta": round(a["obj_to_target_delta"] - o["obj_to_target_delta"], 4)},
            "reward_monitor_disagreement": {"original": o["reward_monitor_disagreement"],
                                            "mw_in_place_off": a["reward_monitor_disagreement"]}}
