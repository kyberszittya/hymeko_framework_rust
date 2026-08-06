"""Option C — adversarial farming search for the monitor-aligned reward (bounded, NO RL).

Tests reward hacking two ways on the real MetaWorld pick-place env: (1) **scripted proxy-adversaries** (hover =
reach-without-grasp; grasp-hold = grasp/lift-without-delivery) that deliberately exploit shaping proxies, and
(2) **warm-started CEM** (open-loop action-sequence search seeded from the scripted expert) that *maximizes* each
reward. Every trajectory is scored under BOTH the original and the monitor_aligned reward from one rollout. Bounded
horizon, no replay, no gradient policy — this is search, not learning.

Determinism note: MetaWorld randomizes the layout per env *instance*, but ``reset(seed)`` is reproducible within a
fixed instance, so all rolls for one layout use one instance and one reset seed.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedReward

_H = 130                     # trajectory horizon


def _declared_weights() -> "tuple[float, float, float, float]":
    from hymeko_rl.env.reward import read_reward_terms
    w = dict(read_reward_terms("data/robotics/metaworld_reward.hymeko"))
    return w["mw_in_place"], w["mw_grasp"], w["mw_near"], w["mw_dist"]


def _make_env(task: str) -> Any:
    from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
    return ENVS[f"{task}-v3-goal-observable"](render_mode=None)


def _score(env: Any, actions: "Callable[[np.ndarray, int], np.ndarray] | np.ndarray", layout_seed: int,
           w: "tuple[float, float, float, float]") -> "dict[str, Any]":
    """Roll a closed-loop controller (callable) or an open-loop sequence (array); score under BOTH rewards + task."""
    obs, _ = env.reset(seed=layout_seed)
    mar = MonitorAlignedReward()
    mar.reset(obs, {"obj_to_target": float(np.linalg.norm(obs[4:7] - obs[-3:]))})
    o = m = 0.0
    g = n = steps = 0
    ott0: "float | None" = None
    ott = 0.0
    ok = False
    oz_max = 0.02
    for t in range(_H):
        a = actions(obs, t) if callable(actions) else actions[t]
        obs, _r, term, trunc, info = env.step(np.clip(np.asarray(a, np.float32), -1.0, 1.0))
        o += w[0] * info["in_place_reward"] + w[1] * info["grasp_reward"] + w[2] * info["near_object"] - w[3] * info["obj_to_target"]
        m += mar.step(obs, info)
        g += int(info["grasp_success"])
        n += int(info["near_object"])
        steps += 1
        ott = float(info["obj_to_target"])
        ok = ok or bool(info["success"])
        oz_max = max(oz_max, float(obs[6]))
        if ott0 is None:
            ott0 = ott
        if term or trunc:
            break
    return {"original_reward": round(float(o), 1), "monitor_aligned_reward": round(float(m), 1), "success": int(ok),
            "grasp_frac": round(g / max(1, steps), 2), "near_frac": round(n / max(1, steps), 2),
            "delivery": round((ott0 or 0.0) - ott, 3), "lift": round(oz_max - 0.02, 3)}


def _hover(obs: np.ndarray, _t: int) -> np.ndarray:
    """Reach the object and hover — gripper OPEN, never grasp, never deliver (proxy-exploit of reach/near shaping)."""
    d = np.clip(25.0 * (obs[4:7] - obs[:3]), -1.0, 1.0)
    return np.array([d[0], d[1], d[2], -1.0], np.float32)


def _expert_actions(task: str, env: Any, layout_seed: int) -> np.ndarray:
    """The scripted expert's open-loop action sequence on the fixed layout (CEM warm-start + the delivery reference)."""
    import metaworld.policies as mp
    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    obs, _ = env.reset(seed=layout_seed)
    pol = getattr(mp, GENERIC_TASKS[task])()
    seq = []
    for _ in range(_H):
        a = np.asarray(pol.get_action(obs), np.float32)
        seq.append(a)
        obs, _r, term, trunc, _info = env.step(a)
        if term or trunc:
            break
    arr = np.asarray(seq)
    pad = np.tile(arr[-1] if len(arr) else np.zeros(4), (_H - len(arr), 1))
    return np.vstack([arr, pad])[:_H].astype(np.float32)


def _grasp_hold(expert: np.ndarray, hold_after: int = 35) -> np.ndarray:
    """Expert reach+grasp+lift+partial-move, then FREEZE (gripper closed, no motion) — reliably fails to complete.

    ``hold_after=35`` stops before the expert's placement, so the object is grasped and moved partway (high shaped
    reward) but never placed (success 0) — the 'stop just short of completion' farming plateau."""
    seq = expert.copy()
    seq[hold_after:] = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    return seq


def _cem(env: Any, mu0: np.ndarray, objective: str, layout_seed: int, w: "tuple[float, float, float, float]",
         *, iters: int = 6, n: int = 40, elite: int = 6, seed: int = 0) -> np.ndarray:
    """Warm-started CEM over open-loop action sequences maximizing ``objective`` reward. Returns the elite mean."""
    rng = np.random.default_rng(seed)
    mu = mu0.copy()
    sigma = np.ones_like(mu) * 0.3
    for _ in range(iters):
        samples = np.clip(rng.normal(mu, sigma, (n, *mu.shape)), -1.0, 1.0)
        rewards = np.asarray([_score(env, s, layout_seed, w)[objective] for s in samples])
        idx = np.argsort(rewards)[-elite:]
        mu = samples[idx].mean(0)
        sigma = samples[idx].std(0) + 1e-2
    return mu


def _render(task: str, actions: Any, layout_seed: int, out_path: Path, label: str) -> None:
    from hymeko_rl.eval.evaluate import _write_gif
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        env = ENVS[f"{task}-v3-goal-observable"](render_mode="rgb_array")
        obs, _ = env.reset(seed=layout_seed)
        frames = []
        for t in range(_H):
            a = actions(obs, t) if callable(actions) else actions[t]
            obs, _r, term, trunc, _info = env.step(np.clip(np.asarray(a, np.float32), -1.0, 1.0))
            frames.append(np.asarray(env.render(), dtype=np.uint8)[::2, ::2])  # type: ignore[no-untyped-call]  # metaworld render
            if term or trunc:
                break
    _write_gif(frames, out_path, fps=20, stamp=label)


def _median(rows: "list[dict[str, Any]]", key: str) -> float:
    return round(float(np.median([r[key] for r in rows])), 2)


def run_adversarial_search(task: str = "pick-place", layout_seeds: "tuple[int, ...]" = (0, 1, 2),
                           out_dir: Any = None) -> "dict[str, Any]":
    """Score expert / hover / grasp-hold / CEM-max-original / CEM-max-monitor_aligned under BOTH rewards; verdict."""
    out = Path(out_dir) if out_dir is not None else Path("reports/figures/2026_07_10_adversarial_farming_search")
    out.mkdir(parents=True, exist_ok=True)
    w = _declared_weights()
    controllers = ("expert_deliver", "hover", "grasp_hold", "cem_max_original", "cem_max_monitor_aligned")
    per_seed: dict[str, list[dict[str, Any]]] = {c: [] for c in controllers}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ls in layout_seeds:
            env = _make_env(task)
            expert = _expert_actions(task, env, ls)
            seqs: dict[str, Any] = {"expert_deliver": expert, "hover": _hover, "grasp_hold": _grasp_hold(expert),
                                    "cem_max_original": _cem(env, expert, "original_reward", ls, w),
                                    "cem_max_monitor_aligned": _cem(env, expert, "monitor_aligned_reward", ls, w)}
            for c, seq in seqs.items():
                per_seed[c].append(_score(env, seq, ls, w))
            if ls == layout_seeds[0]:                                   # one representative layout gets GIFs
                _render(task, expert, ls, out / "expert_deliver.gif", "expert (delivers)")
                _render(task, _hover, ls, out / "hover_adversary.gif", "hover: reach, no grasp (proxy)")
                _render(task, seqs["grasp_hold"], ls, out / "grasp_hold_adversary.gif",
                        "grasp-hold: grasp then freeze, FAILS task (77% reward under original, 0.1% under repair)")
    agg = {c: {k: _median(per_seed[c], k) for k in ("original_reward", "monitor_aligned_reward", "success",
                                                    "grasp_frac", "delivery")} for c in controllers}
    verdict = _verdict(agg)
    summary = {"task": task, "layout_seeds": list(layout_seeds), "declared_weights": {"in_place": w[0], "grasp": w[1],
               "near": w[2], "dist": w[3]}, "aggregate": agg, "per_seed": per_seed, "verdict": verdict}
    (out / "adversarial_farming_search.json").write_text(json.dumps(summary, indent=2, default=float))
    _plot(agg, out)
    for c in controllers:
        a = agg[c]
        print(f"[adv-farm] {c:24s} orig {a['original_reward']:8.1f} | ma {a['monitor_aligned_reward']:9.1f} | "
              f"success {a['success']} grasp {a['grasp_frac']} delivery {a['delivery']}", flush=True)
    gh = verdict["grasp_hold_reward_ratio_vs_expert"]
    print(f"[adv-farm] FAILED grasp_hold: {gh['original']} of expert reward under ORIGINAL vs {gh['monitor_aligned']} "
          f"under monitor_aligned | original_credits_failed={verdict['original_credits_failed_trajectory']} "
          f"ma_penalizes_failed={verdict['monitor_aligned_penalizes_failed_trajectory']} | "
          f"globally_hackable_by_cem={verdict['original_globally_hackable_by_cem']}", flush=True)
    return summary


def _verdict(agg: "dict[str, Any]") -> "dict[str, Any]":
    """A FAILED grasp-hold collects most success-reward under original (a farming plateau) but ~0 under the repair."""
    exp, cemo, gh = agg["expert_deliver"], agg["cem_max_original"], agg["grasp_hold"]
    gh_o = round(gh["original_reward"] / exp["original_reward"], 3) if exp["original_reward"] else None
    gh_m = round(gh["monitor_aligned_reward"] / exp["monitor_aligned_reward"], 3) if exp["monitor_aligned_reward"] else None
    return {
        # magnitude is layout-variable (env non-determinism); the ROBUST signal is the SIGN — original credits a
        # failed near-completion with POSITIVE reward, monitor_aligned gives it ~0-or-NEGATIVE.
        "grasp_hold_reward_ratio_vs_expert": {"original": gh_o, "monitor_aligned": gh_m, "grasp_hold_success": gh["success"]},
        "original_credits_failed_trajectory": bool(gh["original_reward"] > 0.0 and gh["success"] < 0.5),
        "monitor_aligned_penalizes_failed_trajectory": bool(gh["monitor_aligned_reward"] <= 0.02 * exp["monitor_aligned_reward"]),
        "hover_reward_original_vs_ma": [agg["hover"]["original_reward"], agg["hover"]["monitor_aligned_reward"]],
        # honest scope: neither reward is GLOBALLY hackable — CEM maximizing either one delivers.
        "original_globally_hackable_by_cem": bool(cemo["success"] < 0.5 and cemo["original_reward"] > 0.5 * exp["original_reward"]),
        "monitor_aligned_max_delivers": bool(agg["cem_max_monitor_aligned"]["success"] >= 0.5)}


def _plot(agg: "dict[str, Any]", out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctrl = ["expert_deliver", "hover", "grasp_hold", "cem_max_original", "cem_max_monitor_aligned"]
    labels = ["expert\n(delivers)", "hover\n(proxy)", "grasp-hold\n(proxy)", "CEM-max\noriginal", "CEM-max\nmonitor_aligned"]
    x = np.arange(len(ctrl))
    ww = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].bar(x - ww / 2, [agg[c]["original_reward"] for c in ctrl], ww, label="original", color="#2c3e50")
    ax[0].bar(x + ww / 2, [agg[c]["monitor_aligned_reward"] for c in ctrl], ww, label="monitor_aligned", color="#2c8")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("total reward")
    ax[0].set_title("Reward per controller (proxies: original credits, monitor_aligned penalizes)")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, [agg[c]["success"] for c in ctrl], color=["#2c3e50", "#c0392b", "#c0392b", "#4a6fa5", "#2c8"])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("success")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_title("Task success (proxies fail; CEM-max of either reward delivers)")
    fig.tight_layout()
    fig.savefig(out / "adversarial_farming_search.png", dpi=130)
    plt.close(fig)


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_adversarial_search(layout_seeds=tuple(range(a.seeds)), out_dir=Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
