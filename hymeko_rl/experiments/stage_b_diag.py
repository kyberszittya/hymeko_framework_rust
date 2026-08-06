"""From-scratch pick-place PPO sanity/diagnostics — is 0%-vs-0% a harness/metric/PPO bug or a true exploration wall?

Six probes (only the reach-only micro-PPO trains, ≤50k steps): zero-action, random-action, a scripted reach oracle,
the BC policy under the SAME metric collection, a reach-only PPO micro-probe, and a reward-visibility check.
Diagnosis buckets: A harness/control/metric bug · B PPO setup issue · C true exploration wall · D reward-ablation
inconclusive. Reuses ``make_training_env`` / ``_bc_base_policy`` / ``train_ppo_flat`` — no new env.

MetaWorld pick-place obs layout (verified): hand=obs[:3], gripper=obs[3], object=obs[4:7], goal=obs[-3:];
info exposes obj_to_target, near_object, grasp_success, in_place_reward, grasp_reward, unscaled_reward, success.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.eval.cip.reward_ablation_metaworld import ablate_reward_spec
from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig, make_training_env

ActionFn = Callable[[np.ndarray], np.ndarray]
_COMPONENTS = ("in_place_reward", "grasp_reward", "near_object", "obj_to_target")


def _parse(obs: np.ndarray) -> "tuple[np.ndarray, float, np.ndarray, np.ndarray]":
    """(hand_xyz, gripper, object_xyz, goal_xyz) from a MetaWorld pick-place observation."""
    return obs[:3], float(obs[3]), obs[4:7], obs[-3:]


def reach_action(obs: np.ndarray, gain: float = 25.0) -> np.ndarray:
    """Scripted reach oracle: drive the end-effector toward the object (delta control), gripper open. No grasp/place."""
    hand, _g, obj, _goal = _parse(obs)
    delta = np.clip(gain * (obj - hand), -1.0, 1.0)
    return np.array([delta[0], delta[1], delta[2], -1.0], dtype=np.float32)


def _run_episode(env: Any, action_fn: ActionFn, max_steps: int, seed: int) -> "dict[str, Any]":
    """One episode → geometry, metrics, reward components, action buffer, and per-step traces."""
    obs, _ = env.reset(seed=seed)
    hand_obj: list[float] = []
    obj_target: list[float] = []
    near: list[int] = []
    grasp: list[int] = []
    rews: list[float] = []
    acts: list[np.ndarray] = []
    comp_sum = dict.fromkeys(_COMPONENTS, 0.0)
    ok = done = trunc = False
    steps = 0
    for _ in range(max_steps):
        act = np.asarray(action_fn(obs), np.float32)
        acts.append(act)
        obs, r, term, tr, info = env.step(act)
        hand, _g, obj, goal = _parse(obs)
        hand_obj.append(float(np.linalg.norm(hand - obj)))
        obj_target.append(float(info.get("obj_to_target", np.linalg.norm(obj - goal))))
        near.append(int(info.get("near_object", 0)))
        grasp.append(int(info.get("grasp_success", 0)))
        rews.append(float(r))
        for k in _COMPONENTS:
            comp_sum[k] += float(info.get(k, 0.0))
        ok = ok or bool(info.get("success", 0.0))
        steps += 1
        done, trunc = bool(term), bool(tr)
        if term or tr:
            break
    a = np.asarray(acts)
    return {"steps": steps, "min_hand_obj": float(min(hand_obj)), "final_hand_obj": float(hand_obj[-1]),
            "min_obj_target": float(min(obj_target)), "ever_near": int(max(near)), "near_fraction": float(np.mean(near)),
            "ever_grasp": int(max(grasp)), "success": int(ok), "reward_mean": float(np.mean(rews)),
            "components_mean": {k: comp_sum[k] / max(1, steps) for k in _COMPONENTS},
            "action_stats": {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())},
            "done": int(done), "trunc": int(trunc), "trace_hand_obj": hand_obj, "trace_near": near}


def _mean(eps: "list[dict[str, Any]]", key: str) -> float:
    return float(np.mean([e[key] for e in eps]))


def _summarize(cfg: StageBConfig, spec: Any, action_fn: ActionFn, n: int, seed0: int = 0) -> "dict[str, Any]":
    """Aggregate ``n`` episodes: min-distance stats, threshold crossings, near/grasp/success rates, components, actions."""
    eps: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            eps.append(_run_episode(make_training_env(cfg, spec), action_fn, cfg.max_steps, seed0 + i))
    mind = np.asarray([e["min_hand_obj"] for e in eps])
    thr = {f"frac_within_{t}": float(np.mean(mind < t)) for t in (0.20, 0.10, 0.05)}
    comps = {k: float(np.mean([e["components_mean"][k] for e in eps])) for k in _COMPONENTS}
    a_stats = {s: float(np.mean([e["action_stats"][s] for e in eps])) for s in ("mean", "std", "min", "max")}
    return {"n": n, "min_hand_obj_median": float(np.median(mind)), "min_hand_obj_best": float(mind.min()), **thr,
            "frac_ever_near": _mean(eps, "ever_near"), "near_fraction_mean": _mean(eps, "near_fraction"),
            "ever_grasp_frac": _mean(eps, "ever_grasp"), "success_rate": _mean(eps, "success"),
            "components_mean": comps, "action_stats": a_stats}


def _random_fn(rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> ActionFn:
    return lambda _obs: rng.uniform(low, high).astype(np.float32)


def _greedy_fn(policy: Any) -> ActionFn:
    return lambda obs: np.asarray(policy.greedy(obs), np.float32)


class _ReachReward:
    """Reward-override giving ``r = -‖hand − object‖`` — the easy dense reach reward for the PPO micro-probe."""

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kw: Any) -> Any:
        return self.env.reset(**kw)

    def step(self, action: Any) -> Any:
        obs, _r, term, trunc, info = self.env.step(action)
        hand, _g, obj, _goal = _parse(obs)
        return obs, -float(np.linalg.norm(hand - obj)), term, trunc, info


def probe_reach_ppo(cfg: StageBConfig, spec: Any, seed: int, steps: int) -> "dict[str, Any]":
    """Micro-probe: train PPO from scratch on the easy reach reward (-‖hand-obj‖), then measure reach quality."""
    from dataclasses import replace

    from .stage_b_ppo import train_ppo_flat
    rcfg = replace(cfg, total_env_steps=steps, warm_start=False, optimizer="ppo", ppo_entropy_coef=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _ReachReward(make_training_env(rcfg, spec))
        out = train_ppo_flat(rcfg, env, None, seed, log=lambda m: print(f"[diag reach-ppo] {m}", flush=True))
    ev = _summarize(rcfg, spec, _greedy_fn(out["policy"]), n=16, seed0=50_000)
    return {"steps": steps, "returns_first": out["returns"][0] if out["returns"] else None,
            "returns_last": out["returns"][-1] if out["returns"] else None,
            "min_hand_obj_median": ev["min_hand_obj_median"], "frac_ever_near": ev["frac_ever_near"],
            "near_fraction_mean": ev["near_fraction_mean"], "frac_within_0.05": ev["frac_within_0.05"]}


def _diagnose(z: "dict[str, Any]", rnd: "dict[str, Any]", scr: "dict[str, Any]", bc: "dict[str, Any]",
              reach: "dict[str, Any]", full_from_scratch_success: float) -> "tuple[str, str]":
    """Assign A/B/C/D from the probe outcomes (see module docstring for the buckets)."""
    scripted_reaches = scr["frac_within_0.05"] >= 0.5 or scr["frac_ever_near"] >= 0.5
    bc_metric_ok = bc["success_rate"] > 0.0 and bc["frac_ever_near"] > 0.0
    if not scripted_reaches or not bc_metric_ok:
        return "A", (f"scripted-reach within-0.05={scr['frac_within_0.05']:.2f} ever_near={scr['frac_ever_near']:.2f}; "
                     f"BC success={bc['success_rate']:.2f} ever_near={bc['frac_ever_near']:.2f} — the harness/metric "
                     f"does not register reach even when the hand demonstrably closes on the object")
    reach_ppo_improves = (reach["min_hand_obj_median"] < rnd["min_hand_obj_median"] - 0.02
                          and reach["frac_ever_near"] > rnd["frac_ever_near"])
    if not reach_ppo_improves:
        return "B", (f"scripted+BC activate near, but reach-only PPO did not improve reach "
                     f"(min_dist {reach['min_hand_obj_median']:.3f} vs random {rnd['min_hand_obj_median']:.3f}; "
                     f"ever_near {reach['frac_ever_near']:.2f} vs {rnd['frac_ever_near']:.2f}) — PPO/action/obs setup")
    if full_from_scratch_success <= 0.0:
        return "C", (f"scripted+BC+metrics OK and reach-only PPO improves reach "
                     f"(min_dist {reach['min_hand_obj_median']:.3f}, ever_near {reach['frac_ever_near']:.2f}), but full "
                     f"pick-place PPO still discovers no grasp — a genuine grasp/place exploration wall")
    return "D", "both reward variants remain 0% and neither reaches clean pre-grasp behaviour — inconclusive"


_KINDS = ("random", "scripted_reach", "bc", "reach_ppo")
_LABELS = ("random", "scripted\nreach", "BC", "reach-only\nPPO")
_COLS = ("#888", "#e8a33d", "#2c7", "#4a6fa5")


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_distance(diag: "dict[str, Any]", md: "list[float]", out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(list(_LABELS), md, color=list(_COLS))
    for t, c in ((0.20, "gray"), (0.10, "orange"), (0.05, "red")):
        ax.axhline(t, ls="--", lw=.8, color=c, label=f"{t}")
    ax.set_ylabel("median min hand-object distance")
    ax.set_title("Can the controller close on the object?")
    ax.legend(fontsize=7, title="threshold")
    _save(fig, out / "from_scratch_hand_object_distance.png")


def _plot_near(nf: "list[float]", out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(list(_LABELS), nf, color=list(_COLS))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of episodes that ever fire near_object")
    ax.set_title("Does the near_object metric activate?")
    _save(fig, out / "from_scratch_near_fraction.png")


_VIS_KINDS = ("zero", "random", "scripted_reach", "bc")


def _plot_components(vis: "dict[str, Any]", out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(_VIS_KINDS))
    w = .38
    for key, dx, col in (("original", -w / 2, "#2c3e50"), ("mw_in_place_off", w / 2, "#c0392b")):
        ax.bar(x + dx, [vis[key][k]["in_place_reward"] for k in _VIS_KINDS], w, color=col, label=f"{key}: in_place")
    ax.set_xticks(x)
    ax.set_xticklabels(list(_VIS_KINDS))
    ax.set_ylabel("mean in_place_reward component")
    ax.set_title("Reward-component visibility (in_place: the ablated term)")
    ax.legend(fontsize=8)
    _save(fig, out / "from_scratch_reward_components.png")


def _plot_sanity(md: "list[float]", nf: "list[float]", out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar(list(_LABELS[:3]), md[:3], color=list(_COLS[:3]))
    ax[0].axhline(0.05, ls="--", color="red", lw=.8, label="0.05")
    ax[0].set_ylabel("median min hand-object dist")
    ax[0].set_title("min distance")
    ax[0].legend(fontsize=7)
    ax[1].bar(list(_LABELS[:3]), nf[:3], color=list(_COLS[:3]))
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("ever near_object")
    fig.suptitle("Sanity: random vs scripted-reach vs BC", fontweight="bold")
    _save(fig, out / "sanity_random_vs_scripted_vs_bc.png")


def _plot(diag: "dict[str, Any]", out: Path) -> None:
    """The four diagnostic figures (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    md = [diag[k]["min_hand_obj_median"] for k in _KINDS]
    nf = [diag[k]["frac_ever_near"] for k in _KINDS]
    _plot_distance(diag, md, out)
    _plot_near(nf, out)
    _plot_components(diag["reward_visibility"], out)
    _plot_sanity(md, nf, out)


def run_diagnostics(cfg: StageBConfig, out_dir: "Path | None" = None, *, reach_ppo_steps: int = 40_000,
                    full_from_scratch_success: float = 0.0) -> "dict[str, Any]":
    """Run all six probes for original + mw_in_place_off, diagnose A/B/C/D, write JSON + plots."""
    from .exp_metaworld_reward_stageb import _bc_base_policy
    out = out_dir or Path("reports/figures/2026_07_09_pick_place_from_scratch_sanity")
    out.mkdir(parents=True, exist_ok=True)
    orig = ablate_reward_spec(cfg.spec_path)
    off = ablate_reward_spec(cfg.spec_path, drop=["mw_in_place"])
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low = make_training_env(cfg, orig).action_space.low
        high = make_training_env(cfg, orig).action_space.high
    # probes on the ORIGINAL reward env (geometry/metrics are reward-independent)
    zero = _run_episode(make_training_env(cfg, orig), lambda _o: np.zeros(4, np.float32), cfg.max_steps, 0)
    random_s = _summarize(cfg, orig, _random_fn(rng, low, high), n=100)
    scripted = _summarize(cfg, orig, reach_action, n=32)
    bc_policy, bc_info = _bc_base_policy(cfg)
    bc = _summarize(cfg, orig, _greedy_fn(bc_policy), n=32)
    reach = probe_reach_ppo(cfg, orig, seed=0, steps=reach_ppo_steps)
    # reward visibility: mean components under each rollout type for BOTH reward specs
    vis = _reward_visibility(cfg, orig, off, bc_policy, rng, low, high)
    category, reason = _diagnose(zero, random_s, scripted, bc, reach, full_from_scratch_success)
    diag = {"zero_action": zero, "random": random_s, "scripted_reach": scripted, "bc": {**bc, "bc_info": bc_info},
            "reach_ppo": reach, "reward_visibility": vis, "diagnosis": category, "reason": reason}
    (out / "diagnostics.json").write_text(json.dumps(_strip_traces(diag), indent=2, default=float))
    _plot(diag, out)
    print(f"[diag] scripted-reach within-0.05={scripted['frac_within_0.05']:.2f} ever_near={scripted['frac_ever_near']:.2f} "
          f"| BC success={bc['success_rate']:.2f} ever_near={bc['frac_ever_near']:.2f} "
          f"| reach-PPO min_dist={reach['min_hand_obj_median']:.3f} ever_near={reach['frac_ever_near']:.2f}", flush=True)
    print(f"[diag] DIAGNOSIS = {category}: {reason}", flush=True)
    return diag


def _reward_visibility(cfg: StageBConfig, orig: Any, off: Any, bc_policy: Any, rng: np.random.Generator,
                       low: np.ndarray, high: np.ndarray) -> "dict[str, Any]":
    """Mean reward components per rollout type, for BOTH reward specs — is the ablation actually different / non-flat?"""
    fns = {"zero": lambda _o: np.zeros(4, np.float32), "random": _random_fn(rng, low, high),
           "scripted_reach": reach_action, "bc": _greedy_fn(bc_policy)}
    vis: dict[str, Any] = {}
    for name, spec in (("original", orig), ("mw_in_place_off", off)):
        vis[name] = {k: _summarize(cfg, spec, fn, n=8)["components_mean"] for k, fn in fns.items()}
    return vis


def _strip_traces(d: "dict[str, Any]") -> "dict[str, Any]":
    """Drop the per-step trace lists from the zero probe before serializing (keep the JSON compact)."""
    z = {k: v for k, v in d["zero_action"].items() if not k.startswith("trace_")}
    return {**d, "zero_action": z}


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reach-ppo-steps", type=int, default=40_000)
    ap.add_argument("--full-from-scratch-success", type=float, default=0.0,
                    help="the measured full-pick-place from-scratch success (for the C-vs-D call)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_diagnostics(StageBConfig(), Path(a.out) if a.out else None,
                    reach_ppo_steps=a.reach_ppo_steps, full_from_scratch_success=a.full_from_scratch_success)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
