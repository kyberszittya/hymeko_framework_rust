"""Close the spec_bench → CIP loop: the arbitrated HTL success spec as a MetaWorld *reward* (A/B).

Two modes on one entry point (one file, mode arg — §6.5 #13):

* ``--offline`` (Phase 1, no RL): the reward-quality **de-risk**. For synthetic and the saved real coffee-push
  rollouts, score each candidate spec's per-episode robustness-return against native success
  (:func:`~hymeko_rl.eval.spec_bench.spec_reward.spec_reward_separation`). The gating result: the *arbitrated*
  spec's reward separates success from failure (large ``separation``/``auc``); the *raw* over-constrained spec's
  reward is offset-dominated and near-flat — it cannot drive RL. Three-form output (JSON table + a plot).
* ``--rl`` (Phase 2, gated): the drive — a bounded SAC A/B over reward-override arms
  {native, spec_arbitrated, spec_raw, monitor_aligned} on the same env/budget, evaluated on native success.
  Added after the Phase-1 gate; not present until then (no untested RL shipped).

Reuses the ``spec_reward`` bridge, the ``spec_bench`` datasets, and ``evaluate``'s dirs/plots — no
re-implemented metric, no re-implemented rendering (§6.1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hymeko_rl.eval.evaluate import experiment_dir, now_stamp
from hymeko_rl.eval.spec_bench.spec_bench import Rollout, synth_rollouts
from hymeko_rl.eval.spec_bench.spec_reward import (
    ARBITRATED_COFFEE_SPEC,
    RAW_COFFEE_SPEC,
    SpecRewardEnv,
    spec_reward_separation,
)

_REPO = Path(__file__).resolve().parents[2]
_REAL_ROLLOUTS = _REPO / "reports" / "figures" / "2026_07_13_coffee_push" / "coffee_push_rollouts.json"

# candidate spec → label, per dataset (the discriminating triples: arbitrated vs raw vs distractor).
_REAL_SPECS: "dict[str, str]" = {
    "arbitrated (formal)": ARBITRATED_COFFEE_SPEC,
    "arbitrated (weak-model gated)": "F(in_place >= 0.6 AND obj_to_target <= 0.071)",
    "raw (weak-model)": RAW_COFFEE_SPEC,
}
_SYNTH_SPECS: "dict[str, str]" = {
    "target (arbitrated)": "F(in_place >= 0.9)",
    "over-constrained raw": "F(near_object >= 0.7 AND grasp_success >= 0.5 AND in_place >= 0.9)",
    "distractor": "F(grasp_success >= 0.5)",
}


def _load_real_rollouts(path: Path) -> "list[Rollout]":
    data = json.loads(path.read_text())
    return [Rollout(trace=d["trace"], success=bool(d["success"])) for d in data]


def _score_dataset(specs: "dict[str, str]", rollouts: "list[Rollout]") -> "dict[str, Any]":
    """Reward-quality of each labelled spec on one dataset (separation / point-biserial / AUC)."""
    return {label: spec_reward_separation(formula, rollouts).as_dict() for label, formula in specs.items()}


def _plot(summary: "dict[str, Any]", out_path: Path) -> "Path | None":
    """AUC and point-biserial per spec per dataset — scale-free so arbitrated vs raw is directly comparable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001 — plotting is best-effort (§9), not load-bearing
        return None
    datasets = list(summary["datasets"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, ref, title in ((axes[0], "auc", 0.5, "ROC-AUC (return ranks success)"),
                                    (axes[1], "point_biserial", 0.0, "point-biserial corr(return, success)")):
        labels = sorted({lab for ds in datasets for lab in summary["datasets"][ds]})
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(datasets))
        for j, ds in enumerate(datasets):
            vals = [summary["datasets"][ds].get(lab, {}).get(metric, float("nan")) for lab in labels]
            ax.bar(x + j * width, vals, width, label=ds)
        ax.axhline(ref, ls="--", c="gray", lw=1)
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Spec-reward quality: arbitrated separates success/failure; raw is offset-flat", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_offline_derisk(out_dir: "Path | None" = None, *, n_synth: int = 200, seed: int = 0,
                       real_rollouts: Path = _REAL_ROLLOUTS) -> "dict[str, Any]":
    """Phase-1 de-risk: score candidate specs' reward-quality on synthetic + real rollouts. Writes JSON + plot."""
    out = out_dir or experiment_dir("reports/figures", "spec_reward_derisk")
    out.mkdir(parents=True, exist_ok=True)
    datasets: "dict[str, Any]" = {"synthetic": _score_dataset(_SYNTH_SPECS, synth_rollouts(n_synth, seed=seed))}
    if real_rollouts.exists():
        datasets["real coffee-push"] = _score_dataset(_REAL_SPECS, _load_real_rollouts(real_rollouts))
    else:
        print(f"[spec-reward-derisk] real rollouts absent ({real_rollouts}); synthetic only.", flush=True)
    summary: "dict[str, Any]" = {"kind": "spec-reward offline de-risk (reward-quality vs native success)",
                                 "stamp": now_stamp(), "n_synth": n_synth, "seed": seed, "datasets": datasets}
    (out / "spec_reward_derisk.json").write_text(json.dumps(summary, indent=2))
    plot = _plot(summary, out / "spec_reward_derisk.png")
    summary["plot"] = str(plot) if plot else None
    for ds, rows in datasets.items():
        print(f"[spec-reward-derisk] {ds}:", flush=True)
        for label, m in rows.items():
            print(f"    {label:32s} sep={m['separation']:+9.3f}  pb={m['point_biserial']:+.3f}  auc={m['auc']:.3f}"
                  f"  (succ_ret={m['mean_return_success']:+.2f} fail_ret={m['mean_return_failure']:+.2f})",
                  flush=True)
    print(f"[spec-reward-derisk] -> {out}", flush=True)
    return summary


# ── Phase 2: the from-scratch SAC A/B drive (does the arbitrated spec's reward DRIVE learning?) ───────────────
_ARMS = ("native", "spec_arbitrated", "spec_raw", "monitor_aligned")
_COFFEE_ENV_ID = "coffee-push-v3-goal-observable"
_COFFEE_POLICY = "SawyerCoffeePushV3Policy"
_ARM_SPEC = {"spec_arbitrated": ARBITRATED_COFFEE_SPEC, "spec_raw": RAW_COFFEE_SPEC}


class _FixedTaskEnv:
    """Pin the coffee-push task by dropping the reset seed — goal-observable ``reset()`` (no seed) holds a fixed
    goal (verified), so training/eval stay on ONE instance (removes the per-seeded-reset goal-randomisation
    confound). Object init still varies within the task = normal episode stochasticity."""

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **_kw: Any) -> Any:
        return self.env.reset()                              # drop the seed → fixed default task

    def step(self, action: Any) -> Any:
        return self.env.step(action)


def _bare_coffee_env(render_mode: "str | None" = None, *, fixed_task: bool = False) -> Any:
    """A fresh bare coffee-push goal-observable env (no reward override); ``fixed_task`` pins one instance."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]
        env = ENVS[_COFFEE_ENV_ID](render_mode=render_mode)  # type: ignore[arg-type]  # metaworld narrows to Literal
    return _FixedTaskEnv(env) if fixed_task else env


def _reward_env(arm: str, base: Any) -> Any:
    """Wrap ``base`` in the arm's reward override (native = the untouched MetaWorld reward)."""
    from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedEnv
    if arm == "native":
        return base
    if arm in _ARM_SPEC:
        return SpecRewardEnv(base, _ARM_SPEC[arm])
    if arm == "monitor_aligned":
        return MonitorAlignedEnv(base)
    raise ValueError(f"unknown arm {arm!r}; known {_ARMS}")


def _scripted_episode(arm: str, seed_i: int, *, noise: float, max_steps: int = 180, fixed_task: bool = False,
                      ) -> "tuple[list[np.ndarray], float, bool]":
    """One noisy scripted coffee-push episode through the arm's reward env → (obs list, arm-return, native-success)."""
    import warnings
    import metaworld.policies as mp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _reward_env(arm, _bare_coffee_env(fixed_task=fixed_task))
        obs, _ = env.reset(seed=seed_i)
        pol = getattr(mp, _COFFEE_POLICY)()
        rng = np.random.default_rng(seed_i)
        obs_all: "list[np.ndarray]" = []
        ret = 0.0
        ok = False
        for _ in range(max_steps):
            obs_all.append(np.asarray(obs, np.float32))
            act = np.clip(np.asarray(pol.get_action(obs), np.float32)
                          + rng.normal(0, noise, 4).astype(np.float32), -1.0, 1.0)
            obs, r, term, trunc, info = env.step(act)
            ret += float(r)
            ok = ok or bool(info.get("success", 0.0))
            if term or trunc:
                break
    return obs_all, ret, ok


def _fit_coffee_obs_norm(*, n: int = 12, seed: int = 0, noise: float = 0.3, fixed_task: bool = False,
                         ) -> "tuple[np.ndarray, np.ndarray]":
    """Fit obs mean/std from scripted coffee-push rollouts (a broad, task-relevant obs distribution)."""
    obs_all = [o for i in range(n)
               for o in _scripted_episode("native", seed + i, noise=noise, fixed_task=fixed_task)[0]]
    arr = np.asarray(obs_all, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def _mean_or_none(xs: "list[float]") -> "float | None":
    return round(float(np.mean(xs)), 4) if xs else None


def _certify_arm(arm: str, *, n: int = 12, noise: float = 0.7, seed: int = 0, fixed_task: bool = False,
                 ) -> "dict[str, Any]":
    """§3 oracle-certify: does the arm's reward rank scripted **native-success** episodes above failures?

    Rolls noisy scripted coffee-push episodes through the arm's reward env, splits by MetaWorld success, and
    checks mean arm-return(success) > mean(failure). ``delivers`` is only meaningful when both classes appear."""
    rolls = [_scripted_episode(arm, seed + i, noise=noise, fixed_task=fixed_task) for i in range(n)]
    succ = [ret for _obs, ret, ok in rolls if ok]
    fail = [ret for _obs, ret, ok in rolls if not ok]
    ms, mf = _mean_or_none(succ), _mean_or_none(fail)
    delivers = bool(succ and (mf is None or (ms is not None and ms > mf)))
    return {"delivers": delivers, "discriminating": bool(succ and fail), "n_success": len(succ), "n": n,
            "mean_ret_success": ms, "mean_ret_failure": mf}


def run_rl_arm(arm: str, seed: int, steps: int, *, mean: np.ndarray, std: np.ndarray, hidden: int = 256,
               device: str = "cpu", n_eval: int = 12, out_dir: "Path | None" = None, fixed_task: bool = False,
               ) -> "dict[str, Any]":
    """One from-scratch SAC seed on the arm's reward-override coffee-push env; native-success eval curve."""
    import torch

    from hymeko_rl.experiments.exp_metaworld_sac import _ObsNorm, _sac_success_eval
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    env = _ObsNorm(_reward_env(arm, _bare_coffee_env(fixed_task=fixed_task)), mean, std)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    scale = float(np.max(np.abs(np.asarray(env.action_space.high, np.float64))))
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim,
                               action_scale=scale, hidden=hidden, device=device)
    cfg = SACConfig(total_steps=steps, seed=seed, log_every=max(500, steps // 60),
                    eval_every=max(500, steps // 15), start_steps=min(5000, steps // 4), n_eval=n_eval)
    print(f"[spec-rl] arm={arm} seed={seed} steps={steps} obs={obs_dim} act={act_dim} device={device}", flush=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=_sac_success_eval(device, n=n_eval))
    res: "dict[str, Any]" = {"arm": arm, "seed": seed, "steps": steps,
                             "success_curve": [round(c, 4) for c in curve],
                             "final_success": round(curve[-1], 4) if curve else None,
                             "best_success": round(max(curve), 4) if curve else None}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(actor.state_dict(), out_dir / f"sac_{arm}_seed{seed}.pt")
        res["checkpoint"] = str(out_dir / f"sac_{arm}_seed{seed}.pt")
    print(f"[spec-rl] arm={arm} seed={seed} DONE final={res['final_success']} best={res['best_success']}", flush=True)
    return res


def _plot_rl(agg: "dict[str, Any]", per: "dict[str, list[dict[str, Any]]]", out_path: Path) -> "Path | None":
    """Success curves per arm + final-success median/IQR bars (three-form output, §9)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001 — viz best-effort
        return None
    arms = list(per)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm in arms:
        for r in per[arm]:
            c = r["success_curve"]
            if c:
                ax1.plot(range(len(c)), c, alpha=0.7, label=f"{arm} s{r['seed']}")
    ax1.set_xlabel("eval checkpoint")
    ax1.set_ylabel("native success rate")
    ax1.set_title("from-scratch SAC — native success over training")
    ax1.legend(fontsize=7)
    meds = [agg[arm]["final_success"]["median"] for arm in arms]
    q1 = [agg[arm]["final_success"]["median"] - agg[arm]["final_success"]["q1"] for arm in arms]
    q3 = [agg[arm]["final_success"]["q3"] - agg[arm]["final_success"]["median"] for arm in arms]
    ax2.bar(range(len(arms)), meds, yerr=[q1, q3], capsize=4)
    ax2.set_xticks(range(len(arms)))
    ax2.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("final native success (median, IQR)")
    ax2.set_title("does the spec reward DRIVE? arbitrated vs raw vs baselines")
    fig.suptitle("Spec-reward drive A/B (coffee-push)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _certify_and_gate(arms: "Sequence[str]", allow_uncertified: bool, *, fixed_task: bool = False,
                      ) -> "dict[str, Any]":
    """Certify every arm's reward (§3) and raise on a non-discriminating one unless waived."""
    cert = {arm: _certify_arm(arm, fixed_task=fixed_task) for arm in arms}
    for arm in arms:
        c = cert[arm]
        print(f"[spec-rl] certify {arm:16s} delivers={c['delivers']} (succ {c['n_success']}/{c['n']}, "
              f"ret succ={c['mean_ret_success']} fail={c['mean_ret_failure']})", flush=True)
        if not c["delivers"] and not allow_uncertified:
            raise RuntimeError(f"arm {arm!r} reward does not rank success>failure ({c}); pass allow_uncertified "
                               f"to train on it deliberately (a non-discriminating reward cannot drive — §3).")
    return cert


def _aggregate_arms(per: "dict[str, list[dict[str, Any]]]", cert: "dict[str, Any]", median_iqr: Any,
                    ) -> "dict[str, Any]":
    """Median/IQR of final/best native success per arm, carrying each arm's certification."""
    agg: "dict[str, Any]" = {}
    for arm, runs in per.items():
        fin = [r["final_success"] for r in runs if r["final_success"] is not None]
        best = [r["best_success"] for r in runs if r["best_success"] is not None]
        agg[arm] = {"final_success": median_iqr(fin), "best_success": median_iqr(best),
                    "certification": cert[arm]}
    return agg


def run_rl_ab(arms: "Sequence[str]", seeds: "Sequence[int]", steps: int, *, out_dir: "Path | None" = None,
              hidden: int = 256, device: str = "cpu", allow_uncertified: bool = False,
              fixed_task: bool = False) -> "dict[str, Any]":
    """From-scratch SAC A/B over reward arms on coffee-push. Certify → train (per seed) → aggregate → 3-form."""
    from hymeko_rl.eval.cip.reward_ablation_metaworld import _median_iqr
    out = out_dir or experiment_dir("reports/figures", "spec_reward_drive")
    out.mkdir(parents=True, exist_ok=True)
    mean, std = _fit_coffee_obs_norm(fixed_task=fixed_task)  # shared, reward-agnostic obs standardization
    cert = _certify_and_gate(arms, allow_uncertified, fixed_task=fixed_task)
    per: "dict[str, list[dict[str, Any]]]" = {
        arm: [run_rl_arm(arm, seed, steps, mean=mean, std=std, hidden=hidden, device=device, out_dir=out,
                         fixed_task=fixed_task)
              for seed in seeds]
        for arm in arms}
    agg = _aggregate_arms(per, cert, _median_iqr)
    winner = max(arms, key=lambda a: agg[a]["best_success"]["median"])
    summary: "dict[str, Any]" = {"kind": "spec-reward drive A/B (from-scratch SAC, coffee-push)",
                                 "stamp": now_stamp(), "arms": list(arms), "seeds": list(seeds), "steps": steps,
                                 "hidden": hidden, "device": device, "aggregate": agg, "winner": winner,
                                 "per_seed": per}
    (out / "spec_reward_drive.json").write_text(json.dumps(summary, indent=2, default=float))
    plot = _plot_rl(agg, per, out / "spec_reward_drive.png")
    summary["plot"] = str(plot) if plot else None
    for arm in arms:
        a = agg[arm]["best_success"]
        print(f"[spec-rl] {arm:16s} best_success median={a['median']:.3f} [{a['q1']:.3f},{a['q3']:.3f}]", flush=True)
    print(f"[spec-rl] winner={winner} -> {out}", flush=True)
    return summary


# ── Phase 2 (chosen design): BC-warm-start PRESERVATION A/B ──────────────────────────────────────────────────
# From-scratch coffee-push is intractable at a feasible local budget (measured: native ceiling = a single
# transient 1.0 spike at 40k, else 0), so it cannot discriminate reward quality. The preservation design starts
# from a BC-competent policy and asks: does each reward *preserve/improve* native success, or *degrade* it? This
# isolates reward quality from exploration hardness. Reuses the Stage-B BC anchor + PPO fine-tune + native eval.
def _coffee_cfg(steps: int, hidden: int, seed: int, bc_demos: int, n_eval: int, *, bc_epochs: int = 150,
                explore_std: float = 0.1) -> Any:
    """A StageBConfig pinned to coffee-push (via the injectable scripted policy) for BC→PPO warm-start.

    A **deliberately weak BC** (few demos / few epochs) leaves head-room so a *good* reward can be seen to LIFT the
    policy toward the native ceiling while a flat/mismatched reward cannot — the discriminating "does it drive?" test."""
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig
    return StageBConfig(task="coffee-push", policy_name=_COFFEE_POLICY, optimizer="ppo", warm_start=True,
                        total_env_steps=steps, hidden=hidden, seed=seed, bc_demos=bc_demos, bc_epochs=bc_epochs,
                        explore_std=explore_std, eval_episodes=n_eval, eval_episodes_post=n_eval)


def run_preserve_arm(arm: str, cfg: Any, base_state: "dict[str, Any]", spec: Any, *, n_eval: int, seed0: int,
                     out_dir: Path) -> "dict[str, Any]":
    """Fine-tune the shared BC policy under the arm's reward, then measure preserved native success."""
    import torch

    from hymeko_rl.experiments.exp_metaworld_reward_stageb import _policy_success_rate
    from hymeko_rl.experiments.stage_b_ppo import train_ppo_flat

    def _logf(m: str) -> None:
        print(f"[preserve {arm}] {m}", flush=True)
    env = _reward_env(arm, _bare_coffee_env())
    out = train_ppo_flat(cfg, env, base_state, cfg.seed, log=_logf)
    post = _policy_success_rate(cfg, spec, out["policy"], n=n_eval, seed0=seed0)
    torch.save(out["policy"].state_dict(), out_dir / f"ppo_{arm}_seed{cfg.seed}.pt")
    return {"arm": arm, "seed": cfg.seed, "post_success": round(post, 4),
            "final_return": round(out["returns"][-1], 2) if out["returns"] else None}


def run_preserve_ab(arms: "Sequence[str]", seeds: "Sequence[int]", steps: int, *, out_dir: "Path | None" = None,
                    hidden: int = 128, bc_demos: int = 8, bc_epochs: int = 60, explore_std: float = 0.1,
                    n_eval: int = 20, allow_uncertified: bool = False) -> "dict[str, Any]":
    """BC-warm-start drive/improvement A/B: shared (weak) BC policy per seed → per-arm PPO fine-tune → native-success
    delta. A good reward LIFTS the weak anchor toward the native ceiling; a flat/mismatched reward cannot."""
    from hymeko_rl.eval.cip.reward_ablation_metaworld import _median_iqr, ablate_reward_spec
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import _bc_base_policy
    out = out_dir or experiment_dir("reports/figures", "spec_reward_preserve")
    out.mkdir(parents=True, exist_ok=True)
    cert = _certify_and_gate(arms, allow_uncertified)
    per: "dict[str, list[dict[str, Any]]]" = {arm: [] for arm in arms}
    bc_per_seed: "list[float]" = []
    for seed in seeds:
        cfg = _coffee_cfg(steps, hidden, seed, bc_demos, n_eval, bc_epochs=bc_epochs, explore_std=explore_std)
        spec = ablate_reward_spec(cfg.spec_path)
        base, bc_info = _bc_base_policy(cfg)                 # shared BC-competent start (reward-agnostic)
        bc_succ = float(bc_info["bc_success_rate"])
        bc_per_seed.append(bc_succ)
        base_state = base.state_dict()
        print(f"[preserve] seed {seed}: BC success={bc_succ:.3f} ({bc_info['demo_success_episodes']}/{bc_demos} demos)",
              flush=True)
        for arm in arms:
            r = run_preserve_arm(arm, cfg, base_state, spec, n_eval=n_eval, seed0=60_000 + 1000 * seed, out_dir=out)
            r["bc_success"] = round(bc_succ, 4)
            r["preservation_delta"] = round(r["post_success"] - bc_succ, 4)
            per[arm].append(r)
            print(f"[preserve] seed {seed} {arm:16s} post={r['post_success']:.3f} "
                  f"delta={r['preservation_delta']:+.3f} (bc={bc_succ:.3f})", flush=True)
    agg = {arm: {"post_success": _median_iqr([r["post_success"] for r in per[arm]]),
                 "preservation_delta": _median_iqr([r["preservation_delta"] for r in per[arm]]),
                 "certification": cert[arm]} for arm in arms}
    winner = max(arms, key=lambda a: agg[a]["post_success"]["median"])
    summary: "dict[str, Any]" = {"kind": "spec-reward BC-warm-start preservation A/B (coffee-push, PPO fine-tune)",
                                 "stamp": now_stamp(), "arms": list(arms), "seeds": list(seeds), "steps": steps,
                                 "hidden": hidden, "bc_success": _median_iqr(bc_per_seed), "aggregate": agg,
                                 "winner": winner, "per_seed": per}
    (out / "spec_reward_preserve.json").write_text(json.dumps(summary, indent=2, default=float))
    summary["plot"] = str(_plot_preserve(agg, summary["bc_success"], out / "spec_reward_preserve.png") or "")
    for arm in arms:
        a = agg[arm]
        print(f"[preserve] {arm:16s} post={a['post_success']['median']:.3f}"
              f"[{a['post_success']['q1']:.3f},{a['post_success']['q3']:.3f}] "
              f"delta={a['preservation_delta']['median']:+.3f}", flush=True)
    print(f"[preserve] BC baseline={summary['bc_success']['median']:.3f} winner={winner} -> {out}", flush=True)
    return summary


def _plot_preserve(agg: "dict[str, Any]", bc: "dict[str, float]", out_path: Path) -> "Path | None":
    """Post-fine-tune native success per arm vs the BC baseline (preservation = at/above the dashed BC line)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                       # noqa: BLE001 — viz best-effort
        return None
    arms = list(agg)
    meds = [agg[arm]["post_success"]["median"] for arm in arms]
    lo = [agg[arm]["post_success"]["median"] - agg[arm]["post_success"]["q1"] for arm in arms]
    hi = [agg[arm]["post_success"]["q3"] - agg[arm]["post_success"]["median"] for arm in arms]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(arms)), meds, yerr=[lo, hi], capsize=4, color="#4C78A8")
    ax.axhline(bc["median"], ls="--", c="crimson", lw=1.5, label=f"BC baseline {bc['median']:.2f}")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("post-fine-tune native success (median, IQR)")
    ax.set_title("Does the reward PRESERVE a competent policy? (coffee-push, BC→PPO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="Phase 1 reward-quality de-risk (no RL)")
    ap.add_argument("--preserve", action="store_true", help="Phase 2 BC-warm-start preservation A/B (chosen design)")
    ap.add_argument("--rl", action="store_true", help="from-scratch SAC A/B (infeasible at local budget; kept for record)")
    ap.add_argument("--smoke-arm", default=None, help="single-arm single-seed RL smoke (e.g. spec_arbitrated)")
    ap.add_argument("--arms", nargs="+", default=list(_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-uncertified", action="store_true", help="train an arm whose reward does not certify")
    ap.add_argument("--bc-demos", type=int, default=8, help="preserve: scripted demos for the (weak) BC anchor")
    ap.add_argument("--bc-epochs", type=int, default=60, help="preserve: BC epochs (fewer → weaker anchor, more head-room)")
    ap.add_argument("--explore-std", type=float, default=0.1, help="preserve: PPO fine-tune action std")
    ap.add_argument("--fixed-task", action="store_true",
                    help="rl: pin ONE coffee-push instance (removes the goal-randomisation confound)")
    ap.add_argument("--n-synth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if a.smoke_arm:
        mean, std = _fit_coffee_obs_norm(fixed_task=a.fixed_task)
        print(f"[spec-rl] certify {a.smoke_arm}: {_certify_arm(a.smoke_arm, fixed_task=a.fixed_task)}", flush=True)
        run_rl_arm(a.smoke_arm, a.seeds[0], a.steps, mean=mean, std=std, hidden=a.hidden, device=a.device,
                   out_dir=out or experiment_dir("reports/figures", "spec_reward_smoke"), fixed_task=a.fixed_task)
        return 0
    if a.preserve:
        run_preserve_ab(a.arms, a.seeds, a.steps, out_dir=out, hidden=a.hidden, bc_demos=a.bc_demos,
                        bc_epochs=a.bc_epochs, explore_std=a.explore_std, allow_uncertified=a.allow_uncertified)
        return 0
    if a.rl:
        run_rl_ab(a.arms, a.seeds, a.steps, out_dir=out, hidden=a.hidden, device=a.device,
                  allow_uncertified=a.allow_uncertified, fixed_task=a.fixed_task)
        return 0
    if not a.offline:
        print("[spec-reward] no mode given; defaulting to --offline de-risk.", flush=True)
    run_offline_derisk(out, n_synth=a.n_synth, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
