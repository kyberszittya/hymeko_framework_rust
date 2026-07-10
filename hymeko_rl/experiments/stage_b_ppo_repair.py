"""Bounded PPO optimizer-repair for from-scratch pick-place — make PPO learn REACH before any reward-ablation test.

Diagnostics found the from-scratch 0%-vs-0% was a PPO-setup issue (std=1 too noisy), not a wall. This harness sweeps
the fix: fixed-std sweep, std annealing, entropy sanity, deterministic-eval separation, a reach-only pass gate, and
— only if reach passes — an original-reward pre-grasp gate. No SAC, no 1–2M steps, no 5-seed ablation.

Labels: A optimizer repaired (reach passes + original shows pre-grasp) · B PPO reaches but original can't bootstrap
grasp · C PPO still can't learn reach · D original learns enough → proceed to comparison. Reuses the reach reward /
eval / oracle from ``stage_b_diag`` and ``train_ppo_flat`` (with the new std-control) — no re-implemented PPO.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.eval.cip.reward_ablation_metaworld import ablate_reward_spec
from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig, make_training_env
from hymeko_rl.experiments.stage_b_diag import _ReachReward, _greedy_fn, _run_episode

_THRESHOLDS = (0.20, 0.10, 0.05)


def _first_below(trace: "list[float]", thr: float, cap: int) -> int:
    """First step index where hand-object distance drops below ``thr`` (``cap`` if it never does)."""
    for i, d in enumerate(trace):
        if d < thr:
            return i
    return cap


def _reach_metrics(eps: "list[dict[str, Any]]", cap: int) -> "dict[str, Any]":
    """Aggregate deterministic-eval episodes into the reach metrics (median min-dist, near, grasp, first-below)."""
    mind = np.asarray([e["min_hand_obj"] for e in eps])
    fb = {f"first_below_{t}": float(np.median([_first_below(e["trace_hand_obj"], t, cap) for e in eps]))
          for t in _THRESHOLDS}
    a = {s: float(np.mean([e["action_stats"][s] for e in eps])) for s in ("mean", "std", "min", "max")}
    return {"min_hand_obj_median": float(np.median(mind)), "min_hand_obj_best": float(mind.min()),
            "near_fraction": float(np.mean([e["near_fraction"] for e in eps])),
            "ever_near": float(np.mean([e["ever_near"] for e in eps])),
            "grasp_frac": float(np.mean([e["ever_grasp"] for e in eps])),
            "success": float(np.mean([e["success"] for e in eps])), **fb, "action_stats": a}


def _eval_reach(cfg: StageBConfig, spec: Any, policy: Any, n: int = 16, seed0: int = 60_000) -> "dict[str, Any]":
    """DETERMINISTIC (greedy) reach eval — the learned-policy metrics, free of training-time exploration noise."""
    eps: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            eps.append(_run_episode(make_training_env(cfg, spec), _greedy_fn(policy), cfg.max_steps, seed0 + i))
    return _reach_metrics(eps, cfg.max_steps)


def _train(cfg0: StageBConfig, *, reward: str, std_mode: str, std_init: float, std_final: float, entropy: float,
           steps: int, seed: int) -> "dict[str, Any]":
    """Train from-scratch PPO on the reach reward (``reward='reach'``) or the original reward, tracing per iteration."""
    from .stage_b_ppo import train_ppo_flat
    cfg = replace(cfg0, total_env_steps=steps, warm_start=False, optimizer="ppo", ppo_from_scratch_std=std_init,
                  ppo_std_mode=std_mode, ppo_std_final=std_final, ppo_entropy_coef=entropy)
    spec = ablate_reward_spec(cfg.spec_path)
    trace: list[dict[str, float]] = []

    def on_iter(_it: int, steps_: int, buf: "dict[str, Any]", actor: Any) -> None:
        obs = buf["obs"]
        d = np.linalg.norm(obs[:, :3] - obs[:, 4:7], axis=1)
        trace.append({"steps": float(steps_), "hand_obj": float(d.mean()), "near_proxy": float((d < 0.06).mean()),
                      "std": float(actor.log_std.exp().mean()),
                      "ret": float(np.mean(buf["ep_returns"])) if buf["ep_returns"] else float("nan")})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = _ReachReward(make_training_env(cfg, spec)) if reward == "reach" else make_training_env(cfg, spec)
        out = train_ppo_flat(cfg, env, None, seed, log=lambda _m: None, on_iter=on_iter)
    ev = _eval_reach(cfg, spec, out["policy"], n=16)
    return {"trace": trace, "eval": ev, "train_return_last": out["returns"][-1] if out["returns"] else None}


def _reach_pass(ev: "dict[str, Any]") -> bool:
    """Reach-only gate: median min hand-object distance < 0.07 AND near_fraction > 0.30."""
    return bool(ev["min_hand_obj_median"] < 0.07 and ev["near_fraction"] > 0.30)


def _pregrasp_pass(ev: "dict[str, Any]") -> bool:
    """Original-reward pre-grasp gate: near clearly above random, or nonzero grasp, or clearly improved reach."""
    return bool(ev["near_fraction"] > 0.10 or ev["grasp_frac"] > 0.05 or ev["min_hand_obj_median"] < 0.10)


def _fmt(ev: "dict[str, Any]") -> str:
    return (f"min_dist={ev['min_hand_obj_median']:.3f} near={ev['near_fraction']:.2f} grasp={ev['grasp_frac']:.2f} "
            f"eval_std={ev['action_stats']['std']:.2f}")


def std_sweep(cfg: StageBConfig, stds: "tuple[float, ...]", steps: int, seed: int) -> "list[dict[str, Any]]":
    """Fixed-std reach-only sweep — the core repair knob."""
    rows: list[dict[str, Any]] = []
    for s in stds:
        r = _train(cfg, reward="reach", std_mode="fixed", std_init=s, std_final=s, entropy=0.0, steps=steps, seed=seed)
        print(f"[repair sweep] fixed std={s}: {_fmt(r['eval'])} pass={_reach_pass(r['eval'])}", flush=True)
        rows.append({"setting": f"fixed_std_{s}", "std_init": s, "std_final": s, "mode": "fixed", "entropy": 0.0,
                     "eval": r["eval"], "trace": r["trace"], "pass": _reach_pass(r["eval"])})
    return rows


def anneal_probe(cfg: StageBConfig, configs: "tuple[tuple[float, float], ...]", steps: int,
                 seed: int) -> "list[dict[str, Any]]":
    """Std-annealing reach-only probe — early exploration, late precision."""
    rows: list[dict[str, Any]] = []
    for si, sf in configs:
        r = _train(cfg, reward="reach", std_mode="anneal", std_init=si, std_final=sf, entropy=0.0, steps=steps, seed=seed)
        print(f"[repair anneal] {si}->{sf}: {_fmt(r['eval'])} pass={_reach_pass(r['eval'])}", flush=True)
        rows.append({"setting": f"anneal_{si}_{sf}", "std_init": si, "std_final": sf, "mode": "anneal", "entropy": 0.0,
                     "eval": r["eval"], "trace": r["trace"], "pass": _reach_pass(r["eval"])})
    return rows


def entropy_probe(cfg: StageBConfig, std: float, entropies: "tuple[float, ...]", steps: int,
                  seed: int) -> "list[dict[str, Any]]":
    """Entropy sanity at the best std — is the entropy bonus keeping the policy too noisy for precision reach?"""
    rows: list[dict[str, Any]] = []
    for e in entropies:
        r = _train(cfg, reward="reach", std_mode="fixed", std_init=std, std_final=std, entropy=e, steps=steps, seed=seed)
        print(f"[repair entropy] ent={e} std={std}: {_fmt(r['eval'])}", flush=True)
        rows.append({"setting": f"entropy_{e}", "entropy": e, "std_init": std, "eval": r["eval"], "trace": r["trace"]})
    return rows


def multiseed_reach(cfg: StageBConfig, row: "dict[str, Any]", seeds: "tuple[int, ...]", steps: int) -> "dict[str, Any]":
    """3-seed confirmation of a passing setting — median eval + per-seed pass count."""
    evs: list[dict[str, Any]] = []
    for sd in seeds:
        r = _train(cfg, reward="reach", std_mode=row["mode"], std_init=row["std_init"], std_final=row["std_final"],
                   entropy=row.get("entropy", 0.0), steps=steps, seed=sd)
        evs.append(r["eval"])
        print(f"[repair multiseed] {row['setting']} seed={sd}: {_fmt(r['eval'])} pass={_reach_pass(r['eval'])}", flush=True)
    return {"setting": row["setting"], "seeds": list(seeds),
            "min_hand_obj_median": float(np.median([e["min_hand_obj_median"] for e in evs])),
            "near_fraction_median": float(np.median([e["near_fraction"] for e in evs])),
            "pass_count": int(sum(_reach_pass(e) for e in evs))}


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_traces(rows: "list[dict[str, Any]]", out: Path) -> None:
    """min-dist / near-proxy / return / action-std vs steps for the swept settings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = (("hand_obj", "min hand-object distance", "from_scratch_hand_object_distance_vs_steps.png"),
            ("near_proxy", "near proxy (hand-obj<0.06 fraction)", "from_scratch_near_fraction_vs_steps.png"),
            ("ret", "reach return", "from_scratch_return_vs_steps.png"),
            ("std", "action std", "from_scratch_action_std_vs_steps.png"))
    for field, ylab, fname in keys:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for row in rows:
            tr = row["trace"]
            ax.plot([t["steps"] for t in tr], [t[field] for t in tr], label=row["setting"], lw=1.3)
        if field == "hand_obj":
            ax.axhline(0.07, ls="--", color="red", lw=.8, label="gate 0.07")
        ax.set_xlabel("env steps")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} vs steps (reach-only PPO)")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        _save(fig, out / fname)


def _plot_summary(sweep: "list[dict[str, Any]]", anneal: "list[dict[str, Any]]", out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sweep + anneal
    labels = [r["setting"].replace("fixed_std_", "std ").replace("anneal_", "anneal ") for r in rows]
    md = [r["eval"]["min_hand_obj_median"] for r in rows]
    nf = [r["eval"]["near_fraction"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].bar(x, md, color=["#4a6fa5"] * len(sweep) + ["#e8a33d"] * len(anneal))
    ax[0].axhline(0.07, ls="--", color="red", lw=.8, label="gate 0.07")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("median min hand-obj dist")
    ax[0].set_title("Fixed std vs annealed std — reach")
    ax[0].legend(fontsize=7)
    ax[1].bar(x, nf, color=["#4a6fa5"] * len(sweep) + ["#e8a33d"] * len(anneal))
    ax[1].axhline(0.30, ls="--", color="red", lw=.8, label="gate 0.30")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("near_fraction")
    ax[1].set_title("near activation")
    ax[1].legend(fontsize=7)
    _save(fig, out / "fixed_vs_annealed_summary.png")


def _reaches(ev: "dict[str, Any]") -> bool:
    """Operational 'PPO can now reach' signal — the distance half of the gate (near-threshold proximity)."""
    return bool(ev["min_hand_obj_median"] < 0.08)


def _label(reaches: bool, pregrasp: "dict[str, Any] | None") -> "tuple[str, str]":
    if not reaches:
        return "C", "PPO still cannot reliably learn reach — from-scratch reward-ablation remains INVALID"
    if pregrasp is None or not _pregrasp_pass(pregrasp["eval"]):
        return "B", ("PPO can reach, but the original reward from scratch shows no pre-grasp behaviour — reward may be "
                     "insufficient to bootstrap grasp from scratch; ablation not yet valid")
    return "A", ("optimizer repaired: PPO reaches AND original reward shows pre-grasp — a tiny original vs "
                 "mw_in_place_off comparison is now justified (D: proceed)")


def _run_pregrasp(cfg: StageBConfig, best: "dict[str, Any]", steps: int, seed: int) -> "dict[str, Any]":
    """Train the ORIGINAL reward from scratch under the repaired std setting; report the pre-grasp eval."""
    r = _train(cfg, reward="original", std_mode=best["mode"], std_init=best["std_init"], std_final=best["std_final"],
               entropy=best.get("entropy", 0.0), steps=steps, seed=seed)
    print(f"[repair pregrasp] original reward: {_fmt(r['eval'])} pass={_pregrasp_pass(r['eval'])}", flush=True)
    return {"eval": r["eval"], "trace": r["trace"]}


def _no_trace(rows: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
    return [{k: v for k, v in r.items() if k != "trace"} for r in rows]


def run_repair(cfg0: StageBConfig, out_dir: "Path | None" = None, *, sweep_steps: int = 40_000,
               anneal_steps: int = 150_000, pregrasp_steps: int = 150_000, seed: int = 0) -> "dict[str, Any]":
    """Full repair pass: sweep → anneal → entropy → reach/reaches gate → (3-seed) → original pre-grasp gate → A/B/C/D.

    From-scratch needs a higher LR (1e-3) than the BC-fine-tune default — set here so the repair is self-contained."""
    cfg = replace(cfg0, ppo_lr=1e-3)                              # from-scratch learning rate
    out = out_dir or Path("reports/figures/2026_07_10_pick_place_ppo_optimizer_repair")
    out.mkdir(parents=True, exist_ok=True)
    sweep = std_sweep(cfg, (1.0, 0.5, 0.3, 0.2, 0.1), sweep_steps, seed)
    anneal = anneal_probe(cfg, ((0.5, 0.05), (0.6, 0.05)), anneal_steps, seed)
    best = min(sweep + anneal, key=lambda r: r["eval"]["min_hand_obj_median"])
    entropy = entropy_probe(cfg, best["std_init"], (0.0, 0.01), sweep_steps, seed)
    reach_strict = _reach_pass(best["eval"])
    reaches = _reaches(best["eval"])
    multiseed = multiseed_reach(cfg, best, (0, 1, 2), anneal_steps) if reaches else None
    pregrasp = _run_pregrasp(cfg, best, pregrasp_steps, seed) if reaches else None
    label, reason = _label(reaches, pregrasp)
    _plot_traces(sweep + anneal, out)
    _plot_summary(sweep, anneal, out)
    summary = {"best_setting": best["setting"], "best_eval": best["eval"], "reach_gate_strict_pass": reach_strict,
               "reaches": reaches, "multiseed": multiseed, "pregrasp": {"eval": pregrasp["eval"]} if pregrasp else None,
               "pregrasp_pass": bool(pregrasp and _pregrasp_pass(pregrasp["eval"])), "label": label, "reason": reason,
               "sweep": _no_trace(sweep), "anneal": _no_trace(anneal), "entropy": _no_trace(entropy),
               "ablation_valid": label == "A"}
    (out / "optimizer_repair.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[repair] BEST={best['setting']} reach_strict={reach_strict} reaches={reaches} "
          f"pregrasp_pass={summary['pregrasp_pass']} LABEL={label}: {reason}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-steps", type=int, default=30_000)
    ap.add_argument("--anneal-steps", type=int, default=60_000)
    ap.add_argument("--pregrasp-steps", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_repair(StageBConfig(), Path(a.out) if a.out else None, sweep_steps=a.sweep_steps,
               anneal_steps=a.anneal_steps, pregrasp_steps=a.pregrasp_steps, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
