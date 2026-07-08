"""seed_stabilized_demo_mix_v2 — variance-reduction ablation for the (NOT_ROBUST) demo-mix imitation.

The demo-mix BC fine-tune robustly injects sustained contact but robustly trades delivery, with high ft_dom
variance across training seeds (reports/2026-07-08-option-msdm-trainseed-robustness). This driver runs a compact
ablation of variance-reduction recipes (A control / B gentler-LR / C DAgger-anchor / D balanced-batch / E
val-selected checkpointing), each on ``train_seeds`` seeds, multi-seed-evaluating BOTH the final-epoch and the
val-selected checkpoints, and answers the central question: **can a validation gate pick the good seeds before
test?** (val→test predictivity). Deployable checkpoint saved ONLY for a POSITIVE_ROBUST recipe.

Frozen: v2b reward, TaskMonitor verifier, ledgers, selected DAgger baseline. NO scalar TD3/SAC/CQL, NO per-step
residual, NO option-RL, NO reward change, NO CORE edit. Reuses θ*-tagged pools, `_actor_multiseed`, `_classify`,
`compare_ftdom`, and the stabilized-BC trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.eval.critic_benchmark import spearman
from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.eval.multiseed import MultiSeedStats, aggregate, compare_ftdom
from hymeko_rl.eval.push_audit import audit_policy
from hymeko_rl.experiments.exp_galambos_coord_ab import make_env
from hymeko_rl.experiments.exp_option_msdm import (
    _REPORT_KEYS, _THETA_STAR, _actor_multiseed, _classify, _provenance, _stats_row)
from hymeko_rl.experiments.exp_option_retest import _fresh_actor, _option_demo_factory
from hymeko_rl.experiments.exp_vector_retest import _DIFFICULTY, certify_v2b, load_frozen_actor, measure_policy, wire_ledgers
from hymeko_rl.train.demo_mix import collect_tagged_demos
from hymeko_rl.train.stabilized_bc import StabilizedBCConfig, train_stabilized_bc

_VAL_SEED = 7000   # disjoint from the test eval seeds (9000/11000/13000/15000)


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class V2Config:
    stage: str = "full"
    out_dir: str = "experiments/2026_07_08_seed_stabilized"
    n_eval: int = 48
    eval_seeds: tuple[int, ...] = (9000, 11000, 13000, 15000)
    k_sustained: int = 5
    tag_eps: int = 300
    frac: float = 0.25                 # base mix (mix_25 was the headline)
    mix_total: int = 40_000
    train_seeds: int = 5
    val_eps: int = 12
    device: str = "cpu"

    @classmethod
    def smoke(cls) -> "V2Config":
        return cls(stage="smoke", out_dir="experiments/2026_07_08_seed_stabilized_smoke", n_eval=6,
                   eval_seeds=(9000, 11000), tag_eps=30, mix_total=3000, train_seeds=2, val_eps=4)

    @classmethod
    def full(cls, *, device: str = "cpu") -> "V2Config":
        return cls(device=device)


def _recipes(cfg: V2Config) -> dict[str, StabilizedBCConfig]:
    ep = 30 if cfg.stage == "smoke" else 200
    common = dict(frac_sustained=cfg.frac, mix_total=cfg.mix_total, batch=256, device=cfg.device)
    ve = max(10, ep // 2)
    return {
        "A_control": StabilizedBCConfig(lr=1e-3, epochs=ep, anchor_coef=0.0, balanced_batch=False, val_every=ve, **common),
        "B_gentle_lr": StabilizedBCConfig(lr=3e-4, epochs=ep, anchor_coef=0.0, balanced_batch=False, val_every=ve, **common),
        "C_anchor": StabilizedBCConfig(lr=1e-3, epochs=ep, anchor_coef=0.5, balanced_batch=False, val_every=ve, **common),
        "D_balanced": StabilizedBCConfig(lr=1e-3, epochs=ep, anchor_coef=0.0, balanced_batch=True, val_every=ve, **common),
        "E_valselect": StabilizedBCConfig(lr=1e-3, epochs=ep, anchor_coef=0.0, balanced_batch=False,
                                          val_every=max(5, ep // 8), **common),
    }


def _action_dev(actor: Any, dagger: Any, obs_sample: np.ndarray) -> float:
    """Mean L2 action deviation of the actor from the frozen DAgger over a fixed obs sample. Device-safe: the
    actor may be on CPU (moved there for the val gate) while the frozen DAgger is on CUDA (moved there for the
    training anchor) — run each on its own device and reduce on CPU."""
    with torch.no_grad():
        oa = torch.as_tensor(obs_sample, dtype=torch.float32, device=next(actor.parameters()).device)
        od = torch.as_tensor(obs_sample, dtype=torch.float32, device=next(dagger.parameters()).device)
        d = float((actor.action_mean(oa).cpu() - dagger.action_mean(od).cpu()).norm(dim=-1).mean())
    return d


def _make_val_fn(frozen: Any, cfg: V2Config, obs_sample: np.ndarray):
    def val_fn(actor: Any) -> dict:
        m = measure_policy(actor, cfg.val_eps, seed0=_VAL_SEED)
        a = audit_policy(lambda: make_env(coord=False, difficulty=_DIFFICULTY),
                         lambda e: greedy_action_fn(actor), name="val", n_episodes=cfg.val_eps, seed0=_VAL_SEED,
                         k_sustained=cfg.k_sustained)
        return {"ft_dom": m["ft_dom"], "monitor_score": m["monitor_score"],
                "body_driven_exploit": m["body_driven_exploit"], "sustained_push_per_ep": a.sustained_push_per_ep,
                "both_contact_frac": a.both_contact_frac, "ft_progress_in_contact": a.ft_progress_in_contact,
                "body_progress_in_contact": a.body_progress_in_contact, "arm_body_rate": a.arm_body_rate,
                "action_dev_from_dagger": _action_dev(actor, frozen, obs_sample)}
    return val_fn


def _classify_recipe(base: MultiSeedStats, per_seed_stats: list[MultiSeedStats], cfg: V2Config) -> dict:
    """5-way recipe verdict over training seeds (uses the val-selected aggregate)."""
    pooled = aggregate([{k: st.per_seed[k][i] for k in st.per_seed} for st in per_seed_stats
                        for i in range(st.n_seeds)], n_eval=cfg.n_eval)
    tie = compare_ftdom(base, pooled)
    seed_verdicts = [_classify(base, st, cfg)["verdict"] for st in per_seed_stats]
    n_pos = seed_verdicts.count("POSITIVE")
    n_prom = n_pos + seed_verdicts.count("PROMISING_TRADEOFF")
    ns = len(per_seed_stats)
    ftd_iqr = float(np.percentile([st.mean["ft_dom"] for st in per_seed_stats], 75)
                    - np.percentile([st.mean["ft_dom"] for st in per_seed_stats], 25))
    coverage_up = (pooled.mean["sustained_push_per_ep"] > base.mean["sustained_push_per_ep"] + 0.05
                   and pooled.mean["ft_progress_in_contact"] > base.mean["ft_progress_in_contact"])
    monitor_up = pooled.mean["monitor_score"] > base.mean["monitor_score"] + 0.02
    no_bad = (pooled.mean["body_driven_exploit"] <= base.mean["body_driven_exploit"] + 0.02
              and pooled.mean["audit_arm_body"] <= base.mean["audit_arm_body"] + 0.02)
    ftd_ok = tie.decision == "better" or (tie.decision == "tied" and pooled.mean["ft_dom"] >= base.mean["ft_dom"] - 0.02)
    majority_good = n_pos > ns / 2
    if coverage_up and no_bad and monitor_up and ftd_ok and majority_good and ftd_iqr <= 0.15:
        verdict = "POSITIVE_ROBUST"
    elif coverage_up and no_bad and monitor_up and (ftd_ok or n_prom > ns / 2):
        verdict = "PROMISING_BUT_HIGH_VARIANCE"
    elif coverage_up and tie.decision == "worse":
        verdict = "NOT_ROBUST" if ftd_iqr > 0.1 else "NEGATIVE_WITH_MECHANISM"
    else:
        verdict = "INCONCLUSIVE_WITH_NEXT_FIX"
    return {"verdict": verdict, "pooled_ftdom_tie_test": tie.as_dict(), "seed_verdicts": seed_verdicts,
            "n_positive": n_pos, "n_promising_or_better": n_prom, "n_seeds": ns, "ftd_iqr": round(ftd_iqr, 4),
            "coverage_up": bool(coverage_up), "monitor_up": bool(monitor_up), "no_bad": bool(no_bad),
            "pooled": _stats_row(pooled, _REPORT_KEYS)}


_RANK = {"POSITIVE_ROBUST": 4, "PROMISING_BUT_HIGH_VARIANCE": 3, "NEGATIVE_WITH_MECHANISM": 2,
         "NOT_ROBUST": 1, "INCONCLUSIVE_WITH_NEXT_FIX": 0}


def run(cfg: V2Config) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    frozen = load_frozen_actor(env)
    guards = wire_ledgers(env, frozen)
    reward = certify_v2b()
    _log(f"[v2] device={cfg.device} guards {guards['pipeline_schema']}/{guards['policy_provenance']} "
         f"| v2b delivers={reward.get('delivers')} | {cfg.train_seeds} seeds × 5 recipes")
    base = _actor_multiseed(frozen, cfg, tag="baseline")
    option_factory = _option_demo_factory(_THETA_STAR)
    pools = collect_tagged_demos(env, option_factory, n_episodes=cfg.tag_eps, seed0=0, k_sustained=cfg.k_sustained)
    _log(f"[v2] tagged pools: {pools.summary()}")
    obs_sample = pools.deliver_obs[:512] if pools.n_deliver else pools.sustained_obs[:512]
    val_fn = _make_val_fn(frozen, cfg, obs_sample)

    recipes = _recipes(cfg)
    results: dict[str, dict] = {}
    # per-metric (val-selected val metric, val-selected test metric) pairs across all cells, for Spearman(val,test)
    predict: dict[str, list[tuple[float, float]]] = {"ft_dom": [], "monitor_score": [], "sustained_push_per_ep": []}
    final_vs_sel: list[tuple[float, float]] = []       # (final_test_ftdom, val_selected_test_ftdom)
    recipe_best_state: dict[str, tuple[float, dict]] = {}   # rname -> (best-seed val-selected test ft_dom, state)
    for rname, rcfg in recipes.items():
        sel_stats, fin_stats = [], []
        for s in range(cfg.train_seeds):
            rc = StabilizedBCConfig(**{**rcfg.__dict__, "seed": s})
            res = train_stabilized_bc(pools, frozen, rc, fresh_actor_fn=lambda: _fresh_actor(env), val_fn=val_fn, log=_log)
            fin = _fresh_actor(env)
            fin.load_state_dict(res.final_state)
            fin.eval()
            sel = _fresh_actor(env)
            sel.load_state_dict(res.val_selected_state)
            sel.eval()
            fst = _actor_multiseed(fin, cfg, tag=f"{rname}_s{s}_final")
            sst = _actor_multiseed(sel, cfg, tag=f"{rname}_s{s}_sel")
            fin_stats.append(fst)
            sel_stats.append(sst)
            for key in predict:                         # val-selected checkpoint's VAL metric vs its TEST metric
                if key in res.best_val_metrics:
                    predict[key].append((res.best_val_metrics[key], sst.mean[key]))
            final_vs_sel.append((fst.mean["ft_dom"], sst.mean["ft_dom"]))
            if rname not in recipe_best_state or sst.mean["ft_dom"] > recipe_best_state[rname][0]:
                recipe_best_state[rname] = (sst.mean["ft_dom"], res.val_selected_state)
        cls = _classify_recipe(base, sel_stats, cfg)
        results[rname] = {"verdict": cls["verdict"], "classify": cls,
                          "val_selected_pooled": cls["pooled"],
                          "final_pooled": _stats_row(aggregate(
                              [{k: st.per_seed[k][i] for k in st.per_seed} for st in fin_stats
                               for i in range(st.n_seeds)], n_eval=cfg.n_eval), _REPORT_KEYS),
                          "per_seed_val_selected_ftdom": [round(st.mean["ft_dom"], 4) for st in sel_stats],
                          "per_seed_final_ftdom": [round(st.mean["ft_dom"], 4) for st in fin_stats]}
        _log(f"[v2] recipe {rname} → {cls['verdict']} | pooled ft_dom {cls['pooled']['ft_dom']['mean']} "
             f"(tie {cls['pooled_ftdom_tie_test']['decision']}) | {cls['n_positive']}/{cls['n_seeds']} POSITIVE seeds")

    # val→test predictivity per metric (the key decision variable) + does selection beat final on test?
    predictivity = {key: round(spearman([v for v, _ in pairs], [t for _, t in pairs]), 3)
                    for key, pairs in predict.items()}
    sel_beats_final = round(float(np.mean([b for _a, b in final_vs_sel]) - np.mean([a for a, _b in final_vs_sel])), 4)
    overall = max(results.items(), key=lambda kv: (_RANK[kv[1]["verdict"]], kv[1]["classify"]["pooled"]["ft_dom"]["mean"]))
    verdict = overall[1]["verdict"]
    _log(f"[v2] val→test predictivity (spearman): {predictivity} | val-selected − final test ft_dom = {sel_beats_final:+.3f}")
    _log(f"[v2] OVERALL best recipe = {overall[0]} → {verdict}")
    deployable_state = recipe_best_state.get(overall[0], (0.0, None))[1] if verdict == "POSITIVE_ROBUST" else None
    return _finalize(cfg, out, guards, reward, base, pools, results, overall[0], verdict, predictivity,
                     sel_beats_final, deployable_state, t0)


def _finalize(cfg, out, guards, reward, base, pools, results, best_recipe, verdict, predictivity, sel_beats_final,
              deployable_state, t0) -> dict:
    ckpt = None
    if verdict == "POSITIVE_ROBUST" and deployable_state is not None:   # save a checkpoint ONLY if POSITIVE_ROBUST
        ckpt = str(out / f"{best_recipe}_v2.pt")
        torch.save(deployable_state, ckpt)
        _log(f"[v2] POSITIVE_ROBUST → saved deployable checkpoint {ckpt}")
    ftd_pred = predictivity.get("ft_dom", 0.0)
    note = ("val-selection recovers good seeds (val predicts test ft_dom)" if ftd_pred > 0.5
            else "val-selection does NOT reliably recover good seeds (val poorly predicts test ft_dom)")
    result = {"experiment": "seed_stabilized_demo_mix_v2", "verdict": verdict, "best_recipe": best_recipe,
              "wall_s": round(time.perf_counter() - t0, 1), "n_eval": cfg.n_eval, "train_seeds": cfg.train_seeds,
              "eval_seeds": list(cfg.eval_seeds), "device": cfg.device, "guards": guards,
              "provenance_fields": _provenance(), "v2b_reward": reward, "baseline": _stats_row(base, _REPORT_KEYS),
              "tagged_pools": pools.summary(), "val_to_test_predictivity_spearman": predictivity,
              "val_selected_minus_final_test_ftdom": sel_beats_final, "predictivity_note": note,
              "recipes": results, "deployable_checkpoint": ckpt}
    figs = _plot(out, base, results)
    result["figures"] = figs
    (out / "results.json").write_text(json.dumps(result, indent=2))
    _log(f"[v2] VERDICT: {verdict} (best {best_recipe}); {note}")
    _log(f"[v2] wrote {out}/results.json ({result['wall_s']}s); figs={len(figs)}")
    return result


def _plot(out: Path, base: MultiSeedStats, results: dict, log=_log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[plot] matplotlib unavailable: {e}")
        return []
    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(names, [results[n]["classify"]["pooled"]["ft_dom"]["mean"] for n in names],
              yerr=[results[n]["classify"]["pooled"]["ft_dom"]["std"] for n in names], capsize=3)
    ax[0].axhline(base.mean["ft_dom"], ls="--", c="k", lw=0.8, label=f"baseline {base.mean['ft_dom']:.3f}")
    ax[0].set_title("val-selected pooled ft_dom (mean±std)")
    ax[0].legend(fontsize=8)
    for n in names:
        ax[1].scatter([n] * len(results[n]["per_seed_val_selected_ftdom"]),
                      results[n]["per_seed_val_selected_ftdom"], s=20)
    ax[1].axhline(base.mean["ft_dom"], ls="--", c="k", lw=0.8)
    ax[1].set_title("per-training-seed ft_dom (val-selected)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.suptitle("seed_stabilized_demo_mix_v2 — variance-reduction recipes")
    fig.tight_layout()
    p = out / "recipes.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [str(p)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="seed_stabilized_demo_mix_v2")
    ap.add_argument("--stage", choices=("smoke", "full"), default="full")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if args.stage == "smoke":
        cfg = V2Config.smoke()
        cfg.device = args.device            # smoke on the SAME device as the full run (catch device bugs)
    else:
        cfg = V2Config.full(device=args.device)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
