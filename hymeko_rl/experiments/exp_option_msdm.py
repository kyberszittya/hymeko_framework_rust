"""option_dagger_multiseed_demo_mix_v1 — resolve the contact↔delivery trade-off from the option-level run.

Part 1: multi-seed (4 eval seeds × n=48) evaluation of {frozen baseline, regenerated option-DAgger, tuned option
expert} with a 2-proportion tie-test → is the ft_dom drop real or noise? Part 2: a demo-mix grid — tag each expert
state as sustained-PUSH vs delivery-completion and BC fine-tune at frac_sustained ∈ {0,.25,.5,.75,1} + curriculum
(the previous 100%-held BC over-fit holding; mixing in delivery-completion states is the fix). Part 3: a four-way
gate (POSITIVE / PROMISING_TRADEOFF / NEGATIVE_WITH_MECHANISM / INCONCLUSIVE_WITH_NEXT_FIX). Part 4: a bounded-θ
option ES refinement ONLY if a candidate is POSITIVE/PROMISING_TRADEOFF.

Frozen: v2b reward, TaskMonitor verifier, ledgers, selected DAgger baseline. NO scalar TD3/SAC/CQL, NO per-step
motor residual, NO vector actor smoke, NO reward change. Reuses θ* (cached), the DAgger loop, ledgers, audit, and
measure_policy from the prior drivers; adds only demo-mix pools + multi-seed aggregation.
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

from hymeko_rl.eval.evaluate import greedy_action_fn, render_episode_gif
from hymeko_rl.eval.multiseed import MultiSeedStats, aggregate, compare_ftdom
from hymeko_rl.eval.push_audit import audit_policy
from hymeko_rl.eval.task_monitor import TaskMonitor
from hymeko_rl.experiments.exp_galambos_coord_ab import make_env
from hymeko_rl.experiments.exp_option_retest import (
    OptionRetestConfig, _demo_af, _fresh_actor, _md5_actor, _option_demo_factory, dagger_escalate)
from hymeko_rl.experiments.exp_vector_retest import (
    _CKPT, _DIFFICULTY, certify_v2b, load_frozen_actor, measure_policy, wire_ledgers)
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos
from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.demo_mix import collect_tagged_demos, mix_pools

# cached tuned option θ* (experiments/2026_07_08_option_retest branch_BC_option; delivery 0.88, sustained-PUSH 1.71)
_THETA_STAR = (-0.004884286145727313, 0.6413038115033077, -0.026743897604853687,
               0.03873638475792048, 0.01094346099306752)
_HEADLINE = ("ft_dom", "monitor_pass", "monitor_score")
_TOL = 0.02
_COVERAGE_KEYS = ("sustained_push_per_ep", "both_contact_frac", "ft_progress_in_contact")


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class MsdmConfig:
    stage: str = "full"
    out_dir: str = "experiments/2026_07_08_option_msdm"
    n_eval: int = 48
    eval_seeds: tuple[int, ...] = (9000, 11000, 13000, 15000)
    k_sustained: int = 5
    tag_eps: int = 300
    tag_seed: int = 0
    mix_total: int = 40_000
    fracs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    bc_epochs: int = 200
    bc_batch: int = 256
    dagger_rounds: int = 3
    dagger_eps: int = 40
    n_demos: int = 400
    device: str = "cpu"                              # BC training device; eval is always CPU MuJoCo
    # training-seed-robustness stage (kato15 GPU): retrain the promising mixes on N training seeds
    robust_fracs: tuple[float, ...] = (0.25, 0.5, 0.75)
    train_seeds: int = 8

    @classmethod
    def smoke(cls) -> "MsdmConfig":
        return cls(stage="smoke", out_dir="experiments/2026_07_08_option_msdm_smoke", n_eval=6,
                   eval_seeds=(9000, 11000), tag_eps=30, mix_total=3000, fracs=(0.0, 0.5, 1.0),
                   bc_epochs=30, dagger_rounds=1, dagger_eps=8, n_demos=40)

    @classmethod
    def trainseeds(cls, *, device: str = "cpu") -> "MsdmConfig":
        return cls(stage="trainseeds", out_dir="experiments/2026_07_08_option_msdm_trainseeds", device=device,
                   robust_fracs=(0.25, 0.5, 0.75), train_seeds=8)

    @classmethod
    def trainseeds_smoke(cls) -> "MsdmConfig":
        return cls(stage="trainseeds_smoke", out_dir="experiments/2026_07_08_option_msdm_ts_smoke", n_eval=6,
                   eval_seeds=(9000, 11000), tag_eps=30, mix_total=3000, bc_epochs=30,
                   robust_fracs=(0.25, 0.5), train_seeds=2)


# --------------------------------------------------------------------------- multi-seed evaluation

def _actor_multiseed(actor: Any, cfg: MsdmConfig, log=_log, *, tag: str = "") -> MultiSeedStats:
    """Evaluate a (stateless MLP) actor across the eval-seed batches: measure_policy + audit per seed."""
    per_seed: list[dict] = []
    for s in cfg.eval_seeds:
        m = measure_policy(actor, cfg.n_eval, seed0=s)
        a = audit_policy(lambda: make_env(coord=False, difficulty=_DIFFICULTY),
                         lambda e: greedy_action_fn(actor), name=tag, n_episodes=cfg.n_eval, seed0=s,
                         k_sustained=cfg.k_sustained)
        per_seed.append({**m, "sustained_push_per_ep": a.sustained_push_per_ep,
                         "both_contact_frac": a.both_contact_frac,
                         "ft_progress_in_contact": a.ft_progress_in_contact,
                         "body_progress_in_contact": a.body_progress_in_contact,
                         "audit_arm_body": a.arm_body_rate, "audit_exploit": a.exploit_rate})
    st = aggregate(per_seed, n_eval=cfg.n_eval)
    if tag:
        log(f"[msdm/eval] {tag:16s} ft_dom {st.mean['ft_dom']:.3f}±{st.std['ft_dom']:.3f} "
            f"mon_score {st.mean['monitor_score']:.3f} sustainedPUSH/ep {st.mean['sustained_push_per_ep']:.2f}")
    return st


def _expert_multiseed(expert_factory, cfg: MsdmConfig, log=_log) -> MultiSeedStats:
    """Reference eval of the (stateful) tuned option expert — per-episode-fresh via _demo_af; audit + monitor."""
    per_seed: list[dict] = []
    for s in cfg.eval_seeds:
        a = audit_policy(lambda: make_env(coord=False, difficulty=_DIFFICULTY), _demo_af(expert_factory),
                         name="tuned_option_expert", n_episodes=cfg.n_eval, seed0=s, k_sustained=cfg.k_sustained)
        mon = TaskMonitor.from_env(make_env(coord=False, difficulty=_DIFFICULTY)).evaluate_policy(
            lambda: make_env(coord=False, difficulty=_DIFFICULTY), _demo_af(expert_factory), cfg.n_eval, seed0=s)
        per_seed.append({"raw_delivery": a.delivery_rate, "monitor_pass": mon["monitor_pass_rate"],
                         "monitor_score": mon["monitor_score_mean"],
                         "sustained_push_per_ep": a.sustained_push_per_ep, "both_contact_frac": a.both_contact_frac,
                         "ft_progress_in_contact": a.ft_progress_in_contact, "audit_exploit": a.exploit_rate})
    st = aggregate(per_seed, n_eval=cfg.n_eval)
    log(f"[msdm/eval] tuned_option_expert deliv {st.mean['raw_delivery']:.3f} "
        f"mon_score {st.mean['monitor_score']:.3f} sustainedPUSH/ep {st.mean['sustained_push_per_ep']:.2f}")
    return st


# --------------------------------------------------------------------------- Part 3: four-way gate

def _classify(base: MultiSeedStats, cand: MultiSeedStats, cfg: MsdmConfig) -> dict:
    """POSITIVE / PROMISING_TRADEOFF / NEGATIVE_WITH_MECHANISM / INCONCLUSIVE_WITH_NEXT_FIX for one candidate."""
    ftd = compare_ftdom(base, cand, alpha=0.05)
    coverage_up = all(cand.mean[k] > base.mean[k] + (0.05 if k == "sustained_push_per_ep" else 1e-4)
                      for k in _COVERAGE_KEYS)
    no_bad = (cand.mean["body_driven_exploit"] <= base.mean["body_driven_exploit"] + _TOL
              and cand.mean["audit_arm_body"] <= base.mean["audit_arm_body"] + _TOL
              and cand.mean["body_progress_in_contact"] <= base.mean["body_progress_in_contact"] + 1e-3)
    headline_preserved = (cand.mean["monitor_pass"] >= base.mean["monitor_pass"] - _TOL
                          and cand.mean["monitor_score"] >= base.mean["monitor_score"] - _TOL)
    mon_improved = cand.mean["monitor_score"] > base.mean["monitor_score"] + _TOL
    ftd_positive = ftd.decision == "better" or (ftd.decision == "tied"
                                                and cand.mean["ft_dom"] >= base.mean["ft_dom"] - _TOL)
    if coverage_up and no_bad and headline_preserved and ftd_positive:
        verdict = "POSITIVE"
    elif coverage_up and no_bad and mon_improved and ftd.decision == "tied":
        verdict = "PROMISING_TRADEOFF"
    elif coverage_up:
        verdict = "NEGATIVE_WITH_MECHANISM"
    else:
        verdict = "INCONCLUSIVE_WITH_NEXT_FIX"
    return {"verdict": verdict, "ftdom": ftd.as_dict(), "coverage_up": bool(coverage_up),
            "no_bad": bool(no_bad), "headline_preserved": bool(headline_preserved),
            "monitor_score_improved": bool(mon_improved)}


_RANK = {"POSITIVE": 3, "PROMISING_TRADEOFF": 2, "NEGATIVE_WITH_MECHANISM": 1, "INCONCLUSIVE_WITH_NEXT_FIX": 0}


# --------------------------------------------------------------------------- Part 2: demo-mix training

def _train_mix(env: Any, pools, frac: float, cfg: MsdmConfig, *, train_seed: int = 0, log=_log) -> Any:
    obs, acts = mix_pools(pools, frac, total=cfg.mix_total, seed=train_seed)
    actor = _fresh_actor(env, warm_start=True)
    behaviour_clone(actor, obs, acts, n_epochs=cfg.bc_epochs, batch=cfg.bc_batch, seed=train_seed, log_every=0,
                    device=cfg.device)
    actor.eval()
    log(f"[msdm/mix] frac_sustained={frac:.2f} train_seed={train_seed} trained on {len(obs)} samples")
    return actor


def _train_curriculum(env: Any, pools, cfg: MsdmConfig, *, log=_log) -> Any:
    """Delivery-completion first, then sustained-contact fine-tune (two BC passes on one warm-started actor)."""
    deliver = mix_pools(pools, 0.0, total=cfg.mix_total, seed=0)
    sustained = mix_pools(pools, 1.0, total=cfg.mix_total // 2, seed=1)
    actor = _fresh_actor(env, warm_start=True)
    behaviour_clone(actor, deliver[0], deliver[1], n_epochs=cfg.bc_epochs, batch=cfg.bc_batch, seed=0,
                    log_every=0, device="cpu")
    behaviour_clone(actor, sustained[0], sustained[1], n_epochs=cfg.bc_epochs // 2, batch=cfg.bc_batch, seed=1,
                    log_every=0, device="cpu")
    actor.eval()
    log("[msdm/mix] curriculum (delivery→sustained) trained")
    return actor


# --------------------------------------------------------------------------- driver

def run(cfg: MsdmConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    frozen = load_frozen_actor(env)
    guards = wire_ledgers(env, frozen)
    reward = certify_v2b()
    _log(f"[msdm] guards {guards['pipeline_schema']}/{guards['policy_provenance']} | v2b delivers={reward.get('delivers')}")
    option_factory = _option_demo_factory(_THETA_STAR)

    # Part 1 — robust multi-seed eval: baseline, regenerated option-DAgger, tuned expert
    _log(f"[msdm] Part 1: multi-seed eval ({len(cfg.eval_seeds)} seeds × n={cfg.n_eval}) ...")
    base_stats = _actor_multiseed(frozen, cfg, tag="baseline")
    orc = OptionRetestConfig(n_eval=cfg.n_eval, k_sustained=cfg.k_sustained, n_demos=cfg.n_demos,
                             bc_epochs=cfg.bc_epochs, bc_batch=cfg.bc_batch, dagger_rounds=cfg.dagger_rounds,
                             dagger_eps=cfg.dagger_eps)
    seed_o, seed_a = collect_galambos_demos(env, cfg.n_demos, 0, only_success=True, demonstrator=option_factory(env))
    regen = dagger_escalate(lambda: make_env(coord=False, difficulty=_DIFFICULTY), _fresh_actor(env), option_factory,
                            orc, seed_demos=(seed_o, seed_a), seed0=1000)
    regen_stats = _actor_multiseed(regen, cfg, tag="option_dagger")
    expert_stats = _expert_multiseed(option_factory, cfg)
    ftd_regen = compare_ftdom(base_stats, regen_stats)
    _log(f"[msdm] Part 1 tie-test option_dagger vs baseline ft_dom: {ftd_regen.decision} "
         f"(p={ftd_regen.p:.3f}, Δ={ftd_regen.delta:+.3f})")

    # Part 2 — demo-mix grid
    _log("[msdm] Part 2: tagging expert demos (sustained vs delivery-completion) ...")
    pools = collect_tagged_demos(env, option_factory, n_episodes=cfg.tag_eps, seed0=cfg.tag_seed,
                                 k_sustained=cfg.k_sustained)
    _log(f"[msdm] tagged pools: {pools.summary()}")
    candidates: list[dict] = []
    trained: dict[str, Any] = {}
    for frac in cfg.fracs:
        actor = _train_mix(env, pools, frac, cfg)
        st = _actor_multiseed(actor, cfg, tag=f"mix_{int(frac*100)}")
        cls = _classify(base_stats, st, cfg)
        trained[f"mix_{int(frac*100)}"] = actor
        candidates.append({"name": f"mix_{int(frac*100)}", "frac_sustained": frac,
                           "stats": st, "classify": cls, "md5": _md5_actor(actor)})
    cur = _train_curriculum(env, pools, cfg)
    cur_st = _actor_multiseed(cur, cfg, tag="curriculum")
    trained["curriculum"] = cur
    candidates.append({"name": "curriculum", "frac_sustained": None, "stats": cur_st,
                       "classify": _classify(base_stats, cur_st, cfg), "md5": _md5_actor(cur)})

    # Part 3 — pick best candidate + overall verdict
    best = max(candidates, key=lambda c: (_RANK[c["classify"]["verdict"]], c["stats"].mean["ft_dom"]))
    verdict = best["classify"]["verdict"]
    _log(f"[msdm] Part 3: best={best['name']} verdict={verdict}")

    # Part 4 — Branch E gate (conditional; documented, not an uncontrolled sweep)
    branch_e = {"ran": False, "reason": "gate closed: no POSITIVE/PROMISING_TRADEOFF candidate"}
    if verdict in ("POSITIVE", "PROMISING_TRADEOFF"):
        branch_e = _branch_e(cfg, best, trained[best["name"]], base_stats, out)

    return _finalize(cfg, out, guards, reward, base_stats, regen_stats, expert_stats, ftd_regen, pools,
                     candidates, best, trained, verdict, branch_e, option_factory, t0)


def _branch_e(cfg: MsdmConfig, best: dict, best_actor: Any, base: MultiSeedStats, out: Path) -> dict:
    """Bounded-θ option ES refinement — only reached when a candidate is POSITIVE/PROMISING_TRADEOFF. Scope-limited
    to a short option_cem refinement of the option (high-level params only); reports whether a further-tuned option
    would raise sustained-contact coverage. Does NOT retrain a third imitation round (that would be a new sweep)."""
    from hymeko_rl.train.option_search import OptionSearchConfig, option_cem
    _log("[msdm] Part 4: Branch E — bounded-θ option ES refinement (gate open) ...")
    res = option_cem(lambda: make_env(coord=False, difficulty=_DIFFICULTY),
                     OptionSearchConfig(pop=12, elite=4, iters=6, n_eval_eps=6, log_every=1))
    return {"ran": True, "improved_over_scripted": bool(res.improved_over_scripted),
            "theta_named": res.theta_named, "best_metrics": res.best_metrics,
            "note": "option ES can refine the high-level params further; a fresh imitation round is the next step"}


# --------------------------------------------------------------------------- finalize / io

def _stats_row(st: MultiSeedStats, keys: tuple[str, ...]) -> dict:
    return {k: {"mean": round(st.mean.get(k, float("nan")), 4), "std": round(st.std.get(k, 0.0), 4)} for k in keys}


_REPORT_KEYS = ("ft_dom", "raw_delivery", "monitor_pass", "monitor_score", "sustained_push_per_ep",
                "both_contact_frac", "ft_progress_in_contact", "body_progress_in_contact",
                "body_driven_exploit", "audit_arm_body", "audit_exploit")


def _finalize(cfg, out, guards, reward, base, regen, expert, ftd_regen, pools, candidates, best, trained,
              verdict, branch_e, option_factory, t0) -> dict:
    reason = _reason(verdict, best, base)
    ckpt = None
    if verdict == "POSITIVE":                       # save a deployable checkpoint ONLY if POSITIVE
        ckpt = str(out / f"{best['name']}_msdm.pt")
        torch.save(trained[best["name"]].state_dict(), ckpt)
        _log(f"[msdm] POSITIVE → saved deployable checkpoint {ckpt}")
    figs = _plots(out, base, candidates)
    gifs = _gif(out, trained[best["name"]], option_factory)
    result = {
        "experiment": "option_dagger_multiseed_demo_mix_v1", "verdict": verdict, "reason": reason,
        "stage": cfg.stage, "wall_s": round(time.perf_counter() - t0, 1),
        "n_eval": cfg.n_eval, "eval_seeds": list(cfg.eval_seeds),
        "guards": guards, "provenance_fields": _provenance(), "v2b_reward": reward,
        "part1_multiseed": {
            "baseline": _stats_row(base, _REPORT_KEYS), "option_dagger": _stats_row(regen, _REPORT_KEYS),
            "tuned_option_expert": _stats_row(expert, _REPORT_KEYS),
            "option_dagger_vs_baseline_ftdom_tie_test": ftd_regen.as_dict(),
            "baseline_violation_dist": base.violation_dist, "option_dagger_violation_dist": regen.violation_dist},
        "tagged_pools": pools.summary(),
        "part2_candidates": [{"name": c["name"], "frac_sustained": c["frac_sustained"], "md5": c["md5"],
                              "metrics": _stats_row(c["stats"], _REPORT_KEYS),
                              "classify": c["classify"]} for c in candidates],
        "part3_best": {"name": best["name"], "verdict": verdict, "classify": best["classify"]},
        "part4_branch_e": branch_e,
        "deployable_checkpoint": ckpt, "figures": figs, "gifs": gifs,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    _log(f"[msdm] VERDICT: {verdict}")
    _log(f"[msdm] {reason}")
    _log(f"[msdm] wrote {out}/results.json ({result['wall_s']}s); figs={len(figs)} gifs={len(gifs)}")
    return result


def _reason(verdict: str, best: dict, base: MultiSeedStats) -> str:
    st, cl = best["stats"], best["classify"]
    d = cl["ftdom"]
    tail = (f"{best['name']}: ft_dom {st.mean['ft_dom']:.3f}±{st.std['ft_dom']:.3f} vs baseline "
            f"{base.mean['ft_dom']:.3f} (tie-test {d['decision']}, p={d['p']}), monitor_score "
            f"{st.mean['monitor_score']:.3f} vs {base.mean['monitor_score']:.3f}, sustained-PUSH/ep "
            f"{st.mean['sustained_push_per_ep']:.2f} vs {base.mean['sustained_push_per_ep']:.2f}")
    heads = {
        "POSITIVE": "a demo-mix imitation policy improves/preserves every headline metric AND raises "
                    "sustained-contact coverage without exploit — " + tail,
        "PROMISING_TRADEOFF": "a demo-mix policy raises monitor_score + sustained-contact coverage with ft_dom "
                              "statistically TIED to baseline (not significantly worse) and no exploit — " + tail,
        "NEGATIVE_WITH_MECHANISM": "the best demo-mix policy still trades a statistically-significant amount of "
                                   "ft_dom for contact — " + tail,
        "INCONCLUSIVE_WITH_NEXT_FIX": "no demo mixture created sustained-contact coverage above baseline — " + tail,
    }
    return heads[verdict]


def _provenance() -> dict:
    from hymeko_rl.eval.task_monitor.provenance import file_md5
    return {"policy_provenance": "PASS", "actor_checkpoint_hash": file_md5(_CKPT),
            "reward_file": "data/robotics/galambos_task_deliver_v2b.hymeko"}


def _plots(out: Path, base: MultiSeedStats, candidates: list[dict], log=_log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[plots] matplotlib unavailable: {e}")
        return []
    names = ["baseline"] + [c["name"] for c in candidates]
    ftd = [base.mean["ft_dom"]] + [c["stats"].mean["ft_dom"] for c in candidates]
    ftd_e = [base.std["ft_dom"]] + [c["stats"].std["ft_dom"] for c in candidates]
    mon = [base.mean["monitor_score"]] + [c["stats"].mean["monitor_score"] for c in candidates]
    sus = [base.mean["sustained_push_per_ep"]] + [c["stats"].mean["sustained_push_per_ep"] for c in candidates]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].bar(names, ftd, yerr=ftd_e, capsize=3)
    ax[0].axhline(base.mean["ft_dom"], ls="--", c="k", lw=0.8)
    ax[0].set_title("ft_dom (mean±std, dashed=baseline)")
    ax[1].bar(names, mon)
    ax[1].axhline(base.mean["monitor_score"], ls="--", c="k", lw=0.8)
    ax[1].set_title("monitor_score")
    ax[2].bar(names, sus)
    ax[2].axhline(base.mean["sustained_push_per_ep"], ls="--", c="k", lw=0.8)
    ax[2].set_title("sustained-PUSH windows / ep")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.suptitle("Demo-mix grid — multi-seed (4×48); contact↔delivery trade-off")
    fig.tight_layout()
    p = out / "demo_mix_grid.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [str(p)]


def _gif(out: Path, actor: Any, option_factory, log=_log) -> list[str]:
    gifs = []
    try:
        render_episode_gif(make_env(coord=False, difficulty=_DIFFICULTY), greedy_action_fn(actor),
                           str(out / "best_candidate.gif"), seed=9000, width=640)
        gifs.append(str(out / "best_candidate.gif"))
        e = make_env(coord=False, difficulty=_DIFFICULTY)
        render_episode_gif(e, _demo_af(option_factory)(e), str(out / "tuned_option_expert.gif"), seed=9000, width=640)
        gifs.append(str(out / "tuned_option_expert.gif"))
    except Exception as ex:  # noqa: BLE001
        log(f"[gif] render failed: {ex}")
    return gifs


def _summ(arr: list[float]) -> dict:
    """median / IQR / min / max / mean over training seeds."""
    a = np.asarray(arr, dtype=float)
    return {"median": round(float(np.median(a)), 4), "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "mean": round(float(a.mean()), 4),
            "per_seed": [round(float(x), 4) for x in a]}


def run_trainseeds(cfg: MsdmConfig) -> dict[str, Any]:
    """Training-seed robustness: retrain each robust_frac on ``train_seeds`` training seeds, multi-seed-eval each,
    and report median/IQR + a verdict distribution + a pooled tie-test vs baseline. Answers 'is the demo-mix
    POSITIVE training-seed-robust, or a single-seed artifact?' (kato15 GPU layer). Same eval protocol / guards."""
    t0 = time.perf_counter()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    frozen = load_frozen_actor(env)
    guards = wire_ledgers(env, frozen)
    reward = certify_v2b()
    _log(f"[ts] device={cfg.device} | guards {guards['pipeline_schema']}/{guards['policy_provenance']} "
         f"| v2b delivers={reward.get('delivers')} | {cfg.train_seeds} train seeds × fracs {cfg.robust_fracs}")
    base = _actor_multiseed(frozen, cfg, tag="baseline")
    option_factory = _option_demo_factory(_THETA_STAR)
    pools = collect_tagged_demos(env, option_factory, n_episodes=cfg.tag_eps, seed0=cfg.tag_seed,
                                 k_sustained=cfg.k_sustained)
    _log(f"[ts] tagged pools: {pools.summary()}")
    per_frac: list[dict] = []
    for frac in cfg.robust_fracs:
        pooled_eval: list[dict] = []
        ftd, mons, susp, verdicts = [], [], [], []
        for ts in range(cfg.train_seeds):
            actor = _train_mix(env, pools, frac, cfg, train_seed=ts)
            st = _actor_multiseed(actor, cfg, tag=f"mix_{int(frac*100)}_ts{ts}")
            cls = _classify(base, st, cfg)
            pooled_eval.extend([{k: st.per_seed[k][i] for k in st.per_seed} for i in range(st.n_seeds)])
            ftd.append(st.mean["ft_dom"])
            mons.append(st.mean["monitor_score"])
            susp.append(st.mean["sustained_push_per_ep"])
            verdicts.append(cls["verdict"])
        pooled = aggregate(pooled_eval, n_eval=cfg.n_eval)
        tie = compare_ftdom(base, pooled)
        vdist = {v: verdicts.count(v) for v in sorted(set(verdicts))}
        n_pos = verdicts.count("POSITIVE")
        n_prom = n_pos + verdicts.count("PROMISING_TRADEOFF")
        robust = ("ROBUST_POSITIVE" if n_pos > cfg.train_seeds / 2
                  else "ROBUST_PROMISING" if n_prom > cfg.train_seeds / 2
                  else "NOT_ROBUST")
        per_frac.append({"frac_sustained": frac, "robust_verdict": robust, "verdict_distribution": vdist,
                         "ft_dom": _summ(ftd), "monitor_score": _summ(mons), "sustained_push_per_ep": _summ(susp),
                         "pooled_ftdom_tie_test_vs_baseline": tie.as_dict()})
        _log(f"[ts] frac={frac:.2f} → {robust} | ft_dom med {_summ(ftd)['median']} (IQR {_summ(ftd)['iqr']}) "
             f"| mon_score med {_summ(mons)['median']} | verdicts {vdist} | pooled tie-test {tie.decision} p={tie.p:.3f}")
    order = {"ROBUST_POSITIVE": 2, "ROBUST_PROMISING": 1, "NOT_ROBUST": 0}
    best = max(per_frac, key=lambda f: (order[f["robust_verdict"]], f["ft_dom"]["median"]))
    result = {"experiment": "option_dagger_multiseed_demo_mix_v1_trainseeds", "verdict": best["robust_verdict"],
              "best_frac": best["frac_sustained"], "wall_s": round(time.perf_counter() - t0, 1),
              "n_eval": cfg.n_eval, "eval_seeds": list(cfg.eval_seeds), "train_seeds": cfg.train_seeds,
              "device": cfg.device, "guards": guards, "provenance_fields": _provenance(), "v2b_reward": reward,
              "baseline": _stats_row(base, _REPORT_KEYS), "tagged_pools": pools.summary(), "per_frac": per_frac}
    figs = _ts_plot(out, base, per_frac)
    result["figures"] = figs
    (out / "results.json").write_text(json.dumps(result, indent=2))
    _log(f"[ts] VERDICT: {best['robust_verdict']} (best frac {best['frac_sustained']})")
    _log(f"[ts] wrote {out}/results.json ({result['wall_s']}s); figs={len(figs)}")
    return result


def _ts_plot(out: Path, base: MultiSeedStats, per_frac: list[dict], log=_log) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[ts-plot] matplotlib unavailable: {e}")
        return []
    fracs = [f["frac_sustained"] for f in per_frac]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(fracs, [f["ft_dom"]["median"] for f in per_frac],
                   yerr=[[f["ft_dom"]["median"] - f["ft_dom"]["min"] for f in per_frac],
                         [f["ft_dom"]["max"] - f["ft_dom"]["median"] for f in per_frac]],
                   fmt="o-", capsize=4, label="mix (median, min–max over train seeds)")
    ax[0].axhline(base.mean["ft_dom"], ls="--", c="k", lw=0.8, label=f"baseline {base.mean['ft_dom']:.3f}")
    ax[0].set_xlabel("frac_sustained")
    ax[0].set_ylabel("ft_dom")
    ax[0].set_title("ft_dom vs mix (train-seed spread)")
    ax[0].legend(fontsize=8)
    ax[1].plot(fracs, [f["monitor_score"]["median"] for f in per_frac], "o-", label="monitor_score (median)")
    ax[1].plot(fracs, [f["sustained_push_per_ep"]["median"] for f in per_frac], "s-", label="sustained-PUSH/ep (median)")
    ax[1].axhline(base.mean["monitor_score"], ls="--", c="k", lw=0.6)
    ax[1].set_xlabel("frac_sustained")
    ax[1].set_title("monitor_score + sustained-PUSH (median)")
    ax[1].legend(fontsize=8)
    n_ts = len(per_frac[0]["ft_dom"]["per_seed"])
    fig.suptitle(f"Training-seed robustness ({n_ts} seeds × {len(per_frac)} mixes)")
    fig.tight_layout()
    p = out / "trainseed_robustness.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return [str(p)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="option_dagger_multiseed_demo_mix_v1")
    ap.add_argument("--stage", choices=("smoke", "full", "trainseeds", "trainseeds_smoke"), default="full")
    ap.add_argument("--device", default="auto", help="BC training device for the trainseeds stage (auto/cpu/cuda)")
    args = ap.parse_args(argv)
    if args.stage == "trainseeds":
        run_trainseeds(MsdmConfig.trainseeds(device=args.device))
    elif args.stage == "trainseeds_smoke":
        run_trainseeds(MsdmConfig.trainseeds_smoke())
    else:
        run(MsdmConfig.smoke() if args.stage == "smoke" else MsdmConfig())
    return 0


if __name__ == "__main__":
    sys.exit(main())
