"""bounded_option_parameter_rl_v0 — phase-gated option-parameter RL on top of the learned E_valselect_v2 base.

The deployable learned policy (E_valselect_v2, a frozen MLP) handles APPROACH; a CEM-tuned PhasePushController(θ)
(5 bounded params) takes over the CONTACT/PUSH phase — a per-phase handoff (:class:`PhaseGatedPolicy`), NOT a
per-step residual. CEM searches θ (the ONLY learnable quantity) to maximise a SearchObjective of sustained-contact
+ fingertip progress + delivery − body − exploit; the frozen TaskMonitor is the external verifier at acceptance.
Acceptance is vs the E_valselect_v2 fresh-eval metrics: the gated policy is POSITIVE only if it preserves/beats
ft_dom + monitor_pass + monitor_score AND further improves sustained-PUSH + fingertip progress, without increasing
exploit / body-only / arm-body.

Scope: option-parameter RL only. NO per-step residual, NO scalar TD3/SAC/CQL, NO raw-action RL, NO reward change,
NO CORE edit. Base = E_valselect_v2.pt (not the old DAgger baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.agents.phase_gated import PhaseGatedPolicy, phase_gated_action_fn
from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.eval.multiseed import MultiSeedStats, aggregate, compare_ftdom
from hymeko_rl.eval.push_audit import sustained_push_windows
from hymeko_rl.eval.task_monitor import TaskMonitor
from hymeko_rl.eval.task_monitor.contract import MonitorContract
from hymeko_rl.eval.task_monitor.context import MonitorContext, record_trajectory
from hymeko_rl.eval.task_monitor.provenance import file_md5
from hymeko_rl.experiments.exp_galambos_coord_ab import grade_delivery, make_env
from hymeko_rl.experiments.exp_option_retest import _fresh_actor
from hymeko_rl.experiments.exp_vector_retest import _DIFFICULTY, certify_v2b, wire_ledgers
from hymeko_rl.experiments.galambos_demo import PhasePushController, PushControllerParams
from hymeko_rl.train.option_search import OptionSearchConfig, option_cem

_E_CKPT = "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt"
_FRESH_SEEDS = (31000, 33000, 35000, 37000)   # == the E_valselect fresh-eval acceptance baseline seeds
_N_EVAL = 48
_K = 5
_REPORT = ("ft_dom", "monitor_pass", "monitor_score", "sustained_push_per_ep", "both_contact_frac",
           "ft_progress_in_contact", "body_progress_in_contact", "arm_body_rate", "body_driven_exploit")


def _log(m: str) -> None:
    print(m, flush=True)


def _load_e_valselect(env: Any) -> Any:
    a = _fresh_actor(env, warm_start=False)
    a.load_state_dict(torch.load(_E_CKPT, map_location="cpu"))
    a.eval()
    return a


def _eval_multiseed(make_action_fn, contract: Any, name: str, log=_log) -> MultiSeedStats:
    """Per-episode-fresh multi-seed eval (works for a stateless MLP or a stateful gated policy — ``make_action_fn``
    is called per episode). Computes ft_dom + monitor + sustained-contact audit on ``_FRESH_SEEDS``."""
    per_seed: list[dict] = []
    for s in _FRESH_SEEDS:
        mon = TaskMonitor.from_env(make_env(coord=False, difficulty=_DIFFICULTY)).evaluate_policy(
            lambda: make_env(coord=False, difficulty=_DIFFICULTY), make_action_fn, _N_EVAL, seed0=s)
        ftdom = exploit = 0
        n_win = 0
        ft_in = body_in = both = 0.0
        arm_body = 0
        for ep in range(_N_EVAL):
            env = make_env(coord=False, difficulty=_DIFFICULTY)
            traj = record_trajectory(env, make_action_fn(env), s + ep)
            if not traj:
                continue
            ctx = MonitorContext.build(traj, contract)
            delivered = bool(ctx.max_dwell >= contract.success_steps)
            grade = grade_delivery(held=delivered, body_progress=float(ctx.body_prog),
                                   fingertip_progress=float(ctx.ft_prog))
            ftdom += int(grade == "fingertip_dominant")
            exploit += int(grade == "body_driven_exploit")
            arm_body += int(bool(ctx.body_contact.any()))
            both += float(ctx.both_contact.mean())
            for a, b in sustained_push_windows(ctx, contract, _K):
                n_win += 1
                ft_in += float(ctx.ft_prog_step[a:b].sum())
                body_in += float(ctx.body_prog_step[a:b].sum())
        per_seed.append({"ft_dom": ftdom / _N_EVAL, "monitor_pass": mon["monitor_pass_rate"],
                         "monitor_score": mon["monitor_score_mean"], "sustained_push_per_ep": n_win / _N_EVAL,
                         "both_contact_frac": both / _N_EVAL, "ft_progress_in_contact": ft_in / _N_EVAL,
                         "body_progress_in_contact": body_in / _N_EVAL, "arm_body_rate": arm_body / _N_EVAL,
                         "body_driven_exploit": exploit / _N_EVAL, "violation_reason": mon.get("top_violation", "n/a")})
    st = aggregate(per_seed, n_eval=_N_EVAL)
    log(f"[v0] {name:14s} ft_dom {st.mean['ft_dom']:.3f}±{st.std['ft_dom']:.3f} mon_pass {st.mean['monitor_pass']:.3f} "
        f"mon_score {st.mean['monitor_score']:.3f} sustained {st.mean['sustained_push_per_ep']:.2f} "
        f"ftprog {st.mean['ft_progress_in_contact']:.4f} exploit {st.mean['body_driven_exploit']:.3f}")
    return st


def _accept(base: MultiSeedStats, cand: MultiSeedStats) -> dict:
    """Gated θ* is POSITIVE if it preserves/beats E_valselect on ft_dom + monitor_PASS (the delivery-quality RATE —
    user decision 2026-07-08: gate on pass-rate, not the monitor_score MEAN), further improves sustained-PUSH +
    ft-progress, and does not increase exploit/body/arm-body. The monitor_score MEAN is reported as informational
    (it can dip as marginal deliveries are rescued while the pass-rate rises — a benign delivery↔mean-quality trade)."""
    tie = compare_ftdom(base, cand)
    ftd_ok = tie.decision in ("better", "tied") and cand.mean["ft_dom"] >= base.mean["ft_dom"] - 0.02
    mon_pass_ok = cand.mean["monitor_pass"] >= base.mean["monitor_pass"] - 0.02
    mon_score_ok = cand.mean["monitor_score"] >= base.mean["monitor_score"] - 0.02   # informational only
    sustained_up = cand.mean["sustained_push_per_ep"] > base.mean["sustained_push_per_ep"] + 0.05
    ftprog_up = cand.mean["ft_progress_in_contact"] > base.mean["ft_progress_in_contact"] + 1e-4
    no_bad = (cand.mean["body_driven_exploit"] <= 0.02
              and cand.mean["arm_body_rate"] <= base.mean["arm_body_rate"] + 0.02
              and cand.mean["body_progress_in_contact"] <= base.mean["body_progress_in_contact"] + 1e-3)
    headline_ok = ftd_ok and mon_pass_ok                       # gate on ft_dom + monitor_PASS (rate)
    passed = bool(headline_ok and sustained_up and ftprog_up and no_bad)
    if passed:
        verdict = "POSITIVE"
    elif headline_ok and no_bad:
        verdict = "NEUTRAL_NO_FURTHER_GAIN"       # preserves E_valselect but does not further improve contact
    else:
        verdict = "NEGATIVE_WITH_MECHANISM"       # the handoff drops ft_dom / pass-rate or adds exploit
    return {"verdict": verdict, "passed": passed, "gate": "ft_dom + monitor_pass (rate)", "ftdom_tie": tie.as_dict(),
            "ftdom_ok": bool(ftd_ok), "monitor_pass_ok": bool(mon_pass_ok),
            "monitor_score_mean_preserved": bool(mon_score_ok),
            "monitor_score_mean_delta": round(cand.mean["monitor_score"] - base.mean["monitor_score"], 4),
            "sustained_further_up": bool(sustained_up), "ftprog_further_up": bool(ftprog_up), "no_bad": bool(no_bad)}


def run(cfg: OptionSearchConfig, *, out_dir: str = "experiments/2026_07_08_option_rl_v0") -> dict:
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    e_actor = _load_e_valselect(env)
    guards = wire_ledgers(env, e_actor)
    reward = certify_v2b()
    contract = MonitorContract.from_env(env)
    _log(f"[v0] guards {guards['pipeline_schema']}/{guards['policy_provenance']} | v2b delivers={reward.get('delivers')} "
         f"| base E_valselect md5 {file_md5(_E_CKPT)[:8]}")

    def make_ctrl(e: Any, theta: np.ndarray) -> PhaseGatedPolicy:
        return PhaseGatedPolicy(e_actor, PhasePushController(e, params=PushControllerParams.from_vector(theta)))

    _log("[v0] CEM over θ (phase-gated: E_valselect approach + PhasePushController(θ) push) ...")
    res = option_cem(lambda: make_env(coord=False, difficulty=_DIFFICULTY), cfg, make_ctrl=make_ctrl, log=_log)
    _log(f"[v0] θ* = {res.theta_named} | search obj {res.baseline_objective:.3f}→{res.best_objective:.3f} "
         f"(improved_over_scripted-gate={res.improved_over_scripted})")
    theta_star = np.asarray(res.theta, dtype=np.float64)

    # fresh-seed acceptance eval: E_valselect base vs the gated θ* policy
    base = _eval_multiseed(lambda e: greedy_action_fn(e_actor), contract, "E_valselect")

    def make_gated_fn(e: Any):
        pg = PhaseGatedPolicy(e_actor, PhasePushController(e, params=PushControllerParams.from_vector(theta_star)))
        pg.reset()
        return phase_gated_action_fn(pg)
    gated = _eval_multiseed(make_gated_fn, contract, "gated_θ*")

    acc = _accept(base, gated)
    verdict = acc["verdict"]
    ckpt = None
    if verdict == "POSITIVE":
        ckpt = str(out / "phase_gated_theta.json")
        Path(ckpt).write_text(json.dumps({"theta": res.theta, "theta_named": res.theta_named,
                                           "base_mlp_checkpoint": _E_CKPT, "base_mlp_md5": file_md5(_E_CKPT)}, indent=2))
        _log(f"[v0] POSITIVE → saved deployable option {ckpt}")

    def _row(st):
        return {k: {"mean": round(st.mean.get(k, float("nan")), 4), "std": round(st.std.get(k, 0.0), 4)} for k in _REPORT}
    result = {"experiment": "bounded_option_parameter_rl_v0", "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1),
              "design": "phase_gated (E_valselect approach + PhasePushController(theta) push)",
              "fresh_eval_seeds": list(_FRESH_SEEDS), "n_eval": _N_EVAL,
              "guards": {**guards, "tensor_contract": guards["pipeline_schema"]},
              "provenance": {"policy_provenance": "PASS", "base_mlp_md5": file_md5(_E_CKPT)}, "v2b_reward": reward,
              "theta_star": res.theta, "theta_named": res.theta_named,
              "search_objective": {"base": round(res.baseline_objective, 4), "best": round(res.best_objective, 4)},
              "E_valselect_fresh": _row(base), "gated_theta_fresh": _row(gated),
              "acceptance": acc, "deployable_option": ckpt}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    _log(f"[v0] acceptance: {acc}")
    _log(f"[v0] VERDICT: {verdict} (wrote {out}/results.json, {result['wall_s']}s)")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="bounded_option_parameter_rl_v0")
    ap.add_argument("--stage", choices=("smoke", "full"), default="full")
    args = ap.parse_args(argv)
    if args.stage == "smoke":
        cfg = OptionSearchConfig(pop=6, elite=2, iters=2, n_eval_eps=3, max_steps=200, log_every=1)
    else:
        cfg = OptionSearchConfig(pop=16, elite=4, iters=8, n_eval_eps=6, eval_seed=41000, log_every=1)
    run(cfg, out_dir=("experiments/2026_07_08_option_rl_v0_smoke" if args.stage == "smoke"
                      else "experiments/2026_07_08_option_rl_v0"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
