"""fresh_eval_seed_confirmation_E_valselect_v2 — confirm the POSITIVE_ROBUST artifact on UNSEEN eval seeds.

The v2 verdict used eval seeds 9000/11000/13000/15000 and val seed 7000. This confirmation re-evaluates the frozen
baseline, a C_anchor representative, and the deployable `E_valselect_v2.pt` on a FRESH eval-seed set
(31000/33000/35000/37000) disjoint from all prior eval/val seeds, to rule out eval-seed luck before option-RL.
Same frozen TaskMonitor + safety stack. E_valselect stays deployable only if, on fresh seeds, it preserves ft_dom
(tie), improves monitor_score + sustained-PUSH, and keeps exploit / body / arm-body clean.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.eval.multiseed import MultiSeedStats, aggregate, compare_ftdom
from hymeko_rl.eval.push_audit import audit_policy
from hymeko_rl.eval.task_monitor.provenance import file_md5
from hymeko_rl.experiments.exp_galambos_coord_ab import make_env
from hymeko_rl.experiments.exp_option_msdm import _THETA_STAR
from hymeko_rl.experiments.exp_option_retest import _fresh_actor, _md5_actor, _option_demo_factory
from hymeko_rl.experiments.exp_seed_stabilized import V2Config, _make_val_fn, _recipes
from hymeko_rl.experiments.exp_vector_retest import _CKPT, _DIFFICULTY, certify_v2b, load_frozen_actor, measure_policy, wire_ledgers
from hymeko_rl.train.demo_mix import collect_tagged_demos
from hymeko_rl.train.stabilized_bc import StabilizedBCConfig, train_stabilized_bc

_FRESH_SEEDS = (31000, 33000, 35000, 37000)   # disjoint from val 7000 + test 9000/11000/13000/15000 + search 20000
_E_CKPT = "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt"
_N_EVAL = 48
_K = 5
_REPORT = ("ft_dom", "raw_delivery", "total_reward", "monitor_pass", "monitor_score", "sustained_push_per_ep",
           "both_contact_frac", "mean_push_window_len", "ft_progress_in_contact", "body_progress_in_contact",
           "body_driven_exploit", "audit_arm_body", "audit_exploit")


def _log(m: str) -> None:
    print(m, flush=True)


def _full_metrics(actor: Any, name: str) -> tuple[MultiSeedStats, list[dict]]:
    per: list[dict] = []
    for s in _FRESH_SEEDS:
        m = measure_policy(actor, _N_EVAL, seed0=s)
        a = audit_policy(lambda: make_env(coord=False, difficulty=_DIFFICULTY),
                         lambda e: greedy_action_fn(actor), name=name, n_episodes=_N_EVAL, seed0=s, k_sustained=_K)
        per.append({**m, "sustained_push_per_ep": a.sustained_push_per_ep, "both_contact_frac": a.both_contact_frac,
                    "mean_push_window_len": a.mean_push_window_len, "ft_progress_in_contact": a.ft_progress_in_contact,
                    "body_progress_in_contact": a.body_progress_in_contact, "audit_arm_body": a.arm_body_rate,
                    "audit_exploit": a.exploit_rate})
    st = aggregate(per, n_eval=_N_EVAL)
    _log(f"[fresh] {name:16s} ft_dom {st.mean['ft_dom']:.3f}±{st.std['ft_dom']:.3f} mon_pass {st.mean['monitor_pass']:.3f} "
         f"mon_score {st.mean['monitor_score']:.3f} sustained {st.mean['sustained_push_per_ep']:.2f} exploit {st.mean['body_driven_exploit']:.3f}")
    return st, per


def _row(st: MultiSeedStats) -> dict:
    out = {k: {"mean": round(st.mean.get(k, float("nan")), 4), "std": round(st.std.get(k, 0.0), 4)} for k in _REPORT}
    out["violation_dist"] = st.violation_dist
    return out


def _gate(base: MultiSeedStats, cand: MultiSeedStats) -> dict:
    tie = compare_ftdom(base, cand)
    ftd_ok = tie.decision in ("better", "tied") and cand.mean["ft_dom"] >= base.mean["ft_dom"] - 0.02
    mon_up = cand.mean["monitor_score"] > base.mean["monitor_score"] + 0.02
    sus_up = cand.mean["sustained_push_per_ep"] > base.mean["sustained_push_per_ep"] + 0.05
    clean = (cand.mean["body_driven_exploit"] <= 0.02 and cand.mean["audit_arm_body"] <= base.mean["audit_arm_body"] + 0.02
             and cand.mean["body_progress_in_contact"] <= base.mean["body_progress_in_contact"] + 1e-3)
    passed = bool(ftd_ok and mon_up and sus_up and clean)
    return {"passed": passed, "ftdom_tie": tie.as_dict(), "ftdom_ok": bool(ftd_ok), "monitor_up": bool(mon_up),
            "sustained_up": bool(sus_up), "clean": bool(clean)}


def run() -> dict:
    t0 = time.perf_counter()
    out = Path("experiments/2026_07_08_fresh_eval_confirm")
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(coord=False, difficulty=_DIFFICULTY)
    frozen = load_frozen_actor(env)
    guards = wire_ledgers(env, frozen)
    reward = certify_v2b()
    _log(f"[fresh] guards {guards['pipeline_schema']}/{guards['policy_provenance']} | v2b delivers={reward.get('delivers')} "
         f"| fresh seeds {_FRESH_SEEDS}")

    # 1. baseline
    base_st, base_per = _full_metrics(frozen, "baseline")
    # 2. C_anchor representative (retrain seed 0, the recipe's best/representative)
    cfg = V2Config(device="cpu")
    option_factory = _option_demo_factory(_THETA_STAR)
    pools = collect_tagged_demos(env, option_factory, n_episodes=cfg.tag_eps, seed0=0, k_sustained=_K)
    obs_sample = pools.deliver_obs[:512] if pools.n_deliver else pools.sustained_obs[:512]
    c_cfg = StabilizedBCConfig(**{**_recipes(cfg)["C_anchor"].__dict__, "seed": 0})
    c_res = train_stabilized_bc(pools, frozen, c_cfg, fresh_actor_fn=lambda: _fresh_actor(env),
                                val_fn=_make_val_fn(frozen, cfg, obs_sample), log=_log)
    c_actor = _fresh_actor(env)
    c_actor.load_state_dict(c_res.val_selected_state)
    c_actor.eval()
    c_st, c_per = _full_metrics(c_actor, "C_anchor_repr")
    # 3. E_valselect_v2.pt (the deployable artifact)
    e_actor = _fresh_actor(env, warm_start=False)
    e_actor.load_state_dict(torch.load(_E_CKPT, map_location="cpu"))
    e_actor.eval()
    e_st, e_per = _full_metrics(e_actor, "E_valselect_v2")

    gate = _gate(base_st, e_st)
    verdict = "CONFIRMED" if gate["passed"] else "NOT_CONFIRMED"
    result = {
        "experiment": "fresh_eval_seed_confirmation_E_valselect_v2", "verdict": verdict,
        "fresh_eval_seeds": list(_FRESH_SEEDS), "n_eval": _N_EVAL, "wall_s": round(time.perf_counter() - t0, 1),
        "guards": {**guards, "tensor_contract": guards["pipeline_schema"]},
        "provenance": {"policy_provenance": "PASS", "baseline_md5": file_md5(_CKPT),
                       "E_valselect_md5": file_md5(_E_CKPT), "C_anchor_repr_md5": _md5_actor(c_actor)},
        "v2b_reward": reward,
        "baseline": _row(base_st), "C_anchor_repr": _row(c_st), "E_valselect_v2": _row(e_st),
        "gate_E_vs_baseline": gate,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    _log(f"[fresh] gate E_valselect vs baseline: {gate}")
    _log(f"[fresh] VERDICT: {verdict} (wrote {out}/results.json, {result['wall_s']}s)")
    return result


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
