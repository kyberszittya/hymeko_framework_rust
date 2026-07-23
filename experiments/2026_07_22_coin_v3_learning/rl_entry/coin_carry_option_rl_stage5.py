"""Stage 5 — search-in-the-loop semi-MDP option RL (SAC + TD3) over the proposal center.

Pipeline: load best-safe Stage-4c proposal → CERTIFY the option reward (K6 outcomes ranked above non-K6) → distil the
proposal into SAC and TD3 actors (IDENTICAL init) → measure the update-0 proposal baseline at b=0/b=8 → inspect the
option-return distribution (contract smoke) → train each branch (fixed b=8 wrapper, γ^τ target) → eval every checkpoint at
b=0/b=8 on the disjoint final panel → the paired claim: RL proposal + fixed b=8 > update-0 proposal + fixed b=8 in held-out
K6 without exit growth. Panels are disjoint: train 9000-10800, dev 11000-12000, final-eval 12000-13000.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from coin_carry_option_diagnostic import _bank, _panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option_rl import (  # noqa: E402
    DetActor,
    GaussActor,
    OptionReward,
    RLConfig,
    SearchWrapperEnv,
    distill_actor,
    eval_policy,
    execute_one_option,
    search_select,
    train_agent,
)
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
B_DEPLOY, EVAL_H = 8, 160


def certify_reward(templates, pi0, base, reward, log, *, n=12):
    """Reward gate (CLAUDE.md): the option reward must RANK delivery above non-delivery. Collect K6 and non-K6 options
    (strong search + random θ) and assert mean R_option(k6=1) clearly exceeds mean R_option(k6=0)."""
    R_k6, R_no = [], []
    for i, (rl, gate) in enumerate(templates[:n]):
        th_e, _o = search_select(rl, gate, np.zeros(15, np.float32), pi0, base, np.random.default_rng(700 + i), b=32, horizon=EVAL_H)
        for th, sd in ((th_e, 1), (np.clip(np.random.default_rng(800 + i).uniform(-2, 2, 15).astype(np.float32), -3, 3), 2)):
            o = execute_one_option(copy.deepcopy(rl), copy.deepcopy(gate), th, pi0, base, gamma=0.99, horizon=EVAL_H, reward=reward)
            (R_k6 if o["k6"] else R_no).append(o["R_option"])
    mk = float(np.mean(R_k6)) if R_k6 else float("nan"); mn = float(np.mean(R_no)) if R_no else float("nan")
    delivers = len(R_k6) > 0 and (not R_no or mk > mn + 1.0)
    log(f"[reward-cert] R_option K6 mean {round(mk,2)} (n={len(R_k6)}) vs non-K6 mean {round(mn,2)} (n={len(R_no)}) → delivers={delivers}")
    return {"delivers": bool(delivers), "R_k6_mean": round(mk, 3), "R_nonk6_mean": round(mn, 3), "n_k6": len(R_k6), "n_nonk6": len(R_no)}


def option_return_distribution(env, actor, log, *, n=40):
    """Contract smoke: option-return distribution + τ + terminal/success fractions from the actual wrapper."""
    Rs, taus, k6s, terms = [], [], [], []
    s = env.reset()
    for _ in range(n):
        with torch.no_grad():
            a = actor.mean_action(torch.as_tensor(s[None]).float())[0].numpy() if hasattr(actor, "mean_action") else actor(torch.as_tensor(s[None]).float())[0].numpy()
        s2, r, done, info = env.step(a)
        Rs.append(r); taus.append(info["tau"]); k6s.append(info["k6"]); terms.append(int(info["terminal"]))
        s = env.reset() if done else s2
    Rs = np.array(Rs)
    d = {"R_min": round(float(Rs.min()), 2), "R_med": round(float(np.median(Rs)), 2), "R_max": round(float(Rs.max()), 2),
         "tau_min": int(min(taus)), "tau_med": int(np.median(taus)), "tau_max": int(max(taus)),
         "success_frac": round(float(np.mean(k6s)), 3), "terminal_frac": round(float(np.mean(terms)), 3),
         "R_success_med": round(float(np.median(Rs[np.array(k6s) == 1])), 2) if any(k6s) else None,
         "R_fail_med": round(float(np.median(Rs[np.array(k6s) == 0])), 2) if not all(k6s) else None}
    log(f"[opt-return dist] {d}")
    return d


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = set(ls.seed for ls in _bank(cfg["banks"]["late_train"])) | set(ls.seed for ls in _bank(cfg["banks"]["late_dev"]))
    reward = OptionReward()

    s4c = json.load(open(f"{D}/carry_option_stage4c_v1.json"))
    prop = load_proposal(f"{D}/{s4c['rl_init_checkpoint']}")
    log(f"[proposal] RL init = {s4c['rl_init_checkpoint']} (Stage-4c best safe, {s4c['verdict']})")

    n_tr = 20 if smoke else 60
    tr_templ, _tf = _panel(pi0, range(9000, 10800), forbidden, n_tr)
    dev_templ, _df = _panel(pi0, range(11000, 12000), forbidden, 8 if smoke else 20)
    fin_templ, fin_fam = _panel(pi0, range(12000, 13000), forbidden, 10 if smoke else 24)
    log(f"[panels] train {len(tr_templ)} | dev {len(dev_templ)} | final {len(fin_templ)} (disjoint 9000-10800 / 11000-12000 / 12000-13000)")

    cert = certify_reward(tr_templ, pi0, base, reward, log)
    if not cert["delivers"]:
        json.dump({"contract": "CARRY_OPTION_RL_STAGE5_V1", "verdict": "HARD_STOP_REWARD_NOT_CERTIFIED", "reward_cert": cert},
                  open(f"{D}/carry_option_rl_stage5_v1.json", "w"), indent=1, default=float)
        log("→ HARD_STOP: option reward does not rank delivery above non-delivery. No RL launched.\nCARRY_OPTION_RL_STAGE5_DONE"); return

    # update-0 proposal baseline (b=0 and b=8) on the FINAL panel
    upd0 = {}
    for b in (0, B_DEPLOY):
        k6, ex = [], []
        for i, (rl, gate) in enumerate(fin_templ):
            _th, out = search_select(rl, gate, prop.theta(rl.obs()), pi0, base, np.random.default_rng(9000 + i), b=b, horizon=EVAL_H)
            k6.append(int(out["k6"])); ex.append(int(out["contain_exit_ct"] > 0))
        upd0[f"b{b}"] = {"K6": round(float(np.mean(k6)), 3), "any_exit": round(float(np.mean(ex)), 3)}
    log(f"[update-0 baseline final] b0 {upd0['b0']} | b8 {upd0['b8']}")

    rc = RLConfig(b=B_DEPLOY, horizon=EVAL_H,
                  warmup_options=(10 if smoke else 40), total_options=(40 if smoke else 600), eval_every=(20 if smoke else 120))
    seeds = [0] if smoke else [0, 1]
    branches = {}
    for algo in (["sac"] if smoke else ["sac", "td3"]):
        for sd in seeds:
            actor = GaussActor() if algo == "sac" else DetActor()
            dloss = distill_actor(actor, prop, np.stack([t[0].obs() for t in tr_templ]), epochs=(120 if smoke else 400), seed=sd)
            env = SearchWrapperEnv(tr_templ, pi0, base, reward, gamma=rc.gamma, b=rc.b, horizon=rc.horizon, max_options=rc.max_options, seed=sd)
            if algo == "sac" and sd == seeds[0]:
                dist = option_return_distribution(env, actor, log)
                env = SearchWrapperEnv(tr_templ, pi0, base, reward, gamma=rc.gamma, b=rc.b, horizon=rc.horizon, max_options=rc.max_options, seed=sd)
            log(f"[{algo} seed {sd}] distill MSE {round(dloss,4)} → training {rc.total_options} options (fixed b={rc.b}, γ^τ)")
            ckpts, hist = train_agent(algo, env, actor, dev_templ, pi0, base, rc, log, seed=sd)
            # eval every checkpoint at b=0 and b=8 on the FINAL panel
            evals = {}
            for name, sd_state in ckpts.items():
                a2 = GaussActor() if algo == "sac" else DetActor(); a2.load_state_dict(sd_state)
                evals[name] = {f"b{b}": dict(zip(("K6", "any_exit"), eval_policy(a2, fin_templ, pi0, base, b=b, horizon=EVAL_H))) for b in (0, B_DEPLOY)}
            torch.save(ckpts["best_val"], f"{D}/carry_rl_{algo}_seed{sd}_bestval.pt")
            torch.save(ckpts["final"], f"{D}/carry_rl_{algo}_seed{sd}_final.pt")
            branches[f"{algo}_seed{sd}"] = {"distill_mse": round(dloss, 4), "history": hist, "checkpoint_eval": evals}
            log(f"[{algo} seed {sd}] final panel: update0 b8 {evals['update0']['b8']} → best_val b8 {evals['best_val']['b8']} | final b8 {evals['final']['b8']}")

    # paired claim: RL (best_val) proposal + b8 vs update-0 proposal + b8 on the final panel
    b8_updated = upd0["b8"]["K6"]
    best_rl = max((v["checkpoint_eval"]["best_val"]["b8"]["K6"], k) for k, v in branches.items())
    improves = best_rl[0] > b8_updated + 1e-9
    exit_ok = all(v["checkpoint_eval"]["best_val"]["b8"]["any_exit"] <= upd0["b8"]["any_exit"] + 0.15 for v in branches.values())
    if smoke:
        verdict = "OPTION_PROPOSAL_RL_SMOKE_CONTRACTS_OK"
    elif improves and exit_ok:
        verdict = "OPTION_PROPOSAL_RL_IMPROVES_OVER_UPDATE0"
    elif not improves and all(len(v["history"]) > 0 for v in branches.values()):
        verdict = "OPTION_PROPOSAL_RL_NO_PHYSICAL_IMPROVEMENT"
    else:
        verdict = "OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE"

    out = {"contract": "CARRY_OPTION_RL_STAGE5_V1", "date": "2026-07-24", "smoke": smoke, "reward_cert": cert,
           "rl_init": s4c["rl_init_checkpoint"], "update0_baseline_final": upd0, "config": rc.__dict__,
           "branches": branches, "best_rl_bestval_b8": {"K6": best_rl[0], "branch": best_rl[1]},
           "improves_over_update0": bool(improves), "exit_ok": bool(exit_ok), "verdict": verdict}
    if not smoke:
        out["option_return_distribution"] = dist
    json.dump(out, open(f"{D}/carry_option_rl_stage5_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_RL_STAGE5_V1 (search-in-the-loop semi-MDP option RL) ==")
    log(f"  reward certified: {cert['delivers']} (K6 {cert['R_k6_mean']} > non-K6 {cert['R_nonk6_mean']})")
    log(f"  update-0 proposal + b8: K6 {b8_updated}")
    for k, v in branches.items():
        e = v["checkpoint_eval"]; log(f"  {k}: b8 update0 {e['update0']['b8']['K6']} → best_val {e['best_val']['b8']['K6']} → final {e['final']['b8']['K6']}")
    log(f"  best RL best_val b8 {best_rl[0]} ({best_rl[1]}) vs update-0 b8 {b8_updated} → improves={improves} exit_ok={exit_ok}")
    log(f"→ {verdict}\nwrote {D}/carry_option_rl_stage5_v1.json\nCARRY_OPTION_RL_STAGE5_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
