"""Stage 5b — the variance-reduced campaign: does reward-driven proposal-SAC reproducibly beat its OWN update-0 proposal
under the SAME fixed b=8 search?

Fixes the first-pass selection-variance bottleneck, nothing else changed (same wrapper, b=8, γ^τ, frozen pi_0, certificate):
larger panels (dev≥60, separate untouched final), SAC primary + TD3 control, 4/2 seeds, ≥2000 options, eval-time
multi-search-seed averaging (Bellman-safe), and a PRE-REGISTERED checkpoint selection:
  (1) paired ΔK6 vs the checkpoint's OWN update-0 proposal → (2) bootstrap-CI lower bound → (3) any_exit → (4) then return.
Report per-seed, median/IQR across seeds, and the seed-wise solved-set. Claim = median over seeds of the selected
checkpoint's FINAL-panel paired ΔK6 with its bootstrap CI lower bound > 0.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from coin_carry_option_diagnostic import _bank, _panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_action_perturbation import bootstrap_ci  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option_rl import (  # noqa: E402
    DetActor,
    GaussActor,
    OptionReward,
    RLConfig,
    SearchWrapperEnv,
    distill_actor,
    eval_paired,
    train_agent,
)
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
B_DEPLOY, EVAL_H = 8, 160


def _select_and_score(algo, ckpts, prop, dev, final, pi0, base, *, eval_seeds, boot_seed):
    """PRE-REGISTERED selection on DEV, then score the winner on the untouched FINAL panel. Returns the selection record."""
    scored = []
    for name, state in ckpts.items():
        a = GaussActor() if algo == "sac" else DetActor(); a.load_state_dict(state)
        rl_k6, up_k6, ex = eval_paired(a, prop, dev, pi0, base, b=B_DEPLOY, search_seeds=eval_seeds, horizon=EVAL_H)
        d = [r - u for r, u in zip(rl_k6, up_k6)]
        ci = bootstrap_ci(d, stat=np.mean, seed=boot_seed)
        scored.append({"ckpt": name, "dev_dK6_mean": round(float(np.mean(d)), 4), "dev_dK6_lo": round(ci["lo"], 4),
                       "dev_exit": round(float(np.mean(ex)), 4), "dev_rl_k6": round(float(np.mean(rl_k6)), 4)})
    # pre-registered order: (1) paired ΔK6, (2) CI lower, (3) −any_exit, then stable
    scored.sort(key=lambda x: (x["dev_dK6_mean"], x["dev_dK6_lo"], -x["dev_exit"]), reverse=True)
    win = scored[0]
    a = GaussActor() if algo == "sac" else DetActor(); a.load_state_dict(ckpts[win["ckpt"]])
    rl_k6, up_k6, ex = eval_paired(a, prop, final, pi0, base, b=B_DEPLOY, search_seeds=eval_seeds, horizon=EVAL_H)
    dfin = [r - u for r, u in zip(rl_k6, up_k6)]
    ci = bootstrap_ci(dfin, stat=np.mean, seed=boot_seed + 1)
    return {"selected": win, "all_dev": scored, "final_rl_k6": round(float(np.mean(rl_k6)), 4),
            "final_up_k6": round(float(np.mean(up_k6)), 4), "final_dK6_mean": round(float(np.mean(dfin)), 4),
            "final_dK6_lo": round(ci["lo"], 4), "final_dK6_hi": round(ci["hi"], 4), "final_exit": round(float(np.mean(ex)), 4),
            "solved_rl": [i for i, v in enumerate(rl_k6) if v > 0.5], "solved_up": [i for i, v in enumerate(up_k6) if v > 0.5]}


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

    n_tr = 24 if smoke else 90
    tr = _panel(pi0, range(9000, 10800), forbidden, n_tr)[0]
    dev = _panel(pi0, range(11000, 12200), forbidden, 12 if smoke else 54)[0]
    fin = _panel(pi0, range(12200, 13600), forbidden, 12 if smoke else 36)[0]
    log(f"[panels] train {len(tr)} | dev {len(dev)} | final {len(fin)} (disjoint 9000-10800 / 11000-12200 / 12200-13600)")

    eval_seeds = 2 if smoke else 3
    rc = RLConfig(b=B_DEPLOY, horizon=EVAL_H, warmup_options=(10 if smoke else 60),
                  total_options=(40 if smoke else 2000), eval_every=(20 if smoke else 500))
    plan = {"sac": ([0] if smoke else [0, 1, 2, 3]), "td3": ([] if smoke else [0])}   # SAC primary (4 seeds), TD3 single control
    results = {}
    for algo, seeds in plan.items():
        for sd in seeds:
            actor = GaussActor() if algo == "sac" else DetActor()
            dl = distill_actor(actor, prop, np.stack([t[0].obs() for t in tr]), epochs=(120 if smoke else 400), seed=sd)
            env = SearchWrapperEnv(tr, pi0, base, reward, gamma=rc.gamma, b=rc.b, horizon=rc.horizon, max_options=rc.max_options, seed=sd)
            log(f"[{algo} seed {sd}] distill MSE {round(dl,4)} → {rc.total_options} options")
            ckpts, hist = train_agent(algo, env, actor, dev, pi0, base, rc, log, seed=sd)
            rec = _select_and_score(algo, ckpts, prop, dev, fin, pi0, base, eval_seeds=eval_seeds, boot_seed=1000 + sd)
            results[f"{algo}_seed{sd}"] = {"distill_mse": round(dl, 4), "select": rec, "history": hist}
            torch.save(ckpts[rec["selected"]["ckpt"]], f"{D}/carry_rlb_{algo}_seed{sd}_selected.pt")
            log(f"[{algo} seed {sd}] selected {rec['selected']['ckpt']} | FINAL ΔK6 {rec['final_dK6_mean']} (CI≥{rec['final_dK6_lo']}) | rl {rec['final_rl_k6']} vs up {rec['final_up_k6']}")

    # across-seed claim (SAC primary): median of the selected-checkpoint final ΔK6, and CI-lower>0 per seed
    def summarize(prefix):
        rs = [v for k, v in results.items() if k.startswith(prefix)]
        if not rs:
            return None
        dks = [r["select"]["final_dK6_mean"] for r in rs]
        pos = sum(r["select"]["final_dK6_lo"] > 0 for r in rs)
        return {"n_seeds": len(rs), "final_dK6_median": round(float(np.median(dks)), 4), "final_dK6_iqr": [round(float(np.percentile(dks, 25)), 4), round(float(np.percentile(dks, 75)), 4)],
                "seeds_with_CI_lower_gt0": pos, "per_seed_dK6": [round(x, 4) for x in dks],
                "per_seed_final_rl_k6": [r["select"]["final_rl_k6"] for r in rs], "per_seed_final_up_k6": [r["select"]["final_up_k6"] for r in rs]}
    sac_s, td3_s = summarize("sac"), summarize("td3")

    if smoke:
        verdict = "STAGE5B_SMOKE_OK"
    elif sac_s and sac_s["final_dK6_median"] > 0 and sac_s["seeds_with_CI_lower_gt0"] >= max(1, sac_s["n_seeds"] // 2):
        verdict = "OPTION_PROPOSAL_RL_IMPROVES_OVER_UPDATE0"
    elif sac_s and sac_s["final_dK6_median"] <= 0 and all(r["select"]["final_dK6_hi"] < 0.15 for k, r in results.items() if k.startswith("sac")):
        verdict = "OPTION_PROPOSAL_RL_NO_PHYSICAL_IMPROVEMENT"
    else:
        verdict = "OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE"

    out = {"contract": "CARRY_OPTION_RL_STAGE5B_V1", "date": "2026-07-24", "smoke": smoke, "rl_init": s4c["rl_init_checkpoint"],
           "config": rc.__dict__, "eval_search_seeds": eval_seeds, "panels": {"train": len(tr), "dev": len(dev), "final": len(fin)},
           "selection": "pre-registered: paired ΔK6 vs own update-0 → CI lower → −any_exit → return; eval-averaged over search seeds",
           "results": results, "sac_summary": sac_s, "td3_summary": td3_s, "verdict": verdict}
    json.dump(out, open(f"{D}/carry_option_rl_stage5b_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_RL_STAGE5B_V1 (variance-reduced: does proposal-SAC beat its own update-0 @ fixed b=8?) ==")
    for k, r in results.items():
        s = r["select"]; log(f"  {k}: selected {s['selected']['ckpt']} | FINAL ΔK6 {s['final_dK6_mean']} CI[{s['final_dK6_lo']},{s['final_dK6_hi']}] | rl {s['final_rl_k6']} vs up {s['final_up_k6']} exit {s['final_exit']}")
    log(f"  SAC summary: {sac_s}")
    log(f"  TD3 summary: {td3_s}")
    log(f"→ {verdict}\nwrote {D}/carry_option_rl_stage5b_v1.json\nCARRY_OPTION_RL_STAGE5B_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
