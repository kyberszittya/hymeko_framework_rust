"""BALLTIP_COLLISION_ON_V1 — Stage B5: option-level STOCHASTIC-GAUSSIAN SAC on the collision-on ball (not distributional RL).

B3-iteration showed a deterministic proposal absorbs the ball's FRAGILE, multimodal option distribution only slowly
(b=0 0→3, b=8 5/24, ceiling 16/24). A STOCHASTIC actor over θ (with the fixed b=8 search kept in the loop) is the matched
architecture. Same safe pipeline as the Stage-5 clamp SAC — state → ball proposal-init actor → fixed b=8 search →
committed option → frozen clamp pi_0 settling → K6 — reused verbatim from `coin_carry_option_rl` (SearchWrapperEnv /
distill_actor / train_agent / eval_paired) with BALL templates (transplant) and the BALL update-0 proposal.

MANDATORY gate (CLAUDE.md): the TRAINING reward is oracle-certified (`certify_reward`) BEFORE any RL — HARD_STOP if the
option reward does not rank delivery above non-delivery. The claim is PAIRED: SAC vs its OWN ball update-0 (never the
clamp), with a paired bootstrap CI. SAC primary + TD3 control, 2 seeds (multi-seed median per the RL discipline).
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import BALL_PROP, D, _ball_transplant, _bank  # noqa: E402
from coin_carry_option_diagnostic import _panel  # noqa: E402
from coin_carry_option_rl_stage5 import certify_reward, option_return_distribution  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option_rl import (  # noqa: E402
    DetActor, GaussActor, OptionReward, RLConfig, SearchWrapperEnv, distill_actor, eval_paired, train_agent)
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-balltip-b5-sac"
EVAL_H, B_DEPLOY = 160, 8
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}


def _ballify(templ):
    """Transplant each clamp (rl, gate) option-initiation state onto the collision-on ball at the matched state."""
    return [(_ball_transplant(rl), gate) for rl, gate in templ]


def _bootstrap_delta(rl_k6, up_k6, *, iters=10000, seed=0):
    """Paired bootstrap CI of ΔK6 = mean(rl) − mean(up) over the per-state paired lists."""
    rng = np.random.default_rng(seed)
    d = np.asarray(rl_k6) - np.asarray(up_k6)
    boot = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(iters)]
    return round(float(np.mean(d)), 3), (round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3))


def main(smoke=False):
    import os

    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    reward = OptionReward()
    prop = load_proposal(BALL_PROP)                                        # the ball update-0 (B3-iteration best)
    log(f"[B5] ball SAC | RL-init = {BALL_PROP.split('/')[-1]} (ball update-0) | robot = collision-on ball r0.020")

    n_tr = 20 if smoke else 60
    tr = _ballify(_panel(pi0, range(9000, 10800), forbidden, n_tr)[0])
    dev = _ballify(_panel(pi0, range(11000, 12000), forbidden, 8 if smoke else 20)[0])
    fin = _ballify(_panel(pi0, range(14000, 15200), forbidden, 8 if smoke else 24)[0])   # the ball EVAL panel
    log(f"[panels] train {len(tr)} | dev {len(dev)} | final {len(fin)} (ball-transplanted; disjoint seed ranges)")

    # ---- MANDATORY reward certification (HARD_STOP if the option reward is not delivery-aligned on the ball) ----
    cert = certify_reward(tr, pi0, base, reward, log)
    if not cert["delivers"]:
        json.dump({"contract": "BALLTIP_B5_SAC", "verdict": "HARD_STOP_REWARD_NOT_CERTIFIED", "reward_cert": cert},
                  open(f"{OUT}/b5_sac.json", "w"), indent=1, default=float)
        log("→ HARD_STOP: the option reward does not rank ball delivery above non-delivery. No RL launched.\nBALLTIP_B5_DONE")
        return {"verdict": "HARD_STOP_REWARD_NOT_CERTIFIED", "reward_cert": cert}
    log(f"[reward-cert] delivers={cert['delivers']} (R_K6 {cert['R_k6_mean']} > R_nonK6 {cert['R_nonk6_mean']})")

    # ---- ball update-0 baseline (b=0 / b=8) on the final panel ----
    upd0 = {}
    for b in (0, B_DEPLOY):
        k6 = [int(search_select(rl, gate, prop.theta(rl.obs()), pi0, base, np.random.default_rng(9000 + i), b=b, horizon=EVAL_H)[1]["k6"])
              for i, (rl, gate) in enumerate(fin)]
        upd0[f"b{b}"] = round(float(np.mean(k6)), 3)
    log(f"[update-0 ball baseline final] b0 {upd0['b0']} | b8 {upd0['b8']}")

    rc = RLConfig(b=B_DEPLOY, horizon=EVAL_H, warmup_options=(10 if smoke else 40),
                  total_options=(40 if smoke else 600), eval_every=(20 if smoke else 100))
    seeds = [0] if smoke else [0, 1]
    algos = ["sac"] if smoke else ["sac", "td3"]
    branches, dist = {}, None
    for algo in algos:
        for sd in seeds:
            actor = GaussActor() if algo == "sac" else DetActor()
            dloss = distill_actor(actor, prop, np.stack([t[0].obs() for t in tr]), epochs=(120 if smoke else 400), seed=sd)
            env = SearchWrapperEnv(tr, pi0, base, reward, gamma=rc.gamma, b=rc.b, horizon=rc.horizon, max_options=rc.max_options, seed=sd)
            if algo == "sac" and sd == seeds[0]:
                dist = option_return_distribution(env, actor, log)
                env = SearchWrapperEnv(tr, pi0, base, reward, gamma=rc.gamma, b=rc.b, horizon=rc.horizon, max_options=rc.max_options, seed=sd)
            log(f"[{algo} seed {sd}] distill MSE {round(dloss,4)} → train {rc.total_options} options (fixed b={rc.b}, γ^τ)")
            ckpts, hist = train_agent(algo, env, actor, dev, pi0, base, rc, log, seed=sd)
            # PAIRED claim vs the ball update-0 on the final panel (best_val checkpoint)
            a2 = GaussActor() if algo == "sac" else DetActor()
            a2.load_state_dict(ckpts["best_val"])
            rl_k6, up_k6, rl_ex = eval_paired(a2, prop, fin, pi0, base, b=B_DEPLOY, search_seeds=(1 if smoke else 3), horizon=EVAL_H)
            delta, ci = _bootstrap_delta(rl_k6, up_k6)
            torch.save(ckpts["best_val"], f"{D}/carry_rl_balltip_{algo}_seed{sd}_bestval.pt")
            branches[f"{algo}_seed{sd}"] = {"distill_mse": round(dloss, 4), "history": hist,
                                            "rl_b8_K6": round(float(np.mean(rl_k6)), 3), "up_b8_K6": round(float(np.mean(up_k6)), 3),
                                            "delta_K6": delta, "delta_ci95": ci, "rl_exit": round(float(np.mean(rl_ex)), 3),
                                            "rl_k6_per_state": [round(float(x), 3) for x in rl_k6],       # per-state paired bits
                                            "up_k6_per_state": [round(float(x), 3) for x in up_k6]}       # (authoritative eval = coin_balltip_b5_eval.py)
            log(f"[{algo} seed {sd}] PAIRED final: RL b8 {branches[f'{algo}_seed{sd}']['rl_b8_K6']} vs update-0 b8 {branches[f'{algo}_seed{sd}']['up_b8_K6']} | ΔK6 {delta} CI95 {ci}")

    # SEED-AWARE aggregate (no best-seed selection bias): per-seed ΔK6 → median/IQR. 2 seeds ⇒ PILOT cap at most.
    def _agg(prefix):
        seeds = {k: v for k, v in branches.items() if k.startswith(prefix)}
        if not seeds:
            return None
        d = [v["delta_K6"] for v in seeds.values()]
        return {"per_seed_delta": d, "median_delta": round(float(np.median(d)), 3),
                "iqr": [round(float(np.percentile(d, 25)), 3), round(float(np.percentile(d, 75)), 3)],
                "both_seeds_positive": all(x > 0 for x in d), "both_seed_cis_above0": all(v["delta_ci95"][0] > 0 for v in seeds.values())}
    sac_agg = _agg("sac")
    if smoke:
        verdict = "BALLTIP_B5_SMOKE_CONTRACTS_OK"
    elif sac_agg and sac_agg["both_seeds_positive"] and sac_agg["both_seed_cis_above0"]:
        verdict = "BALLTIP_SAC_PILOT_POSITIVE_LEAN"                        # strongest admissible at 2 seeds (NOT established)
    elif sac_agg and sac_agg["median_delta"] > 0:
        verdict = "BALLTIP_SAC_POSITIVE_LEAN_UNDERPOWERED_CI_SPANS_0"
    else:
        verdict = "BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0"
    out = {"contract": "BALLTIP_B5_SAC", "date": "2026-07-24", "smoke": smoke, "baseline": BASELINE, "reward_cert": cert,
           "rl_init": BALL_PROP.split("/")[-1], "update0_baseline_final_DIAGNOSTIC_single_search_seed": upd0, "config": rc.__dict__,
           "branches": branches, "sac_aggregate": sac_agg, "td3_aggregate": _agg("td3"),
           "authoritative_claim": "coin_balltip_b5_eval.py (seed-aware, search-seed-paired, per-(seed,state,search-seed) bits)",
           "option_return_distribution": dist, "verdict": verdict}
    json.dump(out, open(f"{OUT}/b5_sac.json", "w"), indent=1, default=float)

    log("\n== BALLTIP_COLLISION_ON_V1 — Stage B5: option-level SAC ==")
    log(f"  reward certified: {cert['delivers']} | update-0 b8 {upd0['b8']}")
    for k, v in branches.items():
        log(f"  {k}: RL b8 {v['rl_b8_K6']} vs update-0 {v['up_b8_K6']} | ΔK6 {v['delta_K6']} CI95 {v['delta_ci95']}")
    log(f"→ {verdict}\n  artifacts: {OUT}/b5_sac.json\nBALLTIP_B5_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
