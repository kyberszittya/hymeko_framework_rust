"""BALLTIP_COLLISION_ON_V1 — Stage B5 CORRECTED paired evaluation (seed-aware, no selection bias).

Methodology fix (per review): the B5 training script's aggregate picked the BEST seed then read its CI — selection bias
that inflates false positives from two noisy seeds. This re-runs the paired eval FROM THE SAVED CHECKPOINTS (training is
kept, not restarted) and reports the honest seed-aware claim:
  * per-training-seed ΔK6 = mean over states of (RL − its OWN ball update-0), search-seed-paired (SAME search rng for RL
    and update-0 at each (state, search-seed)) — so the two are one estimate, never conflated with the one-shot diagnostic;
  * per-seed bootstrap CI over states;
  * across-seed MEDIAN + IQR and a hierarchical (seed→state) bootstrap;
  * the full per-(seed, state, search-seed) RL and update-0 bits saved to JSON (not just the mean).
With 2 seeds the strongest admissible status is BALLTIP_SAC_PILOT_POSITIVE_LEAN; STATISTICALLY_ESTABLISHED needs more seeds.
The actor is a STOCHASTIC GAUSSIAN SAC policy (not distributional RL in the usual sense).
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_b5_sac import _ballify  # noqa: E402
from coin_balltip_proposal import BALL_PROP, D, _bank  # noqa: E402
from coin_carry_option_diagnostic import _panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option_rl import DetActor, GaussActor, _actor_center  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-balltip-b5-sac"
EVAL_H, B_DEPLOY, SEED0 = 160, 8, 8000
SEARCH_SEEDS = (0, 1, 2)                                   # 3 fixed search seeds per state (paired RL vs update-0)


def eval_matrix(actor, prop, fin, pi0, base):
    """Per-(state, search-seed) RL and update-0 K6, PAIRED: at each (state, search-seed) RL and update-0 use the SAME
    search rng — the only difference is the θ-center (actor mean vs proposal). Returns two [n_state, n_search] int arrays."""
    rl_mat, up_mat = [], []
    for i, (rl, gate) in enumerate(fin):
        c_rl = _actor_center(actor, rl.obs())
        c_up = prop.theta(rl.obs())
        rl_row, up_row = [], []
        for j in SEARCH_SEEDS:
            sd = SEED0 + i * 131 + j
            o_rl = search_select(rl, gate, c_rl, pi0, base, np.random.default_rng(sd), b=B_DEPLOY, horizon=EVAL_H)[1]
            o_up = search_select(rl, gate, c_up, pi0, base, np.random.default_rng(sd), b=B_DEPLOY, horizon=EVAL_H)[1]
            rl_row.append(int(o_rl["k6"]))
            up_row.append(int(o_up["k6"]))
        rl_mat.append(rl_row)
        up_mat.append(up_row)
    return np.asarray(rl_mat, int), np.asarray(up_mat, int)


def _boot_ci(delta_per_state, *, iters=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta_per_state)
    b = [float(np.mean(delta_per_state[rng.integers(0, n, n)])) for _ in range(iters)]
    return round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)


def _hier_boot(per_seed_state_deltas, *, iters=10000, seed=0):
    """Hierarchical bootstrap: resample TRAINING seeds (with replacement) then states within each — the honest 2-level
    uncertainty. Coarse at 2 seeds (flagged), but it does not pretend seed variance away."""
    rng = np.random.default_rng(seed)
    S = len(per_seed_state_deltas)
    means = []
    for _ in range(iters):
        chosen = rng.integers(0, S, S)
        vals = []
        for s in chosen:
            d = per_seed_state_deltas[s]
            vals.append(float(np.mean(d[rng.integers(0, len(d), len(d))])))
        means.append(float(np.mean(vals)))
    return round(float(np.mean(means)), 3), (round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3))


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    prop = load_proposal(BALL_PROP)
    fin = _ballify(_panel(pi0, range(14000, 15200), forbidden, 8 if smoke else 24)[0])
    log(f"[B5-eval] {len(fin)} ball eval states | {len(SEARCH_SEEDS)} search seeds | ckpt update-0 {BALL_PROP.split('/')[-1]}")

    ckpts = sorted(glob.glob(f"{D}/carry_rl_balltip_*_bestval.pt"))
    if not ckpts:
        log("no checkpoints found — is B5 training done?")
        return

    # feedback check (a): the eval path (search_select deep-copies the gate internally) is ORDER-INVARIANT — running the
    # same paired matrix twice must be bit-identical (no gate-contamination across states/seeds like the regression bug).
    a0 = GaussActor() if ckpts[0].split("carry_rl_balltip_")[1].startswith("sac") else DetActor()
    a0.load_state_dict(torch.load(ckpts[0], weights_only=False))
    m1r, m1u = eval_matrix(a0, prop, fin[:2], pi0, base)
    m2r, m2u = eval_matrix(a0, prop, fin[:2], pi0, base)
    assert np.array_equal(m1r, m2r) and np.array_equal(m1u, m2u), "eval path is NOT order-invariant (gate contamination?)"
    log("[invariance] eval path order-invariant ✓ (paired matrix bit-identical on re-run)")

    branches = {}
    for path in ckpts:
        name = path.split("carry_rl_balltip_")[1].replace("_bestval.pt", "")   # e.g. sac_seed0
        actor = GaussActor() if name.startswith("sac") else DetActor()
        actor.load_state_dict(torch.load(path, weights_only=False))
        rl_mat, up_mat = eval_matrix(actor, prop, fin, pi0, base)
        per_state_delta = rl_mat.mean(1) - up_mat.mean(1)                  # avg over search seeds → per-state paired delta
        delta = round(float(per_state_delta.mean()), 3)
        ci = _boot_ci(per_state_delta)
        branches[name] = {"rl_b8": round(float(rl_mat.mean()), 3), "up_b8": round(float(up_mat.mean()), 3),
                          "delta_K6": delta, "delta_ci95_states": ci, "per_state_delta": [round(float(x), 3) for x in per_state_delta],
                          "rl_bits": rl_mat.tolist(), "up_bits": up_mat.tolist()}
        log(f"  {name:12} RL b8 {branches[name]['rl_b8']} vs update-0 {branches[name]['up_b8']} | ΔK6 {delta} CI95(states) {ci}")

    def agg(prefix):
        seeds = {k: v for k, v in branches.items() if k.startswith(prefix)}
        if not seeds:
            return None
        deltas = [v["delta_K6"] for v in seeds.values()]
        per_seed_state = [np.asarray(v["per_state_delta"], float) for v in seeds.values()]
        hb_mean, hb_ci = _hier_boot(per_seed_state)
        both_pos = all(d > 0 for d in deltas)
        both_ci_pos = all(v["delta_ci95_states"][0] > 0 for v in seeds.values())
        return {"n_seeds": len(seeds), "per_seed_delta": deltas, "median_delta": round(float(np.median(deltas)), 3),
                "iqr": [round(float(np.percentile(deltas, 25)), 3), round(float(np.percentile(deltas, 75)), 3)],
                "hier_boot_mean": hb_mean, "hier_boot_ci95": hb_ci, "both_seeds_positive": both_pos, "both_seed_cis_above0": both_ci_pos}

    sac_agg, td3_agg = agg("sac"), agg("td3")
    # verdict — capped at PILOT for 2 seeds; requires BOTH seeds positive AND the hierarchical CI clearing 0
    if sac_agg is None:
        verdict = "BALLTIP_SAC_NO_CHECKPOINTS"
    elif sac_agg["both_seeds_positive"] and sac_agg["both_seed_cis_above0"] and sac_agg["hier_boot_ci95"][0] > 0:
        verdict = "BALLTIP_SAC_PILOT_POSITIVE_LEAN"                        # strongest admissible at 2 seeds
    elif sac_agg["median_delta"] > 0:
        verdict = "BALLTIP_SAC_POSITIVE_LEAN_UNDERPOWERED_CI_SPANS_0"
    else:
        verdict = "BALLTIP_SAC_NO_IMPROVEMENT_OVER_UPDATE0"

    out = {"contract": "BALLTIP_B5_SAC_EVAL", "date": "2026-07-24", "method": "seed-aware paired (search-seed-paired); "
           "per-(seed,state,search-seed) bits; median/IQR + hierarchical bootstrap; 2 seeds ⇒ PILOT cap; stochastic Gaussian SAC",
           "n_eval": len(fin), "search_seeds": list(SEARCH_SEEDS), "update0_ckpt": BALL_PROP.split("/")[-1],
           "branches": branches, "sac_aggregate": sac_agg, "td3_aggregate": td3_agg, "verdict": verdict}
    json.dump(out, open(f"{OUT}/b5_sac_eval.json", "w"), indent=1, default=float)

    log("\n== BALLTIP B5 — corrected seed-aware paired eval ==")
    if sac_agg:
        log(f"  SAC per-seed ΔK6 {sac_agg['per_seed_delta']} | median {sac_agg['median_delta']} IQR {sac_agg['iqr']}")
        log(f"  SAC hierarchical bootstrap ΔK6 {sac_agg['hier_boot_mean']} CI95 {sac_agg['hier_boot_ci95']} "
            f"(2 seeds ⇒ coarse) | both+ {sac_agg['both_seeds_positive']} both CI>0 {sac_agg['both_seed_cis_above0']}")
    if td3_agg:
        log(f"  TD3 per-seed ΔK6 {td3_agg['per_seed_delta']} | median {td3_agg['median_delta']}")
    log(f"→ {verdict}\n  artifact: {OUT}/b5_sac_eval.json\nBALLTIP_B5_EVAL_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
