"""STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — matched obs55 transactional-TD3 campaign (Arm A / Arm B).
Usage: python -m ...coin_markov_ablation_v1 [--smoke] [--seeds N]"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    eval_ablation,
    make_late_actor55_from_pi0,
    train_arm,
    verify_update0,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
CFG = f"{D}/td3_baseline_v1_config.json"
IMPROVE_MARGIN = 0.05          # preregistered "material" K6-rate improvement over pi_0


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1); log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(CFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    stage = dict(cfg["stage1"]); families = tuple(stage["families"])
    if args.smoke:
        stage.update(total_updates=200, critic_warmup_steps=50, collect_every=100, episodes_per_collect=8, checkpoints=[0, 100, 200])
    seeds = [0] if args.smoke else list(range(args.seeds))

    # update-0 manifest (both arms share the same zero-init contract)
    obs48 = [__import__("hymeko_rl.coin_delivery.coin_late_start", fromlist=["reconstruct_handoff"]).reconstruct_handoff(pi0, ls, horizon=360)[3].obs for ls in tb[:16]]
    u0 = verify_update0(pi0, make_late_actor55_from_pi0(pi0, trainable=False), obs48)
    log(f"[update-0 manifest] actor55(obs,strict k)==pi0: max_diff {u0['max_action_diff']:.2e} mean_diff {u0['mean_action_diff']:.2e}")
    assert u0["max_action_diff"] < 1e-5, "update-0 != pi_0"

    # pi_0 baselines (CONTINUATION + RESET) on the dev bank
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    pi0_cont = eval_ablation(pi0, base, db, "A", horizon=stage["horizon"], families=families, reset_strict=False)
    pi0_reset = eval_ablation(pi0, base, db, "A", horizon=stage["horizon"], families=families, reset_strict=True)
    log(f"[pi_0 baseline] CONTINUATION K6 {pi0_cont['k6_rate']} ({pi0_cont['n']} states) | RESET K6 {pi0_reset['k6_rate']}")

    t0 = time.time(); runs = {}
    for arm in ("A", "B"):
        for seed in seeds:
            log(f"\n=== ARM {arm} seed {seed} ({'MARKOV_ONLY' if arm=='A' else 'MARKOV_PLUS_TERMINAL_ALIGNMENT'}) ===")
            runs[f"{arm}_{seed}"] = train_arm(pi0, arm, stage, tb, db, seed=seed, log=log)
    wall = round(time.time() - t0, 1)

    def final_k6(arm):
        vals = []
        for seed in seeds:
            ck = runs[f"{arm}_{seed}"]["checkpoints"]; last = ck[max(ck, key=lambda k: int(k))]
            vals.append(last["CONTINUATION_STRICT"]["k6_rate"])
        return vals
    a_k6, b_k6 = final_k6("A"), final_k6("B")
    a_impr = np.mean(a_k6) - pi0_cont["k6_rate"]; b_impr = np.mean(b_k6) - pi0_cont["k6_rate"]
    a_win = a_impr >= IMPROVE_MARGIN and min(a_k6) > pi0_cont["k6_rate"]           # reproducible (all seeds above)
    b_win = b_impr >= IMPROVE_MARGIN and min(b_k6) > pi0_cont["k6_rate"]
    healthy = all(runs[k]["critic_ever_authorized"] for k in runs)
    if a_win and b_win and b_impr > a_impr + IMPROVE_MARGIN:
        verdict = "BOTH_REPAIRS_LOAD_BEARING"
    elif a_win:
        verdict = "MARKOV_REPAIR_LOAD_BEARING"
    elif b_win:
        verdict = "TERMINAL_ALIGNMENT_LOAD_BEARING"
    elif healthy:
        verdict = "REPAIRS_NOT_SUFFICIENT"
    else:
        verdict = "REPAIRS_INCONCLUSIVE_UNHEALTHY_CRITIC"

    reclass = {"PHASE_SWITCHED_TD3 / TRANSACTIONAL_TD3 / TRANSPORT_DWELL_TD3 / PHASE_GATED_RESIDUAL_CRITIC":
               ("OVERTURNED — a Markov critic improves over pi_0" if (a_win or b_win) else
                "SURVIVE_MARKOV_REPAIR — no improvement even with the counter observed, healthy critics, finite grads")}
    out = {"campaign": "STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1", "date": "2026-07-23", "smoke": args.smoke, "wall_s": wall,
           "update0_manifest": u0, "pi0_baseline": {"CONTINUATION": pi0_cont, "RESET": pi0_reset},
           "stage_config": {k: stage[k] for k in ("families", "horizon", "n_step", "total_updates", "checkpoints", "policy_delay")},
           "final_k6_continuation": {"A": a_k6, "B": b_k6}, "improvement_vs_pi0": {"A": round(float(a_impr), 4), "B": round(float(b_impr), 4)},
           "critic_healthy_all": healthy, "runs": runs, "reclassification": reclass, "verdict": verdict}
    outp = f"{D}/markov_ablation_v1{'_smoke' if args.smoke else ''}.json"
    json.dump(out, open(outp, "w"), indent=1, default=float)
    log(f"\n== ABLATION ==  pi0_K6 {pi0_cont['k6_rate']}  |  Arm A K6 {a_k6} (Δ {a_impr:+.3f})  |  Arm B K6 {b_k6} (Δ {b_impr:+.3f})  healthy {healthy}")
    log(f"→ {verdict}\nwrote {outp}\nMARKOV_ABLATION_DONE")


if __name__ == "__main__":
    main()
