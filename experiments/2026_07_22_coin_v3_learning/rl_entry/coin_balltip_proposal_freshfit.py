"""BALLTIP true-deploy re-fit — re-fit the ball proposal on the FRESH-RECONSTRUCT distribution (O1 finding).

O1 exposed that the ball proposal (fit on TRANSPLANT handoffs in B3) scores ≈0 on the true deploy distribution (pi_0
reconstructed DIRECTLY on the ball), while the expert still solves 0.5–0.69. This re-fits the proposal on fresh-reconstruct
handoffs to get an HONEST true-deploy baseline, and compares it — on the SAME fresh eval states — against the old
transplant-fit proposal. Reuses generate_bank (reconstruct_kwargs = per-object reconstruction), fit_proposal, and the O1
ball reconstruction (no duplication). Canonical coin (r0.020); frozen settling pi_0 + option language unchanged.
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402
from coin_object_variants import _ball_tf, variant_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-balltip-freshfit"
OLD_PROP = f"{D}/carry_proposal_balltip_v1.pt"          # transplant-fit (B3/B5)
FRESH_PROP = f"{D}/carry_proposal_balltip_fresh_v1.pt"  # the true-deploy re-fit
FAMS, EVAL_H, K = ("contact_retention", "transport", "braking"), 160, 6


def eval_prop_fresh(prop, panel, pi0, base):
    """b=0 (direct) and b=8 (search) K6 of a proposal on fresh-reconstruct ball states (each rollout a fresh gate copy)."""
    b0, b8 = 0, 0
    for i, (rl, gate) in enumerate(panel):
        c = prop.theta(rl.obs())
        o0 = structured_carry_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, np.asarray(c, np.float32), horizon=EVAL_H)
        _t, o8 = search_select(rl, gate, c, pi0, base, np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
        b0 += int(o0["k6"])
        b8 += int(o8["k6"])
    n = max(1, len(panel))
    return {"n": len(panel), "b0": b0, "b0_rate": round(b0 / n, 3), "b8": b8, "b8_rate": round(b8 / n, 3)}


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
    shots = 24 if smoke else 160
    rk = {"geom": "POINT", "arm_mjcf_transform": _ball_tf}                 # per-object reconstruction on the canonical ball

    # ---- 1. fresh-reconstruct teacher bank (canonical ball; pi_0 replayed on the ball, not transplant) ----
    log(f"[fresh-fit] teacher bank on fresh-reconstruct ball handoffs ({shots}-shot expert)...")
    train_ls, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(20 if smoke else 200),
                                            families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    obs, theta, prov = generate_bank(pi0, base, train_ls, shots=shots, reconstruct_kwargs=rk, log=log)
    log(f"[fresh-fit] CONFIDENT fresh-reconstruct labels: {len(obs)}")
    if len(obs) < 4:
        log("[fresh-fit] too few labels — abort (widen pool/budget)")
        return {"error": "insufficient_labels", "n": len(obs)}
    kk = min(K, max(2, len(obs) // 4))
    prop_fresh, fit = fit_proposal(np.asarray(obs, np.float32), np.asarray(theta, np.float32), kk,
                                   clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
    save_proposal(prop_fresh, FRESH_PROP)
    log(f"[fresh-fit] proposal K={kk} res_mse {fit['res_mse']:.3f} → {FRESH_PROP}")

    # ---- 2. eval on fresh-reconstruct eval states (14000–15600): new fresh proposal vs old transplant proposal ----
    ev_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 40),
                                         families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    panel = variant_panel(pi0, ev_ls, 0.020, (8 if smoke else 24), log)
    new = eval_prop_fresh(prop_fresh, panel, pi0, base)
    old = eval_prop_fresh(load_proposal(OLD_PROP), panel, pi0, base)

    res = {"contract": "BALLTIP_FRESH_REFIT", "date": "2026-07-24", "smoke": smoke, "n_labels": len(obs), "K": kk,
           "fit": fit, "fresh_proposal_ckpt": FRESH_PROP.split("/")[-1], "eval_distribution": "fresh_reconstruct_canonical_ball",
           "fresh_proposal": new, "old_transplant_proposal_on_fresh_eval": old}
    json.dump(res, open(f"{OUT}/freshfit.json", "w"), indent=1, default=float)

    log("\n== BALLTIP true-deploy re-fit (fresh-reconstruct distribution) ==")
    log(f"  fresh eval states: {new['n']}")
    log(f"  OLD transplant-fit proposal on fresh eval:  b0 {old['b0']}/{old['n']} ({old['b0_rate']})  b8 {old['b8']}/{old['n']} ({old['b8_rate']})")
    log(f"  NEW fresh-fit proposal on fresh eval:        b0 {new['b0']}/{new['n']} ({new['b0_rate']})  b8 {new['b8']}/{new['n']} ({new['b8_rate']})")
    verdict = ("FRESH_REFIT_RECOVERS_DEPLOY_BASELINE" if new["b8"] > old["b8"] else "FRESH_REFIT_NO_BETTER_THAN_TRANSPLANT")
    log(f"→ {verdict}\n  artifact: {OUT}/freshfit.json\nBALLTIP_FRESHFIT_DONE")
    return res


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
