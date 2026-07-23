"""CARRY_OPTION_ACTOR_V1 — amortized-search test (the lever the diagnostic points to).

The diagnostic showed: per-state search delivers (0.533) but no one-shot fixed-θ distillation transfers (≤0.10). The
proposed architecture is NOT search-free distillation but AMORTIZED search:

    state → proposal (template classifier / BC regressor) → small budgeted structured search AROUND the proposal
          → committed option execution → frozen settling pi_0.

Discriminating question: at a fixed small search budget b, does a learned-proposal-centred search recover more of the
0.533 expert than a budget-matched UNIFORM search (no proposal) or a search centred on a RANDOM bank θ? If proposal-centred
> random-centred > uniform, and the K6-vs-b curve rises steeply, the proposal makes search cheap and targeted — validating
the amortized-search + (later) search-in-the-loop-RL path. If proposal-centred ≈ uniform, the proposal does not localise.
Reuses the proposal machinery from the committed diagnostic (no duplication) and the library committed executor.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from coin_carry_option_diagnostic import _bank, _kmeans, _norm, _panel, _train_clf  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option import actor_theta, make_option_actor, option_controller_rollout, train_option_bc  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout, structured_random_around, structured_random_best  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
STD_AMP, STD_DUR, EVAL_H = 0.6, 2.0, 160


def _exec(templ_i, theta, pi0, base):
    return option_controller_rollout(copy.deepcopy(templ_i[0]), copy.deepcopy(templ_i[1]), np.asarray(theta, np.float32), pi0, base, horizon=EVAL_H)


def _around(templ_i, center, pi0, base, rng, b):
    """Budgeted search AROUND center (b shots); b==0 → execute the center directly. Returns the executed outcome dict."""
    if b <= 0:
        return structured_carry_rollout(copy.deepcopy(templ_i[0]), copy.deepcopy(templ_i[1]), pi0, base, np.asarray(center, np.float32), horizon=EVAL_H)
    _th, out = structured_random_around(copy.deepcopy(templ_i[0]), copy.deepcopy(templ_i[1]), pi0, base, rng, shots=b, center=np.asarray(center, np.float32), std_amp=STD_AMP, std_dur=STD_DUR, horizon=EVAL_H)
    return out


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = set(ls.seed for ls in _bank(cfg["banks"]["late_train"])) | set(ls.seed for ls in _bank(cfg["banks"]["late_dev"]))
    K = 4 if smoke else 8
    budgets = [0, 4, 8] if smoke else [0, 4, 8, 16]

    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    bank_obs, bank_th = z["obs"].astype(np.float32), z["theta"].astype(np.float32)
    log(f"[bank] {len(bank_obs)} confident θ labels")

    # proposals (rebuilt reproducibly): TEMPLATE classifier + BC regressor
    lab, medoid = _kmeans(_norm(bank_th), K, seed=0); templates = bank_th[medoid]
    clf = _train_clf(bank_obs, lab, K, epochs=(60 if smoke else 300), seed=0)
    bc = make_option_actor(); train_option_bc(bc, bank_obs, bank_th, epochs=(80 if smoke else 400), lr=1e-3, batch=32, seed=0)

    ev_templ, ev_fam = _panel(pi0, range(11000, 13000), forbidden, 10 if smoke else 30)
    log(f"[panel] disjoint eval {len(ev_templ)} states, K={K}, budgets={budgets}, std_amp={STD_AMP}")

    with torch.no_grad():
        ev_ids = clf(torch.as_tensor(np.stack([ev_templ[i][0].obs() for i in range(len(ev_templ))]))).argmax(1).numpy()
    rng_pick = np.random.default_rng(13); rand_centers = rng_pick.integers(0, len(bank_th), len(ev_templ))
    templ_center = lambda i: templates[ev_ids[i]]
    bc_center = lambda i: actor_theta(bc, ev_templ[i][0].obs())
    rand_center = lambda i: bank_th[rand_centers[i]]

    def curve(center_fn, tag):
        row = {}
        for b in budgets:
            outs = [_around(ev_templ[i], center_fn(i), pi0, base, np.random.default_rng(4000 + b * 100 + i), b) for i in range(len(ev_templ))]
            row[b] = {"K6": round(float(np.mean([o["k6"] for o in outs])), 3), "any_exit": round(float(np.mean([o["contain_exit_ct"] > 0 for o in outs])), 3)}
            log(f"  [{tag:16} b={b:2}] K6 {row[b]['K6']} any_exit {row[b]['any_exit']}")
        return row

    def uniform_curve():
        row = {}
        for b in budgets:
            if b == 0:
                row[b] = {"K6": 0.0, "any_exit": 0.0}; continue           # no proposal, no search → nothing to execute
            outs = [structured_random_best(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.random.default_rng(5000 + b * 100 + i), shots=b, horizon=EVAL_H)[1] for i in range(len(ev_templ))]
            row[b] = {"K6": round(float(np.mean([o["k6"] for o in outs])), 3), "any_exit": round(float(np.mean([o["contain_exit_ct"] > 0 for o in outs])), 3)}
            log(f"  [{'uniform':16} b={b:2}] K6 {row[b]['K6']} any_exit {row[b]['any_exit']}")
        return row

    log("[amortized] TEMPLATE-centred search"); amo_t = curve(templ_center, "amo-template")
    log("[amortized] BC-centred search"); amo_bc = curve(bc_center, "amo-bc")
    log("[amortized] RANDOM-bank-centred search (control)"); amo_r = curve(rand_center, "amo-random")
    log("[control] UNIFORM search (no proposal, budget-matched)"); uni = uniform_curve()

    pi0_out = [sequence_then_pi0(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H) for i in range(len(ev_templ))]
    exp_out = [structured_random_best(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.random.default_rng(900 + i), shots=(16 if smoke else 64), horizon=EVAL_H)[1] for i in range(len(ev_templ))]
    pi0_k6 = round(float(np.mean([o["k6"] for o in pi0_out])), 3); exp_k6 = round(float(np.mean([o["k6"] for o in exp_out])), 3)

    # does proposal-centred beat uniform + random-centred at matched budget?
    b_ref = budgets[1] if len(budgets) > 1 else budgets[0]
    localizes = amo_t[b_ref]["K6"] > uni[b_ref]["K6"] + 1e-9 and amo_t[b_ref]["K6"] >= amo_r[b_ref]["K6"]
    best_amo = max(max(r[b]["K6"] for b in budgets) for r in (amo_t, amo_bc))
    frac = best_amo / exp_k6 if exp_k6 > 0 else 0.0
    if exp_k6 <= pi0_k6 + 0.05:
        verdict = "EXPERT_ADVANTAGE_NOT_REPRODUCED_ON_THIS_PANEL"
    elif best_amo >= 0.8 * exp_k6 and localizes:
        verdict = "AMORTIZED_SEARCH_RECOVERS_EXPERT_PROPOSAL_LOCALIZES"
    elif localizes and best_amo > uni[b_ref]["K6"] + 1e-9:
        verdict = "PROPOSAL_LOCALIZES_SEARCH_PARTIAL_RECOVERY"
    else:
        verdict = "PROPOSAL_DOES_NOT_LOCALIZE_SEARCH_IS_THE_LEVER"

    out = {"contract": "CARRY_OPTION_AMORTIZED_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "eval_n": len(ev_templ), "K_templates": K, "budgets": budgets, "std_amp": STD_AMP, "std_dur": STD_DUR,
           "pi_0_K6": pi0_k6, "expert_K6": exp_k6, "b_ref": b_ref, "proposal_localizes": bool(localizes),
           "best_amortized_K6": best_amo, "best_amortized_frac_of_expert": round(frac, 3),
           "curves": {"amortized_template": amo_t, "amortized_bc": amo_bc, "amortized_random_center": amo_r, "uniform": uni},
           "verdict": verdict}
    json.dump(out, open(f"{D}/carry_option_amortized_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_AMORTIZED_V1 (K6 vs search budget b; recover the 0.533 expert cheaply?) ==")
    log(f"  pi_0 {pi0_k6} | expert(64) {exp_k6} | b_ref={b_ref}")
    hdr = "  budget:      " + "  ".join(f"b={b:<2}" for b in budgets)
    log(hdr)
    for tag, r in (("amo-template", amo_t), ("amo-bc", amo_bc), ("amo-random", amo_r), ("uniform", uni)):
        log(f"  {tag:12} " + "  ".join(f"{r[b]['K6']:<4}" for b in budgets))
    log(f"→ proposal_localizes={localizes} | best_amortized {best_amo} ({round(frac,2)}× expert) → {verdict}")
    log(f"wrote {D}/carry_option_amortized_v1.json\nCARRY_OPTION_AMORTIZED_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
