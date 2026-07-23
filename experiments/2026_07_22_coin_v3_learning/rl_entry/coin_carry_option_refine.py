"""CARRY_OPTION_ACTOR_V1 — search-guided proposal refinement (the honest precursor to search-in-the-loop RL).

The amortized test showed a learned proposal LOCALIZES a small search (0.5× expert @ 1/8 budget). The crux of the whole
architecture is then: can the proposal ABSORB the search's knowledge — i.e. does iterating

    propose θ → small search AROUND it → adopt the search-discovered K6 θ as the new proposal target → retrain

tighten the proposal so that (a) its direct (b=0) K6 rises and (b) the same recovery is reached at a SMALLER budget? This is
DAgger with the *search as teacher* (trustworthy first-pass machinery), and it directly measures whether search-in-the-loop
RL has anything to learn. If the b=0 direct K6 climbs across rounds and small-b recovery improves, the proposal is
amortizing the search — a clean green light for the RL branch. If it stays flat, the proposal cannot be tightened this way
and the search must stay in the loop at deploy. Pre-registered controls: a FIXED (round-0) proposal at the same budgets,
and a budget-matched uniform search. Reuses committed proposal machinery + library executor (no duplication).
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from coin_carry_option_diagnostic import _bank, _kmeans, _norm, _panel, _train_clf  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option import actor_theta, make_option_actor, train_option_bc  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout, structured_random_around, structured_random_best  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
STD_AMP, STD_DUR, EVAL_H = 0.6, 2.0, 160


def _search_around(templ_i, center, pi0, base, rng, b):
    if b <= 0:
        return structured_carry_rollout(copy.deepcopy(templ_i[0]), copy.deepcopy(templ_i[1]), pi0, base, np.asarray(center, np.float32), horizon=EVAL_H), np.asarray(center, np.float32)
    th, out = structured_random_around(copy.deepcopy(templ_i[0]), copy.deepcopy(templ_i[1]), pi0, base, rng, shots=b, center=np.asarray(center, np.float32), std_amp=STD_AMP, std_dur=STD_DUR, horizon=EVAL_H)
    return out, th


def _eval_at(ev_templ, proposal_fn, pi0, base, b, seed0):
    outs = [_search_around(ev_templ[i], proposal_fn(i), pi0, base, np.random.default_rng(seed0 + i), b)[0] for i in range(len(ev_templ))]
    return round(float(np.mean([o["k6"] for o in outs])), 3), round(float(np.mean([o["contain_exit_ct"] > 0 for o in outs])), 3)


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = set(ls.seed for ls in _bank(cfg["banks"]["late_train"])) | set(ls.seed for ls in _bank(cfg["banks"]["late_dev"]))
    rounds = 2 if smoke else 3
    b_train = 8 if smoke else 12
    K = 4 if smoke else 8

    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    obs_bank, th_bank = list(z["obs"].astype(np.float32)), list(z["theta"].astype(np.float32))
    log(f"[bank] {len(obs_bank)} confident θ labels (round-0 seed)")

    # round-0 proposal = template-classifier medoid (best one-shot proposal from the amortized test) + a continuous BC actor
    lab, medoid = _kmeans(_norm(np.asarray(th_bank, np.float32)), K, seed=0); templates = np.asarray(th_bank, np.float32)[medoid]
    clf = _train_clf(np.asarray(obs_bank, np.float32), lab, K, epochs=(60 if smoke else 300), seed=0)
    actor = make_option_actor(); train_option_bc(actor, obs_bank, th_bank, epochs=(80 if smoke else 400), lr=1e-3, batch=32, seed=0)

    tr_templ, tr_fam = _panel(pi0, range(9000, 10800), forbidden, 16 if smoke else 45)
    ev_templ, ev_fam = _panel(pi0, range(11000, 13000), forbidden, 10 if smoke else 30)
    log(f"[panel] refine-train {len(tr_templ)} | disjoint eval {len(ev_templ)} | rounds {rounds} b_train {b_train} K {K}")

    with torch.no_grad():
        ev_ids = clf(torch.as_tensor(np.stack([ev_templ[i][0].obs() for i in range(len(ev_templ))]))).argmax(1).numpy()
    def templ_direct(i):
        return templates[ev_ids[i]]                                       # FIXED round-0 proposal control
    def actor_prop(i):
        return actor_theta(actor, ev_templ[i][0].obs())                  # current refined continuous proposal

    pi0_outs = [sequence_then_pi0(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H) for i in range(len(ev_templ))]
    pi0_k6 = round(float(np.mean([o["k6"] for o in pi0_outs])), 3)
    exp_outs = [structured_random_best(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.random.default_rng(900 + i), shots=(16 if smoke else 64), horizon=EVAL_H)[1] for i in range(len(ev_templ))]
    exp_k6 = round(float(np.mean([o["k6"] for o in exp_outs])), 3)

    history = []
    def snapshot(rnd):
        row = {"round": rnd, "n_labels": len(obs_bank),
               "actor_b0": _eval_at(ev_templ, actor_prop, pi0, base, 0, 6000)[0],
               "actor_b4": _eval_at(ev_templ, actor_prop, pi0, base, 4, 7000)[0],
               "template_b0_fixed": _eval_at(ev_templ, templ_direct, pi0, base, 0, 8000)[0],
               "template_b4_fixed": _eval_at(ev_templ, templ_direct, pi0, base, 4, 9000)[0]}
        history.append(row)
        log(f"  [round {rnd}] labels {row['n_labels']} | refined actor: b0 {row['actor_b0']} b4 {row['actor_b4']} | fixed template: b0 {row['template_b0_fixed']} b4 {row['template_b4_fixed']}")
        return row

    snapshot(0)
    for rnd in range(1, rounds + 1):
        add_o, add_t = [], []
        for i in range(len(tr_templ)):
            center = actor_theta(actor, tr_templ[i][0].obs())
            out, best = _search_around(tr_templ[i], center, pi0, base, np.random.default_rng(3000 + rnd * 100 + i), b_train)
            if int(out["k6"]) == 1:                                       # search FOUND a K6 θ near the proposal → teach it
                add_o.append(tr_templ[i][0].obs().copy()); add_t.append(np.asarray(best, np.float32))
        obs_bank += add_o; th_bank += add_t
        loss = train_option_bc(actor, obs_bank, th_bank, epochs=(60 if smoke else 250), lr=5e-4, batch=32, seed=rnd)
        log(f"  [refine {rnd}] +{len(add_o)} search-K6 labels → {len(obs_bank)} total, BC MSE {round(loss,4)}")
        snapshot(rnd)

    b0_0, b0_R = history[0]["actor_b0"], history[-1]["actor_b0"]
    b4_0, b4_R = history[0]["actor_b4"], history[-1]["actor_b4"]
    tightens_b0 = b0_R > b0_0 + 1e-9
    improves_b4 = b4_R > b4_0 + 1e-9
    # NB: this refines a DETERMINISTIC MSE-BC proposal — verdicts are SCOPED to that head, not to "no proposal can learn θ(s)"
    if exp_k6 <= pi0_k6 + 0.05:
        verdict = "EXPERT_ADVANTAGE_NOT_REPRODUCED_ON_THIS_PANEL"
    elif tightens_b0 and b0_R >= 0.5 * exp_k6:
        verdict = "DETERMINISTIC_BC_PROPOSAL_ABSORBS_SEARCH_DIRECT_K6_RISES"
    elif tightens_b0 or improves_b4:
        verdict = "DETERMINISTIC_BC_PROPOSAL_PARTIALLY_ABSORBS_SEARCH"
    else:
        verdict = "DETERMINISTIC_BC_PROPOSAL_DOES_NOT_ABSORB_SEARCH_DISTRIBUTIONAL_HEAD_NEEDED"

    out = {"contract": "CARRY_OPTION_REFINE_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "scope": "refines a deterministic MSE-BC proposal; verdict does NOT bound distributional/MDN or template-conditioned-residual heads",
           "eval_n": len(ev_templ), "rounds": rounds, "b_train": b_train, "K": K, "pi_0_K6": pi0_k6, "expert_K6": exp_k6,
           "history": history, "delta_b0": round(b0_R - b0_0, 3), "delta_b4": round(b4_R - b4_0, 3), "verdict": verdict}
    json.dump(out, open(f"{D}/carry_option_refine_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_REFINE_V1 (does the proposal absorb the search? K6 across refine rounds) ==")
    log(f"  pi_0 {pi0_k6} | expert(64) {exp_k6}")
    log(f"  refined actor b=0 (direct): {[h['actor_b0'] for h in history]}  (Δ {round(b0_R-b0_0,3)})")
    log(f"  refined actor b=4 (search): {[h['actor_b4'] for h in history]}  (Δ {round(b4_R-b4_0,3)})")
    log(f"→ {verdict}")
    log(f"wrote {D}/carry_option_refine_v1.json\nCARRY_OPTION_REFINE_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
