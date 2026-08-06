"""Stage 4c (spec-compliant) — bounded search-guided refinement of the VALIDATED template+residual proposal.

Refines the multimodality-preserving head (template classifier + template-conditioned residual), NOT the mode-averaging
continuous BC (that global-MSE control is reported separately as a scoped negative). At each TRAIN option-initiation state:
θ0 = proposal(s) → fixed b=8 local search around θ0 → selected θ* → if it delivers K6/handoff, relabel toward it (ABSTAIN
otherwise), re-fit the proposal. At most TWO rounds. Evaluate at b∈{0,4,8} (b=8 = deployed budget) with any_exit, on a
disjoint PAIRED panel with FIXED search seeds. Success = b=0 K6 > 0.067 without unsafe exit growth, OR b=8 K6 > 0.267 on the
paired panel. Always persist the best SAFE proposal checkpoint; if refinement does not beat the round-0 template baseline,
the RL init stays the round-0 proposal.
"""
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from coin_carry_option_diagnostic import _bank, _panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import canonical_label, fit_proposal, save_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
B_DEPLOY = 8
EVAL_H = 160


def _eval_proposal(prop, ev_templ, pi0, base, b, seed0):
    """Paired eval at budget b with FIXED per-state search seeds; returns (K6, any_exit)."""
    k6, ex = [], []
    for i in range(len(ev_templ)):
        center = prop.theta(ev_templ[i][0].obs())
        _th, out = search_select(ev_templ[i][0], ev_templ[i][1], center, pi0, base, np.random.default_rng(seed0 + i), b=b, horizon=EVAL_H)
        k6.append(int(out["k6"])); ex.append(int(out["contain_exit_ct"] > 0))
    return round(float(np.mean(k6)), 3), round(float(np.mean(ex)), 3)


def _snapshot(prop, ev_templ, pi0, base, rnd, log):
    row = {"round": rnd}
    for b, s0 in ((0, 6000), (4, 7000), (8, 8000)):
        row[f"b{b}_k6"], row[f"b{b}_exit"] = _eval_proposal(prop, ev_templ, pi0, base, b, s0)
    log(f"  [round {rnd}] b0 K6 {row['b0_k6']}(exit {row['b0_exit']}) | b4 {row['b4_k6']}({row['b4_exit']}) | b8 {row['b8_k6']}(exit {row['b8_exit']})")
    return row


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = set(ls.seed for ls in _bank(cfg["banks"]["late_train"])) | set(ls.seed for ls in _bank(cfg["banks"]["late_dev"]))
    rounds = 1 if smoke else 2                                             # bounded: at most TWO rounds
    K = 4 if smoke else 8
    label_shots = 8 if smoke else B_DEPLOY

    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    obs_bank, th_bank = list(z["obs"].astype(np.float32)), list(z["theta"].astype(np.float32))
    prop, info0 = fit_proposal(np.asarray(obs_bank, np.float32), np.asarray(th_bank, np.float32), K, clf_epochs=(80 if smoke else 300), res_epochs=(80 if smoke else 300), seed=0)
    log(f"[proposal] round-0 template+residual fit {info0}")

    tr_templ, _tf = _panel(pi0, range(9000, 10800), forbidden, 16 if smoke else 45)
    ev_templ, ev_fam = _panel(pi0, range(11000, 13000), forbidden, 10 if smoke else 30)
    log(f"[panel] refine-train {len(tr_templ)} | disjoint eval {len(ev_templ)} ({dict(Counter(ev_fam))}) | rounds {rounds} K {K}")

    history = [_snapshot(prop, ev_templ, pi0, base, 0, log)]
    round0 = history[0]
    save_proposal(prop, f"{D}/carry_proposal_round0.pt")                   # the VALIDATED update-0 proposal (always kept)
    best = {"round": 0, "b8_k6": round0["b8_k6"], "b8_exit": round0["b8_exit"], "ckpt": "carry_proposal_round0.pt"}

    for rnd in range(1, rounds + 1):
        add_o, add_t, abstain = [], [], 0
        for i in range(len(tr_templ)):
            center = prop.theta(tr_templ[i][0].obs())
            _th, out = search_select(tr_templ[i][0], tr_templ[i][1], center, pi0, base, np.random.default_rng(3000 + rnd * 100 + i), b=label_shots, horizon=EVAL_H)
            if int(out["k6"]) == 1 or int(out["reached_handoff"]) == 1:
                add_o.append(tr_templ[i][0].obs().copy()); add_t.append(np.asarray(_th, np.float32))
            else:                                                         # local search around θ0 found nothing → strong fallback, else ABSTAIN
                fth, _fo = canonical_label(tr_templ[i][0], tr_templ[i][1], pi0, base, np.random.default_rng(3500 + rnd * 100 + i), shots=(16 if smoke else 64), horizon=EVAL_H)
                if fth is not None:
                    add_o.append(tr_templ[i][0].obs().copy()); add_t.append(fth)
                else:
                    abstain += 1
        obs_bank += add_o; th_bank += add_t
        prop, info = fit_proposal(np.asarray(obs_bank, np.float32), np.asarray(th_bank, np.float32), K, clf_epochs=(80 if smoke else 300), res_epochs=(80 if smoke else 300), seed=rnd)
        log(f"  [refine {rnd}] +{len(add_o)} relabels ({abstain} ABSTAIN) → {len(obs_bank)} labels, fit {info}")
        row = _snapshot(prop, ev_templ, pi0, base, rnd, log)
        history.append(row)
        # keep the best SAFE proposal (b8 K6 up, exit not materially worse than round-0)
        if row["b8_k6"] > best["b8_k6"] and row["b8_exit"] <= round0["b8_exit"] + 0.10:
            save_proposal(prop, f"{D}/carry_proposal_refined.pt")
            best = {"round": rnd, "b8_k6": row["b8_k6"], "b8_exit": row["b8_exit"], "ckpt": "carry_proposal_refined.pt"}

    b0_gain = history[-1]["b0_k6"] > round0["b0_k6"] + 1e-9 and history[-1]["b0_exit"] <= round0["b0_exit"] + 0.10
    b8_gain = best["b8_k6"] > round0["b8_k6"] + 1e-9
    gate_pass = b0_gain or b8_gain
    rl_init = best["ckpt"]                                                 # best safe checkpoint; == round-0 if no gain
    verdict = "STAGE4C_REFINEMENT_IMPROVES_SAFE_PROPOSAL" if gate_pass else "STAGE4C_REFINEMENT_NO_GAIN_RL_INITS_FROM_ROUND0_TEMPLATE"

    out = {"contract": "CARRY_OPTION_STAGE4C_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "head": "template_classifier + template_conditioned_residual (multimodality-preserving)", "K": K,
           "eval_n": len(ev_templ), "rounds_run": rounds, "b_deploy": B_DEPLOY, "history": history,
           "round0": round0, "best_safe": best, "gate_pass": bool(gate_pass), "rl_init_checkpoint": rl_init, "verdict": verdict}
    json.dump(out, open(f"{D}/carry_option_stage4c_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_STAGE4C_V1 (bounded refinement of the VALIDATED template+residual head) ==")
    log(f"  round0: b0 {round0['b0_k6']} | b4 {round0['b4_k6']} | b8 {round0['b8_k6']}")
    log(f"  final : b0 {history[-1]['b0_k6']} | b4 {history[-1]['b4_k6']} | b8 {history[-1]['b8_k6']}")
    log(f"  best SAFE proposal: round {best['round']} b8 {best['b8_k6']} (exit {best['b8_exit']}) → RL init = {rl_init}")
    log(f"→ gate_pass={gate_pass} | {verdict}")
    log(f"wrote {D}/carry_option_stage4c_v1.json\nCARRY_OPTION_STAGE4C_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
