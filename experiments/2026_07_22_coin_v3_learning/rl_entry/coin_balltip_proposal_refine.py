"""BALLTIP_COLLISION_ON_V1 — Stage B3-iteration: strengthen the ball proposal (robust-label filter + DAgger refinement).

B3 gave a WEAK update-0 (ball b=8 5/24, b=0 0/24, ceiling 16/24; res_mse 0.40). Two levers, per your go-ahead:
  1. ROBUST-LABEL FILTER — keep only labels whose θ* holds K6 under jitter (robust_k6 ≥ 0.67), de-noising the residual.
  2. DAgger REFINEMENT (search-as-teacher) — each round: propose θ → search AROUND it on the ball → adopt the
     search-discovered K6 θ as a new label → refit. If b=0 (direct) rises and b=8 approaches the ceiling, the proposal is
     absorbing the search ⇒ a strong update-0 ⇒ green light for B5. Reuses generate_bank / fit_proposal / the B3 eval
     helpers / structured_random_around (no duplication). Frozen clamp pi_0 + option language UNCHANGED.
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import (  # noqa: E402
    BALL_PROP, D, FAMS, HORIZON, K, _ball_transplant, _bank, _eval_expert, _eval_proposal)
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_random_around  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-balltip-b3-proposal"
ROBUST_MIN = 0.67                                          # keep labels holding K6 in ≥2/3 jitter re-rolls
B_TRAIN, STD_AMP, STD_DUR = 12, 0.6, 2.0


def _robust_filter(obs, theta, prov):
    """Keep only confident labels whose θ* is robust (robust_k6 ≥ ROBUST_MIN). prov is per-state; obs/θ are the confident
    subset in order, so the confident prov entries align with obs/θ."""
    rk = [(p.get("robust_k6") or 0.0) for p in prov if p["confident"]]
    keep = [i for i, r in enumerate(rk) if r >= ROBUST_MIN]
    return [obs[i] for i in keep], [theta[i] for i in keep], len(rk)


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
    want, shots, rounds = (12, 24, 1) if smoke else (160, 128, 3)

    # ---- round-0 ball bank + ROBUST FILTER ----
    log(f"[iter.0] ball bank {want}×{shots}-shot + robust filter (robust_k6≥{ROBUST_MIN})...")
    train_panel, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=want, families=FAMS,
                                               strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    obs0, th0, prov0 = generate_bank(pi0, base, train_panel, shots=shots, transplant=_ball_transplant, log=log)
    obs_bank, th_bank, n_conf = _robust_filter(obs0, th0, prov0)
    log(f"[iter.0] robust-filtered labels: {len(obs_bank)}/{n_conf} confident (dropped {n_conf - len(obs_bank)} non-robust)")
    if len(obs_bank) < 4:
        log("[iter.0] too few robust labels — abort")
        return {"error": "insufficient_robust_labels", "n": len(obs_bank)}

    # ---- DAgger refine-train states (disjoint from bank 9000-10800 and eval 14000-15200) ----
    rf_want = 8 if smoke else 45
    rf_raw, _c, _s = build_boundary_panel(pi0, range(11000, 13000), forbidden, want=rf_want, families=FAMS,
                                          strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    refine = [(_ball_transplant(rl_c), gate) for rl_c, gate, _h, _r in (reconstruct_handoff(pi0, ls, horizon=360) for ls in rf_raw)]

    # ---- eval panel (24 disjoint ball states) ----
    ev_want = 8 if smoke else 24
    ev_raw, _c, _s = build_boundary_panel(pi0, range(14000, 15200), forbidden, want=ev_want, families=FAMS,
                                          strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    eval_panel = [(*reconstruct_handoff(pi0, ls, horizon=360)[:2], ls) for ls in ev_raw]
    n = len(eval_panel)
    expert_k6, _es = _eval_expert(eval_panel, pi0, base, shots=64 if smoke else 192)

    def fit(ob, th):
        kk = min(K, max(2, len(ob) // 4))
        p, st = fit_proposal(np.asarray(ob, np.float32), np.asarray(th, np.float32), kk,
                             clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
        return p, kk, st

    def evaluate(prop):
        return _eval_proposal(eval_panel, prop, pi0, base, b=0)[0], _eval_proposal(eval_panel, prop, pi0, base, b=8)[0]

    history = []
    prop, kk, st = fit(obs_bank, th_bank)
    b0, b8 = evaluate(prop)
    history.append({"round": 0, "labels": len(obs_bank), "b0": b0, "b8": b8, "res_mse": round(st["res_mse"], 4)})
    log(f"  [round 0] labels {len(obs_bank)} K={kk} res_mse {st['res_mse']:.3f} | b=0 {b0}/{n} b=8 {b8}/{n} (ceiling {expert_k6}/{n})")
    best = {"b8": b8, "prop": prop, "round": 0}

    for rnd in range(1, rounds + 1):
        add_o, add_t = [], []
        for i, (rl, gate) in enumerate(refine):
            center = prop.theta(rl.obs())
            th_best, out = structured_random_around(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                    np.random.default_rng(3000 + rnd * 100 + i), shots=B_TRAIN,
                                                    center=np.asarray(center, np.float32), std_amp=STD_AMP, std_dur=STD_DUR, horizon=HORIZON)
            if int(out["k6"]) == 1:                                     # search found a K6 θ near the proposal → teach it
                add_o.append(rl.obs().copy())
                add_t.append(np.asarray(th_best, np.float32))
        obs_bank += add_o
        th_bank += add_t
        prop, kk, st = fit(obs_bank, th_bank)
        b0, b8 = evaluate(prop)
        history.append({"round": rnd, "labels": len(obs_bank), "b0": b0, "b8": b8, "res_mse": round(st["res_mse"], 4), "added": len(add_o)})
        log(f"  [round {rnd}] +{len(add_o)} search-K6 → {len(obs_bank)} labels K={kk} res_mse {st['res_mse']:.3f} | b=0 {b0}/{n} b=8 {b8}/{n}")
        if b8 >= best["b8"]:
            best = {"b8": b8, "prop": prop, "round": rnd}

    save_proposal(best["prop"], BALL_PROP)                              # overwrite the ball update-0 with the best round
    b0_0, b0_R = history[0]["b0"], history[-1]["b0"]
    b8_0, b8_R = history[0]["b8"], history[-1]["b8"]
    strong = (best["b8"] >= 0.5 * expert_k6) and (b0_R > b0_0)
    verdict = ("BALLTIP_UPDATE0_STRONG_B5_GREENLIT" if strong else
               "BALLTIP_UPDATE0_IMPROVED_STILL_WEAK" if best["b8"] > history[0]["b8"] else
               "BALLTIP_REFINEMENT_DID_NOT_STRENGTHEN")
    manifest = {"contract": "BALLTIP_COLLISION_ON_V1", "stage": "B3-iteration", "date": "2026-07-24", "smoke": smoke,
                "robust_min": ROBUST_MIN, "n_eval": n, "expert_ceiling": expert_k6, "history": history,
                "best_round": best["round"], "best_b8": best["b8"], "delta_b0": b0_R - b0_0, "delta_b8": b8_R - b8_0,
                "ball_proposal_ckpt": BALL_PROP.split("/")[-1], "verdict": verdict,
                "clamp_zeroshot_b8": load_and_eval_clamp(eval_panel, pi0, base)}
    json.dump(manifest, open(f"{OUT}/b3_iteration.json", "w"), indent=1, default=float)

    log("\n== BALLTIP B3-iteration (robust filter + DAgger refinement) ==")
    log(f"  b=0 direct: {[h['b0'] for h in history]}  (Δ {b0_R - b0_0})")
    log(f"  b=8 search: {[h['b8'] for h in history]}  (Δ {b8_R - b8_0}) | ceiling {expert_k6}/{n} | best round {best['round']}")
    log(f"→ {verdict}\n  artifacts: {OUT}/b3_iteration.json + {BALL_PROP}\nBALLTIP_B3ITER_DONE")
    return manifest


def load_and_eval_clamp(eval_panel, pi0, base):
    clamp = load_proposal(f"{D}/carry_proposal_refined.pt")
    return _eval_proposal(eval_panel, clamp, pi0, base, b=8)[0]


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
