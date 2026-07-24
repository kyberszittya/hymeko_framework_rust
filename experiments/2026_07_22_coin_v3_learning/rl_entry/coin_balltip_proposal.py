"""BALLTIP_COLLISION_ON_V1 — Stage B3: ball-tip teacher bank + refit proposal (Case A action).

B1 diagnosed Case D + Case A: the ball is solvable (strong expert 13/24 ≫ frozen clamp proposal 4/24) and the frozen
clamp pi_0 settling TRANSFERS (settling|handoff ≈0.93). So the bottleneck is the PROPOSAL. This stage:

  1. generates a DISJOINT ball-tip teacher bank — the strong structured expert labels ball states (collision-on, matched
     by transplant) with confident (K6-delivering) θ*, on held-out TRAIN seeds 9000–10800 (disjoint from the 14000–15200
     eval panel; ball labels are kept SEPARATE from the clamp bank — no silent mixing);
  2. trains a template+residual ball-tip proposal (`fit_proposal`) — the SAME machinery as the clamp proposal;
  3. evaluates on the disjoint ball eval panel at b=0 (direct), b=8 (proposal+search), and cites the full-expert ceiling,
     compared against the frozen clamp proposal ZERO-SHOT on the ball;
  4. saves the ball-tip update-0 checkpoint `carry_proposal_balltip_v1.pt`.

Option language + frozen clamp pi_0 settling are UNCHANGED (B1 showed both transfer). No SAC here (B5 is gated on a strong
update-0). Reuses the library (generate_bank, option_teacher_label, fit_proposal, search_select) — no duplication.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout, structured_random_best_with_support  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import build_variant_rl, transplant_handoff  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
OUT = "reports/2026-07-24-balltip-b3-proposal"
FAMS = ("contact_retention", "transport", "braking")
BALL = "balltip_nofilter"                                    # BALLTIP_COLLISION_ON_V1
CLAMP_PROP = f"{D}/carry_proposal_refined.pt"
BALL_PROP = f"{D}/carry_proposal_balltip_v1.pt"             # the B3 ball update-0 checkpoint
BALL_BANK = f"{D}/carry_option_balltip_bank_v1.npz"
HORIZON, K, SEARCH_SEED = 160, 6, 9000
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _ball_transplant(rl_clamp):
    return transplant_handoff(build_variant_rl(BALL, seed=0), rl_clamp)   # reset seed irrelevant (transplant overwrites qpos)


def _committed(ball, gate, pi0, base, theta):
    return structured_carry_rollout(ball, gate, pi0, base, np.asarray(theta, np.float32), horizon=HORIZON)


def _eval_proposal(panel, prop, pi0, base, *, b, on_ball_obs=True):
    """K6 rate of a proposal over the eval panel: θ_center = prop.theta(obs); b=0 executes it directly; b>0 wraps a
    Gaussian search around it. ``on_ball_obs`` feeds the BALL obs (deploy scenario) — the clamp proposal zero-shot sees
    the ball obs too, as it would at deploy. Each rollout gets a FRESH deep-copied gate (avoids gate contamination)."""
    import copy
    k6 = 0
    solved = []
    for i, (rl_c, gate_c, ls) in enumerate(panel):
        ball = _ball_transplant(rl_c)
        obs = ball.obs() if on_ball_obs else rl_c.obs()
        center = prop.theta(obs)
        if b == 0:
            o = _committed(ball, copy.deepcopy(gate_c), pi0, base, center)
        else:
            _t, o, _s = structured_random_best_with_support(ball, copy.deepcopy(gate_c), pi0, base,
                                                            np.random.default_rng(SEARCH_SEED + i), shots=b, horizon=HORIZON,
                                                            center=np.asarray(center, np.float32))
        k6 += int(o["k6"])
        if o["k6"]:
            solved.append(i)
    return k6, solved


def _eval_expert(panel, pi0, base, *, shots):
    import copy
    k6, solved = 0, []
    for i, (rl_c, gate_c, ls) in enumerate(panel):
        ball = _ball_transplant(rl_c)
        _t, o, _s = structured_random_best_with_support(ball, copy.deepcopy(gate_c), pi0, base,
                                                       np.random.default_rng(SEARCH_SEED + i), shots=shots, horizon=HORIZON)
        k6 += int(o["k6"])
        if o["k6"]:
            solved.append(i)
    return k6, solved


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
    want, shots = (12, 24) if smoke else (160, 128)

    # ---- 1. ball-tip teacher bank (disjoint TRAIN seeds; transplant to the ball; strong expert) ----
    log(f"[B3.1] ball-tip teacher bank: {want} states × {shots}-shot expert on {BALL} (collision-on)...")
    train_panel, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=want, families=FAMS,
                                               strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    obs, theta, prov = generate_bank(pi0, base, train_panel, shots=shots, transplant=_ball_transplant, log=log)
    n_labels = len(obs)
    log(f"[B3.1] CONFIDENT ball labels: {n_labels}/{len(train_panel)}")
    if n_labels < 4:
        log(f"[B3.1] too few labels ({n_labels}) to fit any templates — abort (widen budget/states)")
        return {"error": "insufficient_labels", "n_labels": n_labels}
    np.savez(BALL_BANK, obs=np.asarray(obs, np.float32), theta=np.asarray(theta, np.float32))

    # ---- 2. train the ball-tip template+residual proposal ----
    kk = min(K, max(2, n_labels // 4))
    prop_ball, fit_stats = fit_proposal(np.asarray(obs, np.float32), np.asarray(theta, np.float32), kk,
                                        clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
    save_proposal(prop_ball, BALL_PROP)
    log(f"[B3.2] fit ball proposal K={kk} clf_ce {fit_stats['clf_ce']:.3f} res_mse {fit_stats['res_mse']:.4f} → {BALL_PROP}")

    # ---- 3. eval on the disjoint ball eval panel (14000–15200) ----
    ev_want = 8 if smoke else 24
    raw, _c, _s = build_boundary_panel(pi0, range(14000, 15200), forbidden, want=ev_want, families=FAMS,
                                       strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    eval_panel = [(*reconstruct_handoff(pi0, ls, horizon=360)[:2], ls) for ls in raw]
    n = len(eval_panel)
    clamp_prop = load_proposal(CLAMP_PROP)
    log(f"[B3.3] eval on {n} disjoint ball states (b=0 direct / b=8 search / expert ceiling / clamp zero-shot)...")
    ball_b0 = _eval_proposal(eval_panel, prop_ball, pi0, base, b=0)
    ball_b8 = _eval_proposal(eval_panel, prop_ball, pi0, base, b=8)
    clamp_b0 = _eval_proposal(eval_panel, clamp_prop, pi0, base, b=0)
    clamp_b8 = _eval_proposal(eval_panel, clamp_prop, pi0, base, b=8)
    expert = _eval_expert(eval_panel, pi0, base, shots=64 if smoke else 192)

    res = {"n_eval": n, "ball_proposal_b0": ball_b0[0], "ball_proposal_b8": ball_b8[0],
           "clamp_proposal_b0_zeroshot": clamp_b0[0], "clamp_proposal_b8_zeroshot": clamp_b8[0],
           "expert_ceiling": expert[0],
           "solved": {"ball_b0": ball_b0[1], "ball_b8": ball_b8[1], "clamp_b8": clamp_b8[1], "expert": expert[1]}}
    manifest = {"contract": "BALLTIP_COLLISION_ON_V1", "stage": "B3", "date": "2026-07-24", "smoke": smoke,
                "baseline": BASELINE, "robot": "ball r0.020 collision-ON", "teacher_shots": shots, "n_labels": n_labels,
                "fit": fit_stats, "K": kk, "ball_proposal_ckpt": BALL_PROP.split("/")[-1], "bank": BALL_BANK.split("/")[-1],
                "eval": res, "provenance_stats": {"scanned": len(train_panel), "confident": n_labels}}
    json.dump(manifest, open(f"{OUT}/b3_proposal.json", "w"), indent=1, default=float)

    log("\n== BALLTIP_COLLISION_ON_V1 — Stage B3: ball-tip proposal ==")
    log(f"  teacher bank: {n_labels} confident ball labels ({shots}-shot expert) | proposal K={kk}")
    log(f"  eval ({n} disjoint ball states):")
    log(f"    ball proposal   b=0 {ball_b0[0]:>2}/{n}   b=8 {ball_b8[0]:>2}/{n}")
    log(f"    clamp zero-shot b=0 {clamp_b0[0]:>2}/{n}   b=8 {clamp_b8[0]:>2}/{n}")
    log(f"    full expert ceiling {expert[0]:>2}/{n}")
    verdict = ("BALLTIP_PROPOSAL_REFIT_SUFFICIENT" if ball_b8[0] > clamp_b8[0] else "BALLTIP_PROPOSAL_REFIT_INSUFFICIENT")
    log(f"  → {verdict} (ball b=8 {ball_b8[0]} vs clamp zero-shot b=8 {clamp_b8[0]}; ceiling {expert[0]})")
    log(f"  artifacts: {OUT}/b3_proposal.json + {BALL_PROP}\nBALLTIP_B3_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
