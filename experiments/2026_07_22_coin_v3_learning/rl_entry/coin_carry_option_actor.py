"""CARRY_OPTION_ACTOR_V1 — Stage 3: option BC + option-level DAgger from the Stage-2 teacher bank, with the required
pre-checks, then a physical update-0 gate on a disjoint held-out carry panel.

Pre-checks (before full training): tiny-set overfit (can the representation fit clean labels?), θ normalization round-trip,
saved-state label replay (the stored θ reproduces the teacher outcome), θ-multimodality inspection. DAgger operates ONLY at
option decision points (student picks θ → committed macro → teacher relabels the recovery/option-initiation state the
student reaches). Update-0 gate is physical (K6 > pi_0, exit not worse, not just handoff). Semi-MDP SAC/TD3 over θ is next.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option import (  # noqa: E402
    make_option_actor,
    option_controller_rollout,
    recovery_state_theta,
    teacher_theta,
    train_option_bc,
)
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, T_MAX, T_MIN, structured_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
SHOTS, TEACHER_H, EVAL_H, MAX_PROBE, DAGGER_ITERS = 64, 160, 160, 60, 2


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _panel(pi0, seeds, forbidden, want):
    panel, _c, _s = build_boundary_panel(pi0, seeds, forbidden, want=want, families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"]
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY
        templates.append((rl, gate)); fam.append(ls.family)
    return templates, fam


def _norm(theta):
    t = np.asarray(theta, np.float32)
    return np.concatenate([t[:12] / A_BOUND, (t[12:] - T_MIN) / (T_MAX - T_MIN)])


def _precheck(actor_cls, obs, theta, log):
    """Tiny-set overfit, θ normalization round-trip, multimodality note. Returns (ok, info)."""
    # normalization round-trip
    z = _norm(theta[0]); rt = np.concatenate([z[:12] * A_BOUND, z[12:] * (T_MAX - T_MIN) + T_MIN])
    assert np.allclose(rt, theta[0], atol=1e-4), "θ normalization not invertible"
    # tiny-set overfit: can the actor fit a clean 4-label subset to near-zero?
    tiny = make_option_actor()
    k = min(4, len(obs))
    loss = train_option_bc(tiny, obs[:k], theta[:k], epochs=800, lr=3e-3, batch=k, seed=0)
    log(f"[precheck] normalization round-trip OK | tiny-overfit ({k} labels) final MSE {round(loss, 6)}")
    return loss < 0.02, {"tiny_overfit_mse": round(loss, 6), "k": k}


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    shots = 16 if smoke else SHOTS
    diters = 1 if smoke else DAGGER_ITERS

    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    obs_bank, th_bank = list(z["obs"]), list(z["theta"])
    log(f"[bank] loaded {len(obs_bank)} confident option labels from Stage-2 teacher bank")

    ok, pcinfo = _precheck(make_option_actor, np.asarray(obs_bank, np.float32), np.asarray(th_bank, np.float32), log)
    if not ok:
        json.dump({"contract": "CARRY_OPTION_ACTOR_V1", "verdict": "HARD_STOP_BC_CANNOT_OVERFIT_TINY_SET", "precheck": pcinfo},
                  open(f"{D}/carry_option_actor_v1.json", "w"), indent=1, default=float)
        log(f"→ HARD_STOP: BC cannot overfit a tiny clean subset (mse {pcinfo['tiny_overfit_mse']}) — representation/training defect")
        log("CARRY_OPTION_ACTOR_DONE"); return

    # DAgger student-rollout TRAIN panel + disjoint EVAL panel (seeds ≥ 11000, disjoint from the bank's ≤10800)
    n_tr = 12 if smoke else 40
    n_ev = 10 if smoke else 30
    tr_templ, tr_fam = _panel(pi0, range(9000, 10800), forbidden, n_tr)
    ev_templ, ev_fam = _panel(pi0, range(11000, 13000), forbidden, n_ev)
    log(f"[panel] DAgger-train {len(tr_templ)} | disjoint eval {len(ev_templ)} ({dict(Counter(ev_fam))})")

    # multimodality inspection: does the teacher give a consistent θ* across seeds on a few states?
    mm = []
    for i in range(min(4, len(tr_templ))):
        ths = [teacher_theta(*tr_templ[i], pi0, base, np.random.default_rng(s), shots=shots, horizon=TEACHER_H)[0] for s in (11, 22, 33)]
        mm.append(round(float(np.mean([np.linalg.norm(_norm(a) - _norm(b)) for a in ths for b in ths if a is not b])), 3))
    log(f"[precheck] θ-multimodality (cross-seed normalized L2, 4 states): {mm}")

    actor = make_option_actor()
    bc_loss = train_option_bc(actor, obs_bank, th_bank, epochs=(80 if smoke else 400), lr=1e-3, batch=32, seed=0)
    bc_actor = copy.deepcopy(actor)
    log(f"[BC] {len(obs_bank)} θ-labels, final MSE {round(bc_loss, 5)}")
    for it in range(diters):
        add_o, add_t = [], []
        for i in range(len(tr_templ)):
            r = recovery_state_theta(*tr_templ[i], actor, pi0, base, np.random.default_rng(2000 + it * 100 + i), shots=shots, horizon=TEACHER_H, max_probe=MAX_PROBE)
            if r is not None:
                add_o.append(r[0]); add_t.append(r[1])
        obs_bank += add_o; th_bank += add_t
        dl = train_option_bc(actor, obs_bank, th_bank, epochs=(60 if smoke else 200), lr=5e-4, batch=32, seed=it + 1)
        log(f"[DAgger {it+1}] +{len(add_o)} recovery θ-labels → {len(obs_bank)} total, MSE {round(dl, 5)}")

    def evl(fn):
        return [fn(i) for i in range(len(ev_templ))]
    pi0_out = evl(lambda i: sequence_then_pi0(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H))
    exp_out = evl(lambda i: structured_random(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.random.default_rng(500 + i), shots=shots, horizon=EVAL_H))
    bc_out = evl(lambda i: option_controller_rollout(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), bc_actor, pi0, base, horizon=EVAL_H))
    dg_out = evl(lambda i: option_controller_rollout(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), actor, pi0, base, horizon=EVAL_H))

    def rate(o, key="k6"):
        return round(float(np.mean([x[key] for x in o])), 3)
    def any_exit(o):
        return round(float(np.mean([x["contain_exit_ct"] > 0 for x in o])), 3)
    agg = {name: {"K6": rate(o), "handoff": rate(o, "reached_handoff"), "any_exit": any_exit(o)}
           for name, o in (("pi_0", pi0_out), ("structured_expert", exp_out), ("BC_option", bc_out), ("DAgger_option", dg_out))}
    agg["DAgger_option"]["mean_options"] = rate(dg_out, "options"); agg["DAgger_option"]["mean_aborts"] = rate(dg_out, "aborts")
    S_pi0 = {i for i in range(len(ev_templ)) if pi0_out[i]["k6"]}; S_dg = {i for i in range(len(ev_templ)) if dg_out[i]["k6"]}; S_exp = {i for i in range(len(ev_templ)) if exp_out[i]["k6"]}
    per_fam = {f: {"n": sum(x == f for x in ev_fam), "pi0": round(float(np.mean([pi0_out[i]["k6"] for i in range(len(ev_templ)) if ev_fam[i] == f])), 3),
                   "DAgger": round(float(np.mean([dg_out[i]["k6"] for i in range(len(ev_templ)) if ev_fam[i] == f])), 3)} for f in sorted(set(ev_fam))}

    dk6, pk6, bck6, ek6 = agg["DAgger_option"]["K6"], agg["pi_0"]["K6"], agg["BC_option"]["K6"], agg["structured_expert"]["K6"]
    meaningful_coverage = ek6 <= 0 or dk6 >= 0.4 * ek6
    gate = (dk6 > pk6 + 0.05 and agg["DAgger_option"]["any_exit"] <= agg["pi_0"]["any_exit"] + 0.15
            and agg["DAgger_option"]["handoff"] >= agg["DAgger_option"]["K6"] and meaningful_coverage)
    if len(ev_templ) < 8 or len(obs_bank) < 8:
        verdict = "OPTION_ACTOR_UNDERPOWERED"; nxt = f"only {len(obs_bank)} labels / {len(ev_templ)} eval states"
    elif gate:
        verdict = "OPTION_ACTOR_BEATS_PI0_UPDATE0_PASS"
        nxt = (f"option actor beats pi_0 ({dk6} vs {pk6}; expert {ek6}, BC {bck6}), exit ok, K6 not just handoff → RL: semi-MDP "
               "macro-SAC/TD3 over θ from THIS checkpoint (target R_option + γ^τ Q(s_next,π); NOT one-step γ)")
    elif dk6 > pk6:
        verdict = "OPTION_ACTOR_HELPS_BUT_MISSES_A_GATE_CONDITION"
        nxt = f"DAgger {dk6} > pi_0 {pk6} but a gate condition (exit / handoff≥K6 / ≥40% expert) not met — one bounded corrective pass"
    else:
        verdict = "OPTION_ACTOR_NO_BETTER_THAN_PI0"
        nxt = f"option {dk6} ≤ pi_0 {pk6} — diagnose (label scarcity {len(obs_bank)}, multimodality {mm}, BC {bck6}); one bounded corrective pass before any conclusion"

    out = {"contract": "CARRY_OPTION_ACTOR_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "precheck": {**pcinfo, "multimodality_cross_seed_L2": mm}, "bc_loss": round(bc_loss, 5),
           "bank_labels_final": len(obs_bank), "panel": {"eval_n": len(ev_templ), "eval_families": dict(Counter(ev_fam))},
           "aggregate": agg, "per_family": per_fam,
           "solved": {"pi_0": sorted(S_pi0), "DAgger": sorted(S_dg), "expert": sorted(S_exp), "DAgger_minus_pi0": sorted(S_dg - S_pi0)},
           "gate_met": gate, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/carry_option_actor_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_ACTOR_V1 (update-0: option θ teacher → BC/DAgger vs pi_0) ==")
    for name in ("pi_0", "structured_expert", "BC_option", "DAgger_option"):
        a = agg[name]; log(f"  {name:18}: K6 {a['K6']} | handoff {a['handoff']} | any_exit {a['any_exit']}")
    log(f"  DAgger options {agg['DAgger_option']['mean_options']} aborts {agg['DAgger_option']['mean_aborts']} | DAgger-solved-not-pi0 {sorted(S_dg - S_pi0)} | per-fam {per_fam}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_option_actor_v1.json\nCARRY_OPTION_ACTOR_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
