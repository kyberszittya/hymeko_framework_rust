"""CARRY_DAGGER_ACTOR_V1 — Phase 4b (part 1): distil the structured carry expert into a low-level 4D carry actor, on the
STUDENT-induced distribution, and evaluate update-0 (BC + DAgger) against pi_0 and the expert on a disjoint held-out panel.

Teacher = structured-random receding-horizon (first-action labels only, ABSTAIN when unsolvable). Actor = low-level obs48→4D,
acts at strict-0 under the carry gate, hands off to FROZEN pi_0 at strict≥1. DAgger: BC on the teacher bank → student
rollout → teacher relabels the states the student actually visited → refit. Update-0 gate (pre-registered, physical): the
DAgger actor beats pi_0 in eventual K6 (not just handoff) with full-containment exit not materially worse. Comparison:
pi_0 / structured-random expert (ceiling) / BC actor / final DAgger actor. RL (SAC/TD3 from this checkpoint) is the next step.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_dagger import (  # noqa: E402
    carry_actor_rollout,
    make_carry_actor,
    teacher_first_action,
    teacher_warmstart_bank,
    train_bc,
)
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
TEACHER_SHOTS, TEACHER_H, ROLL_H, EVAL_H = 16, 100, 120, 160
DAGGER_ITERS = 2


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _panel(pi0, seeds, forbidden, want):
    panel, _c, _s = build_boundary_panel(pi0, seeds, forbidden, want=want, families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"], f"identity mismatch {ls.seed}"
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY
        templates.append((rl, gate)); fam.append(ls.family)
    return templates, fam


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    n_train, n_eval = (8, 8) if smoke else (20, 20)
    tshots = 8 if smoke else TEACHER_SHOTS
    diters = 1 if smoke else DAGGER_ITERS

    log("[panel] TRAIN carry panel (seeds 9000–10400) + disjoint EVAL panel (seeds 10500–12000)...")
    tr_templ, tr_fam = _panel(pi0, range(9000, 10400), forbidden, n_train)
    ev_templ, ev_fam = _panel(pi0, range(10500, 12000), forbidden, n_eval)
    log(f"[panel] train {len(tr_templ)} ({dict(Counter(tr_fam))}) | eval {len(ev_templ)} ({dict(Counter(ev_fam))})")

    # ── initial teacher bank (receding-horizon; abstain on unsolvable) ──
    log("[teacher] building the warm-started RECEDING-horizon teacher bank (replanned first-action labels)...")
    obs_bank, act_bank = [], []; tstats = {"k6": 0, "handoff": 0, "abstained": 0}
    for i in range(len(tr_templ)):
        o, a, st = teacher_warmstart_bank(*tr_templ[i], pi0, base, np.random.default_rng(1000 + i),
                                          strong_shots=tshots * 3, warm_shots=max(4, tshots // 2), teacher_h=TEACHER_H, roll_h=ROLL_H)
        obs_bank += o; act_bank += a
        for k in tstats:
            tstats[k] += st[k]
    log(f"[teacher] {len(obs_bank)} labels | teacher K6 {tstats['k6']}/{len(tr_templ)} handoff {tstats['handoff']} abstained {tstats['abstained']}")

    # ── BC, then DAgger on the student-induced distribution ──
    actor = make_carry_actor(pi0)
    bc_loss = train_bc(actor, obs_bank, act_bank, epochs=(40 if smoke else 120), lr=1e-3, batch=64, seed=0)
    bc_actor = copy.deepcopy(actor)                                       # snapshot the pure-BC actor for the comparison
    log(f"[BC] {len(obs_bank)} labels, final MSE {round(bc_loss, 5)}")
    for it in range(diters):
        add_o, add_a = [], []                                            # relabel the states the STUDENT actually visits
        for i in range(len(tr_templ)):
            _out, visited = carry_actor_rollout(copy.deepcopy(tr_templ[i][0]), copy.deepcopy(tr_templ[i][1]), actor, pi0, base, horizon=ROLL_H, collect=True)
            for j, (rl_s, gate_s, o48) in enumerate(visited[::3]):        # subsample visited states to bound teacher cost
                a, adm, _o = teacher_first_action(rl_s, gate_s, pi0, base, np.random.default_rng(2000 + it * 500 + i * 20 + j), shots=tshots, horizon=TEACHER_H)
                if adm:                                                   # ABSTAIN on states the teacher cannot solve
                    add_o.append(o48); add_a.append(a)
        obs_bank += add_o; act_bank += add_a
        dl = train_bc(actor, obs_bank, act_bank, epochs=(30 if smoke else 60), lr=5e-4, batch=64, seed=it + 1)
        log(f"[DAgger {it+1}] +{len(add_o)} student-visited relabels → {len(obs_bank)} total, MSE {round(dl, 5)}")

    # ── update-0 eval on the disjoint held-out panel ──
    def eval_panel(controller):
        outs = [controller(i) for i in range(len(ev_templ))]
        return outs
    pi0_out = eval_panel(lambda i: sequence_then_pi0(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H))
    exp_out = eval_panel(lambda i: structured_random(*[copy.deepcopy(t) for t in ev_templ[i]], pi0, base, np.random.default_rng(400 + i), shots=tshots * 4, horizon=EVAL_H))
    bc_out = eval_panel(lambda i: carry_actor_rollout(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), bc_actor, pi0, base, horizon=EVAL_H))
    dg_out = eval_panel(lambda i: carry_actor_rollout(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), actor, pi0, base, horizon=EVAL_H))

    def rate(outs, key="k6"):
        return round(float(np.mean([o[key] for o in outs])), 3)
    def any_exit(outs):
        return round(float(np.mean([o["contain_exit_ct"] > 0 for o in outs])), 3)
    agg = {name: {"K6": rate(o), "handoff": rate(o, "reached_handoff"), "any_exit": any_exit(o)}
           for name, o in (("pi_0", pi0_out), ("structured_expert", exp_out), ("BC", bc_out), ("DAgger", dg_out))}
    # teacher–student action error on eval strict-0 start states (admissible only)
    errs = []
    for i in range(len(ev_templ)):
        ta, adm, _o = teacher_first_action(ev_templ[i][0], ev_templ[i][1], pi0, base, np.random.default_rng(500 + i), shots=tshots, horizon=TEACHER_H)
        if adm:
            with torch.no_grad():
                sa = torch.clamp(actor.action_mean(torch.as_tensor(ev_templ[i][0].obs()[None]).float()), -4, 4)[0].numpy()
            errs.append(float(np.linalg.norm(sa - ta)))
    per_fam = {f: {"n": sum(x == f for x in ev_fam), "pi0": round(float(np.mean([pi0_out[i]["k6"] for i in range(len(ev_templ)) if ev_fam[i] == f])), 3),
                   "DAgger": round(float(np.mean([dg_out[i]["k6"] for i in range(len(ev_templ)) if ev_fam[i] == f])), 3)} for f in sorted(set(ev_fam))}

    dk6, pk6 = agg["DAgger"]["K6"], agg["pi_0"]["K6"]
    gate = (dk6 > pk6 and agg["DAgger"]["any_exit"] <= agg["pi_0"]["any_exit"] + 0.15 and agg["DAgger"]["handoff"] >= agg["DAgger"]["K6"])
    if len(ev_templ) < 8:
        verdict, nxt = "CARRY_DAGGER_UNDERPOWERED", "widen the eval panel"
    elif gate:
        verdict = "CARRY_DAGGER_ACTOR_BEATS_PI0_UPDATE0_PASS"
        nxt = (f"update-0 carry actor beats pi_0 in held-out K6 ({dk6} vs {pk6}; expert ceiling {agg['structured_expert']['K6']}, "
               f"BC {agg['BC']['K6']}), exit not worse, K6 up not just handoff → proceed to RL: from THIS DAgger checkpoint run "
               "SAC and TD3 (same replay/start/reward/eval/frozen-pi_0/budget); the claim is SAC/TD3-final > this update-0 in "
               "held-out physical K6, reported per checkpoint (0/early/mid/best-val/final). Re-confirm at larger n in passing")
    elif dk6 > pk6:
        verdict = "CARRY_DAGGER_HELPS_BUT_MISSES_A_GATE_CONDITION"
        nxt = f"DAgger K6 {dk6} > pi_0 {pk6} but exit/handoff-vs-K6 gate not fully met — more DAgger iters / bank / a tuned teacher before RL"
    else:
        verdict = "CARRY_DAGGER_NO_BETTER_THAN_PI0"
        nxt = "the distilled actor does not beat pi_0 — the amortization gap is too large; stronger teacher (structured CEM/CMA), more DAgger iters, or richer bank"

    out = {"contract": "CARRY_DAGGER_ACTOR_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "method": {"teacher": "structured-random receding-horizon, first-action labels, abstain", "actor": "low-level obs48→4D, strict-0 only, frozen pi_0 after handoff",
                      "dagger_iters": diters, "teacher_shots": tshots, "eval_H": EVAL_H, "disjoint_train_eval": True},
           "panel": {"train_n": len(tr_templ), "eval_n": len(ev_templ), "train_families": dict(Counter(tr_fam)), "eval_families": dict(Counter(ev_fam))},
           "teacher_bank": {"n_labels": len(act_bank), **tstats}, "bc_loss": round(bc_loss, 5),
           "aggregate": agg, "per_family": per_fam,
           "teacher_student_action_error": {"mean": round(float(np.mean(errs)), 4) if errs else None, "n": len(errs)},
           "gate_met": gate, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/carry_dagger_actor_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_DAGGER_ACTOR_V1 (update-0: teacher→BC→DAgger vs pi_0, held-out) ==")
    for name in ("pi_0", "structured_expert", "BC", "DAgger"):
        a = agg[name]; log(f"  {name:18}: K6 {a['K6']} | handoff {a['handoff']} | any_exit {a['any_exit']}")
    log(f"  teacher–student action error (mean L2) {out['teacher_student_action_error']['mean']} (n {len(errs)}) | eval per-family {per_fam}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_dagger_actor_v1.json\nCARRY_DAGGER_ACTOR_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
