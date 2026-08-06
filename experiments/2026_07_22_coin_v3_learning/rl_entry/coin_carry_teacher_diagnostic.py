"""CARRY_TEACHER_DIAGNOSTIC_V1 — localise the DAgger teacher bottleneck BEFORE judging the actor.

On the SAME 20 TRAIN carry states, three teachers, three metrics:
  (1) STRONG s0 expert       — one strong structured-random solve from s0 (initial-state K6 CEILING).
  (2) WARM receding (no fb)   — the cheap warm-started receding teacher (receding K6 + first-action label yield).
  (3) WARM receding + FALLBACK — two-tier: warm replan, and on a warm abstain fall back to a full strong solve.

Interpretation: strong-s0 ≈ warm ≈ low → the panel is harder (need more expert coverage, no bug); strong-s0 ≫ warm →
the lightweight replan config is the culprit; warm+fallback ≈ strong-s0 → the fallback fixes it. Abstention reasons are
logged (INITIAL_STRONG_ABSTAIN = panel-hard here vs WARM_THEN_STRONG_ABSTAIN = both fail). No BC/RL here.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_dagger import teacher_warmstart_bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
STRONG_SHOTS, WARM_SHOTS, TEACHER_H, ROLL_H = 64, 8, 160, 120


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    want = 8 if smoke else 20
    strong = 16 if smoke else STRONG_SHOTS

    panel, _c, _s = build_boundary_panel(pi0, range(9000, 10400), forbidden, want=want, families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"]
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY
        templates.append((rl, gate)); fam.append(ls.family)
    n = len(templates)
    log(f"[panel] {n} TRAIN carry states | families {dict(Counter(fam))} | strong_shots {strong}")

    s0_k6, warm, warmfb = [], [], []
    for i in range(n):
        rl0, gate0 = templates[i]
        # (1) strong s0 expert ceiling (one strong solve)
        so = structured_random(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, np.random.default_rng(700 + i), shots=strong, horizon=TEACHER_H)
        s0_k6.append(so)
        # (2) warm receding, NO fallback
        _o, _a, w = teacher_warmstart_bank(rl0, gate0, pi0, base, np.random.default_rng(800 + i), strong_shots=strong, warm_shots=WARM_SHOTS, teacher_h=TEACHER_H, roll_h=ROLL_H, fallback=False)
        warm.append(w)
        # (3) warm receding + strong fallback
        _o2, _a2, wf = teacher_warmstart_bank(rl0, gate0, pi0, base, np.random.default_rng(900 + i), strong_shots=strong, warm_shots=WARM_SHOTS, teacher_h=TEACHER_H, roll_h=ROLL_H, fallback=True)
        warmfb.append(wf)
        if i % 5 == 0 or smoke:
            log(f"  [{i+1}/{n} {fam[i]:16}] s0-ceiling K6 {so['k6']} | warm K6 {w['k6']} lbl {w['n_labels']} ({w['abstain_reason']}) | warm+fb K6 {wf['k6']} lbl {wf['n_labels']} fb {wf['fallbacks']} ({wf['abstain_reason']})")

    def rate(outs, key):
        return round(float(np.mean([o[key] for o in outs])), 3)
    metrics = {
        "strong_s0_ceiling": {"K6": rate(s0_k6, "k6"), "handoff": rate(s0_k6, "reached_handoff")},
        "warm_receding": {"K6": rate(warm, "k6"), "handoff": rate(warm, "handoff"), "mean_labels": rate(warm, "n_labels"),
                          "abstain_reasons": dict(Counter(w["abstain_reason"] for w in warm))},
        "warm_plus_fallback": {"K6": rate(warmfb, "k6"), "handoff": rate(warmfb, "handoff"), "mean_labels": rate(warmfb, "n_labels"),
                               "mean_fallbacks": rate(warmfb, "fallbacks"), "abstain_reasons": dict(Counter(w["abstain_reason"] for w in warmfb))},
    }
    total_labels_fb = int(np.sum([w["n_labels"] for w in warmfb]))
    c, w0, wf0 = metrics["strong_s0_ceiling"]["K6"], metrics["warm_receding"]["K6"], metrics["warm_plus_fallback"]["K6"]

    if c <= 0.3 and w0 <= 0.3:
        verdict = "PANEL_HARD_NEED_MORE_EXPERT_COVERAGE"
        nxt = f"even the strong s0 expert only solves {c} here — this TRAIN panel is harder than the earlier 0.833 fresh panel; not a config bug. Enlarge/rebalance the panel (more solvable carry states) or raise the strong budget toward the validated setting"
    elif c >= w0 + 0.15 and wf0 >= c - 0.1:
        verdict = "REPLAN_CONFIG_WAS_THE_CULPRIT_FALLBACK_FIXES_IT"
        nxt = f"strong s0 {c} ≫ warm {w0}, and warm+fallback {wf0} recovers to ≈ceiling → the lightweight warm replan was collapsing; use the two-tier teacher (fallback=True). Total feedback labels now {total_labels_fb}. Re-run the DAgger update-0 with this teacher"
    elif c >= w0 + 0.15 and wf0 < c - 0.1:
        verdict = "REPLAN_WEAK_FALLBACK_INSUFFICIENT"
        nxt = f"strong s0 {c} ≫ warm {w0}, and even warm+fallback {wf0} < ceiling → the receding trajectory drifts to states the strong solve can't rescue; raise warm budget / replan cadence, or seed the warm start from the s0 plan more tightly"
    else:
        verdict = "TEACHERS_COMPARABLE_INSPECT_LABEL_YIELD"
        nxt = f"strong s0 {c} ≈ warm {w0} ≈ warm+fb {wf0}; the bottleneck is label YIELD/consistency not raw coverage — total labels {total_labels_fb}; enrich the bank / address multimodality"

    out = {"contract": "CARRY_TEACHER_DIAGNOSTIC_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "panel": {"n": n, "families": dict(Counter(fam))}, "strong_shots": strong,
           "metrics": metrics, "total_feedback_labels_fallback": total_labels_fb, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/carry_teacher_diagnostic_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_TEACHER_DIAGNOSTIC_V1 (strong-s0 ceiling vs warm receding vs warm+fallback) ==")
    log(f"  strong s0 ceiling : K6 {metrics['strong_s0_ceiling']['K6']} handoff {metrics['strong_s0_ceiling']['handoff']}")
    log(f"  warm receding     : K6 {metrics['warm_receding']['K6']} handoff {metrics['warm_receding']['handoff']} mean-labels {metrics['warm_receding']['mean_labels']} reasons {metrics['warm_receding']['abstain_reasons']}")
    log(f"  warm + fallback   : K6 {metrics['warm_plus_fallback']['K6']} handoff {metrics['warm_plus_fallback']['handoff']} mean-labels {metrics['warm_plus_fallback']['mean_labels']} fb {metrics['warm_plus_fallback']['mean_fallbacks']} reasons {metrics['warm_plus_fallback']['abstain_reasons']} | total labels {total_labels_fb}")
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_teacher_diagnostic_v1.json\nCARRY_TEACHER_DIAGNOSTIC_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
