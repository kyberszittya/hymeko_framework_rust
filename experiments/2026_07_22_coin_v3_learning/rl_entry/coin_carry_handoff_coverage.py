"""CARRY_HANDOFF_COVERAGE_V1 — Phase 3 of the hierarchical upstream extension. BEFORE building any carry actor/critic,
prove the achievability prerequisite: on held-out strict-0 carry states (contact_retention / transport / braking), does a
SUPPORT-BOUNDED carry action-sequence exist that produces a good settling handoff (→ K6 via the FROZEN pi_0)?

Freeze (Phase 1): pi_0, the strict/certificate contract, the settling result (SETTLING_SKILL_CONFIRMED: strict≥1 → K6
≈0.95), and the trust-region/safety gate are all held fixed. The settling critic stays SETTLING_VALUE and is NOT used
here. Phase 2 (handoff def) is encoded in `sequence_then_pi0`: a good handoff = the coin reaches strict≥1 (contained + slow)
and the frozen pi_0 then delivers (K6).

Controls / expert on the SAME support-bounded candidate class:
  * PI_0            — the frozen baseline (zero offset).
  * RANDOM          — best of N uniformly-random support-bounded carry sequences (needs no search).
  * CEM_EXPERT      — bounded CEM over carry sequences (exact simulator as EXPERT only, to demonstrate EXISTENCE).

Report separately: handoff-reachable rate and K6-reachable rate (coverage) for each, per family. No critic training.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_handoff import carry_cem, carry_random, sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
HORIZON = 120
SHOTS, LENGTH, ITERS, MAG_MAX = 24, 30, 4, 0.20      # stronger existence search for a fair coverage estimate
INIT_STD, ELITE_FRAC = 0.10, 0.25


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    want = 12 if smoke else 30
    shots = 8 if smoke else SHOTS

    log("[panel] scanning held-out strict-0 CARRY states (contact_retention / transport / braking)...")
    panel, comp, strict_hist = build_boundary_panel(pi0, range(6300, 8200), forbidden, want=want,
                                                    families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        rl, gate, _h, _rec = reconstruct_handoff(pi0, ls, horizon=360)
        templates.append((rl, gate)); fam.append(ls.family)
    n = len(templates)
    log(f"[panel] kept {n} held-out strict-0 carry states | families { {k: v for k, v in sorted(Counter(fam).items())} }")

    rows = []
    for i in range(n):
        rl0, gate0 = templates[i]
        p = sequence_then_pi0(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, np.zeros((0, adim), np.float32), horizon=HORIZON)
        rnd = carry_random(rl0, gate0, pi0, base, adim, np.random.default_rng(100 + i), shots=shots, length=LENGTH, mag_max=MAG_MAX, horizon=HORIZON)
        cem = carry_cem(rl0, gate0, pi0, base, adim, np.random.default_rng(200 + i), shots=shots, length=LENGTH, iters=(2 if smoke else ITERS),
                        init_std=INIT_STD, mag_max=MAG_MAX, elite_frac=ELITE_FRAC, horizon=HORIZON)
        rows.append({"family": fam[i], "pi0": p, "random": rnd, "cem": cem})
        if i % 10 == 0 or smoke:
            log(f"  [{i+1}/{n} {fam[i]:16}] pi_0 K6 {p['k6']} handoff {p['reached_handoff']} | random K6 {rnd['k6']} ho {rnd['reached_handoff']} | CEM K6 {cem['k6']} ho {cem['reached_handoff']}")

    def rate(sel, key):
        return round(float(np.mean([sel(r)[key] for r in rows])), 3) if rows else 0.0
    agg = {}
    for name, sel in (("pi_0", lambda r: r["pi0"]), ("random", lambda r: r["random"]), ("cem_expert", lambda r: r["cem"])):
        agg[name] = {"handoff_rate": rate(sel, "reached_handoff"), "k6_coverage": rate(sel, "k6"),
                     "mean_max_strict": round(float(np.mean([sel(r)["max_strict"] for r in rows])), 2)}
    per_fam = {}
    for f in sorted(set(fam)):
        fr = [r for r in rows if r["family"] == f]
        per_fam[f] = {"n": len(fr),
                      "pi0_K6": round(float(np.mean([r["pi0"]["k6"] for r in fr])), 3),
                      "cem_K6": round(float(np.mean([r["cem"]["k6"] for r in fr])), 3),
                      "cem_handoff": round(float(np.mean([r["cem"]["reached_handoff"] for r in fr])), 3)}

    cem_k6, pi0_k6, rnd_k6 = agg["cem_expert"]["k6_coverage"], agg["pi_0"]["k6_coverage"], agg["random"]["k6_coverage"]
    cem_ho = agg["cem_expert"]["handoff_rate"]
    if n < 8:
        verdict, nxt = "CARRY_COVERAGE_UNDERPOWERED", f"only {n} carry states — widen the scan"
    elif cem_k6 >= 0.5 and cem_k6 > pi0_k6 + 0.2:
        rnd_note = " (search needed: random ≪ CEM)" if rnd_k6 < cem_k6 - 0.2 else " (even random finds it)"
        verdict = "UPSTREAM_EXPERT_EXISTS_CARRY_COVERAGE_CONFIRMED"
        nxt = ("a support-bounded carry-prefix reliably produces a good handoff → K6" + rnd_note +
               "; the upstream expert EXISTS → Phase 4: distil its first actions into a carry actor (BC init → DAgger/"
               "receding-horizon labels → reward-driven RL), keeping the frozen settling pi_0 downstream")
    elif cem_ho >= 0.5 and cem_k6 < 0.5:
        verdict = "HANDOFF_REACHABLE_BUT_SETTLING_DOES_NOT_FINISH"
        nxt = "a bounded carry-prefix reaches strict≥1 but the frozen pi_0 does not then deliver — tighten the handoff quality criterion (settle deeper / slower) or lengthen the carry"
    elif cem_k6 >= 0.2 and cem_k6 > pi0_k6 + 0.1 and cem_k6 > rnd_k6:
        verdict = "PARTIAL_CARRY_COVERAGE_CANDIDATE_CLASS_WORKS_FOR_A_SUBSET"
        nxt = ("a bounded carry-prefix delivers end-to-end on a subset (CEM {cem} vs pi_0 {p}, and CEM>random so SEARCH "
               "helps — not just any perturbation); before distilling a carry actor, raise coverage by WIDENING the support "
               "(mag_max / carry length / horizon): if it rises → support-limited; if it plateaus → the candidate class "
               "(pi_0 + offset) is wrong for the carry phase and a carry-specific action structure is needed").format(
                   cem=round(cem_k6, 2), p=round(pi0_k6, 2))
    elif cem_ho < 0.3:
        verdict = "NO_CARRY_COVERAGE_SUPPORT_OR_HORIZON_LIMIT"
        nxt = "even the CEM expert rarely reaches a good handoff under the support bound — increase mag_max / carry length / horizon, or the candidate class is wrong for the carry phase"
    else:
        verdict, nxt = "CARRY_COVERAGE_INCONCLUSIVE", "CEM improves over pi_0 but below the 0.5 bar — widen search / states"

    out = {"contract": "CARRY_HANDOFF_COVERAGE_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "freeze": {"pi_0": "1902454c", "settling": "SETTLING_SKILL_CONFIRMED strict≥1→K6≈0.95", "settling_critic": "SETTLING_VALUE (unused here)"},
           "method": {"handoff": "strict≥1 (contained+slow) then FROZEN pi_0 delivers K6", "candidate": "support-bounded pi_0 + clipped offset seq",
                      "expert": "bounded CEM (existence only, not a learned policy)", "shots": shots, "length": LENGTH,
                      "iters": ITERS, "mag_max": MAG_MAX, "horizon": HORIZON, "controls": ["pi_0", "random"]},
           "panel": {"n": n, "families": {k: v for k, v in sorted(Counter(fam).items())}, "all_held_out": True, "strict": 0},
           "aggregate": agg, "per_family": per_fam, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/carry_handoff_coverage_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_HANDOFF_COVERAGE_V1 (strict-0 carry; existence via bounded CEM expert; frozen settling continuation) ==")
    log(f"  n={n} held-out strict-0 carry states | families { {k: v for k, v in sorted(Counter(fam).items())} }")
    for name in ("pi_0", "random", "cem_expert"):
        a = agg[name]
        log(f"  {name:11}: handoff-reachable {a['handoff_rate']} | K6-coverage {a['k6_coverage']} | mean max_strict {a['mean_max_strict']}")
    log("  per family (pi_0 K6 → CEM K6 / CEM handoff): " + " ".join(f"{f}[{per_fam[f]['pi0_K6']}→{per_fam[f]['cem_K6']}/{per_fam[f]['cem_handoff']}]" for f in per_fam))
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_handoff_coverage_v1.json\nCARRY_HANDOFF_COVERAGE_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
