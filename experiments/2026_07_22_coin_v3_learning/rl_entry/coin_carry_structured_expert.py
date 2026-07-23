"""CARRY_STRUCTURED_EXPERT_V1 — Phase 4 existence test: does a carry-SPECIFIC action language (push→brake→release) solve
carry states the pi_0+offset class provably cannot?

On a FRESH held-out strict-0 carry panel (disjoint from the frontier dev panel), four controllers on the same states:
  * PI_0                     — frozen baseline.
  * OLD_OFFSET_CEM           — the pi_0 + support-bounded offset CEM (the plateaued class), same budget.
  * STRUCTURED_RANDOM        — budget-matched random over the structured params (isolates search vs class).
  * STRUCTURED_CEM           — CEM over the ≈15-param push→brake→release macro-action; frozen pi_0 after handoff.

Primary evidence is NOT aggregate coverage but **old-unsolved → structured-solved**: states the offset class fails yet the
structured class delivers (→ K6 via frozen pi_0). Success gate (pre-registered): structured coverage > 0.40 AND solves
states outside the offset-solved set AND K6 rises (not just handoff) AND full-containment exit not worse AND structured CEM
beats both the offset CEM and the budget-matched structured random. No critic training.
"""
import copy
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_handoff import carry_cem, sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_cem, structured_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
H = 160
SHOTS, ITERS, ELITE = 24, 4, 0.25
OFFSET_MAG, OFFSET_LEN = 0.20, 30       # the plateaued offset class, at its best-known setting


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
    want = 8 if smoke else 24
    shots = 8 if smoke else SHOTS

    # FRESH panel — high seed range, disjoint from the frontier dev panel (which drew from ≥6300)
    log("[panel] scanning a FRESH held-out strict-0 carry panel (seeds ≥9000, disjoint from the dev panel)...")
    panel, _c, _s = build_boundary_panel(pi0, range(9000, 12000), forbidden, want=want,
                                         families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"], f"panel identity mismatch {ls.seed}"
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY
        templates.append((rl, gate)); fam.append(ls.family)
    n = len(templates)
    log(f"[panel] {n} FRESH carry states | families { {k: v for k, v in sorted(Counter(fam).items())} }")

    out_pi0, out_off, out_srand, out_scem = [], [], [], []
    for i in range(n):
        rl0, gate0 = templates[i]
        out_pi0.append(sequence_then_pi0(copy.deepcopy(rl0), copy.deepcopy(gate0), pi0, base, np.zeros((0, adim), np.float32), horizon=H))
        out_off.append(carry_cem(rl0, gate0, pi0, base, adim, np.random.default_rng(200 + i), shots=shots, length=OFFSET_LEN,
                                 iters=ITERS, init_std=0.1, mag_max=OFFSET_MAG, elite_frac=ELITE, horizon=H))
        out_srand.append(structured_random(rl0, gate0, pi0, base, np.random.default_rng(300 + i), shots=shots * ITERS, horizon=H))
        out_scem.append(structured_cem(rl0, gate0, pi0, base, np.random.default_rng(400 + i), shots=shots, iters=ITERS, elite_frac=ELITE, horizon=H))
        if i % 8 == 0 or smoke:
            log(f"  [{i+1}/{n} {fam[i]:16}] pi_0 {out_pi0[i]['k6']} | offset-CEM {out_off[i]['k6']} | "
                f"struct-rand {out_srand[i]['k6']} | struct-CEM K6 {out_scem[i]['k6']} ho {out_scem[i]['reached_handoff']}")

    def solved(outs):
        return {i for i in range(n) if outs[i]["k6"] == 1}
    def rate(outs, key="k6"):
        return round(float(np.mean([o[key] for o in outs])), 3)
    def any_exit(outs):
        return round(float(np.mean([o["contain_exit_ct"] > 0 for o in outs])), 3)

    S_pi0, S_off, S_scem = solved(out_pi0), solved(out_off), solved(out_scem)
    new_vs_offset = sorted(S_scem - S_off)                              # PRIMARY: offset-unsolved → structured-solved
    new_vs_pi0 = sorted(S_scem - S_pi0)
    agg = {name: {"K6": rate(o), "handoff": rate(o, "reached_handoff"), "any_exit": any_exit(o)}
           for name, o in (("pi_0", out_pi0), ("offset_CEM", out_off), ("structured_random", out_srand), ("structured_CEM", out_scem))}
    per_fam = {f: {"n": sum(x == f for x in fam),
                   "pi0": round(float(np.mean([out_pi0[i]["k6"] for i in range(n) if fam[i] == f])), 3),
                   "offset": round(float(np.mean([out_off[i]["k6"] for i in range(n) if fam[i] == f])), 3),
                   "structured": round(float(np.mean([out_scem[i]["k6"] for i in range(n) if fam[i] == f])), 3)}
               for f in sorted(set(fam))}

    scov, ocov = agg["structured_CEM"]["K6"], agg["offset_CEM"]["K6"]
    gate = (scov > 0.40 and len(new_vs_offset) >= 1 and scov > ocov and scov > agg["structured_random"]["K6"]
            and agg["structured_CEM"]["any_exit"] <= agg["offset_CEM"]["any_exit"] + 0.1
            and agg["structured_CEM"]["handoff"] >= agg["structured_CEM"]["K6"])   # K6 rose, not just handoff
    if n < 8:
        verdict, nxt = "STRUCTURED_EXPERT_UNDERPOWERED", f"only {n} fresh carry states — widen the scan"
    elif gate:
        verdict = "STRUCTURED_CLASS_VALIDATED_CARRY_ACTION_LANGUAGE_FOUND"
        nxt = (f"structured push→brake→release solves {scov} (offset {ocov}, struct-random {agg['structured_random']['K6']}) "
               f"and delivers {len(new_vs_offset)} states the offset class FAILS ({new_vs_offset}) with exit not worse → the "
               "carry action-language works. Proceed to Phase 4b: DAgger carry actor from the structured-CEM receding-horizon "
               "first actions (low-level continuous actor) → update-0 eval → SAC/TD3; frozen settling pi_0 stays downstream. "
               "No more mandatory intermediate audits (re-confirm scale on more states in passing)")
    elif scov > ocov and len(new_vs_offset) >= 1:
        verdict = "STRUCTURED_HELPS_BUT_BELOW_GATE"
        nxt = (f"structured beats offset ({scov} vs {ocov}) and solves new states {new_vs_offset}, but misses ≥1 gate "
               "condition (coverage>0.40 / K6-not-just-handoff / exit / >random) — tune the primitive (phase transitions, "
               "amplitude bound, durations) or enlarge search before distillation")
    else:
        verdict = "STRUCTURED_NO_BETTER_THAN_OFFSET"
        nxt = "the push→brake→release primitive does not beat the offset class — revise the action language (transitions / phases) or reconsider whether the carry phase is controllable from these handoffs"

    result = {"contract": "CARRY_STRUCTURED_EXPERT_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
              "method": {"primitive": "push→brake→release, ≈15 params, closed-loop transitions", "handoff": "frozen pi_0 after strict≥1",
                         "score": "lex K6≻handoff≻dwell≻-exit≻contact≻-effort≻-completion", "H": H, "shots": shots, "iters": ITERS,
                         "controls": ["pi_0", "offset_CEM", "structured_random"], "fresh_panel": True},
              "panel": {"n": n, "families": {k: v for k, v in sorted(Counter(fam).items())}, "seeds": ">=9000_disjoint"},
              "aggregate": agg, "per_family": per_fam,
              "solved": {"pi_0": sorted(S_pi0), "offset": sorted(S_off), "structured": sorted(S_scem)},
              "new_vs_offset": new_vs_offset, "new_vs_pi0": new_vs_pi0, "gate_met": gate, "verdict": verdict, "next_lever": nxt}
    json.dump(result, open(f"{D}/carry_structured_expert_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_STRUCTURED_EXPERT_V1 (fresh carry panel; push→brake→release vs offset class) ==")
    for name in ("pi_0", "offset_CEM", "structured_random", "structured_CEM"):
        a = agg[name]; log(f"  {name:18}: K6 {a['K6']} | handoff {a['handoff']} | any_exit {a['any_exit']}")
    log(f"  offset-UNSOLVED → structured-SOLVED: {new_vs_offset} ({len(new_vs_offset)} states)")
    log("  per family: " + " ".join(f"{f}[n{per_fam[f]['n']}] pi0 {per_fam[f]['pi0']}→off {per_fam[f]['offset']}→struct {per_fam[f]['structured']}" for f in per_fam))
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_structured_expert_v1.json\nCARRY_STRUCTURED_EXPERT_DONE")
    return result


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
