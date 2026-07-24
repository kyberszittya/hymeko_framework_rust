"""Stage 2 — build a strong structured OPTION teacher bank (state → θ*) with provenance, robustness, and explicit ABSTAIN.

Teacher = the VALIDATED uniform structured random-shooting expert (CEM is not the default — random shooting outperformed
it). One canonical θ* per option-initiation state, ranked K6-first. A label is CONFIDENT only when the option actually
delivers K6 through the frozen pi_0 continuation (never a merely-least-bad candidate). Enough labels for a meaningful actor
fit — not a 20–30 toy set. Saves obs/θ (npz) + per-state provenance (json).
"""
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_option import option_teacher_label  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
SHOTS, TEACHER_H, ROBUST = 64, 160, 3


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def generate_bank(pi0, base, panel, *, shots, teacher_h=TEACHER_H, robust=ROBUST, transplant=None,
                  reconstruct_kwargs=None, log=print):
    """Label each option-initiation state with the strong structured expert; keep only CONFIDENT (K6-delivering) labels.
    ``transplant`` (a callable rl_clamp → rl_variant at the matched state) retargets the labels to a robot VARIANT at the
    matched canonical state (the B3 TRANSPLANT bank); ``None`` = the canonical clamp. ``reconstruct_kwargs`` (a dict passed
    to ``reconstruct_handoff``, e.g. ``geom``/``arm_mjcf_transform``/``disk_radius_override``) instead reconstructs the pi_0
    prefix DIRECTLY on the variant robot/object (the true FRESH-RECONSTRUCT deploy distribution) — the canonical hash
    verification is skipped (different physics) and only strict==0 carry handoffs are kept. Returns (obs, theta, provenance)."""
    fresh = reconstruct_kwargs is not None
    obs, theta, prov = [], [], []
    for i, ls in enumerate(panel):
        if not fresh:
            v = verify_reconstruction(pi0, ls)
            assert v["obs_ok"] and v["base_ok"] and v["gate_ok"]
        try:
            rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360, **(reconstruct_kwargs or {}))
        except ValueError:
            continue                                                    # variant terminated before the prefix — skip
        if fresh:
            if int(rl._strict) != 0:
                continue                                                # variant: keep only strict==0 carry handoffs
        else:
            assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY
            if transplant is not None:
                rl = transplant(rl)                                     # retarget to the ball-tip robot at the matched state
        th, confident, p = option_teacher_label(rl, gate, pi0, base, np.random.default_rng(1000 + i), shots=shots, horizon=teacher_h, robust_checks=robust)
        p.update({"seed": int(ls.seed), "prefix_steps": int(ls.prefix_steps), "family": ls.family})
        prov.append(p)
        if confident:
            obs.append(rl.obs().copy())
            theta.append(th)
        if i % 20 == 0:
            log(f"  [{i+1}/{len(panel)} {ls.family:16}] confident {confident} reason {p['termination_reason']} robust_k6 {p['robust_k6']} (labels {len(obs)})")
    return obs, theta, prov


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    want = 16 if smoke else 180
    shots = 16 if smoke else SHOTS

    log(f"[panel] scanning held-out TRAIN carry states (seeds 9000–10800) for the option teacher bank (want {want})...")
    panel, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=want, families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    obs, theta, prov = generate_bank(pi0, base, panel, shots=shots, teacher_h=TEACHER_H, robust=ROBUST, log=log)

    reasons = Counter(p["termination_reason"] for p in prov)
    fam_conf = Counter(p["family"] for p in prov if p["confident"])
    robusts = [p["robust_k6"] for p in prov if p["robust_k6"] is not None]
    stats = {"scanned": len(panel), "confident_labels": len(obs),
             "near_miss_handoff_only": reasons.get("HANDOFF_ONLY", 0), "abstain_no_handoff": reasons.get("NO_HANDOFF", 0),
             "confident_by_family": dict(fam_conf), "mean_robust_k6": round(float(np.mean(robusts)), 3) if robusts else None,
             "termination_reasons": dict(reasons)}
    if obs:
        np.savez(f"{D}/carry_option_teacher_bank_v1.npz", obs=np.asarray(obs, np.float32), theta=np.asarray(theta, np.float32))
    json.dump({"contract": "CARRY_OPTION_TEACHER_BANK_V1", "date": "2026-07-24", "smoke": smoke, "shots": shots,
               "teacher": "uniform structured random-shooting (validated expert); K6-primary canonical; robust jitter checks",
               "stats": stats, "provenance": prov}, open(f"{D}/carry_option_teacher_bank_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_TEACHER_BANK_V1 ==")
    log(f"  scanned {stats['scanned']} | CONFIDENT labels {stats['confident_labels']} | near-miss(handoff-only) {stats['near_miss_handoff_only']} | ABSTAIN(no-handoff) {stats['abstain_no_handoff']}")
    log(f"  confident by family {stats['confident_by_family']} | mean robust_k6 {stats['mean_robust_k6']} | reasons {stats['termination_reasons']}")
    log(f"  saved bank ({len(obs)} labels) → {D}/carry_option_teacher_bank_v1.npz\nCARRY_OPTION_TEACHER_BANK_DONE")
    return stats


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
