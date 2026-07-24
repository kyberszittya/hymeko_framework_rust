"""OBJECT_TO_TARGET_VARIANTS_V1 — O1: disk-size ladder on the frozen collision-on ball-tip embodiment.

Keeps the ball-tip robot FIXED (BALLTIP_COIN_BASELINE_V1) and varies ONLY the manipuland size (cylinder radius), which
changes mass/inertia/braking but NOT the contact topology. Per size, runs the core baseline ladder:
  * full structured search EXPERT (192-shot) — the achievable ceiling on that object;
  * frozen ball coin proposal ZERO-SHOT + b=8 — the deployed controller's transfer to the new size.
States are reconstructed PER OBJECT (pi_0 prefix replayed on the variant coin; strict==0 carry handoffs kept) — not
transplanted, since a bigger coin at a canonical fingertip contact would interpenetrate. Frozen scene/robot/reward
untouched (disk_radius_override on the EnvSpec only). This is O1 of the geometry matrix (O2 square/rect, O3 triangle,
O4 ring follow); per the plan it STOPS after the bounded matrix — no RL here (the stricter RL gate is a later stage).
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_random_best_with_support  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import BALLTIP_SPEC, build_arm_mjcf  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_grasp_env import with_fingertip_sites  # noqa: E402

OUT = "reports/2026-07-24-object-variants-o1"
BALL_PROP = f"{D}/carry_proposal_balltip_v1.pt"
FAMS, EVAL_H, EXPERT_SHOTS = ("contact_retention", "transport", "braking"), 160, 192
SIZES = [("small_0.014", 0.014), ("canonical_0.020", 0.020), ("large_0.028", 0.028)]   # canonical coin radius = 0.020


def _ball_tf(_canon):
    return with_fingertip_sites(build_arm_mjcf(BALLTIP_SPEC, "enabled"))                # collision-on ball (no filter)


def variant_panel(pi0, cand_ls, size, want, log):
    """Reconstruct handoff states on the ball-tip + variant-size coin (reuse canonical (seed,prefix) candidates; keep the
    strict==0 carry handoffs that survive the pi_0 prefix on the variant object)."""
    panel, tried = [], 0
    for ls in cand_ls:
        tried += 1
        try:
            rl, gate, _h, _r = reconstruct_handoff(pi0, ls, geom="POINT", arm_mjcf_transform=_ball_tf, disk_radius_override=size)
        except ValueError:
            continue                                                                    # terminated before prefix on this object
        if int(rl._strict) == 0:
            panel.append((rl, gate))
        if len(panel) >= want:
            break
    log(f"    [{size}] reconstructed {len(panel)} strict==0 handoffs (from {tried} candidates)")
    return panel


def eval_size(pi0, base, prop, panel):
    prop_k6, exp_k6 = 0, 0
    for i, (rl, gate) in enumerate(panel):
        rng = np.random.default_rng(9000 + i)
        _t, o_up = search_select(rl, gate, prop.theta(rl.obs()), pi0, base, np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
        _t2, o_ex, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, rng, shots=EXPERT_SHOTS, horizon=EVAL_H)
        prop_k6 += int(o_up["k6"])
        exp_k6 += int(o_ex["k6"])
    n = max(1, len(panel))
    return {"n": len(panel), "proposal_zeroshot_b8": prop_k6, "proposal_rate": round(prop_k6 / n, 3),
            "expert_ceiling": exp_k6, "expert_rate": round(exp_k6 / n, 3)}


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
    prop = load_proposal(BALL_PROP)
    want = 6 if smoke else 16
    # a generous pool of canonical LateStarts (many won't survive the pi_0 prefix on off-size objects → over-provision)
    raw_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 40),
                                          families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    log(f"[O1] {len(raw_ls)} canonical LateStart candidates | ball-tip fixed | sizes {[s for s, _ in SIZES]} | want {want}/size")

    results = {}
    for name, size in ([SIZES[1]] if smoke else SIZES):
        panel = variant_panel(pi0, raw_ls, size, want, log)
        if not panel:
            results[name] = {"error": "no_strict0_handoffs", "size": size}
            continue
        r = eval_size(pi0, base, prop, panel)
        r["size"] = size
        results[name] = r
        log(f"  [{name}] n {r['n']} | ball proposal zero-shot b8 {r['proposal_zeroshot_b8']}/{r['n']} ({r['proposal_rate']}) "
            f"| expert ceiling {r['expert_ceiling']}/{r['n']} ({r['expert_rate']})")

    json.dump({"contract": "OBJECT_TO_TARGET_VARIANTS_V1", "stage": "O1_disk_size", "date": "2026-07-24", "smoke": smoke,
               "embodiment": "collision-on ball-tip (frozen BALLTIP_COIN_BASELINE_V1)", "canonical_radius": 0.020,
               "proposal": BALL_PROP.split("/")[-1], "results": results}, open(f"{OUT}/o1_disk_size.json", "w"), indent=1, default=float)

    log("\n== OBJECT_TO_TARGET_VARIANTS_V1 — O1 disk-size ladder (ball-tip fixed) ==")
    log(f"  {'size':16} {'n':>3} {'proposal_b8':>12} {'expert_ceiling':>15}")
    for name, r in results.items():
        if "error" in r:
            log(f"  {name:16} {r['error']}")
        else:
            log(f"  {name:16} {r['n']:>3} {r['proposal_zeroshot_b8']:>6}/{r['n']:<3} ({r['proposal_rate']:>5}) {r['expert_ceiling']:>6}/{r['n']:<3} ({r['expert_rate']:>5})")
    log(f"\n  artifact: {OUT}/o1_disk_size.json\nOBJECT_O1_DONE")
    return results


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
