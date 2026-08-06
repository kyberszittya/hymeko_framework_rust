"""Update-0 reproduction for PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1: with pi_late an EXACT copy of pi_0, the composed
controller must reproduce the frozen neutral-reset result (headline 3/9, validation 2/30, grasp 9/9, delivered
{1011,1447,1568}) and leak nothing at gate-off. Also verifies deterministic replay-to-handoff reconstruction. No training.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_phase_switched_late import (  # noqa: E402
    PhaseSwitchedController,
    assert_late_is_pi0_copy,
    make_late_actor_from_pi0,
)
from hymeko_rl.coin_delivery.coin_stable_engagement import (  # noqa: E402
    StableEngagementConfig,
    StableEngagementGate,
    stable_engagement_signals,
)
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "experiments/2026_07_22_coin_v3_learning/rl_entry/phase_switched_update0.json"


def eval_phase_switched(ctrl, seeds, horizon=360):
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    grasp = deliv = 0; delivered = []
    for s in seeds:
        env.set_stage(0); env.reset(seed=int(s))
        gate = StableEngagementGate(StableEngagementConfig()); cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        touched = False
        for _t in range(horizon):
            cert.update(_cert_step(inner, cf)); m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            if cert.delivery_certified:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            a = ctrl.act(nf, gate.gate)                                   # phase-switched: gate-off pi_0, gate-on pi_late
            inner.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(cert.delivery_certified))
        d = bool(cert.delivery_certified); grasp += int(touched); deliv += int(d)
        if d:
            delivered.append(int(s))
    return {"n": len(list(seeds)), "grasp": grasp, "deliver": deliv, "delivered": sorted(delivered)}


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    pi_late = make_late_actor_from_pi0(pi0, trainable=True)
    assert_late_is_pi0_copy(pi0, pi_late, tol=0.0)
    ctrl = PhaseSwitchedController(pi0, pi_late)

    head = eval_phase_switched(ctrl, HEADLINE); val = eval_phase_switched(ctrl, VALIDATION)

    # gate-off leakage: perturb pi_late; every gate-off action must still equal pi_0 exactly
    late2 = make_late_actor_from_pi0(pi0, trainable=True)
    with torch.no_grad():
        for p in late2.parameters():
            p.add_(2.0)
    ctrl2 = PhaseSwitchedController(pi0, late2)
    rng = np.random.default_rng(0); maxdiff = 0.0
    for _ in range(400):
        o = rng.standard_normal(48).astype(np.float32)
        off = ctrl2.act(o, 0.0)
        base = np.clip(pi0.action_mean(torch.as_tensor(o[None]))[0].detach().numpy(), -4, 4)
        maxdiff = max(maxdiff, float(np.max(np.abs(off - base))))

    # deterministic replay-to-handoff reconstruction check
    bank = build_late_start_bank(pi0, range(6100, 6130), per_family=1)
    recon = [verify_reconstruction(pi0, ls) for ls in bank]
    recon_ok = all(r["obs_ok"] and r["base_ok"] and r["causal_ok"] and r["gate_ok"] and r["gate_active"] for r in recon)

    checks = {
        "headline_grasp_9": head["grasp"] == 9, "headline_deliver_3": head["deliver"] == 3,
        "headline_delivered_set": head["delivered"] == [1011, 1447, 1568],
        "validation_deliver_2": val["deliver"] == 2,
        "gate_off_leakage_zero": maxdiff == 0.0,
        "late_is_pi0_copy": True, "reconstruction_exact": recon_ok,
    }
    res = {"pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
           "headline": head, "validation": val, "gate_off_max_leakage": maxdiff,
           "late_start_bank_n": len(bank), "reconstruction_verified": len(recon), "reconstruction_ok": recon_ok,
           "checks": checks,
           "verdict": "PHASE_SWITCHED_UPDATE0_REPRODUCED" if all(checks.values()) else "PHASE_SWITCHED_UPDATE0_REGRESSION"}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(res["verdict"], "| wrote", OUT)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
