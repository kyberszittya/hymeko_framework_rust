"""§3 — reverify and PRESERVE the already-built controller gates before adding critic infrastructure.

Emits PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED + EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS evidence:
  - update-0 composite (zero-init residual) neutral-reset: headline 3/9, validation 2/30, grasp 9/9, delivered {1011,1447,1568}
  - forced-residual gate=0 leakage: executed action bit-identical to pi_0 for +0.25/-0.25/random/saturated
  - pi_0 parameter + output fingerprints unchanged
Stops with CONTROLLER_REGRESSION_BLOCKED if any accepted gate no longer reproduces.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_behavior import base_action, eval_gated_residual_identity, gated_composite_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "experiments/2026_07_22_coin_v3_learning/rl_entry/s3_controller_gates.json"
EXPECT = {"headline_grasp": 9, "headline_deliver": 3, "headline_delivered": [1011, 1447, 1568],
          "validation_deliver": 2}


def param_hash(pi0):
    h = hashlib.sha256()
    for _n, p in sorted(pi0.named_parameters()):
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()[:12]


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    h0 = param_hash(pi0)
    probe = torch.tensor(np.random.default_rng(0).standard_normal((16, 48)), dtype=torch.float32)
    with torch.no_grad():
        out0 = hashlib.sha256(pi0.action_mean(probe).numpy().tobytes()).hexdigest()[:12]

    head = eval_gated_residual_identity(pi0, HEADLINE)
    val = eval_gated_residual_identity(pi0, VALIDATION)

    rng = np.random.default_rng(3); maxdiff = 0.0; probes = 0
    for force in ("+0.25", "-0.25", "random", "saturated"):
        for _ in range(200):
            o = rng.standard_normal(48).astype(np.float32); b = base_action(pi0, o)
            d = {"+0.25": np.full(4, 0.25, np.float32), "-0.25": np.full(4, -0.25, np.float32),
                 "random": rng.uniform(-0.25, 0.25, 4).astype(np.float32),
                 "saturated": np.clip(np.full(4, 1e6, np.float32), -0.25, 0.25)}[force]
            comp0 = gated_composite_action(b, 0.0, d)
            maxdiff = max(maxdiff, float(np.max(np.abs(comp0 - np.clip(b, -4, 4))))); probes += 1

    h1 = param_hash(pi0)
    with torch.no_grad():
        out1 = hashlib.sha256(pi0.action_mean(probe).numpy().tobytes()).hexdigest()[:12]

    res = {"pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
           "headline": {"grasp": head["grasp"], "deliver": head["deliver"], "n": head["n"],
                        "delivered": sorted(head["delivered_seeds"])},
           "validation": {"grasp": val["grasp"], "deliver": val["deliver"], "n": val["n"]},
           "forced_residual_gate0_max_leakage": maxdiff, "leakage_probes": probes,
           "pi0_param_hash": [h0, h1], "pi0_output_fp": [out0, out1]}
    checks = {
        "headline_grasp_9": head["grasp"] == EXPECT["headline_grasp"],
        "headline_deliver_3": head["deliver"] == EXPECT["headline_deliver"],
        "headline_delivered_set": sorted(head["delivered_seeds"]) == EXPECT["headline_delivered"],
        "validation_deliver_2": val["deliver"] == EXPECT["validation_deliver"],
        "gate0_leakage_zero": maxdiff == 0.0,
        "pi0_params_unchanged": h0 == h1, "pi0_output_unchanged": out0 == out1,
    }
    res["checks"] = checks
    res["verdict"] = ("PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED+EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS"
                      if all(checks.values()) else "CONTROLLER_REGRESSION_BLOCKED")
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(res["verdict"]); print("wrote", OUT)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
