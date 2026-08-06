"""§3 PHASE_GATED_RESIDUAL_BEHAVIOR_CONTRACT regression: the gated residual collector with residual exploration
DISABLED must reproduce the deployable identity (pi_0: HL 3/9, VAL 2/30, grasp 9/9, delivered {1011,1447,1568}), and
its transitions must store terminated AND truncated separately and be bit-identical to pi_0 at every gate-off step.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_behavior import base_action, collect_gated_residual, eval_gated_residual_identity, gated_composite_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/behavior_reg.json"
INIT = {1011, 1447, 1568}


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    hl = eval_gated_residual_identity(pi0, HEADLINE)
    vl = eval_gated_residual_identity(pi0, list(VALIDATION))
    # collect a small exploratory batch and verify contract fields + gate-off bit-identity
    trs = collect_gated_residual(pi0, list(range(6000, 6010)), explore=True)
    has_trunc = any(t["truncated"] for t in trs); has_term = any(t["terminated"] for t in trs)
    both_stored = all(("terminated" in t and "truncated" in t) for t in trs)
    # gate-off steps must be bit-identical to pi_0 despite exploration
    gate_off_identical = True
    for t in trs:
        if t["gate_mult_t"] == 0.0:
            b = base_action(pi0, t["obs_t"])
            if not np.array_equal(t["action_t"], gated_composite_action(b, 0.0, t["requested_delta"])):
                gate_off_identical = False; break
            if not np.allclose(t["action_t"], np.clip(b, -4, 4), atol=1e-6):
                gate_off_identical = False; break
    # residual applied only when gate active
    off_residual_zero = all(np.allclose(t["executed_residual"], 0, atol=1e-6) for t in trs if t["gate_mult_t"] == 0.0)
    reproduced = (hl["deliver"] == 3 and vl["deliver"] == 2 and hl["grasp"] == 9
                  and set(hl["delivered_seeds"]) == INIT)
    ok = reproduced and both_stored and gate_off_identical and off_residual_zero
    out = {"identity_headline": hl["deliver"], "identity_validation": vl["deliver"], "identity_grasp": hl["grasp"],
           "delivered": sorted(hl["delivered_seeds"]), "n_transitions": len(trs),
           "stores_terminated_and_truncated": both_stored, "has_terminated": has_term, "has_truncated": has_trunc,
           "gate_off_bit_identical_to_pi0": gate_off_identical, "gate_off_residual_zero": off_residual_zero,
           "verdict": "PHASE_GATED_RESIDUAL_BEHAVIOR_CONTRACT_PASS" if ok else "BEHAVIOR_CONTRACT_FAIL"}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"identity: HL {hl['deliver']}/9 VAL {vl['deliver']}/30 grasp {hl['grasp']}/9 delivered {sorted(hl['delivered_seeds'])}", flush=True)
    print(f"transitions {len(trs)} | term&trunc stored {both_stored} (term {has_term} trunc {has_trunc}) | "
          f"gate-off==pi_0 {gate_off_identical} | gate-off residual 0 {off_residual_zero}", flush=True)
    print(out["verdict"], flush=True); print("BEHAVIOR_REG_DONE", flush=True)


if __name__ == "__main__":
    main()
