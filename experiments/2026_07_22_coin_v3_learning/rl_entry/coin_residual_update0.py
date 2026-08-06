"""§3 PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED + §4 EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS (rollout side).

Load the immutable frozen pi_0 (file-SHA 1902454c), a zero-init residual, and STABLE_OBJECT_ENGAGEMENT_V1. Prove the
full composite+gate pipeline reproduces pi_0 exactly at update 0 (3/9 headline, 2/30 validation, 9/9 grasp, delivered
{1011,1447,1568}), and that composite==base for BOTH gate states on a state panel, with pi_0 frozen and unchanged.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_controller import (  # noqa: E402
    CompositeResidualController,
    ZeroInitResidualActor,
    assert_frozen_base,
)
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION  # noqa: E402
from hymeko_rl.coin_delivery.full_action_bc import eval_bc_delivery  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/update0.json"
INIT_SUCCESS = {1011, 1447, 1568}


def eval_composite(ctrl, seeds, horizon=360):
    """Roll the composite+gate from NEUTRAL; grade strict K=6 delivery. Gate advanced from deployable signals."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    fc = grasp = deliv = 0; per = []
    for s in seeds:
        env.set_stage(0); env.reset(seed=int(s))
        gate = StableEngagementGate(StableEngagementConfig()); gate.reset()
        cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        touched = False
        for _t in range(horizon):
            cert.update(_cert_step(inner, cf))
            m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            if cert.delivery_certified:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            a = ctrl.act(nf, gate.gate)                                  # composite with current gate multiplier
            inner.step(a)
            lc, rc, coin, ltip, rtip = stable_engagement_signals(inner)
            gate.update(lc, rc, coin, ltip, rtip, terminated=bool(cert.delivery_certified))
        d = bool(cert.delivery_certified)
        fc += int(touched); grasp += int(touched); deliv += int(d); per.append((int(s), d))
    n = max(1, len(list(seeds)))
    return {"n": n, "first_contact": fc, "grasp": grasp, "deliver": deliv,
            "delivered_seeds": [s for s, dd in per if dd]}


def main():
    file_sha = hashlib.sha256(open(PI0, "rb").read()).hexdigest()
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    assert_frozen_base(pi0)
    pre_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in pi0.parameters())).hexdigest()
    residual = ZeroInitResidualActor()
    ctrl = CompositeResidualController(pi0, residual)

    # composite == base for BOTH gate states on a rollout state panel (update 0, residual==0)
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    env.set_stage(0); env.reset(seed=1011)
    states = []
    for _ in range(200):
        nf = np.asarray(inner.node_features(), np.float32).flatten(); states.append(nf)
        inner.step(np.asarray(pi0.action_mean(torch.tensor(nf[None]))[0].numpy(), np.float32))
    ob = torch.tensor(np.asarray(states, np.float32))
    with torch.no_grad():
        base = ctrl.base_action(ob).numpy()
        comp_g0 = ctrl.composite_action(ob, 0.0).numpy()
        comp_g1 = ctrl.composite_action(ob, 1.0).numpy()
    maxdiff_g0 = float(np.abs(comp_g0 - base).max())
    maxdiff_g1 = float(np.abs(comp_g1 - base).max())

    # full composite+gate rollout delivery must reproduce pi_0
    comp_hl = eval_composite(ctrl, HEADLINE)
    comp_vl = eval_composite(ctrl, list(VALIDATION))
    pi0_hl = eval_bc_delivery(ActorEvalWrap(pi0), HEADLINE)
    pi0_vl = eval_bc_delivery(ActorEvalWrap(pi0), list(VALIDATION))

    post_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in pi0.parameters())).hexdigest()
    delivered = sorted(comp_hl["delivered_seeds"])
    reproduced = (comp_hl["deliver"] == 3 and comp_vl["deliver"] == 2 and comp_hl["grasp"] == 9
                  and set(delivered) == INIT_SUCCESS and maxdiff_g0 == 0.0 and maxdiff_g1 < 1e-6
                  and comp_hl["deliver"] == pi0_hl["deliver"] and comp_vl["deliver"] == pi0_vl["deliver"]
                  and pre_hash == post_hash)
    out = {"pi0_file_sha_prefix": file_sha[:8], "residual_contract_sha": residual.contract_sha256()[:12],
           "residual_contract": residual.contract(),
           "composite_vs_base_maxdiff": {"gate0": maxdiff_g0, "gate1": maxdiff_g1},
           "composite_gate_rollout": {"headline": comp_hl["deliver"], "validation": comp_vl["deliver"],
                                      "grasp": comp_hl["grasp"], "delivered": delivered},
           "pi0_direct": {"headline": pi0_hl["deliver"], "validation": pi0_vl["deliver"], "grasp": pi0_hl["grasp"]},
           "pi0_param_hash_unchanged": pre_hash == post_hash, "pi0_param_hash": pre_hash[:12],
           "verdict": "PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED" if reproduced else "UPDATE0_REPRODUCTION_FAILED"}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"pi0 file SHA {file_sha[:8]} | residual SHA {residual.contract_sha256()[:12]}", flush=True)
    print(f"composite vs base maxdiff: gate0={maxdiff_g0} gate1={maxdiff_g1:.2e}", flush=True)
    print(f"composite+gate rollout: HL {comp_hl['deliver']}/9 VAL {comp_vl['deliver']}/30 grasp {comp_hl['grasp']}/9 "
          f"delivered {delivered}", flush=True)
    print(f"pi_0 direct: HL {pi0_hl['deliver']}/9 VAL {pi0_vl['deliver']}/30 grasp {pi0_hl['grasp']}/9", flush=True)
    print(f"pi_0 param hash unchanged: {pre_hash == post_hash}", flush=True)
    print(out["verdict"], flush=True); print("UPDATE0_DONE", flush=True)


if __name__ == "__main__":
    main()
