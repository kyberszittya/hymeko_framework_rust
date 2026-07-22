"""§12 actor-gradient-support audit: does the hard-clip actor's gradient flow on load-bearing transport/entry/settling
states (where the BC is unsaturated, da/dz=1), or is it gradient-locked by saturation? Uses the calibrated critic.
"""
import importlib.util
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_v3_receding_horizon import _phase_code
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import build_shared_sac_td3
from hymeko_rl.experiments.coin_neutral_start import neutral_env

_CC = "experiments/2026_07_22_coin_v3_learning/rl_entry/critic_calibration.py"
_spec = importlib.util.spec_from_file_location("cc", _CC)
cc = importlib.util.module_from_spec(_spec)
sys.argv = ["x", "/dev/null"]
_spec.loader.exec_module(cc)

PHN = ["APPROACH", "CONTACT_ACQ", "BILATERAL", "TRANSPORT", "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]
OUT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "/dev/null" else "/tmp/grad.json"


def collect_states_by_phase(bc, seeds):
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    states, phases = [], []
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        touched = False
        start = float(inner._planar_metrics.disk_to_zone)
        for _t in range(200):
            m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            states.append(nf)
            phases.append(_phase_code(inner, 0, touched, start))
            inner.step(np.asarray(bc.act(nf), np.float32))
    return np.asarray(states, np.float32), np.asarray(phases)


def main():
    bc = FullActionBC()
    bc.load_state_dict(torch.load(cc.BC))
    bc.eval()
    sac, td3 = build_shared_sac_td3(bc)
    trajs = cc.collect(bc)
    states, phases = collect_states_by_phase(bc, HEADLINE)
    S = torch.tensor(states, dtype=torch.float32)
    out = {}
    for name, actor, is_td3 in (("SAC", sac, False), ("TD3", td3, True)):
        q = cc.fit_critic(trajs, actor, td3=is_td3, steps=6000)
        # da/dz coverage: fraction of action dims with nonzero hard-clip Jacobian (|z|<4) on load-bearing states
        with torch.no_grad():
            z = actor.mu(actor.backbone(S)) if name == "SAC" else actor.head(actor.backbone(S))
            jac = (z.abs() < 4 - 1e-6).float()               # da/dz = 1 unsaturated, 0 saturated
        # actor objective gradient: L = -mean(min Q(s, actor(s))); dL/dθ_actor
        for p in actor.parameters():
            if p.grad is not None:
                p.grad = None
        a = actor.action_mean(S)
        q1, q2 = q(S, a)
        loss = -torch.min(q1, q2).mean()
        loss.backward()
        gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters() if p.grad is not None)))
        # per-phase gradient contribution (grad of -Q on only that phase's states)
        per_phase = {}
        for ph in (1, 3, 4, 5):                              # contact_acq, transport, entry, settling
            mask = phases == ph
            if mask.sum() < 3:
                continue
            for p in actor.parameters():
                if p.grad is not None:
                    p.grad = None
            aa = actor.action_mean(S[mask])
            (-torch.min(*q(S[mask], aa)).mean()).backward()
            g = float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters() if p.grad is not None)))
            per_phase[PHN[ph]] = round(g, 5)
        # dQ/da magnitude on load-bearing (transport/entry/settling) states
        lb = np.isin(phases, [3, 4, 5])
        Slb = S[lb].clone().requires_grad_(False)
        alb = actor.action_mean(Slb).detach().clone().requires_grad_(True)
        (torch.min(*q(Slb, alb)).sum()).backward()
        dqda = float(alb.grad.abs().mean())
        # predicted action change under one small update: |Δa| for a step on the load-bearing objective
        out[name] = {
            "actor_grad_norm_all": round(gnorm, 5),
            "per_phase_grad_norm": per_phase,
            "load_bearing_dQda_mean_abs": round(dqda, 5),
            "jacobian_nonzero_frac_transport": round(float(jac[phases == 3].mean()), 3) if (phases == 3).sum() else None,
            "jacobian_nonzero_frac_entry": round(float(jac[phases == 4].mean()), 3) if (phases == 4).sum() else None,
            "jacobian_nonzero_frac_settling": round(float(jac[phases == 5].mean()), 3) if (phases == 5).sum() else None,
            "finite_gradient": bool(np.isfinite(gnorm) and gnorm > 1e-6),
        }
        print(f"{name}: actor_grad_norm {gnorm:.4f} | per-phase {per_phase} | LB dQ/da {dqda:.4f} | "
              f"jac_nonzero transport {out[name]['jacobian_nonzero_frac_transport']} "
              f"settling {out[name]['jacobian_nonzero_frac_settling']}", flush=True)
    # gate: transport-or-later states contribute meaningful gradient; not solely approach/contact; finite nontrivial
    def ok(a):
        o = out[a]
        lb_grad = sum(o["per_phase_grad_norm"].get(p, 0) for p in ("TRANSPORT", "TARGET_ENTRY", "SETTLING"))
        return (o["finite_gradient"] and lb_grad > 1e-4 and o["load_bearing_dQda_mean_abs"] > 1e-4
                and (o["jacobian_nonzero_frac_transport"] or 0) > 0.5)
    passed = ok("SAC") and ok("TD3")
    verdict = "HARD_CLIP_ACTOR_GRADIENT_SUPPORT_PASS" if passed else "HARD_CLIP_ACTOR_GRADIENT_BLOCKED"
    out["verdict"] = verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
