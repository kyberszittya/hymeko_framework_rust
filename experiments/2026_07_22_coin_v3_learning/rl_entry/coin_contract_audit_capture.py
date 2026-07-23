"""COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1 — capture per-step certificate streams + canonical v3 reward for ONE
controller across the 31 dev handoffs (read-only; controllers/pi_0/reward/certifier unchanged). One controller per
process (serial) to avoid the BLAS-oversubscription that destabilised mp.Pool for this workload."""
import argparse
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_contract_audit import decompose_reward  # noqa: E402
from hymeko_rl.coin_delivery.coin_start_id import start_id  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
TDCFG = f"{D}/transport_dwell_config.json"
CEM = dict(horizon=30, pop=40, iters=6, elite=8)


def _cs_dict(cs):
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else bool(v)) for k, v in asdict(cs).items()}


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _pi0_a(pi0, obs):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0].numpy(), -4, 4).astype(np.float32)


def _make_policy(name, pi0):
    if name == "pi0":
        return lambda rl, step: _pi0_a(pi0, rl.obs())
    if name == "h30":
        from hymeko_rl.coin_delivery.coin_v3_receding_horizon import plan_first_action
        from hymeko_rl.experiments.coin_neutral_start import _clearance
        return lambda rl, step: plan_first_action(rl.inner, rl.cf, _clearance(rl.inner), bool(rl._touched),
                                                  int(rl._strict), step, **CEM)[0]
    if name == "repaired":
        from hymeko_rl.coin_delivery.coin_planner_repair import FeasibilityConfig, plan_first_action_repaired
        from hymeko_rl.experiments.coin_neutral_start import _clearance
        cfg = FeasibilityConfig(boundary="stable_entry", contact_floor=0.75)
        return lambda rl, step: plan_first_action_repaired(rl.inner, rl.cf, _clearance(rl.inner), bool(rl._touched),
                                                           int(rl._strict), step, cfg=cfg, **CEM)[0]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--controller", required=True); args = ap.parse_args()
    torch.set_num_threads(1); log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True); horizon = cfg["horizon"]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    policy = _make_policy(args.controller, pi0)
    rl_clr = CoinRL4Dof()                                          # measure the exact canonical initial clearance per seed
    log(f"[{time.strftime('%H:%M:%S')}] capture {args.controller}: {len(dev)} dev states")
    t = time.time(); out = []; max_decomp_err = 0.0
    for i, ls in enumerate(dev):
        rl_clr.reset(int(ls.seed)); clearance = float(_clearance(rl_clr.inner))     # canonical start clearance (measured)
        rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        terms = rl.inner.reward_spec.terms; transitions = []
        handoff_strict = int(rl._strict)                          # dwell carried from the pi_0 PREFIX replay (load-bearing)
        for step in range(horizon):
            init_cs = _cert_step(rl.inner, rl.cf)                  # certificate at state_t (pre-action)
            a = np.asarray(policy(rl, step), np.float32)
            _o, rw, term, trunc, _ = rl.step(a)                   # -> state_{t+1}
            comps = decompose_reward(rl.inner, rl._dtz(), rl.inner.data.ctrl, terms)   # sum == rw (err 0); PBRS memory is
            max_decomp_err = max(max_decomp_err, abs(sum(comps.values()) - float(rw)))  # left post-step (idempotent for next)
            post_cs = _cert_step(rl.inner, rl.cf)                  # certificate at state_{t+1} (post-action; terminal kept)
            transitions.append({"init": _cs_dict(init_cs), "action": [float(x) for x in a], "reward": float(rw),
                                "components": {k: round(v, 6) for k, v in comps.items()}, "post": _cs_dict(post_cs),
                                "env_strict": int(rl._strict),     # env _strict counter (carries prefix dwell) = arc-canonical
                                "env_touched": bool(rl._touched)})  # rl._touched = ANY planar contact (matches RolloutTrace)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            if term or trunc:
                break
        out.append({"start_id": start_id(ls), "seed": int(ls.seed), "family": ls.family, "prefix": int(ls.prefix_steps),
                    "handoff_strict": handoff_strict, "clearance_measured": clearance, "transitions": transitions})
        if (i + 1) % 8 == 0:
            log(f"    {i+1}/{len(dev)} ({time.time()-t:.0f}s)")
    assert max_decomp_err < 1e-4, f"reward decomposition mismatch {max_decomp_err}"
    json.dump({"controller": args.controller, "reward_terms": [[k, w] for k, w in terms],
               "max_decomp_error": max_decomp_err, "rollouts": out},
              open(f"{D}/audit_trace_{args.controller}.json", "w"), default=float)
    log(f"  ({time.time()-t:.0f}s) decomp_err {max_decomp_err:.2e}; wrote audit_trace_{args.controller}.json\nCAPTURE_{args.controller}_DONE")


if __name__ == "__main__":
    main()
