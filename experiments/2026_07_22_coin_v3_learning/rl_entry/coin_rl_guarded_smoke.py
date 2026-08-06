"""§10-11 guarded TD3 actor-entry smoke: FROZEN 3/9 actor + the AUDITED critic (recovery config D, TD3_CRITIC_FINAL_
AUDIT_PASS), synchronized targets, immutable broadened replay. Actor+critic co-train; guarded EARLY checkpoints
(0/1/2/5/10/25/50/100/500/1000) with actor/critic metrics + authorization panel; PAUSE on sep<=0 / grasp<8 / all-3-
successes-lost / OOD critic exploitation. No actor-LR change, no BC anchor, no rebalance.
"""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
_BASE = sys.argv[1]
sys.argv = ["x", _BASE, "/dev/null"]
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "cr", "/private/tmp/claude-501/-Users-kyberszittya-hakiko-ai-ws-03-implementation-hymeko-framework-rust/63ad1b54-314a-48f8-b561-ba4a163f847c/scratchpad/coin_rl_critic_recovery.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION  # noqa: E402
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC, eval_bc_delivery  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, build_shared_sac_td3  # noqa: E402

import os  # noqa: E402

GAMMA, TAU = 0.99, 0.005
CKPTS = [0, 1, 2, 5, 10, 25, 50, 100, 500, 1000]
OUT = os.environ.get("GS_OUT", "/tmp/gs.json")
INIT_SUCCESS = {1011, 1447, 1568}


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    bc = FullActionBC(); bc.load_state_dict(torch.load(cr.BC)); bc.eval()
    _sac, actor = build_shared_sac_td3(bc); actor_targ = copy.deepcopy(actor)
    # ---- build the audited critic (recovery config D, 10000 steps, seed 0) ----
    train_seeds = set(range(6000, 6080))
    demo = cr.trajs_with_returns(cr.replay_certified(_BASE, train_seeds))
    frozen = cr.trajs_with_returns([cr.roll("actor", s, bc, actor)[0] for s in range(6000, 6050)])
    fails = cr.trajs_with_returns([cr.roll(p, s, bc)[0] for s in range(6000, 6020) for p in ("zero", "away")])
    perturb = cr.trajs_with_returns([cr.roll("bc", s, bc, noise=0.4)[0] for s in range(6000, 6040)])
    D = demo + frozen + fails + perturb
    auth = cr.trajs_with_returns([cr.roll(p, s, bc)[0] for s in list(HEADLINE) + list(range(7000, 7015))
                                  for p in ("bc", "zero", "away")])
    q = cr.train_critic(D, actor_targ, 10000)             # the audited critic
    qt = cr.TwinQ(); qt.load_state_dict(q.state_dict())    # synchronized target critic
    m0 = cr.authorize(q, auth)
    print(f"audited critic (pre-actor): sep {m0['succ_fail_sep']:+.2f} spear {m0['spearman']:+.2f} rank {m0['rank_acc']}",
          flush=True)
    qopt = torch.optim.Adam(q.parameters(), lr=3e-4); aopt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    # immutable replay = D transitions (+ fresh appended)
    S, A, R, S2, Dn = cr.flat(D)
    ce = CoinRL4Dof(); co = ce.reset(int(rng.integers(1000, 100000)))
    panel = torch.tensor(np.concatenate([np.array([t[0] for t in tr["tr"]], np.float32) for tr in demo[:3]])[:400])
    a0 = actor.action_mean(panel).detach().clone()
    a0p = torch.cat([p.flatten() for p in actor.parameters()]).clone()
    recs = []; paused = None
    for upd in range(1001):
        if upd in CKPTS:
            hl = eval_bc_delivery(ActorEvalWrap(actor), HEADLINE); vl = eval_bc_delivery(ActorEvalWrap(actor), VALIDATION)
            mc = cr.authorize(q, auth)
            with torch.no_grad():
                ad = float((actor.action_mean(panel) - a0).abs().mean())
                pd = float((torch.cat([p.flatten() for p in actor.parameters()]) - a0p).abs().mean())
            rec = {"update": upd, "headline": hl["deliver"], "grasp": hl["grasp"], "validation": vl["deliver"],
                   "actor_out_delta": round(ad, 5), "actor_param_delta": round(pd, 6),
                   "critic_sep": mc["succ_fail_sep"], "critic_spear": mc["spearman"], "critic_ood_b": mc["ood_boundary"],
                   "delivered": hl["delivered_seeds"]}
            recs.append(rec)
            print(f"[gs] upd {upd}: HL {hl['deliver']}/9 grasp {hl['grasp']}/9 VAL {vl['deliver']}/30 | aΔ {ad:.4f} "
                  f"| critic sep {mc['succ_fail_sep']:+.1f} spear {mc['spearman']:+.2f} ood_b {mc['ood_boundary']}", flush=True)
            lost_all = (not set(hl["delivered_seeds"]) & INIT_SUCCESS)
            if mc["succ_fail_sep"] <= 0:
                paused = "critic_sep_nonpositive"
            elif hl["grasp"] < 8:
                paused = "grasp_below_8"
            elif lost_all and upd >= 5:
                paused = "all_initial_successes_lost"
            elif mc["ood_boundary"] > 0.5:
                paused = "critic_ood_exploitation"
            if paused:
                print(f"PAUSE: {paused} at upd {upd}", flush=True); break
        if upd == 1000:
            break
        with torch.no_grad():
            act = (actor.action_mean(torch.tensor(co[None]))[0].numpy() + rng.normal(0, 0.3, 4)).clip(-4, 4).astype(np.float32)
        co2, r, term, trunc, _ = ce.step(act)
        S = np.vstack([S, co[None]]); A = np.vstack([A, act[None]]); R = np.append(R, r)
        S2 = np.vstack([S2, co2[None]]); Dn = np.append(Dn, float(term))
        co = ce.reset(int(rng.integers(1000, 100000))) if (term or trunc) else co2
        idx = rng.integers(0, len(S), 256)
        s = torch.tensor(S[idx]); ac = torch.tensor(A[idx]); rw = torch.tensor(R[idx])
        s2 = torch.tensor(S2[idx]); d = torch.tensor(Dn[idx])
        with torch.no_grad():
            a2 = (actor_targ.action_mean(s2) + (0.2 * torch.randn(256, 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
            y = rw + GAMMA * (1 - d) * torch.min(*qt(s2, a2))
        qopt.zero_grad(); (((q(s, ac)[0] - y) ** 2 + (q(s, ac)[1] - y) ** 2).mean()).backward(); qopt.step()
        if upd % 2 == 0:
            aopt.zero_grad(); (-q(s, actor.action_mean(s))[0].mean()).backward(); aopt.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_targ.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    base = recs[0]; best = max(recs, key=lambda r: (r["validation"], r["headline"]))
    out = {"audited_critic_pre": m0, "init_headline": base["headline"], "init_validation": base["validation"],
           "best_headline": best["headline"], "best_validation": best["validation"], "best_update": best["update"],
           "paused": paused, "improved_validation": best["validation"] > base["validation"],
           "improved_headline": best["headline"] > base["headline"], "checkpoints": recs}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\ninit HL {base['headline']} VAL {base['validation']} -> best HL {best['headline']} VAL {best['validation']} "
          f"@upd {best['update']} | paused={paused} | improved_val={out['improved_validation']}\nGUARDED_DONE", flush=True)


if __name__ == "__main__":
    main()
