"""§8 bounded SAC/TD3 smoke from the frozen 3/9 clip-actor init, canonical 4-DoF reward env. Reuses the validated
Bellman critic (CRITIC_CALIBRATION_PASS) + clip actors + reward env. Actor updates authorized. Checkpoints at
0/1/10/100/1000/5000/10000 updates; standalone headline+validation eval + reward-driven-learning metrics at each.
No privileged resets / curriculum / planner / scripted action. SAC=stochastic+entropy; TD3=deterministic+smoothing+delay.
"""
import argparse
import copy
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, build_shared_sac_td3

GAMMA, TAU = 0.99, 0.005
CKPTS = [0, 1, 10, 100, 1000, 5000, 10000]
BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"


def qnet():
    return nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = qnet(), qnet()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def bc_rollout(bc, seed, policy="bc"):
    e = CoinRL4Dof()
    o = e.reset(seed)
    tr = []
    for _t in range(360):
        a = (np.zeros(4, np.float32) if policy == "zero"
             else np.full(4, 3.0, np.float32) if policy == "away" else bc.act(o).astype(np.float32))
        a = np.clip(a, -4, 4)
        o2, r, term, trunc, _ = e.step(a)
        tr.append((o.copy(), a.copy(), float(r), o2.copy(), float(term)))
        o = o2
        if term or trunc:
            break
    return tr


def eval_actor(actor, seeds):
    from hymeko_rl.coin_delivery.full_action_bc import eval_bc_delivery
    return eval_bc_delivery(ActorEvalWrap(actor), seeds)


class Replay:
    def __init__(self, cap=200000):
        self.s = np.zeros((cap, 48), np.float32); self.a = np.zeros((cap, 4), np.float32)
        self.r = np.zeros(cap, np.float32); self.s2 = np.zeros((cap, 48), np.float32); self.d = np.zeros(cap, np.float32)
        self.i = 0; self.n = 0; self.cap = cap; self.n_demo = 0

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b, rng):
        idx = rng.integers(0, self.n, b)
        return (torch.tensor(self.s[idx]), torch.tensor(self.a[idx]), torch.tensor(self.r[idx]),
                torch.tensor(self.s2[idx]), torch.tensor(self.d[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "td3"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    bc = FullActionBC(); bc.load_state_dict(torch.load(BC)); bc.eval()
    sac_a, td3_a = build_shared_sac_td3(bc)
    with torch.no_grad():
        sac_a.log_std.bias.fill_(-2.4)          # SAC stochastic init at the TARGET entropy (-4 for 4-D); the frozen
        #                                         DETERMINISTIC action_mean (= the 3/9 reproduction) is unchanged.
    actor = sac_a if a.algo == "sac" else td3_a
    actor_targ = copy.deepcopy(actor)
    q, qt = TwinQ(), TwinQ(); qt.load_state_dict(q.state_dict())
    qopt = torch.optim.Adam(q.parameters(), lr=3e-4)
    aopt = torch.optim.Adam([p for n, p in actor.named_parameters() if "log_std" not in n or a.algo == "sac"], lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True); alopt = torch.optim.Adam([log_alpha], lr=3e-4); tgt_ent = -4.0
    # initial replay = BC demo transitions on the train_query pilot (same for both algos) + a few failures
    rb = Replay()
    init_seeds = list(range(6000, 6030))
    for s in init_seeds:
        for t in bc_rollout(bc, s):
            rb.add(*t)
    for s in init_seeds[:6]:
        for pol in ("zero", "away"):
            for t in bc_rollout(bc, s, pol):
                rb.add(*t)
    rb.n_demo = rb.n
    replay_sha = hashlib.sha256(rb.s[:rb.n].tobytes() + rb.a[:rb.n].tobytes()).hexdigest()[:16]
    # frozen state panel for actor-delta
    panel = torch.tensor(np.array([t[0] for s in HEADLINE[:3] for t in bc_rollout(bc, s)][:400]), dtype=torch.float32)
    with torch.no_grad():
        panel_a0 = actor.action_mean(panel).clone()
    a0_params = torch.cat([p.flatten() for p in actor.parameters()]).clone()
    # CRITIC WARMUP: fit the twin-Q on the demo replay with the FROZEN actor BEFORE any actor update (the actor learns
    # against a CALIBRATED critic per CRITIC_CALIBRATION_PASS, not a random one). No actor step here.
    warmup = 6000
    for _w in range(warmup):
        s, ac, rw, s2, d = rb.sample(256, rng)
        with torch.no_grad():
            if a.algo == "sac":
                a2, logp2 = actor.sample(s2); q1t, q2t = qt(s2, a2)
                y = rw + GAMMA * (1 - d) * (torch.min(q1t, q2t) - log_alpha.exp() * logp2)
            else:
                a2 = (actor_targ.action_mean(s2) + (0.2 * torch.randn(s2.shape[0], 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
                q1t, q2t = qt(s2, a2)
                y = rw + GAMMA * (1 - d) * torch.min(q1t, q2t)
        q1, q2 = q(s, ac); closs = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        qopt.zero_grad(); closs.backward(); qopt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    # online collector env
    ce = CoinRL4Dof(); co = ce.reset(int(rng.integers(1000, 100000)))
    records = []
    policy_delay = 2 if a.algo == "td3" else 1
    for upd in range(a.updates + 1):
        if upd in CKPTS:
            hl = eval_actor(actor, HEADLINE); vl = eval_actor(actor, VALIDATION)
            with torch.no_grad():
                pa = actor.action_mean(panel)
                adelta = float((pa - panel_a0).abs().mean())
                pdelta = float((torch.cat([p.flatten() for p in actor.parameters()]) - a0_params).abs().mean())
                sb = rb.sample(512, np.random.default_rng(7))
                q1, q2 = q(sb[0], sb[1]); qscale = float(q1.abs().mean()); tw = float((q1 - q2).abs().mean())
            rec = {"update": upd, "env_steps": upd, "headline": hl["deliver"], "headline_grasp": hl["grasp"],
                   "validation": vl["deliver"], "validation_n": vl["n"], "actor_out_delta": round(adelta, 5),
                   "actor_param_delta": round(pdelta, 6), "q_scale": round(qscale, 2), "twin_disagree": round(tw, 3),
                   "alpha": round(float(log_alpha.exp()), 4) if a.algo == "sac" else None,
                   "delivered_headline": hl["delivered_seeds"]}
            records.append(rec)
            print(f"[{a.algo}] upd {upd}: HL {hl['deliver']}/9 grasp {hl['grasp']}/9 VAL {vl['deliver']}/{vl['n']} "
                  f"| aΔout {adelta:.4f} aΔparam {pdelta:.5f} Qscale {qscale:.1f} alpha {rec['alpha']}", flush=True)
            if len(records) >= 2 and records[-1]["headline_grasp"] < 6 and records[-2]["headline_grasp"] < 6:
                print("STOP: grasp < 6/9 for two consecutive checkpoints", flush=True); break
            if not np.isfinite(qscale) or qscale > 1e5:
                print("STOP: Q divergence", flush=True); break
        if upd == a.updates:
            break
        # collect one fresh env step from the current actor (exploration)
        with torch.no_grad():
            if a.algo == "sac":
                act = actor.sample(torch.tensor(co[None]))[0][0].numpy()
            else:
                act = actor.action_mean(torch.tensor(co[None]))[0].numpy() + rng.normal(0, 0.3, 4).astype(np.float32)
        act = np.clip(act, -4, 4)
        co2, r, term, trunc, _ = ce.step(act)
        rb.add(co, act, r, co2, float(term))
        co = ce.reset(int(rng.integers(1000, 100000))) if (term or trunc) else co2
        # updates
        s, ac, rw, s2, d = rb.sample(256, rng)
        with torch.no_grad():
            if a.algo == "sac":
                a2, logp2 = actor.sample(s2); q1t, q2t = qt(s2, a2)
                y = rw + GAMMA * (1 - d) * (torch.min(q1t, q2t) - log_alpha.exp() * logp2)
            else:
                a2 = actor_targ.action_mean(s2) + (0.2 * torch.randn(s2.shape[0], 4)).clamp(-0.5, 0.5)
                a2 = a2.clamp(-4, 4); q1t, q2t = qt(s2, a2)
                y = rw + GAMMA * (1 - d) * torch.min(q1t, q2t)
        q1, q2 = q(s, ac); closs = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        qopt.zero_grad(); closs.backward(); qopt.step()
        if upd % policy_delay == 0:
            if a.algo == "sac":
                ap_, logp = actor.sample(s); qa1, qa2 = q(s, ap_)
                aloss = (log_alpha.exp().detach() * logp - torch.min(qa1, qa2)).mean()
                aopt.zero_grad(); aloss.backward(); aopt.step()
                alloss = -(log_alpha * (logp + tgt_ent).detach()).mean()
                alopt.zero_grad(); alloss.backward(); alopt.step()
            else:
                aloss = -q(s, actor.action_mean(s))[0].mean()
                aopt.zero_grad(); aloss.backward(); aopt.step()
                with torch.no_grad():
                    for p, pt in zip(actor.parameters(), actor_targ.parameters()):
                        pt.mul_(1 - TAU).add_(TAU * p)
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    base = records[0]
    best = max(records, key=lambda r: (r["validation"], r["headline"]))
    out = {"algo": a.algo, "seed": a.seed, "replay_sha16": replay_sha, "n_demo": rb.n_demo, "gamma": GAMMA,
           "init_headline": base["headline"], "init_validation": base["validation"],
           "best_headline": best["headline"], "best_validation": best["validation"], "best_update": best["update"],
           "actor_changed": records[-1]["actor_param_delta"] > 1e-4, "checkpoints": records}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n[{a.algo}] DONE init HL {base['headline']} VAL {base['validation']} -> best HL {best['headline']} "
          f"VAL {best['validation']} @upd {best['update']} | actor_changed {out['actor_changed']}\nSMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
