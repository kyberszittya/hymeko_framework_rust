"""TD3 critic-first distribution recovery (actor FROZEN). Rebuild the critic on BROADENED data with disjoint panels
(training / authorization / final-audit), ordered additive coverage ablations (§9 A demo -> B +frozen rollouts ->
C +failures -> D +perturbations), authorize on a held-out panel (§7), then a single untouched final audit (§8).
"""
import copy
import glob
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import build_shared_sac_td3

GAMMA, TAU = 0.99, 0.005
BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"
BASE = sys.argv[1]          # coin_v3_dataset_train dir (certified delivering trajectories)
OUT = sys.argv[2]
RNG = np.random.default_rng(0)


def qnet():
    return nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = qnet(), qnet()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def roll(policy, seed, bc, actor=None, noise=0.0):
    """Roll a policy in the canonical reward env; return transitions (s,a,r,s2,term) + delivered."""
    e = CoinRL4Dof(); o = e.reset(seed); tr = []; delivered = False
    for _t in range(360):
        if policy == "zero":
            a = np.zeros(4, np.float32)
        elif policy == "away":
            a = np.full(4, 3.0, np.float32)
        elif policy == "actor":
            with torch.no_grad():
                a = actor.action_mean(torch.tensor(o[None]))[0].numpy()
            if noise:
                a = a + RNG.normal(0, noise, 4).astype(np.float32)
        else:  # bc
            a = bc.act(o)
            if noise:
                a = a + RNG.normal(0, noise, 4).astype(np.float32)
        a = np.clip(a, -4, 4).astype(np.float32)
        o2, r, term, trunc, _ = e.step(a)
        delivered = delivered or term
        tr.append((o.copy(), a, float(r), o2.copy(), float(term)))
        o = o2
        if term or trunc:
            break
    return tr, delivered


def replay_certified(base_dir, seeds):
    """Replay recorded certified-delivery actions in the canonical env -> DELIVERING transitions with canonical reward."""
    trs = []
    for f in sorted(glob.glob(base_dir + "/traj_*.npz")):
        seed = int(f.split("traj_")[1].split(".npz")[0])
        if seed not in seeds:
            continue
        z = np.load(f); acts = z["act"]
        e = CoinRL4Dof(); o = e.reset(seed); tr = []; delivered = False
        for a in acts:
            a = np.clip(a, -4, 4).astype(np.float32)
            o2, r, term, trunc, _ = e.step(a)
            delivered = delivered or term
            tr.append((o.copy(), a, float(r), o2.copy(), float(term))); o = o2
            if term or trunc:
                break
        if delivered:
            trs.append(tr)
    return trs


def trajs_with_returns(tr_list):
    out = []
    for tr in tr_list:
        G = 0.0; gs = []
        for t in reversed(tr):
            G = t[2] + GAMMA * G; gs.append(G)
        out.append({"tr": tr, "G": list(reversed(gs)), "delivered": any(t[4] for t in tr)})
    return out


def flat(trajs):
    return (np.array([t[0] for tr in trajs for t in tr["tr"]], np.float32),
            np.array([t[1] for tr in trajs for t in tr["tr"]], np.float32),
            np.array([t[2] for tr in trajs for t in tr["tr"]], np.float32),
            np.array([t[3] for tr in trajs for t in tr["tr"]], np.float32),
            np.array([t[4] for tr in trajs for t in tr["tr"]], np.float32))


def train_critic(trajs, actor_targ, steps, seed=0):
    torch.manual_seed(seed)
    q, qt = TwinQ(), TwinQ(); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    S, A, R, S2, D = flat(trajs)
    S = torch.tensor(S); A = torch.tensor(A); R = torch.tensor(R); S2 = torch.tensor(S2); D = torch.tensor(D)
    n = len(S); rng = np.random.default_rng(seed)
    for _i in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, 256))
        with torch.no_grad():
            a2 = (actor_targ.action_mean(S2[idx]) + (0.2 * torch.randn(256, 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
            y = R[idx] + GAMMA * (1 - D[idx]) * torch.min(*qt(S2[idx], a2))
        loss = ((q(S[idx], A[idx])[0] - y) ** 2 + (q(S[idx], A[idx])[1] - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    return q


def authorize(q, panel):
    S, A, R, S2, D = flat(panel)
    S = torch.tensor(S); A = torch.tensor(A)
    G = np.array([g for tr in panel for g in tr["G"]])
    with torch.no_grad():
        qmin = torch.min(*q(S, A)).numpy()
    spear = float(np.corrcoef(np.argsort(np.argsort(qmin)), np.argsort(np.argsort(G)))[0, 1])
    dq, fq, off = [], [], 0
    for tr in panel:
        m = len(tr["tr"])
        with torch.no_grad():
            qq = float(torch.min(*q(S[off:off + m], A[off:off + m])).mean())
        (dq if tr["delivered"] else fq).append(qq); off += m
    sep = (np.mean(dq) - np.mean(fq)) if dq and fq else float("nan")
    rng = np.random.default_rng(0); i, j = rng.integers(0, len(G), 5000), rng.integers(0, len(G), 5000)
    v = G[i] != G[j]; rank = float(np.mean((qmin[i] > qmin[j])[v] == (G[i] > G[j])[v]))
    with torch.no_grad():
        qb = torch.min(*q(S, torch.full_like(A, 4.0))).numpy(); qr = torch.min(*q(S, torch.rand_like(A) * 8 - 4)).numpy()
        tw = float((q(S, A)[0] - q(S, A)[1]).abs().mean())
    ood_b = float(np.mean(qb > qmin)); ood_r = float(np.mean(qr > qmin))
    return {"spearman": round(spear, 3), "rank_acc": round(rank, 3), "succ_fail_sep": round(float(sep), 2),
            "ood_boundary": round(ood_b, 3), "ood_random": round(ood_r, 3), "twin_disagree": round(tw, 3),
            "n_deliver": len(dq), "n_fail": len(fq)}


def gate(m):
    return (m["succ_fail_sep"] > 0 and m["spearman"] > 0.2 and m["rank_acc"] > 0.6
            and m["ood_boundary"] < 0.5 and m["ood_random"] < 0.5)


def main():
    bc = FullActionBC(); bc.load_state_dict(torch.load(BC)); bc.eval()
    _sac, td3 = build_shared_sac_td3(bc); actor_targ = copy.deepcopy(td3)
    # ---- disjoint panels ----
    train_seeds = set(range(6000, 6080))
    auth_seeds = list(HEADLINE) + list(range(7000, 7015))
    final_seeds = list(range(7015, 7030))
    print("building data (frozen actor)...", flush=True)
    # training coverage classes
    demo = trajs_with_returns(replay_certified(BASE, train_seeds))                       # A: delivering demos
    frozen = trajs_with_returns([roll("actor", s, bc, td3)[0] for s in range(6000, 6050)])  # B: frozen-actor rollouts
    fails = trajs_with_returns([roll(p, s, bc)[0] for s in range(6000, 6020) for p in ("zero", "away")])  # C
    perturb = trajs_with_returns([roll("bc", s, bc, noise=0.4)[0] for s in range(6000, 6040)])            # D
    auth = trajs_with_returns([roll(p, s, bc)[0] for s in auth_seeds for p in ("bc", "zero", "away")])
    final = trajs_with_returns([roll(p, s, bc)[0] for s in final_seeds for p in ("bc", "zero", "away")])
    print(f"demo(deliver) {len(demo)} frozen {len(frozen)} fails {len(fails)} perturb {len(perturb)} "
          f"| auth {len(auth)} ({sum(t['delivered'] for t in auth)} deliver) final {len(final)}", flush=True)
    out = {"panels": {"auth_seeds_sha": hashlib.sha256(json.dumps(auth_seeds).encode()).hexdigest()[:12],
                      "final_seeds_sha": hashlib.sha256(json.dumps(final_seeds).encode()).hexdigest()[:12]},
           "ablations": {}}
    # ---- §9 ordered additive ablations ----
    A = demo; B = A + frozen; C = B + fails; D = C + perturb
    for name, data in (("A_demo", A), ("B_+frozen", B), ("C_+fails", C), ("D_+perturb", D)):
        q = train_critic(data, actor_targ, 10000)
        m = authorize(q, auth)
        out["ablations"][name] = {"n_traj": len(data), "auth": m, "pass": gate(m)}
        print(f"  {name:<12} n={len(data):<4} auth sep {m['succ_fail_sep']:+.2f} spear {m['spearman']:+.2f} "
              f"rank {m['rank_acc']} ood_b {m['ood_boundary']} ood_r {m['ood_random']} -> pass={gate(m)}", flush=True)
    # ---- §7 authorization (two consecutive checkpoints on the best/full config D) ----
    ckpt_pass = []
    qD = None
    for step in (6000, 10000, 15000, 20000):
        q = train_critic(D, actor_targ, step)
        m = authorize(q, auth)
        ok = gate(m); ckpt_pass.append(ok)
        out.setdefault("authorization_ckpts", {})[str(step)] = {"auth": m, "pass": ok}
        print(f"  ckpt {step}: sep {m['succ_fail_sep']:+.2f} spear {m['spearman']:+.2f} pass {ok}", flush=True)
        if len(ckpt_pass) >= 2 and ckpt_pass[-1] and ckpt_pass[-2]:
            qD = q; break
    authorized = qD is not None
    # ---- §8 final audit (untouched panel) ----
    final_m = authorize(qD if authorized else train_critic(D, actor_targ, 20000), final)
    out["final_audit"] = {"metrics": final_m, "pass": gate(final_m)}
    coverage_load_bearing = (not out["ablations"]["A_demo"]["pass"] and out["ablations"]["D_+perturb"]["pass"])
    if not authorized:
        verdict = "POST_WARMUP_CRITIC_CALIBRATION_FAILURE"
    elif not gate(final_m):
        verdict = "CRITIC_AUTHORIZATION_OVERFIT"
    else:
        verdict = "TD3_CRITIC_FINAL_AUDIT_PASS"
    out["verdict"] = verdict; out["coverage_load_bearing"] = coverage_load_bearing
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nfinal_audit {final_m} pass={gate(final_m)} | coverage_load_bearing={coverage_load_bearing}\n{verdict}\nRECOVERY_DONE",
          flush=True)


if __name__ == "__main__":
    main()
