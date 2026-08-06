"""§3 critic calibration for the frozen clip-actor init (SAC + TD3), in canonical executed-action space.

Collect trajectory classes (BC rollouts, zero, away, perturbed) with canonical rewards + discounted return-to-go G_t;
split trajectory-level; fit twin-Q critics via Bellman with the FROZEN actor (no actor step); evaluate held-out
calibration: Q vs empirical G_t (Pearson/Spearman), success/failure separation, terminal-mask audit, OOD boundary
overvaluation, saturation-partitioned. SAC and TD3 differ only in the next-action a' semantics.
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE, VALIDATION
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import build_shared_sac_td3

GAMMA = 0.99
OUT = sys.argv[1]
BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"


def _qnet():
    return nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = _qnet(), _qnet()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def rollout(bc, seed, policy="bc", noise=0.0, rng=None):
    e = CoinRL4Dof()
    o = e.reset(seed)
    tr = []
    for _t in range(360):
        if policy == "zero":
            a = np.zeros(4, np.float32)
        elif policy == "away":
            a = np.full(4, 3.0, np.float32)
        else:
            a = bc.act(o).astype(np.float32)
            if noise:
                a = a + rng.normal(0, noise, 4).astype(np.float32)
        a = np.clip(a, -4, 4)
        o2, r, term, trunc, info = e.step(a)
        tr.append((o.copy(), a.copy(), float(r), o2.copy(), bool(term), bool(trunc)))
        o = o2
        if term or trunc:
            break
    # discounted return-to-go
    G = 0.0
    gts = []
    for (_s, _a, r, _s2, _term, _tr) in reversed(tr):
        G = r + GAMMA * G
        gts.append(G)
    gts = list(reversed(gts))
    delivered = any(t[4] for t in tr)
    return tr, gts, delivered


def collect(bc):
    rng = np.random.default_rng(0)
    trajs = []
    for s in HEADLINE:
        for pol, nz in (("bc", 0.0), ("bc", 0.3), ("zero", 0.0), ("away", 0.0)):
            tr, gts, dl = rollout(bc, s, pol, nz, rng)
            trajs.append({"seed": int(s), "policy": f"{pol}{'+n' if nz else ''}", "tr": tr, "G": gts, "delivered": dl})
    for s in list(VALIDATION)[:12]:
        for pol, nz in (("bc", 0.0), ("bc", 0.3)):
            tr, gts, dl = rollout(bc, s, pol, nz, rng)
            trajs.append({"seed": int(s), "policy": f"{pol}{'+n' if nz else ''}", "tr": tr, "G": gts, "delivered": dl})
    return trajs


def fit_critic(train, actor, *, td3=False, steps=8000, seed=0):
    torch.manual_seed(seed)
    q, qt = TwinQ(), TwinQ()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    S = torch.tensor(np.array([t[0] for tr in train for t in tr["tr"]]), dtype=torch.float32)
    A = torch.tensor(np.array([t[1] for tr in train for t in tr["tr"]]), dtype=torch.float32)
    R = torch.tensor(np.array([t[2] for tr in train for t in tr["tr"]]), dtype=torch.float32)
    S2 = torch.tensor(np.array([t[3] for tr in train for t in tr["tr"]]), dtype=torch.float32)
    D = torch.tensor(np.array([t[4] for tr in train for t in tr["tr"]]), dtype=torch.float32)  # terminated only
    n = len(S)
    rng = np.random.default_rng(seed)
    for _i in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, 256))
        with torch.no_grad():
            if td3:
                a2 = actor.action_mean(S2[idx])
                a2 = (a2 + (0.2 * torch.randn_like(a2)).clamp(-0.5, 0.5)).clamp(-4, 4)   # target smoothing
            else:
                a2, _ = actor.sample(S2[idx])
            q1t, q2t = qt(S2[idx], a2)
            y = R[idx] + GAMMA * (1 - D[idx]) * torch.min(q1t, q2t)                       # terminal: no bootstrap
        q1, q2 = q(S[idx], A[idx])
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(0.995).add_(0.005 * p)
    return q


def calibrate(q, held):
    S = torch.tensor(np.array([t[0] for tr in held for t in tr["tr"]]), dtype=torch.float32)
    A = torch.tensor(np.array([t[1] for tr in held for t in tr["tr"]]), dtype=torch.float32)
    G = np.array([g for tr in held for g in tr["G"]])
    with torch.no_grad():
        q1, q2 = q(S, A)
        qmin = torch.min(q1, q2).numpy()
    pear = float(pearsonr(qmin, G)[0])
    spear = float(spearmanr(qmin, G)[0])
    # success/failure separation (trajectory-level: mean Q vs delivered)
    dq, fq = [], []
    off = 0
    for tr in held:
        m = len(tr["tr"])
        with torch.no_grad():
            qq = torch.min(*q(S[off:off + m], A[off:off + m])).mean().item()
        (dq if tr["delivered"] else fq).append(qq)
        off += m
    sep = (float(np.mean(dq)) - float(np.mean(fq))) if dq and fq else float("nan")
    # ranking accuracy on random pairs
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(G), 4000), rng.integers(0, len(G), 4000)
    valid = G[i] != G[j]
    rank_acc = float(np.mean((qmin[i] > qmin[j])[valid] == (G[i] > G[j])[valid]))
    return {"pearson": round(pear, 3), "spearman": round(spear, 3), "success_fail_sep": round(sep, 3),
            "rank_accuracy": round(rank_acc, 3), "n_deliver_traj": len(dq), "n_fail_traj": len(fq),
            "mean_Q_deliver": round(float(np.mean(dq)), 2) if dq else None,
            "mean_Q_fail": round(float(np.mean(fq)), 2) if fq else None}


def ood_saturation(q, actor, held):
    """OOD boundary overvaluation + saturation partition: at held-out states compare Q(actor a) vs boundary ±4 vs zero vs inward."""
    S = torch.tensor(np.array([t[0] for tr in held for t in tr["tr"]]), dtype=torch.float32)
    with torch.no_grad():
        a_act = actor.action_mean(S)
        raw = actor.mu(actor.backbone(S)) if hasattr(actor, "mu") else actor.head(actor.backbone(S))
        sat = (raw.abs() >= 4 - 1e-6).any(-1).numpy()
        q_act = torch.min(*q(S, a_act)).numpy()
        q_bound = torch.min(*q(S, torch.full_like(a_act, 4.0))).numpy()
        q_zero = torch.min(*q(S, torch.zeros_like(a_act))).numpy()
        q_rand = torch.min(*q(S, (torch.rand_like(a_act) * 8 - 4))).numpy()
    ood_boundary_overvalued = float(np.mean(q_bound > q_act))       # fraction where boundary beats the actor action
    ood_random_overvalued = float(np.mean(q_rand > q_act))
    return {"frac_boundary_beats_actor": round(ood_boundary_overvalued, 3),
            "frac_random_beats_actor": round(ood_random_overvalued, 3),
            "mean_Q_actor": round(float(q_act.mean()), 2), "mean_Q_boundary": round(float(q_bound.mean()), 2),
            "mean_Q_zero": round(float(q_zero.mean()), 2),
            "unsaturated_n": int((~sat).sum()), "saturated_n": int(sat.sum())}


def main():
    bc = FullActionBC()
    bc.load_state_dict(torch.load(BC))
    bc.eval()
    sac, td3 = build_shared_sac_td3(bc)
    trajs = collect(bc)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(trajs))
    held_ids = set(perm[:len(trajs) // 3].tolist())          # trajectory-level held-out split
    train = [t for k, t in enumerate(trajs) if k not in held_ids]
    held = [t for k, t in enumerate(trajs) if k in held_ids]
    print(f"trajectories: {len(trajs)} ({len(train)} train / {len(held)} held-out); "
          f"deliver {sum(t['delivered'] for t in trajs)}", flush=True)
    out = {"gamma": GAMMA, "n_train_traj": len(train), "n_held_traj": len(held)}
    for name, actor, is_td3 in (("SAC", sac, False), ("TD3", td3, True)):
        q = fit_critic(train, actor, td3=is_td3, steps=8000)
        cal = calibrate(q, held)
        ood = ood_saturation(q, actor, held)
        out[name] = {"calibration": cal, "ood_saturation": ood}
        print(f"{name}: pearson {cal['pearson']} spearman {cal['spearman']} rank_acc {cal['rank_accuracy']} "
              f"succ-fail_sep {cal['success_fail_sep']} | Q_deliver {cal['mean_Q_deliver']} Q_fail {cal['mean_Q_fail']} "
              f"| boundary_beats_actor {ood['frac_boundary_beats_actor']} random_beats {ood['frac_random_beats_actor']}",
              flush=True)
    # gate
    def ok(a):
        c, o = out[a]["calibration"], out[a]["ood_saturation"]
        return (c["spearman"] > 0.3 and c["rank_accuracy"] > 0.6 and c["success_fail_sep"] > 0
                and o["frac_boundary_beats_actor"] < 0.5 and o["frac_random_beats_actor"] < 0.5)
    sac_ok, td3_ok = ok("SAC"), ok("TD3")
    verdict = ("CRITIC_CALIBRATION_PASS" if sac_ok and td3_ok else
               "SAC_CRITIC_CALIBRATION_PASS" if sac_ok else "TD3_CRITIC_CALIBRATION_PASS" if td3_ok else
               "CRITIC_RETURN_RANKING_FAILURE")
    out["sac_pass"], out["td3_pass"], out["verdict"] = sac_ok, td3_ok, verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
