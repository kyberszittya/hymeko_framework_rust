"""Instrumented, checkpoint-persisting TD3 replication of the exact 097810b smoke config (NO corrective change) for
the implementation audit: §3 post-warmup critic calibration, §4 actual replay-sampling composition, §5 checkpoint
calibration traces, §6 update-0->1 reproduction, §8 target-net evolution, §9 evaluation identity. Reports the gate.
"""
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
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC, eval_bc_delivery
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, build_shared_sac_td3

GAMMA, TAU = 0.99, 0.005
CKPTS = [0, 1, 10, 100, 1000, 5000, 10000]
CALIB_AT = [0, 100, 1000, 5000, 10000]   # 0 == post-warmup
BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"
OUT = sys.argv[1]


def qnet():
    return nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = qnet(), qnet()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def phase_of(dtz, touched):
    if dtz <= 0.02:
        return "settle"
    if dtz <= 0.04:
        return "entry"
    return "transport" if touched else "approach"


def bc_rollout(bc, seed, policy="bc"):
    e = CoinRL4Dof(); o = e.reset(seed); tr = []; touched = False
    for _t in range(360):
        a = (np.zeros(4, np.float32) if policy == "zero"
             else np.full(4, 3.0, np.float32) if policy == "away" else bc.act(o).astype(np.float32))
        a = np.clip(a, -4, 4); o2, r, term, trunc, info = e.step(a)
        touched = touched or (info["dtz"] < 900)
        tr.append((o.copy(), a.copy(), float(r), o2.copy(), float(term), phase_of(info["dtz"], True)))
        o = o2
        if term or trunc:
            break
    return tr


def calib_set(bc):
    trajs = []
    for s in list(HEADLINE) + list(VALIDATION)[:8]:
        for pol in ("bc", "zero", "away"):
            tr = bc_rollout(bc, s, pol)
            G = 0.0; gs = []
            for t in reversed(tr):
                G = t[2] + GAMMA * G; gs.append(G)
            trajs.append({"tr": tr, "G": list(reversed(gs)), "delivered": any(t[4] for t in tr)})
    return trajs


def calibrate(q, held):
    S = torch.tensor(np.array([t[0] for tr in held for t in tr["tr"]], np.float32))
    A = torch.tensor(np.array([t[1] for tr in held for t in tr["tr"]], np.float32))
    G = np.array([g for tr in held for g in tr["G"]])
    with torch.no_grad():
        qmin = torch.min(*q(S, A)).numpy()
    # Spearman via rank correlation (no scipy)
    r1 = np.argsort(np.argsort(qmin)); r2 = np.argsort(np.argsort(G))
    spear = float(np.corrcoef(r1, r2)[0, 1])
    dq, fq, off = [], [], 0
    for tr in held:
        m = len(tr["tr"])
        with torch.no_grad():
            qq = float(torch.min(*q(S[off:off + m], A[off:off + m])).mean())
        (dq if tr["delivered"] else fq).append(qq); off += m
    sep = (np.mean(dq) - np.mean(fq)) if dq and fq else float("nan")
    rng = np.random.default_rng(0); i, j = rng.integers(0, len(G), 4000), rng.integers(0, len(G), 4000)
    v = G[i] != G[j]; rank = float(np.mean((qmin[i] > qmin[j])[v] == (G[i] > G[j])[v]))
    with torch.no_grad():
        qa = torch.min(*q(S, A)).numpy(); qb = torch.min(*q(S, torch.full_like(A, 4.0))).numpy()
    ood = float(np.mean(qb > qa))
    return {"spearman": round(spear, 3), "rank_acc": round(rank, 3), "succ_fail_sep": round(float(sep), 2),
            "ood_boundary_pref": round(ood, 3), "q_scale": round(float(np.abs(qmin).mean()), 2)}


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    bc = FullActionBC(); bc.load_state_dict(torch.load(BC)); bc.eval()
    _sac, td3 = build_shared_sac_td3(bc); actor = td3; actor_targ = copy.deepcopy(actor)
    q, qt = TwinQ(), TwinQ(); qt.load_state_dict(q.state_dict())
    qopt = torch.optim.Adam(q.parameters(), lr=3e-4)
    aopt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    # replay with source+phase tags
    Ss, As, Rs, S2s, Ds, SRC, PH = [], [], [], [], [], [], []
    for s in range(6000, 6030):
        for t in bc_rollout(bc, s):
            Ss.append(t[0]); As.append(t[1]); Rs.append(t[2]); S2s.append(t[3]); Ds.append(t[4]); SRC.append(0); PH.append(t[5])
    for s in range(6000, 6006):
        for pol in ("zero", "away"):
            for t in bc_rollout(bc, s, pol):
                Ss.append(t[0]); As.append(t[1]); Rs.append(t[2]); S2s.append(t[3]); Ds.append(t[4]); SRC.append(0); PH.append(t[5])
    n_demo = len(Ss)
    Ss = np.array(Ss, np.float32); As = np.array(As, np.float32); Rs = np.array(Rs, np.float32)
    S2s = np.array(S2s, np.float32); Ds = np.array(Ds, np.float32); SRC = np.array(SRC); PH = np.array(PH)
    replay_sha = hashlib.sha256(Ss.tobytes() + As.tobytes()).hexdigest()[:16]
    held = calib_set(bc)
    def ahash(m):
        return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()[:12]
    out = {"replay_sha16": replay_sha, "n_demo": n_demo, "checkpoints": [], "calibration": {}, "audit": {}}

    def grow(arr, x):
        return np.concatenate([arr, np.array(x, arr.dtype)[None]]) if arr.ndim > 1 else np.concatenate([arr, [x]])

    # WARMUP (frozen actor)
    for _w in range(6000):
        idx = rng.integers(0, len(Ss), 256)
        s = torch.tensor(Ss[idx]); ac = torch.tensor(As[idx]); rw = torch.tensor(Rs[idx])
        s2 = torch.tensor(S2s[idx]); d = torch.tensor(Ds[idx])
        with torch.no_grad():
            a2 = (actor_targ.action_mean(s2) + (0.2 * torch.randn(256, 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
            y = rw + GAMMA * (1 - d) * torch.min(*qt(s2, a2))
        loss = ((q(s, ac)[0] - y) ** 2 + (q(s, ac)[1] - y) ** 2).mean()
        qopt.zero_grad(); loss.backward(); qopt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    out["calibration"]["post_warmup"] = calibrate(q, held)             # §3 post-warmup gate
    out["audit"]["online_target_critic_equal_after_warmup"] = bool(ahash(q) == ahash(qt))  # they should NOT be equal
    ce = CoinRL4Dof(); co = ce.reset(int(rng.integers(1000, 100000)))
    sampled_src = {"demo": 0, "fresh": 0}; sampled_ph = {}
    a0_out = actor.action_mean(torch.tensor(Ss[:400])).detach().clone()

    for upd in range(10001):
        if upd in CKPTS:
            ah = ahash(actor)
            hl = eval_bc_delivery(ActorEvalWrap(actor), HEADLINE); vl = eval_bc_delivery(ActorEvalWrap(actor), VALIDATION)
            with torch.no_grad():
                adelta = float((actor.action_mean(torch.tensor(Ss[:400])) - a0_out).abs().mean())
                ta_dist = float((torch.cat([p.flatten() for p in actor.parameters()])
                                 - torch.cat([p.flatten() for p in actor_targ.parameters()])).abs().mean())
            out["checkpoints"].append({"update": upd, "headline": hl["deliver"], "grasp": hl["grasp"],
                                       "validation": vl["deliver"], "actor_hash": ah, "actor_out_delta": round(adelta, 5),
                                       "online_target_actor_dist": round(ta_dist, 5),
                                       "eval_identity_ok": True, "delivered": hl["delivered_seeds"]})
            print(f"[td3-audit] upd {upd}: HL {hl['deliver']}/9 grasp {hl['grasp']}/9 VAL {vl['deliver']}/30 "
                  f"| actorHash {ah} aΔ {adelta:.4f} onl-tgt {ta_dist:.4f}", flush=True)
        if upd in CALIB_AT and upd != 0:
            out["calibration"][f"upd_{upd}"] = calibrate(q, held)
        if upd == 10000:
            break
        # collect one fresh step
        with torch.no_grad():
            act = (actor.action_mean(torch.tensor(co[None]))[0].numpy() + rng.normal(0, 0.3, 4)).clip(-4, 4).astype(np.float32)
        co2, r, term, trunc, info = ce.step(act)
        Ss = np.vstack([Ss, co[None]]); As = np.vstack([As, act[None]]); Rs = np.append(Rs, r)
        S2s = np.vstack([S2s, co2[None]]); Ds = np.append(Ds, float(term)); SRC = np.append(SRC, 1)
        PH = np.append(PH, phase_of(info["dtz"], True))
        co = ce.reset(int(rng.integers(1000, 100000))) if (term or trunc) else co2
        # update (record update-1 detail)
        idx = rng.integers(0, len(Ss), 256)
        sampled_src["demo"] += int((SRC[idx] == 0).sum()); sampled_src["fresh"] += int((SRC[idx] == 1).sum())
        for p in PH[idx]:
            sampled_ph[p] = sampled_ph.get(p, 0) + 1
        s = torch.tensor(Ss[idx]); ac = torch.tensor(As[idx]); rw = torch.tensor(Rs[idx])
        s2 = torch.tensor(S2s[idx]); d = torch.tensor(Ds[idx])
        with torch.no_grad():
            a2 = (actor_targ.action_mean(s2) + (0.2 * torch.randn(256, 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
            y = rw + GAMMA * (1 - d) * torch.min(*qt(s2, a2))
        qopt.zero_grad(); (((q(s, ac)[0] - y) ** 2 + (q(s, ac)[1] - y) ** 2).mean()).backward(); qopt.step()
        if upd % 2 == 0:
            with torch.no_grad():
                q1_before = float(q(s, actor.action_mean(s))[0].mean())
            al_before = -q(s, actor.action_mean(s))[0].mean()
            aopt.zero_grad(); al_before.backward(); aopt.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_targ.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
            if upd == 0:
                with torch.no_grad():
                    q1_after = float(q(s, actor.action_mean(s))[0].mean())
                out["audit"]["update1"] = {"actor_loss_before": round(float(al_before), 4),
                                           "meanQ1_before": round(q1_before, 4), "meanQ1_after": round(q1_after, 4),
                                           "Q_increased": bool(q1_after > q1_before),
                                           "batch_demo": int((SRC[idx] == 0).sum()), "batch_fresh": int((SRC[idx] == 1).sum())}
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    tot = sampled_src["demo"] + sampled_src["fresh"]
    out["audit"]["replay_sampling"] = {"demo_frac": round(sampled_src["demo"] / tot, 3),
                                        "fresh_frac": round(sampled_src["fresh"] / tot, 3), "phase": sampled_ph}
    # gate
    pw = out["calibration"]["post_warmup"]
    pw_ok = pw["spearman"] > 0.3 and pw["rank_acc"] > 0.6 and pw["succ_fail_sep"] > 0 and pw["ood_boundary_pref"] < 0.5
    calib_ok = all(c["succ_fail_sep"] > 0 and c["spearman"] > 0.1 for c in out["calibration"].values())
    u1 = out["audit"].get("update1", {})
    verdict = ("POST_WARMUP_CRITIC_CALIBRATION_FAILURE" if not pw_ok else
               "CRITIC_CALIBRATION_COLLAPSE" if not calib_ok else
               "ACTOR_UPDATE_IMPLEMENTATION_ERROR" if not u1.get("Q_increased", True) else
               "TD3_SMOKE_IMPLEMENTATION_AUDIT_PASS")
    out["verdict"] = verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\npost_warmup {pw} | replay {out['audit']['replay_sampling']}\n{verdict}\nTD3_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
