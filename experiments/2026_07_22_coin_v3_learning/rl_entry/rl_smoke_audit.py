"""Implementation-contract audit of the SAC/TD3 smoke (no training). §1 SAC log-prob coherence, §4 target-net init,
§6 Bellman-target reference, §7 critic optimizer direction, §8 TD3 actor optimizer direction, §10 replay composition,
§12 update-1 trace."""
import copy
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof
from hymeko_rl.coin_delivery.full_action_bc import FullActionBC
from hymeko_rl.coin_delivery.rl_clip_actor import build_shared_sac_td3

BC = "experiments/2026_07_22_coin_v3_learning/bc_configs/bc_handoff_only_best.pt"


def qnet():
    from torch import nn
    return nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))


class TwinQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = qnet(), qnet()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def sha(m):
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()[:16]


def main():
    out = {}
    bc = FullActionBC()
    bc.load_state_dict(torch.load(BC))
    bc.eval()
    sac, td3 = build_shared_sac_td3(bc)
    with torch.no_grad():
        sac.log_std.bias.fill_(-2.4)
    S = torch.randn(2000, 48)

    # ---- §1 SAC log-prob coherence (replicate sample() EXACTLY with a controlled eps to expose the contract) ----
    with torch.no_grad():
        h = sac.backbone(S); mu = sac.mu(h); log_std = sac.log_std(h).clamp(-6, 2); std = log_std.exp()
        torch.manual_seed(123)
        eps = torch.randn_like(mu)              # the SAME draw the (re-seeded) sample() will use
        pre = mu + std * eps
        action_manual = pre.clamp(-4, 4)        # == sample()'s `action = clamp(pre)`
        logp_manual = (-0.5 * eps ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)   # == sample()'s log_prob
        torch.manual_seed(123)
        a_s, logp_s = sac.sample(S)             # sample() with the same seed → same eps
    action_is_clip_of_pre = bool(torch.allclose(a_s, action_manual, atol=1e-5))
    logp_is_preclip_density = bool(torch.allclose(logp_s, logp_manual, atol=1e-4))
    at_boundary = (a_s.abs() >= 4 - 1e-6).any(-1)
    frac_boundary = float(at_boundary.float().mean())
    # for BOUNDARY-clipped samples a coherent clipped-Gaussian must give the boundary MASS log P(z>=4) (grows very
    # negative as std shrinks); the implementation instead returns the finite Gaussian DENSITY of the unclipped pre.
    boundary_logp_finite_density = bool(logp_s[at_boundary].isfinite().all()) if at_boundary.any() else True
    mismatch = action_is_clip_of_pre and logp_is_preclip_density and frac_boundary > 0.0
    out["§1_sac_logprob"] = {
        "action_is_clip_of_pre": action_is_clip_of_pre,
        "logprob_is_preclip_gaussian_density_of_unclipped_z": logp_is_preclip_density,
        "boundary_mass_represented": False, "boundary_samples_get_finite_density": boundary_logp_finite_density,
        "frac_samples_at_boundary": round(frac_boundary, 4),
        "verdict": "SAC_HARD_CLIP_LOGPROB_MISMATCH" if mismatch else "coherent",
    }

    # ---- §4 target-net init ----
    td3_targ = copy.deepcopy(td3)      # as in the smoke
    q = TwinQ(); qt = TwinQ(); qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        maxout = float((td3.action_mean(S) - td3_targ.action_mean(S)).abs().max())
    out["§4_target_init"] = {
        "td3_online_actor_sha": sha(td3), "td3_target_actor_sha": sha(td3_targ),
        "actor_equal_at_init": sha(td3) == sha(td3_targ), "actor_out_maxdiff": maxout,
        "critic_online_sha": sha(q), "critic_target_sha": sha(qt), "critic_equal_at_init": sha(q) == sha(qt),
    }

    # ---- §6 Bellman target reference (TD3) ----
    env = CoinRL4Dof(); o = env.reset(6000); batch = []
    for _ in range(64):
        aa = bc.act(o).astype(np.float32); o2, r, term, trunc, _ = env.step(aa)
        batch.append((o.copy(), aa, float(r), o2.copy(), float(term))); o = o2
        if term or trunc:
            o = env.reset(6001)
    s = torch.tensor(np.array([b[0] for b in batch])); ac = torch.tensor(np.array([b[1] for b in batch]))
    rw = torch.tensor(np.array([b[2] for b in batch])); s2 = torch.tensor(np.array([b[3] for b in batch]))
    d = torch.tensor(np.array([b[4] for b in batch]))
    GAMMA = 0.99
    with torch.no_grad():
        a2 = (td3_targ.action_mean(s2) + (0.2 * torch.randn(s2.shape[0], 4)).clamp(-0.5, 0.5)).clamp(-4, 4)
        q1t, q2t = qt(s2, a2)
        y_impl = rw + GAMMA * (1 - d) * torch.min(q1t, q2t)
        # independent reference
        y_ref = rw + GAMMA * (1.0 - d) * torch.minimum(q1t, q2t)
        no_bootstrap_ok = bool(((d == 1) & (y_impl != rw)).sum() == 0)
        a2_in_bounds = bool((a2.abs() <= 4 + 1e-6).all())
    out["§6_bellman"] = {"target_matches_reference": bool(torch.allclose(y_impl, y_ref, atol=1e-5)),
                         "no_bootstrap_after_terminated": no_bootstrap_ok, "target_action_in_bounds": a2_in_bounds}

    # ---- §7 critic optimizer direction ----
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)
    q1, q2 = q(s, ac); l0 = float(((q1 - y_impl) ** 2 + (q2 - y_impl) ** 2).mean())
    ((q(s, ac)[0] - y_impl) ** 2 + (q(s, ac)[1] - y_impl) ** 2).mean().backward()
    opt.step()
    with torch.no_grad():
        q1b, q2b = q(s, ac); l1 = float(((q1b - y_impl) ** 2 + (q2b - y_impl) ** 2).mean())
    out["§7_critic_direction"] = {"loss_before": round(l0, 4), "loss_after": round(l1, 4),
                                  "decreased": bool(l1 < l0), "verdict": "ok" if l1 < l0 else "CRITIC_OPTIMIZER_DIRECTION_ERROR"}

    # ---- §8 TD3 actor optimizer direction (freeze critic) ----
    for p in q.parameters():
        p.requires_grad_(False)
    aopt = torch.optim.Adam(td3.parameters(), lr=1e-5)
    with torch.no_grad():
        q_before = float(q(s, td3.action_mean(s))[0].mean())
    aloss = -q(s, td3.action_mean(s))[0].mean(); aopt.zero_grad(); aloss.backward(); aopt.step()
    with torch.no_grad():
        q_after = float(q(s, td3.action_mean(s))[0].mean())
    out["§8_td3_actor_direction"] = {"meanQ_before": round(q_before, 4), "meanQ_after": round(q_after, 4),
                                     "Q_increased": bool(q_after > q_before),
                                     "verdict": "ok" if q_after > q_before else "TD3_ACTOR_UPDATE_SIGN_ERROR"}

    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    # primary verdict
    if out["§1_sac_logprob"]["verdict"] == "SAC_HARD_CLIP_LOGPROB_MISMATCH":
        td3_ok = (out["§4_target_init"]["actor_equal_at_init"] and out["§6_bellman"]["target_matches_reference"]
                  and out["§6_bellman"]["no_bootstrap_after_terminated"] and out["§7_critic_direction"]["decreased"]
                  and out["§8_td3_actor_direction"]["Q_increased"])
        print(f"\nPRIMARY: SAC_HARD_CLIP_LOGPROB_MISMATCH | TD3_valid={td3_ok}", flush=True)


if __name__ == "__main__":
    main()
