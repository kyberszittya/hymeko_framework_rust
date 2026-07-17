"""S1 critic-calibration test — separates reward/Q SCALE from actual OVERESTIMATION. Our SAC, 2 configs, 4 seeds.

For each eval checkpoint, roll the greedy policy and compute, at the visited (s,a):
  * Monte-Carlo empirical discounted return G_t (on the SAME reward the critic saw: normalized by the eval-rollout
    RMS when reward_norm is on, raw when off — so a high Q that merely reflects the 1/(1-gamma) return scale shows
    ZERO bias, while a genuinely over-optimistic critic shows Q >> G);
  * Q1,Q2 predictions -> min-Q; signed bias Q - G; Q1-Q2 disagreement; corr(Q, G).
Plus behavioural (min_d, contact<7.5cm, first-contact step, stable-contact over final 3 evals) and policy stats
(entropy = -E[logp], log_std). Configs differ ONLY in reward_norm (both auto-alpha, SB3-matched init/lr).
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import s1_env  # noqa: E402
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac, AlphaMode  # noqa: E402

STEPS = int(os.environ.get("S1_STEPS", "40000"))
EVAL = max(8000, STEPS // 5)
CONTACT_OK = 0.075
SEEDS = (0, 1, 2, 3)


def eval_calib(env, actor, critics, gamma, reward_norm, n=8):
    contacts, min_ds, ents, lstds, ep_ret = [], [], [], [], []
    Qs, MCs, Q1s, Q2s = [], [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=70_000 + i)
        ts, ta, tr = [], [], []
        m = 1e9
        for _ in range(s1_env.HORIZON):
            st = torch.as_tensor(obs[None], dtype=torch.float32)
            with torch.no_grad():
                a = actor.action_mean(st).squeeze(0).numpy()
                _, logp = actor.sample(st)
                ls = actor.log_std(actor.backbone(st)).clamp(-20, 2).mean().item()
            ents.append(-float(logp.item()))
            lstds.append(float(ls))
            ts.append(obs.copy())
            ta.append(a.copy())
            obs, r, term, trunc, info = env.step(a)
            tr.append(float(r))
            m = min(m, info["d_eef_mug"])
            if term or trunc:
                break
        min_ds.append(m)
        contacts.append(int(m < CONTACT_OK))
        r_arr = np.array(tr, dtype=np.float64)
        ep_ret.append(float(r_arr.sum()))
        r_forq = (
            r_arr / (float(np.sqrt(np.mean(r_arr**2))) + 1e-6) if reward_norm else r_arr
        )
        g = np.zeros(len(r_forq))
        acc = 0.0
        for t in range(len(r_forq) - 1, -1, -1):
            acc = r_forq[t] + gamma * acc
            g[t] = acc
        for t in range(len(ts)):
            st = torch.as_tensor(ts[t][None], dtype=torch.float32)
            at = torch.as_tensor(ta[t][None], dtype=torch.float32)
            with torch.no_grad():
                q1 = float(critics[0](st, at).item())
                q2 = float(critics[1](st, at).item())
            Q1s.append(q1)
            Q2s.append(q2)
            Qs.append(min(q1, q2))
            MCs.append(float(g[t]))
    Qs, MCs, Q1s, Q2s = map(np.array, (Qs, MCs, Q1s, Q2s))
    corr = (
        float(np.corrcoef(Qs, MCs)[0, 1])
        if Qs.std() > 1e-9 and MCs.std() > 1e-9
        else 0.0
    )
    return {
        "contact_success": round(float(np.mean(contacts)), 3),
        "min_d_median": round(float(np.median(min_ds)), 4),
        "min_d_best": round(float(np.min(min_ds)), 4),
        "entropy_mean": round(float(np.mean(ents)), 3),
        "logstd_mean": round(float(np.mean(lstds)), 3),
        "empirical_return_raw": round(float(np.mean(ep_ret)), 2),
        "Q_mean": round(float(Qs.mean()), 2),
        "MC_return_mean": round(float(MCs.mean()), 2),
        "Q_minus_MC_bias": round(float((Qs - MCs).mean()), 2),
        "Q_minus_MC_bias_rel": round(
            float((Qs - MCs).mean() / (abs(MCs.mean()) + 1e-6)), 3
        ),
        "q1_q2_disagreement": round(float(np.mean(np.abs(Q1s - Q2s))), 3),
        "corr_Q_MC": round(corr, 3),
    }


CONFIGS = {"ours_auto_rnorm_OFF": dict(reward_norm=False)}


def run(name, kw, mean, std, rv):
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = s1_env.make_s1_env(mean, std, rv)
        actor, critics = build_sac(
            "mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256
        )
        cfg = SACConfig(
            total_steps=STEPS,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            init_alpha=1.0,
            alpha_mode=AlphaMode.AUTO,
            target_entropy=-4.0,
            start_steps=1000,
            capacity=200_000,
            eval_every=EVAL,
            log_every=EVAL,
            **kw,
        )
        log = []
        ee = s1_env.make_s1_env(mean, std, rv)
        train_sac(
            actor,
            critics,
            env,
            cfg,
            eval_fn=lambda _e, a, _l=log, _ee=ee, _c=critics, _rn=kw["reward_norm"]: (
                _l.append(eval_calib(_ee, a, _c, cfg.gamma, _rn))
                or _l[-1]["contact_success"]
            ),
        )
        cs = [m["contact_success"] for m in log]
        fc = next((EVAL * (k + 1) for k, v in enumerate(cs) if v > 0), None)
        per.append(
            {
                "seed": seed,
                "first_contact_step": fc,
                "stable_contact_final3": round(float(np.mean(cs[-3:])), 3),
                "final_contact": cs[-1],
                "min_d_curve": [m["min_d_median"] for m in log],
                "contact_curve": cs,
                "entropy_curve": [m["entropy_mean"] for m in log],
                "logstd_curve": [m["logstd_mean"] for m in log],
                "empirical_return_final": log[-1]["empirical_return_raw"],
                "Q_final": log[-1]["Q_mean"],
                "MC_return_final": log[-1]["MC_return_mean"],
                "Q_minus_MC_bias_final": log[-1]["Q_minus_MC_bias"],
                "q1_q2_disagreement_final": log[-1]["q1_q2_disagreement"],
                "corr_Q_MC_final": log[-1]["corr_Q_MC"],
            }
        )
        print(
            f"[calib] {name} seed{seed}: contact {cs} | stable3 {per[-1]['stable_contact_final3']} | "
            f"Q {log[-1]['Q_mean']} MC {log[-1]['MC_return_mean']} bias {log[-1]['Q_minus_MC_bias']} "
            f"disagree {log[-1]['q1_q2_disagreement']} corr {log[-1]['corr_Q_MC']} | ent {log[-1]['entropy_mean']}",
            flush=True,
        )
    n_stable = sum(1 for p in per if p["stable_contact_final3"] >= 0.5)
    return {
        "config": name,
        "reward_norm": kw["reward_norm"],
        "n_seeds_stable_contact": n_stable,
        "median_bias_final": round(
            float(np.median([p["Q_minus_MC_bias_final"] for p in per])), 2
        ),
        "per_seed": per,
    }


def main():
    mean, std, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    res = [run(n, kw, mean, std, rv) for n, kw in CONFIGS.items()]
    (HERE / "s1_calib_result.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n[calib] SUMMARY")
    for r in res:
        print(
            f"  {r['config']:22s} stable_contact={r['n_seeds_stable_contact']}/4  median_Q-MC_bias={r['median_bias_final']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
