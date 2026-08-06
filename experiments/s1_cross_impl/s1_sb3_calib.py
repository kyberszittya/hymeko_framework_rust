"""SB3 SAC on the exact S1 env — 4 seeds, with the SAME critic-calibration metrics as s1_calib.py.

Matched config (net [256,256], lr 3e-4, batch 256, gamma 0.99, tau 0.005, auto entropy, NO reward_norm — SB3 has
none). The reference for: is our critic's Q-vs-return calibration different from SB3's, or the same? Raw MC return
(no reward_norm). Q from SB3's twin critics.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_env  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

STEPS = int(os.environ.get("S1_STEPS", "40000"))
EVAL = max(8000, STEPS // 5)
CONTACT_OK = 0.075
SEEDS = tuple(range(int(os.environ.get('S1_SEEDS', '4'))))


def eval_sb3_calib(model, env, gamma, n=8):
    contacts, min_ds, ents, ep_ret = [], [], [], []
    Qs, MCs, Q1s, Q2s = [], [], [], []
    dev = model.device
    for i in range(n):
        obs, _ = env.reset(seed=70_000 + i)
        ts, ta, tr = [], [], []
        m = 1e9
        for _ in range(s1_env.HORIZON):
            a, _ = model.predict(obs, deterministic=True)
            ot = torch.as_tensor(obs[None], dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, logp = model.actor.action_log_prob(ot)
            ents.append(-float(logp.item()))
            ts.append(obs.copy())
            ta.append(np.asarray(a, np.float32).copy())
            obs, r, term, trunc, info = env.step(a)
            tr.append(float(r))
            m = min(m, info["d_eef_mug"])
            if term or trunc:
                break
        min_ds.append(m)
        contacts.append(int(m < CONTACT_OK))
        ep_ret.append(float(np.sum(tr)))
        r_arr = np.array(tr, dtype=np.float64)
        g = np.zeros(len(r_arr))
        acc = 0.0
        for t in range(len(r_arr) - 1, -1, -1):
            acc = r_arr[t] + gamma * acc
            g[t] = acc
        for t in range(len(ts)):
            ot = torch.as_tensor(ts[t][None], dtype=torch.float32, device=dev)
            at = torch.as_tensor(ta[t][None], dtype=torch.float32, device=dev)
            with torch.no_grad():
                qs = model.critic(ot, at)
            q1 = float(qs[0].item())
            q2 = float(qs[1].item())
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
        "entropy_mean": round(float(np.mean(ents)), 3),
        "empirical_return_raw": round(float(np.mean(ep_ret)), 2),
        "Q_mean": round(float(Qs.mean()), 2),
        "MC_return_mean": round(float(MCs.mean()), 2),
        "Q_minus_MC_bias": round(float((Qs - MCs).mean()), 2),
        "q1_q2_disagreement": round(float(np.mean(np.abs(Q1s - Q2s))), 3),
        "corr_Q_MC": round(corr, 3),
    }


class Ck(BaseCallback):
    def __init__(self, ee):
        super().__init__()
        self.ee = ee
        self.curve = []

    def _on_step(self):
        if self.num_timesteps % EVAL == 0:
            m = eval_sb3_calib(self.model, self.ee, self.model.gamma)
            m["ent_coef"] = float(
                self.model.logger.name_to_value.get("train/ent_coef", float("nan"))
            )
            self.curve.append(m)
        return True


def run_seed(seed, mean, std, rv):
    env = s1_env.make_s1_env(mean, std, rv)
    env.reset(seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        gamma=0.99,
        batch_size=256,
        learning_rate=3e-4,
        tau=0.005,
        learning_starts=1000,
        ent_coef="auto",
        buffer_size=200_000,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    cb = Ck(s1_env.make_s1_env(mean, std, rv))
    model.learn(total_timesteps=STEPS, callback=cb, progress_bar=False)
    cs = [m["contact_success"] for m in cb.curve]
    fc = next((EVAL * (k + 1) for k, v in enumerate(cs) if v > 0), None)
    print(
        f"[sb3c] seed{seed}: contact {cs} | stable3 {round(float(np.mean(cs[-3:])), 3)} | "
        f"Q {cb.curve[-1]['Q_mean']} MC {cb.curve[-1]['MC_return_mean']} bias {cb.curve[-1]['Q_minus_MC_bias']} "
        f"disagree {cb.curve[-1]['q1_q2_disagreement']} corr {cb.curve[-1]['corr_Q_MC']} ent {cb.curve[-1]['entropy_mean']}",
        flush=True,
    )
    return {
        "seed": seed,
        "first_contact_step": fc,
        "stable_contact_final3": round(float(np.mean(cs[-3:])), 3),
        "final_contact": cs[-1],
        "min_d_curve": [m["min_d_median"] for m in cb.curve],
        "contact_curve": cs,
        "ent_coef_curve": [round(m["ent_coef"], 4) for m in cb.curve],
        "Q_final": cb.curve[-1]["Q_mean"],
        "MC_return_final": cb.curve[-1]["MC_return_mean"],
        "Q_minus_MC_bias_final": cb.curve[-1]["Q_minus_MC_bias"],
        "q1_q2_disagreement_final": cb.curve[-1]["q1_q2_disagreement"],
        "corr_Q_MC_final": cb.curve[-1]["corr_Q_MC"],
    }


def main():
    mean, std, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    per = [run_seed(s, mean, std, rv) for s in SEEDS]
    n_stable = sum(1 for p in per if p["stable_contact_final3"] >= 0.5)
    res = {
        "config": "SB3_matched_rnorm_off",
        "sb3_version": __import__("stable_baselines3").__version__,
        "n_seeds_stable_contact": n_stable,
        "median_bias_final": round(
            float(np.median([p["Q_minus_MC_bias_final"] for p in per])), 2
        ),
        "per_seed": per,
    }
    (HERE / "s1_sb3_calib_result.json").write_text(
        json.dumps(res, indent=2, default=float)
    )
    print(
        f"\n[sb3c] SUMMARY stable_contact={n_stable}/4  median_Q-MC_bias={res['median_bias_final']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
