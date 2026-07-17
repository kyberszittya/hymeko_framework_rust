"""SB3 SAC on the byte-identical S1 env (fixed-mug reach) — the trusted reference for the S1 cross-impl test.

Matched to s1_ours.py: 40k steps, 2 seeds, lr 3e-4, batch 256, gamma 0.99, tau 0.005, auto entropy, net [256,256].
Grades success at the physical contact distance (d<7.5cm / near_object), reports the impossible 5cm too. Runs in the
SB3 venv (stable-baselines3 + metaworld). If SB3 solves reach-contact and ours doesn't -> our config defect; if
neither -> reward/task/setup inadequate; if both -> proceed to contact/push.
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")
import s1_env  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

STEPS = 40_000
CONTACT_OK = 0.075


def eval_sb3(model, env, n=8):
    md, contact, near = [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=70_000 + i)
        m, nr, steps = 1e9, 0, 0
        for _ in range(s1_env.HORIZON):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            m = min(m, info["d_eef_mug"])
            nr += int(info.get("near_object", 0))
            steps += 1
            if term or trunc:
                break
        md.append(m)
        contact.append(int(m < CONTACT_OK))
        near.append(nr / max(1, steps))
    md = np.array(md)
    return {
        "min_d_median": round(float(np.median(md)), 4),
        "min_d_best": round(float(md.min()), 4),
        "success_contact_7_5cm": round(float(np.mean(contact)), 3),
        "frac_within_10cm": round(float(np.mean(md < 0.10)), 3),
        "frac_within_7_5cm": round(float(np.mean(md < 0.075)), 3),
        "frac_within_5cm": round(float(np.mean(md < 0.05)), 3),
        "near_object_rate": round(float(np.median(near)), 4),
    }


class Ck(BaseCallback):
    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env
        self.curve = []

    def _on_step(self):
        if self.num_timesteps % 8000 == 0:
            v = self.model.logger.name_to_value
            m = eval_sb3(self.model, self.eval_env)
            m["ent_coef"] = float(v.get("train/ent_coef", float("nan")))
            m["actor_loss"] = float(v.get("train/actor_loss", float("nan")))
            m["critic_loss"] = float(v.get("train/critic_loss", float("nan")))
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
    cs = cb.curve[-1]["success_contact_7_5cm"] if cb.curve else 0.0
    print(
        f"[sb3] seed{seed}: contact_succ {[m['success_contact_7_5cm'] for m in cb.curve]} | "
        f"min_d {[m['min_d_median'] for m in cb.curve]} | 5cm {[m['frac_within_5cm'] for m in cb.curve]}",
        flush=True,
    )
    return {
        "seed": seed,
        "curve": cb.curve,
        "final_contact_success": cs,
        "best_min_d": min(m["min_d_best"] for m in cb.curve) if cb.curve else None,
    }


def main():
    mean, std, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    per = [run_seed(s, mean, std, rv) for s in (0, 1)]
    cs = float(np.median([p["final_contact_success"] for p in per]))
    res = {
        "impl": "stable_baselines3_SAC",
        "sb3_version": __import__("stable_baselines3").__version__,
        "steps": STEPS,
        "final_contact_success_median": round(cs, 3),
        "SOLVES_reach_contact": bool(cs >= 0.5),
        "per_seed": per,
    }
    (HERE / "s1_sb3_result.json").write_text(json.dumps(res, indent=2, default=float))
    print(
        f"\n[sb3] SUMMARY contact_success={res['final_contact_success_median']} SOLVES={res['SOLVES_reach_contact']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
