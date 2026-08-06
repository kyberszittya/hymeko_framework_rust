"""SB3 reference SAC on Pendulum-v1 — the trusted-implementation baseline for the SAC-correctness gate.

Seed 42, 50k steps, automatic entropy tuning, deterministic eval (10 before / 20 after). Logs return before/after,
improvement, reward curve, actor/critic loss, ent_coef, action variance. Success: improvement >= 500, final > -500,
no NaN, nonzero action variance. Isolated venv (experiments/sac_smoke_test/.venv_sb3) — does not touch the project.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

SEED = 42
STEPS = 50_000
OUT = Path(__file__).resolve().parent


class Recorder(BaseCallback):
    """Every 1000 steps, snapshot SB3's logged losses + rolling episode return + a sampled action std."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 != 0:
            return True
        v = self.model.logger.name_to_value
        ep = self.model.ep_info_buffer
        ep_rew = float(np.mean([e["r"] for e in ep])) if ep else float("nan")
        # sampled (stochastic) action std over 64 replayed observations => "nonzero action variance during training"
        act_std = float("nan")
        if self.model.replay_buffer.size() > 64:
            import torch
            obs = self.model.replay_buffer.observations[:64, 0]
            with torch.no_grad():
                a, _ = self.model.actor.action_log_prob(torch.as_tensor(obs, dtype=torch.float32))
            act_std = float(a.std().item())
        self.rows.append({
            "step": float(self.num_timesteps), "ep_rew_mean": ep_rew,
            "actor_loss": float(v.get("train/actor_loss", float("nan"))),
            "critic_loss": float(v.get("train/critic_loss", float("nan"))),
            "ent_coef": float(v.get("train/ent_coef", float("nan"))),
            "ent_coef_loss": float(v.get("train/ent_coef_loss", float("nan"))),
            "action_std_sampled": act_std,
        })
        return True


def main() -> int:
    env = Monitor(gym.make("Pendulum-v1"))
    env.reset(seed=SEED)
    eval_env = Monitor(gym.make("Pendulum-v1"))

    model = SAC("MlpPolicy", env, seed=SEED, verbose=0, gamma=0.99, batch_size=256,
                learning_rate=3e-4, tau=0.005, learning_starts=100, ent_coef="auto",
                buffer_size=1_000_000, train_freq=1, gradient_steps=1)

    mb, sb = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    rec = Recorder()
    model.learn(total_timesteps=STEPS, callback=rec, progress_bar=False)
    ma, sa = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)

    curve = [r["ep_rew_mean"] for r in rec.rows]
    any_nan = any(not np.isfinite(r["actor_loss"]) and r["step"] > 2000 for r in rec.rows) or \
        not np.isfinite(ma) or not np.isfinite(mb)
    final_act_std = next((r["action_std_sampled"] for r in reversed(rec.rows)
                          if np.isfinite(r["action_std_sampled"])), float("nan"))
    result = {
        "impl": "stable_baselines3", "sb3_version": __import__("stable_baselines3").__version__,
        "env": "Pendulum-v1", "seed": SEED, "steps": STEPS,
        "return_before_mean": round(float(mb), 2), "return_before_std": round(float(sb), 2),
        "return_after_mean": round(float(ma), 2), "return_after_std": round(float(sa), 2),
        "improvement": round(float(ma - mb), 2), "final_action_std_sampled": round(final_act_std, 4),
        "any_nan": bool(any_nan),
        "PASS": bool((ma - mb) >= 500 and ma > -500 and not any_nan and final_act_std > 1e-3),
    }
    (OUT / "sb3_result.json").write_text(json.dumps(result, indent=2))
    with (OUT / "sb3_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.rows[0].keys()))
        w.writeheader(); w.writerows(rec.rows)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
