"""SAC-from-scratch for floating-humanoid balance, under the Lyapunov reward + gate.

Reward = alive − 2·V − control cost (V = COM Lyapunov). The reward-independent
``lyapunov_certificate`` is evaluated on the FINAL policy (never in the reward) —
the campaign's unchanged-external-certificate discipline. No hand-tuned or LQR
baseline exists, so this is genuine RL from scratch (coin R14–R60 regime).

Usage::  PYTHONPATH=. python -m scenarios.humanoid.run_humanoid_sac [--steps N]
SIMULATION. Live [sac] progress every log_every steps (§3 never-run-blind).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

from .balance_env import HumanoidBalanceEnv
from .lyapunov import evaluate_lyapunov

_OUT = Path("reports/2026-07-27-humanoid-sac")


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return actor.action_mean(t).squeeze(0).cpu().numpy()


def _eval_balance(env, actor, seeds) -> tuple[float, float]:
    """Return (mean upright fraction, Lyapunov certificate pass rate) over seeds."""
    fracs, lyap_pass = [], 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, up, steps, vs = False, 0, 0, []
        while not done:
            obs, _r, term, trunc, info = env.step(_greedy(actor, obs))
            steps += 1
            vs.append(info["V"])
            if info["upright"]:
                up = steps
            done = term or trunc
        fracs.append(up / env.max_steps)
        lyap_pass += int(evaluate_lyapunov(vs)["passes"])
    return float(np.mean(fracs)), lyap_pass / len(seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    env = HumanoidBalanceEnv(max_steps=500, seed=0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    torch.manual_seed(0)
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                               action_dim=act_dim, action_scale=1.0, hidden=128)

    eval_seeds = list(range(2000, 2006))

    def eval_fn(e, a) -> float:
        return _eval_balance(e, a, eval_seeds)[0]

    cfg = SACConfig(total_steps=args.steps, start_steps=2_000, batch_size=256,
                    eval_every=15_000, log_every=5_000, seed=0)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    final_frac, lyap_rate = _eval_balance(env, actor, list(range(3000, 3012)))
    torch.save(actor.state_dict(), _OUT / "humanoid_sac_actor.pt")
    result = {
        "verdict": ("SAC_BALANCES_AND_SATISFIES_LYAPUNOV" if final_frac > 0.9 and lyap_rate > 0.8
                    else "SAC_PARTIAL" if final_frac > 0.4 else "SAC_INSUFFICIENT"),
        "final_upright_fraction": round(final_frac, 3),
        "final_lyapunov_pass_rate": round(lyap_rate, 3),
        "eval_curve_upright_fraction": [round(c, 3) for c in curve],
        "total_steps": args.steps,
        "reward": "alive - 2*V(COM Lyapunov) - 1e-3*|a|^2",
        "note": "SIMULATION. SAC-from-scratch (no certified baseline). Lyapunov reward; the "
                "reward-INDEPENDENT lyapunov_certificate is the success/safety gate (eval only). "
                "Gravity-comp is a physics feedforward (it tips alone); SAC learns the balance residual.",
    }
    (_OUT / "sac_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "final_upright_fraction": result["final_upright_fraction"],
                      "final_lyapunov_pass_rate": result["final_lyapunov_pass_rate"],
                      "curve": result["eval_curve_upright_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
